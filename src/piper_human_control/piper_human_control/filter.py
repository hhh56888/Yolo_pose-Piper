# -*- coding: utf-8 -*-
"""
piper_human_control.filter
==========================
三级滤波链 + 死区。

为什么需要三级，而不是只在最后滤一次：
    YOLO 关键点抖动 -> 角度计算放大抖动 -> 关节目标抖动
    每一级都有各自的噪声来源：
        关键点级：像素级抖动（box 抖动、亚像素估计不稳）
        角度级：两个关键点的抖动会放大成角度抖动（尤其点靠得近时）
        关节级：映射与限幅带来的阶跃
    只在最后滤一次，等于让前两级噪声先被非线性映射（含 clamp）
    放大再滤，效果差且难调。逐级滤能把噪声在各自的量纲上压掉。

每一级都是**独立可配置**的，并且都可以单独关掉 —— 便于排查
「抖动到底是哪一级引入的」。

滤波方法：EMA（一阶低通）
    y[k] = y[k-1] + alpha * (x[k] - y[k-1])
    alpha 越小越平滑但延迟越大。
    本模块不做 Kalman —— 对当前「单变量 + 近似高斯噪声」的场景，
    EMA 的性价比更高，且没有额外状态需要整定。
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ============================================================
# 配置
# ============================================================
@dataclass
class FilterConfig:
    """
    单个滤波级的配置。

    Attributes:
        enabled:    是否启用
        alpha:      EMA 系数 (0,1]。1 = 不滤波（完全采用新值）
        deadband:   死区。变化量小于此值时不更新（抑制微动）
        max_jump:   单帧突变门限（与量纲一致）。相邻两帧变化超过此值时，
                    **丢弃新值、保持上一帧输出**（见下）。
                    0 = 关闭该保护。
    """
    enabled: bool = True
    alpha: float = 0.4
    deadband: float = 0.0
    max_jump: float = 0.0

    def validate(self, name: str) -> None:
        if not (0.0 < self.alpha <= 1.0):
            raise ValueError(f"配置错误: {name}.alpha 必须在 (0,1]，收到 {self.alpha}")
        if self.deadband < 0:
            raise ValueError(f"配置错误: {name}.deadband 不能为负")
        if self.max_jump < 0:
            raise ValueError(f"配置错误: {name}.max_jump 不能为负")


# ============================================================
# 标量 EMA + 死区
# ============================================================
class ScalarFilter:
    """
    单变量 EMA + 死区滤波器。

    死区的作用与 EMA 不同，两者互补：
        EMA   ：把噪声的「幅度」压下去（但有延迟）
        死区  ：把人手的「无意义微动」直接吃掉（不产生任何运动）
    只用一个通常不够：只有 EMA 会一直有微小输出；
    只有死区会保留 EMA 来不及压掉的高频抖动。
    """

    def __init__(self, cfg: FilterConfig, name: str = ""):
        cfg.validate(name or "filter")
        self.cfg = cfg
        self.name = name
        self._value: Optional[float] = None
        self._raw: Optional[float] = None

    def reset(self, value: Optional[float] = None) -> None:
        """重置状态（例如标定完成、重新捕获目标时）"""
        self._value = value
        self._raw = value

    def update(self, x: float) -> float:
        """喂入新观测，返回滤波后的值"""
        self._raw = x

        if self._value is None:
            self._value = x
            return x

        if not self.cfg.enabled:
            self._value = x
            return x

        diff = x - self._value
        # 死区：变化太小就不动，避免机械臂持续微调
        if self.cfg.deadband > 0.0 and abs(diff) < self.cfg.deadband:
            return self._value

        self._value = self._value + self.cfg.alpha * diff
        return self._value

    @property
    def value(self) -> Optional[float]:
        """当前滤波值"""
        return self._value

    @property
    def raw(self) -> Optional[float]:
        """最近一次原始观测（用于 raw/filtered 对比）"""
        return self._raw


# ============================================================
# 关键点滤波（2D/3D 通用）
# ============================================================
class KeypointFilter:
    """
    关键点级滤波：对每个关键点的 x / y / z 分别做 EMA。

    ⚠️ 保留 3D 接口：
        z 为 None 时**跳过 z 的滤波**，不做「当作 0」这类伪造。
        这样阶段七接入 RGB-D 后，z 自动被滤波，无需改本模块。
    """

    def __init__(self, cfg: FilterConfig, name: str = "keypoint"):
        cfg.validate(name)
        self.cfg = cfg
        self.name = name
        self._fx: Dict[str, ScalarFilter] = {}
        self._fy: Dict[str, ScalarFilter] = {}
        self._fz: Dict[str, ScalarFilter] = {}

    def _get(self, store: Dict[str, ScalarFilter], key: str) -> ScalarFilter:
        if key not in store:
            store[key] = ScalarFilter(self.cfg, f"{self.name}.{key}")
        return store[key]

    def reset(self) -> None:
        self._fx.clear()
        self._fy.clear()
        self._fz.clear()

    def update(self, kp):
        """
        输入一个 Keypoint，返回滤波后的**新 Keypoint**。

        不修改原对象 —— 保持不可变，避免调用方手里的引用被悄悄改写。
        """
        from piper_human_control.types import Keypoint

        if kp is None:
            return None
        if not kp.is_valid:
            # 无效点（confidence=0）不参与滤波，避免污染滤波器状态
            return kp

        key = kp.name or "unnamed"
        x = self._get(self._fx, key).update(kp.x)
        y = self._get(self._fy, key).update(kp.y)

        z = None
        if kp.z is not None:
            # 只有真的有深度时才滤波 —— 不伪造 z
            z = self._get(self._fz, key).update(kp.z)

        return Keypoint(x=x, y=y, z=z, confidence=kp.confidence, name=kp.name)

    def update_arm(self, arm):
        """
        输入 ArmKeypoints，返回滤波后的新 ArmKeypoints。

        注意：wrist 与 hand_wrist 都要滤 —— 骨架用的是 hand_wrist。
        """
        from piper_human_control.types import ArmKeypoints

        if arm is None:
            return None
        return ArmKeypoints(
            shoulder=self.update(arm.shoulder),
            elbow=self.update(arm.elbow),
            wrist=self.update(arm.wrist),
            side=arm.side,
            hand_wrist=self.update(arm.hand_wrist),
        )


# ============================================================
# 关节滤波（多关节）
# ============================================================
class JointFilter:
    """
    关节级滤波：对 6 个关节分别做 EMA + 死区。

    这是滤波链的最后一级，输出直接作为关节目标。
    """

    def __init__(self, cfg: FilterConfig, joint_names: List[str],
                 name: str = "joint"):
        cfg.validate(name)
        self.cfg = cfg
        self.joint_names = list(joint_names)
        self.name = name
        self._f: Dict[str, ScalarFilter] = {
            j: ScalarFilter(cfg, f"{name}.{j}") for j in self.joint_names
        }

    def reset(self, values: Optional[List[float]] = None) -> None:
        if values is None:
            for f in self._f.values():
                f.reset(None)
        else:
            for j, v in zip(self.joint_names, values):
                self._f[j].reset(v)

    def update(self, q: List[float]) -> List[float]:
        if len(q) != len(self.joint_names):
            raise ValueError(
                f"关节数不匹配: 期望 {len(self.joint_names)}, 收到 {len(q)}")
        return [self._f[j].update(v) for j, v in zip(self.joint_names, q)]

    def raw(self) -> List[Optional[float]]:
        """最近一次原始输入（用于 raw/filtered 对比）"""
        return [self._f[j].raw for j in self.joint_names]

    def value(self) -> List[Optional[float]]:
        return [self._f[j].value for j in self.joint_names]


# ============================================================
# 整条滤波链
# ============================================================
@dataclass
class FilterChainConfig:
    """三级滤波链配置"""
    keypoint: FilterConfig
    angle: FilterConfig
    joint: FilterConfig

    def validate(self) -> None:
        self.keypoint.validate("filter.keypoint")
        self.angle.validate("filter.angle")
        self.joint.validate("filter.joint")


class CircularScalarFilter:
    """
    **角度专用**滤波器（正确处理 ±180° 环绕）。

    为什么不能直接用普通 EMA 滤角度：
        角度是圆形量，而 EMA 是线性运算。
        对 +179° 与 −179° 求平均会得到 0°，但这两个角
        在物理上都在 180° 附近，真实平均应是 ±180°。
        实测踩过这个坑：上臂角在 ±180 附近来回翻转时，
        滤波输出恒为 0，该通道完全失效（关节被钉死在中位）。

    做法（圆形统计的标准解法）：
        1) 把角度映射为单位圆上的点 (cos θ, sin θ)
        2) 对 cos 与 sin **分别**做 EMA —— 它们是线性量，可以平均
        3) 用 atan2 还原角度

    这样 +179° 与 −179° 的滤波结果会正确收敛到 ±180°，而不是 0°。

    死区同样按「环绕归一化后的角度差」判断，避免在 ±180 附近误判为大变化。
    """

    def __init__(self, cfg: FilterConfig, name: str = ""):
        cfg.validate(name or "circular_filter")
        self.cfg = cfg
        self.name = name
        self._cos: Optional[float] = None
        self._sin: Optional[float] = None
        self._value: Optional[float] = None
        self._raw: Optional[float] = None

    def reset(self, value: Optional[float] = None) -> None:
        if value is None:
            self._cos = self._sin = self._value = self._raw = None
        else:
            r = math.radians(value)
            self._cos, self._sin = math.cos(r), math.sin(r)
            self._value, self._raw = value, value

    def update(self, x_deg: float) -> float:
        self._raw = x_deg
        r = math.radians(x_deg)
        c, s = math.cos(r), math.sin(r)

        if self._cos is None:
            self._cos, self._sin = c, s
            self._value = x_deg
            return self._value

        if not self.cfg.enabled:
            self._cos, self._sin = c, s
            self._value = x_deg
            return self._value

        # 死区：用环绕归一化后的角度差判断
        if self.cfg.deadband > 0.0 and self._value is not None:
            d = (x_deg - self._value + 180.0) % 360.0 - 180.0
            if abs(d) < self.cfg.deadband:
                return self._value

        a = self.cfg.alpha
        self._cos += a * (c - self._cos)
        self._sin += a * (s - self._sin)
        self._value = math.degrees(math.atan2(self._sin, self._cos))
        return self._value

    @property
    def value(self) -> Optional[float]:
        return self._value

    @property
    def raw(self) -> Optional[float]:
        return self._raw


class AngleFilter:
    """
    角度级滤波：对多个角度量分别做 EMA + 死区。

    为什么角度级也要单独滤：
        角度由两个关键点相减得到，两点各自的抖动会**放大**成角度抖动
        （尤其当两点靠得很近、或其中一个置信度偏低时）。
        在角度量纲上滤一次，比只在关键点级滤更直接有效。
    """

    # ★ 内部用 CircularScalarFilter，因为本类专门处理**角度**（度）。
    #   普通 EMA 会在 ±180° 附近失效（见 CircularScalarFilter 的说明）。
    #   需要滤非角度量请直接用 ScalarFilter。
    def __init__(self, cfg: FilterConfig, names: List[str],
                 name: str = "angle", max_jump: Optional[Dict[str, float]] = None,
                 circular: Optional[set] = None):
        """
        Args:
            max_jump: 每个量各自的单帧突变门限（度）。
                      **必须按量区分** —— 见 update() 里的说明。
                      None 或某量缺省时退回 cfg.max_jump。
            circular: 哪些量是**周期量**（角度）。集合内的用周期滤波
                      （cos/sin 域），集合外的用**线性 EMA**。

        ⚠️ 为什么必须区分周期量与非周期量（实际踩过）：
            本类原先把**所有**输入都当角度用 CircularScalarFilter 滤。
            但 `thumb_offset` 是「拇指在掌宽上的归一化偏移」（约 ±0.5），
            被当成角度后会先按 360° 取模 ——
            滤波值恒在 0 附近，差值 0.45° 又被死区 1.2 吃掉，
            结果 `delta_thumb_offset` **恒为 0**，joint6 完全不动。
            表面看是「映射没配好」，实际是滤波层把量纲搞错了。
        """
        cfg.validate(name)
        self.cfg = cfg
        self.names = list(names)
        self.max_jump = dict(max_jump or {})
        # 默认为空集合：不显式声明就按**线性量**处理，
        # 这样新加的量不会因为「忘了声明」而被错误地周期化。
        self.circular = set(circular or ())
        self._f: Dict[str, ScalarFilter] = {}
        for n in self.names:
            if n in self.circular:
                self._f[n] = CircularScalarFilter(cfg, f"{name}.{n}")
            else:
                # ⚠️ 线性量（thumb_offset / pinch_distance）必须**清掉死区**。
                #    角度死区 1.2（度）是合理的，但线性量量纲是 0~1，
                #    死区 1.2 会把**所有**变化都吃掉 -> 该量恒为初值、
                #    对应关节完全不动。这一点实测踩过：
                #    delta_thumb_offset 恒为 0，joint6 纹丝不动。
                lin_cfg = FilterConfig(enabled=cfg.enabled, alpha=cfg.alpha,
                                       deadband=0.0, max_jump=cfg.max_jump)
                self._f[n] = ScalarFilter(lin_cfg, f"{name}.{n}")

    def reset(self) -> None:
        for f in self._f.values():
            f.reset(None)

    def update(self, values: Dict[str, float]) -> Dict[str, float]:
        """
        只滤出现过的量；缺失的量原样不返回（不伪造）。

        ★ 单帧突变保护（max_jump）：
            角度量偶尔会出现**假跳变** —— 典型场景是关键点被误估到对侧，
            此时 ``手朝向 - 前臂朝向`` 会跨过 ±180°，单帧跳 300° 以上。
            实测腕俯仰就出现过 9/1265 帧跳变超 90°（最大 353°），
            会让腕关节瞬间翻转。

            这类跳变 EMA 是压不住的（它会把输出拉过去一半），
            因此在 EMA **之前**先做门限：超过 max_jump 就丢弃新值、
            保持上一帧输出。这是「拒绝坏观测」，不是「平滑坏观测」。
        """
        out = {}
        for k, v in values.items():
            f = self._f.get(k)
            if f is None:
                out[k] = v
                continue
            # 门限优先取「该量专属」的，没有则退回全局 cfg.max_jump。
            # 按量区分是必须的：wrist_pitch_deg 的假跳变可达 353°，
            # 而 palm_roll_deg 本身就会跨 ±180°、真实单帧变化能到 100°+。
            lim = self.max_jump.get(k, self.cfg.max_jump)
            if lim > 0.0:
                # value 是 property；用 getattr 兼容两种实现
                prev = getattr(f, "value", None)
                if callable(prev):
                    prev = prev()
                if prev is not None:
                    if k in self.circular:
                        d = abs((v - prev + 180.0) % 360.0 - 180.0)  # 周期量
                    else:
                        d = abs(v - prev)                            # 线性量
                    if d > lim:
                        out[k] = prev                        # 丢弃，保持
                        continue
            out[k] = f.update(v)
        return out

    def raw(self) -> Dict[str, Optional[float]]:
        return {n: self._f[n].raw for n in self.names}

    def value_of(self, name: str) -> float:
        """
        取某个量的当前滤波值；尚无值时返回 0.0。

        返回 0.0 而不是 None 是刻意的：调用方（HUD / CSV）拿到 None
        还要额外判空，而「还没数据」与「角度为 0」在显示上无区别。
        需要区分时用 raw()/value() 取 Optional。
        """
        f = self._f.get(name)
        if f is None or f.value is None:
            return 0.0
        return f.value

    def value(self) -> Dict[str, Optional[float]]:
        return {n: self._f[n].value for n in self.names}
