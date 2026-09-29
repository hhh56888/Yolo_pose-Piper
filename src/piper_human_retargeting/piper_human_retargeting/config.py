# -*- coding: utf-8 -*-
"""
piper_human_retargeting.config
==============================
映射配置加载与校验。

所有参数从 config/retargeting.yaml 读取，主程序零硬编码。
加载时立即校验（关节名、区间合法性、限位一致性），
把配置错误暴露在**启动阶段**，而不是运行到一半才崩。
"""

import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import yaml

from piper_human_control import FilterConfig as _BaseFilterConfig
from piper_human_control import GripperConfig

from .mapping import RangeMapping
from .tracker_state import TrackerConfig

try:
    from ament_index_python.packages import get_package_share_directory
    _HAS_AMENT_INDEX = True
except ImportError:
    _HAS_AMENT_INDEX = False


# 支持的人体测量量
# 哪些人体测量量是**周期量**（角度）。
#
# ⚠️ 这个区分是必须的：周期量要用 cos/sin 域滤波（否则 ±180° 附近
#    会把 +179 与 -179 平均成 0）；而 thumb_offset / pinch_distance
#    这类**线性归一化量**若被当成角度，会先按 360° 取模，
#    再被死区吃掉，最终 delta 恒为 0、对应关节完全不动 —— 实际踩过。
CIRCULAR_HUMAN_MEASURES = {
    "upper_arm_angle_deg",
    "forearm_angle_deg",
    "elbow_angle_deg",
    "wrist_pitch_deg",
    "palm_roll_deg",
    "arm_direction_deg",
}

VALID_HUMAN_MEASURES = {
    # 手臂（来自 arm_geometry）
    "upper_arm_angle_deg",
    "forearm_angle_deg",
    "elbow_angle_deg",
    # 任务空间量
    #   human_u / human_v：手腕相对肩的归一化位置（V1.1 task_space 的**主控量**）
    #   reach：伸展程度 = |S-W| / (|S-E|+|E-W|)，**保留为诊断量**（记录用）
    #   （旧版的 elevation 已随 reach/elevation 主控方案一起移除：
    #     垂直信息现在由 human_v 直接表达，不再需要派生量）
    "human_u",
    "human_v",
    "reach",
    # 手臂方向（来自 arm_direction.arm_direction_deg）
    #   「肩 -> 腕」的方向角，**方向驱动映射**的主控量。
    #   用户要求「更注重腕部相对于肩部的相对角度」，这就是那个角度。
    "arm_direction_deg",
    # 腕部/手部姿态（来自 hand.compute_hand_pose）
    #   wrist_pitch_deg  手相对前臂的俯仰 -> 驱动 joint5（实测是纯 pitch）
    #   thumb_offset     拇指在掌横轴上的偏移（+食指侧 / -小指侧）
    #                    -> 驱动 joint6 的腕部旋转（四指指向镜头时
    #                       掌横轴几乎不动，此时 thumb_offset 才是可观测信号）
    #   pinch_distance   拇指尖到四指指尖最近距 -> 驱动夹爪闭合
    #   palm_roll_deg    掌横轴朝向。**保留作为观测/调试量**，
    #                    在四指指向镜头的姿态下它几乎不变，故不再作为主映射量。
    "wrist_pitch_deg",
    "palm_roll_deg",
    "thumb_offset",
    "pinch_distance",
}


# ============================================================
# 子配置
# ============================================================
@dataclass
class CalibrationConfig:
    capture_seconds: float = 2.0
    require_arm_complete: bool = True
    min_samples: int = 8
    sample_rate_hz: float = 30.0


@dataclass
class HumanConfig:
    side: str = "right"
    min_confidence: float = 0.4
    require_complete: bool = True


@dataclass
class ArmDirectionConfig:
    """
    「方向驱动」映射配置。

    动机（用户实测反馈）：
        逐关节映射不保证机械臂朝人手臂的方向伸出去 ——
        Piper 的 joint2/joint3 是耦合的（joint2 同时抬高与前推，
        joint3 同时改肘弯与伸展），于是「人往前伸、机械臂没往前伸」。

    做法：
        取人体「肩->腕」方向角作主控量，
        用它去反解机械臂的 joint2/joint3（查 FK 表），
        使机械臂「肩->腕」的方向**对齐**人的方向。

    gain:
        人机方向增益。人体方向角在实拍视频里行程约 40°，
        而机械臂可达约 150°，直接 1:1 会显得幅度太小。
        gain=1.0 表示完全一致（最"忠实"但幅度小）。
    neutral_dir_deg:
        标定姿势对应的机械臂方向角。**必须由实机 FK 表查得**，
        不能凭直觉填 —— 填错会让整个方向基准偏掉。
    reach_from_elbow:
        是否用「肘伸展度」决定伸展量。
        True  -> 人手臂越直，机械臂伸得越远（第二自由度有明确来源）
        False -> 只对齐方向，伸展取该方向下最伸展的解
    """
    enabled: bool = False
    table_path: str = ""
    gain: float = 1.0
    neutral_dir_deg: float = 0.0
    reach_from_elbow: bool = True
    # 肘夹角（度）与机械臂伸展（米）的对应：伸直端 / 弯曲端
    elbow_straight_deg: float = 20.0
    elbow_bent_deg: float = 140.0
    reach_extended: float = 0.50
    reach_folded: float = 0.20

    def validate(self) -> None:
        if not (0.1 <= self.gain <= 5.0):
            raise ValueError(
                f"配置错误: arm_direction.gain 应在 0.1~5.0，收到 {self.gain}")
        if self.enabled and not self.table_path:
            raise ValueError("配置错误: arm_direction.enabled=true 但未给 table_path")
        if self.elbow_bent_deg <= self.elbow_straight_deg:
            raise ValueError(
                "配置错误: elbow_bent_deg 必须大于 elbow_straight_deg")
        if self.reach_extended <= self.reach_folded:
            raise ValueError(
                "配置错误: reach_extended 必须大于 reach_folded")


@dataclass
class TaskSpaceRetargetConfig:
    """
    V1.1 的 2D task-space 重映射配置（人的手去哪，机器人 TCP 就去哪）。

    retarget_mode（位于 RetargetingConfig 顶层）：
        "legacy"     上臂角->joint2、肘夹角->joint3（原方案，**保留可回退**）
        "task_space" 人体手腕相对位移 (du,dv) -> 机器人 TCP (x,z) -> J2/J3 数值 IK

    本段只在 task_space 模式生效；legacy 模式完全不读这里。

    轴关系是**配置项**（任务书 §8）：单目 2D 下"图像水平"代表机器人哪个轴
    取决于相机站位，不能硬编码。
    """
    # ---- FK 链路（由 tools/build_fk_chain.py 从实际 URDF 生成）----
    chain_path: str = "data/fk_chain.json"
    # 兼容字段：旧版用 fk_table.json（方向/伸展表），保留以避免旧配置直接报错
    table_path: str = ""

    # ---- 轴映射 ----
    horizontal_axis: str = "x"
    vertical_axis: str = "z"
    sign_h: float = 1.0
    sign_v: float = 1.0

    # ---- 人体位移 -> TCP 位移 ----
    # 增益：归一化人体位移(1.0 = 手臂全长的横向位移) -> 米
    kx: float = 0.55
    kz: float = 0.55
    # 死区（归一化人体位移）
    deadzone_u: float = 0.02
    deadzone_v: float = 0.02

    # ---- TCP 目标行程（base_link 坐标，米）----
    x_min: float = -0.05
    x_max: float = 0.46
    z_min: float = 0.10
    z_max: float = 0.62
    clamp: bool = True

    # ---- IK ----
    wx: float = 1.0
    wz: float = 1.0
    lambda_prev: float = 0.35       # ||q - q_prev||²（抑制跳解，任务书 §12）
    lambda_neutral: float = 0.02    # ||q - q_neutral||²（轻微偏好）
    max_iters: int = 12
    tol: float = 1e-4
    # IK 求解的关节区间（**必须** ⊂ safe ⊂ physical，由 limits 层校验）
    q2_min: float = 0.30
    q2_max: float = 2.30
    q3_min: float = -2.20
    q3_max: float = -0.40

    # ---- 安全余量 ----
    limit_margin_rad: float = 0.10

    # ---- task-space 实验 neutral（不动 legacy 的 neutral_joints）----
    # 空 = 沿用 robot.neutral_joints。由 tools/scan_workspace.py 的评分搜索给出。
    neutral_joints: List[float] = field(default_factory=list)

    # ---- IK 跳变/未收敛判据（用于日志与 A/B 统计）----
    jump_warn_rad: float = 0.25

    def to_mapping_config(self):
        from .task_space import TaskMappingConfig
        return TaskMappingConfig(
            horizontal_axis=self.horizontal_axis,
            vertical_axis=self.vertical_axis,
            sign_h=self.sign_h, sign_v=self.sign_v,
            kx=self.kx, kz=self.kz,
            x_min=self.x_min, x_max=self.x_max,
            z_min=self.z_min, z_max=self.z_max,
            deadzone_u=self.deadzone_u, deadzone_v=self.deadzone_v,
            clamp=self.clamp)

    def validate(self) -> None:
        self.to_mapping_config().validate()          # 轴/符号/区间合法性
        if self.lambda_prev < 0 or self.lambda_neutral < 0:
            raise ValueError("配置错误: task_space.lambda_prev/lambda_neutral 不能为负")
        if self.max_iters < 1:
            raise ValueError("配置错误: task_space.max_iters 必须 >= 1")
        if not (0.0 < self.tol < 1.0):
            raise ValueError("配置错误: task_space.tol 应在 (0,1)")
        if self.q2_min >= self.q2_max or self.q3_min >= self.q3_max:
            raise ValueError("配置错误: task_space.q2/q3 的 min 必须小于 max")
        if self.limit_margin_rad < 0:
            raise ValueError("配置错误: task_space.limit_margin_rad 不能为负")
        if self.neutral_joints and len(self.neutral_joints) != 6:
            raise ValueError("配置错误: task_space.neutral_joints 必须 6 个值")

    # ------------------------------------------------------------
    def retarget_ranges(self) -> Dict[str, Tuple[float, float]]:
        """本模式实际输出的关节区间（供 limits 三层校验）"""
        return {"joint2": (self.q2_min, self.q2_max),
                "joint3": (self.q3_min, self.q3_max)}


@dataclass
class GeometryConfig:
    """
    手臂几何的基准点选择。

    pivot:
        "shoulder"  以**人体肩关键点**为枢轴（原始做法）。
                    上臂角 = 肩→肘方向，前臂角 = 肘→腕方向。
        "anchor"    以**画面锚点**为枢轴（默认）。
                    上臂角 = 锚点→肘方向，前臂角 = 锚点→腕方向。

    为什么要引入 anchor：
        以人体肩点为枢轴时，「上臂角 / 前臂角」是**肢体自身的朝向**，
        而画面是镜像显示的（mirror=True）。人抬高手臂时，
        镜像画面里手臂朝右上、而原始坐标里朝左上，
        于是「角度变大」在观感上对应手臂**往下**，
        映射到机械臂就成了反直觉的反向弯曲。

        改成画面底部中点作枢轴后：
            腕在手正前方（画面中上方） -> 前臂角 ≈ +90°
            手放下（画面下方）         -> 前臂角 ≈ -90°
        即「越伸展、角度越大」是直接读出来的，不再依赖肢体朝向的符号约定。

        副作用（正面的）：原先三条几何量全都依赖肩关键点，
        肩点一抖，三个量一起抖。anchor 模式下肩点不再参与几何，
        只影响绘制，因此对肩部遮挡/出画更鲁棒。

    anchor_x / anchor_y:
        锚点在画面中的归一化位置（0~1）。
        默认 (0.5, 1.0) = 底边中点。
    """
    pivot: str = "anchor"
    anchor_x: float = 0.5
    anchor_y: float = 1.0

    def validate(self) -> None:
        if self.pivot not in ("shoulder", "anchor"):
            raise ValueError(
                f"配置错误: geometry.pivot 必须是 'shoulder' 或 'anchor'，"
                f"收到 {self.pivot!r}")
        if not (0.0 <= self.anchor_x <= 1.0):
            raise ValueError(
                f"配置错误: geometry.anchor_x 必须在 0~1，收到 {self.anchor_x}")
        if not (0.0 <= self.anchor_y <= 1.0):
            raise ValueError(
                f"配置错误: geometry.anchor_y 必须在 0~1，收到 {self.anchor_y}")


@dataclass
class RobotConfig:
    """
    机械臂侧配置。

    ⚠️ 两个必须成对的量：
        initial_pose   机械臂绝对位置基准
        human_neutral  人体相对角度零点（标定得到）
    必须对应同一时刻的「人-机」状态，否则映射零点错位。
    """
    use_current_as_initial: bool = True
    initial_pose: List[float] = field(default_factory=list)


@dataclass
class SafetyConfig:
    # 注意：这里**没有** max_step_rad。
    # 关节速度/加速度限制统一由 piper_human_control.safety.SafetyLimiter
    # 按 min(a_max*dt, v_max*dt) 每关节独立执行（见 retargeting.yaml 说明）。
    # 曾在本层加过 max_step_rad 粗限幅，因位置在映射之后、滤波之前，
    # 会把滤波器输出再截断一次，导致调 alpha 失效，故已移除。
    go_neutral_on_start: bool = True
    return_home_when_lost: bool = True
    lost_grace_seconds: float = 1.0
    return_home_speed: float = 0.6
    resume_on_reacquire: bool = True


@dataclass
class ControlLoopConfig:
    """控制循环配置"""
    control_hz: float = 30.0
    overrun_warn_ms: float = 100.0

    @property
    def period(self) -> float:
        return 1.0 / max(1e-6, self.control_hz)


@dataclass
class DebugConfig:
    """调试与记录配置"""
    csv_enabled: bool = False
    csv_path: str = "/tmp/piper_teleop_log.csv"
    csv_every_n_frames: int = 1
    show_raw_filtered: bool = True
    stats_every_n_frames: int = 60


@dataclass
class GripperMappingConfig:
    """
    夹爪映射配置（人体 openness -> joint7）。

    mode:
        "fixed"  夹爪固定在 fixed_openness（手部识别不可用时的降级方案）
        "hand"   跟随手掌开合度
    """
    mode: str = "fixed"
    fixed_openness: float = 1.0
    joint7_open: float = 0.035
    joint7_closed: float = 0.0
    min_openness: float = 0.15
    max_openness: float = 0.85
    deadband_m: float = 0.0008
    invert: bool = False

    def to_gripper_config(self) -> GripperConfig:
        """转成控制层的 GripperConfig"""
        return GripperConfig(
            joint7_open=self.joint7_open,
            joint7_closed=self.joint7_closed,
            min_openness=self.min_openness,
            max_openness=self.max_openness,
            deadband_m=self.deadband_m,
        )

    def validate(self) -> None:
        if self.mode not in ("fixed", "hand"):
            raise ValueError(
                f"配置错误: gripper.mode 必须是 'fixed' 或 'hand'，收到 {self.mode}")
        if self.joint7_open <= self.joint7_closed:
            raise ValueError(
                f"配置错误: joint7_open({self.joint7_open}) 必须大于 "
                f"joint7_closed({self.joint7_closed})")
        if self.max_openness <= self.min_openness:
            raise ValueError(
                f"配置错误: max_openness({self.max_openness}) 必须大于 "
                f"min_openness({self.min_openness})")
        if self.deadband_m < 0:
            raise ValueError("配置错误: gripper.deadband_m 不能为负")


# ============================================================
# 主配置
# ============================================================
@dataclass
class RetargetingConfig:
    """映射总配置"""
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    human: HumanConfig = field(default_factory=HumanConfig)
    # 重映射模式：legacy（逐关节，默认保留）| task_space（V1.1 新增）
    # 两套实现并存，可用配置切换，便于 A/B 对比与快速回退。
    retarget_mode: str = "legacy"

    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    task_space: TaskSpaceRetargetConfig = field(
        default_factory=TaskSpaceRetargetConfig)
    arm_direction: ArmDirectionConfig = field(
        default_factory=ArmDirectionConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    control: ControlLoopConfig = field(default_factory=ControlLoopConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    gripper: GripperMappingConfig = field(default_factory=GripperMappingConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)

    # 三级滤波
    filter_keypoint: _BaseFilterConfig = field(
        default_factory=lambda: _BaseFilterConfig(alpha=0.45, deadband=0.0))
    filter_angle: _BaseFilterConfig = field(
        default_factory=lambda: _BaseFilterConfig(alpha=0.35, deadband=1.2))
    filter_joint: _BaseFilterConfig = field(
        default_factory=lambda: _BaseFilterConfig(alpha=0.40, deadband=0.008))

    joint_names: List[str] = field(default_factory=list)
    neutral_joints: List[float] = field(default_factory=list)
    fixed_joints: Dict[str, float] = field(default_factory=dict)
    rules: List[RangeMapping] = field(default_factory=list)

    # 映射阶段取「滤波前」还是「滤波后」的人体角。
    #
    # 三级滤波链的语义应当逐级收紧：
    #     关键点 -> [角度滤波] -> 角度差 -> 区间映射 -> [关节滤波] -> 下发
    # 即区间映射的输入理应是**已滤波**的角度；否则角度滤波器只是被
    # 「顺便记录到 CSV」，对真正下发到机械臂的值毫无影响 —— 那一级滤波
    # 就成了摆设，三级滤波退化为两级。
    #
    # 配置项：mapping:
    #            use_filtered_angles: true
    #
    # 实机实测（400 帧录像，相邻帧抖动中位数）：
    #     下发关节抖动   joint2 -78%，joint3 -100%，joint5 -42%
    #     关节行程       joint2 0.647->0.716，joint3 1.194->1.195，
    #                    joint5 1.291->1.178（基本不变，未被过度滤波）
    # 因此仓库配置默认开启；设为 False 可复现历史行为。
    use_filtered_angles: bool = False

    source_path: str = ""

    # ---------------- 加载 ----------------
    @classmethod
    def from_yaml(cls, path: Optional[str] = None) -> "RetargetingConfig":
        if path is None:
            path = cls.default_yaml_path()
        if not os.path.isfile(path):
            raise FileNotFoundError(f"映射配置不存在: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        cfg = cls._from_dict(raw)
        cfg.source_path = path
        cfg.validate()
        return cfg

    @staticmethod
    def default_yaml_path() -> str:
        if _HAS_AMENT_INDEX:
            try:
                share = get_package_share_directory("piper_human_retargeting")
                cand = os.path.join(share, "config", "retargeting.yaml")
                if os.path.isfile(cand):
                    return cand
            except Exception:                              # noqa: BLE001
                pass
        here = os.path.dirname(os.path.abspath(__file__))
        cand = os.path.join(os.path.dirname(here), "config", "retargeting.yaml")
        if os.path.isfile(cand):
            return cand
        raise FileNotFoundError("找不到 retargeting.yaml")

    # ---------------- 解析 ----------------
    @classmethod
    def _from_dict(cls, raw: dict) -> "RetargetingConfig":
        cal = raw.get("calibration", {}) or {}
        hum = raw.get("human", {}) or {}
        rob = raw.get("robot", {}) or {}
        saf = raw.get("safety", {}) or {}
        ctl = raw.get("control", {}) or {}
        dbg = raw.get("debug", {}) or {}
        grp = raw.get("gripper", {}) or {}
        trk = raw.get("tracker", {}) or {}
        flt = raw.get("filter", {}) or {}
        geo = raw.get("geometry", {}) or {}
        adir = raw.get("arm_direction", {}) or {}
        tsk = raw.get("task_space", {}) or {}

        joint_names = cls._load_joint_names()
        limits = cls._load_joint_limits()
        mp = raw.get("mapping", {}) or {}

        # ---- 映射规则 ----
        rules: List[RangeMapping] = []
        # mapping 段里除了「每个关节一条子配置」还有若干**开关型**键
        # （例如 use_filtered_angles: true）。开关的值是 bool，不是 dict，
        # 必须跳过，否则会被当成一条没有 joint 字段的映射规则。
        for key, r in mp.items():
            if not isinstance(r, dict):
                continue
            joint = str(r.get("joint", ""))
            lo, hi = limits.get(joint, (-math.pi, math.pi))
            rules.append(RangeMapping(
                key=key,
                human=str(r.get("human", "")),
                joint=joint,
                human_min=float(r.get("human_min", -180.0)),
                human_max=float(r.get("human_max", 180.0)),
                robot_min=float(r.get("robot_min", 0.0)),
                robot_max=float(r.get("robot_max", 0.0)),
                invert=bool(r.get("invert", False)),
                sign=float(r.get("sign", 1.0)),
                max_jump=float(r.get("max_jump", 0.0)),
                offset=float(r.get("offset", 0.0)),
                clamp=bool(r.get("clamp", True)),
                enabled=bool(r.get("enabled", True)),
                joint_lower=lo,
                joint_upper=hi,
            ))

        def _flt(name: str, da: float, dd: float) -> _BaseFilterConfig:
            d = flt.get(name, {}) or {}
            return _BaseFilterConfig(
                enabled=bool(d.get("enabled", True)),
                alpha=float(d.get("alpha", da)),
                deadband=float(d.get("deadband", dd)),
                max_jump=float(d.get("max_jump", 0.0)),
            )

        return cls(
            calibration=CalibrationConfig(
                capture_seconds=float(cal.get("capture_seconds", 2.0)),
                require_arm_complete=bool(cal.get("require_arm_complete", True)),
                min_samples=int(cal.get("min_samples", 8)),
                sample_rate_hz=float(cal.get("sample_rate_hz", 30.0)),
            ),
            retarget_mode=str(raw.get("retarget_mode", "legacy")),
            task_space=TaskSpaceRetargetConfig(
                chain_path=str(tsk.get("chain_path", "data/fk_chain.json")),
                table_path=str(tsk.get("table_path", "")),
                horizontal_axis=str(tsk.get("horizontal_axis", "x")),
                vertical_axis=str(tsk.get("vertical_axis", "z")),
                sign_h=float(tsk.get("sign_h", 1.0)),
                sign_v=float(tsk.get("sign_v", 1.0)),
                kx=float(tsk.get("kx", 0.55)),
                kz=float(tsk.get("kz", 0.55)),
                deadzone_u=float(tsk.get("deadzone_u", 0.02)),
                deadzone_v=float(tsk.get("deadzone_v", 0.02)),
                x_min=float(tsk.get("x_min", -0.05)),
                x_max=float(tsk.get("x_max", 0.46)),
                z_min=float(tsk.get("z_min", 0.10)),
                z_max=float(tsk.get("z_max", 0.62)),
                clamp=bool(tsk.get("clamp", True)),
                wx=float(tsk.get("wx", 1.0)),
                wz=float(tsk.get("wz", 1.0)),
                lambda_prev=float(tsk.get("lambda_prev", 0.35)),
                lambda_neutral=float(tsk.get("lambda_neutral", 0.02)),
                max_iters=int(tsk.get("max_iters", 12)),
                tol=float(tsk.get("tol", 1e-4)),
                q2_min=float(tsk.get("q2_min", 0.30)),
                q2_max=float(tsk.get("q2_max", 2.30)),
                q3_min=float(tsk.get("q3_min", -2.20)),
                q3_max=float(tsk.get("q3_max", -0.40)),
                limit_margin_rad=float(tsk.get("limit_margin_rad", 0.10)),
                neutral_joints=[float(v) for v in
                                (tsk.get("neutral_joints") or [])],
                jump_warn_rad=float(tsk.get("jump_warn_rad", 0.25)),
            ),
            arm_direction=ArmDirectionConfig(
                enabled=bool(adir.get("enabled", False)),
                table_path=str(adir.get("table_path", "")),
                gain=float(adir.get("gain", 1.0)),
                neutral_dir_deg=float(adir.get("neutral_dir_deg", 0.0)),
                reach_from_elbow=bool(adir.get("reach_from_elbow", True)),
                elbow_straight_deg=float(adir.get("elbow_straight_deg", 20.0)),
                elbow_bent_deg=float(adir.get("elbow_bent_deg", 140.0)),
                reach_extended=float(adir.get("reach_extended", 0.50)),
                reach_folded=float(adir.get("reach_folded", 0.20)),
            ),
            geometry=GeometryConfig(
                pivot=str(geo.get("pivot", "anchor")),
                anchor_x=float(geo.get("anchor_x", 0.5)),
                anchor_y=float(geo.get("anchor_y", 1.0)),
            ),
            human=HumanConfig(
                side=str(hum.get("side", "right")),
                min_confidence=float(hum.get("min_confidence", 0.4)),
                require_complete=bool(hum.get("require_complete", True)),
            ),
            robot=RobotConfig(
                use_current_as_initial=bool(rob.get("use_current_as_initial", True)),
                initial_pose=[float(v) for v in
                              (rob.get("initial_pose")
                               or rob.get("neutral_joints") or [])],
            ),
            safety=SafetyConfig(
                go_neutral_on_start=bool(saf.get("go_neutral_on_start", True)),
                return_home_when_lost=bool(saf.get("return_home_when_lost", True)),
                lost_grace_seconds=float(saf.get("lost_grace_seconds", 1.0)),
                return_home_speed=float(saf.get("return_home_speed", 0.6)),
                resume_on_reacquire=bool(saf.get("resume_on_reacquire", True)),
            ),
            control=ControlLoopConfig(
                control_hz=float(ctl.get("control_hz", 30.0)),
                overrun_warn_ms=float(ctl.get("overrun_warn_ms", 100.0)),
            ),
            debug=DebugConfig(
                csv_enabled=bool(dbg.get("csv_enabled", False)),
                csv_path=str(dbg.get("csv_path", "/tmp/piper_teleop_log.csv")),
                csv_every_n_frames=int(dbg.get("csv_every_n_frames", 1)),
                show_raw_filtered=bool(dbg.get("show_raw_filtered", True)),
                stats_every_n_frames=int(dbg.get("stats_every_n_frames", 60)),
            ),
            gripper=GripperMappingConfig(
                mode=str(grp.get("mode", "fixed")),
                fixed_openness=float(grp.get("fixed_openness", 1.0)),
                joint7_open=float(grp.get("joint7_open", 0.035)),
                joint7_closed=float(grp.get("joint7_closed", 0.0)),
                min_openness=float(grp.get("min_openness", 0.15)),
                max_openness=float(grp.get("max_openness", 0.85)),
                deadband_m=float(grp.get("deadband_m", 0.0008)),
                invert=bool(grp.get("invert", False)),
            ),
            tracker=TrackerConfig(
                short_lost_seconds=float(trk.get("short_lost_seconds", 0.5)),
                long_lost_seconds=float(trk.get("long_lost_seconds", 3.0)),
                min_confidence=float(trk.get("min_confidence", 0.4)),
                required_keypoints=tuple(
                    trk.get("required_keypoints") or
                    # 默认值随枢轴走：
                    #   anchor 模式下肩点不参与几何，再强制要求它可信
                    #   就把这个模式唯一的鲁棒性优势抵消掉了
                    #   （肩部遮挡/出画时明明能算，却被状态机拦成 LOST）。
                    (("shoulder", "elbow", "wrist")
                     if str(geo.get("pivot", "anchor")) == "shoulder"
                     else ("elbow", "wrist"))),
            ),
            filter_keypoint=_flt("keypoint", 0.45, 0.0),
            filter_angle=_flt("angle", 0.35, 1.2),
            filter_joint=_flt("joint", 0.40, 0.008),
            joint_names=joint_names,
            use_filtered_angles=bool(mp.get("use_filtered_angles", False)),
            neutral_joints=[float(v) for v in rob.get("neutral_joints", [])],
            fixed_joints={k: float(v)
                          for k, v in (rob.get("fixed_joints", {}) or {}).items()},
            rules=rules,
        )

    # ---------------- 复用阶段二的权威配置 ----------------
    @staticmethod
    def _load_joint_names() -> List[str]:
        """关节顺序复用阶段二配置，避免两处定义不一致"""
        try:
            from piper_human_control import ControlConfig
            return list(ControlConfig.from_yaml().joint_names)
        except Exception:                                  # noqa: BLE001
            return ["joint1", "joint2", "joint3", "joint4",
                    "joint5", "joint6"]

    @staticmethod
    def _load_joint_limits() -> Dict[str, tuple]:
        """
        关节限位**只从阶段二取**，不在这里再维护一份。

        为什么：两份限位表迟早会不一致，而「安全限位不一致」
        是最危险的一类 bug。
        """
        try:
            from piper_human_control import ControlConfig
            c = ControlConfig.from_yaml()
            return {n: (c.lower_limits[n], c.upper_limits[n])
                    for n in c.joint_names}
        except Exception:                                  # noqa: BLE001
            return {}

    # ---------------- 校验 ----------------
    def validate(self) -> None:
        n = len(self.joint_names)
        if not n:
            raise ValueError("配置错误: 关节名列表为空")
        self.geometry.validate()
        self.arm_direction.validate()
        self.task_space.validate()
        if self.retarget_mode not in ("legacy", "task_space"):
            raise ValueError(
                f"配置错误: retarget_mode 必须是 'legacy' 或 'task_space'，"
                f"收到 {self.retarget_mode!r}")
        if self.retarget_mode == "task_space" and not self.task_space.chain_path:
            raise ValueError(
                "配置错误: retarget_mode=task_space 但未给 task_space.chain_path")
        if self.task_space.neutral_joints and len(self.task_space.neutral_joints) != n:
            raise ValueError("配置错误: task_space.neutral_joints 长度必须等于关节数")
        if len(self.neutral_joints) != n:
            raise ValueError(
                f"配置错误: neutral_joints 长度 {len(self.neutral_joints)} != {n}")
        if self.robot.initial_pose and len(self.robot.initial_pose) != n:
            raise ValueError(
                f"配置错误: initial_pose 长度 {len(self.robot.initial_pose)} != {n}")
        if not self.rules:
            raise ValueError("配置错误: mapping 为空")

        seen: Dict[str, str] = {}
        for r in self.rules:
            if r.human not in VALID_HUMAN_MEASURES:
                raise ValueError(
                    f"配置错误: 规则 {r.key} 的 human='{r.human}' 非法，"
                    f"可选 {sorted(VALID_HUMAN_MEASURES)}")
            if r.joint not in self.joint_names:
                raise ValueError(
                    f"配置错误: 规则 {r.key} 的 joint='{r.joint}' 不在关节列表中")
            r.validate()                                   # RangeMapping 自校验
            if r.enabled:
                if r.joint in seen:
                    raise ValueError(
                        f"配置错误: 关节 {r.joint} 被 {seen[r.joint]} 与 "
                        f"{r.key} 同时控制")
                seen[r.joint] = r.key

        for name, val in self.fixed_joints.items():
            if name not in self.joint_names:
                raise ValueError(f"配置错误: fixed_joints 中的 {name} 不是有效关节")
            if name in seen:
                raise ValueError(
                    f"配置错误: 关节 {name} 既在 mapping 又在 fixed_joints")

        # ---- 标定基准一致性校验（重要）----
        # 标定姿势对应 Δ=0。此时机械臂**应当停在该关节的 neutral 位姿**。
        # 若 human 区间不相对 Δ=0 对称，或 robot 区间中点 ≠ neutral，
        # 则 Δ=0 会落在 ratio≠0.5，机械臂一启动就偏离中立位置。
        #
        # 实测踩过这个坑：human=[20,150]（中点 85）而标定基准在 90 附近，
        # Δ=0 时 ratio=-0.15 被 clamp 到 0，关节直接跑到区间端点。
        # 因此在这里**启动即报错**，而不是等到运行时才发现动作不对。
        for r in self.enabled_rules():
            mid = 0.5 * (r.robot_min + r.robot_max)
            if r.invert:
                mid = r.robot_min + r.robot_max - mid
            mid += r.offset
            mid = max(r.joint_lower, min(r.joint_upper, mid))
            neutral_j = self.neutral_for(r.joint)
            if abs(mid - neutral_j) > 1e-6:
                raise ValueError(
                    f"配置错误: 规则 {r.key} 在 Δ=0 时输出 {mid:.4f}，"
                    f"但 {r.joint} 的 neutral 是 {neutral_j:.4f}。\n"
                    f"  标定姿势应让机械臂停在中立位姿，因此要求：\n"
                    f"    (1) human_min = -human_max（相对 Δ=0 对称），且\n"
                    f"    (2) (robot_min + robot_max)/2 (含 invert/offset) = neutral\n"
                    f"  当前 human=[{r.human_min}, {r.human_max}], "
                    f"robot=[{r.robot_min}, {r.robot_max}], "
                    f"invert={r.invert}, offset={r.offset}")

        self.filter_keypoint.validate("filter.keypoint")
        self.filter_angle.validate("filter.angle")
        self.filter_joint.validate("filter.joint")
        self.tracker.validate()
        self.gripper.validate()

        if self.control.control_hz <= 0:
            raise ValueError("配置错误: control_hz 必须 > 0")
        if not (0.0 < self.safety.return_home_speed <= 1.0):
            raise ValueError("配置错误: return_home_speed 必须在 (0, 1]")
        if self.safety.lost_grace_seconds < 0:
            raise ValueError("配置错误: lost_grace_seconds 不能为负")

    # ---------------- 便捷 ----------------
    def neutral_for(self, joint: str) -> float:
        return self.active_neutral_joints()[self.joint_names.index(joint)]

    # ---- task_space 的实验 neutral（不动 legacy 的 neutral_joints）----
    def active_neutral_joints(self) -> List[float]:
        """
        当前模式实际使用的中立位姿。

        task_space 模式下若配置了 `task_space.neutral_joints`（由 workspace
        扫描的评分搜索给出），就用它；否则沿用 `robot.neutral_joints`。
        legacy 模式**永远**用 `robot.neutral_joints`（保证可回退、行为不变）。
        """
        if (self.retarget_mode == "task_space"
                and self.task_space.neutral_joints
                and len(self.task_space.neutral_joints) == len(self.joint_names)):
            return list(self.task_space.neutral_joints)
        return list(self.neutral_joints)

    def active_initial_pose(self) -> List[float]:
        """
        当前模式实际使用的**初始位姿**。

        ⚠️ 为什么单独一个方法（实时测试踩到的 bug）：
            `robot.initial_pose` 在解析时会被 `robot.neutral_joints` 填充，
            因此它**永远非空**。若启动逻辑写成
                nj = cfg.robot.initial_pose or cfg.active_neutral_joints()
            就会一直拿到 legacy 的 neutral，而 task-space 的
            IK 基线 (x0,z0) 用的是 `active_neutral_joints()` ——
            两处基准不一致：机械臂停在 legacy 位姿，映射却以为它在
            task-space 位姿，开机就带一个固定偏差。
        """
        if (self.retarget_mode == "task_space"
                and self.task_space.neutral_joints):
            return list(self.task_space.neutral_joints)
        return list(self.robot.initial_pose or self.neutral_joints)

    def active_joint_names(self) -> List[str]:
        """
        **当前模式下真正由重映射决定的全部关节**（按 joint_names 顺序）。

        = controlled_joints()（有 legacy 规则且启用的关节）
        ∪ task_space_joints()（task_space 下由 IK 决定的 joint2/joint3）

        ⚠️ 必须统一用这个集合，不能到处写 controlled_joints()：
            task_space 模式下 joint2/joint3 的 legacy 规则被停用，
            它们不在 controlled_joints 里。实测因此踩过三次同类 bug：
              ① CSV 的 joint2/joint3 列恒 0（看起来像没被控制）
              ② 关节丢失回原位时**只回腕关节、大臂冻住**
              ③ HUD/状态行看不到 J2/J3
            凡是"遍历受控关节"的地方都要用它。
        """
        out = list(self.controlled_joints())
        for j in self.task_space_joints():
            if j not in out:
                out.append(j)
        order = {n: i for i, n in enumerate(self.joint_names)}
        out.sort(key=lambda n: order.get(n, 999))
        return out

    def task_space_joints(self) -> List[str]:
        """task_space 模式下由 IK 决定的关节（这些关节的 legacy 规则必须让位）"""
        return ["joint2", "joint3"] if self.retarget_mode == "task_space" else []

    def enabled_rules(self) -> List[RangeMapping]:
        """
        生效的映射规则。

        ⚠️ task_space 模式下 joint2/joint3 由 IK 决定，
        对应的 legacy 规则（上臂角->j2、肘夹角->j3）**必须停用** ——
        否则又变成"肘角直接决定 J3"，与 IK 冲突（任务书 §15）。
        停用只影响本模式的输出；规则本身仍留在配置里，legacy 模式照旧。
        """
        ts_joints = set(self.task_space_joints())
        return [r for r in self.rules if r.enabled and r.joint not in ts_joints]

    def controlled_joints(self) -> List[str]:
        return [r.joint for r in self.enabled_rules()]

    @staticmethod
    def circular_measures() -> set:
        """周期量（角度）集合 —— 交给滤波层决定用哪种滤波"""
        return set(CIRCULAR_HUMAN_MEASURES)

    def measure_max_jump(self) -> Dict[str, float]:
        """
        各人体测量量的单帧突变门限（与量纲一致，度）。0 表示不限制。

        为什么是「每量」而不是全局一个值：
            wrist_pitch_deg 的假跳变可达 353°（关键点被误估到对侧），
            palm_roll_deg 则本身就会跨 ±180°、真实单帧变化能到 100°+。
            用同一个门限，要么放过腕俯仰的假跳变，要么把掌滚转的真实
            转动当跳变丢掉。实机数据同时踩到过这两种。
        """
        out: Dict[str, float] = {}
        for r in self.enabled_rules():
            v = float(r.max_jump)
            if v > 0:
                out[r.human] = min(out[r.human], v) if r.human in out else v
            else:
                out.setdefault(r.human, 0.0)
        return out

    def measure_sign(self) -> Dict[str, float]:
        """
        各人体测量量的符号修正（在「做差」之前施加）。

        为什么需要它（实机实测）：
            以人体肩点为枢轴时，前臂角是「肘->腕」的朝向。
            实测 corr(腕的高 low, 前臂角) = **+0.561** ——
            即**手往下移，前臂角反而变大**。映射到机械臂就成了
            「人抬手、机器人缩回」的反直觉行为。

            在测量量上乘 -1 后，「手抬高 = 数值变大」，方向就正过来了。

        同一个人体量可能被多条规则引用，此时取它们的 sign 之积；
        单个人体量对应单条规则时就是那个规则的 sign。
        """
        out: Dict[str, float] = {}
        for r in self.enabled_rules():
            out[r.human] = out.get(r.human, 1.0) * r.sign
        return out

    def used_measures(self) -> List[str]:
        """
        需要采集/滤波/标定的人体测量量。

        ⚠️ 除了「规则引用到的量」，还要把**方向驱动映射用到**的量算进来。
        `arm_direction_deg` 没有对应的映射规则（它由
        ArmDirectionConfig 直接驱动 joint2/joint3），
        所以不会出现在 enabled_rules 里 ——
        曾经因此**永远取不到该量**，方向映射静默失效：
        日志显示「标定完成」，但关节一次都没下发。

        教训：凡是链路里会用到的量，都必须出现在这个集合里，
        否则滤波/标定/缺失检查都会把它当不存在。
        """
        out: List[str] = []
        for r in self.enabled_rules():
            if r.human not in out:
                out.append(r.human)
        if self.arm_direction.enabled and "arm_direction_deg" not in out:
            out.append("arm_direction_deg")
        # task_space 模式用 human_u/human_v 驱动 J2/J3（经 IK），
        # 它们没有对应的映射规则 —— 必须显式加进来，
        # 否则滤波/标定/缺失检查都会把它们当不存在（这个坑踩过多次：
        # 曾经"标定完成"但关节一次没动，就是因为量不在这个集合里）。
        if self.retarget_mode == "task_space":
            for m in ("human_u", "human_v"):
                if m not in out:
                    out.append(m)
            # 诊断/记录量：不驱动任何关节，但必须滤波、标定、进 CSV。
            #   肘角/上臂角：将来作为 IK 姿态偏好（任务书 §16），本轮只记录
            #   reach/elevation：伸展程度与抬高程度，用于评估与画图
            for m in ("elbow_angle_deg", "upper_arm_angle_deg", "reach"):
                if m not in out:
                    out.append(m)
        return out

    def rule_for(self, joint: str) -> Optional[RangeMapping]:
        for r in self.enabled_rules():
            if r.joint == joint:
                return r
        return None

    def __str__(self) -> str:
        lines = [
            "RetargetingConfig(",
            f"  source={self.source_path}",
            f"  side={self.human.side}  min_conf={self.human.min_confidence}",
            f"  neutral={[round(v, 3) for v in self.neutral_joints]}",
            f"  初始位姿={[round(v, 3) for v in self.robot.initial_pose]}"
            f" (use_current={self.robot.use_current_as_initial})",
            f"  控制频率={self.control.control_hz}Hz",
            f"  固定关节={self.fixed_joints}",
            "  映射:",
        ]
        for r in self.enabled_rules():
            lines.append(
                f"    {r.key:4s} {r.human:22s} -> {r.joint:8s} "
                f"Δ[{r.human_min:+.0f},{r.human_max:+.0f}]° -> "
                f"[{r.robot_min:+.2f},{r.robot_max:+.2f}]rad "
                f"invert={r.invert} offset={r.offset:+.2f} "
                f"(scale={r.scale:+.5f} rad/deg)")
        lines += [
            f"  滤波: kp(a={self.filter_keypoint.alpha},"
            f"db={self.filter_keypoint.deadband}) "
            f"angle(a={self.filter_angle.alpha},"
            f"db={self.filter_angle.deadband}) "
            f"joint(a={self.filter_joint.alpha},"
            f"db={self.filter_joint.deadband})",
            f"  跟踪: short={self.tracker.short_lost_seconds}s "
            f"long={self.tracker.long_lost_seconds}s "
            f"conf>={self.tracker.min_confidence}",
            f"  夹爪: mode={self.gripper.mode} "
            f"open=[{self.gripper.min_openness},{self.gripper.max_openness}] "
            f"-> joint7=[{self.gripper.joint7_closed},{self.gripper.joint7_open}]m",
            f"  丢失: "
            f"{'平滑回初始位姿' if self.safety.return_home_when_lost else '冻结保持'}"
            f" 宽限={self.safety.lost_grace_seconds}s"
            f" 速度={self.safety.return_home_speed}",
            ")",
        ]
        return "\n".join(lines)
