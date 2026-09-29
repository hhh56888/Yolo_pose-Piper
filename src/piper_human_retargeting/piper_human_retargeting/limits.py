# -*- coding: utf-8 -*-
"""
piper_human_retargeting.limits
==============================
关节限位的**三层结构**（本轮 V1.1 要求）：

    physical   —— 真实物理限位：来自当前实际加载的 URDF（robot_description）
    safe       —— 安全限位：physical 两端各留 margin（默认 0.10 rad）
    retarget   —— 重映射输出区间：**必须** ⊂ safe

为什么必须分三层（实测教训，见 `docs/J3_Elbow_Mapping_Audit.md`）：
    * joint3 的映射上端 `robot_max = 0.00` **正好等于**物理上限 0.00
      → live 运行 15.25% 的帧骑在限位上；
    * joint2 的映射下端 `robot_min = −1.00` **小于**物理下限 0.00
      → 该方向 1.0 rad 行程不可达，被安全层削成 0（29.67% 的帧被裁）
    即：重映射区间越过物理限位时，输出会被下游静默裁剪，
    表现为「到某个方向就顶住不动」。三层结构把这类问题**在配置校验期**暴露。

注意：legacy 模式下这两处越界是**已知且暂时保留**的（用户要求 legacy 可回退、行为不变），
      `audit()` 会把它报出来，但不会让 legacy 启动失败；
      task_space 模式则**强制**满足嵌套关系。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .kinematics import ChainModel


@dataclass
class LimitLayer:
    """三层限位的快照 + 审计结果"""
    physical: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    safe: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    retarget: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    margin_rad: float = 0.10
    source: str = ""
    violations: List[str] = field(default_factory=list)

    # ------------------------------------------------------------
    def check_nesting(self) -> List[str]:
        """
        校验 physical ⊇ safe ⊇ retarget，返回违规描述列表（空 = 全部合法）。
        """
        bad: List[str] = []
        for j, (slo, shi) in self.safe.items():
            p = self.physical.get(j)
            if p is None:
                bad.append(f"{j}: safe 区间存在但 physical 缺失")
                continue
            if slo < p[0] - 1e-9 or shi > p[1] + 1e-9:
                bad.append(f"{j}: safe [{slo:.3f},{shi:.3f}] 越出 physical "
                           f"[{p[0]:.3f},{p[1]:.3f}]")
        for j, (rlo, rhi) in self.retarget.items():
            s = self.safe.get(j)
            if s is None:
                bad.append(f"{j}: retarget 区间存在但 safe 缺失")
                continue
            if rlo < s[0] - 1e-9 or rhi > s[1] + 1e-9:
                bad.append(f"{j}: retarget [{rlo:.3f},{rhi:.3f}] 越出 safe "
                           f"[{s[0]:.3f},{s[1]:.3f}]")
        self.violations = bad
        return bad

    def margins(self, q: Dict[str, float]) -> Dict[str, Tuple[float, float]]:
        """给定关节角，返回距 safe 上下界的余量 (to_lower, to_upper)"""
        out = {}
        for j, v in q.items():
            if j not in self.safe:
                continue
            lo, hi = self.safe[j]
            out[j] = (float(v - lo), float(hi - v))
        return out

    def as_dict(self) -> dict:
        return {
            "margin_rad": self.margin_rad,
            "source": self.source,
            "physical": {k: list(v) for k, v in self.physical.items()},
            "safe": {k: list(v) for k, v in self.safe.items()},
            "retarget": {k: list(v) for k, v in self.retarget.items()},
            "violations": list(self.violations),
        }

    def table(self) -> str:
        lines = ["%-8s %-22s %-22s %-22s" % ("joint", "physical", "safe", "retarget")]
        for j in sorted(self.physical):
            p = self.physical[j]
            s = self.safe.get(j, (float("nan"), float("nan")))
            r = self.retarget.get(j)
            rs = "-" if r is None else "[%+.3f, %+.3f]" % r
            lines.append("%-8s [%+.3f, %+.3f]      [%+.3f, %+.3f]      %-22s"
                         % (j, p[0], p[1], s[0], s[1], rs))
        return "\n".join(lines)


DEFAULT_MARGIN_RAD = 0.10


def physical_limits(model: Optional[ChainModel] = None,
                    cross_check_yaml: bool = True
                    ) -> Tuple[Dict[str, Tuple[float, float]], str]:
    """
    取物理限位：**以实际加载的 URDF 为准**，并与控制侧 joint_limits.yaml 交叉验证。

    两者不一致时**报错**而不是二选一 —— 历史上「两份限位表不一致」是最危险的一类 bug。
    """
    if model is None:
        from .kinematics import load_default_chain
        model = load_default_chain()
    phys = model.limits()
    src = "URDF(robot_description) md5=%s" % model.source.get("urdf_md5", "?")

    if cross_check_yaml:
        try:
            from piper_human_control import ControlConfig
            c = ControlConfig.from_yaml()
            # 容差 1e-3 rad（≈0.06°）：joint_limits.yaml 里的十进制值是对
            # URDF 做四舍五入写下的（joint1: 2.61800 vs 2.61790），
            # 属记录精度差异，不是"两份限位表冲突"。
            # 真正的冲突（例如差 0.1 rad 以上）仍会直接报错。
            tol = 1e-3
            worst = 0.0
            for name in phys:
                if name not in c.joint_names:
                    continue
                y_lo, y_hi = c.lower_limits[name], c.upper_limits[name]
                worst = max(worst, abs(y_lo - phys[name][0]),
                            abs(y_hi - phys[name][1]))
                if (abs(y_lo - phys[name][0]) > tol
                        or abs(y_hi - phys[name][1]) > tol):
                    raise ValueError(
                        "关节限位不一致: %s URDF=[%.4f,%.4f] "
                        "joint_limits.yaml=[%.4f,%.4f]"
                        % (name, phys[name][0], phys[name][1], y_lo, y_hi))
            src += " + joint_limits.yaml(一致, 最大偏差 %.1e rad)" % worst
        except ImportError:
            src += " + joint_limits.yaml(不可用，跳过交叉验证)"
    return phys, src


def safe_limits(phys: Dict[str, Tuple[float, float]],
                margin: float = DEFAULT_MARGIN_RAD
                ) -> Dict[str, Tuple[float, float]]:
    """两端各留 margin 的安全区间；退化区间直接报错"""
    out = {}
    for j, (lo, hi) in phys.items():
        if hi - lo <= 2.0 * margin:
            raise ValueError(f"{j} 行程 {hi - lo:.3f} 不足以留 {margin} 的余量")
        out[j] = (lo + margin, hi - margin)
    return out


def build_layers(retarget: Dict[str, Tuple[float, float]],
                 margin: float = DEFAULT_MARGIN_RAD,
                 model: Optional[ChainModel] = None) -> LimitLayer:
    """组装三层并做嵌套校验（不抛异常，violations 里给结论）"""
    phys, src = physical_limits(model)
    layer = LimitLayer(physical=phys, safe=safe_limits(phys, margin),
                       retarget=dict(retarget), margin_rad=margin, source=src)
    layer.check_nesting()
    return layer
