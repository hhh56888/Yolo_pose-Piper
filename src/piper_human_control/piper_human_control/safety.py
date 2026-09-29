# -*- coding: utf-8 -*-
"""
piper_human_control.safety
==========================
安全处理模块：限位、速度/加速度限制、合法性检查。

本模块是「纯函数式」的：不持有 ROS2 连接，不做 IO，
输入目标 + 当前状态 + 时间步，输出安全后的目标 + 处理报告。
这样可以在阶段五被实时控制循环直接复用，也便于单元测试。

处理顺序（顺序很重要）：
    1. 合法性检查   —— 长度、NaN/Inf
    2. 位置限位     —— clip 到 [lower, upper]
    3. 速率限制     —— 限制单个周期内的增量（由加速度与周期推导）
    4. 单步上限     —— 兜底，防止任何情况下的跳变
"""

import math
import time
from typing import List, Optional, Sequence

from .config import ControlConfig
from .types import JointTarget, SafetyReport


class SafetyLimiter:
    """
    关节目标安全限制器。

    使用方式（有状态，用于流式控制）：
        limiter = SafetyLimiter(cfg)
        report = limiter.apply(target, current_positions, dt)
    """

    def __init__(self, cfg: ControlConfig):
        self.cfg = cfg
        self._last_valid: Optional[List[float]] = None

    # ------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------
    def apply(
        self,
        target: Sequence[float],
        current: Optional[Sequence[float]],
        dt: float,
        label: str = "",
    ) -> SafetyReport:
        """
        对目标做安全检查与限制。

        Args:
            target:  期望关节角 (rad)
            current: 当前实测关节角 (rad)；None 表示尚无状态反馈
            dt:      距上一次下发的周期时长 (s)，用于速率限制
            label:   用途标记，仅用于日志

        Returns:
            SafetyReport，其中 positions 为可直接下发的安全目标
        """
        report = SafetyReport()

        # ---------- 1. 合法性检查 ----------
        if target is None:
            report.rejected = True
            report.reason = "target 为 None"
            return report

        vals = list(target)
        if len(vals) != self.cfg.num_joints:
            report.rejected = True
            report.reason = (f"关节数不匹配: 期望 {self.cfg.num_joints}, "
                             f"收到 {len(vals)}")
            return report

        for i, v in enumerate(vals):
            if v is None or not math.isfinite(v):
                report.rejected = True
                report.reason = f"关节 {self.cfg.joint_names[i]} 目标值非法: {v}"
                return report

        # ---------- 2. 位置限位 ----------
        clipped = []
        for i, name in enumerate(self.cfg.joint_names):
            lo = self.cfg.lower_limits[name]
            hi = self.cfg.upper_limits[name]
            v = vals[i]
            if v < lo:
                report.clamped_joints.append(name)
                v = lo
            elif v > hi:
                report.clamped_joints.append(name)
                v = hi
            clipped.append(v)

        # ---------- 3. 速率限制 ----------
        # 依据加速度上限与周期，推导本周期允许的最大增量：
        #     dq_max = a_max * dt
        # 同时受速度上限约束：dq_max <= v_max * dt
        rate = clipped
        if current is not None and len(current) == self.cfg.num_joints and dt > 0.0:
            rate = []
            for i, name in enumerate(self.cfg.joint_names):
                a_max = self.cfg.acceleration_limits[name]
                v_max = self.cfg.velocity_limits[name]
                dq_max = min(a_max * dt, v_max * dt)

                cur = current[i]
                tgt = clipped[i]
                delta = tgt - cur

                if abs(delta) > dq_max:
                    report.rate_limited_joints.append(name)
                    tgt = cur + math.copysign(dq_max, delta)
                rate.append(tgt)

        # ---------- 4. 单步上限兜底 ----------
        step_cap = self.cfg.safety.max_step_per_cycle_rad
        final = []
        has_state = current is not None and len(current) == self.cfg.num_joints
        for i, name in enumerate(self.cfg.joint_names):
            ref = rate[i]
            if has_state:
                delta = ref - current[i]
                if abs(delta) > step_cap:
                    if name not in report.rate_limited_joints:
                        report.rate_limited_joints.append(name)
                    ref = current[i] + math.copysign(step_cap, delta)
            final.append(ref)

        report.positions = final
        self._last_valid = list(final)
        return report

    # ------------------------------------------------------------
    # 辅助检查
    # ------------------------------------------------------------
    def in_limits(self, positions: Sequence[float]) -> bool:
        """判断一组关节角是否全部在限位内"""
        if len(positions) != self.cfg.num_joints:
            return False
        for i, name in enumerate(self.cfg.joint_names):
            if not (self.cfg.lower_limits[name] <= positions[i]
                    <= self.cfg.upper_limits[name]):
                return False
        return True

    def describe_violations(self, positions: Sequence[float]) -> List[str]:
        """返回所有越界关节的描述，便于报错"""
        out = []
        if len(positions) != self.cfg.num_joints:
            return [f"关节数不匹配: {len(positions)} != {self.cfg.num_joints}"]
        for i, name in enumerate(self.cfg.joint_names):
            lo = self.cfg.lower_limits[name]
            hi = self.cfg.upper_limits[name]
            v = positions[i]
            if v < lo or v > hi:
                out.append(f"{name}={v:+.4f} 超出 [{lo:+.4f}, {hi:+.4f}]")
        return out

    def max_reachable_delta(self, dt: float) -> List[float]:
        """
        当前周期各关节允许的最大增量，用于诊断「为什么没到位」。
        """
        out = []
        for name in self.cfg.joint_names:
            a_max = self.cfg.acceleration_limits[name]
            v_max = self.cfg.velocity_limits[name]
            out.append(min(a_max * dt, v_max * dt))
        return out

    def limit_home(self) -> List[float]:
        """返回裁剪到限位内的 HOME 位姿"""
        return [self.cfg.clip(n, self.cfg.home[i])
                for i, n in enumerate(self.cfg.joint_names)]


class EmergencyStop:
    """
    软件急停。

    设计取舍（对应任务书阶段五第 7 条「预留软件急停接口」）：
      急停触发后：锁定当前目标不再更新，机械臂保持在当前位置。
      急停解除后：**不会**自动恢复运动，必须显式 reset，
                  以避免意外恢复造成危险。
    """

    def __init__(self):
        self._engaged = False
        self._engaged_at: Optional[float] = None
        self._reason = ""

    @property
    def engaged(self) -> bool:
        return self._engaged

    @property
    def reason(self) -> str:
        return self._reason

    @property
    def engaged_duration(self) -> float:
        if self._engaged_at is None:
            return 0.0
        return time.monotonic() - self._engaged_at

    def engage(self, reason: str = "") -> None:
        """触发急停"""
        if not self._engaged:
            self._engaged = True
            self._engaged_at = time.monotonic()
        self._reason = reason

    def reset(self) -> None:
        """解除急停（需显式调用）"""
        self._engaged = False
        self._engaged_at = None
        self._reason = ""
