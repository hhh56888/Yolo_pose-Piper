# -*- coding: utf-8 -*-
"""
piper_human_retargeting.tracker_state
=====================================
人体跟踪状态机：TRACKING / LOST_SHORT / LOST_LONG。

为什么需要状态机（而不是几个 if）：
    「识别失败怎么办」看似简单，实际有三种完全不同的语义，
    混在一起写很容易出现「一闪就回位」「丢久了机械臂僵在原地」这类问题：

        TRACKING    正常控制
        LOST_SHORT  短暂丢失 -> **保持最后位置**（不要动！）
        LOST_LONG   长时间丢失 -> 进入安全状态

    ★ 关于 LOST_LONG 的语义，本项目做了一个明确选择：
      **不执行 HOME，也不主动回位，而是冻结最后一个有效目标。**

    理由：
        1. 本项目的机械臂是遥操作设备，人不在时最安全的做法是「不动」；
        2. 主动回 HOME 是一次大幅运动，在人可能突然回到画面的场景下
           反而危险（机械臂正在动，人回来了，映射点又跳回去）；
        3. 用户如果需要「丢久了回初始位姿」，那是**上层策略**，
           由 retargeter 的 return_home 配置决定（默认开启，
           且是平滑回位而非硬跳）。
     所以本状态机只负责「状态判定」，不负责「动作选择」——
     动作由调用方根据状态决定，这样策略可替换。

状态转移：

             连续有效帧
    ┌──────────────────────────────┐
    │                              │
    ▼        丢失 < short_s        │
  TRACKING ────────────────► LOST_SHORT
    ▲                              │
    │  重新有效                     │ 丢失 >= short_s
    └──────────────────────────────┤
                                   ▼
                              LOST_LONG
    ▲                              │
    └──────────────────────────────┘
              重新有效
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class TrackState(Enum):
    """跟踪状态"""
    TRACKING = "TRACKING"        # 正常跟踪
    LOST_SHORT = "LOST_SHORT"    # 短暂丢失：保持最后位置
    LOST_LONG = "LOST_LONG"      # 长时间丢失：进入安全状态

    def __str__(self) -> str:
        return self.value


@dataclass
class TrackerConfig:
    """
    状态机配置。

    Attributes:
        short_lost_seconds: 丢失多久之内算 LOST_SHORT（默认 0.5s）
        long_lost_seconds:  丢失多久之后算 LOST_LONG（默认 3.0s）
        min_confidence:     关键点置信度门限
        required_keypoints: 要求哪些关键点都可信才算有效帧
    """
    short_lost_seconds: float = 0.5
    long_lost_seconds: float = 3.0
    min_confidence: float = 0.4
    required_keypoints: tuple = ("shoulder", "elbow", "wrist")

    def validate(self) -> None:
        if self.short_lost_seconds < 0 or self.long_lost_seconds < 0:
            raise ValueError("配置错误: 丢失时间阈值不能为负")
        if self.long_lost_seconds < self.short_lost_seconds:
            raise ValueError(
                f"配置错误: long_lost_seconds({self.long_lost_seconds}) "
                f"必须 >= short_lost_seconds({self.short_lost_seconds})")
        if not (0.0 <= self.min_confidence <= 1.0):
            raise ValueError("配置错误: min_confidence 必须在 [0,1]")


class TrackerStateMachine:
    """
    跟踪状态机（有状态）。

    典型用法：
        sm = TrackerStateMachine(cfg)
        # 每帧：
        st = sm.update(arm, now=time.monotonic())
        if st is TrackState.TRACKING:
            ...正常映射...
        elif st is TrackState.LOST_SHORT:
            ...保持最后位置...
        else:
            ...安全状态...
    """

    def __init__(self, cfg: TrackerConfig):
        cfg.validate()
        self.cfg = cfg
        self.state = TrackState.TRACKING
        self._lost_since: Optional[float] = None
        self._last_time: Optional[float] = None
        self._last_reason: str = ""
        self._last_good_time: Optional[float] = None
        # 统计
        self.n_tracking = 0
        self.n_lost_short = 0
        self.n_lost_long = 0
        self.n_transitions = 0

    # ------------------------------------------------------------
    def _check_keypoints(self, arm) -> bool:
        """
        检查关键点是否全部可信。

        这是需求里明确要求的：「针对 Shoulder/Elbow/Wrist 分别检查 confidence，
        任意关键点 confidence < threshold 则这一帧不更新机械臂」。
        """
        if arm is None:
            self._last_reason = "无手臂数据"
            return False

        for name in self.cfg.required_keypoints:
            kp = getattr(arm, name, None)
            if kp is None:
                self._last_reason = f"{name} 缺失"
                return False
            if not kp.is_valid:
                self._last_reason = f"{name} 无效(conf=0)"
                return False
            if kp.confidence < self.cfg.min_confidence:
                self._last_reason = (
                    f"{name} 置信度 {kp.confidence:.2f} < "
                    f"{self.cfg.min_confidence:.2f}")
                return False

        self._last_reason = ""
        return True

    # ------------------------------------------------------------
    def update(self, arm, now: float) -> TrackState:
        """
        喂入本帧的手臂关键点，返回当前状态。

        Args:
            arm: ArmKeypoints（可为 None）
            now: 当前时刻（**绝对值**，time.monotonic()）

        Returns:
            TrackState
        """
        # 时间基准保护：倒退则重置（与 retargeter 的处理一致）
        if self._last_time is not None and now < self._last_time:
            self._lost_since = None
        prev = self._last_time
        self._last_time = now

        ok = self._check_keypoints(arm)

        if ok:
            self._last_good_time = now
            self._lost_since = None
            new_state = TrackState.TRACKING
        else:
            if self._lost_since is None:
                # 丢失起点取上一帧时刻：让第一帧的丢失时长被正确计入
                # （若取当前帧，会凭空少算一个控制周期）
                self._lost_since = prev if prev is not None else now
            lost_for = now - self._lost_since
            if lost_for < self.cfg.short_lost_seconds:
                new_state = TrackState.LOST_SHORT
            else:
                new_state = TrackState.LOST_LONG

        if new_state is not self.state:
            self.n_transitions += 1
            self.state = new_state

        if new_state is TrackState.TRACKING:
            self.n_tracking += 1
        elif new_state is TrackState.LOST_SHORT:
            self.n_lost_short += 1
        else:
            self.n_lost_long += 1

        return self.state

    # ------------------------------------------------------------
    @property
    def lost_duration(self) -> float:
        """已连续丢失的时长（秒）；未丢失时为 0"""
        if self._lost_since is None or self._last_time is None:
            return 0.0
        return max(0.0, self._last_time - self._lost_since)

    @property
    def reason(self) -> str:
        """最近一次判定为「无效」的原因，便于调试与 HUD 显示"""
        return self._last_reason

    @property
    def is_safe_state(self) -> bool:
        """是否处于需要「冻结/安全」处理的状态"""
        return self.state is TrackState.LOST_LONG

    def reset(self) -> None:
        self.state = TrackState.TRACKING
        self._lost_since = None
        self._last_reason = ""

    def stats(self) -> dict:
        return {
            "tracking": self.n_tracking,
            "lost_short": self.n_lost_short,
            "lost_long": self.n_lost_long,
            "transitions": self.n_transitions,
        }

    def describe(self) -> str:
        return (f"{self.state}"
                + (f" ({self.lost_duration:.1f}s)" if self._lost_since else "")
                + (f" 原因: {self._last_reason}" if self._last_reason else ""))
