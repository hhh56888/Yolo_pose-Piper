# -*- coding: utf-8 -*-
"""
piper_human_retargeting.hand_calibration
========================================
手部标定状态机 + 夹爪双点标定（V1.1 任务书第十三、十四节）。

为什么手部要单独标定
====================
实测 MediaPipe 手部要到约 **11.6s** 才首次稳定检出，而人体标定窗口只有 2s。
原先的做法是「第一次识别到手就直接当 neutral」（lazy baseline），
这不够严谨：
    · 手刚进入画面时往往只有一帧，抖动大、可能误检；
    · 没有样本量、没有离散度检查，基准可能建立在一个坏值上。

本模块改成显式状态机：
    HAND_WAITING -> HAND_STABLE -> (采样 N 帧) -> HAND_CALIBRATED
只有连续稳定 N 帧之后才开始采样，并且记录
    采样起点 / 终点 / 样本数 / 均值 / 标准差
便于事后判断这次标定是否可信。

夹爪双点标定（第十四节）
======================
原先握拳阈值是经验值。改成分别标定
    openness_open   （手完全张开）
    openness_closed （手完全握紧）
然后按
    gripper_ratio = clip((openness - closed) / (open - closed), 0, 1)
映射到 Piper 夹爪，不再依赖固定阈值。
"""

from __future__ import annotations

import math
import statistics as st
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class HandCalState(str, Enum):
    """手部标定状态"""
    WAITING = "HAND_WAITING"        # 还没稳定检出手
    STABLE = "HAND_STABLE"          # 已连续稳定，正在采样
    CALIBRATED = "HAND_CALIBRATED"  # 采样完成


@dataclass
class HandCalibrationConfig:
    """手部标定参数"""
    # 需要连续稳定多少帧才开始采样（抑制「刚检出一帧就当基准」）
    stable_frames: int = 12
    # 采样窗口长度（帧）
    sample_frames: int = 20
    # 判定「稳定」的条件：手部关键点置信度下限
    min_confidence: float = 0.5
    # 样本离散度上限：标准差超过它就认为姿势在动、不建立基准
    max_std_pitch_deg: float = 12.0
    max_std_thumb: float = 0.25

    def validate(self) -> None:
        if self.stable_frames < 1:
            raise ValueError("配置错误: stable_frames 必须 >= 1")
        if self.sample_frames < 3:
            raise ValueError("配置错误: sample_frames 必须 >= 3")


@dataclass
class HandCalibrationResult:
    """手部标定结果（含可审计的统计量）"""
    ok: bool = False
    state: HandCalState = HandCalState.WAITING
    n_samples: int = 0
    mean: Dict[str, float] = field(default_factory=dict)
    std: Dict[str, float] = field(default_factory=dict)
    start_frame: int = -1
    end_frame: int = -1
    reason: str = ""

    def summary(self) -> str:
        if not self.ok:
            return f"{self.state.value} ({self.reason})"
        m = ", ".join(f"{k}={v:.3f}" for k, v in self.mean.items())
        s = ", ".join(f"{k}±{v:.3f}" for k, v in self.std.items())
        return (f"{self.state.value}: n={self.n_samples} "
                f"[{self.start_frame}..{self.end_frame}] 均值({m}) 标准差({s})")


class HandCalibrator:
    """
    手部标定状态机。

    用法（每帧调用一次）::

        hc = HandCalibrator(cfg)
        ...
        res = hc.update(hand_pose, frame_idx)
        if res.ok:
            # 把 res.mean 作为 J5/J6 的零点
    """

    def __init__(self, cfg: Optional[HandCalibrationConfig] = None):
        self.cfg = cfg or HandCalibrationConfig()
        self.cfg.validate()
        self.state = HandCalState.WAITING
        self._stable_run = 0
        self._samples: Dict[str, List[float]] = {}
        self._start_frame = -1
        self._last_frame = -1
        self.result = HandCalibrationResult()

    # ------------------------------------------------------------
    @property
    def calibrated(self) -> bool:
        return self.state is HandCalState.CALIBRATED

    def reset(self) -> None:
        self.state = HandCalState.WAITING
        self._stable_run = 0
        self._samples = {}
        self._start_frame = -1
        self.result = HandCalibrationResult()

    # ------------------------------------------------------------
    def update(self, hand_pose, frame: int = -1) -> HandCalibrationResult:
        """
        喂入一帧手部姿态。

        Args:
            hand_pose: 需含 valid / wrist_pitch_deg / thumb_offset
            frame:     帧号（仅用于记录采样区间）

        Returns:
            当前标定结果（ok=True 表示本次刚完成标定）
        """
        self._last_frame = frame
        if self.state is HandCalState.CALIBRATED:
            return self.result

        valid = (hand_pose is not None
                 and getattr(hand_pose, "valid", False))
        if not valid:
            # 丢了就重置稳定计数 —— 不连续稳定不算稳定
            self._stable_run = 0
            if self.state is HandCalState.STABLE:
                self.state = HandCalState.WAITING
                self._samples = {}
                self._start_frame = -1
            self.result = HandCalibrationResult(
                state=self.state, reason="手部未检出/无效")
            return self.result

        if self.state is HandCalState.WAITING:
            self._stable_run += 1
            if self._stable_run >= self.cfg.stable_frames:
                self.state = HandCalState.STABLE
                self._samples = {}
                self._start_frame = frame
                self.result = HandCalibrationResult(
                    state=self.state, reason="开始采样",
                    start_frame=frame)
            else:
                self.result = HandCalibrationResult(
                    state=self.state,
                    reason=f"稳定中 {self._stable_run}/{self.cfg.stable_frames}")
            return self.result

        # ---- STABLE：采样 ----
        self._samples.setdefault("wrist_pitch_deg", []).append(
            float(getattr(hand_pose, "wrist_pitch_deg", 0.0)))
        self._samples.setdefault("thumb_offset", []).append(
            float(getattr(hand_pose, "thumb_offset", 0.0)))

        n = len(self._samples["wrist_pitch_deg"])
        if n < self.cfg.sample_frames:
            self.result = HandCalibrationResult(
                state=self.state,
                n_samples=n, start_frame=self._start_frame,
                reason=f"采样中 {n}/{self.cfg.sample_frames}")
            return self.result

        # ---- 采样完成：算均值与标准差 ----
        mean = {k: float(st.mean(v)) for k, v in self._samples.items()}
        std = {k: float(st.pstdev(v)) if len(v) > 1 else 0.0
               for k, v in self._samples.items()}

        bad = []
        if std["wrist_pitch_deg"] > self.cfg.max_std_pitch_deg:
            bad.append(f"腕俯仰σ={std['wrist_pitch_deg']:.1f}°")
        if std["thumb_offset"] > self.cfg.max_std_thumb:
            bad.append(f"拇指偏移σ={std['thumb_offset']:.3f}")

        if bad:
            # 姿势在动 -> 丢弃这批，回到 WAITING 重新等稳定
            self.state = HandCalState.WAITING
            self._stable_run = 0
            self._samples = {}
            self.result = HandCalibrationResult(
                state=self.state,
                reason="离散度过大，重新等待稳定: " + ", ".join(bad))
            return self.result

        self.state = HandCalState.CALIBRATED
        self.result = HandCalibrationResult(
            ok=True, state=self.state, n_samples=n,
            mean=mean, std=std,
            start_frame=self._start_frame, end_frame=frame,
            reason="标定完成")
        return self.result


# ============================================================
# 夹爪双点标定（任务书第十四节）
# ============================================================
class GripperCalState(str, Enum):
    NEED_OPEN = "GRIPPER_NEED_OPEN"      # 等手张开
    NEED_CLOSED = "GRIPPER_NEED_CLOSED"  # 等手握拳
    DONE = "GRIPPER_CALIBRATED"


@dataclass
class GripperCalibration:
    """
    夹爪双点标定：分别记录张开与握拳时的 openness。

    原先用固定经验阈值（min_openness/max_openness）把人体 openness
    拉伸到关节全行程。问题是这两个阈值是**估**的，
    不同人手型/机位下会偏，夹爪要么永远闭不上、要么一直全开。

    改成标定两点后：
        gripper_ratio = clip((openness - closed) / (open - closed), 0, 1)
    """
    openness_open: Optional[float] = None
    openness_closed: Optional[float] = None
    state: GripperCalState = GripperCalState.NEED_OPEN

    # 判据：连续多少帧认为人「保持住了」某个姿势
    hold_frames: int = 10
    # 张开/闭合的合法性检查：两者差太小说明没做动作
    min_span: float = 0.15

    _open_samples: List[float] = field(default_factory=list)
    _closed_samples: List[float] = field(default_factory=list)

    def reset(self) -> None:
        self.openness_open = None
        self.openness_closed = None
        self.state = GripperCalState.NEED_OPEN
        self._open_samples = []
        self._closed_samples = []

    @property
    def done(self) -> bool:
        return self.state is GripperCalState.DONE

    def mark_open(self, openness: float) -> None:
        """把当前 openness 记为「张开」端"""
        self._open_samples.append(float(openness))
        if len(self._open_samples) >= self.hold_frames:
            self.openness_open = float(st.mean(self._open_samples))
            self.state = GripperCalState.NEED_CLOSED

    def mark_closed(self, openness: float) -> None:
        """把当前 openness 记为「握拳」端"""
        self._closed_samples.append(float(openness))
        if len(self._closed_samples) >= self.hold_frames:
            self.openness_closed = float(st.mean(self._closed_samples))
            if (self.openness_open is not None
                    and abs(self.openness_open - self.openness_closed)
                    >= self.min_span):
                self.state = GripperCalState.DONE

    def ratio(self, openness: float) -> float:
        """
        把当前 openness 归一化到 [0,1]。

        未完成标定时**回退到原始值**（不改动既有行为），
        避免「没标定就完全不可用」。
        """
        if (self.openness_open is None or self.openness_closed is None
                or abs(self.openness_open - self.openness_closed)
                < self.min_span):
            return float(min(1.0, max(0.0, openness)))
        r = ((openness - self.openness_closed)
             / (self.openness_open - self.openness_closed))
        return float(min(1.0, max(0.0, r)))

    def summary(self) -> str:
        if self.openness_open is None:
            return f"{self.state.value} (未记录张开)"
        if self.openness_closed is None:
            return (f"{self.state.value} (张开={self.openness_open:.3f}，"
                    f"待记录握拳)")
        return (f"{self.state.value}: open={self.openness_open:.3f} "
                f"closed={self.openness_closed:.3f} "
                f"span={abs(self.openness_open-self.openness_closed):.3f}")
