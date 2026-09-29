# -*- coding: utf-8 -*-
"""
手部标定状态机 + 夹爪双点标定测试（V1.1 第十三、十四节）
=====================================================
背景：MediaPipe 手部实测约 11.6s 才首次稳定检出，而标定窗口只有 2s。
原先「第一次检出手就直接当 neutral」不够严谨 ——
只有一帧、可能误检、没有样本量与离散度检查。

运行：
    python3 -m pytest test/test_hand_calibration.py -q -p no:anyio
"""

import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_HERE)
_SRC = os.path.dirname(_PKG_ROOT)
for p in (_PKG_ROOT, os.path.join(_SRC, "piper_human_control")):
    if p not in sys.path:
        sys.path.insert(0, p)

from piper_human_retargeting.hand_calibration import (   # noqa: E402
    GripperCalState, GripperCalibration, HandCalState, HandCalibrator,
    HandCalibrationConfig)


class FakePose:
    def __init__(self, pitch=0.0, thumb=0.0, valid=True):
        self.wrist_pitch_deg = pitch
        self.thumb_offset = thumb
        self.valid = valid


# ============================================================
# 1. 手部标定状态机
# ============================================================
class TestHandCalibrator:
    def test_starts_in_waiting(self):
        hc = HandCalibrator()
        assert hc.state is HandCalState.WAITING
        assert not hc.calibrated

    def test_first_frame_does_not_calibrate(self):
        """
        核心要求：**刚检出手的第一帧不能当基准**。
        必须连续稳定 stable_frames 帧。
        """
        hc = HandCalibrator(HandCalibrationConfig(stable_frames=10,
                                                  sample_frames=5))
        r = hc.update(FakePose(), frame=0)
        assert not r.ok
        assert hc.state is HandCalState.WAITING

    def test_full_sequence_reaches_calibrated(self):
        cfg = HandCalibrationConfig(stable_frames=5, sample_frames=8)
        hc = HandCalibrator(cfg)
        done = None
        for k in range(40):
            r = hc.update(FakePose(pitch=20.0, thumb=0.3), frame=k)
            if r.ok:
                done = r
                break
        assert done is not None, "应能在稳定+采样后完成标定"
        assert hc.state is HandCalState.CALIBRATED
        assert done.n_samples == 8
        assert done.mean["wrist_pitch_deg"] == pytest.approx(20.0, abs=1e-6)
        assert done.mean["thumb_offset"] == pytest.approx(0.3, abs=1e-6)

    def test_records_sample_window_and_stats(self):
        """必须记录采样起止帧、样本数、均值、标准差（可审计）"""
        hc = HandCalibrator(HandCalibrationConfig(stable_frames=3,
                                                  sample_frames=6))
        done = None
        for k in range(30):
            r = hc.update(FakePose(), frame=k)
            if r.ok:
                done = r
                break
        assert done.start_frame >= 0
        assert done.end_frame > done.start_frame
        assert done.n_samples == 6
        assert "wrist_pitch_deg" in done.std
        assert done.summary().startswith("HAND_CALIBRATED")

    def test_dropout_resets_stability(self):
        """中途丢失手 -> 稳定计数必须清零（不连续不算稳定）"""
        hc = HandCalibrator(HandCalibrationConfig(stable_frames=6))
        for k in range(5):
            hc.update(FakePose(), frame=k)
        hc.update(FakePose(valid=False), frame=5)     # 丢一帧
        r = hc.update(FakePose(), frame=6)
        assert hc.state is HandCalState.WAITING, "丢失后应回到 WAITING"

    def test_high_variance_is_rejected(self):
        """
        采样期间姿势在动（离散度大）-> 丢弃这批、重新等待稳定。
        这是「不要把正在动的手当成基准」的保障。
        """
        hc = HandCalibrator(HandCalibrationConfig(
            stable_frames=2, sample_frames=6, max_std_pitch_deg=5.0))
        k = 0
        for _ in range(2):
            hc.update(FakePose(), frame=k); k += 1
        # 采样期间大幅摆动
        r = None
        for i in range(6):
            r = hc.update(FakePose(pitch=(-60 if i % 2 else 60)), frame=k); k += 1
        assert not r.ok
        assert hc.state is HandCalState.WAITING
        assert "离散度" in r.reason

    def test_reset_returns_to_waiting(self):
        hc = HandCalibrator(HandCalibrationConfig(stable_frames=2,
                                                  sample_frames=3))
        for k in range(10):
            hc.update(FakePose(), frame=k)
        assert hc.calibrated
        hc.reset()
        assert hc.state is HandCalState.WAITING

    def test_stays_calibrated_after_completion(self):
        hc = HandCalibrator(HandCalibrationConfig(stable_frames=2,
                                                  sample_frames=3))
        for k in range(10):
            hc.update(FakePose(), frame=k)
        first = hc.result.mean["wrist_pitch_deg"]
        for k in range(20):
            hc.update(FakePose(pitch=999.0), frame=100 + k)
        assert hc.result.mean["wrist_pitch_deg"] == pytest.approx(first), \
            "标定完成后不应被后续帧改动"

    def test_bad_config_rejected(self):
        with pytest.raises(ValueError):
            HandCalibrationConfig(stable_frames=0).validate()
        with pytest.raises(ValueError):
            HandCalibrationConfig(sample_frames=2).validate()


# ============================================================
# 2. 夹爪双点标定
# ============================================================
class TestGripperCalibration:
    def test_two_point_sequence(self):
        gc = GripperCalibration(hold_frames=4, min_span=0.2)
        assert gc.state is GripperCalState.NEED_OPEN
        for _ in range(4):
            gc.mark_open(0.9)
        assert gc.state is GripperCalState.NEED_CLOSED
        for _ in range(4):
            gc.mark_closed(0.2)
        assert gc.done
        assert gc.openness_open == pytest.approx(0.9)
        assert gc.openness_closed == pytest.approx(0.2)

    def test_ratio_maps_endpoints(self):
        gc = GripperCalibration(hold_frames=2, min_span=0.2)
        for _ in range(2):
            gc.mark_open(0.8)
        for _ in range(2):
            gc.mark_closed(0.2)
        assert gc.done
        assert gc.ratio(0.2) == pytest.approx(0.0)
        assert gc.ratio(0.8) == pytest.approx(1.0)
        assert gc.ratio(0.5) == pytest.approx(0.5)

    def test_ratio_clips_out_of_range(self):
        gc = GripperCalibration(hold_frames=2, min_span=0.2)
        for _ in range(2):
            gc.mark_open(0.8)
        for _ in range(2):
            gc.mark_closed(0.2)
        assert gc.ratio(1.5) == pytest.approx(1.0)
        assert gc.ratio(-1.0) == pytest.approx(0.0)

    def test_ratio_falls_back_before_calibration(self):
        """
        未完成标定时必须**回退到原始值**，而不是返回常数 ——
        否则「没标定」会变成「夹爪完全不能动」，比不做还差。
        """
        gc = GripperCalibration()
        assert gc.ratio(0.42) == pytest.approx(0.42)
        assert gc.ratio(1.7) == pytest.approx(1.0)

    def test_too_small_span_not_calibrated(self):
        """张开与握拳差别太小 -> 说明人没做动作，不能算标定完成"""
        gc = GripperCalibration(hold_frames=2, min_span=0.3)
        for _ in range(2):
            gc.mark_open(0.55)
        for _ in range(2):
            gc.mark_closed(0.50)
        assert not gc.done, "行程过小不应判定标定完成"

    def test_reset(self):
        gc = GripperCalibration(hold_frames=2, min_span=0.1)
        for _ in range(2):
            gc.mark_open(0.9)
        gc.reset()
        assert gc.state is GripperCalState.NEED_OPEN
        assert gc.openness_open is None

    def test_summary_reports_both_ends(self):
        gc = GripperCalibration(hold_frames=2, min_span=0.2)
        for _ in range(2):
            gc.mark_open(0.9)
        for _ in range(2):
            gc.mark_closed(0.1)
        s = gc.summary()
        assert "open=0.900" in s and "closed=0.100" in s
