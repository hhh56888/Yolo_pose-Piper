# -*- coding: utf-8 -*-
"""
跟踪状态机 + 三级滤波链 测试
============================
覆盖需求中明确要求的：
    * TRACKING / LOST_SHORT / LOST_LONG 三态
    * 「任意关键点 confidence < threshold 则不更新」
    * 短暂丢失 -> 保持（不动作）
    * 长时间丢失 -> 安全状态（且**不执行 HOME**）
    * EMA 滤波 + 死区
    * 关键点滤波保留 z 的 3D 语义（z=None 不伪造）

运行：
    python3 -m pytest test/test_tracker_filter.py -q -p no:anyio
"""

import math
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_HERE)
_SRC = os.path.dirname(_PKG_ROOT)
for p in (_PKG_ROOT, os.path.join(_SRC, "piper_human_control")):
    if p not in sys.path:
        sys.path.insert(0, p)

from piper_human_control import (AngleFilter, FilterConfig,   # noqa: E402
                                 JointFilter, KeypointFilter,
                                 ScalarFilter)
from piper_human_control import ArmKeypoints, Keypoint        # noqa: E402
from piper_human_retargeting import (TrackerConfig,           # noqa: E402
                                     TrackerStateMachine, TrackState)


def arm(conf_s=0.9, conf_e=0.9, conf_w=0.9, with_wrist=True):
    return ArmKeypoints(
        shoulder=Keypoint(0, 0, confidence=conf_s, name="right_shoulder"),
        elbow=Keypoint(100, 0, confidence=conf_e, name="right_elbow"),
        wrist=(Keypoint(100, 100, confidence=conf_w, name="right_wrist")
               if with_wrist else None),
        side="right",
    )


# ============================================================
# 状态机
# ============================================================
class TestTrackerStateMachine:
    @staticmethod
    def sm(short=0.5, long=3.0, conf=0.4):
        return TrackerStateMachine(TrackerConfig(
            short_lost_seconds=short, long_lost_seconds=long,
            min_confidence=conf))

    # ---------- 正常 ----------
    def test_starts_tracking(self):
        assert self.sm().state is TrackState.TRACKING

    def test_good_arm_is_tracking(self):
        sm = self.sm()
        assert sm.update(arm(), now=0.0) is TrackState.TRACKING

    # ---------- 置信度检查（需求明确要求逐点检查）----------
    def test_low_shoulder_confidence_not_tracking(self):
        sm = self.sm(conf=0.4)
        st = sm.update(arm(conf_s=0.2), now=0.0)
        assert st is not TrackState.TRACKING
        assert "shoulder" in sm.reason

    def test_low_elbow_confidence_not_tracking(self):
        sm = self.sm(conf=0.4)
        st = sm.update(arm(conf_e=0.3), now=0.0)
        assert st is not TrackState.TRACKING
        assert "elbow" in sm.reason

    def test_low_wrist_confidence_not_tracking(self):
        sm = self.sm(conf=0.4)
        st = sm.update(arm(conf_w=0.1), now=0.0)
        assert st is not TrackState.TRACKING
        assert "wrist" in sm.reason

    def test_missing_wrist_not_tracking(self):
        sm = self.sm()
        st = sm.update(arm(with_wrist=False), now=0.0)
        assert st is not TrackState.TRACKING
        assert "wrist" in sm.reason

    def test_none_arm_not_tracking(self):
        sm = self.sm()
        assert sm.update(None, now=0.0) is not TrackState.TRACKING

    def test_confidence_exactly_at_threshold_is_ok(self):
        sm = self.sm(conf=0.4)
        assert sm.update(arm(0.4, 0.4, 0.4), now=0.0) is TrackState.TRACKING

    # ---------- 三态转移 ----------
    def test_lost_short_within_window(self):
        sm = self.sm(short=0.5, long=3.0)
        sm.update(arm(), now=0.0)
        assert sm.update(None, now=0.2) is TrackState.LOST_SHORT

    def test_lost_long_after_window(self):
        sm = self.sm(short=0.5, long=3.0)
        sm.update(arm(), now=0.0)
        assert sm.update(None, now=1.0) is TrackState.LOST_LONG

    def test_short_then_long_progression(self):
        sm = self.sm(short=0.5, long=3.0)
        sm.update(arm(), now=0.0)
        assert sm.update(None, now=0.3) is TrackState.LOST_SHORT
        assert sm.update(None, now=0.6) is TrackState.LOST_LONG

    def test_recover_to_tracking(self):
        sm = self.sm()
        sm.update(arm(), now=0.0)
        sm.update(None, now=5.0)
        assert sm.update(arm(), now=6.0) is TrackState.TRACKING

    def test_transition_counted(self):
        sm = self.sm(short=0.5)
        sm.update(arm(), now=0.0)
        sm.update(None, now=1.0)          # -> LOST_LONG (1 transition)
        sm.update(arm(), now=2.0)         # -> TRACKING  (2 transitions)
        assert sm.stats()["transitions"] == 2

    def test_safe_state_flag(self):
        sm = self.sm(short=0.5)
        sm.update(arm(), now=0.0)
        sm.update(None, now=0.2)
        assert not sm.is_safe_state          # LOST_SHORT 不算安全状态
        sm.update(None, now=1.0)
        assert sm.is_safe_state              # LOST_LONG 是安全状态

    # ---------- 丢失计时 ----------
    def test_first_lost_frame_counts(self):
        """
        回归保护：丢失计时必须从**上一帧**起算。
        若从当前帧起算，第一帧的丢失时长会算成 0，
        宽限期凭空多一个控制周期。
        """
        sm = self.sm(short=0.5)
        sm.update(arm(), now=0.0)
        sm.update(None, now=1.2)
        # 应已丢失 1.2s（而不是 0s）
        assert sm.lost_duration == pytest.approx(1.2, abs=1e-6)
        assert sm.state is TrackState.LOST_LONG

    def test_lost_duration_zero_when_tracking(self):
        sm = self.sm()
        sm.update(arm(), now=0.0)
        assert sm.lost_duration == 0.0

    def test_time_rollback_does_not_produce_negative(self):
        """时间基准倒退时不应算出负数时长"""
        sm = self.sm(short=0.5)
        sm.update(arm(), now=1e6)
        sm.update(None, now=0.1)
        assert sm.lost_duration >= 0.0

    # ---------- 配置校验 ----------
    def test_rejects_long_less_than_short(self):
        with pytest.raises(ValueError, match="long_lost_seconds"):
            TrackerConfig(short_lost_seconds=2.0, long_lost_seconds=1.0).validate()

    def test_rejects_negative_time(self):
        with pytest.raises(ValueError):
            TrackerConfig(short_lost_seconds=-1.0).validate()

    def test_rejects_bad_confidence(self):
        with pytest.raises(ValueError, match="min_confidence"):
            TrackerConfig(min_confidence=1.5).validate()


# ============================================================
# 标量滤波
# ============================================================
class TestScalarFilter:
    def test_first_value_passthrough(self):
        f = ScalarFilter(FilterConfig(alpha=0.5), "t")
        assert f.update(10.0) == pytest.approx(10.0)

    def test_ema_formula(self):
        f = ScalarFilter(FilterConfig(alpha=0.5, deadband=0.0), "t")
        f.update(0.0)
        assert f.update(10.0) == pytest.approx(5.0)     # 0 + 0.5*(10-0)
        assert f.update(10.0) == pytest.approx(7.5)     # 5 + 0.5*(10-5)

    def test_alpha_one_is_passthrough(self):
        f = ScalarFilter(FilterConfig(alpha=1.0, deadband=0.0), "t")
        f.update(0.0)
        assert f.update(10.0) == pytest.approx(10.0)

    def test_deadband_suppresses_small_change(self):
        f = ScalarFilter(FilterConfig(alpha=1.0, deadband=1.0), "t")
        f.update(0.0)
        assert f.update(0.5) == pytest.approx(0.0)      # 死区内不动
        assert f.update(0.9) == pytest.approx(0.0)

    def test_deadband_allows_large_change(self):
        f = ScalarFilter(FilterConfig(alpha=1.0, deadband=1.0), "t")
        f.update(0.0)
        assert f.update(2.0) == pytest.approx(2.0)

    def test_disabled_is_passthrough(self):
        f = ScalarFilter(FilterConfig(enabled=False, alpha=0.1), "t")
        f.update(0.0)
        assert f.update(10.0) == pytest.approx(10.0)

    def test_raw_and_value_tracked(self):
        f = ScalarFilter(FilterConfig(alpha=0.5, deadband=0.0), "t")
        f.update(0.0)
        f.update(10.0)
        assert f.raw == pytest.approx(10.0)
        assert f.value == pytest.approx(5.0)

    def test_reset(self):
        f = ScalarFilter(FilterConfig(alpha=0.5), "t")
        f.update(0.0)
        f.update(10.0)
        f.reset()
        assert f.value is None

    def test_rejects_bad_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            ScalarFilter(FilterConfig(alpha=0.0), "t").update(1.0)


# ============================================================
# 关节滤波
# ============================================================
class TestJointFilter:
    JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]

    def test_first_passthrough(self):
        jf = JointFilter(FilterConfig(alpha=0.5), self.JOINTS)
        q = [0.0, 1.0, -1.0, 0.0, 0.5, 0.0]
        assert jf.update(q) == pytest.approx(q)

    def test_smooths_step(self):
        jf = JointFilter(FilterConfig(alpha=0.5, deadband=0.0), self.JOINTS)
        jf.update([0.0] * 6)
        out = jf.update([1.0] * 6)
        assert out == pytest.approx([0.5] * 6)

    def test_wrong_length_raises(self):
        jf = JointFilter(FilterConfig(), self.JOINTS)
        with pytest.raises(ValueError, match="关节数不匹配"):
            jf.update([0.0, 1.0])

    def test_converges(self):
        jf = JointFilter(FilterConfig(alpha=0.3, deadband=0.0), self.JOINTS)
        jf.update([0.0] * 6)
        for _ in range(100):
            out = jf.update([1.0] * 6)
        assert out == pytest.approx([1.0] * 6, abs=1e-3)

    def test_reset_with_values(self):
        jf = JointFilter(FilterConfig(alpha=0.5), self.JOINTS)
        jf.reset([0.0, 0.6, -0.6, 0.0, 0.0, 0.0])
        assert jf.value()[1] == pytest.approx(0.6)

    def test_deadband_stops_micro_jitter(self):
        """人体静止时机械臂应基本不动 —— 死区的核心作用"""
        jf = JointFilter(FilterConfig(alpha=0.5, deadband=0.01), self.JOINTS)
        jf.update([0.0] * 6)
        # 反复喂入极小的抖动
        for _ in range(50):
            out = jf.update([0.005, -0.005, 0.004, -0.003, 0.002, -0.001])
        assert out == pytest.approx([0.0] * 6)


# ============================================================
# 关键点滤波（含 3D 语义）
# ============================================================
class TestKeypointFilter:
    def test_none_passthrough(self):
        kf = KeypointFilter(FilterConfig())
        assert kf.update(None) is None

    def test_invalid_keypoint_not_filtered(self):
        """confidence=0 的点不参与滤波，避免污染滤波器状态"""
        kf = KeypointFilter(FilterConfig(alpha=0.5))
        bad = Keypoint(5.0, 5.0, confidence=0.0, name="k")
        out = kf.update(bad)
        assert out.confidence == 0.0
        assert out.x == pytest.approx(5.0)

    def test_smooths_position(self):
        kf = KeypointFilter(FilterConfig(alpha=0.5, deadband=0.0))
        kf.update(Keypoint(0, 0, confidence=0.9, name="k"))
        out = kf.update(Keypoint(10, 20, confidence=0.9, name="k"))
        assert out.x == pytest.approx(5.0)
        assert out.y == pytest.approx(10.0)

    def test_preserves_z_none(self):
        """阶段三契约：z 为 None 时必须保持 None，不能伪造成 0"""
        kf = KeypointFilter(FilterConfig(alpha=0.5))
        kf.update(Keypoint(0, 0, z=None, confidence=0.9, name="k"))
        out = kf.update(Keypoint(10, 10, z=None, confidence=0.9, name="k"))
        assert out.z is None

    def test_filters_z_when_present(self):
        """阶段七：有深度时应被滤波（证明确实预留了 3D 通路）"""
        kf = KeypointFilter(FilterConfig(alpha=0.5, deadband=0.0))
        kf.update(Keypoint(0, 0, z=1.0, confidence=0.9, name="k"))
        out = kf.update(Keypoint(0, 0, z=2.0, confidence=0.9, name="k"))
        assert out.z == pytest.approx(1.5)

    def test_preserves_confidence_and_name(self):
        kf = KeypointFilter(FilterConfig(alpha=0.5))
        kf.update(Keypoint(0, 0, confidence=0.9, name="right_elbow"))
        out = kf.update(Keypoint(4, 4, confidence=0.77, name="right_elbow"))
        assert out.confidence == pytest.approx(0.77)
        assert out.name == "right_elbow"

    def test_update_arm_filters_all(self):
        kf = KeypointFilter(FilterConfig(alpha=0.5, deadband=0.0))
        kf.update_arm(arm())
        out = kf.update_arm(ArmKeypoints(
            shoulder=Keypoint(10, 10, confidence=0.9, name="right_shoulder"),
            elbow=Keypoint(110, 0, confidence=0.9, name="right_elbow"),
            wrist=Keypoint(100, 110, confidence=0.9, name="right_wrist"),
            side="right"))
        assert out.shoulder.x == pytest.approx(5.0)
        assert out.elbow.x == pytest.approx(105.0)

    def test_update_arm_preserves_side(self):
        kf = KeypointFilter(FilterConfig(alpha=0.5))
        out = kf.update_arm(arm())
        assert out.side == "right"

    def test_update_arm_none(self):
        kf = KeypointFilter(FilterConfig())
        assert kf.update_arm(None) is None

    def test_different_names_have_separate_state(self):
        kf = KeypointFilter(FilterConfig(alpha=0.5, deadband=0.0))
        kf.update(Keypoint(0, 0, confidence=0.9, name="a"))
        kf.update(Keypoint(0, 0, confidence=0.9, name="b"))
        oa = kf.update(Keypoint(10, 0, confidence=0.9, name="a"))
        ob = kf.update(Keypoint(0, 20, confidence=0.9, name="b"))
        assert oa.x == pytest.approx(5.0)
        assert ob.y == pytest.approx(10.0)


# ============================================================
# 角度滤波
# ============================================================
class TestAngleFilter:
    NAMES = ["upper_arm_angle_deg", "elbow_angle_deg", "forearm_angle_deg"]

    def test_smooths(self):
        af = AngleFilter(FilterConfig(alpha=0.5, deadband=0.0), self.NAMES, circular=set(self.NAMES))
        af.update({"elbow_angle_deg": 0.0})
        out = af.update({"elbow_angle_deg": 10.0})
        assert out["elbow_angle_deg"] == pytest.approx(5.0)

    def test_unknown_name_passthrough(self):
        af = AngleFilter(FilterConfig(alpha=0.5), self.NAMES, circular=set(self.NAMES))
        out = af.update({"unknown_measure": 42.0})
        assert out["unknown_measure"] == pytest.approx(42.0)

    def test_value_of_defaults_to_zero(self):
        af = AngleFilter(FilterConfig(alpha=0.5), self.NAMES, circular=set(self.NAMES))
        assert af.value_of("elbow_angle_deg") == 0.0

    def test_value_of_returns_filtered(self):
        af = AngleFilter(FilterConfig(alpha=0.5, deadband=0.0), self.NAMES, circular=set(self.NAMES))
        af.update({"elbow_angle_deg": 0.0})
        af.update({"elbow_angle_deg": 10.0})
        assert af.value_of("elbow_angle_deg") == pytest.approx(5.0)

    def test_deadband_in_degrees(self):
        """角度级死区是「度」，与关节级（弧度）量纲不同，不要混"""
        af = AngleFilter(FilterConfig(alpha=1.0, deadband=1.2), self.NAMES, circular=set(self.NAMES))
        af.update({"elbow_angle_deg": 0.0})
        assert af.update({"elbow_angle_deg": 0.8})["elbow_angle_deg"] == 0.0
        assert af.update({"elbow_angle_deg": 2.0})["elbow_angle_deg"] == 2.0

    def test_reset(self):
        af = AngleFilter(FilterConfig(alpha=0.5), self.NAMES, circular=set(self.NAMES))
        af.update({"elbow_angle_deg": 10.0})
        af.reset()
        assert af.value_of("elbow_angle_deg") == 0.0
