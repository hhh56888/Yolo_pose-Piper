# -*- coding: utf-8 -*-
"""
安全限制与配置的单元测试（不依赖 ROS2）
=======================================
safety.py / config.py / types.py 均为纯逻辑模块，
可以脱离 ROS2 环境单独测试，便于快速回归。

运行：
    pytest src/piper_human_control/test/test_safety.py -v
"""

import math
import os
import sys

import pytest

# 允许直接以源码目录方式导入（无需 colcon install）
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_HERE)
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from piper_human_control.config import ControlConfig          # noqa: E402
from piper_human_control.safety import EmergencyStop, SafetyLimiter  # noqa: E402
from piper_human_control.types import ArmKeypoints, Keypoint  # noqa: E402


# ============================================================
# 夹具
# ============================================================
@pytest.fixture(scope="module")
def cfg():
    """加载真实配置文件"""
    yaml_path = os.path.join(_PKG_ROOT, "config", "joint_limits.yaml")
    return ControlConfig.from_yaml(yaml_path)


@pytest.fixture
def limiter(cfg):
    return SafetyLimiter(cfg)


# ============================================================
# 配置
# ============================================================
class TestConfig:
    def test_joint_count(self, cfg):
        assert cfg.num_joints == 6
        assert cfg.joint_names[0] == "joint1"

    def test_limits_match_official_sdk(self, cfg):
        """限位必须与官方 piper_sdk 一致（回归保护）"""
        expected = {
            "joint1": (-2.6179, 2.6179),
            "joint2": (0.0, 3.14),
            "joint3": (-2.967, 0.0),
            "joint4": (-1.745, 1.745),
            "joint5": (-1.22, 1.22),
            "joint6": (-2.09439, 2.09439),
        }
        for name, (lo, hi) in expected.items():
            assert cfg.lower_limits[name] == pytest.approx(lo), name
            assert cfg.upper_limits[name] == pytest.approx(hi), name

    def test_home_within_limits(self, cfg):
        for i, name in enumerate(cfg.joint_names):
            assert cfg.lower_limits[name] <= cfg.home[i] <= cfg.upper_limits[name]

    def test_clip(self, cfg):
        assert cfg.clip("joint2", -5.0) == 0.0
        assert cfg.clip("joint2", 99.0) == pytest.approx(3.14)
        assert cfg.clip("joint3", 5.0) == 0.0


# ============================================================
# 合法性检查
# ============================================================
class TestValidation:
    def test_reject_wrong_length(self, limiter, cfg):
        rep = limiter.apply([0.0, 0.5], cfg.home, 0.05)
        assert rep.rejected
        assert "关节数不匹配" in rep.reason

    def test_reject_nan(self, limiter, cfg):
        bad = list(cfg.home)
        bad[2] = float("nan")
        rep = limiter.apply(bad, cfg.home, 0.05)
        assert rep.rejected
        assert "非法" in rep.reason

    def test_reject_inf(self, limiter, cfg):
        bad = list(cfg.home)
        bad[1] = float("inf")
        rep = limiter.apply(bad, cfg.home, 0.05)
        assert rep.rejected

    def test_reject_none(self, limiter, cfg):
        rep = limiter.apply(None, cfg.home, 0.05)
        assert rep.rejected


# ============================================================
# 位置限位
# ============================================================
class TestPositionLimits:
    def test_clamp_upper(self, limiter, cfg):
        bad = list(cfg.home)
        bad[1] = 99.0        # joint2 上限 3.14
        bad[2] = 5.0         # joint3 上限 0
        rep = limiter.apply(bad, None, 0.05)   # 无状态 -> 不做速率限制
        assert "joint2" in rep.clamped_joints
        assert "joint3" in rep.clamped_joints
        assert rep.positions[1] == pytest.approx(3.14)
        assert rep.positions[2] == pytest.approx(0.0)

    def test_clamp_lower(self, limiter, cfg):
        bad = list(cfg.home)
        bad[1] = -1.0        # joint2 下限 0
        rep = limiter.apply(bad, None, 0.05)
        assert "joint2" in rep.clamped_joints
        assert rep.positions[1] == pytest.approx(0.0)

    def test_no_clamp_when_valid(self, limiter, cfg):
        rep = limiter.apply(cfg.home, None, 0.05)
        assert not rep.clamped_joints
        assert not rep.rejected

    def test_in_limits_helper(self, limiter, cfg):
        assert limiter.in_limits(cfg.home)
        assert not limiter.in_limits([0, -1, 0, 0, 0, 0])
        assert not limiter.in_limits([0, 0])

    def test_describe_violations(self, limiter, cfg):
        v = limiter.describe_violations([0.0, -1.0, 0.0, 0.0, 0.0, 0.0])
        assert len(v) == 1 and "joint2" in v[0]


# ============================================================
# 速率限制
# ============================================================
class TestRateLimit:
    def test_rate_limit_engages_on_large_jump(self, limiter, cfg):
        """大阶跃应被速率限制，而不是一次跳过去"""
        cur = list(cfg.home)
        tgt = list(cfg.home)
        tgt[1] = 3.0                       # joint2 从 0.6 -> 3.0，差 2.4
        rep = limiter.apply(tgt, cur, 0.05)
        assert "joint2" in rep.rate_limited_joints
        # 单周期增量不得超过 min(a*dt, v*dt) 与 step_cap
        delta = abs(rep.positions[1] - cur[1])
        assert delta <= cfg.safety.max_step_per_cycle_rad + 1e-9

    def test_rate_limit_allows_small_step(self, limiter, cfg):
        cur = list(cfg.home)
        tgt = list(cfg.home)
        tgt[1] = cur[1] + 0.001            # 极小步长
        rep = limiter.apply(tgt, cur, 0.05)
        assert "joint2" not in rep.rate_limited_joints
        assert rep.positions[1] == pytest.approx(cur[1] + 0.001)

    def test_no_rate_limit_without_state(self, limiter, cfg):
        """没有状态反馈时不做速率限制（只能做位置限位）"""
        tgt = list(cfg.home)
        tgt[1] = 3.0
        rep = limiter.apply(tgt, None, 0.05)
        assert not rep.rate_limited_joints

    def test_convergence_over_many_cycles(self, limiter, cfg):
        """连续迭代应逐步收敛到目标（验证不会卡死）"""
        cur = list(cfg.home)
        tgt = list(cfg.home)
        tgt[1] = 2.5
        for _ in range(500):
            rep = limiter.apply(tgt, cur, 0.05)
            cur = list(rep.positions)
        assert cur[1] == pytest.approx(2.5, abs=1e-3)

    def test_step_cap_is_hard_bound(self, limiter, cfg):
        """无论 dt 多大，单周期增量不得超过 step_cap"""
        cur = list(cfg.home)
        tgt = [0.0, 3.0, -2.9, 1.7, 1.2, 2.0]
        rep = limiter.apply(tgt, cur, 100.0)     # 极大 dt
        for i, name in enumerate(cfg.joint_names):
            assert abs(rep.positions[i] - cur[i]) <= cfg.safety.max_step_per_cycle_rad + 1e-9


# ============================================================
# 急停
# ============================================================
class TestEmergencyStop:
    def test_initial_state(self):
        es = EmergencyStop()
        assert not es.engaged
        assert es.engaged_duration == 0.0

    def test_engage_and_reset(self):
        es = EmergencyStop()
        es.engage("测试")
        assert es.engaged and es.reason == "测试"
        es.reset()
        assert not es.engaged
        assert es.engaged_duration == 0.0


# ============================================================
# 数据结构
# ============================================================
class TestTypes:
    def test_keypoint_z_default_none(self):
        """阶段三约定：RGB 模式下 z 必须为 None"""
        kp = Keypoint(x=1.0, y=2.0, confidence=0.9)
        assert kp.z is None
        assert not kp.has_depth()
        assert kp.is_valid

    def test_keypoint_with_depth(self):
        """阶段七约定：RGB-D 模式下 z 有值，接口不变"""
        kp = Keypoint(x=1.0, y=2.0, z=1.5, confidence=0.9)
        assert kp.has_depth()
        assert kp.z == 1.5

    def test_invalid_keypoint(self):
        kp = Keypoint(x=0.0, y=0.0, confidence=0.0)
        assert not kp.is_valid

    def test_arm_keypoints_incomplete(self):
        ak = ArmKeypoints(shoulder=Keypoint(1, 2, confidence=0.9))
        assert not ak.is_complete

    def test_arm_keypoints_complete(self):
        ak = ArmKeypoints(
            shoulder=Keypoint(1, 2, confidence=0.9),
            elbow=Keypoint(3, 4, confidence=0.8),
            wrist=Keypoint(5, 6, confidence=0.7),
        )
        assert ak.is_complete
        assert len(ak.as_list()) == 3
