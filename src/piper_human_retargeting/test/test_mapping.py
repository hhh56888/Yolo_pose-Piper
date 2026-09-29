# -*- coding: utf-8 -*-
"""
统一映射函数 map_human_to_robot 测试
====================================
映射是整条链路里最容易出错、也最需要反复调参的一环，
因此单独用一个文件把它钉死。

覆盖：
    * 区间归一化公式的正确性（端点 / 中点 / 外推 clamp）
    * invert 语义
    * offset 叠加
    * 机械臂限位兜底
    * 配置校验（非法区间、无效限位、映射区间完全落在限位外）
    * scale 推导
    * map_all 批量映射与缺测量量的保守处理

运行：
    python3 -m pytest test/test_mapping.py -q -p no:anyio
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

from piper_human_retargeting import (RangeMapping, map_all,   # noqa: E402
                                     map_human_to_robot)


def rule(**kw) -> RangeMapping:
    """构造一个默认规则的便捷函数"""
    base = dict(key="test", human="elbow_angle_deg", joint="joint3",
                human_min=-60.0, human_max=60.0,
                robot_min=-1.0, robot_max=1.0,
                invert=False, offset=0.0, clamp=True,
                joint_lower=-2.0, joint_upper=2.0)
    base.update(kw)
    return RangeMapping(**base)


# ============================================================
# 公式正确性
# ============================================================
class TestFormula:
    def test_lower_endpoint(self):
        """Δ = human_min -> robot_min"""
        r = rule()
        assert map_human_to_robot(-60.0, r) == pytest.approx(-1.0)

    def test_upper_endpoint(self):
        """Δ = human_max -> robot_max"""
        r = rule()
        assert map_human_to_robot(60.0, r) == pytest.approx(1.0)

    def test_midpoint(self):
        r = rule()
        assert map_human_to_robot(0.0, r) == pytest.approx(0.0)

    def test_quarter_point(self):
        r = rule()
        # Δ=-30 是下界到中点的一半 -> robot = -0.5
        assert map_human_to_robot(-30.0, r) == pytest.approx(-0.5)

    def test_clamp_below(self):
        r = rule(clamp=True)
        assert map_human_to_robot(-1000.0, r) == pytest.approx(-1.0)

    def test_clamp_above(self):
        r = rule(clamp=True)
        assert map_human_to_robot(+1000.0, r) == pytest.approx(1.0)

    def test_no_clamp_extrapolates(self):
        """clamp=false 时允许外推（随后仍受关节限位约束）"""
        r = rule(clamp=False, joint_lower=-10.0, joint_upper=10.0)
        # human 区间 [-60,60] 跨度 120；Δ=120 -> ratio=1.5（不是 2.0）
        # robot 区间 [-1,1] -> -1 + 1.5*2 = 2.0
        assert map_human_to_robot(120.0, r) == pytest.approx(2.0)
        # 再外推一点，确认是线性而非被 clamp
        assert map_human_to_robot(180.0, r) == pytest.approx(3.0)

    def test_no_clamp_still_limited_by_joint(self):
        """外推不能突破关节限位"""
        r = rule(clamp=False, joint_lower=-1.5, joint_upper=1.5)
        assert map_human_to_robot(1000.0, r) == pytest.approx(1.5)

    def test_reversed_robot_range(self):
        """robot_min > robot_max（人增大时关节减小）也能正确工作"""
        r = rule(robot_min=1.0, robot_max=-1.0)
        assert map_human_to_robot(-60.0, r) == pytest.approx(1.0)
        assert map_human_to_robot(+60.0, r) == pytest.approx(-1.0)


# ============================================================
# invert 语义
# ============================================================
class TestInvert:
    def test_invert_flips_endpoints(self):
        r = rule(invert=True)
        assert map_human_to_robot(-60.0, r) == pytest.approx(1.0)
        assert map_human_to_robot(+60.0, r) == pytest.approx(-1.0)

    def test_invert_keeps_center(self):
        r = rule(invert=True)
        assert map_human_to_robot(0.0, r) == pytest.approx(0.0)

    def test_invert_equals_swapped_range(self):
        """
        invert=true 必须与「交换 robot_min/robot_max」等价 ——
        这是 invert 的定义，保证只有一种方向处理方式。
        """
        a = rule(invert=True)
        b = rule(robot_min=1.0, robot_max=-1.0)
        for d in (-60, -30, 0, 30, 60):
            assert (map_human_to_robot(d, a)
                    == pytest.approx(map_human_to_robot(d, b)))

    def test_no_scattered_negation(self):
        """
        回归保护：方向翻转必须由 invert 完成。
        这里验证 invert 确实作用在区间上，
        而不是靠调用方在外部写 angle = -angle。
        """
        normal = rule(invert=False)
        flipped = rule(invert=True)
        assert (map_human_to_robot(30.0, normal)
                == pytest.approx(-map_human_to_robot(30.0, flipped)))


# ============================================================
# offset
# ============================================================
class TestOffset:
    def test_offset_added(self):
        r = rule(offset=0.3)
        assert map_human_to_robot(0.0, r) == pytest.approx(0.3)

    def test_offset_then_limited(self):
        """offset 后仍要过限位"""
        r = rule(offset=5.0, joint_lower=-2.0, joint_upper=2.0)
        assert map_human_to_robot(0.0, r) == pytest.approx(2.0)


# ============================================================
# 关节限位兜底
# ============================================================
class TestJointLimit:
    def test_result_always_within_limits(self):
        r = rule(robot_min=-5.0, robot_max=5.0,
                 joint_lower=-1.0, joint_upper=1.0)
        for d in range(-200, 201, 17):
            v = map_human_to_robot(float(d), r)
            assert -1.0 - 1e-9 <= v <= 1.0 + 1e-9

    def test_info_reports_limiting(self):
        r = rule(robot_min=-5.0, robot_max=5.0,
                 joint_lower=-1.0, joint_upper=1.0)
        _v, info = r.map(60.0)
        assert info["limited"] is True

    def test_info_reports_not_limiting(self):
        r = rule()
        _v, info = r.map(0.0)
        assert info["limited"] is False


# ============================================================
# scale 推导
# ============================================================
class TestScale:
    def test_scale_from_ranges(self):
        r = rule(human_min=0.0, human_max=100.0,
                 robot_min=0.0, robot_max=2.0)
        assert r.scale == pytest.approx(0.02)

    def test_scale_negative_for_reversed(self):
        r = rule(human_min=0.0, human_max=100.0,
                 robot_min=1.0, robot_max=-1.0)
        assert r.scale == pytest.approx(-0.02)

    def test_human_span(self):
        assert rule(human_min=-30.0, human_max=90.0).human_span == pytest.approx(120.0)


# ============================================================
# 调试信息
# ============================================================
class TestInfo:
    def test_info_keys(self):
        _v, info = rule().map(0.0)
        for k in ("ratio_raw", "ratio", "after_map", "after_invert",
                  "after_offset", "before_limit", "limited"):
            assert k in info

    def test_ratio_clamped_in_info(self):
        _v, info = rule(clamp=True).map(1000.0)
        assert info["ratio"] == pytest.approx(1.0)
        assert info["ratio_raw"] > 1.0


# ============================================================
# 配置校验
# ============================================================
class TestValidation:
    def test_rejects_bad_human_range(self):
        with pytest.raises(ValueError, match="human_max"):
            rule(human_min=10.0, human_max=-10.0).validate()

    def test_rejects_degenerate_robot_range(self):
        with pytest.raises(ValueError):
            rule(robot_min=0.5, robot_max=0.5, offset=0.0).validate()

    def test_allows_degenerate_range_with_offset(self):
        """区间退化但有 offset 时仍可动，应放行"""
        rule(robot_min=0.5, robot_max=0.5, offset=0.2).validate()

    def test_rejects_bad_joint_limit(self):
        with pytest.raises(ValueError, match="限位"):
            rule(joint_lower=1.0, joint_upper=-1.0).validate()

    def test_rejects_mapping_entirely_outside_limit(self):
        """映射区间完全落在关节限位外 -> 配置写错了"""
        with pytest.raises(ValueError, match="完全落在"):
            rule(robot_min=10.0, robot_max=12.0,
                 joint_lower=-1.0, joint_upper=1.0).validate()

    def test_accepts_partial_overlap(self):
        """部分重叠是允许的（会被限位截断），不应报错"""
        rule(robot_min=-3.0, robot_max=3.0,
             joint_lower=-1.0, joint_upper=1.0).validate()


# ============================================================
# 批量映射
# ============================================================
class TestMapAll:
    def test_maps_all_rules(self):
        rules = [
            rule(key="a", human="upper_arm_angle_deg", joint="joint2",
                 robot_min=0.0, robot_max=2.0, joint_lower=0.0, joint_upper=3.0),
            rule(key="b", human="elbow_angle_deg", joint="joint3"),
        ]
        out, infos = map_all(
            {"upper_arm_angle_deg": 0.0, "elbow_angle_deg": 0.0},
            rules, {"joint1": 0.0})
        assert out["joint1"] == pytest.approx(0.0)
        assert out["joint2"] == pytest.approx(1.0)
        assert out["joint3"] == pytest.approx(0.0)
        assert set(infos.keys()) == {"joint2", "joint3"}

    def test_missing_measure_is_skipped_conservatively(self):
        """
        缺少某个测量量时，该关节**不出现**在结果里
        （而不是填 0）—— 让调用方保持上一帧目标，
        避免把「没有数据」当成「角度为 0」下发。
        """
        rules = [rule(key="a", human="elbow_angle_deg", joint="joint3")]
        out, infos = map_all({}, rules, {"joint1": 0.0})
        assert "joint3" not in out
        assert "joint3" not in infos
        assert out["joint1"] == pytest.approx(0.0)

    def test_disabled_rule_skipped(self):
        rules = [rule(key="a", joint="joint3", enabled=False)]
        out, _ = map_all({"elbow_angle_deg": 10.0}, rules, {})
        assert "joint3" not in out

    def test_fixed_joints_preserved(self):
        out, _ = map_all({}, [], {"joint1": 0.3, "joint4": -0.2, "joint6": 0.1})
        assert out == {"joint1": 0.3, "joint4": -0.2, "joint6": 0.1}


# ============================================================
# 方向正确性（回归保护）
# ============================================================
class TestMotionDirection:
    """
    回归背景：
        初版 invert 是凭直觉写的，实测发现 J3 方向反了：
        人伸直肘时机械臂反而更弯。根因是没把「人体角度增大的
        物理含义」与「关节正增量使腕往哪走」对上。

        这组测试把方向判断**固化成可验证的规则**，
        以后改配置若把方向弄反，会立刻失败。

    依据（阶段一用 TF 正运动学实测，见 retargeting.yaml 注释）：
        d(腕角)/dq2 = -33.7 °/rad    q2 增大 -> 腕下降
        d(腕角)/dq3 = -10.8 °/rad    q3 增大 -> 前臂上抬
        d(腕角)/dq5 =  -2.7 °/rad    q5 增大 -> 腕下降
    其中「腕角」为竖直平面内角度（向上为正）。
    """

    # ---- 实测灵敏度 ----
    # WRIST_DQ: 腕在竖直平面内的角度（向上为正）对关节的导数，单位 度/弧度。
    #   来源：阶段一用 TF 正运动学实测（见 retargeting.yaml 注释）。
    WRIST_DQ = {"joint2": -33.7, "joint3": -10.8, "joint5": -2.7}

    # DZ_DQ: 腕部**高度**(z) 对关节的导数，单位 米/弧度。
    #   来源：本项目直接实测（只改一个关节，读 TF 里 link6 相对 base_link 的 z）：
    #       q3 = -1.10 -> z = 0.5215
    #       q3 = -0.85 -> z = 0.4478
    #       q3 = -0.35 -> z = 0.2802
    #       q3 = -0.10 -> z = 0.1970
    #   z 随 q3 单调**下降**，用端点线性估算：
    #       dz/dq3 ≈ (0.1970 - 0.5215) / (-0.10 - (-1.10)) = **-0.3245** m/rad
    #
    # ⚠️ 修正记录（J3 专项审计 §10.2）：
    #   这里原来写的是 +0.3245，与自己列出的 4 个实测点算出来的符号**相反**，
    #   注释也写成"z 随 q3 单调上升"。因为下面两个方向守卫测试直接调
    #   r.map(human_max)（delta 空间，绕过了 measure_sign），
    #   两个错误正好互相抵消，测试一直是 PASS 的 —— 也就守不住真实方向。
    #   现已按实测改正，并把"人弯肘"的端到端语义移到
    #   test/test_direction_e2e.py（走完整 Retargeter + FK）。
    DZ_DQ = {"joint3": -0.3245}

    @pytest.fixture
    def cfg(self):
        import piper_human_retargeting as prt
        import os as _os
        root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
        return prt.RetargetingConfig.from_yaml(
            _os.path.join(root, "config", "retargeting.yaml"))

    def _wrist_angle_change(self, joint: str, dq: float) -> float:
        return self.WRIST_DQ[joint] * dq

    def test_raising_arm_raises_wrist(self, cfg):
        """
        人抬上臂（上臂角增大）-> 腕应上升。
        J2 上：腕角变化 = d(腕角)/dq2 * (q2(human_max) - q2(0))
        """
        r = cfg.rule_for("joint2")
        q_at_0, _ = r.map(0.0)
        q_at_max, _ = r.map(r.human_max)
        d_wrist = self._wrist_angle_change("joint2", q_at_max - q_at_0)
        assert d_wrist > 0, (
            f"人抬臂时腕应上升，实际腕角变化 {d_wrist:+.1f}° "
            f"(q2: {q_at_0:+.3f} -> {q_at_max:+.3f}, invert={r.invert})")

    def test_elbow_rule_moves_wrist_down_at_human_min(self, cfg):
        """
        **delta 空间**的方向守卫，符号与运行时一致：

            elbow_angle_deg 的 measure_sign = −1
            => 人**弯肘**（raw 增大）对应 delta = human_min
            => 规则把 q3 送到区间另一端（更接近 0 = 收臂）
            => 由实测 d(z)/dq3 = −0.3245，腕高度应**下降**

        ⚠️ 修正记录（J3 专项审计 §10.2）：这条断言原来写在 human_max 上，
        而 human_max 在运行时对应的是"人伸肘" —— 测试与生产语义正好错开，
        配上当时写反的 DZ_DQ，两个错误互相抵消，永远 PASS。
        现在符号改正 + 端点对调，**把 invert 改错会立刻失败**。
        """
        r = cfg.rule_for("joint3")
        assert cfg.measure_sign()["elbow_angle_deg"] == -1.0, (
            "本测试的端点选择依赖 elbow 的 sign=−1；若改了 sign 必须同步改测试")
        q0, _ = r.map(0.0)
        q_bend, _ = r.map(r.human_min)          # 人弯肘 -> delta = human_min
        dz = self.DZ_DQ["joint3"] * (q_bend - q0)
        assert dz < 0, (
            f"人弯肘时腕应下降，实际 dz={dz:+.4f} m "
            f"(q3 {q0:+.3f} -> {q_bend:+.3f}, invert={r.invert})")

    def test_elbow_rule_moves_wrist_up_at_human_max(self, cfg):
        """另一端：人**伸肘** -> delta = human_max -> q3 减小 -> 腕上升（伸臂）"""
        r = cfg.rule_for("joint3")
        q0, _ = r.map(0.0)
        q_ext, _ = r.map(r.human_max)
        dz = self.DZ_DQ["joint3"] * (q_ext - q0)
        assert dz > 0, (
            f"人伸肘时腕应上升，实际 dz={dz:+.4f} m "
            f"(q3 {q0:+.3f} -> {q_ext:+.3f})")

    def test_direction_guard_catches_flipped_invert(self, cfg):
        """
        守卫的自检：把 joint3 的 invert 翻错，上面的断言必须失败。
        没有这条，"方向守卫"可能只是恰好通过的摆设（审计 §10.2 的教训）。
        """
        r = cfg.rule_for("joint3")
        r.invert = not r.invert
        q0, _ = r.map(0.0)
        q_bend, _ = r.map(r.human_min)
        dz = self.DZ_DQ["joint3"] * (q_bend - q0)
        assert dz > 0, "invert 翻错后 dz 应变号 —— 否则守卫无效"

    def test_neutral_anchored_for_all(self, cfg):
        """所有受控关节在 Δ=0 时必须输出 neutral（防单边/错位）"""
        for r in cfg.enabled_rules():
            v, info = r.map(0.0)
            n = cfg.neutral_for(r.joint)
            assert v == pytest.approx(n, abs=1e-9), (
                f"{r.joint}: Δ=0 输出 {v:+.4f} != neutral {n:+.4f}")
            assert info["ratio"] == pytest.approx(0.5)

    def test_all_monotonic(self, cfg):
        """所有受控关节在人体输入增大时都应单调变化（不能来回折）"""
        for r in cfg.enabled_rules():
            xs = [r.human_min + i * (r.human_max - r.human_min) / 20
                  for i in range(21)]
            ys = [r.map(x)[0] for x in xs]
            diffs = [b - a for a, b in zip(ys, ys[1:])]
            assert all(d >= -1e-12 for d in diffs) or \
                   all(d <= 1e-12 for d in diffs), \
                f"{r.joint} 映射非单调"
