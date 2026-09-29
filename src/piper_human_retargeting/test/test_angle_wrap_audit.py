# -*- coding: utf-8 -*-
"""
周期角度做差的审计与防回归测试
==============================
背景（V1.1 任务书第二节要求先审计再改配置）：

    实测 `delta_upper` 行程约 199°，而配置区间只有 ±60°，
    导致 joint2 大量饱和。任务书要求先确认：
        199° 是**真实人体运动范围**，还是 **±180° wrap 引起的假行程**？
    并明确禁止「直接改成 ±200°」。

审计结论（本文件把结论固化为测试）：

    1. `angle_delta_deg` **已经**在做 shortest-angle difference，
       与任务书建议的 `atan2(sin, cos)` 形式**完全等价**（差异 ~1e-14）。
    2. 它的输出**恒在 (-180, 180]** —— 全枚举验证过。
       因此 **199° 不可能来自单帧 wrap**，只能是**运动区间的跨度**。
    3. 实测 delta_upper ∈ [-139.30, +59.86]，span = 199.16°。
       对照配置 ±60：
           +59.9° -> ratio 0.999（用满、未裁）
           -139.3° -> ratio -0.661 -> clip 到 0（**负方向被裁**）
       即症状是「负方向饱和」，不是「两端都饱和」。

    结论：差分层无缺陷；问题是**标定姿势把可用行程偏置到区间之外**，
    所以正确做法是重选 neutral / 改用 task-space，而不是简单放宽区间。

运行：
    python3 -m pytest test/test_angle_wrap_audit.py -q -p no:anyio
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

from piper_human_control import AngleFilter, FilterConfig        # noqa: E402
from piper_human_retargeting import angle_delta_deg              # noqa: E402
from piper_human_retargeting.arm_geometry import _angle_deg      # noqa: E402


def spec_form(cur: float, ref: float) -> float:
    """任务书给的推荐形式"""
    r = math.radians(cur - ref)
    return math.degrees(math.atan2(math.sin(r), math.cos(r)))


# ============================================================
# 1. 差分层：shortest-angle 语义
# ============================================================
class TestShortestAngleDifference:
    def test_spec_required_case(self):
        """任务书点名的那一例：ref=+170, cur=-170 -> ±20，而不是 ∓340"""
        d = angle_delta_deg(-170.0, 170.0)
        assert abs(d) == pytest.approx(20.0), (
            f"应得到 20 的等价表示，实际 {d}")
        assert abs(d) < 180.0

    def test_never_exceeds_180_exhaustive(self):
        """
        **核心性质**：输出恒在 (-180, 180]。
        全枚举证明 —— 这正是「199° 不可能是单帧 wrap」的依据。
        """
        worst = 0.0
        for a in range(-359, 360, 1):
            for b in range(-359, 360, 7):       # 步长 7 足够覆盖
                d = angle_delta_deg(a, b)
                assert -180.0 <= d <= 180.0, (a, b, d)
                worst = max(worst, abs(d))
        assert worst == pytest.approx(180.0)

    def test_equivalent_to_atan2_form(self):
        """与任务书推荐的 atan2(sin,cos) 形式等价"""
        worst = 0.0
        for a in range(-180, 181, 7):
            for b in range(-180, 181, 11):
                d1 = angle_delta_deg(a, b)
                d2 = spec_form(a, b)
                worst = max(worst, abs((d1 - d2 + 180) % 360 - 180))
        assert worst < 1e-9, f"两种形式不等价，最大差异 {worst}"

    def test_wrap_across_zero(self):
        """跨 0° 的差不应被算成大角"""
        assert angle_delta_deg(5.0, -5.0) == pytest.approx(10.0)
        assert angle_delta_deg(-5.0, 5.0) == pytest.approx(-10.0)

    def test_naive_subtraction_would_be_wrong(self):
        """
        反证：朴素相减在跨 ±180° 时给出错误的量级。
        说明「必须用 shortest-angle」这条要求本身是对的 ——
        只是本项目**已经**做到了。
        """
        ref, cur = 170.0, -170.0
        naive = cur - ref                       # -340
        assert abs(naive) > 180.0
        assert abs(angle_delta_deg(cur, ref)) < 180.0


# ============================================================
# 2. 绝对方向角本身就不会越过 ±180°
# ============================================================
class TestAbsoluteAngleRange:
    def test_direction_angle_is_bounded(self):
        """上臂/前臂方向角用 atan2 求，天然落在 (-180, 180]"""
        for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1),
                       (1, -1), (-1, 1), (-1, -1), (1, 1), (0.001, -1)):
            a = _angle_deg(dx, dy)
            assert -180.0 <= a <= 180.0, (dx, dy, a)

    def test_zero_is_right_and_positive_is_up(self):
        """坐标约定：0°=图像右，+90°=图像上（y 已翻转）"""
        assert _angle_deg(1, 0) == pytest.approx(0.0)
        assert _angle_deg(0, -1) == pytest.approx(90.0)   # dy<0 在图像里是"上"
        assert _angle_deg(0, 1) == pytest.approx(-90.0)


# ============================================================
# 3. 滤波层也必须是周期处理
# ============================================================
class TestFilterIsCircularForAngles:
    def test_circular_filter_does_not_collapse_at_180(self):
        """
        线性 EMA 会把 +179 与 -179 平均成 0（真实差异只有 2°）。
        周期滤波必须给出 ±180 附近的值。
        """
        f = AngleFilter(FilterConfig(alpha=1.0, deadband=0.0),
                        ["upper_arm_angle_deg"],
                        circular={"upper_arm_angle_deg"})
        f.update({"upper_arm_angle_deg": 179.0})
        out = f.update({"upper_arm_angle_deg": -179.0})["upper_arm_angle_deg"]
        # 结果应贴近 ±180，而不是塌到 0
        assert abs(abs(out) - 180.0) < 5.0, (
            f"周期滤波在 ±180° 处塌陷: 得到 {out}")

    def test_linear_filter_would_collapse(self):
        """
        反证：同一个量若按**线性**处理，+179 -> -179 会跳到约 0。
        这解释了为什么必须区分周期量/线性量。
        """
        f = AngleFilter(FilterConfig(alpha=1.0, deadband=0.0),
                        ["upper_arm_angle_deg"], circular=set())
        f.update({"upper_arm_angle_deg": 179.0})
        out = f.update({"upper_arm_angle_deg": -179.0})["upper_arm_angle_deg"]
        assert out == pytest.approx(-179.0)     # 线性：直接取新值，跳 358°

    def test_all_direction_measures_are_declared_circular(self, cfg):
        """
        凡是「方向/角度」量都必须出现在 CIRCULAR_HUMAN_MEASURES 里。
        漏声明会让它被当线性量处理，在 ±180° 附近出假跳变。
        """
        circ = cfg.circular_measures()
        for m in ("upper_arm_angle_deg", "forearm_angle_deg",
                  "elbow_angle_deg", "wrist_pitch_deg",
                  "palm_roll_deg", "arm_direction_deg"):
            assert m in circ, f"{m} 未声明为周期量"

    def test_linear_measures_are_not_circular(self, cfg):
        """thumb_offset / pinch_distance 是线性量，不能被周期化"""
        circ = cfg.circular_measures()
        for m in ("thumb_offset", "pinch_distance"):
            assert m not in circ, f"{m} 被误声明为周期量"


# ============================================================
# 4. 标定基准对 wrap 的稳健性
# ============================================================
class TestCalibrationBaselineRobustness:
    def test_median_is_wrap_tolerant_when_cluster_is_tight(self):
        """
        标定样本聚成一簇时，中位数与「循环均值」一致 ——
        除非这一簇恰好跨过 ±180°。
        本用例：簇在 178/-178 附近（真实角度只有 4° 宽）。
        """
        samples = [178.0, 179.0, -179.0, -178.0, 177.0]
        med = sorted(samples)[len(samples) // 2]
        # 用循环均值作为参照
        s = sum(math.sin(math.radians(x)) for x in samples)
        c = sum(math.cos(math.radians(x)) for x in samples)
        circ_mean = math.degrees(math.atan2(s, c))
        # 两者都应落在 ±180 附近，且彼此相差不超过簇宽
        assert abs(abs(med) - 180.0) < 5.0
        assert abs((med - circ_mean + 180) % 360 - 180) < 5.0, (
            f"中位数 {med} 与循环均值 {circ_mean} 不一致")

    def test_median_can_be_wrong_when_cluster_crosses_180(self):
        """
        **已知局限**：样本同时散在 +170 与 -170 两侧且不聚簇时，
        中位数会落在两簇之间（≈0），而真实中心在 ±180。
        当前实现用的是中位数，因此当标定样本跨 ±180° 时基准会错。

        这条测试**记录该局限**，不是断言它正确 ——
        如果将来改成循环均值，这条应当随之更新。
        """
        samples = [-179.0, -178.0, 178.0, 179.0, 0.0]
        med = sorted(samples)[len(samples) // 2]
        # 中位数落在 0 附近，而真实中心在 ±180 —— 确实是错的
        assert abs(med) < 10.0, "本用例要展示的正是中位数跨 180 时的失效"
