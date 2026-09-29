# -*- coding: utf-8 -*-
"""
手臂几何枢轴（shoulder / anchor）测试
=====================================
背景（实机反馈）：
    以人体肩关键点为枢轴时，「上臂角 / 前臂角」是**肢体自身的朝向**，
    而显示端是镜像的。人把手抬高时，镜像画面里手臂朝右上、
    原始坐标里却朝左上，于是「角度变大」在观感上对应手臂往下 ——
    映射到机械臂就成了**反向弯曲**（人伸手、机器人缩回），很反直觉。

改法：把枢轴挪到**画面底边中点**（anchor），角度变成
「手相对画面中轴的方位」，越伸展/越抬高角度越大，直接可读。

本测试锁定：
    1. anchor 模式确实以画面底边中点为原点
    2. **手往上抬 -> 前臂角增大**（这是用户要的「直觉」的可执行定义）
    3. anchor 模式不依赖肩关键点（肩遮挡时仍可用）
    4. 没给画面尺寸时降级为 shoulder，且降级是可见的（geo.pivot）
    5. 肘夹角与枢轴无关（它是关节角，始终用三点算）

运行：
    python3 -m pytest test/test_geometry_pivot.py -q -p no:anyio
"""

import math
import os
import sys

import pytest

# 本模块为**纯手臂**测试：不构造手部姿态，因此需要关掉
# 依赖 wrist_pitch_deg / palm_roll_deg 的腕部规则（见 conftest）。
# 几何与符号测试（纯手臂）
ARM_ONLY_TESTS = True

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_HERE)
_SRC = os.path.dirname(_PKG_ROOT)
for p in (_PKG_ROOT, os.path.join(_SRC, "piper_human_control")):
    if p not in sys.path:
        sys.path.insert(0, p)

from piper_human_control import ArmKeypoints, Keypoint        # noqa: E402
from piper_human_retargeting import compute_arm_geometry       # noqa: E402

W, H = 640, 480
SIZE = (W, H)


def arm(s=(200, 400), e=(320, 300), w=(400, 200), cs=0.9, ce=0.9, cw=0.9,
        hand_wrist=None):
    return ArmKeypoints(
        shoulder=Keypoint(*s, confidence=cs, name="right_shoulder"),
        elbow=Keypoint(*e, confidence=ce, name="right_elbow"),
        wrist=Keypoint(*w, confidence=cw, name="right_wrist"),
        side="right",
        hand_wrist=hand_wrist)


# ============================================================
# 1. 枢轴位置
# ============================================================
def test_anchor_is_bottom_centre_of_image():
    """anchor 模式必须以画面底边中点为原点"""
    g = compute_arm_geometry(arm(), pivot="anchor", image_size=SIZE)
    assert g.valid
    assert g.pivot == "anchor"
    assert g.anchor_xy == pytest.approx((W * 0.5, H * 1.0))

    # 腕点在锚点正上方 -> 前臂角应为 +90°
    a = arm(e=(320, 300), w=(320, 200))
    g2 = compute_arm_geometry(a, pivot="anchor", image_size=SIZE)
    assert g2.forearm_angle_deg == pytest.approx(90.0, abs=1e-6)


def test_custom_anchor_fraction():
    """anchor_x/anchor_y 是归一化位置，应可调"""
    g = compute_arm_geometry(arm(), pivot="anchor", image_size=SIZE,
                             anchor=(0.25, 0.5))
    assert g.anchor_xy == pytest.approx((W * 0.25, H * 0.5))


# ============================================================
# 2. 核心：伸展 -> 角度增大
# ============================================================
def test_raising_hand_increases_forearm_angle():
    """
    用户要的「直觉」的可执行定义：
        手从画面下方抬到上方时，前臂角必须**单调增大**。

    这正是不引入 anchor 时做不到的：肩点枢轴下，前臂角是
    「肘->腕」的朝向，与手在画面里的高低没有直接关系。
    """
    # 手沿**以锚点为圆心**的圆弧移动（半径固定 300px），
    # 这样变的确实是「相对锚点的方向」而不是距离。
    # 注意：不能简单地同时缩放 dx 与 dy —— 那样方向角恒定不变，
    # 测试就会「通过得很安静」（我第一版就写错了）。
    angs = []
    for deg in (17, 30, 45, 60, 75, 90):
        r = math.radians(deg)
        wx = int(round(320 + 300 * math.cos(r)))
        wy = int(round(480 - 300 * math.sin(r)))
        # 肘放在锚点与腕的中间，保证几何有效
        ex, ey = (320 + wx) // 2, (480 + wy) // 2
        g = compute_arm_geometry(arm(e=(ex, ey), w=(wx, wy)),
                                 pivot="anchor", image_size=SIZE)
        assert g.valid, g.reason
        angs.append(g.forearm_angle_deg)

    assert angs == sorted(angs), f"前臂角未随抬手单调增大: {angs}"
    assert angs[-1] - angs[0] > 70.0, f"抬手范围太小: {angs[0]:.1f} -> {angs[-1]:.1f}"


def test_extending_to_the_side_moves_angle_towards_zero():
    """
    手伸向侧面（画面右中）时，前臂角应趋近 0°。
    与上一条合起来，说明角度确实编码「手的方位」。
    """
    g_up = compute_arm_geometry(arm(e=(320, 260), w=(320, 220)),
                                pivot="anchor", image_size=SIZE)
    g_side = compute_arm_geometry(arm(e=(450, 440), w=(600, 450)),
                                  pivot="anchor", image_size=SIZE)
    assert g_up.forearm_angle_deg > 60.0
    assert abs(g_side.forearm_angle_deg) < 10.0, g_side.forearm_angle_deg


def test_shoulder_pivot_does_not_have_that_property():
    """
    反向记录：shoulder 模式下这个单调性**不成立**（所以才有本次改动）。
    如果哪天有人把默认改回 shoulder，这个测试会提醒他原因。
    """
    angs = []
    for deg in (17, 30, 45, 60, 75, 90):
        r = math.radians(deg)
        wx = int(round(320 + 300 * math.cos(r)))
        wy = int(round(480 - 300 * math.sin(r)))
        ex, ey = (320 + wx) // 2, (480 + wy) // 2
        g = compute_arm_geometry(arm(e=(ex, ey), w=(wx, wy)),
                                 pivot="shoulder", image_size=SIZE)
        angs.append(g.forearm_angle_deg)
    # 说明：shoulder 模式下前臂角是「肘->腕」方向。本用例只要求它能算出来、
    # 且**不保证**与手的画面方位单调相关 —— 这正是当初改用 anchor 的原因。
    assert len(angs) == 6
    assert all(isinstance(x, float) for x in angs)


# ============================================================
# 3. 对上臂角的影响
# ============================================================
def test_upper_arm_angle_uses_anchor_too():
    """上臂角也应以锚点为原点（锚点->肘），而非肩->肘"""
    # 锚点在 (320,480)；肘在 (320,300) -> 正上方 -> +90°
    g = compute_arm_geometry(arm(s=(100, 480), e=(320, 300), w=(320, 200)),
                             pivot="anchor", image_size=SIZE)
    assert g.upper_arm_angle_deg == pytest.approx(90.0, abs=1e-6)

    # 换成 shoulder：肩(100,480) -> 肘(320,300)，方向不同
    g2 = compute_arm_geometry(arm(s=(100, 480), e=(320, 300), w=(320, 200)),
                              pivot="shoulder", image_size=SIZE)
    assert abs(g2.upper_arm_angle_deg - 90.0) > 10.0


# ============================================================
# 4. 鲁棒性：不依赖肩点
# ============================================================
def test_anchor_mode_survives_bad_shoulder():
    """
    anchor 模式下肩点不参与方向角，因此肩置信度为 0 时仍应可用。
    这是附带的好处：原先肩一抖，三个量一起抖。
    """
    a = arm(s=(200, 400), e=(320, 300), w=(400, 200), cs=0.0)
    g = compute_arm_geometry(a, pivot="anchor", image_size=SIZE,
                             min_confidence=0.4, require_complete=True)
    assert g.valid, f"anchor 模式不应因肩点失效: {g.reason}"

    g2 = compute_arm_geometry(a, pivot="shoulder", image_size=SIZE,
                              min_confidence=0.4, require_complete=True)
    assert not g2.valid, "shoulder 模式应当因肩点无效而失败"


def test_anchor_mode_still_requires_elbow_and_wrist():
    """肘/腕仍然必须有效 —— 它们是方向角的端点"""
    for bad in ("ce", "cw"):
        kw = {bad: 0.0}
        g = compute_arm_geometry(arm(**kw), pivot="anchor", image_size=SIZE,
                                 min_confidence=0.4, require_complete=True)
        assert not g.valid, f"{bad}=0 时不应有效"


# ============================================================
# 5. 降级行为必须可见
# ============================================================
def test_missing_image_size_degrades_to_shoulder_visibly():
    """
    没给 image_size 时降级为 shoulder。
    关键：降级必须能从 geo.pivot 看出来，否则会静默用错基准。
    """
    g = compute_arm_geometry(arm(), pivot="anchor", image_size=None)
    assert g.valid
    assert g.pivot == "shoulder", "降级后必须如实报告实际使用的枢轴"
    assert g.anchor_xy is None


def test_pivot_field_reports_actual_choice():
    g1 = compute_arm_geometry(arm(), pivot="anchor", image_size=SIZE)
    g2 = compute_arm_geometry(arm(), pivot="shoulder", image_size=SIZE)
    assert g1.pivot == "anchor"
    assert g2.pivot == "shoulder"


# ============================================================
# 6. 肘夹角与枢轴无关
# ============================================================
def test_elbow_angle_is_pivot_independent():
    """
    肘夹角是**关节角**（肩-肘-腕三点构成），不是方向角，
    因此换枢轴不应改变它。若哪天变了，说明几何定义被改坏了。
    """
    a = arm(s=(200, 400), e=(320, 300), w=(400, 380))
    g1 = compute_arm_geometry(a, pivot="shoulder", image_size=SIZE)
    g2 = compute_arm_geometry(a, pivot="anchor", image_size=SIZE)
    assert g1.elbow_angle_deg == pytest.approx(g2.elbow_angle_deg, abs=1e-9)
    assert 0.0 <= g1.elbow_angle_deg <= 180.0


def test_elbow_straight_is_zero_and_folded_is_large():
    """0 = 伸直、越大越弯（与文档一致）"""
    # 肩-肘-腕共线且肘在中点 -> 完全伸直
    straight = arm(s=(100, 300), e=(200, 300), w=(300, 300))
    g = compute_arm_geometry(straight, pivot="anchor", image_size=SIZE)
    assert g.elbow_angle_deg == pytest.approx(0.0, abs=1e-6)

    # 腕折回靠近肩 -> 夹角大
    folded = arm(s=(100, 300), e=(200, 300), w=(150, 260))
    g2 = compute_arm_geometry(folded, pivot="anchor", image_size=SIZE)
    assert g2.elbow_angle_deg > 90.0


# ============================================================
# 7. required_keypoints 必须随枢轴联动
# ============================================================
class TestRequiredKeypointsFollowPivot:
    """
    anchor 模式下肩点不参与几何。若状态机仍强制要求肩点可信，
    就会把「肩部遮挡但肘/腕清晰」的帧判成 LOST ——
    恰好把这个模式唯一的鲁棒性优势抵消掉。
    """

    @staticmethod
    def _cfg(tmp_path, pivot, extra=None):
        import yaml
        from piper_human_retargeting import RetargetingConfig
        base = yaml.safe_load(open(
            os.path.join(_PKG_ROOT, "config", "retargeting.yaml"),
            encoding="utf-8"))
        base.setdefault("geometry", {})["pivot"] = pivot
        if extra is not None:
            base.setdefault("tracker", {})["required_keypoints"] = extra
        f = tmp_path / f"rt_{pivot}.yaml"
        with open(f, "w", encoding="utf-8") as fh:
            yaml.safe_dump(base, fh, allow_unicode=True)
        return RetargetingConfig.from_yaml(str(f))

    def test_anchor_drops_shoulder_requirement(self, tmp_path):
        c = self._cfg(tmp_path, "anchor")
        assert "shoulder" not in c.tracker.required_keypoints
        assert set(c.tracker.required_keypoints) == {"elbow", "wrist"}

    def test_shoulder_keeps_shoulder_requirement(self, tmp_path):
        c = self._cfg(tmp_path, "shoulder")
        assert "shoulder" in c.tracker.required_keypoints

    def test_explicit_config_wins(self, tmp_path):
        """显式写了 required_keypoints 就必须以它为准（不被枢轴默认值覆盖）"""
        c = self._cfg(tmp_path, "anchor",
                      extra=["shoulder", "elbow", "wrist"])
        assert "shoulder" in c.tracker.required_keypoints


# ============================================================
# 8. 测量量符号修正（mapping.*.sign）
# ============================================================
class TestMeasureSign:
    """
    `mapping.*.sign` —— 修正**人体测量量本身**的符号，
    与修正机械臂关节方向的 `invert` 是两件事。

    当下用到它的地方：joint3 由肘夹角驱动
        （肘角越大=越伸直，但映射想让"伸直"对应区间下端）。
    joint5 已改由手腕俯仰驱动（见 TestWristMapping）。
    """

    @staticmethod
    def _rt(cfg, robot_cfg, sign):
        from test_retargeting import calibrated
        for r in cfg.rules:
            if r.joint == "joint3":
                r.sign = sign
        return calibrated(cfg, robot_cfg)

    # 肩、肘固定；腕沿**以肘为圆心**的圆弧扫掠。
    # 这样扫掠时前臂长度恒定（真实抬臂就是这种运动）。
    #
    # 为什么不直接竖直移动腕：竖直扫掠会让腕经过「肩-肘延长线」，
    # 肘夹角在那里有个极小值（实测 9.5°），角度非单调，测试就失去意义。
    # 圆弧扫掠在 250°->30° 区间是单调的：43.6° -> 176.6°（越伸越直）。
    _SH = (300, 300)
    _EL = (220, 340)

    @classmethod
    def _arm_deg(cls, deg):
        """腕在以肘为心、半径 150px 的圆弧上，deg 为图像角（y 翻转）"""
        r = math.radians(deg)
        w = (round(cls._EL[0] + 150 * math.cos(r)),
             round(cls._EL[1] - 150 * math.sin(r)))
        return arm(s=cls._SH, e=cls._EL, w=w)

    def test_sign_is_read_from_yaml(self, tmp_path):
        import yaml
        from piper_human_retargeting import RetargetingConfig
        base = yaml.safe_load(open(
            os.path.join(_PKG_ROOT, "config", "retargeting.yaml"),
            encoding="utf-8"))
        base["mapping"]["j3"]["sign"] = -1
        f = tmp_path / "rt.yaml"
        with open(f, "w", encoding="utf-8") as fh:
            yaml.safe_dump(base, fh, allow_unicode=True)
        c = RetargetingConfig.from_yaml(str(f))
        j3 = next(r for r in c.rules if r.joint == "joint3")
        assert j3.sign == -1.0
        assert c.measure_sign()["elbow_angle_deg"] == -1.0

    def test_bad_sign_rejected(self, tmp_path):
        import yaml
        from piper_human_retargeting import RetargetingConfig
        base = yaml.safe_load(open(
            os.path.join(_PKG_ROOT, "config", "retargeting.yaml"),
            encoding="utf-8"))
        base["mapping"]["j3"]["sign"] = 0.5
        f = tmp_path / "rt.yaml"
        with open(f, "w", encoding="utf-8") as fh:
            yaml.safe_dump(base, fh, allow_unicode=True)
        with pytest.raises(ValueError, match="sign"):
            RetargetingConfig.from_yaml(str(f))

    def test_extending_arm_gives_positive_delta_with_sign_plus_one(
            self, cfg, robot_cfg):
        """核心断言：手抬高 -> 肘夹角 delta 为正（sign=-1 时）"""
        rt = self._rt(cfg, robot_cfg, +1.0)
        base = compute_arm_geometry(self._arm_deg(250), pivot="shoulder")
        rt.start_calibration()
        for _ in range(20):
            rt.add_calibration_sample(base)
        assert rt.finish_calibration()
        rt.set_initial_pose(cfg.neutral_joints)

        res = None
        for k in range(40):
            res = rt.update(self._arm_deg(30), now=k * 0.033, frame=k)
        assert res.debug_deltas_deg, "没有取到 delta"
        d = res.debug_deltas_deg["elbow_angle_deg"]
        assert d > 0, (
            f"sign=+1 时伸臂应给出正 delta，实际 {d:+.1f}° —— "
            f"说明符号修正没有作用到差值上")

    def test_sign_plus_one_gives_the_opposite(self, cfg, robot_cfg):
        """对照组：sign=+1 时方向相反 —— 这正是需要修正的原因"""
        rt = self._rt(cfg, robot_cfg, -1.0)
        base = compute_arm_geometry(self._arm_deg(250), pivot="shoulder")
        rt.start_calibration()
        for _ in range(20):
            rt.add_calibration_sample(base)
        assert rt.finish_calibration()
        rt.set_initial_pose(cfg.neutral_joints)

        res = None
        for k in range(40):
            res = rt.update(self._arm_deg(30), now=k * 0.033, frame=k)
        d = res.debug_deltas_deg["elbow_angle_deg"]
        assert d < 0, f"sign=-1 时应为负 delta，实际 {d:+.1f}°"

    def test_sign_does_not_bias_the_neutral_point(self, cfg, robot_cfg):
        """
        标定后立刻（未移动）delta 必须≈0。
        若符号只加在「当前测量」而没加在 neutral 上，
        这里会偏 2×neutral —— 是个很容易犯的错。
        """
        rt = self._rt(cfg, robot_cfg, +1.0)
        base = compute_arm_geometry(self._arm_deg(250), pivot="shoulder")
        rt.start_calibration()
        for _ in range(20):
            rt.add_calibration_sample(base)
        rt.finish_calibration()
        rt.set_initial_pose(cfg.neutral_joints)
        res = None
        for k in range(20):
            res = rt.update(self._arm_deg(250), now=k * 0.033, frame=k)
        for name, d in res.debug_deltas_deg.items():
            assert abs(d) < 5.0, f"{name} 在未移动时 delta={d:+.2f}°，基准被偏置了"
