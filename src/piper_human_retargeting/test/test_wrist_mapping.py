# -*- coding: utf-8 -*-
"""
腕部映射测试（joint5 / joint6 由手部姿态驱动）
==============================================
背景：
    原先 joint5 由「前臂方向角」驱动、joint6 固定为 0。
    实测发现 joint5 对腕**位置**影响极小（-2.7°/rad），
    真正需要的是**腕部姿态**——于是改为：

        wrist_pitch_deg（手朝向相对前臂）  -> joint5   （实测纯 pitch）
        palm_roll_deg  （掌横轴图像内朝向）-> joint6   （实测纯 roll）

    实测依据（TF 正运动学，base_link -> link6）：
        joint5 -1.0 -> 1.0：末端指向在竖直平面内转约 100°，
                            末端位置几乎不动（x 0.108~0.125）
        joint6 -2.0 -> 2.0：末端姿态绕自身轴滚转约 220°，
                            末端位置几乎不动

本测试锁定：
    1. 两个量确实取自手部 21 点（而不是手臂）
    2. 手部缺失时**不伪造 0**，相关关节保持不动
    3. 腕俯仰/掌滚转变化能真正驱动 joint5 / joint6
    4. 区间对称、neutral 落在中点（否则启动校验会拒绝）

运行：
    python3 -m pytest test/test_wrist_mapping.py -q -p no:anyio
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

from piper_human_control import (ArmKeypoints, Keypoint,        # noqa: E402
                                 HandKeypoints)
from piper_human_perception import (HandDetection,              # noqa: E402
                                    compute_hand_pose)
from piper_human_retargeting import (ArmGeometry, Retargeter,   # noqa: E402
                                     RetargetingConfig, TrackState)

W, H = 640, 480


# ============================================================
# 构造：一只可摆姿态的手
# ============================================================
def make_arm(s=(300, 300), e=(220, 340), w=(160, 380)):
    return ArmKeypoints(
        shoulder=Keypoint(*s, confidence=0.9, name="left_shoulder"),
        elbow=Keypoint(*e, confidence=0.9, name="left_elbow"),
        wrist=Keypoint(*w, confidence=0.9, name="left_wrist"),
        side="left",
        hand_wrist=Keypoint(*w, confidence=0.9, name="left_wrist_from_hand"))


def make_hand(root_xy, mcp_xy, index_xy, pinky_xy, side="left", thumb=None,
              tips=None):
    """
    构造手部关键点。

    必填：middle_mcp / index_mcp / pinky_mcp（定掌面与掌横轴）
    选填：thumb_tip（拇指偏移与捏合）、四指 *_tip（汇聚度与捏合）
    """
    lms = {
        "middle_mcp": Keypoint(*mcp_xy, confidence=0.9, name="middle_mcp"),
        "index_mcp": Keypoint(*index_xy, confidence=0.9, name="index_mcp"),
        "pinky_mcp": Keypoint(*pinky_xy, confidence=0.9, name="pinky_mcp"),
    }
    if thumb is not None:
        lms["thumb_tip"] = Keypoint(*thumb, confidence=0.9, name="thumb_tip")
    if tips:
        for f, xy in tips.items():
            lms[f"{f}_tip"] = Keypoint(*xy, confidence=0.9, name=f"{f}_tip")
    h = HandKeypoints(
        side=side,
        wrist=Keypoint(*root_xy, confidence=0.9, name="left_wrist"),
        hand_root=Keypoint(*root_xy, confidence=0.95, name="hand_root"),
        landmarks=lms,
        palm_center=lms["middle_mcp"],
        source="test",
    )
    return HandDetection(hand=h)


def make_hand_at(root, mcp, index, pinky, arm, thumb=None, tips=None):
    """手的根点与 arm 的腕点一致，避免几何不一致"""
    return make_hand(root, mcp, index, pinky, side=arm.side,
                     thumb=thumb, tips=tips)


def rotate_hand(arm, hand_dir_deg, palm_axis_deg=None, thumb_side=0.0):
    """
    构造一只**整只手绕腕旋转到指定朝向**的手。

    Args:
        hand_dir_deg:  腕->middle_mcp 的方向角（度，y 翻转后「向上为正」）
        palm_axis_deg: index_mcp->pinky_mcp 的方向角；None 表示与手的朝向垂直
                       （即掌横轴随手一起转，纯俯仰）

    为什么需要它：手写关键点很容易「只想改俯仰，却顺带改了掌横轴」，
    于是独立性测试会误报。用角度参数构造，两个自由度才真正解耦。
    """
    def pt(deg, r):
        a = math.radians(deg)
        return (round(arm.effective_wrist.x + r * math.cos(a)),
                round(arm.effective_wrist.y - r * math.sin(a)))

    mcp = pt(hand_dir_deg, 80)
    if palm_axis_deg is None:
        palm_axis_deg = hand_dir_deg - 90.0        # 与手朝向垂直（正常握姿）
    idx = pt(palm_axis_deg, 40)
    pky = pt(palm_axis_deg + 180.0, 40)
    # 拇指位置：沿掌横轴按 thumb_side 偏移。
    # ⚠️ 不能「沿掌横轴的垂直方向、按角度放点」—— 那样放出来的点恰好
    #    落在掌横轴上，投影被 offset 归一化抵消，thumb_offset 恒为 0。
    #    直接按「掌中心 + thumb_side × 掌宽 × 掌横轴单位向量」构造。
    ax = math.radians(palm_axis_deg)
    lat_u = (math.cos(ax), -math.sin(ax))       # pinky -> index 方向
    cx = (idx[0] + pky[0]) / 2.0
    cy = (idx[1] + pky[1]) / 2.0
    L = 80.0                                     # 掌宽（idx 到 pky 距离）
    thumb = (round(cx + thumb_side * L * lat_u[0]),
             round(cy + thumb_side * L * lat_u[1]))
    # 四指指尖：沿手朝向伸出去。给出四个略有差异的点，
    # 便于 finger_tips_converged / pinch_distance 有真实值。
    tips = {}
    for i, f in enumerate(("index", "middle", "ring", "pinky")):
        off = (i - 1.5) * 12.0                      # 指尖之间横向散开 12px
        tips[f] = pt(hand_dir_deg, 150.0 + off * 0.0)
        # 沿掌横轴做一点偏移，避免四点重合
        ax = math.radians(palm_axis_deg)
        tips[f] = (round(tips[f][0] + off * math.cos(ax)),
                   round(tips[f][1] - off * math.sin(ax)))
    return make_hand((arm.effective_wrist.x, arm.effective_wrist.y),
                     mcp, idx, pky, arm.side, thumb=thumb, tips=tips)


# ============================================================
# 1. 姿态量的定义
# ============================================================
class TestHandPoseDefinition:
    # 前臂方向（肘(220,340) -> 腕(160,380)）= (-60, +40) -> 约 146°
    # 图像角（y 翻转后「向上为正」）：肘(220,340) -> 腕(160,380) 指向左下
    _FOREARM_DIR = 214.0

    def test_pitch_zero_when_hand_along_forearm(self):
        """手与前臂成一直线时，腕俯仰 ≈ 0"""
        a = make_arm(s=(300, 300), e=(220, 340), w=(160, 380))
        d = rotate_hand(a, self._FOREARM_DIR)
        hp = compute_hand_pose(a, d)
        assert hp.valid, hp.reason
        assert abs(hp.wrist_pitch_deg) < 8.0, hp.wrist_pitch_deg

    def test_pitch_negative_when_hand_bends_up(self):
        """
        手往上翘 -> 腕俯仰为**负**。

        符号来源（容易搞反，这里写死约定）：
            pitch = wrap(手朝向 - 前臂朝向)。
            本用例的前臂朝**右下**（肘(220,340)->腕(160,380)，约 -146°），
            手朝**上**（约 +90°），相减并归一化后落在负半轴。
            所以「手往上翘 = pitch 负」，映射里用 sign/invert 去对齐方向。
        """
        a = make_arm(s=(300, 300), e=(220, 340), w=(160, 380))
        d = rotate_hand(a, 90.0)          # 手朝上 = 相对前臂上翘
        hp = compute_hand_pose(a, d)
        assert hp.valid, hp.reason
        assert hp.wrist_pitch_deg < -20.0, hp.wrist_pitch_deg

    def test_pitch_positive_when_hand_bends_down(self):
        """
        符号约定（写死在这里，避免反复搞错）：
            本用例前臂指向**左下**（图像角 214°）。
            pitch = wrap(手朝向 - 前臂朝向)，因此
                手转到 270°（正下）   -> pitch = +56°
                手转到  90°（正上）   -> pitch = -124°
            即前臂在左下时，「手往下/往右偏」得正 pitch。
        """
        a = make_arm(s=(300, 300), e=(220, 340), w=(160, 380))
        d = rotate_hand(a, 270.0)
        hp = compute_hand_pose(a, d)
        assert hp.valid, hp.reason
        assert hp.wrist_pitch_deg == pytest.approx(56.0, abs=3.0), \
            hp.wrist_pitch_deg

    def test_pitch_sign_matches_forearm_relative_definition(self):
        """pitch 就是「手朝向 - 前臂朝向」的归一化差值（定义自证）"""
        a = make_arm(s=(300, 300), e=(220, 340), w=(160, 380))
        for deg in (214.0, 270.0, 90.0, 320.0):
            hp = compute_hand_pose(a, rotate_hand(a, deg))
            expect = (deg - 214.0 + 180.0) % 360.0 - 180.0
            assert hp.wrist_pitch_deg == pytest.approx(expect, abs=3.0), \
                (deg, hp.wrist_pitch_deg, expect)

    def test_thumb_offset_sign_follows_thumb_side(self):
        """
        拇指偏向食指侧 -> thumb_offset 为正；偏向小指侧 -> 为负。
        符号约定写死在这里：掌横轴取「小指掌指 -> 食指掌指」。
        """
        a = make_arm(s=(300, 300), e=(220, 340), w=(160, 380))
        pos = compute_hand_pose(a, rotate_hand(a, 141.0, -166.0, thumb_side=+0.45))
        neg = compute_hand_pose(a, rotate_hand(a, 141.0, -166.0, thumb_side=-0.45))
        assert pos.valid and neg.valid, (pos.reason, neg.reason)
        assert pos.thumb_offset > 0.05, pos.thumb_offset
        assert neg.thumb_offset < -0.05, neg.thumb_offset
        assert abs(pos.thumb_offset - neg.thumb_offset) > 0.2, (
            pos.thumb_offset, neg.thumb_offset)

    def test_palm_roll_still_available_as_observation(self):
        """palm_roll 不再驱动关节，但仍应作为观测量给出"""
        a = make_arm(s=(300, 300), e=(220, 340), w=(160, 380))
        d1 = rotate_hand(a, 141.0, -166.0)
        d2 = rotate_hand(a, 141.0, -60.0)
        r1 = compute_hand_pose(a, d1).palm_roll_deg
        r2 = compute_hand_pose(a, d2).palm_roll_deg
        assert abs(r1 - r2) > 20.0, (r1, r2)

    def test_pinch_distance_shrinks_when_thumb_touches_fingers(self):
        """拇指贴近四指 -> pinch_distance 变小"""
        a = make_arm(s=(300, 300), e=(220, 340), w=(160, 380))
        far = compute_hand_pose(a, rotate_hand(a, 141.0, -166.0, thumb_side=0.0))
        assert far.valid, far.reason
        # 手写一只「拇指尖几乎碰到四指指尖」的手
        d = make_hand_at((160, 380), (110, 420), (150, 410), (70, 430), a,
                         thumb=(120, 300), tips={"index": (118, 302),
                                                 "middle": (122, 303),
                                                 "ring": (116, 304),
                                                 "pinky": (124, 305)})
        near = compute_hand_pose(a, d)
        assert near.valid, near.reason
        assert near.pinch_distance < far.pinch_distance, (
            near.pinch_distance, far.pinch_distance)

    def test_missing_hand_is_invalid_not_zero(self):
        """没有手部数据时必须 valid=False —— 不能悄悄返回 0 冒充观测"""
        a = make_arm()
        hp = compute_hand_pose(a, None)
        assert not hp.valid
        assert "手部" in hp.reason

    def test_degenerate_palm_axis_is_invalid(self):
        a = make_arm()
        d = make_hand_at((160, 380), (110, 420), (110, 420), (110, 420), a,
                         thumb=(120, 300),
                         tips={"index": (140, 300), "middle": (140, 305),
                               "ring": (140, 310), "pinky": (140, 315)})
        hp = compute_hand_pose(a, d)
        assert not hp.valid
        assert "掌横轴" in hp.reason


# ============================================================
# 2. 配置
# ============================================================
class TestWristConfig:
    @staticmethod
    def _cfg(tmp_path, mutate=None):
        import yaml
        base = yaml.safe_load(open(
            os.path.join(_PKG_ROOT, "config", "retargeting.yaml"),
            encoding="utf-8"))
        if mutate:
            mutate(base)
        f = tmp_path / "rt.yaml"
        with open(f, "w", encoding="utf-8") as fh:
            yaml.safe_dump(base, fh, allow_unicode=True)
        return RetargetingConfig.from_yaml(str(f))

    def test_repo_config_has_wrist_rules(self, tmp_path):
        c = self._cfg(tmp_path)
        by_joint = {r.joint: r for r in c.rules}
        assert "joint5" in by_joint and "joint6" in by_joint
        assert by_joint["joint5"].human == "wrist_pitch_deg"
        # joint6 改由「拇指相对四指的位置」驱动（用户定义）
        assert by_joint["joint6"].human == "thumb_offset"

    def test_joint5_and_6_no_longer_fixed(self, tmp_path):
        c = self._cfg(tmp_path)
        assert "joint5" not in c.fixed_joints
        assert "joint6" not in c.fixed_joints
        assert {"joint5", "joint6"} <= set(c.controlled_joints())

    def test_ranges_are_anchored(self, tmp_path):
        """
        human 区间必须关于 Δ=0 对称、robot 区间中点必须等于 neutral，
        否则启动校验会拒绝（这是防止「一上电就偏离中立位」的硬约束）。
        """
        c = self._cfg(tmp_path)
        for r in c.rules:
            assert r.human_min == pytest.approx(-r.human_max), r.key
            mid = (r.robot_min + r.robot_max) / 2.0
            if r.invert:
                mid = r.robot_min + r.robot_max - mid
            assert mid + r.offset == pytest.approx(
                c.neutral_joints[c.joint_names.index(r.joint)], abs=1e-9), r.key

    def test_ranges_have_usable_overlap_with_joint_limits(self, tmp_path):
        """
        映射区间与关节限位必须有**足够的可用重叠**。

        注意：不要求区间完全落在限位内。j2 的区间就是刻意略超下限的
        （robot_min=-1.00 < 限位下限 0），超出部分由 clamp 裁掉 ——
        这是已知且有意的取舍，写在 YAML 注释里。
        真正要防的是「重叠太小，一动就被裁光」。
        """
        c = self._cfg(tmp_path)
        for r in c.rules:
            lo = max(r.robot_min, r.joint_lower)
            hi = min(r.robot_max, r.joint_upper)
            span = hi - lo
            assert span > 0.5, f"{r.key} 可用行程只剩 {span:.2f} rad，太小"


# ============================================================
# 3. 端到端：手部姿态真的驱动 joint5 / joint6
# ============================================================
class TestWristDrivesJoints:
    @staticmethod
    def _rt(cfg, robot_cfg, arm0, hand0):
        rt = Retargeter(cfg, robot_cfg)
        # 用第一帧的几何做标定基准
        from piper_human_retargeting import compute_arm_geometry
        g = compute_arm_geometry(arm0, pivot="shoulder")
        assert g.valid, g.reason
        from piper_human_perception import compute_hand_pose as _chp
        hp0 = _chp(arm0, hand0)
        assert hp0.valid, hp0.reason
        rt.start_calibration()
        for _ in range(20):
            # 腕部量必须一起喂进标定，否则腕关节基准为空、永远不动
            rt.add_calibration_sample(g, extra_measures=hp0.as_dict())
        assert rt.finish_calibration()
        rt.set_initial_pose(cfg.neutral_joints)
        return rt

    def _run(self, rt, arm, hand, n=40):
        res = None
        for k in range(n):
            res = rt.update(arm, now=k * 0.033, frame=k,
                            hand_pose=compute_hand_pose(arm, hand))
        return res

    def test_wrist_pitch_moves_joint5(self, cfg, robot_cfg):
        a = make_arm(s=(300, 300), e=(220, 340), w=(160, 380))
        neutral_hand = rotate_hand(a, 141.0, -166.0, thumb_side=0.0)
        rt = self._rt(cfg, robot_cfg, a, neutral_hand)
        # 先跑一段中立姿势
        self._run(rt, a, neutral_hand, n=30)
        base = rt.target[cfg.joint_names.index("joint5")]

        # 手往上翘
        up_hand = rotate_hand(a, 90.0, -166.0, thumb_side=0.0)
        res = self._run(rt, a, up_hand, n=60)
        moved = rt.target[cfg.joint_names.index("joint5")] - base
        assert abs(moved) > 0.15, f"joint5 未随腕俯仰变化: Δ={moved:+.4f}"
        assert res.debug_deltas_deg, "没有 delta 记录"

    def test_thumb_offset_moves_joint6(self, cfg, robot_cfg):
        a = make_arm(s=(300, 300), e=(220, 340), w=(160, 380))
        h0 = rotate_hand(a, 141.0, -166.0, thumb_side=0.0)
        rt = self._rt(cfg, robot_cfg, a, h0)
        self._run(rt, a, h0, n=30)
        base = rt.target[cfg.joint_names.index("joint6")]

        # 拇指从小指侧移到食指侧
        h1 = rotate_hand(a, 141.0, -166.0, thumb_side=+0.45)
        self._run(rt, a, h1, n=60)
        moved = rt.target[cfg.joint_names.index("joint6")] - base
        assert abs(moved) > 0.15, f"joint6 未随拇指位置变化: Δ={moved:+.4f}"

    # 中立手：整只手用 rotate_hand 构造，掌横轴固定在一个方向。
    # ⚠️ 不要用手写的关键点当中立姿势：手写点的掌横轴与 rotate_hand 的
    #    默认垂直约定会差十几度，于是「纯俯仰」实验里混进了一点滚转，
    #    独立性测试就会误报（实际踩过，误报值 0.155 rad）。
    _NEUTRAL_HAND_DIR = 141.0
    _NEUTRAL_PALM_AXIS = -166.0

    def test_joint5_and_joint6_are_independent(self, cfg, robot_cfg):
        """腕俯仰只动 joint5、掌滚转只动 joint6（互不串扰）"""
        a = make_arm(s=(300, 300), e=(220, 340), w=(160, 380))
        h0 = rotate_hand(a, self._NEUTRAL_HAND_DIR, self._NEUTRAL_PALM_AXIS)
        rt = self._rt(cfg, robot_cfg, a, h0)
        self._run(rt, a, h0, n=30)
        i5 = cfg.joint_names.index("joint5")
        i6 = cfg.joint_names.index("joint6")
        b5, b6 = rt.target[i5], rt.target[i6]

        # 纯俯仰：手朝向改变，掌横轴**保持与中立完全相同**
        up = rotate_hand(a, 90.0, self._NEUTRAL_PALM_AXIS)
        self._run(rt, a, up, n=60)
        d5_pitch = rt.target[i5] - b5
        d6_pitch = rt.target[i6] - b6
        assert abs(d5_pitch) > 0.15, "俯仰应驱动 joint5"
        assert abs(d6_pitch) < 0.10, f"俯仰不应明显驱动 joint6: {d6_pitch:+.4f}"

    def test_missing_hand_keeps_wrist_still(self, cfg, robot_cfg):
        """
        手部丢失时腕关节必须**保持不动**，而不是被拉到 0
        —— 这是「不伪造观测」的可执行断言。
        """
        a = make_arm(s=(300, 300), e=(220, 340), w=(160, 380))
        h0 = rotate_hand(a, 141.0, -166.0, thumb_side=0.0)
        rt = self._rt(cfg, robot_cfg, a, h0)
        self._run(rt, a, h0, n=40)
        i5 = cfg.joint_names.index("joint5")
        i6 = cfg.joint_names.index("joint6")
        b5, b6 = rt.target[i5], rt.target[i6]

        # hand_pose 传 None（相当于手部识别失败）
        for k in range(40):
            rt.update(a, now=(40 + k) * 0.033, frame=40 + k, hand_pose=None)
        assert rt.target[i5] == pytest.approx(b5, abs=1e-6), \
            f"手部丢失后 joint5 被改动: {b5:+.4f} -> {rt.target[i5]:+.4f}"
        assert rt.target[i6] == pytest.approx(b6, abs=1e-6), \
            f"手部丢失后 joint6 被改动: {b6:+.4f} -> {rt.target[i6]:+.4f}"


# ============================================================
# 4. 可观测性：腕部量必须进 CSV，否则「动没动」看不出来
# ============================================================
class TestWristInCsv:
    """
    腕部列最初漏加，导致运行后完全无法判断 joint5/joint6 是否受控 ——
    与 joint7、delta_* 是同一类「算出来了却没记录」的漏洞。
    """

    def test_csv_has_wrist_columns(self, cfg, robot_cfg):
        a = make_arm(s=(300, 300), e=(220, 340), w=(160, 380))
        h0 = rotate_hand(a, 141.0, -166.0, thumb_side=0.0)
        from piper_human_retargeting import compute_arm_geometry
        g = compute_arm_geometry(a, pivot="shoulder")
        rt = Retargeter(cfg, robot_cfg)
        hp0 = compute_hand_pose(a, h0)
        rt.start_calibration()
        for _ in range(20):
            rt.add_calibration_sample(g, extra_measures=hp0.as_dict())
        rt.finish_calibration()
        rt.set_initial_pose(cfg.neutral_joints)

        res = rt.update(a, now=0.0, frame=0, hand_pose=hp0)
        header = res.debug.csv_header()
        for col in ("raw_wrist_pitch", "flt_wrist_pitch", "delta_wrist_pitch",
                    "raw_palm_roll", "flt_palm_roll", "delta_palm_roll"):
            assert col in header, f"CSV 表头缺少 {col}"

    def test_wrist_columns_are_populated(self, cfg, robot_cfg):
        """腕部列不能恒为 0 —— 那说明又没回填"""
        a = make_arm(s=(300, 300), e=(220, 340), w=(160, 380))
        h0 = rotate_hand(a, 141.0, -166.0, thumb_side=0.0)
        from piper_human_retargeting import compute_arm_geometry
        g = compute_arm_geometry(a, pivot="shoulder")
        rt = Retargeter(cfg, robot_cfg)
        hp0 = compute_hand_pose(a, h0)
        rt.start_calibration()
        for _ in range(20):
            rt.add_calibration_sample(g, extra_measures=hp0.as_dict())
        rt.finish_calibration()
        rt.set_initial_pose(cfg.neutral_joints)

        rows = []
        for k in range(40):
            # 手来回摆，让腕部量有变化
            deg = 110.0 if k % 2 else 170.0
            hp = compute_hand_pose(a, rotate_hand(a, deg, -166.0, thumb_side=0.0))
            res = rt.update(a, now=k * 0.033, frame=k, hand_pose=hp)
            rows.append(res.debug.to_csv_row())
        for col in ("raw_wrist_pitch", "flt_wrist_pitch", "delta_wrist_pitch",
                    "raw_palm_roll"):
            vals = [float(r[col]) for r in rows]
            assert any(abs(v) > 1e-9 for v in vals), f"{col} 恒为 0（没回填）"


# ============================================================
# 5. 单帧突变保护（max_jump）
# ============================================================
class TestJumpProtection:
    """
    实机数据：腕俯仰有 9/1265 帧单帧跳变 >90°（最大 353°）——
    关键点被误估到对侧时 `手朝向 - 前臂朝向` 跨过 ±180° 的**假跳变**。
    EMA 压不住这种跳变（会把输出拉过去一半），必须在 EMA 之前拒绝。

    同时必须**按量设置门限**：掌滚转本身就跨 ±180°，
    真实单帧变化就能到 100°+，用同一个门限必然误伤。
    """

    def test_spike_is_rejected_and_holds_previous(self):
        from piper_human_control import AngleFilter, FilterConfig
        cfg = FilterConfig(alpha=0.5, deadband=0.0, max_jump=60.0)
        f = AngleFilter(cfg, ["wrist_pitch_deg"], "angle",
                        max_jump={"wrist_pitch_deg": 60.0})
        # 正常序列
        for v in (0.0, 5.0, 10.0, 12.0):
            out = f.update({"wrist_pitch_deg": v})
        good = out["wrist_pitch_deg"]
        # 假跳变：单帧跳到 -170（差 180+）
        out = f.update({"wrist_pitch_deg": -170.0})
        assert abs(out["wrist_pitch_deg"] - good) < 1.0, (
            f"跳变未被拒绝: {good:.2f} -> {out['wrist_pitch_deg']:.2f}")

    def test_slow_change_is_not_rejected(self):
        from piper_human_control import AngleFilter, FilterConfig
        cfg = FilterConfig(alpha=1.0, deadband=0.0, max_jump=60.0)
        f = AngleFilter(cfg, ["wrist_pitch_deg"], "angle",
                        max_jump={"wrist_pitch_deg": 60.0})
        f.update({"wrist_pitch_deg": 0.0})
        out = f.update({"wrist_pitch_deg": 30.0})   # 30 < 60，应通过
        assert out["wrist_pitch_deg"] == pytest.approx(30.0, abs=1e-6)

    def test_wide_limit_allows_large_real_roll(self):
        """掌滚转的真实大转动不能被 60° 门限误伤（这正是按量设限的原因）"""
        from piper_human_control import AngleFilter, FilterConfig
        cfg = FilterConfig(alpha=1.0, deadband=0.0, max_jump=0.0)
        f = AngleFilter(cfg, ["palm_roll_deg"], "angle",
                        max_jump={"palm_roll_deg": 170.0})
        f.update({"palm_roll_deg": 14.0})
        out = f.update({"palm_roll_deg": 120.0})    # 真实滚转 ~106°
        assert out["palm_roll_deg"] == pytest.approx(120.0, abs=1e-6),             "掌滚转的真实大转动被误判为跳变"

    def test_config_exposes_per_measure_limits(self, cfg):
        lim = cfg.measure_max_jump()
        assert lim.get("wrist_pitch_deg", 0) > 0, "腕俯仰应有门限"
        # 拇指偏移是归一化量（无 ±180° 问题），不需要宽门限
        assert "thumb_offset" in lim


# ============================================================
# 6. 关节列必须覆盖全部受控关节
# ============================================================
class TestAllControlledJointsInCsv:
    def test_joint6_has_columns(self, cfg, robot_cfg):
        """
        CSV 的关节列曾硬编码为 joint2/joint3/joint5，
        joint6 加入受控集合后**连列都没有** —— 动没动无从判断。
        """
        a = make_arm(s=(300, 300), e=(220, 340), w=(160, 380))
        h0 = rotate_hand(a, 141.0, -166.0, thumb_side=0.0)
        from piper_human_retargeting import compute_arm_geometry
        g = compute_arm_geometry(a, pivot="shoulder")
        rt = Retargeter(cfg, robot_cfg)
        hp0 = compute_hand_pose(a, h0)
        rt.start_calibration()
        for _ in range(20):
            rt.add_calibration_sample(g, extra_measures=hp0.as_dict())
        rt.finish_calibration()
        rt.set_initial_pose(cfg.neutral_joints)
        res = rt.update(a, now=0.0, frame=0, hand_pose=hp0)

        header = set(res.debug.csv_header())
        for j in cfg.controlled_joints():
            assert j in header, f"受控关节 {j} 在 CSV 表头里没有列"
            assert f"mapped_{j}" in header
        row = res.debug.to_csv_row()
        for c in res.debug.csv_header():
            assert c in row, f"表头列 {c} 在行里缺失"


# ============================================================
# 7. 基准懒建立（标定窗口内缺失的量）
# ============================================================
class _FakeHandPose:
    """够用的 HandPose 替身：只需要 valid / as_dict"""

    def __init__(self, wrist_pitch_deg, thumb_offset, valid=True):
        self.wrist_pitch_deg = wrist_pitch_deg
        self.thumb_offset = thumb_offset
        self.valid = valid
        self.palm_roll_deg = 0.0
        self.pinch_distance = 1.0
        self.finger_tips_converged = 0.0

    def as_dict(self):
        return {"wrist_pitch_deg": self.wrist_pitch_deg,
                "thumb_offset": self.thumb_offset}


class TestLazyBaseline:
    """
    实测问题：手部姿态要到约 11.6s 才首次稳定检出，
    而标定窗口只有 2s —— wrist_pitch / thumb_offset 收不到样本，
    基准为空、delta 恒为 0，joint5/joint6 **永远不动**。

    对策：某量首次可用时，把当下值当作它的零点（懒建立）。
    语义等价于「以手进入画面的那一刻为基准」。
    """

    @staticmethod
    def _prepare(cfg, robot_cfg):
        from piper_human_retargeting import (Retargeter,
                                             compute_arm_geometry)
        import test_retargeting as T
        a = T.make_arm(e=(100, 0), w=(100, 100))
        g = compute_arm_geometry(a, pivot="shoulder")
        rt = Retargeter(cfg, robot_cfg)
        rt.start_calibration()
        # 标定时**只给手臂量**，手部量缺失
        extra = {k: v for k, v in g.as_dict().items()}
        for _ in range(20):
            rt.add_calibration_sample(g, extra_measures=extra)
        assert rt.finish_calibration()
        rt.set_initial_pose(cfg.neutral_joints)
        return rt, a

    def test_missing_at_calibration_then_adopted(self, cfg, robot_cfg):
        rt, a = self._prepare(cfg, robot_cfg)
        assert rt.neutral.get("wrist_pitch_deg") is None, \
            "本用例前提：标定时腕部量应当缺失"

        def run(hp, n):
            res = None
            for k in range(n):
                res = rt.update(a, now=k * 0.033, frame=k, hand_pose=hp)
            return res

        # 手部量首次出现 -> 懒建立
        res = run(_FakeHandPose(33.0, 0.42), 25)
        assert rt.neutral.get("wrist_pitch_deg") is not None, \
            "懒建立没有生效：腕部基准仍为空"
        assert "wrist_pitch_deg" in rt._late_baseline
        d0 = abs(res.debug_deltas_deg.get("wrist_pitch_deg", 0.0))
        assert d0 < 3.0, f"刚建立基准时 delta 应≈0，实际 {d0:.2f}"

        # 换一个明显不同的值 -> 应产生明显 delta（说明真的受控）
        res = run(_FakeHandPose(85.0, 0.42), 40)
        d1 = abs(res.debug_deltas_deg.get("wrist_pitch_deg", 0.0))
        assert d1 > 15.0, f"懒建立后腕部量仍未参与映射，delta={d1:.2f}"

    def test_existing_baseline_is_not_overwritten(self, cfg, robot_cfg):
        """已经标定好的量，不能被懒建立改掉"""
        rt, a = self._prepare(cfg, robot_cfg)
        before = rt.neutral.get("elbow_angle_deg")
        assert before is not None
        for k in range(10):
            rt.update(a, now=k * 0.033, frame=k,
                      hand_pose=_FakeHandPose(10.0, 0.1))
        assert rt.neutral.get("elbow_angle_deg") == pytest.approx(before), \
            "已有基准被懒建立覆盖了"
        assert "elbow_angle_deg" not in rt._late_baseline
