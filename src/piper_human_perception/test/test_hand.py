# -*- coding: utf-8 -*-
"""
手部识别单元测试（不依赖摄像头 / MediaPipe / ROS2）
==================================================
覆盖：
    * MediaPipe 21 点定义与索引一致性
    * 手指链、指尖映射
    * HandKeypoints 的「腕部双来源」契约
    * derive_hand_geometry 的朝向 / 张开度 / 分散度（用合成手部数据）
    * YOLOWristHandProvider 基线行为
    * CascadingHandProvider 的自动回退

运行：
    python3 -m pytest test/test_hand.py -q -p no:anyio
"""

import math
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_HERE)
_SRC_ROOT = os.path.dirname(_PKG_ROOT)
for p in (_PKG_ROOT, os.path.join(_SRC_ROOT, "piper_human_control")):
    if p not in sys.path:
        sys.path.insert(0, p)

from piper_human_control import (FINGER_NAMES, ArmKeypoints,     # noqa: E402
                                 HandKeypoints, Keypoint)
from piper_human_perception.hand import (FINGERTIP_NAMES,       # noqa: E402
                                         FINGER_LANDMARK_NAMES,
                                         HAND_CONNECTIONS,
                                         HAND_LANDMARK_INDEX,
                                         HAND_LANDMARK_NAMES,
                                         PALM_CENTER_NAME,
                                         CascadingHandProvider,
                                         HandDetection, HandProvider,
                                         YOLOWristHandProvider,
                                         derive_hand_geometry)
from piper_human_perception.perception import Frame              # noqa: E402


# ============================================================
# 合成手部数据（用于可控地验证几何计算）
# ============================================================
def make_open_hand(cx=320.0, cy=240.0, scale=100.0, angle_deg=90.0):
    """
    构造一只「张开的手」。

    angle_deg: 手掌朝向（度）。90 = 手指朝图像上方（y 减小）。
    scale:     手腕到中指指尖的距离（像素）。

    比例参照**真实实测数据**（见阶段三报告 §0.3）：
        掌宽 / 手长 ≈ 0.65
        中指指尖到手腕 / 掌宽 ≈ 2.86  -> 与 scale 自洽
    这样合成手与 MediaPipe 真实输出的量级一致，
    几何阈值测试才有意义。
    """
    a = math.radians(angle_deg)
    ux, uy = math.cos(a), -math.sin(a)          # 图像坐标下的「手指方向」

    def pt(along, side):
        """along: 沿手指方向的距离; side: 垂直方向偏移"""
        x = cx + ux * along - uy * side
        y = cy + uy * along + ux * side
        return Keypoint(x=x, y=y, z=None, confidence=0.95)

    lm = {}
    lm["wrist"] = pt(0.0, 0.0)

    # 掌指关节：约在 35% 手长处（实测掌宽/手长≈0.65）
    mcp_along = 0.35
    lm["middle_mcp"] = pt(mcp_along * scale, 0.0)
    lm["index_mcp"] = pt(mcp_along * scale, -0.33 * scale)
    lm["ring_mcp"] = pt(mcp_along * scale, 0.16 * scale)
    lm["pinky_mcp"] = pt(mcp_along * scale, 0.32 * scale)
    # 拇指链起点
    lm["thumb_cmc"] = pt(0.15 * scale, 0.20 * scale)

    # 指尖：中指到 scale，其余按实测比例缩短
    # 实测（指尖到手腕 / 掌宽）：thumb 2.17 index 2.79 middle 2.86 ring 2.63 pinky 2.37
    palm_w = 0.65 * scale
    tips_along = {
        "thumb": 2.17 * palm_w / scale,
        "index": 2.79 * palm_w / scale,
        "middle": 2.86 * palm_w / scale,
        "ring": 2.63 * palm_w / scale,
        "pinky": 2.37 * palm_w / scale,
    }
    tips_side = {"thumb": 0.55, "index": -0.33, "middle": 0.0,
                 "ring": 0.16, "pinky": 0.32}

    for f in FINGER_NAMES:
        chain = FINGER_LANDMARK_NAMES[f]
        along_tip = tips_along[f] * scale
        side_tip = tips_side[f] * scale
        # 中间指节点做线性插值
        mcp = lm[chain[0]]
        tip = pt(along_tip, side_tip)
        lm[chain[-1]] = tip
        for k, name in enumerate(chain[1:-1], start=1):
            t = k / (len(chain) - 1)
            lm[name] = Keypoint(
                x=mcp.x + (tip.x - mcp.x) * t,
                y=mcp.y + (tip.y - mcp.y) * t,
                z=None, confidence=0.95)

    hand = HandKeypoints(
        side="right",
        wrist=Keypoint(cx, cy, z=None, confidence=0.9),
        hand_root=lm["wrist"],
        landmarks=lm,
        palm_center=lm[PALM_CENTER_NAME],
        fingertips={f: lm[FINGERTIP_NAMES[f]] for f in FINGER_NAMES},
        source="synthetic",
    )
    return hand


def make_fist(cx=320.0, cy=240.0, scale=100.0):
    """
    构造一只「握拳的手」：指尖被拉回到靠近手腕处。
    """
    hand = make_open_hand(cx, cy, scale)
    lm = hand.landmarks
    root = lm["wrist"]
    for f in FINGER_NAMES:
        tip_name = FINGERTIP_NAMES[f]
        # 指尖缩回到 ~35% 手长处
        tip = lm[tip_name]
        lm[tip_name] = Keypoint(
            x=root.x + (tip.x - root.x) * 0.35,
            y=root.y + (tip.y - root.y) * 0.35,
            z=None, confidence=0.95)
    hand.fingertips = {f: lm[FINGERTIP_NAMES[f]] for f in FINGER_NAMES}
    return hand


# ============================================================
# 关键点定义
# ============================================================
class TestLandmarkDefinition:
    def test_21_landmarks(self):
        assert len(HAND_LANDMARK_NAMES) == 21

    def test_index_consistency(self):
        for i, n in enumerate(HAND_LANDMARK_NAMES):
            assert HAND_LANDMARK_INDEX[n] == i

    def test_mediapipe_official_order(self):
        """MediaPipe 固定顺序，改动会破坏与模型的对应关系"""
        assert HAND_LANDMARK_NAMES[0] == "wrist"
        assert HAND_LANDMARK_NAMES[4] == "thumb_tip"
        assert HAND_LANDMARK_NAMES[8] == "index_tip"
        assert HAND_LANDMARK_NAMES[12] == "middle_tip"
        assert HAND_LANDMARK_NAMES[16] == "ring_tip"
        assert HAND_LANDMARK_NAMES[20] == "pinky_tip"
        assert HAND_LANDMARK_NAMES[9] == "middle_mcp"

    def test_finger_chains(self):
        for f in FINGER_NAMES:
            chain = FINGER_LANDMARK_NAMES[f]
            assert len(chain) == 4
            assert chain[-1] == FINGERTIP_NAMES[f]
            assert chain[-1].endswith("_tip")

    def test_palm_center_is_middle_mcp(self):
        assert PALM_CENTER_NAME == "middle_mcp"

    def test_connections_reference_valid_names(self):
        valid = set(HAND_LANDMARK_NAMES)
        for a, b in HAND_CONNECTIONS:
            assert a in valid, a
            assert b in valid, b


# ============================================================
# 腕部双来源契约
# ============================================================
class TestWristDualSource:
    def test_has_hand_requires_hand_root(self):
        """只有人体腕部时，has_hand 必须为 False"""
        h = HandKeypoints(side="right",
                          wrist=Keypoint(10, 20, confidence=0.9))
        assert not h.has_hand

    def test_is_complete_requires_both(self):
        """缺少人体腕部 -> 不完整（即使手部模型给了结果）"""
        hand = make_open_hand()
        hand.wrist = None
        assert not hand.is_complete

    def test_complete_synthetic_hand(self):
        hand = make_open_hand()
        assert hand.has_hand
        assert hand.is_complete

    def test_wrist_and_hand_root_are_distinct_fields(self):
        """
        两者必须能取到不同的值 —— 这是本结构存在的前提。
        若实现里把二者当成同一个量，这个测试会失败。
        """
        hand = make_open_hand()
        hand.wrist = Keypoint(100.0, 100.0, confidence=0.9)
        hand.hand_root = Keypoint(105.0, 103.0, confidence=0.9)
        assert hand.wrist.x != hand.hand_root.x

    def test_num_landmarks(self):
        hand = make_open_hand()
        assert hand.num_landmarks == 21

    def test_fingertip_lookup(self):
        hand = make_open_hand()
        assert hand.fingertip("index") is not None
        assert hand.fingertip("nonexistent") is None


# ============================================================
# 几何计算
# ============================================================
class TestHandGeometry:
    def test_returns_none_without_hand(self):
        h = HandKeypoints(side="right", wrist=Keypoint(1, 2, confidence=0.9))
        assert derive_hand_geometry(h) is None

    def test_open_hand_orientation(self):
        """手指朝图像上方 -> 朝向角 +90°（已做 y 翻转）"""
        hand = make_open_hand(angle_deg=90.0)
        g = derive_hand_geometry(hand)
        assert g is not None
        assert g.orientation_deg == pytest.approx(90.0, abs=1.0)

    def test_right_hand_orientation(self):
        hand = make_open_hand(angle_deg=0.0)
        g = derive_hand_geometry(hand)
        assert g.orientation_deg == pytest.approx(0.0, abs=1.0)

    def test_open_hand_has_high_openness(self):
        g = derive_hand_geometry(make_open_hand())
        assert g.openness > 0.9
        assert not g.is_fist

    def test_fist_has_low_openness(self):
        g = derive_hand_geometry(make_fist())
        assert g.openness < 0.5
        assert g.is_fist

    def test_openness_ordering(self):
        """张开的手 openness 必须显著大于握拳"""
        g_open = derive_hand_geometry(make_open_hand())
        g_fist = derive_hand_geometry(make_fist())
        assert g_open.openness > g_fist.openness

    def test_openness_is_scale_invariant(self):
        """
        关键性质：人离相机远近变化时，openness 不应改变。
        这是「用掌宽归一化」而不是用像素绝对值的目的。
        """
        g1 = derive_hand_geometry(make_open_hand(scale=100.0))
        g2 = derive_hand_geometry(make_open_hand(scale=200.0))
        assert g1.openness == pytest.approx(g2.openness, abs=0.02)

    def test_orientation_is_translation_invariant(self):
        g1 = derive_hand_geometry(make_open_hand(cx=100, cy=100))
        g2 = derive_hand_geometry(make_open_hand(cx=500, cy=400))
        assert g1.orientation_deg == pytest.approx(g2.orientation_deg, abs=0.5)

    def test_finger_extension_all_present(self):
        g = derive_hand_geometry(make_open_hand())
        for f in FINGER_NAMES:
            assert f in g.finger_extension
            assert 0.0 <= g.finger_extension[f] <= 1.0

    def test_palm_width_positive(self):
        g = derive_hand_geometry(make_open_hand())
        assert g.palm_width > 0

    def test_fist_fingers_less_extended(self):
        g_open = derive_hand_geometry(make_open_hand())
        g_fist = derive_hand_geometry(make_fist())
        for f in ("index", "middle", "ring", "pinky"):
            assert g_fist.finger_extension[f] < g_open.finger_extension[f]

    def test_missing_fingertip_returns_none(self):
        hand = make_open_hand()
        hand.fingertips.pop("pinky")
        assert derive_hand_geometry(hand) is None


# ============================================================
# YOLO 腕部基线提供者
# ============================================================
class TestYOLOWristBaseline:
    @staticmethod
    def frame():
        return Frame(rgb=np.zeros((480, 640, 3), dtype=np.uint8))

    def test_no_wrist_gives_empty(self):
        p = YOLOWristHandProvider()
        det = p.detect(self.frame(), None)
        assert not det.has_hand
        assert det.hand is None or det.hand.wrist is None

    def test_invalid_wrist_gives_empty(self):
        p = YOLOWristHandProvider()
        det = p.detect(self.frame(), Keypoint(10, 10, confidence=0.0))
        assert not det.has_hand

    def test_valid_wrist_produces_hand_root(self):
        p = YOLOWristHandProvider()
        det = p.detect(self.frame(), Keypoint(100, 200, confidence=0.8))
        assert det.hand is not None
        assert det.hand.hand_root is not None
        assert det.hand.hand_root.x == pytest.approx(100.0)
        assert det.hand.wrist.x == pytest.approx(100.0)

    def test_baseline_has_no_fingertips(self):
        """基线**不应**伪造指尖 —— 没数据就是没数据"""
        p = YOLOWristHandProvider()
        det = p.detect(self.frame(), Keypoint(100, 200, confidence=0.8))
        assert det.hand.fingertips == {}
        assert det.geometry is None
        assert not det.is_complete

    def test_describe_mentions_baseline(self):
        p = YOLOWristHandProvider()
        assert "基线" in p.describe()


# ============================================================
# 级联回退
# ============================================================
class _AlwaysFailProvider(HandProvider):
    """永远检不出手（模拟 MediaPipe 失败）"""

    def detect(self, frame, arm_wrist=None, side="right"):
        return HandDetection(hand=HandKeypoints(side=side,
                                                wrist=arm_wrist,
                                                source="fail"))

    def describe(self):
        return "always-fail"


class _AlwaysSucceedProvider(HandProvider):
    """永远成功（模拟 MediaPipe 正常）"""

    def __init__(self):
        self.hand = make_open_hand()

    def detect(self, frame, arm_wrist=None, side="right"):
        h = make_open_hand()
        h.wrist = arm_wrist
        return HandDetection(hand=h, geometry=derive_hand_geometry(h))

    def describe(self):
        return "always-succeed"


class TestCascading:
    @staticmethod
    def frame():
        return Frame(rgb=np.zeros((480, 640, 3), dtype=np.uint8))

    def test_uses_primary_when_available(self):
        c = CascadingHandProvider(_AlwaysSucceedProvider(),
                                  YOLOWristHandProvider())
        det = c.detect(self.frame(), Keypoint(100, 100, confidence=0.9))
        assert det.has_hand
        assert c.last_source == "primary"

    def test_falls_back_when_primary_fails(self):
        c = CascadingHandProvider(_AlwaysFailProvider(),
                                  YOLOWristHandProvider())
        det = c.detect(self.frame(), Keypoint(100, 100, confidence=0.9))
        # 回退后至少有 hand_root
        assert det.hand is not None
        assert det.hand.hand_root is not None
        assert c.last_source == "fallback"

    def test_fallback_never_loses_wrist(self):
        """
        最重要的保证：只要人体姿态给了腕部，
        即使手部模型完全失败，结果里也必须有腕点 —— 手臂链路不能断。
        """
        c = CascadingHandProvider(_AlwaysFailProvider(),
                                  YOLOWristHandProvider())
        det = c.detect(self.frame(), Keypoint(123, 456, confidence=0.7))
        assert det.hand.wrist is not None
        assert det.hand.wrist.x == pytest.approx(123.0)
        assert det.hand.wrist.y == pytest.approx(456.0)


# ============================================================
# 可视化健壮性：hand_det.hand 可能为 None，不能崩
# ============================================================
class TestVisualizerRobustness:
    """
    回归背景：
        pose_demo 曾在 hand_det.hand 为 None 时崩溃：
            AttributeError: 'NoneType' object has no attribute 'wrist'
        原因是 HUD 里写了 `elif hand_det.hand.wrist is not None`，
        没有先判断 hand_det.hand 是否为 None。
        该场景在「未提供腕部锚点且 MediaPipe 也没检出」时真实出现。
    """

    @staticmethod
    def _pose_det():
        from piper_human_perception.pose import PoseDetection
        return PoseDetection(arm=ArmKeypoints(
            shoulder=Keypoint(10, 10, confidence=0.9),
            elbow=Keypoint(50, 50, confidence=0.9),
            wrist=Keypoint(90, 90, confidence=0.9)), num_persons=1)

    def test_hud_with_none_hand(self):
        from piper_human_perception.visualization import PoseVisualizer
        vis = PoseVisualizer()
        img = np.zeros((120, 160, 3), dtype=np.uint8)
        # 不应抛异常
        vis._draw_hud(img, Frame(rgb=img), self._pose_det(), None,
                      HandDetection(hand=None))

    def test_render_with_none_hand(self):
        from piper_human_perception.visualization import PoseVisualizer
        vis = PoseVisualizer()
        frame = Frame(rgb=np.zeros((120, 160, 3), dtype=np.uint8))
        out = vis.render(frame, self._pose_det(), HandDetection(hand=None))
        assert out is not None and out.shape == (120, 160, 3)

    def test_render_with_hand_det_none(self):
        from piper_human_perception.visualization import PoseVisualizer
        vis = PoseVisualizer()
        frame = Frame(rgb=np.zeros((120, 160, 3), dtype=np.uint8))
        out = vis.render(frame, self._pose_det(), None)
        assert out is not None

    def test_render_with_wrist_only_hand(self):
        """只有腕部基线（hand_root 有效但无指尖）也不能崩"""
        from piper_human_perception.visualization import PoseVisualizer
        from piper_human_control import HandKeypoints
        vis = PoseVisualizer()
        frame = Frame(rgb=np.zeros((120, 160, 3), dtype=np.uint8))
        hd = HandDetection(hand=HandKeypoints(
            side="right",
            wrist=Keypoint(90, 90, confidence=0.8),
            hand_root=Keypoint(92, 92, confidence=0.8)))
        out = vis.render(frame, self._pose_det(), hd)
        assert out is not None
