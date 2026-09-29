# -*- coding: utf-8 -*-
"""
腕部接入测试：**用手部识别的腕部替换人体识别的腕部**
====================================================
本项目的一条核心规则：
    肘部的连接点是**手部模型定位的腕关节**（hand_root），
    而不是人体姿态模型给出的腕部。

本文件专门验证这条规则，防止将来被无意改回。

运行：
    python3 -m pytest test/test_wrist_merge.py -q -p no:anyio
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

from piper_human_control import ArmKeypoints, Keypoint          # noqa: E402
from piper_human_perception.hand import (HandDetection,         # noqa: E402
                                         merge_hand_wrist)
from piper_human_perception.visualization import compute_arm_metrics  # noqa: E402

from test_hand import make_open_hand                            # noqa: E402


# ============================================================
# 构造：人体腕部与手部腕部**故意错开**，便于分辨用的是哪个
# ============================================================
POSE_WRIST = (100.0, 200.0)      # 人体姿态给的腕部
HAND_WRIST = (130.0, 215.0)      # 手部模型给的腕关节（相差约 33 px）


def make_arm(pose_wrist=POSE_WRIST, with_hand_wrist=None):
    s = Keypoint(0.0, 0.0, confidence=0.9, name="right_shoulder")
    e = Keypoint(60.0, 100.0, confidence=0.9, name="right_elbow")
    w = (Keypoint(*pose_wrist, confidence=0.9, name="right_wrist")
         if pose_wrist is not None else None)
    hw = (Keypoint(*with_hand_wrist, confidence=0.95, name="right_wrist_from_hand")
          if with_hand_wrist is not None else None)
    return ArmKeypoints(shoulder=s, elbow=e, wrist=w, side="right", hand_wrist=hw)


def make_hand_det(hand_wrist=HAND_WRIST, valid=True):
    """构造一个手部检测结果，其 hand_root 位于 hand_wrist"""
    hand = make_open_hand()
    if valid:
        root = Keypoint(*hand_wrist, confidence=0.95, name="right_wrist_from_hand")
        hand.hand_root = root
        hand.wrist = Keypoint(*POSE_WRIST, confidence=0.9)
    else:
        hand.hand_root = None
    return HandDetection(hand=hand)


# ============================================================
# effective_wrist 的优先级
# ============================================================
class TestEffectiveWrist:
    def test_prefers_hand_wrist(self):
        """有手部腕部时，effective_wrist 必须是手部腕部"""
        arm = make_arm(with_hand_wrist=HAND_WRIST)
        assert arm.effective_wrist.x == pytest.approx(HAND_WRIST[0])
        assert arm.effective_wrist.y == pytest.approx(HAND_WRIST[1])

    def test_falls_back_to_pose_wrist(self):
        """没有手部腕部时，回退到人体腕部（链路不能断）"""
        arm = make_arm()
        assert arm.effective_wrist.x == pytest.approx(POSE_WRIST[0])

    def test_pose_wrist_is_not_discarded_in_field(self):
        """
        人体腕部字段本身**保留**（它是 ROI 锚点与兜底），
        只是不再作为 effective_wrist —— 两者必须能取到不同的值。
        """
        arm = make_arm(with_hand_wrist=HAND_WRIST)
        assert arm.wrist.x == pytest.approx(POSE_WRIST[0])
        assert arm.effective_wrist.x == pytest.approx(HAND_WRIST[0])
        assert arm.wrist.x != arm.effective_wrist.x

    def test_wrist_source_hand(self):
        arm = make_arm(with_hand_wrist=HAND_WRIST)
        assert arm.wrist_source == "hand"

    def test_wrist_source_pose(self):
        arm = make_arm()
        assert arm.wrist_source == "pose"

    def test_wrist_source_none(self):
        arm = make_arm(pose_wrist=None)
        assert arm.wrist_source == "none"
        assert arm.effective_wrist is None

    def test_invalid_hand_wrist_is_ignored(self):
        """手部腕点 confidence=0 视为无效，必须回退"""
        arm = make_arm()
        bad = Keypoint(HAND_WRIST[0], HAND_WRIST[1], confidence=0.0)
        arm = arm.with_hand_wrist(bad)
        assert arm.wrist_source == "pose"
        assert arm.effective_wrist.x == pytest.approx(POSE_WRIST[0])


# ============================================================
# merge_hand_wrist
# ============================================================
class TestMergeHandWrist:
    def test_merges_when_hand_valid(self):
        arm = make_arm()
        merged = merge_hand_wrist(arm, make_hand_det())
        assert merged.wrist_source == "hand"
        assert merged.effective_wrist.x == pytest.approx(HAND_WRIST[0])

    def test_returns_new_object_not_mutating(self):
        """不可变式返回：原对象不应被改写"""
        arm = make_arm()
        merged = merge_hand_wrist(arm, make_hand_det())
        assert arm.hand_wrist is None
        assert arm.wrist_source == "pose"
        assert merged is not arm

    def test_hand_det_none_keeps_pose(self):
        arm = make_arm()
        merged = merge_hand_wrist(arm, None)
        assert merged is arm
        assert merged.wrist_source == "pose"

    def test_arm_none_returns_none(self):
        assert merge_hand_wrist(None, make_hand_det()) is None

    def test_invalid_hand_root_keeps_pose(self):
        arm = make_arm()
        merged = merge_hand_wrist(arm, make_hand_det(valid=False))
        assert merged.wrist_source == "pose"

    def test_shoulder_elbow_preserved(self):
        """替换腕部不应影响肩/肘"""
        arm = make_arm()
        merged = merge_hand_wrist(arm, make_hand_det())
        assert merged.shoulder.x == pytest.approx(arm.shoulder.x)
        assert merged.elbow.x == pytest.approx(arm.elbow.x)


# ============================================================
# 几何计算必须使用手部腕部
# ============================================================
class TestGeometryUsesHandWrist:
    def test_forearm_length_changes_with_hand_wrist(self):
        """
        前臂长度（肘->腕）必须按**手部腕部**计算。
        若实现误用人体腕部，两者算出的长度会不同 —— 本测试据此判别。
        """
        arm_pose = make_arm()
        arm_hand = make_arm(with_hand_wrist=HAND_WRIST)

        m_pose = compute_arm_metrics(arm_pose)
        m_hand = compute_arm_metrics(arm_hand)

        len_pose = float(m_pose["前臂长"].rstrip("px"))
        len_hand = float(m_hand["前臂长"].rstrip("px"))

        assert len_pose != pytest.approx(len_hand, abs=1.0)

        # 手部腕部的期望值：肘(60,100) -> 手部腕(130,215)
        expect = math.hypot(HAND_WRIST[0] - 60.0, HAND_WRIST[1] - 100.0)
        assert len_hand == pytest.approx(expect, abs=0.5)

    def test_elbow_angle_uses_hand_wrist(self):
        """肘夹角也必须基于手部腕部"""
        arm_pose = make_arm()
        arm_hand = make_arm(with_hand_wrist=HAND_WRIST)
        a_pose = float(compute_arm_metrics(arm_pose)["肘夹角"].rstrip("°"))
        a_hand = float(compute_arm_metrics(arm_hand)["肘夹角"].rstrip("°"))
        assert a_pose != pytest.approx(a_hand, abs=0.5)

    def test_as_list_uses_effective_wrist(self):
        arm = make_arm(with_hand_wrist=HAND_WRIST)
        lst = arm.as_list()
        assert len(lst) == 3
        assert lst[2].x == pytest.approx(HAND_WRIST[0])

    def test_as_list_falls_back(self):
        arm = make_arm()
        lst = arm.as_list()
        assert lst[2].x == pytest.approx(POSE_WRIST[0])


# ============================================================
# is_complete 语义：描述的是人体姿态链路，不是手部
# ============================================================
class TestIsCompleteSemantics:
    def test_is_complete_ignores_hand_wrist(self):
        """
        is_complete 描述「人体姿态链路是否完整」。
        只有肩+肘（无人体腕部）时应为 False，即使手部腕部存在。
        """
        arm = make_arm(pose_wrist=None, with_hand_wrist=HAND_WRIST)
        assert not arm.is_complete
        assert arm.effective_wrist is not None

    def test_is_complete_true_with_pose_wrist(self):
        arm = make_arm()
        assert arm.is_complete
