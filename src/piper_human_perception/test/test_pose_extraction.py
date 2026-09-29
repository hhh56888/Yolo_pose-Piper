# -*- coding: utf-8 -*-
"""
YOLOPoseProvider 关键点提取逻辑测试（用 Mock 替代真实模型）
==========================================================
为什么需要这组测试：
    真实模型的行为不可控（取决于画面里有没有人、手臂是否入镜），
    因此「提取逻辑」本身（低置信度过滤、选人策略、坐标映射、
    手臂装配）必须用可控输入单独验证，否则一旦出问题很难定位
    是模型的锅还是代码的锅。

这里直接替换 provider._model，用假的 predict 返回值驱动，
不加载任何模型、不需要 GPU、不需要摄像头。
"""

import os
import sys

import numpy as np
import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_HERE)
_SRC_ROOT = os.path.dirname(_PKG_ROOT)
for p in (_PKG_ROOT, os.path.join(_SRC_ROOT, "piper_human_control")):
    if p not in sys.path:
        sys.path.insert(0, p)

from piper_human_perception.perception import Frame                # noqa: E402
from piper_human_perception.pose import (ARM_KEYPOINT_NAMES,       # noqa: E402
                                         COCO_KEYPOINT_NAMES, KP_INDEX,
                                         YOLOPoseProvider)


# ============================================================
# 构造假的 ultralytics 返回结构
# ============================================================
class FakeBoxes:
    """模拟 ultralytics 的 Boxes 对象"""

    def __init__(self, xyxy, conf):
        self.xyxy = torch.tensor(xyxy, dtype=torch.float32)
        self.conf = torch.tensor(conf, dtype=torch.float32)


class FakeKeypoints:
    """模拟 ultralytics 的 KeyPoints 对象（只有 .data）"""

    def __init__(self, data):
        # data: (N, 17, 3)
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return self.data.shape[0]


class FakeResult:
    def __init__(self, keypoints=None, boxes=None):
        self.keypoints = keypoints
        self.boxes = boxes


class FakeModel:
    """假的 YOLO 模型：直接返回预设结果"""

    def __init__(self, result):
        self._result = result
        self.task = "pose"

    def predict(self, *args, **kwargs):
        return [self._result]


def make_frame():
    return Frame(rgb=np.zeros((480, 640, 3), dtype=np.uint8))


def kp_array(overrides=None, default_conf=0.9):
    """
    构造一个人的 17 点数据。

    overrides: {名称: (x, y, conf)}，未指定的用默认值。
    """
    data = np.zeros((17, 3), dtype=np.float32)
    for i in range(17):
        data[i] = [100.0 + i, 200.0 + i, default_conf]
    if overrides:
        for name, (x, y, c) in overrides.items():
            data[KP_INDEX[name]] = [x, y, c]
    return data


def provider_with(result, **kwargs):
    """创建一个 provider，并把模型替换成假的"""
    p = YOLOPoseProvider(**kwargs)
    p._model = FakeModel(result)
    return p


# ============================================================
# 无检测
# ============================================================
class TestNoDetection:
    def test_no_keypoints_attribute(self):
        p = provider_with(FakeResult(keypoints=None))
        det = p.detect(make_frame())
        assert det.num_persons == 0
        assert not det.has_person
        assert det.arm is None

    def test_empty_keypoints(self):
        p = provider_with(FakeResult(keypoints=FakeKeypoints(
            np.zeros((0, 17, 3), dtype=np.float32))))
        det = p.detect(make_frame())
        assert det.num_persons == 0


# ============================================================
# 关键点提取与过滤
# ============================================================
class TestKeypointExtraction:
    def test_extracts_all_17(self):
        p = provider_with(FakeResult(keypoints=FakeKeypoints(
            kp_array()[None]), boxes=FakeBoxes([[0, 0, 200, 400]], [0.9])))
        det = p.detect(make_frame())
        assert len(det.all_keypoints) == 17
        for n in COCO_KEYPOINT_NAMES:
            assert n in det.all_keypoints

    def test_coordinates_mapped_correctly(self):
        p = provider_with(FakeResult(keypoints=FakeKeypoints(
            kp_array({"right_shoulder": (11.0, 22.0, 0.9),
                      "right_elbow": (33.0, 44.0, 0.8),
                      "right_wrist": (55.0, 66.0, 0.7)})[None]),
            boxes=FakeBoxes([[0, 0, 200, 400]], [0.9])))
        det = p.detect(make_frame())
        assert det.arm.shoulder.x == pytest.approx(11.0)
        assert det.arm.elbow.x == pytest.approx(33.0)
        assert det.arm.wrist.x == pytest.approx(55.0)
        assert det.arm.wrist.confidence == pytest.approx(0.7, abs=1e-5)

    def test_low_confidence_zeroed(self):
        """低于门限的点必须被置为 confidence=0（而不是保留垃圾坐标）"""
        p = provider_with(FakeResult(keypoints=FakeKeypoints(
            kp_array({"right_elbow": (0.0, 480.0, 0.011)})[None]),
            boxes=FakeBoxes([[0, 0, 200, 400]], [0.9])), conf_threshold=0.5)
        det = p.detect(make_frame())
        assert det.all_keypoints["right_elbow"].confidence == 0.0
        assert not det.all_keypoints["right_elbow"].is_valid

    def test_threshold_boundary(self):
        """刚好等于门限的点应视为有效"""
        p = provider_with(FakeResult(keypoints=FakeKeypoints(
            kp_array({"right_elbow": (10.0, 20.0, 0.5)})[None]),
            boxes=FakeBoxes([[0, 0, 200, 400]], [0.9])), conf_threshold=0.5)
        det = p.detect(make_frame())
        assert det.all_keypoints["right_elbow"].confidence == pytest.approx(0.5)

    def test_z_always_none_in_stage3(self):
        """阶段三契约：所有关键点 z 必须为 None"""
        p = provider_with(FakeResult(keypoints=FakeKeypoints(
            kp_array()[None]), boxes=FakeBoxes([[0, 0, 200, 400]], [0.9])))
        det = p.detect(make_frame())
        for kp in det.all_keypoints.values():
            assert kp.z is None


# ============================================================
# 手臂装配
# ============================================================
class TestArmAssembly:
    def test_right_arm_uses_right_keypoints(self):
        p = provider_with(FakeResult(keypoints=FakeKeypoints(
            kp_array({"right_shoulder": (10, 10, 0.9),
                      "left_shoulder": (99, 99, 0.9)})[None]),
            boxes=FakeBoxes([[0, 0, 200, 400]], [0.9])), target_side="right")
        det = p.detect(make_frame())
        assert det.arm.side == "right"
        assert det.arm.shoulder.x == pytest.approx(10.0)

    def test_left_arm_uses_left_keypoints(self):
        p = provider_with(FakeResult(keypoints=FakeKeypoints(
            kp_array({"right_shoulder": (10, 10, 0.9),
                      "left_shoulder": (99, 99, 0.9)})[None]),
            boxes=FakeBoxes([[0, 0, 200, 400]], [0.9])), target_side="left")
        det = p.detect(make_frame())
        assert det.arm.side == "left"
        assert det.arm.shoulder.x == pytest.approx(99.0)

    def test_arm_complete_when_all_confident(self):
        p = provider_with(FakeResult(keypoints=FakeKeypoints(
            kp_array()[None]), boxes=FakeBoxes([[0, 0, 200, 400]], [0.9])))
        det = p.detect(make_frame())
        assert det.arm_complete

    def test_arm_incomplete_when_wrist_low_conf(self):
        """腕部低置信度 -> 手臂不完整（阶段四据此跳过该帧）"""
        p = provider_with(FakeResult(keypoints=FakeKeypoints(
            kp_array({"right_wrist": (5, 5, 0.02)})[None]),
            boxes=FakeBoxes([[0, 0, 200, 400]], [0.9])))
        det = p.detect(make_frame())
        assert not det.arm_complete

    def test_invalid_target_side_rejected(self):
        with pytest.raises(ValueError):
            YOLOPoseProvider(target_side="middle")


# ============================================================
# 多人选人策略
# ============================================================
class TestPersonSelection:
    def test_picks_largest_bbox_not_first(self):
        """
        关键回归：第二个人框更大，应选中第二个。
        若按「取第一个」实现，多人场景关键点会在人之间跳变。
        """
        d1 = kp_array({"right_shoulder": (10, 10, 0.9)})     # 小框的人
        d2 = kp_array({"right_shoulder": (500, 500, 0.9)})   # 大框的人
        data = np.stack([d1, d2])
        boxes = FakeBoxes([[0, 0, 50, 50],       # 面积 2500
                           [0, 0, 400, 400]],    # 面积 160000
                          [0.8, 0.9])
        p = provider_with(FakeResult(keypoints=FakeKeypoints(data), boxes=boxes))
        det = p.detect(make_frame())
        assert det.num_persons == 2
        assert det.arm.shoulder.x == pytest.approx(500.0)

    def test_single_person_no_boxes(self):
        """只有一个人且没有 boxes 时不应崩溃"""
        p = provider_with(FakeResult(keypoints=FakeKeypoints(
            kp_array()[None]), boxes=None))
        det = p.detect(make_frame())
        assert det.num_persons == 1
        assert det.arm_complete

    def test_box_count_mismatch_falls_back_to_first(self):
        """boxes 数量与关键点数不一致时安全回退，不抛异常"""
        data = np.stack([kp_array(), kp_array()])
        boxes = FakeBoxes([[0, 0, 10, 10]], [0.9])          # 只有 1 个框
        p = provider_with(FakeResult(keypoints=FakeKeypoints(data), boxes=boxes))
        det = p.detect(make_frame())
        assert det.num_persons == 2
        assert det.arm is not None

    def test_bbox_and_score_reported(self):
        p = provider_with(FakeResult(keypoints=FakeKeypoints(
            kp_array()[None]), boxes=FakeBoxes([[1, 2, 3, 4]], [0.77])))
        det = p.detect(make_frame())
        assert det.bbox == (1.0, 2.0, 3.0, 4.0)
        assert det.score == pytest.approx(0.77, abs=1e-5)


# ============================================================
# 延迟统计
# ============================================================
class TestLatency:
    def test_latency_measured(self):
        p = provider_with(FakeResult(keypoints=FakeKeypoints(
            kp_array()[None]), boxes=FakeBoxes([[0, 0, 9, 9]], [0.9])))
        det = p.detect(make_frame())
        assert det.latency_ms > 0.0
