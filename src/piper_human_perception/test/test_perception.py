# -*- coding: utf-8 -*-
"""
感知层单元测试（不依赖摄像头与 ROS2）
=====================================
覆盖：
    * Frame 的深度查询（阶段七的接口约定）
    * CameraIntrinsics 反投影
    * 手臂几何量计算
    * 数据结构的 z=None 约定
    * 阶段七占位类确实未实现（避免误用）
    * 关键点索引定义与 COCO-17 顺序一致

运行：
    python3 -m pytest test/test_perception.py -q -p no:anyio
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

from piper_human_control import ArmKeypoints, Keypoint            # noqa: E402
from piper_human_perception.perception import (CameraIntrinsics,  # noqa: E402
                                               Frame, RealSenseSource)
from piper_human_perception.pose import (COCO_KEYPOINT_NAMES,     # noqa: E402
                                         ARM_KEYPOINT_NAMES, KP_INDEX,
                                         RGBDPoseProvider)
from piper_human_perception.visualization import compute_arm_metrics  # noqa: E402


def make_frame(depth=None):
    return Frame(rgb=np.zeros((480, 640, 3), dtype=np.uint8), depth=depth)


# ============================================================
# 关键点定义
# ============================================================
class TestKeypointDefinition:
    def test_coco_has_17_points(self):
        assert len(COCO_KEYPOINT_NAMES) == 17

    def test_index_matches_order(self):
        for i, n in enumerate(COCO_KEYPOINT_NAMES):
            assert KP_INDEX[n] == i

    def test_coco_official_order(self):
        """COCO-17 官方顺序不能改（YOLO 输出依赖它）"""
        assert COCO_KEYPOINT_NAMES[0] == "nose"
        assert COCO_KEYPOINT_NAMES[5] == "left_shoulder"
        assert COCO_KEYPOINT_NAMES[6] == "right_shoulder"
        assert COCO_KEYPOINT_NAMES[7] == "left_elbow"
        assert COCO_KEYPOINT_NAMES[8] == "right_elbow"
        assert COCO_KEYPOINT_NAMES[9] == "left_wrist"
        assert COCO_KEYPOINT_NAMES[10] == "right_wrist"

    def test_arm_mapping(self):
        assert ARM_KEYPOINT_NAMES["right"]["shoulder"] == "right_shoulder"
        assert ARM_KEYPOINT_NAMES["right"]["wrist"] == "right_wrist"
        assert ARM_KEYPOINT_NAMES["left"]["elbow"] == "left_elbow"


# ============================================================
# Frame / 深度
# ============================================================
class TestFrame:
    def test_rgb_only_has_no_depth(self):
        """阶段三约定：depth 必须为 None"""
        f = make_frame()
        assert not f.has_depth()
        assert f.depth_at(100, 100) is None

    def test_dimensions(self):
        f = make_frame()
        assert f.width == 640 and f.height == 480

    def test_depth_query_median(self):
        """小窗口中值采样：应忽略空洞（0）"""
        depth = np.zeros((480, 640), dtype=np.float32)
        depth[98:103, 98:103] = 1.5
        depth[100, 100] = 0.0            # 人为制造空洞
        f = make_frame(depth=depth)
        assert f.has_depth()
        assert f.depth_at(100, 100) == pytest.approx(1.5, abs=1e-6)

    def test_depth_query_out_of_bounds(self):
        depth = np.ones((480, 640), dtype=np.float32)
        f = make_frame(depth=depth)
        assert f.depth_at(-10, 100) is None
        assert f.depth_at(100, 9999) is None

    def test_depth_query_all_invalid(self):
        depth = np.zeros((480, 640), dtype=np.float32)
        f = make_frame(depth=depth)
        assert f.depth_at(100, 100) is None

    def test_depth_query_ignores_nan(self):
        depth = np.full((480, 640), np.nan, dtype=np.float32)
        depth[99:102, 99:102] = 2.0
        f = make_frame(depth=depth)
        assert f.depth_at(100, 100) == pytest.approx(2.0, abs=1e-6)


# ============================================================
# 相机内参
# ============================================================
class TestCameraIntrinsics:
    def test_valid(self):
        K = CameraIntrinsics(fx=600, fy=600, cx=320, cy=240)
        assert K.is_valid()

    def test_invalid(self):
        assert not CameraIntrinsics(fx=0, fy=600, cx=320, cy=240).is_valid()

    def test_backproject_center_is_on_axis(self):
        """主点反投影应落在光轴上 (X=Y=0, Z=depth)"""
        K = CameraIntrinsics(fx=600, fy=600, cx=320, cy=240)
        X, Y, Z = K.backproject(320, 240, 1.0)
        assert X == pytest.approx(0.0, abs=1e-9)
        assert Y == pytest.approx(0.0, abs=1e-9)
        assert Z == pytest.approx(1.0)

    def test_backproject_known_value(self):
        """u 偏 cx 600 像素、fx=600、depth=2m -> X=2m"""
        K = CameraIntrinsics(fx=600, fy=600, cx=320, cy=240)
        X, Y, Z = K.backproject(920, 240, 2.0)
        assert X == pytest.approx(2.0, abs=1e-9)
        assert Y == pytest.approx(0.0, abs=1e-9)

    def test_backproject_rejects_bad_depth(self):
        K = CameraIntrinsics(fx=600, fy=600, cx=320, cy=240)
        assert K.backproject(320, 240, None) is None
        assert K.backproject(320, 240, 0.0) is None
        assert K.backproject(320, 240, -1.0) is None


# ============================================================
# 手臂几何
# ============================================================
class TestArmMetrics:
    @staticmethod
    def arm(sx, sy, ex, ey, wx, wy):
        return ArmKeypoints(
            shoulder=Keypoint(sx, sy, confidence=0.9, name="right_shoulder"),
            elbow=Keypoint(ex, ey, confidence=0.9, name="right_elbow"),
            wrist=Keypoint(wx, wy, confidence=0.9, name="right_wrist"),
            side="right")

    def test_incomplete_returns_empty(self):
        ak = ArmKeypoints(shoulder=Keypoint(1, 2, confidence=0.9))
        assert compute_arm_metrics(ak) == {}

    def test_straight_arm_elbow_angle_is_180(self):
        """肘完全伸直 -> 夹角 180°"""
        # S(0,0) E(100,0) W(200,0)
        m = compute_arm_metrics(self.arm(0, 0, 100, 0, 200, 0))
        ang = float(m["肘夹角"].rstrip("°"))
        assert ang == pytest.approx(180.0, abs=0.5)

    def test_right_angle_elbow_angle_is_90(self):
        """肘弯 90° -> 夹角 90°"""
        # S(0,0) E(100,0) W(100,100)
        m = compute_arm_metrics(self.arm(0, 0, 100, 0, 100, 100))
        ang = float(m["肘夹角"].rstrip("°"))
        assert ang == pytest.approx(90.0, abs=0.5)

    def test_upper_arm_angle_upward_positive(self):
        """上臂指向画面上方 -> 角度为正（已做 y 翻转）"""
        # S(0,100) E(0,0)：在图像坐标里 y 减小 = 向上
        m = compute_arm_metrics(self.arm(0, 100, 0, 0, 0, 0))
        ang = float(m["上臂角"].rstrip("°"))
        assert ang == pytest.approx(90.0, abs=0.5)

    def test_upper_arm_angle_rightward_zero(self):
        m = compute_arm_metrics(self.arm(0, 0, 100, 0, 200, 0))
        ang = float(m["上臂角"].rstrip("°"))
        assert ang == pytest.approx(0.0, abs=0.5)

    def test_lengths_reported(self):
        m = compute_arm_metrics(self.arm(0, 0, 100, 0, 200, 0))
        assert float(m["上臂长"].rstrip("px")) == pytest.approx(100.0, abs=0.5)
        assert float(m["前臂长"].rstrip("px")) == pytest.approx(100.0, abs=0.5)


# ============================================================
# 阶段七占位
# ============================================================
class TestStage7Placeholders:
    def test_realsense_not_implemented(self):
        with pytest.raises(NotImplementedError):
            RealSenseSource()

    def test_rgbd_provider_not_implemented(self):
        with pytest.raises(NotImplementedError):
            RGBDPoseProvider()

    def test_placeholder_message_mentions_stage7(self):
        try:
            RGBDPoseProvider()
        except NotImplementedError as exc:
            assert "阶段七" in str(exc)


# ============================================================
# 数据结构约定（跨阶段契约）
# ============================================================
class TestDataContract:
    def test_keypoint_z_none_in_stage3(self):
        """阶段三产出的关键点 z 必须为 None"""
        kp = Keypoint(x=100.0, y=200.0, z=None, confidence=0.9)
        assert kp.z is None
        assert not kp.has_depth()

    def test_arm_keypoints_completeness_uses_confidence(self):
        """conf=0 的点视为无效，因此手臂判定为不完整"""
        ak = ArmKeypoints(
            shoulder=Keypoint(1, 2, confidence=0.9),
            elbow=Keypoint(3, 4, confidence=0.0),   # 无效
            wrist=Keypoint(5, 6, confidence=0.9))
        assert not ak.is_complete

    def test_arm_keypoints_complete_when_all_valid(self):
        ak = ArmKeypoints(
            shoulder=Keypoint(1, 2, confidence=0.9),
            elbow=Keypoint(3, 4, confidence=0.8),
            wrist=Keypoint(5, 6, confidence=0.7))
        assert ak.is_complete
