# -*- coding: utf-8 -*-
"""
piper_human_perception
======================
人体手臂姿态感知层（阶段三）。

职责：
    RGB 摄像头 / 视频 / 图片
        -> FrameSource        （帧源抽象，阶段七可换 RGB-D）
        -> PoseProvider       （人体姿态：肩 / 肘 / 腕）
        -> HandProvider       （手部：手根 / 掌心 / 五指指尖）
        -> PoseVisualizer     （实时可视化）

与其它模块的边界（保持解耦）：
    * 本包 **不依赖 ROS2**，也不依赖 piper_human_control 的控制逻辑，
      只复用其数据结构（Keypoint / ArmKeypoints / HandKeypoints），
      保证全项目数据模型统一。
    * 本包 **不做** 人体->关节映射（阶段四），也不做机械臂控制（阶段二）。
    * 阶段四是本包的唯一消费者，接口是 PoseDetection.arm 与 HandDetection.hand。

腕部的两个来源（**重要，勿混用**）：
    ArmKeypoints.wrist    人体姿态模型的腕部 —— 用于手臂链路（阶段四 J2/J3/J5）
    HandKeypoints.hand_root 手部模型的掌心根 —— 用于手部几何（阶段八 J6 / 夹爪）
    详见 piper_human_control.types.HandKeypoints 的文档。

阶段七扩展方式（无需改动本包已有代码）：
    新增 RGBDPoseProvider / 在 HandProvider 中填 Keypoint.z 即可。
"""

from .perception import (CameraIntrinsics, Frame, FrameSource,
                         ImageFileSource, OpenCVCameraSource,
                         RealSenseSource, VideoFileSource)
from .pose import (ARM_KEYPOINT_NAMES, COCO_KEYPOINT_NAMES, KP_INDEX,
                   PoseDetection, PoseProvider, RGBDPoseProvider,
                   YOLOPoseProvider)
from .hand import (FINGER_LANDMARK_NAMES, FINGERTIP_NAMES,
                   HAND_CONNECTIONS, HAND_LANDMARK_INDEX,
                   HAND_LANDMARK_NAMES, PALM_CENTER_NAME,
                   CascadingHandProvider, HandDetection, HandGeometry,
                   HandProvider, MediaPipeHandProvider,
                   YOLOWristHandProvider, compute_hand_pose,
                   derive_hand_geometry,
                   make_hand_provider, merge_hand_wrist)
from .visualization import PoseVisualizer, compute_arm_metrics

__version__ = "0.2.0"

__all__ = [
    # 帧源
    "Frame",
    "FrameSource",
    "OpenCVCameraSource",
    "VideoFileSource",
    "ImageFileSource",
    "RealSenseSource",          # 阶段七预留
    "CameraIntrinsics",
    # 人体姿态
    "PoseProvider",
    "YOLOPoseProvider",
    "RGBDPoseProvider",         # 阶段七预留
    "PoseDetection",
    "COCO_KEYPOINT_NAMES",
    "KP_INDEX",
    "ARM_KEYPOINT_NAMES",
    # 手部
    "HandProvider",
    "MediaPipeHandProvider",
    "YOLOWristHandProvider",
    "CascadingHandProvider",
    "HandDetection",
    "HandGeometry",
    "HandPose",
    "compute_hand_pose",
    "derive_hand_geometry",
    "make_hand_provider",
    "merge_hand_wrist",
    "HAND_LANDMARK_NAMES",
    "HAND_LANDMARK_INDEX",
    "FINGER_LANDMARK_NAMES",
    "FINGERTIP_NAMES",
    "HAND_CONNECTIONS",
    "PALM_CENTER_NAME",
    # 可视化
    "PoseVisualizer",
    "compute_arm_metrics",
]
