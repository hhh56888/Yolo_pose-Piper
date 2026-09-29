# -*- coding: utf-8 -*-
"""
piper_human_perception.pose
===========================
人体姿态提取层。

职责边界（保持解耦）：
    本层只负责「从一帧图像里把肩/肘/腕找出来」，
    不做手臂几何计算（阶段四），不做机械臂映射，不碰 ROS2。

核心抽象 PoseProvider：
    上层只依赖这个接口。阶段三提供 YOLOPoseProvider（RGB，z=None）；
    阶段七新增 RGBDPoseProvider（RGB-D，z=真实深度）即可，
    **不需要改本文件的其他任何代码**，也不需要改上层。
    这正是任务书「不要修改原来的 YOLO 2D 核心逻辑」的落地方式。
"""

from abc import ABC, abstractmethod
import time
from typing import Dict, List, Optional

import numpy as np

from piper_human_control import ArmKeypoints, Keypoint

from .perception import CameraIntrinsics, Frame


# ============================================================
# COCO-17 关键点定义（YOLO Pose 输出顺序）
# ============================================================
# 顺序是 ultralytics YOLO pose 模型的固定输出顺序，不可更改。
COCO_KEYPOINT_NAMES: List[str] = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]

# 名称 -> 索引
KP_INDEX: Dict[str, int] = {n: i for i, n in enumerate(COCO_KEYPOINT_NAMES)}

# 每侧手臂需要的三个关键点
ARM_KEYPOINT_NAMES: Dict[str, Dict[str, str]] = {
    "right": {
        "shoulder": "right_shoulder",
        "elbow": "right_elbow",
        "wrist": "right_wrist",
    },
    "left": {
        "shoulder": "left_shoulder",
        "elbow": "left_elbow",
        "wrist": "left_wrist",
    },
}


# ============================================================
# 检测结果
# ============================================================
class PoseDetection:
    """
    单帧姿态检测结果。

    Attributes:
        arm:        目标侧手臂的三个关键点
        all_keypoints: 17 个关键点全量（便于调试与可视化）
        bbox:       人体检测框 (x1, y1, x2, y2)；无检测时为 None
        score:      人体检测置信度
        num_persons: 画面中检测到的人数
        latency_ms: 本帧推理耗时
    """

    __slots__ = ("arm", "all_keypoints", "bbox", "score",
                 "num_persons", "latency_ms")

    def __init__(self, arm: Optional[ArmKeypoints] = None,
                 all_keypoints: Optional[Dict[str, Keypoint]] = None,
                 bbox=None, score: float = 0.0,
                 num_persons: int = 0, latency_ms: float = 0.0):
        self.arm = arm
        self.all_keypoints = all_keypoints or {}
        self.bbox = bbox
        self.score = score
        self.num_persons = num_persons
        self.latency_ms = latency_ms

    @property
    def has_person(self) -> bool:
        return self.num_persons > 0

    @property
    def arm_complete(self) -> bool:
        return self.arm is not None and self.arm.is_complete


# ============================================================
# 抽象接口
# ============================================================
class PoseProvider(ABC):
    """
    姿态提供者抽象接口。

    阶段三：YOLOPoseProvider（RGB，z=None）
    阶段七：RGBDPoseProvider（RGB-D，z=真实深度）
    两者对本接口的实现完全一致，因此上层代码零改动。
    """

    @abstractmethod
    def detect(self, frame: Frame) -> PoseDetection:
        """从一帧图像中提取姿态"""

    @abstractmethod
    def describe(self) -> str:
        """返回提供者描述"""

    def close(self) -> None:
        """释放资源（默认无需释放）"""


# ============================================================
# YOLO Pose 实现（阶段三）
# ============================================================
class YOLOPoseProvider(PoseProvider):
    """
    基于 ultralytics YOLO Pose 的姿态提取（2D）。

    关键参数：
        conf_threshold: 关键点置信度门限。低于此值的点会被标记为无效
                        （confidence=0），而不是把「坐标垃圾」当真值传下去。
                        实测 YOLO 对画面外/遮挡关节会输出 (0,480) 这类
                        边界坐标且 conf 极低，必须过滤。
        person_conf:    人体检测置信度门限
        target_side:    "right" / "left"，当前项目优先右手
        device:         "0" 用 GPU，"" / "cpu" 用 CPU

    关于「选哪个人」：
        多人场景下取检测框面积最大者（通常离相机最近、最清晰），
        而不是取第一个 —— YOLO 输出顺序不保证稳定，
        取第一个会导致关键点在多人之间来回跳。
    """

    def __init__(self,
                 model_path: str = "yolo11n-pose.pt",
                 conf_threshold: float = 0.4,
                 person_conf: float = 0.5,
                 target_side: str = "right",
                 device: str = "0",
                 imgsz: int = 960):
        """
        参数默认值来自实测（见阶段三报告 §4）：
            imgsz=960 + conf=0.4 在手臂近景视频上取得最佳平衡
            （三点齐全率 44.3%，肘夹角均值 87.2°，几何一致性好）。

            实测对比（431 帧手臂近景视频，"三点齐全率"）：
                imgsz=640 conf=0.5 ->  8.6%
                imgsz=640 conf=0.3 -> 38.7%
                imgsz=960 conf=0.5 -> 31.6%
                imgsz=960 conf=0.4 -> 44.3%   ← 采用
                imgsz=960 conf=0.3 -> 54.3%   （齐全率更高，但离群点更多）
                imgsz=1280 conf=0.3 -> 43.6%  （1280 反而变差）

            降 conf 能提高齐全率，但低置信度点会带来更大的偶发跳变
            （conf=0.3 时肘夹角最大跳变达 111°），
            因此取 0.4 更稳；后续阶段五的 EMA 滤波可以再放宽到 0.3。
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.person_conf = person_conf
        self.target_side = target_side
        self.device = device
        self.imgsz = imgsz

        if target_side not in ARM_KEYPOINT_NAMES:
            raise ValueError(f"target_side 必须是 'left' 或 'right'，收到 {target_side}")

        self._model = None

    # ------------------------------------------------------------
    def load(self) -> None:
        """加载模型（惰性，便于测试时不加载）"""
        if self._model is not None:
            return
        from ultralytics import YOLO
        self._model = YOLO(self.model_path)

    def describe(self) -> str:
        return (f"YOLO Pose (模型={self.model_path}, device={self.device}, "
                f"kp_conf>={self.conf_threshold}, person_conf>={self.person_conf}, "
                f"目标侧={self.target_side})")

    # ------------------------------------------------------------
    def detect(self, frame: Frame) -> PoseDetection:
        """对一帧做姿态检测"""
        self.load()
        t0 = time.perf_counter()

        results = self._model.predict(
            frame.rgb,
            device=self.device,
            imgsz=self.imgsz,
            conf=self.person_conf,
            verbose=False,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        result = results[0]
        kps = getattr(result, "keypoints", None)

        if kps is None or len(kps) == 0:
            return PoseDetection(num_persons=0, latency_ms=latency_ms)

        num_persons = len(kps)

        # 注意：人体框在 result.boxes 上，不在 kps 上。
        # （ultralytics 的 KeyPoints 对象没有 .boxes 属性）
        boxes = getattr(result, "boxes", None)

        # ---- 选人：取包围盒面积最大者 ----
        idx = self._select_person(kps, boxes)
        if idx is None:
            return PoseDetection(num_persons=num_persons, latency_ms=latency_ms)

        # kps.data 形状 (N, 17, 3) -> [x, y, conf]
        data = kps.data[idx].cpu().numpy()
        bbox = self._bbox_of(boxes, idx)
        score = self._score_of(boxes, idx)

        # ---- 构造 17 个关键点 ----
        all_kp: Dict[str, Keypoint] = {}
        for i, name in enumerate(COCO_KEYPOINT_NAMES):
            x, y, c = float(data[i][0]), float(data[i][1]), float(data[i][2])
            if c < self.conf_threshold:
                # 低置信度：坐标不可信，标记为无效，避免下游把垃圾当真值
                all_kp[name] = Keypoint(x=x, y=y, z=None,
                                        confidence=0.0, name=name)
            else:
                all_kp[name] = Keypoint(x=x, y=y, z=None,
                                        confidence=c, name=name)

        # ---- 取出目标侧手臂 ----
        names = ARM_KEYPOINT_NAMES[self.target_side]
        arm = ArmKeypoints(
            shoulder=all_kp[names["shoulder"]],
            elbow=all_kp[names["elbow"]],
            wrist=all_kp[names["wrist"]],
            side=self.target_side,
        )

        return PoseDetection(arm=arm, all_keypoints=all_kp,
                             bbox=bbox, score=score,
                             num_persons=num_persons,
                             latency_ms=latency_ms)

    # ------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------
    @staticmethod
    def _select_person(kps, boxes) -> Optional[int]:
        """
        选人策略：包围盒面积最大者。

        理由：YOLO 不保证输出顺序稳定；按面积选通常等价于「离相机最近」，
        且多人切换时不会因为顺序变化导致关键点跳变。
        """
        n = len(kps)
        if n == 0:
            return None
        if n == 1:
            return 0

        if boxes is None or len(boxes.xyxy) != n:
            return 0

        xyxy = boxes.xyxy.cpu().numpy()
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        return int(np.argmax(areas))

    @staticmethod
    def _bbox_of(boxes, idx: int):
        if boxes is None or idx >= len(boxes.xyxy):
            return None
        xyxy = boxes.xyxy[idx].cpu().numpy()
        return tuple(float(v) for v in xyxy)

    @staticmethod
    def _score_of(boxes, idx: int) -> float:
        if boxes is None or boxes.conf is None or idx >= len(boxes.conf):
            return 0.0
        return float(boxes.conf[idx].cpu().numpy())


# ============================================================
# 阶段七预留
# ============================================================
class RGBDPoseProvider(PoseProvider):
    """
    RGB-D 姿态提供者 —— **阶段七预留，本阶段未实现**。

    与 YOLOPoseProvider 的唯一区别：
        在拿到 (u, v, conf) 之后，用 Frame.depth 与 CameraIntrinsics
        把 z 补上，使 Keypoint 从 (x, y, None, conf) 变成
        (X, Y, Z, conf)（相机坐标系，单位米）。

    这样上层（阶段四/八）拿到的数据结构完全一致，
    只是 z 从 None 变成有效值，因此不需要重写整个系统。

    阶段七实现要点（届时补全）：
        1. 复用 YOLOPoseProvider 做 2D 检测（组合而非继承，
           避免复制一份推理逻辑）；
        2. 对 shoulder/elbow/wrist 各调用 frame.depth_at(u, v)
           取中值深度（Frame 已内置该稳健采样）；
        3. 用 intrinsics.backproject(u, v, depth) 得到 (X, Y, Z)；
        4. 深度无效（空洞/超量程）时 z 仍置 None，
           并降低该点 confidence —— 不要伪造 z=0。
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "RGBDPoseProvider 属于阶段七（RGB-D / 3D）内容，当前阶段未实现。\n"
            "阶段三请使用 YOLOPoseProvider（z 恒为 None）。"
        )

    def detect(self, frame: Frame) -> PoseDetection:      # pragma: no cover
        raise NotImplementedError

    def describe(self) -> str:                            # pragma: no cover
        return "RGB-D Pose（未实现）"
