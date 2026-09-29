# -*- coding: utf-8 -*-
"""
piper_human_perception.perception
=================================
帧源抽象层。

设计目标（对应任务书总体原则第 11 条「保留未来 RGB-D / 3D 接口」）：
    上层（pose / visualization）只依赖 FrameSource 这个抽象，
    不关心画面来自 USB 摄像头、RealSense、视频文件还是单张图片。
    阶段七接入 RGB-D 时，只需新增一个 FrameSource 子类，
    并让它额外提供深度图，不需要改动 pose 层与可视化层。

当前实现：
    OpenCVCameraSource   USB / V4L2 摄像头（阶段三使用）
    VideoFileSource      视频文件（用于离线调试与回归测试）
    ImageFileSource      单张图片（用于单元测试）
    RealSenseSource      阶段七预留（本阶段故意不实现，接口先占位）
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# ============================================================
# 帧数据结构
# ============================================================
@dataclass
class Frame:
    """
    一帧图像数据。

    Attributes:
        rgb:        BGR 图像 (H, W, 3)，uint8。沿用 OpenCV 约定。
        depth:      深度图 (H, W)，float32，单位米。
                    **阶段三恒为 None**；阶段七由 RGB-D 相机填充。
        timestamp:  采集时刻（单调时钟，秒）
        frame_id:   帧序号
        source_name:来源名称，便于日志定位

    为什么 depth 放在 Frame 而不是 Keypoint：
        深度是「整幅图」的属性，关键点只是其中若干像素。
        放在 Frame 上，阶段七只需一次深度查询即可为所有关键点补 z，
        避免每个关键点各自去访问相机。
    """
    rgb: np.ndarray
    depth: Optional[np.ndarray] = None
    timestamp: float = 0.0
    frame_id: int = 0
    source_name: str = ""

    @property
    def height(self) -> int:
        return int(self.rgb.shape[0])

    @property
    def width(self) -> int:
        return int(self.rgb.shape[1])

    def has_depth(self) -> bool:
        """是否携带深度信息（阶段七才会为 True）"""
        return self.depth is not None

    def depth_at(self, x: float, y: float, radius: int = 2) -> Optional[float]:
        """
        查询某个像素的深度值（米）。

        采用「小窗口有效值中位数」而不是单点采样：
        深度图普遍存在空洞（无效值 0 / NaN），单点采样极不稳定。

        Returns:
            深度值（米）；无深度图或该处全为无效值时返回 None。
        """
        if self.depth is None:
            return None

        h, w = self.depth.shape[:2]
        xi, yi = int(round(x)), int(round(y))
        if not (0 <= xi < w and 0 <= yi < h):
            return None

        x0, x1 = max(0, xi - radius), min(w, xi + radius + 1)
        y0, y1 = max(0, yi - radius), min(h, yi + radius + 1)
        patch = self.depth[y0:y1, x0:x1].astype(np.float32)

        valid = patch[np.isfinite(patch) & (patch > 0.0)]
        if valid.size == 0:
            return None
        return float(np.median(valid))


# ============================================================
# 帧源抽象
# ============================================================
class FrameSource(ABC):
    """帧源抽象基类"""

    @abstractmethod
    def open(self) -> bool:
        """打开设备/文件。返回是否成功。"""

    @abstractmethod
    def read(self) -> Optional[Frame]:
        """读取一帧；失败返回 None。"""

    @abstractmethod
    def release(self) -> None:
        """释放资源。"""

    @abstractmethod
    def describe(self) -> str:
        """返回来源描述，用于日志。"""

    # 支持 with 语法
    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False


# ============================================================
# 摄像头
# ============================================================
class OpenCVCameraSource(FrameSource):
    """
    USB / V4L2 摄像头帧源。

    参数说明：
        index:      摄像头序号（/dev/video0 -> 0）
        width/height: 期望分辨率（驱动可能给出最接近的可用值）
        fps:        期望帧率
        backend:    OpenCV 后端。Linux 上显式使用 CAP_V4L2 更稳定。
        warmup:     打开后丢弃的帧数。
                    摄像头刚打开时前几帧常为自动曝光未收敛的暗帧，
                    丢弃它们可避免阶段三首帧全黑/过暗导致误检。
    """

    def __init__(self, index: int = 0, width: int = 640, height: int = 480,
                 fps: int = 30, backend: int = None, warmup: int = 5,
                 name: str = ""):
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self.warmup = warmup
        self.name = name or f"camera{index}"

        # 默认在 Linux 上使用 V4L2 后端
        if backend is None:
            backend = getattr(__import__("cv2"), "CAP_V4L2", 0)
        self.backend = backend

        self._cap = None
        self._frame_id = 0

    def open(self) -> bool:
        import cv2

        self._cap = cv2.VideoCapture(self.index, self.backend)
        if not self._cap.isOpened():
            self._cap = None
            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)

        # 丢弃预热帧，等待自动曝光/白平衡收敛
        for _ in range(self.warmup):
            self._cap.read()

        return True

    def read(self) -> Optional[Frame]:
        if self._cap is None:
            return None

        import time
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None

        self._frame_id += 1
        return Frame(rgb=frame, depth=None,
                     timestamp=time.monotonic(),
                     frame_id=self._frame_id,
                     source_name=self.name)

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def describe(self) -> str:
        actual = ""
        if self._cap is not None:
            import cv2
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual = f" 实际={w}x{h}"
        return f"OpenCV 摄像头 index={self.index}{actual}"


class VideoFileSource(FrameSource):
    """
    视频文件帧源。

    用途：无需真人在场即可回归测试整条感知链路，
    也方便用录制视频反复调参。
    """

    def __init__(self, path: str, loop: bool = False):
        self.path = path
        self.loop = loop
        self._cap = None
        self._frame_id = 0

    def open(self) -> bool:
        import cv2

        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            self._cap = None
            return False
        return True

    def read(self) -> Optional[Frame]:
        import time

        if self._cap is None:
            return None

        ok, frame = self._cap.read()
        if not ok:
            if self.loop:
                self._cap.set(2, 0)          # cv2.CAP_PROP_POS_FRAMES = 0
                ok, frame = self._cap.read()
            if not ok or frame is None:
                return None

        self._frame_id += 1
        return Frame(rgb=frame, depth=None,
                     timestamp=time.monotonic(),
                     frame_id=self._frame_id,
                     source_name=self.path)

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def describe(self) -> str:
        return f"视频文件 {self.path}"


class ImageFileSource(FrameSource):
    """
    单张图片帧源（只产出一次）。

    用途：单元测试与快速验证，不需要摄像头。
    """

    def __init__(self, path: str):
        self.path = path
        self._img = None
        self._emitted = False

    def open(self) -> bool:
        import cv2

        self._img = cv2.imread(self.path)
        self._emitted = False
        return self._img is not None

    def read(self) -> Optional[Frame]:
        import time

        if self._img is None or self._emitted:
            return None
        self._emitted = True
        return Frame(rgb=self._img, depth=None,
                     timestamp=time.monotonic(), frame_id=1,
                     source_name=self.path)

    def release(self) -> None:
        self._img = None

    def describe(self) -> str:
        return f"图片文件 {self.path}"


# ============================================================
# 阶段七预留
# ============================================================
class RealSenseSource(FrameSource):
    """
    Intel RealSense RGB-D 帧源 —— **阶段七预留，本阶段未实现**。

    保留此占位类的目的：
        明确「RGB-D 从哪进来」这一架构问题的答案，
        使阶段三的 pose 层不必为将来改动而妥协设计。

    阶段七实现要点（届时补全）：
        1. 用 pyrealsense2 打开 pipeline，同时取 color 与 depth 流；
        2. 用 rs.align(rs.stream.color) 做深度-彩色对齐
           （否则深度图与彩色图像素不对应，查深度会错位）；
        3. read() 返回的 Frame 需同时填 rgb 与 depth（单位米，float32）；
        4. 相机内参直接填入 CameraIntrinsics。
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "RealSenseSource 属于阶段七（RGB-D）内容，当前阶段未实现。\n"
            "阶段三请使用 OpenCVCameraSource / VideoFileSource / ImageFileSource。"
        )

    def open(self) -> bool:                       # pragma: no cover
        raise NotImplementedError

    def read(self) -> Optional[Frame]:            # pragma: no cover
        raise NotImplementedError

    def release(self) -> None:                    # pragma: no cover
        pass

    def describe(self) -> str:                    # pragma: no cover
        return "RealSense（未实现）"


# ============================================================
# 相机内参（阶段七使用，阶段三先定义好）
# ============================================================
@dataclass
class CameraIntrinsics:
    """
    针孔相机内参。

    阶段三：不使用（因为没有深度）。
    阶段七：配合 Frame.depth 把像素 (u, v) 反投影为三维点：
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
    现在定义好，是为了让阶段七只需「填参数」而不必重构接口。
    """
    fx: float
    fy: float
    cx: float
    cy: float

    def is_valid(self) -> bool:
        return self.fx > 0 and self.fy > 0

    def backproject(self, u: float, v: float, depth_m: float
                    ) -> Optional[Tuple[float, float, float]]:
        """
        像素 + 深度 -> 相机坐标系三维点 (米)。

        阶段三不会调用（depth 为 None），阶段七直接可用。
        """
        if depth_m is None or depth_m <= 0 or not self.is_valid():
            return None
        x = (u - self.cx) * depth_m / self.fx
        y = (v - self.cy) * depth_m / self.fy
        return (x, y, float(depth_m))
