# -*- coding: utf-8 -*-
"""
piper_human_perception.hand
===========================
手部关键点识别。

职责边界：
    本层只负责「把手找出来，并给出可用的手部几何量」，
    不做机械臂关节映射（阶段四/八），不碰 ROS2。

两类提供者（保持与 pose.py 相同的抽象风格）：

    YOLOWristHandProvider     **零新增依赖**的基线。
        直接复用人体姿态模型给出的腕部，并据此派生
        「腕点 + 手掌近似几何」。信息量有限，但永远可用，
        且保证「腕部」这一项不会因为手部模型失败而丢失。

    MediaPipeHandProvider     精确的 21 点手部关键点。
        以人体腕部为锚点做 ROI 裁剪后送入 MediaPipe Hands，
        得到手根/掌心/五指指尖等完整信息。

    两者的输出结构完全一致（HandKeypoints），
    因此阶段四/八可以在任一提供者上工作，也可以级联使用
    （先用 MediaPipe，失败时回退到 YOLO 基线）。

关于「腕部」的重要说明见 piper_human_control.types.HandKeypoints 的文档：
    wrist（来自人体姿态）与 hand_root（来自手部模型）是**两个不同的量**，
    前者用于手臂链路，后者用于手部几何，不可混用。
"""

from abc import ABC, abstractmethod
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from piper_human_control import (FINGER_NAMES, ArmKeypoints,
                                 HandKeypoints, Keypoint)

from .perception import Frame


# ============================================================
# MediaPipe 21 点手部关键点定义
# ============================================================
# 该编号顺序是 MediaPipe Hands 的固定输出顺序，不可更改。
HAND_LANDMARK_NAMES: List[str] = [
    "wrist",                                                    # 0
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",          # 1-4
    "index_mcp", "index_pip", "index_dip", "index_tip",         # 5-8
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",     # 9-12
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",             # 13-16
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",         # 17-20
]

HAND_LANDMARK_INDEX: Dict[str, int] = {
    n: i for i, n in enumerate(HAND_LANDMARK_NAMES)
}

# 每根手指的四个关键点（掌指关节 -> 指尖）
FINGER_LANDMARK_NAMES: Dict[str, List[str]] = {
    "thumb": ["thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip"],
    "index": ["index_mcp", "index_pip", "index_dip", "index_tip"],
    "middle": ["middle_mcp", "middle_pip", "middle_dip", "middle_tip"],
    "ring": ["ring_mcp", "ring_pip", "ring_dip", "ring_tip"],
    "pinky": ["pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"],
}

# 指尖名称
FINGERTIP_NAMES: Dict[str, str] = {
    f: FINGER_LANDMARK_NAMES[f][-1] for f in FINGER_NAMES
}

# 掌心参考点：中指掌指关节（MediaPipe 官方建议的手掌中心近似）
PALM_CENTER_NAME = "middle_mcp"
# 手部原点
HAND_ROOT_NAME = "wrist"

# 用于绘制手部骨架的连接（MediaPipe 官方 HAND_CONNECTIONS 的等价定义）
HAND_CONNECTIONS: List[Tuple[str, str]] = (
    # 掌心环
    [("wrist", "thumb_cmc"), ("thumb_cmc", "index_mcp"),
     ("index_mcp", "middle_mcp"), ("middle_mcp", "ring_mcp"),
     ("ring_mcp", "pinky_mcp"), ("pinky_mcp", "wrist")]
    # 拇指
    + [("thumb_cmc", "thumb_mcp"), ("thumb_mcp", "thumb_ip"),
       ("thumb_ip", "thumb_tip")]
    # 其余四指
    + [(f"{f}_mcp", f"{f}_pip") for f in ("index", "middle", "ring", "pinky")]
    + [(f"{f}_pip", f"{f}_dip") for f in ("index", "middle", "ring", "pinky")]
    + [(f"{f}_dip", f"{f}_tip") for f in ("index", "middle", "ring", "pinky")]
)


# ============================================================
# 手部几何量
# ============================================================
@dataclass
class HandGeometry:
    """
    从手部关键点派生出的、可直接用于映射的几何量。

    之所以单独成一个结构而不是塞进 HandKeypoints：
        关键点是「观测」，几何量是「解释」。
        阶段八可能换用不同的几何定义，届时只需替换本结构的计算函数，
        观测数据结构不受影响。

    Attributes:
        orientation_deg: 手掌在图像平面内的朝向角（度）。
                         定义为「手腕 -> 中指掌指关节」向量的方向，
                         已做 y 翻转使「向上为正」，与 compute_arm_metrics 一致。
        openness:        手掌张开度 [0, 1]。0=握拳，1=完全张开。
                         由「各指尖到手腕的距离」相对掌宽归一化得到。
        spread:          五指张开程度 [0, 1]，描述手指之间的分散度，
                         与 openness 不同（手指可伸直但不分开）。
        finger_extension: 每根手指的伸展比例，1=完全伸直，0=完全弯曲。
        palm_width:      掌宽（像素），用于归一化，衡量手的尺度。
        hand_scale:      手的整体尺度（像素），≈ 手腕到中指指尖的距离。
        is_fist:         是否判定为握拳（openness < 阈值）
    """
    orientation_deg: float = 0.0
    openness: float = 0.0
    spread: float = 0.0
    finger_extension: Dict[str, float] = field(default_factory=dict)
    palm_width: float = 0.0
    hand_scale: float = 0.0
    is_fist: bool = False


# ============================================================
# 检测结果
# ============================================================
@dataclass
class HandDetection:
    """
    单帧手部检测结果。

    Attributes:
        hand:     手部关键点（统一结构）
        geometry: 派生几何量（手部模型成功时才有）
        latency_ms: 本帧耗时
    """
    hand: Optional[HandKeypoints] = None
    geometry: Optional[HandGeometry] = None
    latency_ms: float = 0.0

    @property
    def has_hand(self) -> bool:
        return self.hand is not None and self.hand.has_hand

    @property
    def is_complete(self) -> bool:
        return self.hand is not None and self.hand.is_complete


# ============================================================
# 抽象接口
# ============================================================
class HandProvider(ABC):
    """手部提供者抽象接口"""

    @abstractmethod
    def detect(self, frame: Frame,
               arm_wrist: Optional[Keypoint] = None,
               side: str = "right") -> HandDetection:
        """
        从一帧中提取手部。

        Args:
            frame:     图像帧
            arm_wrist: 人体姿态给出的腕部，作为 ROI 锚点（可为 None）
            side:      目标侧
        """

    @abstractmethod
    def describe(self) -> str:
        """返回提供者描述"""

    def close(self) -> None:
        """释放资源"""


# ============================================================
# 几何计算
# ============================================================
def derive_hand_geometry(hand: HandKeypoints) -> Optional[HandGeometry]:
    """
    从手部关键点计算几何量。

    需要 hand_root + 五指指尖 + 掌心可用；不满足时返回 None。

    归一化思路（重要）：
        所有距离都用「手部自身尺度」归一化，而不是用像素绝对值，
        这样人离相机远近变化时，openness / spread 保持稳定，
        阶段八做映射时不必再为距离单独标定。
    """
    if not hand.has_hand:
        return None

    root = hand.hand_root
    palm = hand.palm_center
    if root is None or palm is None or not root.is_valid or not palm.is_valid:
        return None

    tips = [hand.fingertips.get(f) for f in FINGER_NAMES]
    if any(t is None or not t.is_valid for t in tips):
        return None

    p_root = np.array([root.x, root.y], dtype=np.float64)
    p_palm = np.array([palm.x, palm.y], dtype=np.float64)

    # ---- 手掌尺度 ----
    # hand_scale: 手腕 -> 中指指尖（手的整体长度）
    middle_tip = hand.fingertips["middle"]
    p_mid_tip = np.array([middle_tip.x, middle_tip.y], dtype=np.float64)
    hand_scale = float(np.linalg.norm(p_mid_tip - p_root))

    # palm_width: 食指掌指关节 <-> 小指掌指关节（掌宽）
    idx_mcp = hand.landmarks.get("index_mcp")
    pky_mcp = hand.landmarks.get("pinky_mcp")
    if (idx_mcp is not None and pky_mcp is not None
            and idx_mcp.is_valid and pky_mcp.is_valid):
        palm_width = float(np.linalg.norm(
            np.array([idx_mcp.x, idx_mcp.y]) - np.array([pky_mcp.x, pky_mcp.y])))
    else:
        palm_width = 0.0

    # 归一化基准：优先用掌宽（对伸展不敏感），退化时用手长
    norm = palm_width if palm_width > 1e-6 else hand_scale
    if norm <= 1e-6:
        return None

    # ---- 朝向：手腕 -> 掌心，y 翻转使向上为正 ----
    dx, dy = p_palm[0] - p_root[0], p_palm[1] - p_root[1]
    orientation_deg = math.degrees(math.atan2(-dy, dx))

    # ---- 伸展度：每根手指的 (指尖到手腕距离) / 掌宽 ----
    # ⚠️ 这里**必须**用掌宽（palm_width）做归一化，不能用 hand_scale。
    #    因为 hand_scale 定义为「腕 -> 中指指尖」，握拳时它会随指尖一起
    #    缩小，分子分母同时变小 -> 比值几乎不变 -> 握拳检测失效。
    #    掌宽由食指/小指掌指关节决定，**不随手指弯曲而改变**，
    #    是稳定的尺度基准。这一点由 test_fist_has_low_openness 守住。
    #
    # 经验比值（指尖到手腕距离 / 掌宽）
    # ---- 标定来源：真实手臂近景视频实测（见阶段三报告 §0.3）----
    #   手指完全伸直（实测中位）：index 2.79 / middle 2.86 / ring 2.63 / pinky 2.37
    #   取各指实测 5 分位作为「完全伸直」下界：
    #       index 1.92 / middle 1.73 / ring 1.47 / pinky 1.36
    #   握拳时指尖缩到掌指关节附近，经验约为完全伸直的 50%~60%，
    #   故「完全握拳」上界取 0.55 × 伸直下界。
    #
    # ⚠️ 局限说明：标定视频中测试者全程张开手掌，
    #    因此 **伸直侧是实测值，握拳侧是经验估计**。
    #    阶段五接入真实握拳动作后应重新标定（届时可直接用同样的脚本采集）。
    RATIO_BOUNDS: Dict[str, Tuple[float, float]] = {
        # finger: (握拳比值, 伸直比值)
        "thumb": (0.90, 1.83),
        "index": (1.05, 1.92),
        "middle": (0.95, 1.73),
        "ring": (0.80, 1.47),
        "pinky": (0.75, 1.36),
    }

    extension: Dict[str, float] = {}
    for f in FINGER_NAMES:
        tip = hand.fingertips[f]
        d = float(np.linalg.norm(np.array([tip.x, tip.y]) - p_root))
        ratio = d / norm if norm > 1e-6 else 0.0

        lo, hi = RATIO_BOUNDS[f]
        ext = (ratio - lo) / (hi - lo)
        extension[f] = float(np.clip(ext, 0.0, 1.0))

    # ---- 张开度：四指（不含拇指）伸展度的均值 ----
    four = [extension[f] for f in ("index", "middle", "ring", "pinky")]
    openness = float(np.clip(np.mean(four), 0.0, 1.0))

    # ---- 分散度：指尖两两平均距离 / 掌宽 ----
    tip_pts = [np.array([hand.fingertips[f].x, hand.fingertips[f].y])
               for f in FINGER_NAMES]
    pair_d = []
    for i in range(len(tip_pts)):
        for j in range(i + 1, len(tip_pts)):
            pair_d.append(float(np.linalg.norm(tip_pts[i] - tip_pts[j])))
    spread_raw = float(np.mean(pair_d)) if pair_d else 0.0
    # 经验上「五指张开」时平均指尖间距约为掌宽的 1.2~1.6 倍
    spread = float(np.clip(spread_raw / (1.5 * norm), 0.0, 1.0))

    return HandGeometry(
        orientation_deg=orientation_deg,
        openness=openness,
        spread=spread,
        finger_extension=extension,
        palm_width=palm_width,
        hand_scale=hand_scale,
        is_fist=openness < 0.45,
    )


# ============================================================
# 腕部姿态（用于驱动机械臂的腕关节）
# ============================================================
@dataclass
class HandPose:
    """
    腕部姿态量 —— 用于驱动机械臂的**腕关节**。

    Attributes:
        wrist_pitch_deg: 腕俯仰 = 「手朝向」相对「前臂朝向」的夹角（度）。
                         手与前臂成一直线时 ≈ 0；手向上翘为正。
                         对应机械臂 joint5
                         （实测：joint5 让末端指向在竖直平面内转 ±50°，是纯 pitch）。
        palm_roll_deg:   掌心滚转 = 掌横轴（index_mcp -> pinky_mcp）
                         在图像平面内的朝向（度）。
                         对应机械臂 joint6
                         （实测：joint6 只让末端绕自身轴滚转 ±110°，
                           末端位置几乎不动，是纯 roll）。
        valid:           两个量是否都可用
        reason:          不可用时的原因

    ⚠️ 诚实的局限（单目 2D 固有限制）：
        只有**在图像平面内可观测**的转动才测得到。
        掌心正对/背对相机时的「拧」在 2D 投影里几乎不改变掌横轴方向，
        因此 palm_roll 对这类转动不敏感。
        要真正解耦需要深度相机（阶段七的 RGB-D 接口已预留）。

    ---- 手掌特征（按用户定义新增，用于腕部与夹爪）----

        thumb_offset:  拇指尖在**掌横轴**上的归一化投影。
                       基准是手自身：掌横轴取「小指掌指 -> 食指掌指」。
                       +1 ≈ 偏向食指侧，-1 ≈ 偏向小指侧。
                       用途：四指指向镜头时掌横轴几乎不变，
                       但拧腕会让拇指明显偏到某一侧 —— 这时它才是
                       腕部旋转的可观测信号。
        finger_tips_converged:
                       四指指尖在**图像里**的离散度（归一化到掌宽）。
                       四指并拢指向镜头时，透视会让指尖聚到一起，
                       该值显著变小。用于判断「手正对镜头」。
        pinch_distance:
                       拇指尖到四指指尖的**最近距离**（归一化到掌宽）。
                       拇指与四指靠近 -> 变小。用于夹爪闭合。
                       ⚠️ 与 openness 是**不同动作**：openness 量的是
                       「指尖到手腕的距离」（手指朝掌心收），
                       pinch 量的是「拇指与四指相互靠近」（捏合）。
                       两者可以独立发生，不能互相替代。
    """
    wrist_pitch_deg: float = 0.0
    palm_roll_deg: float = 0.0
    thumb_offset: float = 0.0
    finger_tips_converged: float = 0.0
    pinch_distance: float = 1.0
    valid: bool = False
    reason: str = ""

    def as_dict(self) -> dict:
        return {
            "wrist_pitch_deg": self.wrist_pitch_deg,
            "palm_roll_deg": self.palm_roll_deg,
            "thumb_offset": self.thumb_offset,
            "pinch_distance": self.pinch_distance,
        }


def _dir_angle_deg(dx: float, dy: float) -> float:
    """图像向量 -> 方向角（度），y 翻转使「向上为正」，与 arm_geometry 一致"""
    return math.degrees(math.atan2(-dy, dx))


def _wrap180(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def compute_hand_pose(arm, hand_det) -> HandPose:
    """
    从手臂 + 手部关键点计算腕部姿态。

    Args:
        arm:      ArmKeypoints（用 effective_wrist 作为权威腕点）
        hand_det: HandDetection（需要 landmarks 里的掌指关节）

    Returns:
        HandPose；不可用时 valid=False 并给出 reason
    """
    if arm is None:
        return HandPose(valid=False, reason="无手臂数据")
    if hand_det is None or hand_det.hand is None:
        return HandPose(valid=False, reason="无手部数据")

    h = hand_det.hand
    w = arm.effective_wrist
    e = arm.elbow
    if w is None or e is None or not w.is_valid or not e.is_valid:
        return HandPose(valid=False, reason="腕/肘不可用")

    lm = h.landmarks or {}
    mcp = lm.get("middle_mcp")
    idx = lm.get("index_mcp")
    pky = lm.get("pinky_mcp")
    for name, kp in (("middle_mcp", mcp), ("index_mcp", idx),
                     ("pinky_mcp", pky)):
        if kp is None or not kp.is_valid:
            return HandPose(valid=False, reason=f"缺少 {name}")

    # 前臂方向（肘 -> 腕）与手朝向（腕 -> 中指掌指）
    fore = _dir_angle_deg(w.x - e.x, w.y - e.y)
    hand_dir = _dir_angle_deg(mcp.x - w.x, mcp.y - w.y)
    pitch = _wrap180(hand_dir - fore)

    # 掌横轴方向（index_mcp -> pinky_mcp）作为掌面滚转
    roll = _dir_angle_deg(pky.x - idx.x, pky.y - idx.y)

    # 退化保护：掌横轴过短说明三点共线（遮挡/姿态极端）
    lat_len = math.hypot(pky.x - idx.x, pky.y - idx.y)
    if lat_len < 4.0:
        return HandPose(valid=False, reason="掌横轴过短")

    # ---- 拇指偏移：拇指尖在掌横轴上的归一化投影 ----
    # 掌横轴单位向量取「小指掌指 -> 食指掌指」，因此
    #   + 表示偏食指侧，- 表示偏小指侧。
    # 用掌宽归一化，与人离相机远近无关。
    lat = np.array([idx.x - pky.x, idx.y - pky.y], dtype=np.float64)
    lat_u = lat / lat_len
    palm_c = np.array([(idx.x + pky.x) / 2.0, (idx.y + pky.y) / 2.0])
    thumb_tip = lm.get("thumb_tip")
    if thumb_tip is None or not thumb_tip.is_valid:
        return HandPose(valid=False, reason="缺少 thumb_tip")
    tv = np.array([thumb_tip.x - palm_c[0], thumb_tip.y - palm_c[1]],
                  dtype=np.float64)
    thumb_off = float(np.dot(tv, lat_u) / lat_len)

    # ---- 四指指尖汇聚度：指尖在图像里的离散度 / 掌宽 ----
    # 四指并拢指向镜头时透视会让指尖聚拢，该值显著变小。
    tips = []
    for f in ("index", "middle", "ring", "pinky"):
        kp = lm.get(f"{f}_tip")
        if kp is None or not kp.is_valid:
            return HandPose(valid=False, reason=f"缺少 {f}_tip")
        tips.append(np.array([kp.x, kp.y], dtype=np.float64))
    m = np.mean(tips, axis=0)
    # 指尖到其质心的平均距离：对「聚成一点」比两两距离更直接
    conv = float(np.mean([np.linalg.norm(t - m) for t in tips]) / lat_len)

    # ---- 捏合距离：拇指尖到四指指尖的最近距离 / 掌宽 ----
    pinch = float(min(np.linalg.norm(thumb_tip_pt - t)
                      for thumb_tip_pt, t in zip(
                          [np.array([thumb_tip.x, thumb_tip.y])] * 4, tips))
                  / lat_len)

    return HandPose(wrist_pitch_deg=pitch,
                    palm_roll_deg=roll,
                    thumb_offset=thumb_off,
                    finger_tips_converged=conv,
                    pinch_distance=pinch,
                    valid=True)


# ============================================================
# 提供者 1：零依赖基线（复用人体腕部）
# ============================================================
class YOLOWristHandProvider(HandProvider):
    """
    基于人体姿态腕部的**基线**手部提供者。

    它能提供什么：
        * wrist（人体姿态的腕部，用于手臂链路）
        * hand_root（这里就用人体腕部代替，作为手掌原点）

    它**不能**提供什么：
        * 五指指尖、掌心、张开度、朝向
          —— 人体姿态模型只给出一个腕点，没有手部细节。

    存在的意义（不是凑数）：
        1. 保证「腕部」这一项在任何情况下都不会丢失 ——
           手部模型可能因遮挡/出画/初始化失败而返回空，
           此时上层至少还有腕点可用，不会整条链路断掉；
        2. 作为 MediaPipeHandProvider 的自动回退目标；
        3. 让阶段四可以先只依赖腕部把关节映射跑起来，
           不必等手部模型完全稳定。

    实现代价：**零**（不加载任何额外模型，只是搬运一个已有的点）。
    """

    def detect(self, frame: Frame,
               arm_wrist: Optional[Keypoint] = None,
               side: str = "right") -> HandDetection:
        t0 = time.perf_counter()

        if arm_wrist is None or not arm_wrist.is_valid:
            return HandDetection(latency_ms=(time.perf_counter() - t0) * 1000.0)

        # 用腕点构造一个最小可用的 HandKeypoints
        wrist = Keypoint(x=arm_wrist.x, y=arm_wrist.y, z=arm_wrist.z,
                         confidence=arm_wrist.confidence,
                         name=f"{side}_wrist")
        root = Keypoint(x=arm_wrist.x, y=arm_wrist.y, z=arm_wrist.z,
                        confidence=arm_wrist.confidence,
                        name=HAND_ROOT_NAME)
        hand = HandKeypoints(
            side=side,
            wrist=wrist,
            hand_root=root,
            landmarks={HAND_ROOT_NAME: root},
            palm_center=None,
            fingertips={},
            source="yolo_wrist_baseline",
        )
        return HandDetection(hand=hand,
                             geometry=None,
                             latency_ms=(time.perf_counter() - t0) * 1000.0)

    def describe(self) -> str:
        return "YOLO 腕部基线（仅腕点，无手部细节，零额外依赖）"


# ============================================================
# 提供者 2：MediaPipe 21 点
# ============================================================
class MediaPipeHandProvider(HandProvider):
    """
    基于 MediaPipe HandLandmarker（**Tasks API**）的 21 点手部关键点识别。

    ⚠️ 关于 MediaPipe 版本的重要区别：
        mediapipe <= 0.10.x 提供旧版 `mp.solutions.hands`；
        **mediapipe 1.x 已移除 `mp.solutions`**，只保留 Tasks API
        （`mediapipe.tasks.python.vision.HandLandmarker`）。
        本实现按 Tasks API 编写，并在 load() 时给出明确报错提示，
        避免出现 `module 'mediapipe' has no attribute 'solutions'`
        这类难以理解的失败。

    Tasks API 需要一个 .task 模型文件（不随 pip 包分发），
    默认从 ~/Yolo_pose+piper/models/hand_landmarker.task 读取。
    模型下载见 README / 报告。

    ROI 策略（效果好坏的关键）：
        以 YOLO Pose 给出的腕点为中心裁一块 ROI 再送入 MediaPipe，
        显著优于整图送入 —— 因为整图上「大画幅 + 小手」的检出率很低。

    参数：
        model_path:  .task 模型路径
        running_mode: "VIDEO"（逐帧同步，默认，带跟踪，最省算力）
                      "IMAGE"（每帧独立检测，无跟踪）
        num_hands:   最多检测几只手
        roi_scale:   ROI 边长 = roi_scale × 估计手长（像素）
        use_roi:     是否使用 ROI 裁剪
    """

    # 运行模式常量（避免 import 期就依赖 mediapipe）
    MODE_IMAGE = "IMAGE"
    MODE_VIDEO = "VIDEO"

    def __init__(self,
                 model_path: Optional[str] = None,
                 running_mode: str = MODE_VIDEO,
                 num_hands: int = 1,
                 min_hand_detection_confidence: float = 0.5,
                 min_hand_presence_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 roi_scale: float = 2.5,
                 use_roi: bool = True,
                 default_hand_size: int = 200):
        if model_path is None:
            model_path = os.path.expanduser(
                "~/Yolo_pose+piper/models/hand_landmarker.task")
        self.model_path = model_path
        self.running_mode = running_mode
        self.num_hands = num_hands
        self.min_hand_detection_confidence = min_hand_detection_confidence
        self.min_hand_presence_confidence = min_hand_presence_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.roi_scale = roi_scale
        self.use_roi = use_roi
        self.default_hand_size = default_hand_size

        self._landmarker = None
        self._mode = None
        self._ts_ms = 0

    # ------------------------------------------------------------
    def load(self) -> None:
        """惰性初始化 HandLandmarker（Tasks API）"""
        if self._landmarker is not None:
            return

        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(
                f"找不到手部模型: {self.model_path}\n"
                f"请下载 hand_landmarker.task：\n"
                f"  curl -L -o {self.model_path} \\\n"
                f"    https://storage.googleapis.com/mediapipe-models/"
                f"hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")

        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
        except ImportError as exc:
            raise ImportError(
                "mediapipe 的 Tasks API 不可用。"
                "mediapipe 1.x 已移除旧的 mp.solutions 接口，"
                f"本实现需要 tasks API。原始错误: {exc}") from exc

        base = mp_python.BaseOptions(model_asset_path=self.model_path)

        if self.running_mode == self.MODE_VIDEO:
            opts = mp_vision.HandLandmarkerOptions(
                base_options=base,
                running_mode=mp_vision.RunningMode.VIDEO,
                num_hands=self.num_hands,
                min_hand_detection_confidence=self.min_hand_detection_confidence,
                min_hand_presence_confidence=self.min_hand_presence_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
            self._mode = mp_vision.RunningMode.VIDEO
        else:
            opts = mp_vision.HandLandmarkerOptions(
                base_options=base,
                running_mode=mp_vision.RunningMode.IMAGE,
                num_hands=self.num_hands,
                min_hand_detection_confidence=self.min_hand_detection_confidence,
                min_hand_presence_confidence=self.min_hand_presence_confidence,
            )
            self._mode = mp_vision.RunningMode.IMAGE

        self._landmarker = mp_vision.HandLandmarker.create_from_options(opts)
        self._mp_vision = mp_vision
        self._ts_ms = 0

    def describe(self) -> str:
        return (f"MediaPipe HandLandmarker/Tasks (21点, 模式={self.running_mode}, "
                f"ROI={'开' if self.use_roi else '关'}, "
                f"模型={os.path.basename(self.model_path)})")

    # ------------------------------------------------------------
    @staticmethod
    def is_available() -> bool:
        """
        MediaPipe Tasks 手部检测是否可用。

        注意：这里必须检查 **Tasks API**（HandLandmarker），
        而不是仅仅 `import mediapipe` —— mediapipe 1.x 能 import
        但没有 mp.solutions，只检查 import 会得出错误结论。
        同时检查模型文件是否存在，否则调用时才发现失败。
        """
        try:
            from mediapipe.tasks.python import vision as mp_vision
            assert hasattr(mp_vision, "HandLandmarker")
        except Exception:                                  # noqa: BLE001
            return False

        default_model = os.path.expanduser(
            "~/Yolo_pose+piper/models/hand_landmarker.task")
        return os.path.isfile(default_model)

    # ------------------------------------------------------------
    def _compute_roi(self, frame: Frame, arm_wrist: Optional[Keypoint]):
        """返回 (roi, ox, oy)"""
        img = frame.rgb
        h, w = img.shape[:2]

        if not (self.use_roi and arm_wrist is not None and arm_wrist.is_valid):
            return img, 0, 0

        size = max(64, int(self.default_hand_size * self.roi_scale))
        cx, cy = int(round(arm_wrist.x)), int(round(arm_wrist.y))
        x0 = max(0, cx - size // 2)
        y0 = max(0, cy - size // 2)
        x1 = min(w, x0 + size)
        y1 = min(h, y0 + size)

        if x1 - x0 < 32 or y1 - y0 < 32:
            return img, 0, 0
        return img[y0:y1, x0:x1], x0, y0

    def detect(self, frame: Frame,
               arm_wrist: Optional[Keypoint] = None,
               side: str = "right") -> HandDetection:
        t0 = time.perf_counter()
        self.load()

        import cv2
        import mediapipe as mp

        roi, ox, oy = self._compute_roi(frame, arm_wrist)

        # ---- 构造 wrist 关键点（来自人体姿态，保持手臂链路语义）----
        wrist_kp = None
        if arm_wrist is not None and arm_wrist.is_valid:
            wrist_kp = Keypoint(x=arm_wrist.x, y=arm_wrist.y, z=arm_wrist.z,
                                confidence=arm_wrist.confidence,
                                name=f"{side}_wrist")

        # ---- 推理 ----
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        if self._mode == self._mp_vision.RunningMode.VIDEO:
            # VIDEO 模式要求时间戳单调递增
            self._ts_ms += 33
            result = self._landmarker.detect_for_video(mp_image, self._ts_ms)
        else:
            result = self._landmarker.detect(mp_image)

        latency_ms = (time.perf_counter() - t0) * 1000.0

        hands = getattr(result, "hand_landmarks", None)
        if not hands:
            hand = HandKeypoints(side=side, wrist=wrist_kp,
                                 source="mediapipe_handlandmarker")
            return HandDetection(hand=hand, latency_ms=latency_ms)

        # ---- 取第一只手（单臂场景）----
        lms = hands[0]
        rh, rw = roi.shape[:2]

        score = 1.0
        try:
            handedness = getattr(result, "handedness", None)
            if handedness:
                score = float(handedness[0][0].score)
        except Exception:                                  # noqa: BLE001
            score = 1.0

        landmarks: Dict[str, Keypoint] = {}
        for i, name in enumerate(HAND_LANDMARK_NAMES):
            lm = lms[i]
            px = ox + float(lm.x) * rw
            py = oy + float(lm.y) * rh
            # lm.z 是相对深度（以手腕为原点的相对值），不是米制深度，
            # **不能**冒充 Keypoint.z —— 阶段七由 RGB-D 提供真实深度。
            landmarks[name] = Keypoint(x=px, y=py, z=None,
                                       confidence=score, name=name)

        fingertips = {f: landmarks[FINGERTIP_NAMES[f]] for f in FINGER_NAMES}

        hand = HandKeypoints(
            side=side,
            wrist=wrist_kp,
            hand_root=landmarks[HAND_ROOT_NAME],
            landmarks=landmarks,
            palm_center=landmarks[PALM_CENTER_NAME],
            fingertips=fingertips,
            source="mediapipe_handlandmarker",
        )
        geometry = derive_hand_geometry(hand)
        return HandDetection(hand=hand, geometry=geometry, latency_ms=latency_ms)

    def close(self) -> None:
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:                              # noqa: BLE001
                pass
            self._landmarker = None


# ============================================================
# 级联提供者：优先精细模型，失败自动回退
# ============================================================
class CascadingHandProvider(HandProvider):
    """
    串联两个提供者：优先用 primary（MediaPipe），
    当它没检出手部时自动回退到 fallback（YOLO 腕部基线）。

    这样上层无论何时都能拿到一个「至少有腕部」的结果，
    不会因为手部模型偶发失败而整帧丢失手部信息。
    """

    def __init__(self, primary: HandProvider, fallback: HandProvider):
        self.primary = primary
        self.fallback = fallback
        self.last_source = ""

    def detect(self, frame: Frame,
               arm_wrist: Optional[Keypoint] = None,
               side: str = "right") -> HandDetection:
        det = self.primary.detect(frame, arm_wrist, side)
        if det.has_hand:
            self.last_source = "primary"
            return det

        fb = self.fallback.detect(frame, arm_wrist, side)
        self.last_source = "fallback"
        # 保留 primary 的耗时统计，便于观察真实开销
        fb.latency_ms = det.latency_ms + fb.latency_ms
        return fb

    def describe(self) -> str:
        return f"级联[{self.primary.describe()} -> 回退 {self.fallback.describe()}]"

    def close(self) -> None:
        self.primary.close()
        self.fallback.close()


def make_hand_provider(prefer_mediapipe: bool = True,
                       **kwargs) -> HandProvider:
    """
    工厂函数：按可用性选择手部提供者。

    Args:
        prefer_mediapipe: 是否优先使用 MediaPipe（不可用时自动降级）
        **kwargs:         传给 MediaPipeHandProvider 的参数

    Returns:
        HandProvider。
        MediaPipe Tasks 可用（且模型文件存在）时返回级联提供者，
        否则只返回 YOLO 腕部基线 —— 保证上层永远拿得到腕点。
    """
    fallback = YOLOWristHandProvider()
    if prefer_mediapipe and MediaPipeHandProvider.is_available():
        return CascadingHandProvider(MediaPipeHandProvider(**kwargs), fallback)
    return fallback


# ============================================================
# 腕部接入：用手部识别的腕部替换人体识别的腕部
# ============================================================
def merge_hand_wrist(arm: Optional[ArmKeypoints],
                     hand_det: Optional["HandDetection"]
                     ) -> Optional[ArmKeypoints]:
    """
    把**手部识别的腕部**接入手臂链路，作为肘部的连接点。

    这是本项目的一条核心规则：
        肘部的连接对象是**手部模型定位的腕关节**（hand_root），
        而**不是**人体姿态模型给出的腕部。

    为什么：
        人体姿态模型同时要负责全身 17 个点，腕部只是其中一个，
        在有遮挡/侧向时误差明显；而手部模型的第 0 个关键点
        天生就是腕关节，专门为手部定位训练，精度更高。
        实测两者中位相差约 26 px —— 足以让机械臂末端明显偏位。

    行为：
        * 手部识别成功   -> 返回新的 ArmKeypoints，hand_wrist = 手部腕点，
                            此后 effective_wrist 一律取手部腕点。
        * 手部识别失败   -> 返回原 arm 对象（effective_wrist 自动回退到人体腕部），
                            保证链路不中断，而不是让腕部凭空消失。

    Args:
        arm:      人体姿态给出的手臂关键点（可为 None）
        hand_det: 手部检测结果（可为 None）

    Returns:
        接入后的 ArmKeypoints；arm 为 None 时返回 None。
    """
    if arm is None:
        return None

    if (hand_det is not None
            and hand_det.hand is not None
            and hand_det.hand.hand_root is not None
            and hand_det.hand.hand_root.is_valid):
        root = hand_det.hand.hand_root
        return arm.with_hand_wrist(
            Keypoint(x=root.x, y=root.y, z=root.z,
                     confidence=root.confidence,
                     name=f"{arm.side}_wrist_from_hand"))

    # 手部失败：保留人体腕部作为兜底，链路不断
    return arm

