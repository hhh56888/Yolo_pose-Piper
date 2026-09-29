# -*- coding: utf-8 -*-
"""
piper_human_control.types
=========================
基础数据结构定义。

设计原则：
    本模块只定义「数据长什么样」，不做任何计算，也不依赖 ROS2。
    这样阶段三/四的人体姿态模块可以复用同一套结构，
    而不会把感知逻辑和控制逻辑耦合在一起。

与后续阶段的衔接：
    Keypoint 保留了 z 字段与 source 字段。
    阶段三（RGB YOLO）时 z=None；
    阶段七（RGB-D）时 z 填入真实深度，source 变为 "rgbd"。
    这样上层代码无需改动即可支持 3D。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


# ============================================================
# 人体关键点
# ============================================================
@dataclass
class Keypoint:
    """
    单个人体关键点。

    Attributes:
        x:          图像横坐标（像素）；RGB-D 模式下可为归一化坐标
        y:          图像纵坐标（像素）
        z:          深度（米）。当前 RGB 模式恒为 None，保留字段以便
                    阶段七升级 RGB-D 时无需修改整个系统的数据结构。
        confidence: 置信度 [0, 1]
        name:       关键点名称，如 "right_shoulder"

    注意：z 允许为 None 是本项目「保留 3D 接口」的关键约定，
    所有下游代码必须显式处理 z is None 的情况，而不是假装它是 0。
    """
    x: float
    y: float
    z: Optional[float] = None
    confidence: float = 0.0
    name: str = ""

    @property
    def is_valid(self) -> bool:
        """关键点是否可用（有坐标且置信度非负）"""
        return self.confidence > 0.0

    def has_depth(self) -> bool:
        """是否携带有效深度信息（阶段七才会为 True）"""
        return self.z is not None


@dataclass
class ArmKeypoints:
    """
    一侧手臂的三个关键点。

    ===== 关于腕部来源的重要约定 =====
    本项目的**权威腕部来自手部识别**，不是人体姿态识别。

    理由：
        人体姿态模型（YOLO Pose）对腕部只是「手臂链路的粗略末端」，
        而手部模型（MediaPipe）21 个点里的第 0 点就是腕关节，
        是专门为手部定位训练的，精度明显更高。

    因此：
        * `wrist`      保留为**人体姿态给出的腕部**，仅用于两件事：
                         (a) 手部检测的 ROI 锚点（搜索起点）
                         (b) 手部识别完全失败时的兜底
                       **不应**作为手臂骨架末端或映射输入。
        * `hand_wrist` **手部识别给出的腕部 —— 权威值**。
        * `effective_wrist` 对外统一出口：优先 hand_wrist，回退 wrist。
                       骨架绘制、几何计算、阶段四映射都应使用它，
                       使「抛弃人体腕部」这条规则只有一处实现。

    ⚠️ 注意 wrist 与 hand_wrist **不重合**（实测中位相差约 26 px）。
       下游若误用 wrist，骨架末端会明显偏离真实手腕。
    """
    shoulder: Optional[Keypoint] = None
    elbow: Optional[Keypoint] = None
    wrist: Optional[Keypoint] = None          # 人体姿态腕部（仅 ROI 锚点 / 兜底）
    side: str = "right"                       # "right" / "left"
    hand_wrist: Optional[Keypoint] = None     # 手部识别腕部（权威）

    @property
    def is_complete(self) -> bool:
        """
        手臂三点是否都有效。

        这里刻意用**人体姿态的 wrist** 判断，而不是 effective_wrist ——
        本属性描述的是「人体姿态链路本身是否完整」；
        手部是否可用由 HandKeypoints.is_complete 单独描述。
        """
        return all(kp is not None and kp.is_valid
                   for kp in (self.shoulder, self.elbow, self.wrist))

    @property
    def effective_wrist(self) -> Optional[Keypoint]:
        """
        对外统一使用的腕部：**优先手部识别，其次人体姿态**。

        所有绘制与几何计算都应通过本属性取腕部。
        """
        if self.hand_wrist is not None and self.hand_wrist.is_valid:
            return self.hand_wrist
        return self.wrist

    @property
    def wrist_source(self) -> str:
        """当前生效的腕部来自哪里：'hand' / 'pose' / 'none'"""
        if self.hand_wrist is not None and self.hand_wrist.is_valid:
            return "hand"
        if self.wrist is not None and self.wrist.is_valid:
            return "pose"
        return "none"

    def with_hand_wrist(self, hand_wrist: Optional[Keypoint]) -> "ArmKeypoints":
        """
        返回一个「手部腕部已接入」的新 ArmKeypoints。

        用不可变式返回而不是就地修改，避免同一对象被反复改写
        而产生难以追踪的状态。
        """
        return ArmKeypoints(
            shoulder=self.shoulder,
            elbow=self.elbow,
            wrist=self.wrist,
            side=self.side,
            hand_wrist=hand_wrist,
        )

    def as_list(self) -> List[Optional[Keypoint]]:
        """返回 [肩, 肘, 有效腕] —— 供几何计算使用"""
        return [self.shoulder, self.elbow, self.effective_wrist]


# ============================================================
# 手部关键点
# ============================================================
# 手指名称（顺序：拇指 -> 小指）
FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")


@dataclass
class HandKeypoints:
    """
    一侧手部的关键点集合。

    ===== 这里必须先说清楚一个概念区分 =====
    「腕部(wrist)」有两个不同来源，本结构同时保留，不要混淆：

      1. wrist —— 来自**人体姿态模型**（YOLO Pose 的 COCO-17 第 9/10 点）。
         它是「手臂链路」的粗略末端，
         **仅用于**：(a) 手部检测的 ROI 锚点；(b) 手部失败时的兜底。

      2. hand_root —— 来自**手部模型**（MediaPipe 的 landmark 0）。
         它就是手部模型定位的腕关节，是本项目**权威的腕部**，
         也是 21 个手部关键点的原点。

    ★ 本项目的规则是：**手臂骨架末端与所有映射，统一使用 hand_root。**
      （ArmKeypoints.effective_wrist 实现了这条规则，
        它会优先取 hand_root 并把人体腕部降级为兜底。）
      wrist 字段保留下来只是为了保证「手部模型失败时链路不断」。

    Attributes:
        side:         "right" / "left"
        wrist:        人体姿态给出的腕部（ROI 锚点 / 兜底，非权威）
        hand_root:    手部模型给出的腕关节 —— **权威腕部**
        landmarks:    手部模型原始关键点字典，键为 landmark 名称
                      （RGB 模式 z=None；阶段七填深度）
        palm_center:  掌心（landmark 9，中指掌指关节）
        fingertips:   五个指尖，键为 FINGER_NAMES
        source:       产生本结果的手部模型名称，便于排查
    """
    side: str = "right"
    wrist: Optional[Keypoint] = None
    hand_root: Optional[Keypoint] = None
    landmarks: Dict[str, Keypoint] = field(default_factory=dict)
    palm_center: Optional[Keypoint] = None
    fingertips: Dict[str, Keypoint] = field(default_factory=dict)
    source: str = ""

    @property
    def has_hand(self) -> bool:
        """是否有手部模型的输出（不只是人体姿态的腕部）"""
        return self.hand_root is not None and self.hand_root.is_valid

    @property
    def is_complete(self) -> bool:
        """
        『手部链路完整』的判据：
        人体腕部有效 + 手部模型给出有效 hand_root + 五个指尖齐全。

        注意：这里故意要求 wrist 也有效 —— 因为阶段四/八需要
        「手臂 + 手」同时可用才能做有意义的重定向；
        只有手没有臂（或反之）都不算完整。
        """
        if self.wrist is None or not self.wrist.is_valid:
            return False
        if not self.has_hand:
            return False
        return all(kp is not None and kp.is_valid
                   for kp in self.fingertips.values()) and \
            len(self.fingertips) == len(FINGER_NAMES)

    @property
    def num_landmarks(self) -> int:
        return len(self.landmarks)

    def fingertip(self, finger: str) -> Optional[Keypoint]:
        """按名称取指尖"""
        return self.fingertips.get(finger)

    def named_landmarks(self) -> List[str]:
        """返回所有已有关键点的名称（便于调试/日志）"""
        return sorted(self.landmarks.keys())


# ============================================================
# 关节目标
# ============================================================
@dataclass
class JointTarget:
    """
    一次关节目标请求。

    这是「上层意图」的表达，尚未经过任何安全处理。
    经过 SafetyLimiter 处理后才会变成可直接下发的值。
    """
    positions: Sequence[float]              # 目标关节角 (rad)，长度 = 关节数
    velocities: Optional[Sequence[float]] = None   # 可选：目标速度
    label: str = ""                          # 用途标记，便于日志追踪

    def as_list(self) -> List[float]:
        return [float(v) for v in self.positions]


@dataclass
class JointState:
    """
    机器人当前关节状态快照。

    由控制器从 /joint_states 缓存而来。
    """
    positions: List[float] = field(default_factory=list)
    velocities: List[float] = field(default_factory=list)
    stamp: float = 0.0                       # 接收时刻（本地单调时钟，秒）
    received: bool = False                   # 是否曾经收到过有效状态

    def is_stale(self, now: float, timeout_s: float) -> bool:
        """状态是否已过期"""
        if not self.received:
            return True
        return (now - self.stamp) > timeout_s


@dataclass
class SafetyReport:
    """
    单次安全处理的结果记录。

    把「做了什么限制」显式记录下来，而不是静默修改数值，
    便于定位问题（对应任务书「不要硬编码临时绕过」的要求）。
    """
    clamped_joints: List[str] = field(default_factory=list)      # 触发位置限位
    rate_limited_joints: List[str] = field(default_factory=list)  # 触发速率限制
    rejected: bool = False                                        # 是否整体拒绝
    reason: str = ""                                              # 拒绝原因
    # 经过处理后的最终值
    positions: List[float] = field(default_factory=list)

    @property
    def was_modified(self) -> bool:
        return bool(self.clamped_joints or self.rate_limited_joints)
