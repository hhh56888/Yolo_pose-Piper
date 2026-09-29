# -*- coding: utf-8 -*-
"""
piper_human_control.streaming
=============================
关节指令流式发送器。

为什么需要它：
    任务书要求「平滑连续发送」。直接把目标写成阶跃下发会产生冲击，
    而且 JointTrajectoryController 在收到单点轨迹时，会以「当前位置」
    作为新轨迹起点重新规划；若轨迹时间窗设置不当，会导致跟踪严重滞后。

    本模块把「上层想要的目标」与「实际下发的轨迹」解耦：
      * set_target()   —— 上层随时更新目标（可以高频，读 YOLO 结果）
      * 按 publish_rate_hz 定时把「当前内部目标」用固定时间窗发下去

实测依据（见阶段一报告与参数扫描）：
    trajectory_horizon_s = 0.1 s, publish_rate = 20 Hz
    → 稳态跟踪误差 < 1e-4 rad
    horizon = 1.0 s 时同样条件下滞后明显（3s 只到 1.434 / 目标 1.5）
    因此时间窗取小值是关键。

与阶段五的衔接：
    本模块不含滤波与置信度逻辑，只负责「把已确定的目标平滑发出去」。
    EMA / Deadband / Lost Tracking 在阶段五的 control 层加入，
    这样职责单一，便于分别测试。
"""

import time
from typing import List, Optional, Sequence

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from rclpy.publisher import Publisher
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from .config import ControlConfig


class TrajectoryStreamer:
    """
    把关节目标以固定时间窗的 JointTrajectory 持续下发。

    注意 QoS：
        arm_controller 的 joint_trajectory 订阅是 BEST_EFFORT。
        发布端若用默认 RELIABLE，会与之不兼容（收不到）。
    """

    def __init__(self, cfg: ControlConfig, publisher: Publisher, logger=None):
        self.cfg = cfg
        self.pub = publisher
        self.logger = logger

        self._target: Optional[List[float]] = None
        self._last_sent: Optional[List[float]] = None
        self._publish_count = 0
        self._failure_count = 0
        self._last_publish_time: Optional[float] = None

    # ------------------------------------------------------------
    # QoS
    # ------------------------------------------------------------
    @staticmethod
    def make_qos(depth: int = 10) -> QoSProfile:
        """
        构造与 arm_controller 兼容的 QoS。

        arm_controller 订阅端为 BEST_EFFORT / VOLATILE，
        因此发布端必须同样是 BEST_EFFORT，否则话题能建立但收不到数据。
        """
        return QoSProfile(
            depth=depth,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
        )

    # ------------------------------------------------------------
    # 目标管理
    # ------------------------------------------------------------
    def set_target(self, positions: Sequence[float]) -> None:
        """更新内部目标（不立即下发，由 publish() 定时下发）"""
        self._target = [float(v) for v in positions]

    @property
    def target(self) -> Optional[List[float]]:
        return None if self._target is None else list(self._target)

    @property
    def last_sent(self) -> Optional[List[float]]:
        return None if self._last_sent is None else list(self._last_sent)

    @property
    def publish_count(self) -> int:
        return self._publish_count

    @property
    def failure_count(self) -> int:
        return self._failure_count

    def has_subscriber(self) -> bool:
        """当前是否有订阅者（用于诊断「为什么机械臂不动」）"""
        try:
            return self.pub.get_subscription_count() > 0
        except Exception:
            return False

    # ------------------------------------------------------------
    # 下发
    # ------------------------------------------------------------
    def publish(self, force: bool = False) -> bool:
        """
        按当前内部目标下发一条轨迹。

        Args:
            force: 为 True 时忽略「目标未变化」的跳过逻辑

        Returns:
            是否成功发布
        """
        if self._target is None:
            return False

        # 目标未变化且非强制时，仍然重发：
        # JointTrajectoryController 不会「记住」目标，
        # 单次下发后若不再重发，控制器会停止跟踪。
        # 因此这里保持持续重发（这是流式控制的必要条件）。
        traj = JointTrajectory()
        traj.joint_names = list(self.cfg.joint_names)

        pt = JointTrajectoryPoint()
        pt.positions = [float(v) for v in self._target]
        pt.velocities = [0.0] * self.cfg.num_joints

        # 时间窗 -> sec + nanosec（nanosec 必须 < 1e9，否则消息非法）
        ns_total = int(self.cfg.controller.trajectory_horizon_s * 1e9)
        pt.time_from_start.sec = ns_total // 1_000_000_000
        pt.time_from_start.nanosec = ns_total % 1_000_000_000

        traj.points.append(pt)

        try:
            self.pub.publish(traj)
            self._last_sent = list(self._target)
            self._publish_count += 1
            self._last_publish_time = time.monotonic()
            return True
        except Exception as exc:                      # noqa: BLE001
            self._failure_count += 1
            if self.logger is not None:
                self.logger.error(f"轨迹发布失败: {exc}")
            return False

    def seconds_since_publish(self) -> Optional[float]:
        if self._last_publish_time is None:
            return None
        return time.monotonic() - self._last_publish_time
