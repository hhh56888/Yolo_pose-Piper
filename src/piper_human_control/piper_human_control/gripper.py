# -*- coding: utf-8 -*-
"""
piper_human_control.gripper
===========================
Piper 夹爪控制（joint7 / joint8 双指）。

为什么单独一个类，而不是塞进 PiperJointController：
    1. **接口不同**：手臂走 `/arm_controller/joint_trajectory`，
       夹爪走 `/gripper_controller/joint_trajectory` 与
       `/gripper8_controller/joint_trajectory`，是两个独立控制器；
    2. **量纲不同**：手臂关节是旋转（rad），夹爪是移动（m），
       限位、速度、滤波参数都不同；
    3. **可独立性**：夹爪坏掉/没接不应影响手臂控制，
       反之亦然。分开后可以单独启动、单独测试。

Piper 夹爪的事实（阶段一实测确认）：
    joint7  prismatic  [0, 0.035] m   轴 (0,0,+1)   手指 A
    joint8  prismatic  [-0.035, 0] m  轴 (0,0,-1)   手指 B
    两者**轴方向相反**，因此 joint8 = -joint7 才能同步张开/闭合。
    官方 `joint8_ctrl.py` 做的就是这件事（订阅 joint7 状态 -> 反向发布 joint8）。

本类的做法：
    直接同时下发 joint7 与 joint8，不依赖官方的镜像节点，
    这样夹爪的开合完全由我们自己的配置决定（例如将来支持
    「只动一根手指」或「夹持力控制」时不会被镜像节点干扰）。
"""

import time
from dataclasses import dataclass
from typing import Optional

# 注意：这里**不能**在模块顶层 import rclpy。
#   GripperConfig / openness_to_joint7 是纯逻辑，
#   需要能在没有 ROS2 的机器上用于单元测试与配置校验。
#   rclpy 与消息类型只在 GripperController 真正实例化时才导入。


# ============================================================
# 夹爪配置
# ============================================================
@dataclass
class GripperConfig:
    """
    夹爪配置。

    Attributes:
        joint_open:   完全张开时的 joint7 位置 (m)，通常 0.035
        joint_closed: 完全闭合时的 joint7 位置 (m)，通常 0.0
        min_openness: openness 低于此值视为完全闭合（避免噪声抖动）
        max_openness: openness 高于此值视为完全张开
        trajectory_horizon_s: 轨迹时窗
        publish_rate_hz:      重发频率
        deadband_m:           目标变化小于此值不更新（抑制夹爪微动）
    """
    joint7_open: float = 0.035
    joint7_closed: float = 0.0
    min_openness: float = 0.15
    max_openness: float = 0.85
    trajectory_horizon_s: float = 0.15
    publish_rate_hz: float = 20.0
    deadband_m: float = 0.0008      # 0.8 mm


def openness_to_joint7(openness: float, cfg: GripperConfig) -> float:
    """
    归一化张开度 -> joint7 位置。

    Args:
        openness: [0, 1]。0 = 握紧，1 = 完全张开
        cfg:      夹爪配置

    Returns:
        joint7 位置 (m)，已 clamp 到 [joint7_closed, joint7_open]

    归一化处理：
        先把 openness 从 [min_openness, max_openness] 重映射到 [0, 1]，
        再线性映射到关节行程。这样做的目的是**充分利用行程**：
        实测人手张开度很难稳定达到 0 或 1，
        若直接用 [0,1] 映射，夹爪永远走不满行程。
    """
    lo, hi = cfg.min_openness, cfg.max_openness
    if hi <= lo:
        raise ValueError(f"夹爪配置错误: max_openness({hi}) 必须大于 min_openness({lo})")

    r = (float(openness) - lo) / (hi - lo)
    r = max(0.0, min(1.0, r))

    j7 = cfg.joint7_closed + r * (cfg.joint7_open - cfg.joint7_closed)
    return max(cfg.joint7_closed, min(cfg.joint7_open, j7))


# ============================================================
# 夹爪控制器
# ============================================================
class GripperController:
    """
    Piper 夹爪控制器。

    用法：
        g = GripperController(node, cfg)
        g.set_openness(0.8)      # 张开 80%
        g.set_openness(0.0)      # 握紧
        g.open() / g.close() / g.hold()

    注意：本类**不继承 Node**，而是复用传入的节点来创建发布器，
    这样整个程序只有一个 ROS 节点，避免节点爆炸。
    """

    TOPIC_JOINT7 = "/gripper_controller/joint_trajectory"
    TOPIC_JOINT8 = "/gripper8_controller/joint_trajectory"

    def __init__(self, node, cfg: Optional[GripperConfig] = None,
                 logger=None):
        # 惰性导入：只有真正要控制夹爪时才需要 ROS2
        from rclpy.node import Node as _Node            # noqa: F401
        from trajectory_msgs.msg import (JointTrajectory,
                                         JointTrajectoryPoint)
        from .streaming import TrajectoryStreamer

        self._JointTrajectory = JointTrajectory
        self._JTP = JointTrajectoryPoint
        self._TrajectoryStreamer = TrajectoryStreamer

        self.node = node
        self.cfg = cfg or GripperConfig()
        self.logger = logger or node.get_logger()

        qos = self._TrajectoryStreamer.make_qos(depth=10)
        self.pub7 = node.create_publisher(self._JointTrajectory,
                                          self.TOPIC_JOINT7, qos)
        self.pub8 = node.create_publisher(self._JointTrajectory,
                                          self.TOPIC_JOINT8, qos)

        # 当前目标（joint7 位置）与滤波状态
        self._target_j7: float = self.cfg.joint7_closed
        self._last_cmd_j7: float = self.cfg.joint7_closed
        self._openness: float = 0.0

        self._publish_count = 0
        self._last_publish_time: Optional[float] = None

    # ------------------------------------------------------------
    # 目标设置
    # ------------------------------------------------------------
    def set_openness(self, openness: float) -> bool:
        """
        设置归一化张开度。

        Args:
            openness: [0,1]，0=握紧，1=完全张开

        Returns:
            是否产生了实际更新（受 deadband 影响）
        """
        j7 = openness_to_joint7(openness, self.cfg)
        self._openness = float(openness)

        # 死区：抑制夹爪微动
        if abs(j7 - self._target_j7) < self.cfg.deadband_m:
            return False
        self._target_j7 = j7
        return True

    def set_joint7(self, value: float) -> None:
        """直接设置 joint7 位置（m），供手动测试用"""
        self._target_j7 = max(self.cfg.joint7_closed,
                              min(self.cfg.joint7_open, float(value)))

    def open(self) -> None:
        """完全张开"""
        self.set_openness(1.0)

    def close(self) -> None:
        """完全握紧"""
        self.set_openness(0.0)

    # ------------------------------------------------------------
    # 下发
    # ------------------------------------------------------------
    def _make_msg(self, joint_name: str, position: float):
        JointTrajectory = self._JointTrajectory
        JointTrajectoryPoint = self._JTP
        traj = JointTrajectory()
        traj.joint_names = [joint_name]
        pt = JointTrajectoryPoint()
        pt.positions = [float(position)]
        pt.velocities = [0.0]
        ns = int(self.cfg.trajectory_horizon_s * 1e9)
        pt.time_from_start.sec = ns // 1_000_000_000
        pt.time_from_start.nanosec = ns % 1_000_000_000
        traj.points.append(pt)
        return traj

    def publish(self) -> bool:
        """
        下发当前目标。

        joint8 必须取 **-joint7**：两个关节轴方向相反
        （joint7 轴 (0,0,+1)，joint8 轴 (0,0,-1)），
        且限位互为镜像 [0,0.035] 与 [-0.035,0]。
        """
        j7 = self._target_j7
        j8 = -j7
        ok = True
        try:
            self.pub7.publish(self._make_msg("joint7", j7))
            self.pub8.publish(self._make_msg("joint8", j8))
            self._last_cmd_j7 = j7
            self._publish_count += 1
            self._last_publish_time = time.monotonic()
        except Exception as exc:                          # noqa: BLE001
            ok = False
            self.logger.error(f"夹爪下发失败: {exc}")
        return ok

    # ------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------
    @property
    def target_joint7(self) -> float:
        return self._target_j7

    @property
    def openness(self) -> float:
        return self._openness

    @property
    def publish_count(self) -> int:
        return self._publish_count

    def has_subscriber(self) -> bool:
        """夹爪控制器是否在线（用于诊断「夹爪为什么不动」）"""
        try:
            return (self.pub7.get_subscription_count() > 0
                    or self.pub8.get_subscription_count() > 0)
        except Exception:                                  # noqa: BLE001
            return False

    def describe(self) -> str:
        return (f"夹爪 joint7[{self.cfg.joint7_closed}, {self.cfg.joint7_open}]m, "
                f"openness[{self.cfg.min_openness}, {self.cfg.max_openness}], "
                f"死区 {self.cfg.deadband_m*1000:.1f}mm")
