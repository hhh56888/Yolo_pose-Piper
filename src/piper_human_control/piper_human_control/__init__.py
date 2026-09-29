# -*- coding: utf-8 -*-
"""
piper_human_control
===================
Piper 机械臂独立关节控制接口（阶段二）。

对外主要接口：
    from piper_human_control import ControlConfig, PiperJointController

    cfg   = ControlConfig.from_yaml()
    robot = PiperJointController(cfg)
    robot.start()
    robot.send_joint_target([q1, q2, q3, q4, q5, q6])

模块划分（感知 / 计算 / 控制解耦）：
    types.py      数据结构（Keypoint / JointTarget / JointState）
    config.py     配置加载与校验
    safety.py     限位、速率限制、合法性检查、急停
    streaming.py  轨迹流式发送
    controller.py 编排层，对外统一接口

后续阶段将新增（不改动本层）：
    pose/          阶段三：YOLO Pose 关键点提取
    retargeting/   阶段四：人体姿态 -> 关节角映射
    perception/    阶段七：RGB-D 深度接入
"""

from .config import ControlConfig, ControllerConfig, LoggingConfig, SafetyConfig
from .filter import (AngleFilter, FilterChainConfig, FilterConfig,
                     JointFilter, KeypointFilter, ScalarFilter)
# gripper 里 GripperConfig / openness_to_joint7 是纯逻辑（无 rclpy），
# 而 GripperController 需要 rclpy。因此这里只导入纯逻辑部分，
# GripperController 放到下面的惰性导入里 ——
# 保证「映射配置类」在没有 ROS2 的机器上也能用于单元测试。
from .gripper import GripperConfig, openness_to_joint7
from .safety import EmergencyStop, SafetyLimiter
from .types import (FINGER_NAMES, ArmKeypoints, HandKeypoints, JointState,
                    JointTarget, Keypoint, SafetyReport)

# controller / streaming / gripper 控制器依赖 rclpy。
# 这里做「惰性导入」：让 config / safety / types / filter 这些纯逻辑模块
# 可以在没有 ROS2 环境的机器上单独导入与单元测试。
try:
    from .controller import PiperJointController, wrap_to_pi
    from .gripper import GripperController
    from .streaming import TrajectoryStreamer

    _ROS_AVAILABLE = True
except ImportError:                                    # pragma: no cover
    PiperJointController = None                         # type: ignore
    GripperController = None                            # type: ignore
    TrajectoryStreamer = None                           # type: ignore
    _ROS_AVAILABLE = False

    def wrap_to_pi(angle: float) -> float:
        """rclpy 不可用时的降级实现（纯数学，无依赖）"""
        import math
        return (angle + math.pi) % (2.0 * math.pi) - math.pi


ROS_AVAILABLE = _ROS_AVAILABLE

__version__ = "0.1.0"

__all__ = [
    # 配置
    "ControlConfig",
    "ControllerConfig",
    "SafetyConfig",
    "LoggingConfig",
    # 控制
    "PiperJointController",
    "TrajectoryStreamer",
    "SafetyLimiter",
    "EmergencyStop",
    "wrap_to_pi",
    # 数据结构
    "Keypoint",
    "ArmKeypoints",
    "HandKeypoints",
    "FINGER_NAMES",
    "JointTarget",
    "JointState",
    "SafetyReport",
    # 滤波
    "FilterConfig",
    "FilterChainConfig",
    "ScalarFilter",
    "KeypointFilter",
    "AngleFilter",
    "JointFilter",
    # 夹爪
    "GripperConfig",
    "GripperController",
    "openness_to_joint7",
    # 环境标识
    "ROS_AVAILABLE",
]
