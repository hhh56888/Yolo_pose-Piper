# -*- coding: utf-8 -*-
"""
piper_human_retargeting
=======================
人体手臂姿态 -> Piper 机械臂关节 映射层。

职责：
    ArmGeometry（肩/肘/有效腕）
        -> arm_geometry    计算上臂方向 / 肘夹角 / 前臂方向
        -> mapping         区间归一化映射（**唯一**的人体->关节转换入口）
        -> tracker_state   TRACKING / LOST_SHORT / LOST_LONG 状态机
        -> Retargeter      标定 + 相对角度 + 映射 + 限幅
        -> 6 关节目标

分层（保持解耦）：
    * 本包只做**数学与映射**，不依赖 ROS2。
    * 感知由 piper_human_perception 提供；
    * 关节下发、滤波与安全由 piper_human_control 提供。
    * retarget_demo 是唯一把它们串起来的节点，属于演示层。

当前 V1（2D RGB）只映射 J2 / J3 / J5；J1 / J4 / J6 保持安全初值。
将来升级 RGB-D / 3D 时，替换的是本包的几何与映射层，
上层（控制、安全、滤波）与数据结构（Keypoint 含 z 字段）均可复用。
"""

from .arm_geometry import ArmGeometry, angle_delta_deg, compute_arm_geometry
from .config import (CalibrationConfig, ControlLoopConfig, DebugConfig,
                     GripperMappingConfig, HumanConfig, RetargetingConfig,
                     RobotConfig, SafetyConfig)
from .kinematics import ChainModel, load_default_chain
from .limits import (LimitLayer, build_layers, physical_limits, safe_limits)
from .mapping import RangeMapping, map_all, map_human_to_robot
from .retargeter import HumanNeutral, Retargeter, RetargetResult
from .task_space import (HumanWristTask, IkSolution, TaskMappingConfig,
                         TaskSpaceIk, compute_human_wrist_task,
                         map_human_to_task, neutral_candidates, scan_workspace)
from .tracker_state import TrackerConfig, TrackerStateMachine, TrackState

__version__ = "0.2.0"

__all__ = [
    # 几何
    "ArmGeometry",
    "compute_arm_geometry",
    "angle_delta_deg",
    # 映射（统一入口）
    "RangeMapping",
    "map_human_to_robot",
    "map_all",
    # 配置
    "RetargetingConfig",
    "CalibrationConfig",
    "HumanConfig",
    "RobotConfig",
    "SafetyConfig",
    "ControlLoopConfig",
    "DebugConfig",
    "GripperMappingConfig",
    # 重定向
    "Retargeter",
    "RetargetResult",
    "HumanNeutral",
    # 运动学 / 限位（V1.1）
    "ChainModel",
    "load_default_chain",
    "LimitLayer",
    "physical_limits",
    "safe_limits",
    "build_layers",
    # 2D task-space（V1.1）
    "HumanWristTask",
    "compute_human_wrist_task",
    "TaskMappingConfig",
    "map_human_to_task",
    "TaskSpaceIk",
    "IkSolution",
    "scan_workspace",
    "neutral_candidates",
    # 状态机
    "TrackState",
    "TrackerConfig",
    "TrackerStateMachine",
]
