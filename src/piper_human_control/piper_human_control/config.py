# -*- coding: utf-8 -*-
"""
piper_human_control.config
==========================
配置加载模块。

所有控制参数（限位、速度、加速度、HOME 位姿、话题名、安全阈值）
统一从 config/joint_limits.yaml 读取，不在代码中硬编码。

用法：
    cfg = ControlConfig.from_yaml()                  # 自动查找包内配置
    cfg = ControlConfig.from_yaml("/path/to.yaml")   # 显式指定
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import yaml

try:
    from ament_index_python.packages import get_package_share_directory
    _HAS_AMENT_INDEX = True
except ImportError:      # 允许在没有 ROS2 环境时做纯配置解析测试
    _HAS_AMENT_INDEX = False


# ============================================================
# 子配置结构
# ============================================================
@dataclass
class ControllerConfig:
    """底层控制接口参数"""
    update_rate_hz: float = 500.0
    publish_rate_hz: float = 20.0
    trajectory_horizon_s: float = 0.1
    trajectory_topic: str = "/arm_controller/joint_trajectory"
    joint_state_topic: str = "/joint_states"
    state_timeout_s: float = 0.5


@dataclass
class SafetyConfig:
    """安全策略参数"""
    max_initial_error_rad: float = 3.2
    max_step_per_cycle_rad: float = 0.15
    max_publish_failures: int = 10
    lost_tracking_hold_s: float = 1.0


@dataclass
class LoggingConfig:
    """日志参数"""
    debug_every_n_publishes: int = 0


# ============================================================
# 主配置
# ============================================================
@dataclass
class ControlConfig:
    """
    Piper 关节控制完整配置。

    之所以用 dataclass 而不是直接透传 dict：
    可以在加载时立即校验（例如上下限顺序、长度一致性），
    把配置错误暴露在启动阶段，而不是运行到一半才崩。
    """
    joint_names: List[str] = field(default_factory=list)
    lower_limits: Dict[str, float] = field(default_factory=dict)
    upper_limits: Dict[str, float] = field(default_factory=dict)
    velocity_limits: Dict[str, float] = field(default_factory=dict)
    acceleration_limits: Dict[str, float] = field(default_factory=dict)
    home: List[float] = field(default_factory=list)

    controller: ControllerConfig = field(default_factory=ControllerConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    source_path: str = ""

    # ---------------- 加载 ----------------
    @classmethod
    def from_yaml(cls, path: Optional[str] = None) -> "ControlConfig":
        """
        从 YAML 加载配置。

        Args:
            path: 显式路径；为 None 时自动查找 piper_human_control/config/joint_limits.yaml
        """
        if path is None:
            path = cls.default_yaml_path()

        if not os.path.isfile(path):
            raise FileNotFoundError(f"配置文件不存在: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        cfg = cls._from_dict(raw)
        cfg.source_path = path
        cfg.validate()
        return cfg

    @staticmethod
    def default_yaml_path() -> str:
        """定位包内默认配置文件"""
        if _HAS_AMENT_INDEX:
            share = get_package_share_directory("piper_human_control")
            candidate = os.path.join(share, "config", "joint_limits.yaml")
            if os.path.isfile(candidate):
                return candidate

        # 回退：从本文件位置向上找 config/joint_limits.yaml
        # 便于在没有 colcon install 的源码树里直接运行
        here = os.path.dirname(os.path.abspath(__file__))
        pkg_root = os.path.dirname(here)
        candidate = os.path.join(pkg_root, "config", "joint_limits.yaml")
        if os.path.isfile(candidate):
            return candidate

        raise FileNotFoundError("找不到 joint_limits.yaml")

    @classmethod
    def _from_dict(cls, raw: dict) -> "ControlConfig":
        joints = raw.get("joints", {})
        names = list(joints.get("names", []))

        limits = joints.get("limits", {})
        lower, upper = {}, {}
        for j in names:
            entry = limits.get(j, {})
            lower[j] = float(entry.get("lower", 0.0))
            upper[j] = float(entry.get("upper", 0.0))

        ctrl_raw = raw.get("controller", {})
        safe_raw = raw.get("safety", {})
        log_raw = raw.get("logging", {})

        return cls(
            joint_names=names,
            lower_limits=lower,
            upper_limits=upper,
            velocity_limits={k: float(v) for k, v in
                             joints.get("velocity_limits", {}).items()},
            acceleration_limits={k: float(v) for k, v in
                                 joints.get("acceleration_limits", {}).items()},
            home=[float(v) for v in raw.get("home", [])],
            controller=ControllerConfig(
                update_rate_hz=float(ctrl_raw.get("update_rate_hz", 500.0)),
                publish_rate_hz=float(ctrl_raw.get("publish_rate_hz", 20.0)),
                trajectory_horizon_s=float(ctrl_raw.get("trajectory_horizon_s", 0.1)),
                trajectory_topic=str(ctrl_raw.get("trajectory_topic",
                                                  "/arm_controller/joint_trajectory")),
                joint_state_topic=str(ctrl_raw.get("joint_state_topic", "/joint_states")),
                state_timeout_s=float(ctrl_raw.get("state_timeout_s", 0.5)),
            ),
            safety=SafetyConfig(
                max_initial_error_rad=float(safe_raw.get("max_initial_error_rad", 3.2)),
                max_step_per_cycle_rad=float(safe_raw.get("max_step_per_cycle_rad", 0.15)),
                max_publish_failures=int(safe_raw.get("max_publish_failures", 10)),
                lost_tracking_hold_s=float(safe_raw.get("lost_tracking_hold_s", 1.0)),
            ),
            logging=LoggingConfig(
                debug_every_n_publishes=int(log_raw.get("debug_every_n_publishes", 0)),
            ),
        )

    # ---------------- 校验 ----------------
    def validate(self) -> None:
        """
        配置合法性校验。启动即失败，优于运行中出错。
        """
        if not self.joint_names:
            raise ValueError("配置错误: joints.names 为空")

        for j in self.joint_names:
            if j not in self.lower_limits or j not in self.upper_limits:
                raise ValueError(f"配置错误: 关节 {j} 缺少 limits 定义")
            if self.lower_limits[j] >= self.upper_limits[j]:
                raise ValueError(
                    f"配置错误: 关节 {j} 下限 {self.lower_limits[j]} "
                    f">= 上限 {self.upper_limits[j]}")

        if len(self.home) != len(self.joint_names):
            raise ValueError(
                f"配置错误: home 长度 {len(self.home)} "
                f"!= 关节数 {len(self.joint_names)}")

        for i, j in enumerate(self.joint_names):
            if not (self.lower_limits[j] <= self.home[i] <= self.upper_limits[j]):
                raise ValueError(
                    f"配置错误: home[{i}]={self.home[i]} 超出关节 {j} "
                    f"范围 [{self.lower_limits[j]}, {self.upper_limits[j]}]")

        if self.controller.publish_rate_hz <= 0:
            raise ValueError("配置错误: publish_rate_hz 必须 > 0")
        if self.controller.trajectory_horizon_s <= 0:
            raise ValueError("配置错误: trajectory_horizon_s 必须 > 0")

        # 速度/加速度缺失时给出默认，但发出明确提示（不静默）
        missing_v = [j for j in self.joint_names if j not in self.velocity_limits]
        missing_a = [j for j in self.joint_names if j not in self.acceleration_limits]
        if missing_v:
            raise ValueError(f"配置错误: 关节 {missing_v} 缺少 velocity_limits")
        if missing_a:
            raise ValueError(f"配置错误: 关节 {missing_a} 缺少 acceleration_limits")

    # ---------------- 便捷访问 ----------------
    @property
    def num_joints(self) -> int:
        return len(self.joint_names)

    def index_of(self, joint_name: str) -> int:
        return self.joint_names.index(joint_name)

    def clip(self, joint_name: str, value: float) -> float:
        """把单个关节值裁剪到限位内"""
        lo = self.lower_limits[joint_name]
        hi = self.upper_limits[joint_name]
        return max(lo, min(hi, value))

    def __str__(self) -> str:
        lines = [
            "ControlConfig(",
            f"  source={self.source_path}",
            f"  joints={self.joint_names}",
            f"  home={[round(v, 4) for v in self.home]}",
            f"  publish_rate={self.controller.publish_rate_hz} Hz",
            f"  horizon={self.controller.trajectory_horizon_s} s",
            f"  traj_topic={self.controller.trajectory_topic}",
            ")",
        ]
        return "\n".join(lines)
