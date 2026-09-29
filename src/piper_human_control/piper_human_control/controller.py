# -*- coding: utf-8 -*-
"""
piper_human_control.controller
==============================
PiperJointController —— 独立关节控制接口。

对外提供类似：
    robot.send_joint_target([q1, q2, q3, q4, q5, q6])
    robot.move_to_home()
    robot.get_joint_positions()

职责划分（保持解耦，对应任务书总体原则第 4 条）：
    TrajectoryStreamer  —— 怎么把指令发出去（传输层）
    SafetyLimiter       —— 什么值可以发（安全层）
    PiperJointController—— 组织上述两者 + 状态缓存 + 时钟（编排层）
    上层（阶段三/四）只与本类交互，不接触 ROS2 细节。

不修改 AgileX 官方 piper_ros 的任何文件。
"""

import math
import time
from typing import Callable, List, Optional, Sequence

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState as JointStateMsg
from trajectory_msgs.msg import JointTrajectory

from .config import ControlConfig
from .safety import EmergencyStop, SafetyLimiter
from .streaming import TrajectoryStreamer
from .types import JointState


class PiperJointController(Node):
    """
    Piper 关节控制器（ROS2 节点）。

    典型用法：
        rclpy.init()
        cfg = ControlConfig.from_yaml()
        robot = PiperJointController(cfg)
        robot.start()                       # 启动定时发布
        robot.send_joint_target([0, 0.8, -0.8, 0, 0, 0])
        robot.wait_until_reached(timeout=10)
        robot.move_to_home()
        robot.stop()
        rclpy.shutdown()
    """

    def __init__(self, cfg: ControlConfig, node_name: str = "piper_joint_controller"):
        super().__init__(node_name)
        self.cfg = cfg

        # ---------- 状态缓存 ----------
        self._state = JointState()
        self._joint_index_map = {}        # joint_name -> /joint_states 中的下标

        # ---------- 安全 ----------
        self.limiter = SafetyLimiter(cfg)
        self.estop = EmergencyStop()

        # ---------- 发布器 ----------
        self.pub = self.create_publisher(
            JointTrajectory,
            cfg.controller.trajectory_topic,
            TrajectoryStreamer.make_qos(depth=10),
        )
        self.streamer = TrajectoryStreamer(cfg, self.pub, logger=self.get_logger())

        # ---------- 状态订阅 ----------
        # /joint_states 由 joint_state_broadcaster 发布，默认 RELIABLE
        self.create_subscription(
            JointStateMsg,
            cfg.controller.joint_state_topic,
            self._on_joint_state,
            QoSProfile(depth=10,
                       reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.VOLATILE,
                       history=HistoryPolicy.KEEP_LAST),
        )

        # ---------- 时钟 ----------
        self._last_cycle_time: Optional[float] = None
        self._lost_since: Optional[float] = None
        self._warned_no_sub = False

        # ---------- 期望目标 ----------
        # _desired 是上层最终想要到达的位置；
        # streamer 里的目标则是经速率限制后、本周期真正下发的位置。
        # 两者分离，到位判断必须针对 _desired。
        self._desired: Optional[List[float]] = None
        self._desired_label: str = ""

        # ---------- 定时发布 ----------
        period = 1.0 / cfg.controller.publish_rate_hz
        self._timer = self.create_timer(period, self._publish_cycle)
        self._timer_active = True
        # 建节点时定时器即生效；若调用方希望先配置再启动，可调用 stop()
        self._timer.cancel()
        self._timer_active = False

        self.get_logger().info(
            f"PiperJointController 初始化完成 | 关节={cfg.joint_names} | "
            f"发布频率={cfg.controller.publish_rate_hz}Hz | "
            f"时间窗={cfg.controller.trajectory_horizon_s}s | "
            f"话题={cfg.controller.trajectory_topic}"
        )

    # ================================================================
    # 生命周期
    # ================================================================
    def start(self) -> None:
        """启动定时发布循环"""
        if not self._timer_active:
            period = 1.0 / self.cfg.controller.publish_rate_hz
            self._timer = self.create_timer(period, self._publish_cycle)
            self._timer_active = True
            self._last_cycle_time = None

    def stop(self) -> None:
        """停止定时发布（机械臂将保持最后目标）"""
        if self._timer_active:
            self._timer.cancel()
            self._timer_active = False

    @property
    def is_running(self) -> bool:
        return self._timer_active and not self.estop.engaged

    # ================================================================
    # 状态读取
    # ================================================================
    def _on_joint_state(self, msg: JointStateMsg) -> None:
        """缓存 /joint_states"""
        pos_map = dict(zip(msg.name, msg.position))
        vel_map = dict(zip(msg.name, msg.velocity))

        if not self._joint_index_map:
            self._joint_index_map = {n: i for i, n in enumerate(msg.name)}

        # 只取我们关心的 6 个手臂关节，顺序以 cfg.joint_names 为准
        positions, velocities = [], []
        for name in self.cfg.joint_names:
            positions.append(float(pos_map.get(name, 0.0)))
            velocities.append(float(vel_map.get(name, 0.0)))

        self._state.positions = positions
        self._state.velocities = velocities
        self._state.stamp = time.monotonic()
        self._state.received = True

    def get_joint_state(self) -> JointState:
        """返回当前关节状态快照（副本）"""
        return JointState(
            positions=list(self._state.positions),
            velocities=list(self._state.velocities),
            stamp=self._state.stamp,
            received=self._state.received,
        )

    def get_joint_positions(self) -> Optional[List[float]]:
        """
        返回当前 6 个手臂关节角 (rad)。
        尚无状态反馈时返回 None（而不是伪造 0，避免上层误判）。
        """
        if not self._state.received:
            return None
        return list(self._state.positions)

    def has_state(self) -> bool:
        return self._state.received

    def is_state_stale(self) -> bool:
        return self._state.is_stale(time.monotonic(), self.cfg.controller.state_timeout_s)

    # ================================================================
    # 目标下发（本模块的核心接口）
    # ================================================================
    def send_joint_target(
        self,
        positions: Sequence[float],
        label: str = "",
    ) -> bool:
        """
        发送关节目标（流式，平滑）。

        这是对外的主接口：
            robot.send_joint_target([q1, q2, q3, q4, q5, q6])

        与直接发布不同之处：
          * 目标经 SafetyLimiter 做限位 + 速率限制后才生效
          * 实际下发由定时器按固定时间窗持续推进，保证平滑
          * 急停状态下拒绝新目标

        Args:
            positions: 6 个关节角 (rad)，顺序 = cfg.joint_names
            label:     用途标记，便于日志排查

        Returns:
            True 表示目标已被接受（不代表已到位）
        """
        if self.estop.engaged:
            self.get_logger().warn(
                f"[急停中] 拒绝目标 {label}: {self.estop.reason}", throttle_duration_sec=2.0)
            return False

        # 第一步：只做「合法性检查 + 位置限位」，得到最终期望目标。
        # 注意这里 current=None：不做速率限制。
        # 速率限制必须放在控制循环里逐周期做，否则单次调用只会前进
        # 一个步长（见下方 _advance_target 注释）。
        report = self.limiter.apply(positions, None, 0.0, label=label)

        if report.rejected:
            self.get_logger().error(f"目标被拒绝 ({label}): {report.reason}")
            return False

        if report.clamped_joints:
            self.get_logger().warn(
                f"[限位] {label} 以下关节被裁剪: {report.clamped_joints}",
                throttle_duration_sec=2.0)

        # 第二步：记录期望目标；实际下发由 _advance_target 每周期推进
        self._desired = list(report.positions)
        self._desired_label = label
        self.streamer.set_target(self._desired)
        return True

    def get_target(self) -> Optional[List[float]]:
        """返回当前下发的关节目标（可能仍在向 desired 逼近）"""
        return self.streamer.target

    def get_desired_target(self) -> Optional[List[float]]:
        """返回上层最终期望的目标（未经速率限制）"""
        return None if self._desired is None else list(self._desired)

    def _advance_target(self, dt: float) -> None:
        """
        按速率限制把 streamer 目标朝 _desired 推进一步。

        为什么必须放在控制循环里：
            SafetyLimiter 是「基于当前实测位置」计算本周期允许的增量的。
            如果只在 send_joint_target() 里调用一次，那么一次调用最多只能
            让目标前进 min(a*dt, v*dt, step_cap) 一个步长，
            机械臂永远到不了最终目标。
            正确做法是每个控制周期都重新计算一次，逐步逼近。
        """
        if self._desired is None:
            return

        current = self.get_joint_positions()
        report = self.limiter.apply(self._desired, current, dt)

        # 限位/合法性异常时不下发（保持上一周期目标）
        if report.rejected:
            return

        self.streamer.set_target(report.positions)

    # ================================================================
    # 便捷动作
    # ================================================================
    def move_to_home(self, label: str = "home") -> bool:
        """前往配置中的 HOME 位姿"""
        return self.send_joint_target(self.limiter.limit_home(), label=label)

    def hold_current(self, label: str = "hold") -> bool:
        """
        保持当前位置（用于丢失跟踪时的安全处理）。
        注意：这是「保持」，不是「回 HOME」——避免突然的大幅运动。
        """
        cur = self.get_joint_positions()
        if cur is None:
            return False
        return self.send_joint_target(cur, label=label)

    def emergency_stop(self, reason: str = "手动触发") -> None:
        """软件急停：锁定当前目标"""
        self.estop.engage(reason)
        self.get_logger().error(f"** 软件急停触发: {reason} **")

    def reset_emergency_stop(self) -> None:
        """解除急停（需显式调用，不会自动恢复运动）"""
        self.estop.reset()
        self.get_logger().warn("急停已解除（机械臂不会自动恢复运动）")

    # ================================================================
    # 到位判断
    # ================================================================
    def reached_target(self, tol: float = 0.02) -> bool:
        """
        当前是否已到达「期望目标」。

        注意：这里必须比较 _desired（上层期望值），
        而不能比较 streamer 的内部目标 —— 后者是逐步逼近中的中间值，
        拿它做比较会导致「刚开始动就报到位」。
        """
        cur = self.get_joint_positions()
        tgt = self._desired
        if cur is None or tgt is None:
            return False
        return all(abs(c - t) <= tol for c, t in zip(cur, tgt))

    def target_error(self) -> Optional[List[float]]:
        """返回各关节 (当前 - 期望目标) 的误差列表"""
        cur = self.get_joint_positions()
        tgt = self._desired
        if cur is None or tgt is None:
            return None
        return [c - t for c, t in zip(cur, tgt)]

    def max_target_error(self) -> Optional[float]:
        """返回各关节误差绝对值的最大值"""
        err = self.target_error()
        return None if err is None else max(abs(e) for e in err)

    def wait_until_reached(
        self,
        tol: float = 0.02,
        timeout: float = 15.0,
        settle: float = 0.8,
        on_tick: Optional[Callable[["PiperJointController"], None]] = None,
    ) -> bool:
        """
        阻塞等待到位。

        先等到误差首次进入容差，再额外等待 settle 秒让其收敛稳定，
        这样返回时的误差接近稳态值，而不是刚好卡在容差边缘。

        Args:
            tol:     到位容差 (rad)
            timeout: 总超时 (s)
            settle:  进入容差后的稳定观察时长 (s)
            on_tick: 每次循环回调，便于上层打印进度

        Returns:
            是否在超时前稳定到位
        """
        t0 = time.monotonic()
        entered_at: Optional[float] = None

        while time.monotonic() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.01)
            now = time.monotonic()

            if self.reached_target(tol):
                if entered_at is None:
                    entered_at = now
                elif now - entered_at >= settle:
                    return True
            else:
                entered_at = None          # 掉出容差则重新计时

            if on_tick is not None:
                on_tick(self)

        return self.reached_target(tol)

    # ================================================================
    # 内部：定时发布循环
    # ================================================================
    def _current_dt(self) -> float:
        """距上次控制周期的时长，用于速率限制"""
        now = time.monotonic()
        if self._last_cycle_time is None:
            return 1.0 / self.cfg.controller.publish_rate_hz
        dt = now - self._last_cycle_time
        # 防止长时间停顿后 dt 过大导致一次跳变
        return max(1e-3, min(dt, 0.25))

    def _publish_cycle(self) -> None:
        """定时器回调：每个周期做一次安全处理 + 下发"""
        now = time.monotonic()
        dt = self._current_dt()

        # ---------- 急停：保持最后目标不动 ----------
        if self.estop.engaged:
            self.streamer.publish()
            self._last_cycle_time = now
            return

        # ---------- 状态丢失处理 ----------
        if self.is_state_stale():
            if self._lost_since is None:
                self._lost_since = now
            lost_for = now - self._lost_since

            # 短时间丢失：保持最后目标（不做任何新动作）
            # 长时间丢失：同样保持，但提高日志级别提醒
            if lost_for > self.cfg.safety.lost_tracking_hold_s:
                self.get_logger().warn(
                    f"/joint_states 已丢失 {lost_for:.1f}s，保持最后目标（不执行 HOME）",
                    throttle_duration_sec=2.0)
            self.streamer.publish()
            self._last_cycle_time = now
            return
        else:
            self._lost_since = None

        # ---------- 订阅者检查（只提示一次，方便定位「不动」的原因）----------
        if not self.streamer.has_subscriber() and not self._warned_no_sub:
            self.get_logger().warn(
                f"话题 {self.cfg.controller.trajectory_topic} 当前无订阅者，"
                f"指令不会生效（请确认仿真已启动）")
            self._warned_no_sub = True
        elif self.streamer.has_subscriber():
            self._warned_no_sub = False

        # ---------- 连续发布失败保护 ----------
        if self.streamer.failure_count >= self.cfg.safety.max_publish_failures:
            self.get_logger().error(
                f"连续发布失败 {self.streamer.failure_count} 次，进入安全急停")
            self.estop.engage("连续发布失败")
            self._last_cycle_time = now
            return

        # ---------- 正常下发 ----------
        # 每周期都朝期望目标推进一步（速率限制在此生效）
        self._advance_target(dt)
        self.streamer.publish()
        self._last_cycle_time = now

    def __del__(self):
        try:
            self.stop()
        except Exception:                              # noqa: BLE001
            pass


def wrap_to_pi(angle: float) -> float:
    """
    把角度归一化到 (-pi, pi]。

    阶段四计算人体关节夹角时会用到，放在此处供复用。
    """
    return (angle + math.pi) % (2.0 * math.pi) - math.pi
