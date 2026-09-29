# -*- coding: utf-8 -*-
"""
关节手动控制节点 (joint_teleop)
================================
用于人工验证：按键盘选择关节并增减角度，实时观察 Gazebo 中的 Piper。

按键：
    q / a   选择上一个 / 下一个关节
    w / s   当前关节 + / - 步长
    h       回 HOME
    space   保持当前位置
    e       急停 / 解除急停
    x       退出

运行：
    ros2 run piper_human_control joint_teleop
"""

import select
import sys
import termios
import time
import tty

import rclpy

from .config import ControlConfig
from .controller import PiperJointController

STEP = 0.1        # 每次调整的步长 (rad)


def fmt(vals, nd=3):
    if vals is None:
        return "None"
    return "[" + ", ".join(f"{v:+.{nd}f}" for v in vals) + "]"


def get_key(timeout=0.05):
    """非阻塞读取一个按键"""
    dr, _, _ = select.select([sys.stdin], [], [], timeout)
    if dr:
        return sys.stdin.read(1)
    return None


def render(robot, cfg, sel):
    cur = robot.get_joint_positions()
    tgt = robot.get_target()
    lines = [
        "\033[2J\033[H",                       # 清屏
        "=" * 72,
        "Piper 关节手动控制",
        "=" * 72,
        f"  当前: {fmt(cur)}",
        f"  目标: {fmt(tgt)}",
        f"  限位: " + "  ".join(
            f"{cfg.joint_names[i]}[{cfg.lower_limits[cfg.joint_names[i]]:+.2f},"
            f"{cfg.upper_limits[cfg.joint_names[i]]:+.2f}]"
            for i in range(cfg.num_joints)),
        "-" * 72,
        f"  已选关节: {cfg.joint_names[sel]}  (步长 {STEP} rad)",
        f"  急停: {'是 <<<' if robot.estop.engaged else '否'}",
        "-" * 72,
        "  q/a 选关节   w/s 增减   h 回HOME   space 保持   e 急停/解除   x 退出",
        "=" * 72,
    ]
    print("\n".join(lines))


def main(args=None):
    rclpy.init(args=args)
    cfg = ControlConfig.from_yaml()
    robot = PiperJointController(cfg, node_name="piper_joint_teleop")

    # 等待状态
    print("等待 /joint_states ...")
    t0 = time.monotonic()
    while time.monotonic() - t0 < 10.0:
        rclpy.spin_once(robot, timeout_sec=0.1)
        if robot.has_state():
            break
    if not robot.has_state():
        print("收不到 /joint_states，请确认仿真已启动")
        rclpy.shutdown()
        return

    robot.start()
    # 起始目标 = 当前位姿，避免一启动就运动
    robot.send_joint_target(robot.get_joint_positions(), label="init")

    sel = 0
    old_term = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        while rclpy.ok():
            render(robot, cfg, sel)
            key = get_key(0.05)
            rclpy.spin_once(robot, timeout_sec=0.01)

            if key is None:
                continue
            if key == "x":
                break
            elif key == "q":
                sel = (sel - 1) % cfg.num_joints
            elif key == "a":
                sel = (sel + 1) % cfg.num_joints
            elif key in ("w", "s"):
                tgt = robot.get_target() or robot.get_joint_positions()
                if tgt is None:
                    continue
                q = list(tgt)
                delta = STEP if key == "w" else -STEP
                q[sel] += delta
                name = cfg.joint_names[sel]
                clipped = cfg.clip(name, q[sel])
                if abs(clipped - q[sel]) > 1e-9:
                    q[sel] = clipped
                robot.send_joint_target(q, label=f"teleop {name}")
            elif key == "h":
                robot.move_to_home()
            elif key == " ":
                robot.hold_current()
            elif key == "e":
                if robot.estop.engaged:
                    robot.reset_emergency_stop()
                else:
                    robot.emergency_stop("teleop 手动触发")
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_term)
        robot.stop()
        robot.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print("\n已退出。")


if __name__ == "__main__":
    main()
