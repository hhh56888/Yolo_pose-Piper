#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
J3 审计 · §5 joint3 单关节仿真测试
==================================
保持不变：joint1/2/4/5/6 = neutral
只动：joint3 = neutral + {0, -0.15, +0.15, -0.30, +0.30}
每个位姿：等稳定后采样 /joint_states(实际 q3) 与 TF(TCP 位置)。

只读 + 走既有 PiperJointController（含安全限速），不修改任何配置。
"""
import json
import math
import sys
import time

import rclpy
import tf2_ros

from piper_human_control import ControlConfig, PiperJointController
from piper_human_retargeting import RetargetingConfig

FRAMES = ("link2", "link3", "link4", "link6", "link7", "link8", "gripper_base")
DELTAS = (0.0, -0.3, -0.6, -0.9, -1.2, -1.5, +0.3, +0.6)
HOLD_S = 2.6
SAMPLE_S = 1.2


def main():
    rc = RetargetingConfig.from_yaml()
    neutral = list(rc.neutral_joints)
    names = list(rc.joint_names)
    i3 = names.index("joint3")

    rclpy.init()
    cfg = ControlConfig.from_yaml()
    robot = PiperJointController(cfg, node_name="j3_audit_probe")
    buf = tf2_ros.Buffer()
    tf2_ros.TransformListener(buf, robot)

    t0 = time.monotonic()
    while time.monotonic() - t0 < 10 and not robot.has_state():
        rclpy.spin_once(robot, timeout_sec=0.1)
    if not robot.has_state():
        print("[FAIL] 收不到 /joint_states")
        return 2
    robot.start()
    print("neutral = %s" % [round(v, 4) for v in neutral])
    print("joint3 限位来自 joint_limits.yaml: [-2.967, 0.0]；"
          "本测试只用 neutral±0.30，远离限位")

    def hold(target, seconds):
        t = time.monotonic()
        while time.monotonic() - t < seconds:
            robot.send_joint_target(target, label="j3_audit")
            rclpy.spin_once(robot, timeout_sec=0.01)

    def tf_pos(frame):
        try:
            tr = buf.lookup_transform("base_link", frame, rclpy.time.Time())
            p = tr.transform.translation
            return [p.x, p.y, p.z]
        except Exception:                                  # noqa: BLE001
            return None

    def sample(seconds):
        """窗口内平均：/joint_states 的 q3 实测 + 各 link 位置"""
        acc, n = {}, 0
        t = time.monotonic()
        while time.monotonic() - t < seconds:
            rclpy.spin_once(robot, timeout_sec=0.01)
            pos = robot.get_joint_positions()
            if pos is None or len(pos) != len(names):
                continue
            n += 1
            acc["q3"] = acc.get("q3", 0.0) + pos[i3]
            for f in FRAMES:
                p = tf_pos(f)
                if p is None:
                    continue
                acc.setdefault(f, [0.0, 0.0, 0.0])
                for k in range(3):
                    acc[f][k] += p[k]
        if n == 0:
            return None
        out = {"n": n, "q3": acc["q3"] / n}
        for f in FRAMES:
            if f in acc:
                out[f] = [v / n for v in acc[f]]
        return out

    rows = []
    # 先回到 neutral
    hold(neutral, HOLD_S)
    for d in DELTAS:
        target = list(neutral)
        target[i3] = neutral[i3] + d
        hold(target, HOLD_S)
        s = sample(SAMPLE_S)
        if s is None:
            print("[FAIL] 采样失败 dq3=%+.2f" % d)
            return 3
        s["dq3_cmd"] = d
        s["q3_cmd"] = target[i3]
        rows.append(s)
        tcp = s.get("gripper_base") or s.get("link6")
        print("dq3=%+0.2f  q3_cmd=%+.4f  q3_actual=%+.4f  "
              "link6=(%+.4f,%+.4f,%+.4f)  tcp=(%+.4f,%+.4f,%+.4f)"
              % (d, target[i3], s["q3"], *(s.get("link6") or [0, 0, 0]),
                 *(tcp or [0, 0, 0])))

    # 回到 neutral
    hold(neutral, HOLD_S)

    with open("/tmp/j3_curve.json", "w", encoding="utf-8") as f:
        json.dump({"neutral": neutral, "names": names, "rows": rows}, f,
                  ensure_ascii=False, indent=1)

    print("\n%-8s %-9s %-9s %-10s %-10s %-10s %-10s %-10s" % (
        "dq3", "q3_cmd", "q3_act", "tcp_x", "tcp_y", "tcp_z", "|tcp|", "reach24"))
    for s in rows:
        tcp = s.get("gripper_base")[0:3] if "gripper_base" in s else s["link6"]
        p2, p4 = s.get("link2"), s.get("link4")
        reach = (math.dist(p2, p4) if p2 and p4 else float("nan"))
        print("%-8.2f %-9.4f %-9.4f %-10.4f %-10.4f %-10.4f %-10.4f %-10.4f" % (
            s["dq3_cmd"], s["q3_cmd"], s["q3"],
            tcp[0], tcp[1], tcp[2], math.dist([0, 0, 0], tcp), reach))
    print("\n→ 写入 /tmp/j3_curve.json")
    robot.destroy_node()
    try:
        rclpy.shutdown()
    except Exception:                                      # noqa: BLE001
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
