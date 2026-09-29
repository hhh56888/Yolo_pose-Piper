#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
只读探针 v2：记录 joint5 与**工具指向**（link6 -> link8 的物理方向），
用偏回归判定 d(工具指向角)/d(joint5) 的符号（控制 joint2/joint3/joint6）。

不发布任何指令。
"""
import math
import sys
import time

import rclpy
import tf2_ros
from rclpy.node import Node
from sensor_msgs.msg import JointState


def quat_rotate(q, v):
    x, y, z, w = q
    t = (2.0 * (y * v[2] - z * v[1]),
         2.0 * (z * v[0] - x * v[2]),
         2.0 * (x * v[1] - y * v[0]))
    return (v[0] + w * t[0] + (y * t[2] - z * t[1]),
            v[1] + w * t[1] + (z * t[0] - x * t[2]),
            v[2] + w * t[2] + (x * t[1] - y * t[0]))


class Probe(Node):
    def __init__(self):
        super().__init__("probe_j5b")
        self.buf = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buf, self)
        self.js = None
        self.create_subscription(JointState, "/joint_states", self._on_js, 20)

    def _on_js(self, msg):
        self.js = (list(msg.name), list(msg.position))

    def run(self, seconds):
        t0 = time.time()
        rows = []
        while time.time() - t0 < seconds:
            rclpy.spin_once(self, timeout_sec=0.02)
            if self.js is None:
                continue
            names, pos = self.js
            q = {n: p for n, p in zip(names, pos)}
            try:
                T = {}
                for f in ("link2", "link4", "link6", "link7", "link8",
                          "gripper_base"):
                    T[f] = self.buf.lookup_transform("base_link", f,
                                                     rclpy.time.Time())
            except Exception:                            # noqa: BLE001
                continue
            r6 = T["link6"].transform.rotation
            xax = quat_rotate([r6.x, r6.y, r6.z, r6.w], [1.0, 0.0, 0.0])
            zax = quat_rotate([r6.x, r6.y, r6.z, r6.w], [0.0, 0.0, 1.0])
            p2 = T["link2"].transform.translation
            p4 = T["link4"].transform.translation
            p6 = T["link6"].transform.translation
            p8 = T["link8"].transform.translation
            pg = T["gripper_base"].transform.translation
            rows.append((
                time.time() - t0,
                q.get("joint2", float("nan")), q.get("joint3", float("nan")),
                q.get("joint5", float("nan")), q.get("joint6", float("nan")),
                math.degrees(math.atan2(p8.z - p6.z, p8.x - p6.x)),
                math.degrees(math.atan2(pg.z - p6.z, pg.x - p6.x)),
                math.degrees(math.atan2(xax[2], xax[0])),
                math.degrees(math.atan2(zax[2], zax[0])),
                math.degrees(math.atan2(p4.z - p2.z, p4.x - p2.x)),
                p6.x, p6.z))
        return rows


HEADER = ("t,joint2,joint3,joint5,joint6,tip_l6_l8_deg,tip_l6_gb_deg,"
          "l6x_deg,l6z_deg,wrist_dir_deg,l6x_m,l6z_m")


def main():
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 45.0
    out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/probe_j5b.csv"
    rclpy.init()
    n = Probe()
    try:
        rows = n.run(seconds)
    finally:
        n.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:                                # noqa: BLE001
            pass
    with open(out, "w", encoding="utf-8") as f:
        f.write(HEADER + "\n")
        for r in rows:
            f.write(",".join("%.6f" % v for v in r) + "\n")
    print("采样 %d 行 -> %s" % (len(rows), out))


if __name__ == "__main__":
    main()
