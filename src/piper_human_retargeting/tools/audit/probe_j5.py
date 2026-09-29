#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
只读探针：记录 joint5 与末端指向方向（TF 实测），用于判定
    d(末端指向角) / d(joint5)  的符号
    d(腕部方向角) / d(joint5)  与 FK 表的 -2.7°/rad 是否一致

不发布任何指令，不影响正在运行的遥操作。
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
    # 四元数旋转向量
    t = (2.0 * (y * v[2] - z * v[1]),
         2.0 * (z * v[0] - x * v[2]),
         2.0 * (x * v[1] - y * v[0]))
    return (v[0] + w * t[0] + (y * t[2] - z * t[1]),
            v[1] + w * t[1] + (z * t[0] - x * t[2]),
            v[2] + w * t[2] + (x * t[1] - y * t[0]))


class Probe(Node):
    def __init__(self):
        super().__init__("probe_j5")
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
                t6 = self.buf.lookup_transform("base_link", "link6",
                                               rclpy.time.Time())
                t2 = self.buf.lookup_transform("base_link", "link2",
                                               rclpy.time.Time())
                t4 = self.buf.lookup_transform("base_link", "link4",
                                               rclpy.time.Time())
            except Exception:                            # noqa: BLE001
                continue
            r6 = t6.transform.rotation
            xax = quat_rotate([r6.x, r6.y, r6.z, r6.w], [1.0, 0.0, 0.0])
            tip = math.degrees(math.atan2(xax[2], xax[0]))
            p2 = t2.transform.translation
            p4 = t4.transform.translation
            dx, dz = p4.x - p2.x, p4.z - p2.z
            wdir = math.degrees(math.atan2(dz, dx))
            rows.append((time.time() - t0,
                         q.get("joint2", float("nan")),
                         q.get("joint3", float("nan")),
                         q.get("joint5", float("nan")),
                         q.get("joint6", float("nan")),
                         tip, wdir, dx, dz))
        return rows


def main():
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
    out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/probe_j5.csv"
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
        f.write("t,joint2,joint3,joint5,joint6,tip_deg,wrist_dir_deg,dx,dz\n")
        for r in rows:
            f.write(",".join("%.6f" % v for v in r) + "\n")
    print("采样 %d 行 -> %s" % (len(rows), out))


if __name__ == "__main__":
    main()
