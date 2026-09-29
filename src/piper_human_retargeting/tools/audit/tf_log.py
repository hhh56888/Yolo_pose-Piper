#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
只读 TF/关节记录器（审计用）
    python3 /tmp/tf_log.py <out.csv> <seconds>
记录：t, joint1..joint6（/joint_states 实测），
      link2/link4/link6/link8/gripper_base 的 base_link 坐标
不发布任何指令。
"""
import sys
import time

import rclpy
import tf2_ros
from rclpy.node import Node
from sensor_msgs.msg import JointState

FRAMES = ("link2", "link4", "link6", "link8", "gripper_base")
JOINTS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")


class Log(Node):
    def __init__(self):
        super().__init__("j3_tf_log")
        self.buf = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buf, self)
        self.js = None
        self.create_subscription(JointState, "/joint_states", self._on, 20)

    def _on(self, m):
        self.js = dict(zip(m.name, m.position))

    def run(self, seconds):
        t0 = time.time()
        rows = []
        while time.time() - t0 < seconds:
            rclpy.spin_once(self, timeout_sec=0.01)
            if self.js is None:
                continue
            rec = [time.time() - t0] + [self.js.get(j, float("nan"))
                                        for j in JOINTS]
            ok = True
            for f in FRAMES:
                try:
                    tr = self.buf.lookup_transform("base_link", f,
                                                   rclpy.time.Time())
                    p = tr.transform.translation
                    rec += [p.x, p.y, p.z]
                except Exception:                      # noqa: BLE001
                    ok = False
                    break
            if ok:
                rows.append(rec)
        return rows


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/tf_log.csv"
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
    rclpy.init()
    n = Log()
    try:
        rows = n.run(seconds)
    finally:
        n.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:                              # noqa: BLE001
            pass
    cols = ["t"] + list(JOINTS)
    for f in FRAMES:
        cols += [f + "_x", f + "_y", f + "_z"]
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join("%.6f" % v for v in r) + "\n")
    print("记录 %d 行 -> %s" % (len(rows), out))


if __name__ == "__main__":
    main()
