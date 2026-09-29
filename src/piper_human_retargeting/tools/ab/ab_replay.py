#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
J3 审计 · §7 A/B 测试（同一段真实人体输入，仿真实测）
====================================================
输入：/tmp/live_run.csv 中一段 TRACKING 帧（真实 live 运行记录的人体角度）
      用 (raw_upper, raw_forearm) 重建 ArmKeypoints：
          肘 = 肩 + R(upper)*100        （与 test/verify_joints.py 同一构造）
          腕 = 肘 + R(fore)*100
      => 重建后的肘夹角 = |fore - upper|，与记录的 raw_elbow 同源。

对 invert=false / invert=true 各跑一遍（只改这一行），
每帧：Retargeter（生产代码）-> PiperJointController（生产代码，含安全限速）
      -> 同时读 /joint_states 实测 q3 与 TF 的 TCP 位置。

输出：/tmp/ab_replay_<mode>.csv（逐帧全链路，含 TCP）
"""
import csv
import math
import os
import shutil
import subprocess
import sys
import time

import rclpy
import tf2_ros

from piper_human_control import (ArmKeypoints, ControlConfig, Keypoint,
                                 PiperJointController)
from piper_human_retargeting import Retargeter, RetargetingConfig
from piper_human_retargeting.arm_geometry import compute_arm_geometry
from piper_human_retargeting.retargeter import HumanNeutral

SRC_CFG = os.path.expanduser("~/Yolo_pose+piper/src/piper_human_retargeting/config")
LIVE_CSV = "/tmp/live_run.csv"
_w = os.environ.get("AUDIT_WIN", "831600,831660").split(",")
WIN_LO, WIN_HI = float(_w[0]), float(_w[1])   # 默认: 探针窗口；可用 AUDIT_WIN 覆盖
HZ = 30.0
# 与 live 运行日志一致的人体中立值（原始值，符号由配置施加）
NEUTRAL_RAW = {"upper_arm_angle_deg": -82.1, "elbow_angle_deg": 126.6,
               "wrist_pitch_deg": -6.1, "thumb_offset": 0.0}


def patch_config(mode):
    dst = "/tmp/ab_cfg_%s" % mode
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(SRC_CFG, dst)
    path = os.path.join(dst, "retargeting.yaml")
    lines = open(path, encoding="utf-8").read().split("\n")
    in_j3 = False
    for i, ln in enumerate(lines):
        if ln.startswith("  j3:"):
            in_j3 = True
            continue
        if in_j3 and ln[:2] not in ("  ", "") and ln.strip():
            in_j3 = False
        if in_j3 and ln.strip().startswith("invert:"):
            lines[i] = "    invert: %s" % ("true" if mode == "true" else "false")
            break
    open(path, "w", encoding="utf-8").write("\n".join(lines))
    d = subprocess.run(["diff", "-u", os.path.join(SRC_CFG, "retargeting.yaml"),
                        path], capture_output=True, text=True).stdout
    print("[%s] 配置 diff:\n%s" % (mode, d.strip() or "(无)"))
    return path


def load_frames():
    rows = []
    with open(LIVE_CSV, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("state") != "TRACKING":
                continue
            try:
                t = float(r["timestamp"])
            except Exception:                          # noqa: BLE001
                continue
            if WIN_LO <= t <= WIN_HI:
                rows.append(r)
    return rows


def make_arm(upper_deg, fore_deg):
    ea = math.radians(upper_deg)
    e = (100.0 * math.cos(ea), -100.0 * math.sin(ea))
    fa = math.radians(fore_deg)
    w = (e[0] + 100.0 * math.cos(fa), e[1] - 100.0 * math.sin(fa))
    return ArmKeypoints(
        shoulder=Keypoint(0.0, 0.0, confidence=0.9, name="left_shoulder"),
        elbow=Keypoint(*e, confidence=0.9, name="left_elbow"),
        wrist=Keypoint(*w, confidence=0.9, name="left_wrist"),
        side="left",
        hand_wrist=Keypoint(*w, confidence=0.9, name="left_wrist_from_hand"))


def run_one(mode, cfg_path, frames):
    out_csv = os.environ.get("AUDIT_OUT", "/tmp/ab_replay_%s.csv") % mode
    rc = RetargetingConfig.from_yaml(cfg_path)
    ctrl_cfg = ControlConfig.from_yaml()
    rclpy.init()
    robot = PiperJointController(ctrl_cfg, node_name="ab_replay_%s" % mode)
    buf = tf2_ros.Buffer()
    tf2_ros.TransformListener(buf, robot)
    t0 = time.monotonic()
    while time.monotonic() - t0 < 10 and not robot.has_state():
        rclpy.spin_once(robot, timeout_sec=0.1)
    robot.start()
    rt = Retargeter(rc, ctrl_cfg)
    rt.set_initial_pose(list(rc.neutral_joints))
    rt.neutral = HumanNeutral(values=dict(NEUTRAL_RAW), samples=999,
                              signs=dict(rc.measure_sign()))
    i3 = rc.joint_names.index("joint3")
    print("[%s] 回放 %d 帧 @%.0fHz（neutral 用 live 日志值 %s）"
          % (mode, len(frames), HZ, NEUTRAL_RAW))
    dt = 1.0 / HZ
    rows_out = []
    nxt = time.monotonic()
    for i, r in enumerate(frames):
        up, fore = float(r["raw_upper"]), float(r["raw_forearm"])
        arm = make_arm(up, fore)
        geo = compute_arm_geometry(arm)
        res = rt.update(arm, now=time.monotonic(), frame=i, raw_geometry=geo)
        robot.send_joint_target(res.positions, label="ab_replay")
        rclpy.spin_once(robot, timeout_sec=0.001)
        pos = robot.get_joint_positions()
        tcp = [float("nan")] * 3
        try:
            tr = buf.lookup_transform("base_link", "gripper_base",
                                      rclpy.time.Time())
            tcp = [tr.transform.translation.x, tr.transform.translation.y,
                   tr.transform.translation.z]
        except Exception:                              # noqa: BLE001
            pass
        d = res.debug
        rows_out.append([
            i, float(r["raw_upper"]), float(r["raw_forearm"]),
            geo.elbow_angle_deg, d.flt_elbow, d.delta_elbow,
            res.mapped_joints.get("joint3", float("nan")),
            res.positions[i3],
            pos[i3] if pos and len(pos) > i3 else float("nan"),
            tcp[0], tcp[1], tcp[2]])
        nxt += dt
        sleep = nxt - time.monotonic()
        if sleep > 0:
            time.sleep(sleep)
        else:
            nxt = time.monotonic()
    robot.send_joint_target(list(rc.neutral_joints), label="ab_done")
    for _ in range(60):
        rclpy.spin_once(robot, timeout_sec=0.01)
    with open(out_csv, "w", encoding="utf-8") as fh:
        fh.write("frame,raw_upper,raw_forearm,elbow_rebuilt,flt_elbow,"
                 "delta_elbow,mapped_j3,target_j3,actual_j3,tcp_x,tcp_y,tcp_z\n")
        for row in rows_out:
            fh.write(",".join("%.6f" % v for v in row) + "\n")
    print("[%s] 写出 %s（%d 帧）" % (mode, out_csv, len(rows_out)))
    robot.destroy_node()
    try:
        rclpy.shutdown()
    except Exception:                                  # noqa: BLE001
        pass


def main():
    frames = load_frames()
    if len(frames) < 100:
        print("[FAIL] live_run.csv 里窗口内样本不足: %d" % len(frames))
        return 2
    for mode in ("false", "true"):
        run_one(mode, patch_config(mode), frames)
    print("A/B 回放完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
