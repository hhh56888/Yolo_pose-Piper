#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V1.1 A/B 实测运行器（legacy vs task_space，同一段人体输入）
=========================================================
两种输入：
    --input synth : 四个基础动作（前 / 后 / 上 / 下），单轴、干净
    --input real  : /tmp/live_run.csv 里真实记录的人体角度序列（重建关键点）

两种模式：
    --mode legacy | task_space（只改 retarget_mode 一行，其余配置不动）

每帧同时记录：人体 u/v、du/dv、目标 (x,z)、IK 解、**实测关节角**与
**TF 实测 TCP**（gripper_base），写入 /tmp/abv11_<mode>_<input>.csv。

只走既有生产代码（Retargeter + PiperJointController），不修改任何配置。
"""
import argparse
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
REAL_WIN = (831794.3, 831884.3)        # 肘部行程最大的一段真实数据（90 s）
HZ = 30.0

# 合成动作：肩在原点，L1=L2=100px；腕位置以归一化 u/v 给出
BASE_U, BASE_V = 0.55, 0.30
ACTIONS = [
    # (名称, 目标 u, 目标 v, 段时长 s)
    ("fwd", BASE_U + 0.20, BASE_V, 3.0),
    ("hold_fwd", BASE_U + 0.20, BASE_V, 1.0),
    ("back_to_base", BASE_U, BASE_V, 2.0),
    ("settle1", BASE_U, BASE_V, 2.0),
    ("back", BASE_U - 0.20, BASE_V, 3.0),
    ("hold_back", BASE_U - 0.20, BASE_V, 1.0),
    ("back_to_base2", BASE_U, BASE_V, 2.0),
    ("settle2", BASE_U, BASE_V, 2.0),
    ("up", BASE_U, BASE_V + 0.30, 3.0),
    ("hold_up", BASE_U, BASE_V + 0.30, 1.0),
    ("back_to_base3", BASE_U, BASE_V, 2.0),
    ("settle3", BASE_U, BASE_V, 2.0),
    ("down", BASE_U, BASE_V - 0.25, 3.0),
    ("hold_down", BASE_U, BASE_V - 0.25, 1.0),
    ("back_to_base4", BASE_U, BASE_V, 2.0),
    ("settle4", BASE_U, BASE_V, 2.0),
]


def make_arm_from_uv(u, v, L1=100.0, L2=100.0, branch=+1):
    """
    由归一化腕位置 (u,v) 反解两连杆，得到一组**刚性**手臂关键点。

        u = wx/(L1+L2), v = -wy/(L1+L2)
    肘取 branch 指定的一支（+1 = 肘在下），|SE|、|EW| 恒为 L1/L2。
    """
    wx, wy = u * (L1 + L2), -v * (L1 + L2)
    d = math.hypot(wx, wy)
    d = max(abs(L1 - L2) + 1e-6, min(L1 + L2 - 1e-6, d))
    ang = math.atan2(-wy, wx)                       # 数学角（y 向上）
    cosb = (d * d + L1 * L1 - L2 * L2) / (2 * d * L1)
    beta = math.acos(max(-1.0, min(1.0, cosb)))
    th1 = ang + branch * beta
    ex, ey = L1 * math.cos(th1), -L1 * math.sin(th1)
    # 腕点由目标反算（避免浮点误差）
    wr = math.hypot(wx - ex, wy - ey)
    th2 = math.atan2(-(wy - ey), wx - ex)
    fx, fy = ex + L2 * math.cos(th2), ey - L2 * math.sin(th2)
    del wr, fy
    w = (fx, ey - L2 * math.sin(th2))
    w = (wx, wy)
    return ArmKeypoints(
        shoulder=Keypoint(0.0, 0.0, confidence=0.9, name="left_shoulder"),
        elbow=Keypoint(ex, ey, confidence=0.9, name="left_elbow"),
        wrist=Keypoint(*w, confidence=0.9, name="left_wrist"),
        side="left",
        hand_wrist=Keypoint(*w, confidence=0.9,
                            name="left_wrist_from_hand"))


def synth_frames():
    """四个基础动作：每帧 (u, v) 线性插值"""
    frames, cur = [], (BASE_U, BASE_V)
    for name, tu, tv, secs in ACTIONS:
        n = max(1, int(secs * HZ))
        for k in range(1, n + 1):
            a = k / n
            frames.append((name, cur[0] + (tu - cur[0]) * a,
                           cur[1] + (tv - cur[1]) * a))
        cur = (tu, tv)
    return frames


def real_frames():
    rows = []
    with open(LIVE_CSV, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("state") != "TRACKING":
                continue
            try:
                t = float(r["timestamp"])
                up, fore = float(r["raw_upper"]), float(r["raw_forearm"])
            except Exception:                              # noqa: BLE001
                continue
            if REAL_WIN[0] <= t <= REAL_WIN[1]:
                rows.append((up, fore))
    return rows


def make_arm_angles(upper_deg, fore_deg):
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


def patch_config(mode, lam=None):
    dst = "/tmp/abv11_cfg_%s" % mode
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(SRC_CFG, dst)
    # 配置里的 data/*.json 是相对「配置目录」解析的，临时目录必须带上 data/
    data_src = os.path.join(os.path.dirname(SRC_CFG), "piper_human_retargeting",
                            "data")
    if os.path.isdir(data_src):
        shutil.copytree(data_src, os.path.join(dst, "data"))
    path = os.path.join(dst, "retargeting.yaml")
    lines = open(path, encoding="utf-8").read().split("\n")
    for i, ln in enumerate(lines):
        if ln.startswith("retarget_mode:"):
            lines[i] = 'retarget_mode: "%s"' % mode
        if lam is not None and ln.strip().startswith("lambda_prev:"):
            lines[i] = "  lambda_prev: %s" % lam
    open(path, "w", encoding="utf-8").write("\n".join(lines))
    d = subprocess.run(["diff", "-u", os.path.join(SRC_CFG, "retargeting.yaml"),
                        path], capture_output=True, text=True).stdout
    print("[%s] 配置 diff:\n%s" % (mode, d.strip() or "(无)"))
    return path


def run(mode, kind, out_csv, lam=None):
    rc = RetargetingConfig.from_yaml(patch_config(mode, lam))
    if lam is not None:
        print("[%s] lambda_prev 覆盖为 %s" % (mode, rc.task_space.lambda_prev))
    neutral = list(rc.active_neutral_joints())
    rclpy.init()
    robot = PiperJointController(ControlConfig.from_yaml(),
                                 node_name="abv11_%s_%s" % (mode, kind))
    buf = tf2_ros.Buffer()
    tf2_ros.TransformListener(buf, robot)
    t0 = time.monotonic()
    while time.monotonic() - t0 < 10 and not robot.has_state():
        rclpy.spin_once(robot, timeout_sec=0.1)
    robot.start()

    def hold(target, seconds):
        t = time.monotonic()
        while time.monotonic() - t < seconds:
            robot.send_joint_target(target, label="abv11")
            rclpy.spin_once(robot, timeout_sec=0.01)

    hold(neutral, 4.0)          # 先到本模式的 neutral（两种模式各自的初始位姿）

    rt = Retargeter(rc, ControlConfig.from_yaml())
    rt.set_initial_pose(neutral)
    # 标定：用 base 姿势（两种输入共用同一零点定义）
    if kind == "synth":
        cal_arm = make_arm_from_uv(BASE_U, BASE_V)
    else:
        cal_arm = make_arm_angles(0.0, 90.0)
    cal_geo = compute_arm_geometry(cal_arm)
    rt.start_calibration()
    for _ in range(30):
        rt.add_calibration_sample_for(cal_arm, cal_geo)
    ok = rt.finish_calibration()
    print("[%s/%s] neutral=%s 标定=%s" % (mode, kind,
                                          [round(v, 3) for v in neutral], ok))

    frames = synth_frames() if kind == "synth" else real_frames()
    print("[%s/%s] 回放 %d 帧" % (mode, kind, len(frames)))
    i3 = rc.joint_names.index("joint3")
    dt, nxt = 1.0 / HZ, time.monotonic()
    rows = []
    for i, item in enumerate(frames):
        if kind == "synth":
            label, u, v = item
            arm = make_arm_from_uv(u, v)
        else:
            label, arm = "real", make_arm_angles(item[0], item[1])
        geo = compute_arm_geometry(arm)
        res = rt.update(arm, now=time.monotonic(), frame=i, raw_geometry=geo)
        robot.send_joint_target(res.positions, label="abv11")
        rclpy.spin_once(robot, timeout_sec=0.001)
        pos = robot.get_joint_positions()
        tcp = [float("nan")] * 3
        try:
            tr = buf.lookup_transform("base_link", "gripper_base",
                                      rclpy.time.Time())
            tcp = [tr.transform.translation.x, tr.transform.translation.y,
                   tr.transform.translation.z]
        except Exception:                                  # noqa: BLE001
            pass
        d = res.debug
        rows.append([
            i, label, geo.human_u, geo.human_v,
            d.human_u_flt, d.human_v_flt, d.du, d.dv,
            d.tcp_target_x, d.tcp_target_z, d.ik_q2, d.ik_q3,
            d.ik_jump, d.ik_time_ms, d.reach_raw, geo.elbow_angle_deg,
            res.positions[1], res.positions[2], res.positions[i3],
            pos[1] if pos else float("nan"), pos[2] if pos else float("nan"),
            tcp[0], tcp[1], tcp[2],
            math.hypot(tcp[0], tcp[2]) if tcp[0] == tcp[0] else float("nan"),
        ])
        nxt += dt
        sl = nxt - time.monotonic()
        if sl > 0:
            time.sleep(sl)
        else:
            nxt = time.monotonic()

    hold(neutral, 1.0)
    with open(out_csv, "w", encoding="utf-8") as fh:
        fh.write("frame,label,u,v,u_flt,v_flt,du,dv,tgt_x,tgt_z,ik_q2,ik_q3,"
                 "ik_jump,ik_ms,reach,elbow,j2_cmd,j3_cmd,j3i_cmd,j2_act,"
                 "j3_act,tcp_x,tcp_y,tcp_z,tcp_r\n")
        for r in rows:
            fh.write(",".join(
                (r[1] if isinstance(r[1], str) else "%.6f" % r[1])
                if j == 1 else
                (("%.6f" % r[j]) if isinstance(r[j], float) else str(r[j]))
                for j in range(len(r))) + "\n")
    print("[%s/%s] 写出 %s（%d 帧）" % (mode, kind, out_csv, len(rows)))
    st = rt.ik_stats()
    if st.get("n"):
        print("   IK: n=%d 中位 %.3f ms P95 %.3f ms 峰值 %.3f ms 未收敛 %d"
              % (st["n"], st["median_ms"], st["p95_ms"], st["max_ms"],
                 st["failures"]))
    robot.destroy_node()
    try:
        rclpy.shutdown()
    except Exception:                                      # noqa: BLE001
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("legacy", "task_space"), required=True)
    ap.add_argument("--input", choices=("synth", "real"), required=True)
    ap.add_argument("--lambda-prev", type=float, default=None)
    ap.add_argument("--tag", type=str, default="")
    a = ap.parse_args()
    out = "/tmp/abv11_%s_%s%s.csv" % (a.mode, a.input, a.tag)
    run(a.mode, a.input, out, a.lambda_prev)
    return 0


if __name__ == "__main__":
    sys.exit(main())
