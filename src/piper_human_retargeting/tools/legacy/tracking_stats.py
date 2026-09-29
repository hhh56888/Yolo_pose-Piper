# -*- coding: utf-8 -*-
"""
阶段三验收量化：手臂三点跟踪稳定性
评估项：
  1. 三点齐全率
  2. 连续跟踪段（连续多少帧不丢失）
  3. 关键点抖动（相邻帧位移，需按臂长归一化）
  4. 肘夹角连续性
"""
import sys, math
sys.path.insert(0, "/home/zxmy/Yolo_pose+piper/src/piper_human_perception")
sys.path.insert(0, "/home/zxmy/Yolo_pose+piper/src/piper_human_control")

import numpy as np
from piper_human_perception import (YOLOPoseProvider, VideoFileSource,
                                    compute_arm_metrics)

IMGSZ, CONF = 960, 0.4
p = YOLOPoseProvider("/home/zxmy/Yolo_pose+piper/models/yolo11n-pose.pt",
                     conf_threshold=CONF, target_side="right",
                     device="0", imgsz=IMGSZ)
p.load()
s = VideoFileSource("/tmp/arm_test.mp4"); s.open()

valid = []          # 每帧是否完整
pts = []            # (S, E, W) 坐标
angles = []         # 肘夹角
n = 0
while True:
    f = s.read()
    if f is None:
        break
    n += 1
    d = p.detect(f)
    if d.arm_complete:
        valid.append(True)
        a = d.arm
        pts.append((np.array([a.shoulder.x, a.shoulder.y]),
                    np.array([a.elbow.x, a.elbow.y]),
                    np.array([a.wrist.x, a.wrist.y])))
        try:
            angles.append(float(compute_arm_metrics(a)["肘夹角"].rstrip("°")))
        except Exception:
            angles.append(np.nan)
    else:
        valid.append(False)
s.release()

valid = np.array(valid)
nvalid = int(valid.sum())

# 连续跟踪段
runs, cur = [], 0
for v in valid:
    if v: cur += 1
    else:
        if cur: runs.append(cur)
        cur = 0
if cur: runs.append(cur)

print("=" * 62)
print("阶段三验收 · 手臂三点跟踪稳定性")
print("=" * 62)
print("配置            : imgsz=%d, conf=%.1f, yolo11n-pose" % (IMGSZ, CONF))
print("总帧数          : %d" % n)
print("三点齐全帧数    : %d  (%.1f%%)" % (nvalid, 100.0*nvalid/n))
if runs:
    print("连续跟踪段数    : %d" % len(runs))
    print("最长连续跟踪    : %d 帧" % max(runs))
    print("平均连续跟踪    : %.1f 帧" % np.mean(runs))

# 抖动：相邻「都有效」的帧之间，各关键点位移（像素）
if len(pts) > 1:
    # 只在帧号连续时算抖动，避免把跨跳跃当成抖动
    dS, dE, dW, arm_len = [], [], [], []
    for i in range(1, len(pts)):
        s0, e0, w0 = pts[i-1]
        s1, e1, w1 = pts[i]
        dS.append(np.linalg.norm(s1-s0))
        dE.append(np.linalg.norm(e1-e0))
        dW.append(np.linalg.norm(w1-w0))
        arm_len.append(np.linalg.norm(e1-s1) + np.linalg.norm(w1-e1))
    med_arm = float(np.median(arm_len))
    print("-" * 62)
    print("平均臂长(像素)  : %.0f" % med_arm)
    print("关键点抖动(中位数/90分位, 像素, 归一化为臂长百分比):")
    for name, arr in (("肩", dS), ("肘", dE), ("腕", dW)):
        a = np.array(arr)
        print("   %s  中位 %5.1f px (%.2f%% 臂长)   90分位 %5.1f px (%.2f%%)"
              % (name, np.median(a), 100*np.median(a)/med_arm,
                 np.percentile(a, 90), 100*np.percentile(a, 90)/med_arm))

# 肘夹角
a = np.array([x for x in angles if not math.isnan(x)])
if len(a) > 1:
    jumps = np.abs(np.diff(a))
    print("-" * 62)
    print("肘夹角          : 均值 %.1f°  标准差 %.1f°  范围 [%.1f, %.1f]"
          % (a.mean(), a.std(), a.min(), a.max()))
    print("肘夹角相邻跳变  : 中位 %.1f°  90分位 %.1f°  最大 %.1f°"
          % (np.median(jumps), np.percentile(jumps, 90), jumps.max()))

print("=" * 62)
