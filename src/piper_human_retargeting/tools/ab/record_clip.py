#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
J3 审计 · 录制 A/B 测试用的人体动作片段（需要人配合）
=====================================================
用法：
    source /opt/ros/humble/setup.bash
    python3 /tmp/record_clip.py

流程（脚本会自己在画面上打提示）：
    5 秒准备倒计时
    阶段 A  伸肘（手臂接近伸直）        3 秒
    阶段 B  屈肘约 45°                 3 秒
    阶段 C  屈肘约 90°                 3 秒
    阶段 D  重新伸肘                    3 秒
    以上 4 阶段重复 2 遍，共 24 秒
输出：/tmp/elbow_ab.mp4（同时在屏幕上显示画面与提示）

要求：肩部与手腕尽量不动，**只做肘部屈伸**。
"""
import sys
import time

import cv2

OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/elbow_ab.mp4"
PHASE_S = 3.0
REPEATS = 2
WARMUP_S = 5.0
PHASES = [("A  STRAIGHT (extend)", (0, 255, 0)),
          ("B  BEND ~45deg", (0, 200, 255)),
          ("C  BEND ~90deg", (0, 0, 255)),
          ("D  STRAIGHT again", (0, 255, 0))]

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[FAIL] 打不开摄像头 /dev/video0")
    sys.exit(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
ok, frame = cap.read()
if not ok:
    print("[FAIL] 读不到帧")
    sys.exit(3)
h, w = frame.shape[:2]
vw = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
print("录制分辨率 %dx%d -> %s" % (w, h, OUT))
print("准备……请站到摄像头前，保持肩部不动，只做肘部屈伸")

t0 = time.monotonic()
n = 0
while True:
    el = time.monotonic() - t0
    total = WARMUP_S + PHASE_S * len(PHASES) * REPEATS
    if el >= total:
        break
    ok, frame = cap.read()
    if not ok:
        continue
    if el < WARMUP_S:
        label, color = ("READY %d" % (WARMUP_S - el + 1), (255, 255, 255))
    else:
        k = int((el - WARMUP_S) // PHASE_S)
        idx = k % len(PHASES)
        rep = k // len(PHASES) + 1
        label, color = ("%s   [rep %d]" % (PHASES[idx][0], rep), PHASES[idx][1])
    vw.write(frame)
    n += 1
    if n % 2 == 0:
        show = frame.copy()
        cv2.rectangle(show, (0, 0), (w, 46), (0, 0, 0), -1)
        cv2.putText(show, label, (10, 32), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, color, 2)
        try:
            cv2.imshow("record", show)      # GUI 失败不影响录制
            cv2.waitKey(1)
        except Exception:                   # noqa: BLE001
            pass
    time.sleep(0.001)

vw.release()
cap.release()
cv2.destroyAllWindows()
print("已保存 %s（%d 帧, %.1f 秒）" % (OUT, n, n / 30.0))
print("接下来由审计脚本做 A/B 回放，无需再操作")
