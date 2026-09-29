"""离线：视频里「肩->腕」方向角的分布与时间线"""
import sys, statistics as st
sys.path.insert(0, "/home/zxmy/Yolo_pose+piper/src/piper_human_retargeting")
from piper_human_perception import (VideoFileSource, YOLOPoseProvider,
                                    make_hand_provider, merge_hand_wrist)
from piper_human_retargeting.arm_direction import arm_direction_deg
VID="/home/zxmy/Yolo_pose+piper/testdata/arm_fwd_up.mp4"
src=VideoFileSource(VID, loop=False); src.open()
p=YOLOPoseProvider(model_path="/home/zxmy/Yolo_pose+piper/models/yolo11n-pose.pt",
                   conf_threshold=0.4, person_conf=0.5, target_side="right", device=None, imgsz=960)
p.load(); hand=make_hand_provider()
dirs=[]; reaches=[]; idx=[]
i=0
while True:
    fr=src.read()
    if fr is None: break
    i+=1
    d=p.detect(fr)
    if d.arm is None: continue
    a=merge_hand_wrist(d.arm, hand.detect(fr, d.arm.wrist, "right"))
    ang=arm_direction_deg(a.shoulder, a.effective_wrist)
    if ang is None: continue
    import math
    r=math.hypot(a.effective_wrist.x-a.shoulder.x, a.effective_wrist.y-a.shoulder.y)
    dirs.append(ang); reaches.append(r); idx.append(i)
src.release(); p.close(); hand.close()
print(f"有效 {len(dirs)}/{i} 帧")
print(f"方向角 [{min(dirs):+7.1f},{max(dirs):+7.1f}] 中位 {st.median(dirs):+7.1f} 行程 {max(dirs)-min(dirs):.1f}°")
print(f"肩腕像素距 [{min(reaches):.0f},{max(reaches):.0f}] 中位 {st.median(reaches):.0f}")
print("\n时间线（每 40 帧取样）：")
print(f"{'帧':>6}{'方向角':>10}{'像素距':>9}")
for k in range(0,len(idx),max(1,len(idx)//15)):
    print(f"{idx[k]:>6}{dirs[k]:>10.1f}{reaches[k]:>9.0f}")
# 判断：朝前平举时手臂在画面什么方向？
import collections
h=collections.Counter(int(x//20)*20 for x in dirs)
print("\n方向角分布（20°一档）：")
for k in sorted(h): print(f"  [{k:+4d},{k+20:+4d}) : {h[k]:4d}  {'#'*(h[k]//8)}")
