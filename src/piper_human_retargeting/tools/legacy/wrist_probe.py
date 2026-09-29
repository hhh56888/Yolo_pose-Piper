"""实测：从手部21点+手臂，能提取哪些腕部姿态量，范围多大"""
import sys, time, math, statistics as st
sys.path.insert(0, "/home/zxmy/Yolo_pose+piper/src/piper_human_retargeting")
from piper_human_perception import (OpenCVCameraSource, YOLOPoseProvider,
                                    make_hand_provider, merge_hand_wrist)
import numpy as np

SIDE = sys.argv[1] if len(sys.argv)>1 else "left"
N = int(sys.argv[2]) if len(sys.argv)>2 else 120
cam=OpenCVCameraSource(0); assert cam.open()
p=YOLOPoseProvider(model_path="/home/zxmy/Yolo_pose+piper/models/yolo11n-pose.pt",
                   conf_threshold=0.4, person_conf=0.5, target_side=SIDE, device=None, imgsz=960)
p.load(); hand=make_hand_provider()

def ang_deg(dx,dy):   # y 翻转，与几何模块一致
    return math.degrees(math.atan2(-dy,dx))
def wrap(a): 
    a=(a+180.0)%360.0-180.0
    return a

rows=[]
print(f"采集 {N} 帧（{SIDE} 手）。请做这些动作：手腕上下摆、手掌左右转")
for i in range(N):
    fr=cam.read()
    if fr is None: continue
    d=p.detect(fr)
    if d.arm is None: continue
    hd=hand.detect(fr, d.arm.wrist, SIDE)
    if hd is None or hd.hand is None: continue
    h=hd.hand
    if not h.is_complete: continue
    a=merge_hand_wrist(d.arm, hd)
    w=a.effective_wrist; e=a.elbow
    lm=h.landmarks
    mcp=lm.get("middle_mcp"); idx=lm.get("index_mcp"); pky=lm.get("pinky_mcp")
    if None in (w,e,mcp,idx,pky): continue
    # 前臂方向（肘->腕）
    fore = ang_deg(w.x-e.x, w.y-e.y)
    # 手朝向（腕->中指掌指）
    handdir = ang_deg(mcp.x-w.x, mcp.y-w.y)
    # 腕俯仰 = 手朝向相对前臂的夹角
    wrist_pitch = wrap(handdir - fore)
    # 掌心横轴（index_mcp -> pinky_mcp）的朝向 = 掌面在图像内的滚转
    palm_roll = ang_deg(pky.x-idx.x, pky.y-idx.y)
    rows.append((fore, handdir, wrist_pitch, palm_roll))
if len(rows)<20:
    print(f"有效帧太少({len(rows)})"); cam.release(); p.close(); hand.close(); sys.exit(1)
cols=list(zip(*rows))
names=["前臂朝向","手朝向","腕俯仰(相对前臂)","掌心横轴"]
print(f"\n有效帧 {len(rows)}")
print(f"{'量':<24}{'最小':>9}{'中位':>9}{'最大':>9}{'行程':>9}")
for n,v in zip(names,cols):
    print(f"{n:<24}{min(v):>9.1f}{st.median(v):>9.1f}{max(v):>9.1f}{max(v)-min(v):>9.1f}")
cam.release(); p.close(); hand.close()
