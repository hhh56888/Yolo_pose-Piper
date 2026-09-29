"""对比几种「抬臂」代理量的可分辨性"""
import sys, statistics as st, math
sys.path.insert(0, "/home/zxmy/Yolo_pose+piper/src/piper_human_retargeting")
from piper_human_perception import (VideoFileSource, YOLOPoseProvider,
                                    make_hand_provider, merge_hand_wrist)
from piper_human_retargeting.arm_direction import arm_direction_deg
VID="/home/zxmy/Yolo_pose+piper/testdata/arm_fwd_up.mp4"
src=VideoFileSource(VID, loop=False); src.open()
p=YOLOPoseProvider(model_path="/home/zxmy/Yolo_pose+piper/models/yolo11n-pose.pt",
                   conf_threshold=0.4, person_conf=0.5, target_side="right", device=None, imgsz=960)
p.load(); hand=make_hand_provider()
rows=[]
i=0
while True:
    fr=src.read()
    if fr is None: break
    i+=1
    d=p.detect(fr)
    if d.arm is None: continue
    a=merge_hand_wrist(d.arm, hand.detect(fr, d.arm.wrist, "right"))
    sh,wr,el=a.shoulder,a.effective_wrist,a.elbow
    ak=d.all_keypoints or {}
    hip=ak.get("right_hip") or ak.get("left_hip")
    if None in (sh,wr,el) or not (sh.is_valid and wr.is_valid and el.is_valid): continue
    # 躯干尺度：肩到髋（无髋则用肩到肘*2 兜底）
    if hip is not None and hip.is_valid:
        torso=math.hypot(hip.x-sh.x, hip.y-sh.y)
    else:
        torso=2*math.hypot(el.x-sh.x, el.y-sh.y)
    if torso<20: continue
    ang=arm_direction_deg(sh,wr)
    # 腕相对肩的「抬高」（y 向上为正），用躯干长度归一化
    lift=(sh.y-wr.y)/torso            # 腕高于肩为正
    lift_el=(sh.y-el.y)/torso         # 肘高于肩为正
    rows.append((i, ang, lift, lift_el, (wr.x-sh.x)/torso))
src.release(); p.close(); hand.close()
print(f"有效 {len(rows)} 帧")
q=lambda v,p: sorted(v)[min(len(v)-1,int(len(v)*p))]
names=["方向角(°)","腕抬高/躯干","肘抬高/躯干","前向距离/躯干"]
for k,nm in enumerate(names, start=1):
    v=[r[k] for r in rows]
    print(f"{nm:<18} 最小{min(v):+8.2f} 5%{q(v,.05):+8.2f} 中位{st.median(v):+8.2f} "
          f"95%{q(v,.95):+8.2f} 最大{max(v):+8.2f}  行程{max(v)-min(v):7.2f}")
print("\n按腕抬高排序的峰值帧：")
for r in sorted(rows,key=lambda x:-x[2])[:5]:
    print(f"  帧{r[0]:>4} 方向{r[1]:+7.1f}° 腕抬高{r[2]:+.3f} 肘抬高{r[3]:+.3f} 前向{r[4]:+.3f}")
print("按腕抬高排序的最低帧：")
for r in sorted(rows,key=lambda x:x[2])[:5]:
    print(f"  帧{r[0]:>4} 方向{r[1]:+7.1f}° 腕抬高{r[2]:+.3f} 肘抬高{r[3]:+.3f} 前向{r[4]:+.3f}")
