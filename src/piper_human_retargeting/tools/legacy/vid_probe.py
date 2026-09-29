"""离线探测：这段视频里人体手臂关键点的可识别性"""
import sys, os, statistics as st
sys.path.insert(0, "/home/zxmy/Yolo_pose+piper/src/piper_human_retargeting")
from piper_human_perception import (VideoFileSource, YOLOPoseProvider,
                                    make_hand_provider, merge_hand_wrist)
from piper_human_retargeting import compute_arm_geometry
VID="/home/zxmy/Yolo_pose+piper/testdata/arm_fwd_up.mp4"
src=VideoFileSource(VID, loop=False); assert src.open(), "视频打不开"
p=YOLOPoseProvider(model_path="/home/zxmy/Yolo_pose+piper/models/yolo11n-pose.pt",
                   conf_threshold=0.4, person_conf=0.5, target_side="right", device=None, imgsz=960)
p.load(); hand=make_hand_provider()
conf={"shoulder":[], "elbow":[], "wrist":[]}
n=0; ok_geom=0; up_ang=[]; fo_ang=[]; el_ang=[]
while True:
    fr=src.read()
    if fr is None: break
    n+=1
    d=p.detect(fr)
    if d.arm is None: continue
    a=d.arm
    for k,kp in (("shoulder",a.shoulder),("elbow",a.elbow),("wrist",a.wrist)):
        if kp is not None: conf[k].append(kp.confidence)
    hd=hand.detect(fr, a.wrist, "right")
    a2=merge_hand_wrist(a,hd)
    g=compute_arm_geometry(a2, 0.4, True, pivot="shoulder")
    if g.valid:
        ok_geom+=1; up_ang.append(g.upper_arm_angle_deg)
        fo_ang.append(g.forearm_angle_deg); el_ang.append(g.elbow_angle_deg)
src.release(); p.close(); hand.close()
print(f"共 {n} 帧")
print(f"{'关键点':<10}{'有效帧':>8}{'中位置信度':>12}")
for k,v in conf.items():
    ge=sum(1 for x in v if x>=0.4)
    m=f"{st.median(v):.3f}" if v else " - "
    print(f"{k:<10}{ge:>8}/{n}{m:>12}")
print(f"\n几何有效 {ok_geom}/{n} ({ok_geom/max(n,1)*100:.0f}%)")
if up_ang:
    print(f"{'量':<14}{'最小':>9}{'中位':>9}{'最大':>9}{'行程':>9}")
    for nm,v in (("上臂角",up_ang),("前臂角",fo_ang),("肘夹角",el_ang)):
        print(f"{nm:<14}{min(v):>9.1f}{st.median(v):>9.1f}{max(v):>9.1f}{max(v)-min(v):>9.1f}")
