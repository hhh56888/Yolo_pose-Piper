import sys, time
sys.path.insert(0, "/home/zxmy/Yolo_pose+piper/src/piper_human_retargeting")
from piper_human_perception import OpenCVCameraSource, YOLOPoseProvider, make_hand_provider
SIDE="right"
cam=OpenCVCameraSource(0); cam.open()
p=YOLOPoseProvider(model_path="/home/zxmy/Yolo_pose+piper/models/yolo11n-pose.pt",
                   conf_threshold=0.4, person_conf=0.5, target_side=SIDE, device=None, imgsz=960)
p.load(); hand=make_hand_provider()
for i in range(8):
    fr=cam.read()
    if fr is None: print(f"{i}: 无帧"); continue
    d=p.detect(fr)
    arm=d.arm
    print(f"{i}: 人数={d.num_persons} arm={'有' if arm else '无'}", end="")
    if arm:
        print(f" 肩{arm.shoulder.confidence:.2f} 肘{arm.elbow.confidence:.2f} 腕{arm.wrist.confidence:.2f}", end="")
    hd=hand.detect(fr, arm.wrist if arm else None, SIDE)
    if hd is None or hd.hand is None:
        print("  手部=无")
        continue
    h=hd.hand
    lm=h.landmarks or {}
    missing=[k for k in ("wrist","index_mcp","pinky_mcp","thumb_tip","thumb_mcp") if k not in lm or not lm[k].is_valid]
    print(f"  手部=有 complete={h.is_complete} 关键点数={len(lm)} 缺失/无效={missing}")
cam.release(); p.close(); hand.close()
