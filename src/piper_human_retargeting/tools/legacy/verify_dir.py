"""离线验证：视频 -> 标定 -> 方向驱动映射，检查机械臂方向是否跟随人体"""
import sys, statistics as st, math
sys.path.insert(0, "/home/zxmy/Yolo_pose+piper/src/piper_human_retargeting")
from piper_human_perception import (VideoFileSource, YOLOPoseProvider,
                                    make_hand_provider, merge_hand_wrist,
                                    compute_hand_pose)
from piper_human_retargeting import RetargetingConfig, Retargeter, compute_arm_geometry
from piper_human_retargeting.arm_direction import arm_direction_deg
from piper_human_control import ControlConfig
VID="/home/zxmy/Yolo_pose+piper/testdata/arm_fwd_up.mp4"
cfg=RetargetingConfig.from_yaml(); rc=ControlConfig.from_yaml()
rt=Retargeter(cfg, rc)
print(f"enabled={cfg.arm_direction.enabled} gain={cfg.arm_direction.gain} "
      f"neutral_dir={cfg.arm_direction.neutral_dir_deg}")

src=VideoFileSource(VID, loop=False); src.open()
p=YOLOPoseProvider(model_path="/home/zxmy/Yolo_pose+piper/models/yolo11n-pose.pt",
                   conf_threshold=0.4, person_conf=0.5, target_side="right", device=None, imgsz=960)
p.load(); hand=make_hand_provider()

# 先缓存所有有效帧（避免视频只能读一遍）
cache=[]
i=0
while True:
    fr=src.read()
    if fr is None: break
    i+=1
    d=p.detect(fr)
    if d.arm is None: continue
    hd=hand.detect(fr, d.arm.wrist, "right")
    a=merge_hand_wrist(d.arm, hd)
    g=compute_arm_geometry(a, 0.4, True, pivot="shoulder")
    if not g.valid: continue
    hp=compute_hand_pose(a, hd)
    h,w=fr.rgb.shape[:2]
    cache.append((i,a,hp,(w,h),g))
src.release(); p.close(); hand.close()
print(f"有效帧 {len(cache)}")

# 用前 150 帧标定（对应视频里手臂前伸的稳定段）
rt.start_calibration()
ncal=0
for i,a,hp,sz,g in cache[:150]:
    if rt.add_calibration_sample(g, extra_measures=hp.as_dict() if hp.valid else None):
        ncal+=1
print(f"标定样本 {ncal}  完成={rt.finish_calibration()}")
rt.set_initial_pose(cfg.neutral_joints)
print("neutral:", {k:round(v,1) for k,v in rt.neutral.values.items()})

rows=[]
for i,a,hp,sz,g in cache[150:]:
    res=rt.update(a, now=i*0.033, frame=i, hand_pose=hp, image_size=sz)
    j2=res.positions[cfg.joint_names.index("joint2")]
    j3=res.positions[cfg.joint_names.index("joint3")]
    # 机械臂实际方向（查表）
    e=min(rt.dir_mapper.table, key=lambda x:(x.j2-j2)**2+(x.j3-j3)**2)
    rows.append((i, arm_direction_deg(a.shoulder,a.effective_wrist),
                 e.dir_deg, e.reach, j2, j3))
def corr(a,b):
    ma,mb=st.mean(a),st.mean(b)
    cov=sum((x-ma)*(y-mb) for x,y in zip(a,b))/len(a)
    sa=(sum((x-ma)**2 for x in a)/len(a))**.5; sb=(sum((y-mb)**2 for y in b)/len(b))**.5
    return cov/(sa*sb) if sa*sb>1e-12 else float('nan')
h=[r[1] for r in rows]; rd=[r[2] for r in rows]
print(f"\n映射后 {len(rows)} 帧")
print(f"人体方向 [{min(h):+7.1f},{max(h):+7.1f}] 行程 {max(h)-min(h):6.1f}°")
print(f"机械臂方向 [{min(rd):+7.1f},{max(rd):+7.1f}] 行程 {max(rd)-min(rd):6.1f}°")
print(f"corr(人体方向, 机械臂方向) = {corr(h,rd):+.3f}   (应显著正 = 同向跟随)")
print(f"\n{'人体方向':>10}{'机械臂方向':>12}{'伸展':>8}{'j2':>8}{'j3':>8}")
for k in range(0,len(rows),max(1,len(rows)//12)):
    r=rows[k]
    print(f"{r[1]:>10.1f}{r[2]:>12.1f}{r[3]:>8.3f}{r[4]:>8.2f}{r[5]:>8.2f}")
