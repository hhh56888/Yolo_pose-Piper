# -*- coding: utf-8 -*-
"""
摄像头自检工具 (camera_check)
=============================
用途：
    在正式跑阶段三之前，快速确认摄像头可用性并**评估摆位是否合适**。

为什么需要「摆位评估」：
    阶段三只关心肩/肘/腕三个点。若人距离相机过近、
    或手臂不在画面内，YOLO 会输出置信度极低的肘/腕
    （实测出现过 conf=0.011，坐标还被压到画面边缘 (0, 480)），
    这种数据进入阶段四只会产生垃圾角度。
    本工具会明确提示「手臂是否完整入镜」。

运行：
    ros2 run piper_human_perception camera_check
    python3 -m piper_human_perception.camera_check --device 0 --frames 60
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

from .perception import OpenCVCameraSource
from .pose import ARM_KEYPOINT_NAMES, KP_INDEX, YOLOPoseProvider


DEFAULT_MODEL = os.path.expanduser("~/Yolo_pose+piper/models/yolo11n-pose.pt")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="摄像头 + YOLO Pose 自检",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--device", type=int, default=0, help="摄像头序号")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--frames", type=int, default=60, help="采样帧数")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--device-infer", type=str, default="0",
                   help="推理设备：0=GPU, cpu=CPU")
    p.add_argument("--side", choices=["right", "left"], default="right")
    p.add_argument("--imgsz", type=int, default=960, help="推理输入尺寸")
    p.add_argument("--conf", type=float, default=0.4, help="关键点置信度门限")
    p.add_argument("--save", type=str, default=None, help="保存一张标注图到该路径")
    p.add_argument("--display", action="store_true", help="开窗口预览")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    print("=" * 74)
    print("摄像头自检 · Piper 阶段三")
    print("=" * 74)

    # ---------- 1. 设备与权限 ----------
    print("\n[1/4] 检查设备")
    dev = f"/dev/video{args.device}"
    if os.path.exists(dev):
        readable = os.access(dev, os.R_OK)
        print(f"  {dev} 存在，可读={readable}")
        if not readable:
            print("  [警告] 不可读。请确认用户在 video 组，或设备有 ACL：")
            print(f"         ls -l {dev}; getfacl {dev}")
    else:
        print(f"  {dev} 不存在")
        print("  可用设备：", [f"/dev/video{i}" for i in range(8)
                              if os.path.exists(f"/dev/video{i}")])

    # ---------- 2. 打开与抓帧 ----------
    print("\n[2/4] 打开摄像头并抓帧")
    source = OpenCVCameraSource(index=args.device, width=args.width,
                                height=args.height, fps=args.fps)
    if not source.open():
        print(f"  [FAIL] 无法打开 {dev}")
        return 1
    print(f"  {source.describe()}")

    frames = []
    t0 = time.monotonic()
    fails = 0
    for _ in range(args.frames):
        f = source.read()
        if f is None:
            fails += 1
        else:
            frames.append(f)
    dt = time.monotonic() - t0

    if not frames:
        print("  [FAIL] 一帧都没抓到")
        source.release()
        return 1

    print(f"  抓取成功 {len(frames)}/{args.frames} 帧，失败 {fails}")
    print(f"  采集帧率约 {len(frames)/max(1e-6, dt):.1f} FPS")

    # 亮度检查（过暗/过曝都会影响 YOLO）
    brightness = float(np.mean(frames[-1].rgb))
    print(f"  平均亮度 {brightness:.1f}/255", end="")
    if brightness < 40:
        print("  [警告] 画面过暗，建议补光")
    elif brightness > 220:
        print("  [警告] 画面过曝")
    else:
        print("  [正常]")

    # ---------- 3. 姿态检测 ----------
    print("\n[3/4] YOLO Pose 检测")
    provider = YOLOPoseProvider(model_path=args.model,
                                target_side=args.side,
                                device=args.device_infer,
                                imgsz=args.imgsz,
                                conf_threshold=args.conf)
    try:
        provider.load()
    except Exception as exc:                              # noqa: BLE001
        print(f"  [FAIL] 模型加载失败: {exc}")
        source.release()
        return 1
    print(f"  {provider.describe()}")

    n_person = 0
    n_arm_ok = 0
    conf_sum = {"shoulder": 0.0, "elbow": 0.0, "wrist": 0.0}
    conf_cnt = {"shoulder": 0, "elbow": 0, "wrist": 0}
    last_det = None
    last_frame = None

    for f in frames:
        det = provider.detect(f)
        last_det, last_frame = det, f
        if det.has_person:
            n_person += 1
        if det.arm_complete:
            n_arm_ok += 1
            for key, kp in (("shoulder", det.arm.shoulder),
                            ("elbow", det.arm.elbow),
                            ("wrist", det.arm.wrist)):
                conf_sum[key] += kp.confidence
                conf_cnt[key] += 1

    n = len(frames)
    print(f"  检出人体   : {n_person}/{n}  ({100.0*n_person/n:.0f}%)")
    print(f"  手臂三点齐全: {n_arm_ok}/{n}  ({100.0*n_arm_ok/n:.0f}%)")
    for k in ("shoulder", "elbow", "wrist"):
        if conf_cnt[k]:
            print(f"  {k:9s} 平均置信度: {conf_sum[k]/conf_cnt[k]:.3f}")
        else:
            print(f"  {k:9s} 平均置信度: (从未齐全)")

    # ---------- 4. 摆位建议 ----------
    print("\n[4/4] 摆位评估")
    ratio = n_arm_ok / n
    if n_person == 0:
        print("  [不合格] 画面中没有人。请坐进画面。")
    elif ratio < 0.3:
        print("  [不合格] 手臂关键点很少齐全 —— 大概率是手臂不在画面内，")
        print("           或人离相机太近/太偏。建议：")
        print("           * 后退，让肩、肘、腕都能看到")
        print("           * 把手臂完全放进画面，尤其是手腕")
        print("           * 避免手臂被桌子/身体遮挡")
    elif ratio < 0.8:
        print("  [勉强] 手臂时有时无。建议调整位置，让手臂更完整入镜。")
    else:
        print("  [合格] 手臂关键点稳定检出，可以开始阶段三。")

    # 保存标注图便于目视确认
    save_path = args.save
    if save_path and last_det is not None and last_frame is not None:
        from .visualization import PoseVisualizer
        vis = PoseVisualizer(mirror=False)
        canvas = vis.render(last_frame, last_det)
        cv2.imwrite(save_path, canvas)
        print(f"\n  标注图已保存: {save_path}")

    if args.display and last_det is not None and last_frame is not None:
        from .visualization import PoseVisualizer
        vis = PoseVisualizer()
        cv2.imshow("camera_check", vis.render(last_frame, last_det))
        cv2.waitKey(3000)
        cv2.destroyAllWindows()

    source.release()
    provider.close()

    print("\n" + "=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
