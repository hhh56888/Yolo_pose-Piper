# -*- coding: utf-8 -*-
"""
阶段三演示程序：YOLO Pose 2D 人体手臂识别
==========================================
功能：
    RGB 摄像头 -> YOLO Pose -> 肩/肘/腕关键点 -> 实时可视化

运行示例：
    # 实时摄像头（默认 /dev/video0）
    python3 -m piper_human_perception.pose_demo

    # 指定摄像头与目标手臂
    python3 -m piper_human_perception.pose_demo --camera 0 --side right

    # 用图片测试（不开窗口，输出结果图）
    python3 -m piper_human_perception.pose_demo --image /tmp/cam_test.jpg --no-display

    # 用视频文件
    python3 -m piper_human_perception.pose_demo --video test.mp4

    # 无显示环境（纯统计 + 存图）
    python3 -m piper_human_perception.pose_demo --no-display --save-dir /tmp/out

按键（显示模式下）：
    q / ESC  退出
    s        保存当前帧（原图 + 标注图）
    m        切换镜像显示
    k        切换完整骨架显示

设计说明：
    本程序只做「感知 + 可视化」，**不控制机械臂**。
    它与阶段二的控制包完全解耦，二者通过数据结构（ArmKeypoints）衔接，
    由阶段四的映射模块把它们连起来。
"""

import argparse
import os
import sys
import time
from typing import Optional

import cv2

from .hand import (CascadingHandProvider, HandDetection, HandProvider,
                   MediaPipeHandProvider, YOLOWristHandProvider,
                   make_hand_provider, merge_hand_wrist)
from .perception import (Frame, FrameSource, ImageFileSource,
                         OpenCVCameraSource, VideoFileSource)
from .pose import PoseDetection, YOLOPoseProvider
from .visualization import PoseVisualizer


DEFAULT_MODEL = os.path.expanduser("~/Yolo_pose+piper/models/yolo11n-pose.pt")
WINDOW_NAME = "Piper Pose (stage 3)"


def build_source(args) -> FrameSource:
    """根据命令行参数选择帧源"""
    if args.image:
        return ImageFileSource(args.image)
    if args.video:
        return VideoFileSource(args.video, loop=args.loop)
    return OpenCVCameraSource(index=args.camera,
                              width=args.width,
                              height=args.height,
                              fps=args.fps)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="阶段三：YOLO Pose 2D 人体手臂识别演示",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    src = p.add_argument_group("帧源")
    src.add_argument("--camera", type=int, default=0, help="摄像头序号")
    src.add_argument("--image", type=str, default=None, help="用单张图片")
    src.add_argument("--video", type=str, default=None, help="用视频文件")
    src.add_argument("--loop", action="store_true", help="视频循环播放")
    src.add_argument("--width", type=int, default=640, help="期望宽度")
    src.add_argument("--height", type=int, default=480, help="期望高度")
    src.add_argument("--fps", type=int, default=30, help="期望帧率")

    det = p.add_argument_group("检测")
    det.add_argument("--model", type=str, default=DEFAULT_MODEL, help="YOLO pose 模型路径")
    det.add_argument("--side", choices=["right", "left"], default="right",
                     help="目标手臂（当前项目优先右手）")
    det.add_argument("--conf", type=float, default=0.4, help="关键点置信度门限")
    det.add_argument("--person-conf", type=float, default=0.5, help="人体检测置信度门限")
    det.add_argument("--device", type=str, default="0", help="推理设备：0=GPU, cpu=CPU")
    det.add_argument("--imgsz", type=int, default=960, help="推理输入尺寸")

    hand = p.add_argument_group("手部")
    hand.add_argument("--no-hand", action="store_true",
                      help="关闭手部关键点识别（只做手臂）")
    hand.add_argument("--no-hand-roi", action="store_true",
                      help="关闭手部 ROI 裁剪（整图送入 MediaPipe，更慢且更易漏检）")
    hand.add_argument("--hand-model", type=str, default=None,
                      help="MediaPipe hand_landmarker.task 模型路径")

    ui = p.add_argument_group("显示")
    ui.add_argument("--no-display", action="store_true",
                    help="不开窗口（无 X11 环境或批量处理时使用）")
    ui.add_argument("--no-skeleton", action="store_true", help="不画完整 17 点骨架")
    ui.add_argument("--no-mirror", action="store_true", help="不镜像显示")
    ui.add_argument("--save-dir", type=str, default=None,
                    help="把每帧标注图保存到此目录")
    ui.add_argument("--max-frames", type=int, default=0,
                    help="最多处理多少帧（0=不限）")
    ui.add_argument("--stats-every", type=int, default=30,
                    help="每多少帧打印一次统计（0=不打印）")

    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    # ---------- 帧源 ----------
    source = build_source(args)
    if not source.open():
        print(f"[FAIL] 无法打开帧源: {source.describe()}")
        return 1

    print("=" * 74)
    print("阶段三 · YOLO Pose 2D 人体手臂识别")
    print("=" * 74)
    print(f"  帧源     : {source.describe()}")
    print(f"  模型     : {args.model}")

    # ---------- 检测器 ----------
    provider = YOLOPoseProvider(
        model_path=args.model,
        conf_threshold=args.conf,
        person_conf=args.person_conf,
        target_side=args.side,
        device=args.device,
        imgsz=args.imgsz,
    )
    try:
        provider.load()
    except Exception as exc:                              # noqa: BLE001
        print(f"[FAIL] 模型加载失败: {exc}")
        source.release()
        return 1
    print(f"  人体检测 : {provider.describe()}")

    # ---------- 手部检测器 ----------
    hand_provider = None
    if not args.no_hand:
        hand_kwargs = {"use_roi": not args.no_hand_roi}
        if args.hand_model:
            hand_kwargs["model_path"] = args.hand_model
        hand_provider = make_hand_provider(prefer_mediapipe=True, **hand_kwargs)
        print(f"  手部检测 : {hand_provider.describe()}")
        if not MediaPipeHandProvider.is_available():
            print("             [提示] 未安装 mediapipe，仅能提供腕点（基线模式）")
    else:
        print("  手部检测 : 已关闭")

    # ---------- 可视化 ----------
    visualizer = PoseVisualizer(
        show_all_skeleton=not args.no_skeleton,
        show_metrics=True,
        mirror=not args.no_mirror,
    )

    if not args.no_display:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        print(f"  存图目录 : {args.save_dir}")

    # ---------- 统计 ----------
    n_frame = 0
    n_person = 0
    n_arm_complete = 0
    n_hand = 0
    n_hand_complete = 0
    n_wrist_from_hand = 0
    latencies = []
    hand_latencies = []
    t_start = time.monotonic()

    print("-" * 74)
    print("  开始处理（按 q 或 ESC 退出）")
    print("-" * 74)

    try:
        while True:
            frame = source.read()
            if frame is None:
                if args.image or args.video:
                    break                                  # 文件播完
                time.sleep(0.01)
                continue

            det = provider.detect(frame)

            # ---- 手部识别 ----
            # ROI 锚点仍用**人体姿态的腕部**：它只是「去哪找手」的搜索起点，
            # 不是最终腕部值（最终腕部由手部模型给出）。
            hand_det = None
            if hand_provider is not None:
                anchor = det.arm.wrist if det.arm is not None else None
                hand_det = hand_provider.detect(frame, anchor, args.side)

            # ---- 用手部识别的腕部替换人体腕部 ----
            # 肘部的连接点改为手部模型定位的腕关节（更准）。
            # 手部失败时 merge_hand_wrist 会保留人体腕部作为兜底，链路不断。
            if det.arm is not None:
                det.arm = merge_hand_wrist(det.arm, hand_det)

            n_frame += 1
            latencies.append(det.latency_ms)
            if det.has_person:
                n_person += 1
            if det.arm_complete:
                n_arm_complete += 1
            if det.arm is not None and det.arm.wrist_source == "hand":
                n_wrist_from_hand += 1
            if hand_det is not None:
                if hand_det.has_hand:
                    n_hand += 1
                if hand_det.is_complete:
                    n_hand_complete += 1
                hand_latencies.append(hand_det.latency_ms)

            canvas = visualizer.render(frame, det, hand_det)

            if args.save_dir:
                cv2.imwrite(os.path.join(args.save_dir, f"frame_{n_frame:06d}.jpg"), canvas)

            if args.stats_every and n_frame % args.stats_every == 0:
                avg_lat = sum(latencies[-args.stats_every:]) / len(latencies[-args.stats_every:])
                rate = n_frame / max(1e-6, time.monotonic() - t_start)
                print(f"  帧 {n_frame:5d} | 人体 {n_person:5d} | 手臂完整 {n_arm_complete:5d} "
                      f"| 手部 {n_hand:5d} | 推理 {avg_lat:5.1f}ms | 平均 {rate:5.1f} FPS")

            if not args.no_display:
                cv2.imshow(WINDOW_NAME, canvas)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    print("  用户退出")
                    break
                if key == ord('s'):
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    raw_path = f"/tmp/pose_raw_{ts}.jpg"
                    ann_path = f"/tmp/pose_ann_{ts}.jpg"
                    cv2.imwrite(raw_path, frame.rgb)
                    cv2.imwrite(ann_path, canvas)
                    print(f"  已保存: {raw_path} / {ann_path}")
                if key == ord('m'):
                    visualizer.mirror = not visualizer.mirror
                    print(f"  镜像显示: {'开' if visualizer.mirror else '关'}")
                if key == ord('k'):
                    visualizer.show_all_skeleton = not visualizer.show_all_skeleton
                    print(f"  完整骨架: {'开' if visualizer.show_all_skeleton else '关'}")

            if args.max_frames and n_frame >= args.max_frames:
                print(f"  已达最大帧数 {args.max_frames}")
                break

    except KeyboardInterrupt:
        print("\n  用户中断")
    finally:
        source.release()
        provider.close()
        if hand_provider is not None:
            hand_provider.close()
        if not args.no_display:
            cv2.destroyAllWindows()

    # ---------- 汇总 ----------
    elapsed = time.monotonic() - t_start
    print("-" * 74)
    print("  统计汇总")
    print("-" * 74)
    print(f"  处理帧数     : {n_frame}")
    print(f"  检出人体帧数 : {n_person}  ({100.0*n_person/max(1,n_frame):.1f}%)")
    print(f"  手臂完整帧数 : {n_arm_complete}  ({100.0*n_arm_complete/max(1,n_frame):.1f}%)")
    if hand_provider is not None:
        print(f"  手部检出帧数 : {n_hand}  ({100.0*n_hand/max(1,n_frame):.1f}%)")
        print(f"  腕部取自手部 : {n_wrist_from_hand}  "
              f"({100.0*n_wrist_from_hand/max(1,n_frame):.1f}%)  "
              f"[其余回退人体腕部]")
        print(f"  手部完整帧数 : {n_hand_complete}  "
              f"({100.0*n_hand_complete/max(1,n_frame):.1f}%)")
    if latencies:
        print(f"  姿态推理延迟 : 平均 {sum(latencies)/len(latencies):.1f} ms "
              f"(最小 {min(latencies):.1f} / 最大 {max(latencies):.1f})")
    if hand_latencies:
        print(f"  手部推理延迟 : 平均 {sum(hand_latencies)/len(hand_latencies):.1f} ms")
    print(f"  总耗时       : {elapsed:.1f} s ({n_frame/max(1e-6, elapsed):.1f} FPS)")

    # 验收判定：只要跑通并有人体检测即算链路正常
    ok = n_frame > 0
    print(f"\n  >>> 阶段三感知链路: {'正常' if ok else '异常'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
