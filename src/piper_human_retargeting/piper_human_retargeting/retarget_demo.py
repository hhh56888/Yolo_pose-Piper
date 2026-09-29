# -*- coding: utf-8 -*-
"""
演示节点：人体手臂 -> Piper 机械臂 实时遥操作
==============================================
完整控制链：

    RGB 摄像头
      -> YOLO Pose            （肩 / 肘 / 腕）
      -> 手部识别              （权威腕关节 + 手掌开合度）
      -> KeypointFilter       （① 关键点滤波）
      -> compute_arm_geometry （上臂方向 / 肘夹角 / 前臂方向）
      -> AngleFilter          （② 角度滤波）
      -> map_human_to_robot   （③ 区间归一化映射，唯一的人体->关节入口）
      -> JointFilter          （④ 关节滤波 + 死区）
      -> SafetyLimiter        （⑤ 关节限位 / 速度 / 加速度限制）
      -> TrajectoryStreamer   （⑥ 平滑下发）
      -> Piper (Gazebo)
    同时：HandGeometry.openness -> GripperController -> joint7/joint8

设计要点：
  * **控制循环显式限频**（control_hz，默认 30Hz）。
    若不限频，控制周期由推理耗时决定，YOLO 首帧 1.1s 或偶发卡顿
    会直接表现为控制周期拉长 —— 这不该出现在实时遥操作里。
  * **推理与控制分离**：感知在自己的线程里跑，控制循环按固定周期
    取「最新一帧」结果。这样即使推理偶发卡顿，控制周期仍稳定。
  * **参数全在配置里**，本文件不含任何 magic number。

运行：
    export DISPLAY=:1
    source ~/Yolo_pose+piper/install/setup.bash
    ros2 run piper_human_retargeting retarget_demo

按键（可视化窗口内）：
    c / r  开始 / 重新标定
    n      回初始位姿
    k      切换逐关节测试模式（J3 -> J2 -> J5 -> J2+J3 -> 全开）
    g      切换夹爪模式（hand / fixed）
    e      急停 / 解除
    q      退出
"""

import argparse
import os
import sys
import threading
import time
from typing import Optional

import cv2
import numpy as np

import rclpy

from piper_human_control import (ControlConfig, GripperController,
                                 PiperJointController)
from piper_human_perception import (ImageFileSource, OpenCVCameraSource,
                                    VideoFileSource, YOLOPoseProvider,
                                    compute_hand_pose, make_hand_provider,
                                    merge_hand_wrist)
from piper_human_perception.visualization import PoseVisualizer

from .arm_direction import arm_direction_deg
from .arm_geometry import ArmGeometry, compute_arm_geometry
from .config import RetargetingConfig
from .debug_log import CsvDebugLogger
from .hand_calibration import (GripperCalibration,
                               HandCalibrationConfig, HandCalibrator)
from .retargeter import Retargeter
from .tracker_state import TrackState

DEFAULT_POSE_MODEL = os.path.expanduser(
    "~/Yolo_pose+piper/models/yolo11n-pose.pt")
WINDOW = "Piper Teleop (2D V1)"

# 逐关节测试顺序（按需求文档要求）
# 每个阶段给出「只启用哪些关节映射」的关节名集合
JOINT_TEST_STAGES = [
    ("仅 J3", ["joint3"]),
    ("仅 J2", ["joint2"]),
    ("仅 J5", ["joint5"]),
    ("J2+J3", ["joint2", "joint3"]),
    ("J2+J3+J5", ["joint2", "joint3", "joint5"]),
]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="人体手臂 -> Piper 机械臂 实时遥操作（2D V1）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--video", type=str, default=None)
    p.add_argument("--image", type=str, default=None)
    p.add_argument("--pose-model", type=str, default=DEFAULT_POSE_MODEL)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--retarget-config", type=str, default=None)

    p.add_argument("--control-hz", type=float, default=None,
                   help="覆盖配置里的控制频率")
    p.add_argument("--csv", type=str, default=None,
                   help="启用 CSV 记录并指定路径（覆盖配置）")
    p.add_argument("--no-csv", action="store_true", help="关闭 CSV 记录")

    p.add_argument("--no-gripper", action="store_true", help="不控制夹爪")
    p.add_argument("--no-robot", action="store_true",
                   help="只算映射，不控制机械臂（离线调参）")
    p.add_argument("--no-current-initial", action="store_true",
                   help="不使用启动时当前位姿，改用配置里的 initial_pose")
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--auto-calibrate", action="store_true")
    p.add_argument("--test-stage", type=int, default=-1,
                   help="直接进入第 N 个逐关节测试阶段（0..4），-1=全开")
    p.add_argument("--filtered-angles", action="store_true",
                   help="映射阶段使用**滤波后**的人体角（默认用原始角，"
                        "保持历史行为）。开启后角度滤波才真正作用于下发值")
    p.add_argument("--side", choices=("left", "right"), default=None,
                   help="识别哪一侧手臂，覆盖配置里的 human.side。"
                        "实战中很关键：若某侧肘部在画面外，该侧几何会一直无效"
                        "（elbow confidence=0），换另一侧即可正常")
    return p.parse_args(argv)


# ============================================================
# 感知线程
# ============================================================
class PerceptionWorker:
    """
    把「相机采集 + 推理」放到独立线程。

    为什么分离：
        推理耗时波动很大（首帧 1.1s，稳态 6-10ms）。
        若推理与控制在同一循环里，控制周期会随推理抖动，
        导致下发时间不均匀 —— 对底层控制器是不必要的扰动。
        分离后控制循环按固定周期取「最新可用结果」，
        即使推理偶发卡顿，控制周期仍稳定（只是画面略滞后）。
    """

    def __init__(self, args, cfg: RetargetingConfig, visualizer=None,
                 allow_render: bool = False):
        self.args = args
        self.cfg = cfg
        # 可视化放在**感知线程**里做，这在架构上是有意的：
        #
        #   * OpenCV 的窗口只能在主线程刷新（非 GUI 线程里 cv2.imshow 会阻塞，
        #     实测 6 秒内只投递成功 0~1 次，窗口全黑），所以 imshow 必须留在
        #     控制循环所在的主线程；
        #   * 但 viz.render 是纯 Python、反复抢 GIL，若也放主线程，
        #     它会和本线程的推理互相踩踏 —— 实测渲染从 7.6ms 涨到 45ms，
        #     控制周期 33ms -> 49ms（30Hz -> 20.3Hz）。
        #
        # 于是拆成两半：**渲染在这里（感知线程）**，**imshow 在主线程**，
        # 中间用单槽邮箱交接「已经画好的画布」。这样控制循环每周期
        # 只花约 2ms 在显示上，渲染再怎么抖动也落不到控制节拍上。
        self.visualizer = visualizer
        self.allow_render = allow_render
        self.lock = threading.Lock()
        self._stop = threading.Event()
        # 显示交接：渲染好的画布（单槽，覆盖即丢帧）
        self._canvas: Optional[np.ndarray] = None
        self._canvas_lock = threading.Lock()
        self.n_rendered = 0
        self.n_render_dropped = 0
        self.render_ms = 0.0
        self.max_render_ms = 0.0
        # HUD 文本由控制线程每周期写入（它才知道状态机/映射结果），
        # 感知线程渲染时读取。HUD 滞后一帧是无害的。
        self.hud_provider = None

        # 最新结果
        self.arm = None
        self.hand_det = None
        self.pose_det = None
        self.frame = None
        self.geo_raw: Optional[ArmGeometry] = None
        self.openness: Optional[float] = None
        self.hand_pose = None
        self.n_frames = 0
        self.n_pose_ms = 0.0
        self.n_hand_ms = 0.0
        self.error: Optional[str] = None

        self._thread = None

    def start(self) -> bool:
        a = self.args
        cfg = self.cfg

        if a.image:
            self.source = ImageFileSource(a.image)
        elif a.video:
            self.source = VideoFileSource(a.video, loop=True)
        else:
            self.source = OpenCVCameraSource(a.camera)
        if not self.source.open():
            self.error = f"无法打开帧源: {self.source.describe()}"
            return False

        # 模型文件不存在时必须给出可读的失败信息，而不是抛 FileNotFoundError 堆栈。
        # 帧源打不开、配置非法都有 [FAIL] 提示，唯独模型路径漏了 ——
        # 同样是「用户输入错误」，报错风格不一致最容易让人误判成程序 bug。
        if not os.path.isfile(a.pose_model):
            self.error = f"姿态模型文件不存在: {a.pose_model}"
            return False

        self.pose = YOLOPoseProvider(
            model_path=a.pose_model, conf_threshold=cfg.human.min_confidence,
            target_side=cfg.human.side, device=a.device, imgsz=a.imgsz)
        try:
            self.pose.load()
        except Exception as exc:                            # noqa: BLE001
            self.error = f"姿态模型加载失败 ({a.pose_model}): {exc}"
            return False
        self.hand = make_hand_provider()

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return True

    def _run(self):
        cfg = self.cfg
        while not self._stop.is_set():
            try:
                frame = self.source.read()
                if frame is None:
                    if self.args.image or self.args.video:
                        time.sleep(0.005)
                        continue
                    time.sleep(0.005)
                    continue

                t0 = time.monotonic()
                det = self.pose.detect(frame)
                t1 = time.monotonic()

                hand_det = self.hand.detect(
                    frame, det.arm.wrist if det.arm is not None else None,
                    cfg.human.side)
                t2 = time.monotonic()

                if det.arm is not None:
                    # 用手部识别的腕关节替换人体腕部（项目核心规则）
                    det.arm = merge_hand_wrist(det.arm, hand_det)

                # 原始几何（未经滤波）—— 只用于 debug 的 raw 列。
                # 必须与 Retargeter 用**同一套枢轴配置**，否则 debug 的
                # raw 列和实际参与映射的值不是同一个基准，对比就失去意义。
                h, w = frame.rgb.shape[:2]
                geo_raw = compute_arm_geometry(
                    det.arm, min_confidence=0.0, require_complete=False,
                    pivot=self.cfg.geometry.pivot,
                    image_size=(w, h),
                    anchor=(self.cfg.geometry.anchor_x,
                            self.cfg.geometry.anchor_y))

                openness = None
                if hand_det is not None and hand_det.geometry is not None:
                    openness = hand_det.geometry.openness

                # 腕部姿态（驱动 joint5/joint6）。在感知线程算，
                # 不占控制循环时间。
                hand_pose = compute_hand_pose(det.arm, hand_det)

                with self.lock:
                    self.frame = frame
                    self.pose_det = det
                    self.hand_det = hand_det
                    self.arm = det.arm
                    self.geo_raw = geo_raw
                    self.openness = openness
                    self.hand_pose = hand_pose
                    self.n_frames += 1
                    self.n_pose_ms = (t1 - t0) * 1000.0
                    self.n_hand_ms = (t2 - t1) * 1000.0

                # ---- 渲染（在本线程做，见 __init__ 里的说明）----
                if self.allow_render and self.visualizer is not None:
                    self._render(frame, det, hand_det)
            except Exception as exc:                        # noqa: BLE001
                self.error = f"感知线程异常: {exc}"
                time.sleep(0.1)

    def _render(self, frame, det, hand_det) -> None:
        """渲染一帧到画布并放进单槽（覆盖即丢帧）"""
        extra = self.hud_provider() if self.hud_provider else []
        t0 = time.monotonic()
        canvas = self.visualizer.render(frame, det, hand_det,
                                        extra_lines=extra)
        dt = (time.monotonic() - t0) * 1000.0
        self.render_ms += dt
        self.max_render_ms = max(self.max_render_ms, dt)
        with self._canvas_lock:
            if self._canvas is not None:
                self.n_render_dropped += 1
            self._canvas = canvas
        self.n_rendered += 1

    def take_canvas(self):
        """取走待显示画布；没有则返回 None（控制线程调用，非阻塞）"""
        with self._canvas_lock:
            c, self._canvas = self._canvas, None
        return c

    def snapshot(self):
        """取最新一帧（线程安全）"""
        with self.lock:
            return (self.frame, self.pose_det, self.hand_det, self.arm,
                    self.geo_raw, self.openness, self.n_frames,
                    self.hand_pose)

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        try:
            self.source.release()
            self.pose.close()
            self.hand.close()
        except Exception:                                   # noqa: BLE001
            pass


# ============================================================
# 主程序
# ============================================================
class TeleopApp:
    def __init__(self, args):
        self.args = args
        self.cfg = RetargetingConfig.from_yaml(args.retarget_config)
        if args.control_hz:
            self.cfg.control.control_hz = float(args.control_hz)
        if args.filtered_angles:
            self.cfg.use_filtered_angles = True
        if args.side:
            self.cfg.human.side = args.side

        self.robot_cfg = ControlConfig.from_yaml()
        self.retargeter = Retargeter(self.cfg, self.robot_cfg)
        self.visualizer = PoseVisualizer(
            show_all_skeleton=False, show_metrics=True, mirror=True,
            pivot=self.cfg.geometry.pivot,
            anchor=(self.cfg.geometry.anchor_x, self.cfg.geometry.anchor_y))

        self.robot: Optional[PiperJointController] = None
        self.gripper: Optional[GripperController] = None
        self.perception: Optional[PerceptionWorker] = None
        self.csv: Optional[CsvDebugLogger] = None

        # ---- 逐关节测试模式 ----
        self.test_stage = args.test_stage
        if self.test_stage >= 0:
            self._apply_test_stage(self.test_stage)

        # ---- 状态 ----
        self.message = "按 c 开始标定"
        self.gripper_mode = self.cfg.gripper.mode

        # ---- V1.1 手部标定状态机 ----
        # 原先「第一次检出手就当基准」不够严谨（可能只有一帧、可能误检）。
        # 现在要求连续稳定 stable_frames 帧后再采 sample_frames 帧，
        # 并检查离散度；完成后把均值作为 J5/J6 的零点。
        self.hand_cal = HandCalibrator(HandCalibrationConfig())
        # ---- V1.1 夹爪双点标定 ----
        # 按 o 记录"张开"、按 p 记录"握拳"，之后夹爪按两点归一化映射，
        # 不再依赖固定经验阈值。
        self.gripper_cal = GripperCalibration()

        # ---- 统计 ----
        self.n_loops = 0
        self.n_overrun = 0
        self.n_sent = 0
        self.max_loop_ms = 0.0
        self.sum_loop_ms = 0.0
        self.t_start = time.monotonic()
        # 「环路耗时」只统计 step() 内部，不能反映真实周期。
        # 真实周期还包含 sleep 等待，且会被 GIL 争用拉长
        #（感知线程在做 PyTorch/MediaPipe 推理）。
        # 因此单独统计相邻两次 tick 的真实间隔 —— 这才是控制频率的真值。
        self._last_tick: Optional[float] = None
        self.periods: list = []
        # HUD 文本由控制线程写、感知线程读。
        # 只做「整体替换一个 list 引用」，list 本身构造完就不再修改，
        # 因此无需加锁 —— 读取方拿到的要么是旧的整份、要么是新的整份。
        self._hud_cache: list = []

    # ------------------------------------------------------------
    def _apply_test_stage(self, stage: int):
        """只启用指定关节的映射，其余禁用（逐关节验证用）"""
        stage = max(0, min(stage, len(JOINT_TEST_STAGES) - 1))
        _name, joints = JOINT_TEST_STAGES[stage]
        for r in self.cfg.rules:
            r.enabled = (r.joint in joints)
        self.test_stage = stage

    @property
    def test_stage_name(self) -> str:
        if self.test_stage < 0:
            return "全开"
        return JOINT_TEST_STAGES[self.test_stage][0]

    # ------------------------------------------------------------
    def setup(self) -> bool:
        a = self.args

        # ---- 感知 ----
        # 渲染交给感知线程，窗口留主线程（详见 PerceptionWorker.__init__）
        self.perception = PerceptionWorker(
            a, self.cfg, visualizer=self.visualizer,
            allow_render=not a.no_display)
        self.perception.hud_provider = self._current_hud_lines
        if not self.perception.start():
            print(f"[FAIL] {self.perception.error}")
            return False

        # ---- 机械臂 ----
        if not a.no_robot:
            self.robot = PiperJointController(self.robot_cfg,
                                              node_name="piper_teleop")
            t0 = time.monotonic()
            while time.monotonic() - t0 < 10 and not self.robot.has_state():
                rclpy.spin_once(self.robot, timeout_sec=0.1)
            if not self.robot.has_state():
                print("[FAIL] 收不到 /joint_states，请确认 Gazebo 已启动")
                return False

            # 初始位姿 = 当前实际位姿（与标定基准成对）
            if self.cfg.robot.use_current_as_initial and not a.no_current_initial:
                cur = self.robot.get_joint_positions()
                self.retargeter.set_initial_pose(cur, also_set_neutral=True)
                print(f"  [初始位姿] 采用启动时当前位姿: "
                      f"{[round(v, 4) for v in cur]}", flush=True)
            else:
                # ⚠️ 用 active_neutral_joints()：task_space 模式有独立的实验
                # neutral（由 workspace 评分搜索给出），legacy 仍用 robot.neutral_joints。
                # 直接写 cfg.neutral_joints 会让 task_space 模式的初始位姿
                # 与 IK 的 q_neutral / 标定基准不一致。
                # 用 active_initial_pose()：它保证"机械臂起始位姿"与
                # "IK/task-space 基线"是同一个位姿（曾因 robot.initial_pose
                # 永远非空而错用 legacy neutral，实测踩到）
                nj = self.cfg.active_initial_pose()
                self.retargeter.set_initial_pose(nj, also_set_neutral=True)
                print(f"  [初始位姿] 配置中立位姿({self.cfg.retarget_mode}): "
                      f"{[round(v, 4) for v in nj]}", flush=True)

            self.robot.start()
            if self.cfg.safety.go_neutral_on_start:
                self.robot.send_joint_target(self.retargeter.target,
                                             label="neutral")

            # ---- 夹爪 ----
            if not a.no_gripper:
                self.gripper = GripperController(
                    self.robot, self.cfg.gripper.to_gripper_config(),
                    logger=self.robot.get_logger())
                self.gripper.set_openness(
                    self.cfg.gripper.fixed_openness
                    if self.gripper_mode == "fixed" else 1.0)

        # ---- CSV ----
        use_csv = self.cfg.debug.csv_enabled and not a.no_csv
        if a.csv:
            self.cfg.debug.csv_path = a.csv
            use_csv = True
        if use_csv:
            self.csv = CsvDebugLogger(self.cfg.debug.csv_path,
                                      self.cfg.debug.csv_every_n_frames,
                                      logger=self.robot.get_logger()
                                      if self.robot else None)
        return True

    # ------------------------------------------------------------
    def print_banner(self):
        c = self.cfg
        print("=" * 78)
        print("人体手臂 -> Piper 机械臂 实时遥操作（2D V1）")
        print("=" * 78)
        print(f"  帧源       : {self.perception.source.describe()}")
        print(f"  映射配置   : {c.source_path}")
        print(f"  目标侧     : {c.human.side}   置信度门限: {c.human.min_confidence}")
        _pv = ("画面锚点(底边中点)"
               if c.geometry.pivot == "anchor" else "人体肩关键点")
        print(f"  几何枢轴   : {c.geometry.pivot} = {_pv}  "
              f"锚点=({c.geometry.anchor_x:.2f}, {c.geometry.anchor_y:.2f})")
        print(f"  初始位姿   : {[round(v, 3) for v in self.retargeter.initial_pose]}")
        print(f"  控制频率   : {c.control.control_hz} Hz")
        print(f"  固定关节   : {c.fixed_joints}")
        print("  映射:")
        for r in c.enabled_rules():
            print(f"    {r.key:4s} {r.human:22s} -> {r.joint:8s} "
                  f"Δ[{r.human_min:+.0f},{r.human_max:+.0f}]° -> "
                  f"[{r.robot_min:+.2f},{r.robot_max:+.2f}]rad "
                  f"invert={r.invert}")
        print(f"  滤波       : kp α={c.filter_keypoint.alpha} | "
              f"angle α={c.filter_angle.alpha} db={c.filter_angle.deadband}° | "
              f"joint α={c.filter_joint.alpha} db={c.filter_joint.deadband}")
        print(f"  状态机     : LOST_SHORT<{c.tracker.short_lost_seconds}s  "
              f"LOST_LONG>={c.tracker.long_lost_seconds}s")
        print(f"  夹爪       : {self.gripper_mode} "
              f"({c.gripper.min_openness}~{c.gripper.max_openness} -> "
              f"joint7 {c.gripper.joint7_closed}~{c.gripper.joint7_open}m)")
        print(f"  机械臂     : {'已连接' if self.robot else '未连接(仅计算)'}")
        print(f"  逐关节测试 : {self.test_stage_name}")
        if self.csv:
            print(f"  CSV        : {self.csv.describe()}")
        print("-" * 78)
        print(f"  按 c 标定：保持自然姿势约 {c.calibration.capture_seconds:.0f} 秒")
        print("  按键: c标定 r重标 n回位 k切换测试阶段 g切换夹爪\n        o记张开 p记握拳 e急停 q退出")
        print("-" * 78)

    # ------------------------------------------------------------
    def key_handler(self, key: int) -> bool:
        # 每个按键都打一行日志。
        # 原因：按键只改 HUD 文字的话，事后看日志完全无法判断
        # 「这个键到底按到没有/生效没有」—— 实测排查按键问题时
        # 就因为没有日志而只能靠猜。
        if key in (ord('c'), ord('r')):
            self.retargeter.start_calibration()
            self.message = "标定中... 保持姿势不动"
            print(f"  [按键{'c' if key == ord('c') else 'r'}] 开始重新标定", flush=True)
        elif key == ord('n'):
            if self.robot:
                self.robot.send_joint_target(self.retargeter.target,
                                             label="neutral")
            self.message = "回初始位姿"
            tgt = [round(v, 3) for v in self.retargeter.target]
            print(f"  [按键n] 回初始位姿 -> {tgt}", flush=True)
        elif key == ord('k'):
            nxt = (self.test_stage + 1) % (len(JOINT_TEST_STAGES) + 1)
            if nxt == len(JOINT_TEST_STAGES):
                self.test_stage = -1
                for r in self.cfg.rules:
                    r.enabled = True
            else:
                self._apply_test_stage(nxt)
            self.message = f"测试阶段: {self.test_stage_name}"
            print(f"  [测试阶段] -> {self.test_stage_name}", flush=True)
        elif key == ord('g'):
            self.gripper_mode = ("fixed" if self.gripper_mode == "hand"
                                 else "hand")
            self.message = f"夹爪模式: {self.gripper_mode}"
            print(f"  [按键g] 夹爪模式 -> {self.gripper_mode}", flush=True)
        elif key == ord('e'):
            if self.robot:
                if self.robot.estop.engaged:
                    self.robot.reset_emergency_stop()
                    self.message = "急停解除"
                else:
                    self.robot.emergency_stop("手动触发")
                    self.message = "急停中"
        elif key == ord('o'):
            # 记录夹爪"张开"端
            if self.perception is not None:
                _op = getattr(self.perception, "openness", None)
                if _op is not None:
                    self.gripper_cal.mark_open(_op)
            self.message = f"夹爪标定: {self.gripper_cal.summary()}"
            print(f"  [按键o] {self.message}", flush=True)
        elif key == ord('p'):
            # 记录夹爪"握拳"端
            if self.perception is not None:
                _op = getattr(self.perception, "openness", None)
                if _op is not None:
                    self.gripper_cal.mark_closed(_op)
            self.message = f"夹爪标定: {self.gripper_cal.summary()}"
            print(f"  [按键p] {self.message}", flush=True)
        elif key in (ord('q'), 27):
            return False
        return True

    # ------------------------------------------------------------
    def run(self) -> int:
        if not self.setup():
            return 1
        self.print_banner()
        if self.args.auto_calibrate:
            self.retargeter.start_calibration()
        if not self.args.no_display:
            # 窗口必须在**主线程**创建与刷新。
            #
            # 曾经把渲染+imshow 挪到独立线程以避开 GIL 争用，
            # 实测行不通：OpenCV 非 GUI 线程里的 cv2.imshow 会阻塞
            # （6 秒内只成功投递 0~1 次，窗口全黑）。
            # 真正让 30Hz 达标的是**把单帧渲染降到 ~8ms**
            # （见 piper_human_perception.textdraw 的性能说明），
            # 而不是换线程。8ms 在 33ms 周期里留有充分余量。
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

        period = self.cfg.control.period
        next_tick = time.monotonic()

        try:
            while rclpy.ok():
                loop_t0 = time.monotonic()

                if not self.step():
                    break

                # ---- 真实周期记录（用于诊断控制频率是否达标）----
                now_tick = time.monotonic()
                if self._last_tick is not None:
                    self.periods.append(now_tick - self._last_tick)
                self._last_tick = now_tick

                # ---- 控制循环限频 ----
                self.n_loops += 1
                loop_ms = (time.monotonic() - loop_t0) * 1000.0
                self.sum_loop_ms += loop_ms
                self.max_loop_ms = max(self.max_loop_ms, loop_ms)
                if loop_ms > self.cfg.control.overrun_warn_ms:
                    self.n_overrun += 1

                next_tick += period
                sleep = next_tick - time.monotonic()
                if sleep > 0:
                    time.sleep(sleep)
                else:
                    # 落后了：不补跑，直接对齐到下一个周期，
                    # 避免「积压补偿」造成一串密集下发
                    next_tick = time.monotonic()

                if self.args.max_frames and self.n_loops >= self.args.max_frames:
                    break
        except KeyboardInterrupt:
            print("\n  用户中断")
        finally:
            self.cleanup()

        self.print_summary()
        return 0

    # ------------------------------------------------------------
    def step(self) -> bool:
        a = self.args

        (frame, pose_det, hand_det, arm, geo_raw,
         openness, n_frames, hand_pose) = self.perception.snapshot()

        if frame is None or pose_det is None:
            if self.robot:
                rclpy.spin_once(self.robot, timeout_sec=0.001)
            return True

        now = time.monotonic()

        # ---- 标定采样（用**未滤波**的原始几何，避免滤波器未收敛时污染基准）----
        if self.retargeter.calibrating:
            # 手部姿态也要一起标定 —— 腕关节的量不在 ArmGeometry 里，
            # 不传的话腕部基准为空、joint5/joint6 永远不动。
            # 用 _for 版本：它会自动补齐「不在 ArmGeometry 里」的量
            #（方向角、腕部量）。这条路径曾两次因为漏补而静默失效。
            self.retargeter.add_calibration_sample_for(
                arm, geo_raw, hand_pose=hand_pose)
            if self.retargeter.calibration_elapsed >= \
                    self.cfg.calibration.capture_seconds:
                if self.retargeter.finish_calibration():
                    self.message = "标定完成"
                    print(f"  [标定完成] 人体中立值: "
                          f"{ {k: round(v, 1) for k, v in self.retargeter.neutral.values.items()} }",
                          flush=True)
                    if self.robot:
                        self.robot.send_joint_target(self.retargeter.target,
                                                     label="post-cal")
                else:
                    self.message = self.retargeter.last_status

        # ---- 核心映射 ----
        # 画面尺寸是 pivot="anchor" 的前提（锚点 = 底边中点）
        _h, _w = frame.rgb.shape[:2]
        res = self.retargeter.update(arm, openness=openness, now=now,
                                     frame=self.n_loops,
                                     raw_geometry=geo_raw,
                                     image_size=(_w, _h),
                                     hand_pose=hand_pose)

        # ---- 下发关节 ----
        if self.robot and res.changed and not self.robot.estop.engaged:
            self.robot.send_joint_target(res.positions, label="teleop")
            self.n_sent += 1

        # ---- 手部标定状态机 ----
        # 手部量一旦标定完成，就用它的均值作为 J5/J6 零点
        #（取代先前「第一次检出手即基准」的懒建立）。
        if not self.hand_cal.calibrated and hand_pose is not None:
            hc = self.hand_cal.update(hand_pose, frame=self.n_loops)
            if hc.ok:
                print(f"  [手部标定] {hc.summary()}", flush=True)
                for k, v in hc.mean.items():
                    if self.retargeter.neutral is not None:
                        self.retargeter.neutral.values[k] = float(v)

        # ---- 夹爪 ----
        if self.gripper is not None:
            if self.gripper_mode == "hand" and openness is not None:
                # 已完成双点标定 -> 按标定区间归一化；未完成 -> 回退原始值
                self.gripper.set_openness(self.gripper_cal.ratio(openness))
            else:
                self.gripper.set_openness(self.cfg.gripper.fixed_openness)
            self.gripper.publish()

        # ---- CSV ----
        # 把夹爪状态补进调试快照 —— Retargeter 只负责手臂 6 关节，
        # 夹爪由本层驱动，因此这两列必须在这里填。
        # （曾经漏填，导致 CSV 里 joint7 恒为 0，误以为夹爪没动作。）
        if self.csv is not None and res.debug is not None:
            if self.gripper is not None:
                res.debug.openness = self.gripper.openness
                res.debug.joint7 = self.gripper.target_joint7
            elif openness is not None:
                res.debug.openness = float(openness)
            # ---- tcp_actual_x/z：用**实测关节角**经 FK 算 ----
            # 必须用实测值：用目标值经 FK 算等于自证（目标=解，永远"跟得上"），
            # 看不出控制器/限速造成的实际偏差。
            self._fill_tcp_actual(res.debug)
            self.csv.log(res.debug)

        # ---- 可视化 ----
        # 窗口只能在主线程刷新（OpenCV 限制）。
        # 渲染已在感知线程完成，这里只取现成画布贴出来，约 2ms。
        if not a.no_display:
            self._hud_cache = self._hud_lines(res, geo_raw, n_frames)
            canvas = self.perception.take_canvas() if self.perception else None
            if canvas is not None:
                cv2.imshow(WINDOW, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key != 255 and not self.key_handler(key):
                return False

        # ---- 行统计 ----
        every = self.cfg.debug.stats_every_n_frames
        if every and self.n_loops % every == 0:
            self._print_stats(res, geo_raw)

        if self.robot:
            rclpy.spin_once(self.robot, timeout_sec=0.001)
        return True

    # ------------------------------------------------------------
    def _current_hud_lines(self) -> list:
        """供感知线程渲染时取用（返回的是不可变使用的整份快照）"""
        return self._hud_cache

    # ------------------------------------------------------------
    def _fill_tcp_actual(self, debug) -> None:
        """
        用**实测关节角**经 FK 算出 TCP (x,z)，写进 debug（CSV 的 tcp_actual_*）。

        为什么要实测而不是用目标：目标就是 IK 的解，用目标经 FK 算等于自证，
        永远显示"跟得上"，看不出控制器限速/跟踪误差。
        """
        model = getattr(self.retargeter, "ts_model", None)
        if model is None:
            return
        pos = self.robot.get_joint_positions()
        if pos is None or len(pos) != len(self.cfg.joint_names):
            return
        q = {n: float(v) for n, v in zip(self.cfg.joint_names, pos)}
        for n in ("joint1", "joint4", "joint5", "joint6"):
            q.setdefault(n, 0.0)
        try:
            p = model.tcp(q)
        except Exception:                                  # noqa: BLE001
            return
        debug.tcp_actual_x = float(p[0])
        debug.tcp_actual_z = float(p[2])

    def _hud_lines(self, res, geo_raw, n_frames) -> list:
        """构造 HUD 附加行（raw / filtered 对比）"""
        c = self.cfg
        d = res.debug
        lines = []

        # 状态机
        st = res.state
        lines.append(f"状态: {st}" + (f"  {res.status}" if st is not TrackState.TRACKING else ""))
        if self.retargeter.is_calibrated:
            lines.append(f"标定: 完成({self.retargeter.neutral.samples}样本)")
        elif self.retargeter.calibrating:
            lines.append(f"标定中 {self.retargeter.calibration_progress*100:.0f}%"
                         f" ({self.retargeter.calibration_sample_count}样本)")
        else:
            lines.append("标定: 未完成 (按 c)")

        if geo_raw is not None and geo_raw.valid:
            lines.append(f"几何: 上臂{geo_raw.upper_arm_angle_deg:+.1f}° "
                         f"肘{geo_raw.elbow_angle_deg:.1f}° "
                         f"前臂{geo_raw.forearm_angle_deg:+.1f}°")
        else:
            lines.append(f"几何: {(geo_raw.reason if geo_raw else '无数据')}")

        # raw / filtered 对比（需求明确要求）
        if c.debug.show_raw_filtered and d is not None:
            lines.append("人体角 raw -> filtered:")
            lines.append(f"  上臂 {d.raw_upper:+7.1f} -> {d.flt_upper:+7.1f}")
            lines.append(f"  肘   {d.raw_elbow:+7.1f} -> {d.flt_elbow:+7.1f}")
            lines.append(f"  前臂 {d.raw_forearm:+7.1f} -> {d.flt_forearm:+7.1f}")
            lines.append("关节 mapped -> filtered:")
            tgt = self.retargeter.target
            # ⚠️ task_space 模式下 joint2/joint3 不在 controlled_joints 里
            # （它们的 legacy 规则被停用，改由 IK 决定）。若只按
            # controlled_joints 显示，HUD 会**看不到最重要的两个关节**——
            # 实测踩到：状态行只剩 "5=... 6=..."，排查时完全看不出 J2/J3 在动。
            shown = list(c.controlled_joints())
            for j in c.task_space_joints():
                if j not in shown:
                    shown.append(j)
            for j in shown:
                m = d.mapped_joints.get(j, tgt[c.joint_names.index(j)])
                idx = c.joint_names.index(j)
                lines.append(f"  {j:7s} {m:+.3f} -> {tgt[idx]:+.3f}")
            if c.retarget_mode == "task_space":
                lines.append("task-space (人体 -> TCP 目标 -> IK):")
                lines.append(f"  u {d.human_u_raw:+.3f} -> flt {d.human_u_flt:+.3f}"
                             f"   du {d.du:+.3f}")
                lines.append(f"  v {d.human_v_raw:+.3f} -> flt {d.human_v_flt:+.3f}"
                             f"   dv {d.dv:+.3f}")
                lines.append(f"  TCP目标 ({d.tcp_target_x:+.3f}, {d.tcp_target_z:+.3f})"
                             f"  -> IK ({d.ik_x:+.3f}, {d.ik_z:+.3f})")
                lines.append(f"  IK 误差 ({d.ik_err_x*1000:+.1f}, {d.ik_err_z*1000:+.1f}) mm"
                             f"  跳变 {d.ik_jump:.3f} rad  {d.ik_iters:.0f} 迭代"
                             f"  {d.ik_time_ms:.2f} ms")

        # 夹爪
        if self.gripper is not None:
            lines.append(f"夹爪[{self.gripper_mode}]: "
                         f"open={self.gripper.openness:.2f} "
                         f"j7={self.gripper.target_joint7*1000:.1f}mm")
        else:
            lines.append("夹爪: 未启用")

        lines.append(f"测试阶段: {self.test_stage_name}")
        lines.append(f"控制 {self.cfg.control.control_hz:.0f}Hz  "
                     f"超时 {self.n_overrun}")
        lines.append(self.message)
        return lines

    def _print_stats(self, res, geo_raw):
        c = self.cfg
        tgt = self.retargeter.target
        avg_ms = self.sum_loop_ms / max(1, self.n_loops)
        # 状态行必须包含**真正受控**的关节：task_space 模式下 joint2/joint3
        # 由 IK 决定、不在 controlled_joints 里。若照旧只列 controlled_joints，
        # 终端就只剩 "5=... 6=..."，操作者看不到最主要两个关节（实测踩到）。
        _js = list(c.controlled_joints())
        for _j in c.task_space_joints():
            if _j not in _js:
                _js.append(_j)
        js = " ".join(f"{j[-1]}={tgt[c.joint_names.index(j)]:+.2f}" for j in _js)
        print(f"  [{self.n_loops:5d}] {res.state} | "
              f"环路 {avg_ms:5.1f}ms(峰{self.max_loop_ms:5.1f}) 超时{self.n_overrun} | "
              f"{js} | 下发 {self.n_sent}", flush=True)

    def print_summary(self):
        c = self.cfg
        elapsed = time.monotonic() - self.t_start
        print("-" * 78)
        print("  汇总")
        print("-" * 78)
        print(f"  控制周期数 : {self.n_loops}")
        print(f"  运行时长   : {elapsed:.1f} s (含启动阶段，仅作参考)")
        # 频率一律从「相邻 tick 的真实间隔」算，而不是从进程墙钟时间算。
        # 墙钟包含了 setup()（加载 YOLO 模型、等 /joint_states、把机械臂
        # 平滑送到中立位），那段开销与稳态控制节拍无关，混进来会把
        # 30Hz 的控制循环报成 18Hz —— 是个会误导人的数字。
        ps = sorted(self.periods) if self.periods else []
        if ps:
            span = sum(ps)
            print(f"  控制频率   : {len(ps)/span:.1f} Hz (目标 {c.control.control_hz}, "
                  f"按真实周期计)")
        else:
            print(f"  控制频率   : 无有效周期样本 (目标 {c.control.control_hz})")
        print(f"  环路耗时   : 平均 {self.sum_loop_ms/max(1,self.n_loops):.1f} ms, "
              f"峰值 {self.max_loop_ms:.1f} ms, 超时 {self.n_overrun} 次")
        if ps:
            n = len(ps)
            p50 = ps[n // 2] * 1000.0
            p95 = ps[min(n - 1, int(n * 0.95))] * 1000.0
            # 启动瞬间（首帧推理、模型预热、等状态）会有少量超长周期，
            # 把它们排除后再看稳态 —— 稳定性比平均值更能反映控制质量。
            cap = c.control.period * 5.0
            steady = [x for x in ps if x <= cap]
            n_out = n - len(steady)
            print(f"  真实周期   : 中位 {p50:.1f} ms ({1000.0/p50:.1f} Hz), "
                  f"95分位 {p95:.1f} ms, 峰值 {ps[-1]*1000:.1f} ms")
            if steady:
                s_mean = sum(steady) / len(steady)
                print(f"  稳态周期   : 均值 {s_mean*1000:.1f} ms ({1.0/s_mean:.1f} Hz), "
                      f"样本 {len(steady)}/{n}"
                      + (f" (剔除启动期 {n_out} 个 >{cap*1000:.0f}ms 样本)" if n_out else ""))
            print(f"               (目标 {c.control.period*1000:.1f} ms; "
                  f"周期不稳通常来自感知线程的 GIL 争用)")
        print(f"  关节下发   : {self.n_sent} 次")
        if self.cfg.retarget_mode == "task_space":
            st = self.retargeter.ik_stats()
            if st.get("n"):
                print(f"  IK 求解    : n={st['n']} 均值 {st['mean_ms']:.3f} ms / "
                      f"中位 {st['median_ms']:.3f} / P95 {st['p95_ms']:.3f} / "
                      f"峰值 {st['max_ms']:.3f} ms, 平均迭代 {st['mean_iters']:.1f}, "
                      f"未收敛 {st['failures']}")
            tb = self.retargeter.task_baseline()
            if tb:
                print(f"  task 基线  : u0={tb[0]:+.4f} v0={tb[1]:+.4f} "
                      f"x0={tb[2]:+.4f} z0={tb[3]:+.4f}")
        print(f"  感知帧数   : {self.perception.n_frames if self.perception else 0}")
        print(f"  已标定     : {self.retargeter.is_calibrated}")
        if self.retargeter.neutral:
            print(f"  人体中立值 : "
                  f"{ {k: round(v,1) for k,v in self.retargeter.neutral.values.items()} }")
        print(f"  最终关节   : {[round(v,3) for v in self.retargeter.target]}")
        print(f"  状态统计   : {self.retargeter.stats}")
        if self.csv:
            print(f"  CSV        : {self.csv.rows_written} 行"
                  f" (共 {self.csv.frames_seen} 帧) -> {self.csv.path}")
        if self.perception is not None and self.perception.allow_render:
            p = self.perception
            print(f"  渲染(感知线): {p.n_rendered} 帧, "
                  f"平均 {p.render_ms/max(1,p.n_rendered):.1f} ms, "
                  f"峰值 {p.max_render_ms:.1f} ms")
            print(f"  显示         : 未及时上屏 {p.n_render_dropped} 帧"
                  f"（渲染快于上屏时覆盖，属正常丢帧）")

    def cleanup(self):
        if self.csv:
            self.csv.close()
        if self.perception:
            self.perception.stop()
        if self.robot:
            try:
                self.robot.stop()
                self.robot.destroy_node()
            except Exception:                               # noqa: BLE001
                pass
        if not self.args.no_display:
            cv2.destroyAllWindows()


def main(argv=None) -> int:
    args = parse_args(argv)

    # 关键：显式开启行缓冲。
    # 否则 stdout 重定向到文件时会变成块缓冲（约 4KB 才刷一次），
    # 用 nohup 跑长任务时日志会「看起来没有任何输出」，严重妨碍排查。
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:                                       # noqa: BLE001
        pass

    rclpy.init()
    # 配置/参数错误属于「用户输入问题」，应当打印一行可读的 [FAIL] 后退出，
    # 而不是抛出几十行 traceback 让人以为是程序崩溃。
    # （实测：YAML 里 neutral_joints 写成空列表时抛 ValueError 堆栈，
    #   与「帧源打不开」的 [FAIL] 风格不一致。）
    try:
        app = TeleopApp(args)
    except Exception as exc:                                # noqa: BLE001
        print(f"[FAIL] 配置加载失败: {exc}", flush=True)
        if rclpy.ok():
            rclpy.shutdown()
        return 1
    try:
        rc = app.run()
    finally:
        if rclpy.ok():
            rclpy.shutdown()
    return rc


if __name__ == "__main__":
    sys.exit(main())
