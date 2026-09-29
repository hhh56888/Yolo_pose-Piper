# -*- coding: utf-8 -*-
"""
piper_human_retargeting.retargeter
==================================
人体手臂 -> Piper 关节目标 的映射核心。

这是整条遥操作链路的「大脑」，把下面几件事按固定顺序串起来：

    ArmKeypoints（原始关键点）
        ↓ ① 关键点滤波        KeypointFilter
    ArmGeometry（上臂角/肘夹角/前臂角）
        ↓ ② 角度滤波          AngleFilter
    相对标定姿态的角度差 Δ
        ↓ ③ 区间映射          map_human_to_robot()   ← 唯一的人体->关节入口
    原始关节目标
        ↓ ④ 关节滤波+死区      JointFilter
    最终关节目标（交给 piper_human_control 做真正的速度/加速度限制）

设计要点：

  1. **不做「人体角度 = 关节角」直接赋值**
     先标定中立姿态，只映射**相对角度**，消除摄像头安装角度、
     人的位置/身高/初始姿势的影响。

  2. **三级滤波放在这里统一做**
     关键点级 → 角度级 → 关节级，每级量纲不同、噪声来源不同。
     放在同一个类里是为了让顺序**不可能被写错**。

  3. **状态机决定行为，不靠 if 堆**
     TRACKING / LOST_SHORT / LOST_LONG 三态语义完全不同：
       TRACKING    正常映射
       LOST_SHORT  保持最后位置（**不动**）
       LOST_LONG   平滑回初始位姿（可配置为冻结）
     ★ 本项目**不会**因检测失败突然执行 HOME ——
       HOME 是大幅运动，遥操作场景下反而危险。

  4. **本模块不碰 ROS2、不碰机械臂**
     只做数学，因此能完全脱机单元测试。

  5. **为 3D 升级留路**
     输入仍是「人体角度」（由 arm_geometry 从 Keypoint 算出）。
     将来升级 RGB-D 时，替换的是 arm_geometry 与 mapping，
     本类的滤波链、状态机、限幅逻辑均可复用。
"""

import time
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from piper_human_control import (AngleFilter, ControlConfig, JointFilter,
                                 KeypointFilter)

from .arm_direction import (ArmDirectionMapper, arm_direction_deg,
                            wrap180)
from .arm_geometry import ArmGeometry, angle_delta_deg, compute_arm_geometry
from .config import RetargetingConfig
from .mapping import map_all
from .task_space import (TaskSpaceIk, compute_human_wrist_task,
                         map_human_to_task)
from .tracker_state import TrackState, TrackerStateMachine


# ============================================================
# 数据
# ============================================================
@dataclass
class HumanNeutral:
    """标定得到的人体中立姿态（相对角度零点）"""
    values: Dict[str, float] = field(default_factory=dict)
    samples: int = 0

    # 各测量量的符号修正（在「做差」之前施加）。
    # 由 Retargeter 在加载配置时注入；空字典表示全部 +1。
    signs: Dict[str, float] = field(default_factory=dict)

    def get(self, name: str) -> Optional[float]:
        """
        取中立值。

        ⚠️ 这里返回的是**已施加符号修正**的值。
        这样做的好处：``angle_delta_deg(meas, neutral.get(name))``
        两边都带同一个符号因子，等价于对差值取反，
        不需要在每个做差的地方各写一次负号。
        """
        v = self.values.get(name)
        if v is None:
            return None
        return v * self.signs.get(name, 1.0)


@dataclass
class DebugSnapshot:
    """
    单帧调试快照 —— 用于 HUD 显示与 CSV 记录。

    这里刻意同时保留 raw 与 filtered：
        raw      能看出「人体到底动了多少」
        filtered 能看出「我们实际用了多少」
    两者之差就是滤波带来的平滑量与延迟，
    是调 alpha / deadband 的唯一依据。
    """
    t: float = 0.0
    frame: int = 0
    state: str = ""

    # 原始关键点（像素）
    shoulder_xy: tuple = (0.0, 0.0)
    elbow_xy: tuple = (0.0, 0.0)
    wrist_xy: tuple = (0.0, 0.0)

    # 人体角度：raw（滤波前）与 filtered（滤波后）
    raw_upper: float = 0.0
    raw_elbow: float = 0.0
    raw_forearm: float = 0.0
    flt_upper: float = 0.0
    flt_elbow: float = 0.0
    flt_forearm: float = 0.0

    # 腕部姿态（来自手部 21 点，驱动 joint5/joint6）
    # ⚠️ 这两列最初漏加，导致「腕关节到底动没动」在 CSV 里完全看不出来 ——
    #    与 joint7/delta_* 是同一类「算出来了却没记录」的漏洞。
    raw_wrist_pitch: float = 0.0
    raw_palm_roll: float = 0.0
    flt_wrist_pitch: float = 0.0
    flt_palm_roll: float = 0.0

    # 手掌特征（拇指相对四指 / 捏合）
    # 这几列同样是「加了新映射就必须同步加列」——
    # joint6 的列曾经漏加过一次，不想再犯。
    # ---- task-space（V1.1）----
    # 任务书第十五节：任何参与控制的量都必须能在 debug/CSV 里看到
    # raw -> mapped -> 最终 q 的完整链路。
    # ---- 人体 2D task-space（V1.1 主控量）----
    #   human_u/v      手腕相对肩的归一化位置（raw 与 filtered 都要）
    #   du/dv          相对标定姿势的位移（死区前/后）
    human_u_raw: float = 0.0
    human_v_raw: float = 0.0
    human_u_flt: float = 0.0
    human_v_flt: float = 0.0
    du: float = 0.0
    dv: float = 0.0
    du_deadzoned: float = 0.0
    dv_deadzoned: float = 0.0
    # ---- 机器人 TCP 目标与 IK ----
    tcp_target_x: float = 0.0
    tcp_target_z: float = 0.0
    tcp_actual_x: float = 0.0
    tcp_actual_z: float = 0.0
    ik_q2: float = 0.0
    ik_q3: float = 0.0
    ik_iters: float = 0.0
    ik_time_ms: float = 0.0
    # 诊断量（不驱动任何关节，仅记录）
    reach_raw: float = 0.0
    reach_flt: float = 0.0
    x_target: float = 0.0
    z_target: float = 0.0
    x_target_raw: float = 0.0
    z_target_raw: float = 0.0
    x_clamped: float = 0.0
    z_clamped: float = 0.0
    ik_x: float = 0.0
    ik_z: float = 0.0
    ik_err_x: float = 0.0
    ik_err_z: float = 0.0
    ik_jump: float = 0.0

    raw_thumb_offset: float = 0.0
    flt_thumb_offset: float = 0.0
    delta_thumb_offset: float = 0.0
    raw_pinch_distance: float = 1.0
    finger_tips_converged: float = 0.0

    # 相对标定姿态的角度差（度）
    delta_upper: float = 0.0
    delta_elbow: float = 0.0
    delta_forearm: float = 0.0
    delta_wrist_pitch: float = 0.0
    delta_palm_roll: float = 0.0

    # 关节目标：映射后（滤波前）与最终（滤波后）
    mapped_joints: Dict[str, float] = field(default_factory=dict)
    final_joints: Dict[str, float] = field(default_factory=dict)

    # 夹爪
    # 注意：openness / joint7 由**调用方**（retarget_demo）填写，
    # 因为夹爪不在 Retargeter 的职责范围内（它只算手臂 6 关节）。
    # 若调用方忘了填，CSV 里这两列会恒为 0 —— 看起来像"夹爪没动"。
    openness: float = 0.0
    joint7: float = 0.0

    def to_csv_row(self) -> Dict[str, object]:
        """转成 CSV 一行（键名与需求文档给出的列一致）"""
        row = {
            "timestamp": f"{self.t:.4f}",
            "frame": self.frame,
            "state": self.state,
            "shoulder_x": f"{self.shoulder_xy[0]:.1f}",
            "shoulder_y": f"{self.shoulder_xy[1]:.1f}",
            "elbow_x": f"{self.elbow_xy[0]:.1f}",
            "elbow_y": f"{self.elbow_xy[1]:.1f}",
            "wrist_x": f"{self.wrist_xy[0]:.1f}",
            "wrist_y": f"{self.wrist_xy[1]:.1f}",
            "raw_upper": f"{self.raw_upper:.2f}",
            "filtered_upper": f"{self.flt_upper:.2f}",
            "raw_elbow": f"{self.raw_elbow:.2f}",
            "filtered_elbow": f"{self.flt_elbow:.2f}",
            "raw_forearm": f"{self.raw_forearm:.2f}",
            "filtered_forearm": f"{self.flt_forearm:.2f}",
            "delta_upper": f"{self.delta_upper:.2f}",
            "delta_elbow": f"{self.delta_elbow:.2f}",
            "delta_forearm": f"{self.delta_forearm:.2f}",
            "raw_wrist_pitch": f"{self.raw_wrist_pitch:.2f}",
            "flt_wrist_pitch": f"{self.flt_wrist_pitch:.2f}",
            "delta_wrist_pitch": f"{self.delta_wrist_pitch:.2f}",
            "raw_palm_roll": f"{self.raw_palm_roll:.2f}",
            "flt_palm_roll": f"{self.flt_palm_roll:.2f}",
            "delta_palm_roll": f"{self.delta_palm_roll:.2f}",
            "raw_thumb_offset": f"{self.raw_thumb_offset:.4f}",
            "flt_thumb_offset": f"{self.flt_thumb_offset:.4f}",
            "delta_thumb_offset": f"{self.delta_thumb_offset:.4f}",
            "raw_pinch_distance": f"{self.raw_pinch_distance:.4f}",
            "finger_tips_converged": f"{self.finger_tips_converged:.4f}",
            "human_u_raw": f"{self.human_u_raw:.5f}",
            "human_v_raw": f"{self.human_v_raw:.5f}",
            "human_u_flt": f"{self.human_u_flt:.5f}",
            "human_v_flt": f"{self.human_v_flt:.5f}",
            "du": f"{self.du:.5f}",
            "dv": f"{self.dv:.5f}",
            "du_deadzoned": f"{self.du_deadzoned:.5f}",
            "dv_deadzoned": f"{self.dv_deadzoned:.5f}",
            "tcp_target_x": f"{self.tcp_target_x:.5f}",
            "tcp_target_z": f"{self.tcp_target_z:.5f}",
            "tcp_actual_x": f"{self.tcp_actual_x:.5f}",
            "tcp_actual_z": f"{self.tcp_actual_z:.5f}",
            "ik_q2": f"{self.ik_q2:.5f}",
            "ik_q3": f"{self.ik_q3:.5f}",
            "ik_iters": f"{self.ik_iters:.0f}",
            "ik_time_ms": f"{self.ik_time_ms:.3f}",
            "reach_raw": f"{self.reach_raw:.4f}",
            "reach_flt": f"{self.reach_flt:.4f}",
            "x_target": f"{self.x_target:.4f}",
            "z_target": f"{self.z_target:.4f}",
            "x_target_raw": f"{self.x_target_raw:.4f}",
            "z_target_raw": f"{self.z_target_raw:.4f}",
            "x_clamped": f"{self.x_clamped:.0f}",
            "z_clamped": f"{self.z_clamped:.0f}",
            "ik_x": f"{self.ik_x:.4f}",
            "ik_z": f"{self.ik_z:.4f}",
            "ik_err_x": f"{self.ik_err_x:.4f}",
            "ik_err_z": f"{self.ik_err_z:.4f}",
            "ik_jump": f"{self.ik_jump:.4f}",
        }
        # ⚠️ 这里曾写成 ("joint2","joint3","joint5") 的**硬编码**列表，
        #    结果 joint6 加入受控集合后 CSV 里根本没有它的列 ——
        #    「joint6 到底动没动」完全无法从日志判断。
        #    与 joint7 / delta_* 是同一类「算出来了却没记录」的漏洞。
        #    改为覆盖 6 个手臂关节：未受控的关节按 0 写出（列存在、值为 0，
        #    一眼就能看出「这个关节没被控制」，比缺列清楚得多）。
        for j in ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6"):
            row[f"mapped_{j}"] = f"{self.mapped_joints.get(j, 0.0):.4f}"
            row[f"{j}"] = f"{self.final_joints.get(j, 0.0):.4f}"
        row["openness"] = f"{self.openness:.3f}"
        row["joint7"] = f"{self.joint7:.5f}"
        return row

    @staticmethod
    def csv_header() -> List[str]:
        cols = ["timestamp", "frame", "state",
                "shoulder_x", "shoulder_y", "elbow_x", "elbow_y",
                "wrist_x", "wrist_y",
                "raw_upper", "filtered_upper", "raw_elbow", "filtered_elbow",
                "raw_forearm", "filtered_forearm",
                "delta_upper", "delta_elbow", "delta_forearm",
                "raw_wrist_pitch", "flt_wrist_pitch", "delta_wrist_pitch",
                "raw_palm_roll", "flt_palm_roll", "delta_palm_roll",
                "raw_thumb_offset", "flt_thumb_offset", "delta_thumb_offset",
                "raw_pinch_distance", "finger_tips_converged",
                "human_u_raw", "human_v_raw", "human_u_flt", "human_v_flt",
                "du", "dv", "du_deadzoned", "dv_deadzoned",
                "tcp_target_x", "tcp_target_z",
                "tcp_actual_x", "tcp_actual_z",
                "ik_q2", "ik_q3", "ik_iters", "ik_time_ms",
                "reach_raw", "reach_flt",
                "x_target", "z_target",
                "x_target_raw", "z_target_raw", "x_clamped", "z_clamped",
                "ik_x", "ik_z", "ik_err_x", "ik_err_z", "ik_jump"]
        # 与 to_csv_row 保持一致：覆盖全部 6 个手臂关节
        for j in ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6"):
            cols += [f"mapped_{j}", j]
        cols += ["openness", "joint7"]
        return cols


@dataclass
class RetargetResult:
    """一次更新的结果"""
    positions: Optional[List[float]] = None
    changed: bool = False
    status: str = ""
    state: TrackState = TrackState.TRACKING
    deltas: Dict[str, float] = field(default_factory=dict)
    debug: Optional[DebugSnapshot] = None

    # 映射后（关节滤波前）的关节值。
    # 保留它是为了在 HUD / CSV 上区分抖动来源：
    #     mapped 就抖            -> 抖动来自人体角度/映射
    #     mapped 稳但 final 抖   -> 滤波器本身有问题
    mapped_joints: Dict[str, float] = field(default_factory=dict)

    # 本帧使用的相对角度差（度），便于排查映射输入
    debug_deltas_deg: Dict[str, float] = field(default_factory=dict)


# ============================================================
# 映射器
# ============================================================
class Retargeter:
    """
    人体手臂 -> Piper 关节 映射器（有状态）。

    典型用法：
        rt = Retargeter(cfg, robot_cfg)
        rt.set_initial_pose(robot.get_joint_positions())
        rt.start_calibration()
        # 循环：
        #   res = rt.update(arm_keypoints, openness, now)
        #   robot.send_joint_target(res.positions)
    """

    def __init__(self, cfg: RetargetingConfig,
                 robot_cfg: Optional[ControlConfig] = None):
        self.cfg = cfg
        self.robot_cfg = robot_cfg

        # ---- 标定 ----
        self.neutral: Optional[HumanNeutral] = None
        self.calibrating = False
        self._cal_samples: Dict[str, List[float]] = {}
        self._cal_started_at: Optional[float] = None

        # ---- 输出目标 ----
        self._target: List[float] = list(cfg.neutral_joints)
        self._apply_fixed_joints()

        # ---- 三级滤波 ----
        self.kp_filter = KeypointFilter(cfg.filter_keypoint, "kp")
        # ---- 方向驱动映射器（可选）----
        # 见 config.ArmDirectionConfig：逐关节映射不保证机械臂朝人手臂的
        # 方向伸出去，改用「肩->腕方向角」查 FK 表反解 joint2/joint3。
        # ---- V1.1 task-space 求解器（可选，与 legacy 并存）----
        # V1.1：J2/J3 由 (x,z) 目标经数值 IK 联合求解（限位在迭代内）。
        # 旧的网格最近邻 JointSolver 已整体替换：网格步长(0.2×0.25 rad)限死了
        # 精度，实测产生大量跳解，且无法把限位放进优化。
        self.task_solver = None
        self.ts_model = None
        self.ts_mapper = None
        self._ts_baseline = None       # (u0, v0, x0, z0)
        self._ts_warned: set = set()
        if cfg.retarget_mode == "task_space":
            from .kinematics import ChainModel
            chain = self._resolve_table_path(cfg.task_space.chain_path,
                                             cfg.source_path)
            self.ts_model = ChainModel.load(chain)
            self.task_solver = TaskSpaceIk.from_config(cfg, self.ts_model)
            self.ts_mapper = cfg.task_space.to_mapping_config()
            # 三层限位校验：retarget ⊂ safe ⊂ physical
            from .limits import build_layers
            layers = build_layers(cfg.task_space.retarget_ranges(),
                                  cfg.task_space.limit_margin_rad,
                                  self.ts_model)
            if layers.violations:
                raise ValueError(
                    "配置错误: task_space 的 IK 关节区间越界 → "
                    + "; ".join(layers.violations))

        self.dir_mapper = None
        if cfg.arm_direction.enabled:
            path = self._resolve_table_path(cfg.arm_direction.table_path,
                                            cfg.source_path)
            self.dir_mapper = ArmDirectionMapper(path)

        # 每量各自的突变门限（见 RetargetingConfig.measure_max_jump 的说明）
        # circular=周期量集合：thumb_offset / pinch_distance 是线性量，
        # 必须走线性 EMA，否则会被按 360° 取模 + 死区吃掉（实测 delta 恒 0）
        self.angle_filter = AngleFilter(cfg.filter_angle, cfg.used_measures(),
                                        "angle",
                                        max_jump=cfg.measure_max_jump(),
                                        circular=cfg.circular_measures())
        self.joint_filter = JointFilter(cfg.filter_joint, cfg.joint_names,
                                        "joint")

        # ---- 状态机 ----
        self.tracker = TrackerStateMachine(cfg.tracker)

        # ---- 回位状态 ----
        self._was_lost = False

        # ---- 统计 ----
        self._update_count = 0
        self._skip_count = 0
        self._last_status = "未标定"
        self._last_geom: Optional[ArmGeometry] = None

    # ============================================================
    # 初始位姿
    # ============================================================
    def set_initial_pose(self, positions: List[float],
                         also_set_neutral: bool = True) -> None:
        """
        把给定关节角设为「初始位姿」。

        同时（默认）设为 neutral_joints —— 这两个**必须一致**：
            initial_pose   机械臂绝对位置基准
            neutral_joints 映射公式里的位置基准
        不一致会导致映射输出整体偏移。
        """
        if len(positions) != len(self.cfg.joint_names):
            raise ValueError(
                f"initial_pose 长度 {len(positions)} != {len(self.cfg.joint_names)}")
        vals = [float(v) for v in positions]
        self.cfg.robot.initial_pose = list(vals)
        if also_set_neutral:
            self.cfg.neutral_joints = list(vals)

        self._target = list(vals)
        self._apply_fixed_joints()
        # 关节滤波器也要跟着重置，否则会把旧位姿当成「历史值」慢慢爬
        self.joint_filter.reset(self._target)

    @property
    def initial_pose(self) -> List[float]:
        if self.cfg.robot.initial_pose:
            return list(self.cfg.robot.initial_pose)
        return list(self.cfg.neutral_joints)

    def _apply_fixed_joints(self) -> None:
        """把 J1/J4/J6 固定为配置的安全初值"""
        for name, val in self.cfg.fixed_joints.items():
            self._target[self.cfg.joint_names.index(name)] = float(val)

    # ============================================================
    # 标定
    # ============================================================
    def start_calibration(self, now: Optional[float] = None) -> None:
        self.calibrating = True
        self.neutral = None
        self._hand_pose = None
        self._arm_dir = None
        # 标定时缺失、由懒建立补上的量（供日志/HUD 提示）
        self._late_baseline: set = set()
        self._task_state = None
        self._last_task_info: Optional[dict] = None
        self._cal_samples = {}
        self._cal_started_at = now if now is not None else time.monotonic()
        self._last_status = "标定中"

    @property
    def calibration_elapsed(self) -> float:
        if self._cal_started_at is None:
            return 0.0
        return time.monotonic() - self._cal_started_at

    @property
    def calibration_progress(self) -> float:
        if not self.calibrating:
            return 1.0 if self.neutral is not None else 0.0
        dur = max(1e-6, self.cfg.calibration.capture_seconds)
        return min(1.0, self.calibration_elapsed / dur)

    @property
    def calibration_sample_count(self) -> int:
        if not self._cal_samples:
            return 0
        return len(next(iter(self._cal_samples.values())))

    def add_calibration_sample_for(self, arm, geo: ArmGeometry,
                                   hand_pose=None) -> bool:
        """
        采集标定样本（推荐入口）—— 自动补齐所有**不在 ArmGeometry 里**的量。

        为什么要有这个入口：
            方向角（arm_direction_deg）与腕部量（wrist_pitch_deg 等）
            都不在 ArmGeometry 里。每次新增这样一个量，
            调用方就得记得手动补进 extra_measures ——
            实测因此漏过两次（先是腕部量、再是方向角），
            表现都是「标定完成但对应关节永远不动」。
            把「该补什么」收进 Retargeter 内部，调用方只需给原始数据。
        """
        extra: Dict[str, float] = {}
        hp = hand_pose
        if hp is not None and getattr(hp, "valid", False):
            extra.update(hp.as_dict())
        if arm is not None:
            d = arm_direction_deg(arm.shoulder, arm.effective_wrist)
            if d is not None:
                extra["arm_direction_deg"] = d
            # task-space 量：human_u/human_v 已在 ArmGeometry.as_dict() 里，
            # 这里补上 reach（诊断量）并保留 ts.as_dict() 的其余键。
            # ⚠️ 漏了这些量的后果特别隐蔽：add_calibration_sample 要求
            #    **所有**在用量本帧都有值，缺一个就整条样本不采纳 ——
            #    标定仍然返回 True（其余量够），但 reach/elevation 基准为空，
            #    于是 task_space 静默不生效（两种模式输出完全相同）。
            ts = compute_human_wrist_task(arm.shoulder, arm.elbow,
                                          arm.effective_wrist)
            if ts.valid:
                extra.update(ts.as_dict())
        return self.add_calibration_sample(geo, extra_measures=(extra or None))

    def add_calibration_sample(self, geo: ArmGeometry,
                               extra_measures: Optional[dict] = None) -> bool:
        """
        采集一个标定样本（几何无效时不采纳）。

        Args:
            geo:           手臂几何
            extra_measures: 额外的测量量（例如手部姿态给出的
                           wrist_pitch_deg / palm_roll_deg）。

        ⚠️ 为什么需要 extra_measures：
            腕关节由手部姿态驱动，那两个量**不在 ArmGeometry 里**。
            若标定只采手臂几何，腕部量的基准就是空的，
            映射时会一直报「缺少测量量 wrist_pitch_deg」、腕关节**永远不动** ——
            这个坑实际踩过：整条映射链看起来通了，只有腕部纹丝不动。
        """
        if not self.calibrating or not geo.valid:
            return False
        geo_dict = geo.as_dict()
        if extra_measures:
            geo_dict.update(extra_measures)
        # ⚠️ 逐量采样，不再"缺一个就整条丢弃"。
        #
        # 原实现要求**所有**在用量本帧都有值，否则整条样本不采纳。
        # 后果（本轮实测踩到）：task_space 模式下 used_measures 同时包含
        # 手臂量与人手量；用合成手臂回放（没有手部数据）时
        # wrist_pitch_deg / thumb_offset 永远缺失 -> 样本数恒为 0 ->
        # `标定失败: 无样本`，task-space 链路完全跑不起来。
        #
        # 语义上也该逐量：某个量在标定窗口内一直拿不到，应当
        # **跳过它并报告**（finish_calibration 的 short 列表），
        # 而不是连累其它已经可用的量 —— 这与"部分标定"的既定设计一致。
        got = 0
        for name in self.cfg.used_measures():
            v = geo_dict.get(name)
            if v is None:
                continue
            self._cal_samples.setdefault(name, []).append(float(v))
            got += 1
        return got > 0

    def finish_calibration(self) -> bool:
        """结束标定。样本不足则失败，neutral 保持 None"""
        self.calibrating = False
        req = self.cfg.calibration.min_samples

        if not self._cal_samples:
            self._last_status = "标定失败: 无样本"
            return False

        # ---- 部分标定 ----
        # 只对「样本充足」的量建基准；样本不足的量**跳过并报告**，
        # 而不是让整个标定失败。
        #
        # 为什么必须这样（实测踩过）：
        #   手部姿态常常在标定窗口内识别不到（手在画面外/被挡），
        #   wrist_pitch / thumb_offset 于是收不到样本。
        #   原先只要有一个量样本不足就 return False ——
        #   连**手臂与方向角**的基准都建不起来，
        #   结果是 tracking 正常但「关节下发 0 次」，
        #   而日志只显示一句「标定失败」，很难定位到是腕部拖累的。
        med: Dict[str, float] = {}
        short: List[str] = []
        for name, vals in self._cal_samples.items():
            if len(vals) < req:
                short.append(f"{name}({len(vals)}<{req})")
                continue
            # 用中位数而非均值：人对准姿势时会不自觉抖动，
            # 中位数对偶发跳变更稳健。
            med[name] = float(sorted(vals)[len(vals) // 2])

        if not med:
            self._last_status = (
                f"标定失败: 所有量样本不足 [{', '.join(short)}]")
            return False

        # samples 取「已建基准的那些量」的样本数中位数。
        # ⚠️ 不要用 min()：部分标定后可能有个量样本为 0，
        #    那样 samples 会被算成 0，HUD 上显示「完成(0样本)」误导排查。
        _cnt = sorted(len(v) for n, v in self._cal_samples.items() if n in med)
        self.neutral = HumanNeutral(
            values=med,
            samples=(_cnt[len(_cnt) // 2] if _cnt else 0),
            # 符号修正随标定一起注入：values 里存的是**原始**测量量，
            # 读取时统一乘 signs，保证「标定基准」与「当前测量」
            # 处在同一个符号约定下。
            signs=dict(self.cfg.measure_sign()))
        if short:
            self._last_status = (f"标定完成({self.neutral.samples}样本)"
                                 f"，未标定: {', '.join(short)}")
        else:
                self._last_status = f"标定完成({self.neutral.samples}样本)"

        # ---- task-space 基线 ----
        # 标定姿势对应「机器人 neutral 末端」，所以
        #   (reach0, elevation0) = 标定得到的人体量
        #   (x0, z0)             = FK(neutral_joints) 的末端位置
        # ---- task-space 基线：人体 (u0, v0) + 机器人 TCP (x0, z0) ----
        # 标定姿势既是人体零点也是机器人零点（任务书 §9）。
        if self.cfg.retarget_mode == "task_space" and self.task_solver is not None:
            u0 = self.neutral.get("human_u")
            v0 = self.neutral.get("human_v")
            if u0 is not None and v0 is not None:
                names = self.cfg.joint_names
                try:
                    q2 = self.cfg.active_neutral_joints()[names.index("joint2")]
                    q3 = self.cfg.active_neutral_joints()[names.index("joint3")]
                    x0, z0 = self.task_solver.fk_xz(q2, q3)
                    self.set_task_baseline(u0, v0, x0, z0)
                except (ValueError, IndexError):
                    pass

        # 标定完成后重置角度/关节滤波，
        # 否则滤波器还带着标定期间的历史值，会造成初始漂移。
        self.angle_filter.reset()
        self.joint_filter.reset(self._target)
        return True

    # ============================================================
    # 主更新
    # ============================================================
    def update(self,
               arm,
               openness: Optional[float] = None,
               now: Optional[float] = None,
               frame: int = 0,
               raw_geometry: Optional[ArmGeometry] = None,
               image_size: Optional[Tuple[int, int]] = None,
               hand_pose=None
               ) -> RetargetResult:
        """
        单帧更新。

        Args:
            arm:          原始 ArmKeypoints（**未滤波**）
            openness:     手掌张开度 [0,1]（None 表示无手部数据）
            now:          当前时刻（**绝对值**，time.monotonic()）
            frame:        帧号（仅用于调试记录）
            raw_geometry: 已用原始关键点算好的几何量（可选）。
                          传入时用于 DebugSnapshot 的 raw 列，
                          避免重复计算。
            image_size:   (width, height)。配置 geometry.pivot="anchor" 时
                          **必须**提供，否则几何无效（不静默退回错误的基准）。
            hand_pose:    手部姿态（perception.compute_hand_pose 的结果），
                          提供 wrist_pitch_deg / palm_roll_deg 用于驱动腕关节。
                          为 None 时腕部量缺失，相关规则会被跳过（不是伪造 0）。

        Returns:
            RetargetResult
        """
        if now is None:
            now = time.monotonic()

        self._update_count += 1

        # ---- ① 关键点滤波 ----
        arm_f = self.kp_filter.update_arm(arm)

        # ---- 状态机判定（用滤波后的关键点，与后续计算保持一致）----
        state = self.tracker.update(arm_f, now)

        # ---- 未标定：什么都不做 ----
        if self.neutral is None:
            self._skip_count += 1
            return RetargetResult(
                positions=list(self._target), changed=False,
                status="未标定", state=state,
                debug=self._make_debug(frame, now, state, arm, raw_geometry,
                                       openness, {}))

        # ---- ② 几何（用滤波后的关键点）----
        # 枢轴选择来自配置：shoulder = 人体肩点（原始），
        # anchor = 画面底边中点（伸展方向更直观，且不依赖肩点质量）。
        geo = compute_arm_geometry(
            arm_f,
            min_confidence=self.cfg.human.min_confidence,
            require_complete=self.cfg.human.require_complete,
            pivot=self.cfg.geometry.pivot,
            image_size=image_size,
            anchor=(self.cfg.geometry.anchor_x, self.cfg.geometry.anchor_y))
        self._last_geom = geo

        # 腕部姿态（来自 hand.compute_hand_pose）挂在实例上，
        # 由 _do_tracking 在构建 measure 字典时合并 ——
        # 那里才是「所有测量量汇成一份字典」的地方。
        self._hand_pose = hand_pose
        # 手臂方向（肩->腕），方向驱动映射的主控量。
        # 用**滤波后**的关键点算，与链路其余部分一致；
        # 在 update 里算好挂实例，_do_tracking 再并进 measure 字典。
        self._arm_dir = (arm_direction_deg(arm_f.shoulder, arm_f.effective_wrist)
                         if arm_f is not None else None)
        # task-space 人体状态（手腕相对肩的归一化位置 u/v），用滤波后关键点
        self._task_state = (compute_human_wrist_task(
            arm_f.shoulder, arm_f.elbow, arm_f.effective_wrist)
            if arm_f is not None else None)

        # 原始几何（仅用于 debug 对比）
        geo_raw = raw_geometry if raw_geometry is not None else geo

        # ---- 状态分派 ----
        if state is TrackState.TRACKING:
            res = self._do_tracking(geo, now)
        elif state is TrackState.LOST_SHORT:
            # 短暂丢失 -> 保持最后位置，绝对不动
            self._skip_count += 1
            res = RetargetResult(
                positions=list(self._target), changed=False,
                status=f"手臂丢失 {self.tracker.lost_duration:.1f}s（保持）",
                state=state)
        else:
            # 长时间丢失 -> 安全状态
            res = self._do_lost_long(now)

        res.debug = self._make_debug(frame, now, state, arm, geo_raw,
                                     openness, res.mapped_joints
                                     if hasattr(res, "mapped_joints") else {})
        res.debug.final_joints = {
            n: self._target[self.cfg.joint_names.index(n)]
            for n in self.cfg.active_joint_names()
        }
        # 补全 filtered 角度。
        # ⚠️ 这里必须把所有受控测量量都填上 ——
        #    曾经只填了 upper，导致 CSV 里 filtered_elbow / filtered_forearm
        #    恒为 0，看起来像"滤波把角度抹平了"，实际只是没记录。
        flt_now = self.angle_filter.value()
        res.debug.flt_upper = self._fnum(flt_now.get("upper_arm_angle_deg"))
        res.debug.flt_elbow = self._fnum(flt_now.get("elbow_angle_deg"))
        res.debug.flt_forearm = self._fnum(flt_now.get("forearm_angle_deg"))

        # 补全相对标定姿态的角度差。
        # ⚠️ 同一类陷阱：deltas_deg 在 _do_tracking 里算好了、也放进了
        #    RetargetResult，但 **没有转写到 DebugSnapshot**，
        #    于是 CSV 的 delta_* 三列恒为 0 —— 而关节列明明在动。
        #    这种"看起来像映射没生效"的假象，根源都是一处状态被算出来
        #    却没被记录。凡 RetargetResult 里已算出的量，快照就必须同步。
        dd = getattr(res, "debug_deltas_deg", None) or {}
        res.debug.delta_upper = self._fnum(dd.get("upper_arm_angle_deg"))
        res.debug.delta_elbow = self._fnum(dd.get("elbow_angle_deg"))
        res.debug.delta_forearm = self._fnum(dd.get("forearm_angle_deg"))
        res.debug.delta_wrist_pitch = self._fnum(dd.get("wrist_pitch_deg"))
        res.debug.delta_palm_roll = self._fnum(dd.get("palm_roll_deg"))

        # 腕部原始/滤波值（同样必须回填，否则这两列恒 0）
        hp = getattr(self, "_hand_pose", None)
        if hp is not None and getattr(hp, "valid", False):
            res.debug.raw_wrist_pitch = self._fnum(hp.wrist_pitch_deg)
            res.debug.raw_palm_roll = self._fnum(hp.palm_roll_deg)
        # task-space：raw / filtered / 目标 / IK 结果 全部回填
        ts = getattr(self, "_task_state", None)
        if ts is not None and getattr(ts, "valid", False):
            res.debug.human_u_raw = float(ts.u)
            res.debug.human_v_raw = float(ts.v)
            res.debug.reach_raw = float(ts.reach)
        res.debug.human_u_flt = self._fnum(flt_now.get("human_u"))
        res.debug.human_v_flt = self._fnum(flt_now.get("human_v"))
        res.debug.reach_flt = self._fnum(flt_now.get("reach"))
        ti = getattr(self, "_last_task_info", None)
        if ti:
            res.debug.du = float(ti.get("du", 0.0))
            res.debug.dv = float(ti.get("dv", 0.0))
            res.debug.du_deadzoned = float(ti.get("du_deadzoned", 0.0))
            res.debug.dv_deadzoned = float(ti.get("dv_deadzoned", 0.0))
            res.debug.tcp_target_x = float(ti.get("x_target", 0.0))
            res.debug.tcp_target_z = float(ti.get("z_target", 0.0))
            res.debug.x_target = float(ti.get("x_target", 0.0))
            res.debug.z_target = float(ti.get("z_target", 0.0))
            res.debug.x_target_raw = float(ti.get("x_target_raw", 0.0))
            res.debug.z_target_raw = float(ti.get("z_target_raw", 0.0))
            res.debug.x_clamped = float(ti.get("x_clamped", 0.0))
            res.debug.z_clamped = float(ti.get("z_clamped", 0.0))
            res.debug.ik_x = float(ti.get("ik_x", 0.0))
            res.debug.ik_z = float(ti.get("ik_z", 0.0))
            res.debug.ik_q2 = float(ti.get("ik_q2", 0.0))
            res.debug.ik_q3 = float(ti.get("ik_q3", 0.0))
            res.debug.ik_err_x = float(ti.get("ik_err_x", 0.0))
            res.debug.ik_err_z = float(ti.get("ik_err_z", 0.0))
            res.debug.ik_jump = float(ti.get("ik_jump", 0.0))
            res.debug.ik_iters = float(ti.get("ik_iters", 0.0))
            res.debug.ik_time_ms = float(ti.get("ik_time_ms", 0.0))

        res.debug.flt_wrist_pitch = self._fnum(flt_now.get("wrist_pitch_deg"))
        res.debug.flt_palm_roll = self._fnum(flt_now.get("palm_roll_deg"))
        res.debug.flt_thumb_offset = self._fnum(flt_now.get("thumb_offset"))
        res.debug.delta_thumb_offset = self._fnum(dd.get("thumb_offset"))
        if hp is not None and getattr(hp, "valid", False):
            res.debug.raw_thumb_offset = self._fnum(hp.thumb_offset)
            res.debug.raw_pinch_distance = self._fnum(hp.pinch_distance)
            res.debug.finger_tips_converged = self._fnum(
                hp.finger_tips_converged)
        return res

    @staticmethod
    def _resolve_table_path(table_path: str, source_path: str) -> str:
        """
        解析 FK 表路径。

        候选顺序（都在「配置所在目录」及其**上一级**下找）：
            1) 配置目录 / table_path
            2) 配置目录的上一级 / table_path
            3) 包内 piper_human_retargeting / table_path

        为什么要试上一级：
            安装后配置在 share/<pkg>/config/retargeting.yaml，
            而 data_files 把 FK 表放在 share/<pkg>/data/。
            两者不同层。写死哪一层都会在另一种布局下失效 ——
            实测就因为这个，装好的表找不到、方向映射静默不生效。
        """
        if os.path.isabs(table_path):
            return table_path
        here = os.path.dirname(os.path.abspath(source_path))
        cands = [
            os.path.join(here, table_path),
            os.path.join(os.path.dirname(here), table_path),
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         table_path),
        ]
        for c in cands:
            if os.path.isfile(c):
                return c
        raise FileNotFoundError(
            f"FK 表未找到: {table_path}；已尝试 {cands}")

    def set_task_baseline(self, u0: float, v0: float,
                          x0: float, z0: float) -> None:
        """
        记录 task-space 标定基准（V1.1）。

            (u0, v0) = 标定姿势下的人体手腕归一化位置
            (x0, z0) = 同姿势下机器人 TCP 的位置（由 FK 算出）

        映射用的是**相对位移** (u-u0, v-v0)，因此人体绝对像素坐标不参与控制
        （任务书 §9）。
        """
        self._ts_baseline = (float(u0), float(v0), float(x0), float(z0))

    # ------------------------------------------------------------
    def task_baseline(self) -> Optional[Tuple[float, float, float, float]]:
        return self._ts_baseline

    def ik_stats(self) -> Dict[str, float]:
        """IK 求解耗时统计（任务书 §14），供 demo 打印与 A/B 对比"""
        if self.task_solver is None:
            return {"n": 0}
        return self.task_solver.stats.summary()

    def _apply_task_space(self, target: List[float],
                          geo_dict: Dict[str, float]) -> Optional[dict]:
        """
        V1.1 task-space 重映射：

            人体 (u, v) --相对标定--> (du, dv) --死区/增益--> TCP 目标 (x, z)
                        --数值 IK（限位在迭代内）--> (q2, q3)

        ⚠️ 肘角**不参与**这里。task_space 模式下 q2/q3 只由 TCP 目标决定
        （任务书 §15）；肘角仍进 CSV，作为将来的姿态偏好（§16）。

        Returns:
            info dict（供 debug/CSV）；未就绪时 None
        """
        if self.task_solver is None or self.ts_mapper is None:
            return None
        u = geo_dict.get("human_u")
        v = geo_dict.get("human_v")
        if u is None or v is None:
            return None

        # ---- 基线懒建立 ----
        # 标定窗口内 u/v 有可能一直拿不到（关键点缺失，或样本因其它量不齐
        # 未采纳）。那样 task-space 会**静默失效**：两种模式输出完全相同，
        # 日志上却只显示「标定完成」。实测踩过这个坑，所以这里退一步：
        # 用第一次拿到有效值的时刻作为基准，并在状态里标注。
        if self._ts_baseline is None:
            names = self.cfg.joint_names
            try:
                nj = self.cfg.active_neutral_joints()
                q2 = nj[names.index("joint2")]
                q3 = nj[names.index("joint3")]
                x0, z0 = self.task_solver.fk_xz(q2, q3)
            except (ValueError, IndexError):
                return None
            self.set_task_baseline(u, v, x0, z0)
            self._late_baseline.add("task_space_baseline")

        u0, v0, x0, z0 = self._ts_baseline
        xt, zt, info = map_human_to_task(u - u0, v - v0, x0, z0,
                                         self.ts_mapper)

        names = self.cfg.joint_names
        q_prev = None
        if "joint2" in names and "joint3" in names:
            q_prev = (target[names.index("joint2")],
                      target[names.index("joint3")])
        nj = self.cfg.active_neutral_joints()
        q_neutral = (nj[names.index("joint2")], nj[names.index("joint3")])

        sol = self.task_solver.solve(xt, zt, q_prev=q_prev,
                                     q_neutral=q_neutral)
        info.update(sol.as_dict())
        info["ik_iters"] = float(sol.iters)
        info["ik_valid"] = float(bool(sol.valid))
        self._last_task_info = info

        if not sol.valid:
            # ⚠️ 不静默继续：未收敛时保持上一帧关节值，并把原因写进状态。
            if sol.reason not in self._ts_warned:
                self._ts_warned.add(sol.reason)
            self._last_status = f"IK {sol.reason}"
            return info

        for name, val in (("joint2", sol.q2), ("joint3", sol.q3)):
            if name in names:
                target[names.index(name)] = float(val)
        if sol.jump > self.cfg.task_space.jump_warn_rad:
            self._last_status = f"IK 跳变 {sol.jump:.3f} rad"
        return info

    def _apply_arm_direction(self, target: List[float],
                             deltas_deg: Dict[str, float],
                             signs: Dict[str, float]) -> None:
        """
        方向驱动：把「人体肩->腕方向角」映射成机械臂 joint2/joint3。

        映射关系（相对标定姿势）：
            人方向变化 Δ = 人当前方向 - 标定时的方向
            机械臂目标方向 = neutral_dir_deg + gain × Δ
            再用 FK 表反解出 (joint2, joint3)

        Δ 的符号即方向旋转方向：人臂在图像里逆时针转，
        机械臂也在矢状面内逆时针转（两者都取「向上为正」）。
        """
        if self.dir_mapper is None or self.neutral is None:
            return
        cfg = self.cfg.arm_direction
        base = self.neutral.get("arm_direction_deg")
        if base is None:
            return
        cur = deltas_deg.get("arm_direction_deg")
        if cur is None:
            return

        # deltas_deg 里存的已经是「当前 - 基准」。
        # ⚠️ 方向角**不参与 sign 修正**：它的正负本身就是旋转方向，
        #    乘 -1 等于把「抬手」变成「放手」，是语义错误。
        #    因此这里直接用 cur，不乘 signs。
        delta = cur

        target_dir = cfg.neutral_dir_deg + cfg.gain * delta

        # 伸展量由肘伸展度决定（可选）
        reach = None
        if cfg.reach_from_elbow:
            eb = self._last_geom.elbow_angle_deg if self._last_geom else None
            if eb is not None:
                lo, hi = cfg.elbow_straight_deg, cfg.elbow_bent_deg
                t = (eb - lo) / (hi - lo) if hi > lo else 0.0
                t = min(1.0, max(0.0, t))
                reach = cfg.reach_extended + t * (cfg.reach_folded
                                                  - cfg.reach_extended)

        sol = self.dir_mapper.solve(target_dir, reach)
        if sol is None:
            return
        j2, j3 = sol
        for name, val in (("joint2", j2), ("joint3", j3)):
            if name in self.cfg.joint_names:
                target[self.cfg.joint_names.index(name)] = float(val)

    @staticmethod
    def _fnum(v) -> float:
        """None -> 0.0，便于写入 CSV 的数值列"""
        return 0.0 if v is None else float(v)

    # ------------------------------------------------------------
    def _do_tracking(self, geo: ArmGeometry, now: float) -> RetargetResult:
        """TRACKING：正常映射"""
        # 若刚从丢失恢复，把人的基准重置为当前姿势，
        # 否则机械臂会从初始位姿猛跳回旧映射点
        resumed = self._was_lost
        if resumed:
            self.angle_filter.reset()
            self.joint_filter.reset(self._target)
            if self.cfg.safety.resume_on_reacquire:
                geo_dict = geo.as_dict()
                for name in self.cfg.used_measures():
                    v = geo_dict.get(name)
                    if v is not None:
                        # 存**原始**测量量；符号在 HumanNeutral.get 里施加。
                        # 与「重新标定即重置基准」同理：恢复映射时也要把
                        # 人当下的姿势当作新的零点，否则机械臂会跳回旧映射点。
                        self.neutral.values[name] = float(v)
        self._was_lost = False

        if not geo.valid:
            # 状态机说 TRACKING 但几何算不出来（例如关键点过短）
            # -> 保守保持不动，不放行错误值
            self._skip_count += 1
            return RetargetResult(
                positions=list(self._target), changed=False,
                status=f"几何无效({geo.reason})",
                state=TrackState.TRACKING)

        # ---- ③ 角度滤波 ----
        geo_dict = geo.as_dict()
        # 合并腕部姿态量。它们在 ArmGeometry 里没有（那是纯手臂几何），
        # 来自 hand.compute_hand_pose。
        # ⚠️ hand_pose 无效时**不填 0**：相关规则会走「缺少测量量」分支
        #    保持不动，而不是把腕关节拉到 0 位（那是伪造观测）。
        hp = getattr(self, "_hand_pose", None)
        if hp is not None and getattr(hp, "valid", False):
            geo_dict.update(hp.as_dict())

        _adir = getattr(self, "_arm_dir", None)
        if _adir is not None:
            geo_dict["arm_direction_deg"] = _adir

        # task-space 量（reach / elevation）
        ts = getattr(self, "_task_state", None)
        if ts is not None and getattr(ts, "valid", False):
            geo_dict.update(ts.as_dict())
        flt = self.angle_filter.update(
            {k: geo_dict[k] for k in self.cfg.used_measures()
             if k in geo_dict})

        # ---- ④ 相对标定姿态的角度差 + 区间映射 ----
        #
        # ⚠️ 这里取 raw 还是 filtered，是有语义差别的一步：
        #    若映射直接吃「未滤波」的 geo_dict，角度滤波器对**下发值毫无影响**
        #    （只被记录进 CSV），三级滤波实际退化为两级。
        #    实测差异很大（300 帧录像，相邻帧抖动中位数）：
        #        上臂 4.81° -> 0.93°，肘 3.90° -> 2.43°，前臂 2.80° -> 2.02°
        #    由 mapping.use_filtered_angles 显式选择，
        #    默认 False 保持历史行为，不悄悄改变已标定的手感。
        src = flt if self.cfg.use_filtered_angles else geo_dict

        # 符号修正（配置 mapping.*.sign）。
        # 为什么要有这一步：以人体肩点为枢轴时，前臂角是「肘->腕」的朝向，
        # 实测 corr(手的高度, 前臂角) = +0.561 —— **手往下移、角度反而变大**，
        # 映射到机械臂就是「人抬手、机器人缩回」的反直觉行为。
        # 在测量量上乘 -1，「手抬高 = 数值变大」就成立了。
        # ⚠️ neutral.get() 也施加同一个符号，所以做差时符号自动一致。
        signs = self.cfg.measure_sign()

        # 逐量算 delta。**某个量缺失时只跳过依赖它的规则**，
        # 不再整帧返回 ——
        # 否则「手部没识别到」会把手臂关节一起停掉
        # （实测：手部量缺失导致 1180/1268 帧被整体跳过、关节一次没动）。
        deltas_deg: Dict[str, float] = {}
        missing: List[str] = []
        for name in self.cfg.used_measures():
            base = self.neutral.get(name)
            cur = src.get(name)
            if base is None and cur is not None:
                # ---- 基准懒建立 ----
                # 某个量在标定窗口内一直不可用（典型：手部姿态要到
                # 11s 后才稳定检出），它的基准就是空的，
                # 对应关节会**永远不动** —— 实测踩过：
                # delta_thumb_offset 恒为 0、joint6 全程 0。
                #
                # 这里退一步：把该量**首次可用时的值**当作零点。
                # 语义上等价于「以你手进入画面的那一刻为基准」，
                # 对操作者是自然的；代价是这一次的绝对姿势不再参与标定。
                # 只对「标定时就缺」的量生效，不影响正常标定的量。
                self.neutral.values[name] = float(cur)
                self.neutral.signs[name] = 1.0   # 存原始值，符号在 get() 施加
                base = self.neutral.get(name)
                self._late_baseline.add(name)
            if base is None or cur is None:
                missing.append(name)
                continue
            cur = cur * signs.get(name, 1.0)
            # 周期归一化：角度是周期量，170 与 -170 的真实差是 20 而不是 340
            deltas_deg[name] = angle_delta_deg(cur, base)

        # 只保留「其测量量本帧可用」的规则
        rules = [r for r in self.cfg.enabled_rules() if r.human in deltas_deg]
        # ⚠️ task_space 模式下 joint2/joint3 由 IK 决定，legacy 规则会被全部停用；
        # 若此时腕部量也缺失（例如测试里显式关掉腕部规则），rules 会是空的 ——
        # 但**不能因此整帧跳过**，否则 task-space 链路根本不执行（静默失效）。
        if not rules and self.task_solver is None:
            self._skip_count += 1
            return RetargetResult(
                positions=list(self._target), changed=False,
                status=f"缺少测量量 {','.join(missing) or '全部'}",
                state=TrackState.TRACKING)
        if not rules:
            self._last_status = "task_space(IK)"
        # 只报告「本该有却没收到」的量：基准里就没有的量
        #（标定时就没采到）不算每帧缺失，否则状态会一直被刷成缺失，
        # 反而掩盖了真正的问题。
        truly = [m for m in missing if self.neutral.get(m) is not None]
        if truly:
            self._last_status = f"部分量缺失({','.join(truly)})"

        mapped, _infos = map_all(
            deltas_deg, rules,
            {n: self._target[self.cfg.joint_names.index(n)]
             for n in self.cfg.fixed_joints})

        # ---- ⑤ 关节滤波 + 死区 ----
        #
        # ⚠️ 这里**故意不再加**「单周期 max_step 粗限幅」。
        #    历史上曾在映射之后、滤波之前加过一层 max_step_rad 截断，
        #    结果是：滤波器算出的值又被截断一次，
        #    最终输出由「alpha 与限幅谁先谁后」决定，
        #    调 alpha 的效果被限幅掩盖（实测：设 target=1.55 却输出 0.7）。
        #
        #    真正的速度/加速度限制在 piper_human_control.safety.SafetyLimiter
        #    里按 min(a_max*dt, v_max*dt) **每关节独立**计算并执行，
        #    那里才是正确的限速位置（知道 dt、知道实测关节角）。
        #    在这里再加一层只会破坏滤波的线性性，且掩盖真实限速行为。
        new_target = list(self._target)
        for name in self.cfg.controlled_joints():
            new_target[self.cfg.joint_names.index(name)] = mapped.get(
                name, self._target[self.cfg.joint_names.index(name)])

        # ---- ④a task-space 覆盖（V1.1）----
        # 与 legacy 互斥：task_space 模式下 J2/J3 由 reach/elevation 联合求解，
        # legacy 的 j2/j3 规则结果被覆盖。
        if self.cfg.retarget_mode == "task_space":
            self._apply_task_space(new_target, geo_dict)

        # ---- ④b 方向驱动覆盖（可选）----
        # 用「人肩->腕方向」反解 joint2/joint3，直接对齐机械臂的臂方向。
        # 这一步**取代**逐关节映射对 joint2/joint3 的结果，
        # 因为后者无法保证机械臂朝人的方向伸出去（见 ArmDirectionConfig）。
        self._apply_arm_direction(new_target, deltas_deg, signs)

        raw_after_map = list(new_target)

        new_target = self.joint_filter.update(new_target)
        # 固定关节不受滤波影响
        for name, val in self.cfg.fixed_joints.items():
            new_target[self.cfg.joint_names.index(name)] = float(val)

        changed = any(abs(a - b) > 1e-9
                      for a, b in zip(new_target, self._target))
        self._target = new_target
        self._last_status = "正常(已重置基准)" if resumed else "正常"

        res = RetargetResult(
            positions=list(self._target), changed=changed,
            status=self._last_status, state=TrackState.TRACKING,
            deltas={n: self._target[self.cfg.joint_names.index(n)]
                    - self.cfg.neutral_for(n)
                    for n in self.cfg.controlled_joints()})
        # 记录映射后（滤波前）的值，供 debug 对比
        # ⚠️ 受控关节集合要带上 task_space 的关节（joint2/joint3）。
        #    它们的 legacy 规则被停用、不在 controlled_joints 里，
        #    只按 controlled_joints 记录会让 CSV 的 mapped_joint2/3 恒为 0 ——
        #    实时测试实测踩到：IK 明明输出 q2=0.924/q3=-0.907，
        #    CSV 里却写着 0.0000，看起来像"这两个关节没被控制"。
        _ctrl = self.cfg.active_joint_names()
        res.mapped_joints = {
            n: raw_after_map[self.cfg.joint_names.index(n)] for n in _ctrl}
        res.debug_deltas_deg = deltas_deg
        return res

    # ------------------------------------------------------------
    def _do_lost_long(self, now: float) -> RetargetResult:
        """
        LOST_LONG：安全状态。

        两种策略（配置决定）：
            return_home_when_lost=true   平滑回初始位姿
            false                        冻结最后目标（不执行 HOME、不回位）

        ★ 无论哪种，都**不会**突然跳到 HOME。
        """
        self._was_lost = True
        lost_for = self.tracker.lost_duration

        if not self.cfg.safety.return_home_when_lost:
            self._skip_count += 1
            return RetargetResult(
                positions=list(self._target), changed=False,
                status=f"手臂长时间丢失 {lost_for:.1f}s（冻结保持）",
                state=TrackState.LOST_LONG)

        if lost_for < self.cfg.safety.lost_grace_seconds:
            self._skip_count += 1
            return RetargetResult(
                positions=list(self._target), changed=False,
                status=(f"手臂丢失 {lost_for:.1f}s/"
                        f"{self.cfg.safety.lost_grace_seconds:.1f}s（保持）"),
                state=TrackState.LOST_LONG)

        # ---- 平滑回初始位姿 ----
        # ⚠️ 必须遍历 active_joint_names()（含 task_space 的 joint2/joint3）。
        #    只遍历 controlled_joints() 时，task_space 模式下**大臂不会回位**，
        #    只有腕关节归零 —— 实测表现："识别不到时机械臂停在原地不动"。
        home = self.initial_pose
        speed = self.cfg.safety.return_home_speed
        new_target = list(self._target)
        for name in self.cfg.active_joint_names():
            idx = self.cfg.joint_names.index(name)
            cur, tgt = self._target[idx], home[idx]
            if abs(tgt - cur) < 1e-4:
                new_target[idx] = tgt
            else:
                new_target[idx] = cur + (tgt - cur) * speed
        # 回位路径也要过一遍关节滤波，保证与正常控制同样的平滑度
        new_target = self.joint_filter.update(new_target)
        for name, val in self.cfg.fixed_joints.items():
            new_target[self.cfg.joint_names.index(name)] = float(val)

        moved = any(abs(a - b) > 1e-9
                    for a, b in zip(new_target, self._target))
        self._target = new_target

        # "已回到原位"的判据必须**大于关节滤波死区**，否则永远判不到：
        # 实测 deadband=0.008 rad，于是残差停在 ~0.011 rad（0.6°），
        # 状态行会一直显示"回初始位姿中"，看起来像没回完。
        # 0.6° 在物理上就是原位了，故取 max(1e-3, 2×deadband)。
        _tol = max(1e-3, 2.0 * float(getattr(self.cfg.filter_joint,
                                            "deadband", 0.0)))
        at_home = all(
            abs(self._target[self.cfg.joint_names.index(n)]
                - home[self.cfg.joint_names.index(n)]) < _tol
            for n in self.cfg.active_joint_names())

        return RetargetResult(
            positions=list(self._target), changed=moved,
            status=("手臂丢失（已回到初始位姿）" if at_home
                    else f"手臂长时间丢失 {lost_for:.1f}s（回初始位姿中）"),
            state=TrackState.LOST_LONG)

    # ------------------------------------------------------------
    def _make_debug(self, frame: int, now: float, state: TrackState,
                    arm, geo_raw: Optional[ArmGeometry],
                    openness: Optional[float],
                    mapped: Dict[str, float]) -> DebugSnapshot:
        """构造调试快照（raw / filtered 对比 + CSV 记录用）"""
        d = DebugSnapshot(t=now, frame=frame, state=str(state))
        if arm is not None:
            w = arm.effective_wrist
            if arm.shoulder is not None:
                d.shoulder_xy = (arm.shoulder.x, arm.shoulder.y)
            if arm.elbow is not None:
                d.elbow_xy = (arm.elbow.x, arm.elbow.y)
            if w is not None:
                d.wrist_xy = (w.x, w.y)
        if geo_raw is not None and geo_raw.valid:
            d.raw_upper = geo_raw.upper_arm_angle_deg
            d.raw_elbow = geo_raw.elbow_angle_deg
            d.raw_forearm = geo_raw.forearm_angle_deg
        d.mapped_joints = dict(mapped)
        # ---- 兜底：关节列必须永远写全 ----
        # 早退路径（手臂丢失 / 标定中 / 缺测量量）不会走到"填 final_joints"
        # 那一步，于是 CSV 里 joint2/joint3 会被 to_csv_row 的 .get(j, 0.0)
        # 填成 **0.0000** —— 而机械臂其实稳稳保持在 1.504/−0.99。
        # 实测踩到：实时测试前 25 帧看起来像"关节归零"。
        # 这里统一用当前实际目标兜底，保证"记录到的值 = 真正下发目标"。
        for _j in self.cfg.active_joint_names():
            _idx = self.cfg.joint_names.index(_j)
            d.final_joints.setdefault(_j, float(self._target[_idx]))
            d.mapped_joints.setdefault(_j, float(self._target[_idx]))
        if openness is not None:
            d.openness = float(openness)
        return d

    # ============================================================
    # 查询
    # ============================================================
    @property
    def target(self) -> List[float]:
        return list(self._target)

    @property
    def is_calibrated(self) -> bool:
        return self.neutral is not None

    @property
    def state(self) -> TrackState:
        return self.tracker.state

    @property
    def last_geometry(self) -> Optional[ArmGeometry]:
        return self._last_geom

    @property
    def last_status(self) -> str:
        return self._last_status

    @property
    def stats(self) -> Dict[str, int]:
        s = {"updates": self._update_count, "skips": self._skip_count}
        s.update(self.tracker.stats())
        return s

    def reset_target_to_neutral(self) -> List[float]:
        self._target = list(self.cfg.neutral_joints)
        self._apply_fixed_joints()
        self.joint_filter.reset(self._target)
        return list(self._target)
