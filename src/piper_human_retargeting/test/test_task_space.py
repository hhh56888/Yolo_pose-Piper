# -*- coding: utf-8 -*-
"""
2D Task-Space Retargeting V1.1 测试
===================================
覆盖：人体 u/v 定义、死区、映射、数值 IK（限位在迭代内 / 不跳解 / 耗时统计）、
workspace 扫描与 neutral 评分、两种模式共存。

运行：
    python3 -m pytest test/test_task_space.py -q -p no:anyio
"""
import math
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_HERE)
_SRC = os.path.dirname(_PKG_ROOT)
for _p in (_PKG_ROOT, os.path.join(_SRC, "piper_human_control"),
           os.path.join(_SRC, "piper_human_perception")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from piper_human_retargeting.task_space import (          # noqa: E402
    TaskMappingConfig, TaskSpaceIk, apply_deadzone, compute_human_wrist_task,
    map_human_to_task, neutral_candidates, scan_workspace)


class Kp:
    """最小关键点替身（只需要 x/y/is_valid）"""

    def __init__(self, x, y, valid=True):
        self.x, self.y, self.is_valid = float(x), float(y), bool(valid)


# ============================================================
# ① 人体 u / v
# ============================================================
class TestHumanWristTask:
    def test_shoulder_elbow_wrist_colinear_gives_reach_one(self):
        t = compute_human_wrist_task(Kp(0, 0), Kp(100, 0), Kp(200, 0))
        assert t.valid
        assert t.reach == pytest.approx(1.0, abs=1e-9)
        assert t.u == pytest.approx(1.0, abs=1e-9)     # 图像右
        assert t.v == pytest.approx(0.0, abs=1e-9)

    def test_v_is_positive_when_hand_above_shoulder(self):
        """图像 y 向下：手在肩上方 -> v > 0（这是"抬手"的语义）"""
        up = compute_human_wrist_task(Kp(0, 200), Kp(70, 130), Kp(140, 60))
        assert up.valid and up.v > 0
        down = compute_human_wrist_task(Kp(0, 200), Kp(70, 270), Kp(140, 340))
        assert down.valid and down.v < 0

    def test_u_sign_follows_image_right(self):
        right = compute_human_wrist_task(Kp(0, 0), Kp(50, 0), Kp(100, 0))
        left = compute_human_wrist_task(Kp(200, 0), Kp(150, 0), Kp(100, 0))
        assert right.u > 0 > left.u

    def test_scale_invariance(self):
        """u/v 用 |SE|+|EW| 归一化 -> 与人体尺度、与相机距离无关"""
        a = compute_human_wrist_task(Kp(0, 0), Kp(100, 0), Kp(150, 86.6))
        b = compute_human_wrist_task(Kp(0, 0), Kp(200, 0), Kp(300, 173.2))
        assert a.u == pytest.approx(b.u, abs=1e-9)
        assert a.v == pytest.approx(b.v, abs=1e-9)

    @pytest.mark.parametrize("s,e,w", [
        (None, Kp(100, 0), Kp(200, 0)),
        (Kp(0, 0, False), Kp(100, 0), Kp(200, 0)),
        (Kp(0, 0), Kp(100, 0), None),
        (Kp(0, 0), Kp(1, 0), Kp(200, 0)),       # 上臂过短
        (Kp(0, 0), Kp(100, 0), Kp(101, 0)),     # 前臂过短
    ])
    def test_invalid_cases_have_reason(self, s, e, w):
        t = compute_human_wrist_task(s, e, w)
        assert not t.valid and t.reason

    def test_as_dict_keys(self):
        t = compute_human_wrist_task(Kp(0, 0), Kp(100, 0), Kp(150, 86.6))
        d = t.as_dict()
        assert {"human_u", "human_v", "reach"} <= set(d)


# ============================================================
# ② 死区
# ============================================================
class TestDeadzone:
    def test_inside_is_zero(self):
        assert apply_deadzone(0.01, 0.02) == 0.0
        assert apply_deadzone(-0.019, 0.02) == 0.0

    def test_outside_is_shifted_not_passed_through(self):
        """超界后必须减去死区，否则边界处有台阶（0 -> 0.02 的跳变）"""
        assert apply_deadzone(0.05, 0.02) == pytest.approx(0.03)
        assert apply_deadzone(-0.05, 0.02) == pytest.approx(-0.03)

    def test_continuity_at_boundary(self):
        eps = 1e-9
        lo = apply_deadzone(0.02 - eps, 0.02)
        hi = apply_deadzone(0.02 + eps, 0.02)
        assert abs(hi - lo) < 1e-6

    def test_zero_deadzone_is_identity(self):
        assert apply_deadzone(0.37, 0.0) == pytest.approx(0.37)


# ============================================================
# ③ 人体 -> TCP 目标
# ============================================================
class TestTaskMapping:
    def cfg(self, **kw):
        base = dict(horizontal_axis="x", vertical_axis="z",
                    sign_h=1.0, sign_v=1.0, kx=0.5, kz=0.5,
                    x_min=-0.2, x_max=0.6, z_min=0.0, z_max=0.8,
                    deadzone_u=0.0, deadzone_v=0.0)
        base.update(kw)
        return TaskMappingConfig(**base)

    def test_zero_delta_keeps_baseline(self):
        x, z, info = map_human_to_task(0.0, 0.0, 0.12, 0.35, self.cfg())
        assert (x, z) == pytest.approx((0.12, 0.35))
        assert info["x_clamped"] == 0.0 and info["z_clamped"] == 0.0

    def test_horizontal_only_moves_x(self):
        x, z, _ = map_human_to_task(0.3, 0.0, 0.10, 0.30, self.cfg())
        assert x == pytest.approx(0.10 + 0.5 * 0.3)
        assert z == pytest.approx(0.30)          # 轴解耦：水平不该动 z

    def test_vertical_only_moves_z(self):
        x, z, _ = map_human_to_task(0.0, -0.2, 0.10, 0.30, self.cfg())
        assert x == pytest.approx(0.10)
        assert z == pytest.approx(0.30 - 0.5 * 0.2)

    def test_sign_flips_direction(self):
        cfg = self.cfg(sign_h=-1.0, sign_v=-1.0)
        x, z, _ = map_human_to_task(0.3, 0.2, 0.0, 0.4, cfg)
        assert x < 0.0 and z < 0.4

    def test_clamp_reported(self):
        x, z, info = map_human_to_task(5.0, 5.0, 0.0, 0.4, self.cfg())
        assert x == pytest.approx(0.6) and z == pytest.approx(0.8)
        assert info["x_clamped"] == 1.0 and info["z_clamped"] == 1.0

    def test_deadzone_applied_in_mapping(self):
        cfg = self.cfg(deadzone_u=0.05, kx=1.0)
        x1, _, _ = map_human_to_task(0.04, 0.0, 0.0, 0.0, cfg)
        x2, _, _ = map_human_to_task(0.10, 0.0, 0.0, 0.0, cfg)
        assert x1 == pytest.approx(0.0)
        assert x2 == pytest.approx(0.05)          # 0.10 - 0.05 死区

    def test_axis_is_configurable(self):
        """相机从正面看时，"图像水平"不该被硬编码成机器人 X"""
        cfg = self.cfg(horizontal_axis="y")
        assert cfg.horizontal_axis == "y"
        cfg.validate()

    @pytest.mark.parametrize("kw", [
        dict(horizontal_axis="q"),
        dict(horizontal_axis="z", vertical_axis="z"),
        dict(sign_h=2.0),
        dict(x_min=0.5, x_max=0.1),
        dict(z_min=0.5, z_max=0.1),
        dict(kx=0.0),
        dict(deadzone_u=-1.0),
    ])
    def test_bad_config_rejected(self, kw):
        with pytest.raises(ValueError):
            self.cfg(**kw).validate()


# ============================================================
# ④ 数值 IK
# ============================================================
class TestTaskSpaceIk:
    @pytest.fixture
    def ik(self):
        return TaskSpaceIk()

    def test_hits_reachable_targets(self, ik):
        """
        目标取自已扫描出的**真实可达点**（由 FK 生成），必须精确命中。

        注意：不能拿 (x,z) 矩形里的随机点当"可达目标" ——
        两连杆的可达集是平面上的厚环带，矩形里大量点根本不可达
        （实测 27/200）。那种情况 IK 只能给最近点，属**正确行为**
        （由 test_unreachable_target_reports_invalid_not_silent 覆盖）。
        """
        pts = scan_workspace(n2=25, n3=25)
        rng = np.random.default_rng(0)
        worst, qp = 0.0, None
        for _ in range(200):
            p = pts[int(rng.integers(0, len(pts)))]
            s = ik.solve(p.x, p.z, q_prev=qp, q_neutral=(1.5, -1.0))
            qp = (s.q2, s.q3)
            worst = max(worst, math.hypot(s.err_x, s.err_z))
        assert worst < ik.pos_tol, f"最大误差 {worst * 1000:.2f} mm"

    def test_solution_always_inside_box(self, ik):
        """限位在迭代内：任何解都必须落在 IK 区间内（不是最后再 clamp）"""
        rng = np.random.default_rng(1)
        qp = None
        for _ in range(300):
            x, z = rng.uniform(-0.5, 0.8), rng.uniform(0.0, 0.9)
            s = ik.solve(x, z, q_prev=qp, q_neutral=(1.0, -1.2))
            qp = (s.q2, s.q3)
            assert ik.q_min[0] - 1e-12 <= s.q2 <= ik.q_max[0] + 1e-12
            assert ik.q_min[1] - 1e-12 <= s.q3 <= ik.q_max[1] + 1e-12

    def test_unreachable_target_reports_invalid_not_silent(self, ik):
        """目标不可达时必须显式无效 + 给原因，不能静默返回"看起来正常"的解"""
        s = ik.solve(5.0, 5.0, q_prev=(1.0, -1.0))
        assert not s.valid
        assert s.reason

    def test_smooth_trajectory_has_no_solution_jump(self, ik):
        """沿平滑轨迹走：warm start + lambda_prev 应让相邻帧解连续"""
        xs = np.linspace(0.05, 0.35, 120)
        zs = 0.35 + 0.10 * np.sin(np.linspace(0, 2 * math.pi, 120))
        qp, jumps = None, []
        for x, z in zip(xs, zs):
            s = ik.solve(float(x), float(z), q_prev=qp, q_neutral=(1.0, -1.2))
            if qp is not None:
                jumps.append(math.hypot(s.q2 - qp[0], s.q3 - qp[1]))
            qp = (s.q2, s.q3)
        assert max(jumps) < 0.15, f"最大单帧跳变 {max(jumps):.3f} rad"

    def test_lambda_prev_reduces_jump(self):
        """lambda_prev 的作用必须可测：调大它，单帧跳变不应变大"""
        def run(lam):
            ik = TaskSpaceIk(lambda_prev=lam, lambda_neutral=0.0)
            qp, jumps = None, []
            for x in np.linspace(0.05, 0.35, 60):
                s = ik.solve(float(x), 0.30, q_prev=qp,
                             q_neutral=(1.0, -1.2))
                if qp:
                    jumps.append(math.hypot(s.q2 - qp[0], s.q3 - qp[1]))
                qp = (s.q2, s.q3)
            return max(jumps)
        assert run(2.0) <= run(0.0) + 1e-9

    def test_cost_penalises_distance_from_neutral(self):
        """目标函数层面：同一个 q，离 neutral 越远代价越大"""
        ik = TaskSpaceIk(lambda_neutral=0.1)
        q = np.array([1.2, -1.0])
        near = ik.cost(q, 0.3, 0.3, None, np.array([1.2, -1.0]))
        far = ik.cost(q, 0.3, 0.3, None, np.array([2.2, -2.0]))
        assert near < far

    def test_neutral_never_overrides_position_target(self):
        """
        任务书 §12：neutral 只是轻微偏好，**不能压过 TCP 位置目标**。
        把 lambda_neutral 从 0 拉到 1.0，可达目标的命中误差都必须保持合格。
        """
        pts = scan_workspace(n2=21, n3=21)
        p = pts[len(pts) // 2]
        for lam in (0.0, 0.01, 0.1, 1.0):
            ik = TaskSpaceIk(lambda_neutral=lam)
            s = ik.solve(p.x, p.z, q_neutral=(0.3, -0.4))
            assert math.hypot(s.err_x, s.err_z) < ik.pos_tol, (
                f"lambda_neutral={lam} 时位置目标被压过: "
                f"err={math.hypot(s.err_x, s.err_z) * 1000:.1f}mm")

    def test_box_is_injective_over_reachable_set(self):
        """
        记录一个实测事实：在配置的关节盒内，可达 (x,z) 对应的 (q2,q3) 是**唯一**的
        （实测 120 个随机可达目标、两种初值都收敛到同一解）。
        因此 lambda_prev 的作用是"平滑"而不是"在两个 IK 分支间选边"，
        这也是不依赖滞回逻辑（旧实现的 ik_hysteresis）的原因。
        """
        pts = scan_workspace(n2=21, n3=21)
        rng = np.random.default_rng(1)
        for _ in range(40):
            p = pts[int(rng.integers(0, len(pts)))]
            a = TaskSpaceIk(lambda_prev=0, lambda_neutral=0).solve(
                p.x, p.z, q_prev=(0.35, -0.4), record_stats=False)
            b = TaskSpaceIk(lambda_prev=0, lambda_neutral=0).solve(
                p.x, p.z, q_prev=(2.35, -2.25), record_stats=False)
            assert a.valid and b.valid
            assert math.hypot(a.q2 - b.q2, a.q3 - b.q3) < 0.3

    def test_stats_recorded(self, ik):
        for _ in range(20):
            ik.solve(0.2, 0.3, q_neutral=(1.0, -1.0))
        s = ik.stats.summary()
        assert s["n"] == 20
        assert s["median_ms"] <= s["p95_ms"] <= s["max_ms"]
        assert s["mean_iters"] > 0

    def test_solve_is_fast_enough_for_30hz(self, ik):
        """
        30 Hz 主循环 = 33 ms/帧，IK 必须远低于此。

        用**可达点**测（FK 生成），这样统计的是正常跟踪路径（不做重试）；
        不可达目标的耗时由 test_unreachable... 覆盖。

        阈值取 P95 < 3 ms（30 Hz 预算 33 ms 的 1/10）：
          * 空载实测 0.55 ms（本机）/ 0.63 ms（远端）
          * 远端**同时跑实时识别**（YOLO 30 Hz）时实测 ~2 ms ——
            这是负载差异，不是回归，故留足余量。
        真实数字记录在 docs/2D_TaskSpace_Retargeting_V1_1_Report.md §7.2。
        """
        pts = scan_workspace(n2=25, n3=25)
        qp = None
        for i in range(100):                     # 预热（不计入统计）
            p = pts[i % len(pts)]
            s = ik.solve(p.x, p.z, q_prev=qp, q_neutral=(1.5, -1.0),
                         record_stats=False)
            qp = (s.q2, s.q3)
        ik.stats.reset()
        for i in range(300):
            p = pts[i % len(pts)]
            s = ik.solve(p.x, p.z, q_prev=qp, q_neutral=(1.5, -1.0))
            qp = (s.q2, s.q3)
        assert ik.stats.summary()["p95_ms"] < 3.0

    def test_at_limit_flags(self, ik):
        """目标在区间外沿时，应报告"贴限位"，而不是假装正常"""
        s = ik.solve(-1.0, -1.0, q_neutral=(1.0, -1.0))
        assert any(s.at_lower) or any(s.at_upper)

    def test_bad_box_rejected(self):
        with pytest.raises(ValueError):
            TaskSpaceIk(q_min=(1.0, -1.0), q_max=(0.5, -0.5))


# ============================================================
# ⑤ Workspace 扫描 + Neutral 搜索
# ============================================================
class TestWorkspaceAndNeutral:
    @pytest.fixture(scope="class")
    def points(self):
        return scan_workspace(n2=21, n3=21)

    def test_scan_shape_and_fields(self, points):
        assert len(points) == 21 * 21
        p = points[0]
        for k in ("q2", "q3", "x", "y", "z", "m2_lo", "m2_hi",
                  "m3_lo", "m3_hi", "sigma_min", "manipulability"):
            assert hasattr(p, k)
        assert all(p.sigma_min >= 0 for p in points)

    def test_scan_is_continuous(self, points):
        """相邻网格点的 TCP 位置不能跳变（FK 连续）"""
        pts = {(round(p.q2, 6), round(p.q3, 6)): p for p in points}
        q2s = sorted({p.q2 for p in points})
        q3s = sorted({p.q3 for p in points})
        d = 0.0
        for q2 in q2s:
            for i in range(len(q3s) - 1):
                a = pts[(round(q2, 6), round(q3s[i], 6))]
                b = pts[(round(q2, 6), round(q3s[i + 1], 6))]
                d = max(d, math.hypot(a.x - b.x, a.z - b.z))
        assert d < 0.05, f"相邻网格 TCP 跳变 {d:.4f} m"

    def test_top5_sorted_and_fields(self, points):
        cands = neutral_candidates(points, top=5)
        assert len(cands) == 5
        assert cands[0].score >= cands[-1].score
        for c in cands:
            assert set(c.margins) == {"joint2", "joint3"}
            assert 0.0 <= c.parts["workspace"] <= 1.0
            assert 0.0 <= c.parts["limit"] <= 1.0
            assert 0.0 <= c.parts["manip"] <= 1.0

    def test_limit_weight_prefers_margin(self, points):
        """只有限位项时，选出的解应离边界足够远"""
        cands = neutral_candidates(points, w_workspace=0.0, w_manip=0.0, top=1)
        best = cands[0]
        assert min(best.margins["joint2"][0], best.margins["joint2"][1],
                   best.margins["joint3"][0], best.margins["joint3"][1]) > 0.1

    def test_empty_input(self):
        assert neutral_candidates([]) == []


# ============================================================
# ⑥ 配置：两种模式共存
# ============================================================
class TestConfigSwitch:
    def cfg(self):
        from piper_human_retargeting import RetargetingConfig
        return RetargetingConfig.from_yaml()

    def test_repo_defaults_to_legacy(self):
        assert self.cfg().retarget_mode == "legacy"

    def test_legacy_keeps_all_rules(self):
        c = self.cfg()
        assert {"joint2", "joint3"} <= set(c.controlled_joints())

    def test_task_space_disables_elbow_to_j3_rule(self):
        """任务书 §15：task_space 模式下肘角不能直接决定 joint3"""
        c = self.cfg()
        c.retarget_mode = "task_space"
        joints = set(c.controlled_joints())
        assert "joint2" not in joints and "joint3" not in joints
        assert {"joint5", "joint6"} <= joints          # 腕部仍由 legacy 规则驱动

    def test_task_space_measures_include_u_v(self):
        c = self.cfg()
        c.retarget_mode = "task_space"
        ms = c.used_measures()
        assert {"human_u", "human_v"} <= set(ms)
        # 诊断/记录量也必须在列里（否则 CSV 里看不到肘角）
        assert "elbow_angle_deg" in ms

    def test_task_space_neutral_is_separate_from_legacy(self):
        c = self.cfg()
        c.retarget_mode = "task_space"
        legacy = list(c.neutral_joints)
        c.task_space.neutral_joints = [0.0, 1.3, -1.2, 0.0, 0.0, 0.0]
        assert c.active_neutral_joints() == [0.0, 1.3, -1.2, 0.0, 0.0, 0.0]
        c.retarget_mode = "legacy"
        assert c.active_neutral_joints() == legacy

    def test_active_initial_pose_matches_mode_neutral(self):
        """
        回归：初始位姿必须与当前模式的中立位姿一致。
        `robot.initial_pose` 在解析时会被 robot.neutral_joints 填充、永远非空，
        启动逻辑若写成 `robot.initial_pose or active_neutral_joints()`，
        task_space 就会用 legacy 位姿启动，而 IK 基线用的是 task-space 位姿
        —— 开机即带固定偏差（实时测试实测踩到）。
        """
        c = self.cfg()
        legacy = list(c.robot.initial_pose)
        c.retarget_mode = "task_space"
        assert c.active_initial_pose() == list(c.task_space.neutral_joints)
        assert c.active_initial_pose() != legacy
        c.retarget_mode = "legacy"
        assert c.active_initial_pose() == legacy

    def test_bad_mode_rejected(self):
        c = self.cfg()
        c.retarget_mode = "task-space"     # 拼错
        with pytest.raises(ValueError):
            c.validate()

    def test_ik_bounds_must_be_inside_safe(self):
        """retarget ⊂ safe ⊂ physical：越界必须在配置校验期被拒绝"""
        from piper_human_retargeting.limits import build_layers
        c = self.cfg()
        ts = c.task_space
        bad = build_layers({"joint2": (ts.q2_min, 3.2),      # 超 physical 上限
                            "joint3": (ts.q3_min, ts.q3_max)},
                           ts.limit_margin_rad)
        assert bad.violations
        ok = build_layers(ts.retarget_ranges(), ts.limit_margin_rad)
        assert not ok.violations
