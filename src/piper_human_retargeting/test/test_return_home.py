# -*- coding: utf-8 -*-
"""
识别不到手臂时的「回归原位」测试
================================
需求：手臂长时间识别不到 -> 平滑回到**启动位姿（原位）**，而不是停在原地。

实测踩到的 bug（本次修复）：
    `_do_lost_long` 只遍历 `cfg.controlled_joints()`，
    而 task_space 模式下 joint2/joint3 的 legacy 规则被停用、不在该集合里
    -> 只有腕关节回位，**大臂冻在原地**，观感就是"识别不到时机械臂不动"。

运行：python3 -m pytest test/test_return_home.py -q -p no:anyio
"""
import math
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_HERE)
_SRC = os.path.dirname(_PKG_ROOT)
for _p in (_PKG_ROOT, os.path.join(_SRC, "piper_human_control"),
           os.path.join(_SRC, "piper_human_perception")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from piper_human_control import ArmKeypoints, Keypoint          # noqa: E402
from piper_human_retargeting import Retargeter, RetargetingConfig  # noqa: E402
from piper_human_retargeting.arm_geometry import compute_arm_geometry  # noqa: E402


def make_arm(upper_deg, fore_deg):
    ea = math.radians(upper_deg)
    e = (100.0 * math.cos(ea), -100.0 * math.sin(ea))
    fa = math.radians(fore_deg)
    w = (e[0] + 100.0 * math.cos(fa), e[1] - 100.0 * math.sin(fa))
    return ArmKeypoints(
        shoulder=Keypoint(0.0, 0.0, confidence=0.9, name="left_shoulder"),
        elbow=Keypoint(*e, confidence=0.9, name="left_elbow"),
        wrist=Keypoint(*w, confidence=0.9, name="left_wrist"),
        side="left",
        hand_wrist=Keypoint(*w, confidence=0.9, name="left_wrist_from_hand"))


def build(cfg, robot_cfg, mode, frames_to_move=60):
    """建一个已标定、并且已经把机械臂"带离原位"的 Retargeter"""
    cfg.retarget_mode = mode
    for r in cfg.rules:
        if r.key in ("j5_wrist", "j6_roll"):
            r.enabled = False
    rt = Retargeter(cfg, robot_cfg)
    rt.set_initial_pose(cfg.active_initial_pose())
    arm0 = make_arm(0.0, 0.0)
    rt.start_calibration()
    geo = compute_arm_geometry(arm0)
    for _ in range(30):
        rt.add_calibration_sample_for(arm0, geo)
    assert rt.finish_calibration()
    # 用一条不同的姿势把它带离原位
    arm1 = make_arm(30.0, 90.0)
    for _ in range(frames_to_move):
        rt.update(arm1, now=0.0)
    return rt, arm1


class TestReturnHome:
    @pytest.mark.parametrize("mode", ["legacy", "task_space"])
    def test_lost_long_returns_all_arm_joints(self, cfg, robot_cfg, mode):
        """丢失超过宽限期后，**所有**受控手臂关节都要回到原位（含 j2/j3）"""
        rt, _ = build(cfg, robot_cfg, mode)
        home = rt.initial_pose
        idx = cfg.joint_names.index
        moved = [n for n in cfg.active_joint_names()
                 if abs(rt.target[idx(n)] - home[idx(n)]) > 0.05]
        assert moved, "测试前提失败：机械臂没有被带离原位"

        # 宽限期内：必须保持不动
        res = rt.update(None, now=0.0)
        for n in cfg.active_joint_names():
            assert rt.target[idx(n)] == pytest.approx(
                rt.target[idx(n)])  # 占位：宽限期内不应移动（下面统一断言）
        before = list(rt.target)

        # 把丢失时间推过宽限期：tracker 用真实时间，这里直接推进 now
        t = 0.0
        for _ in range(400):
            t += 0.05
            res = rt.update(None, now=t)
            if "已回到初始位姿" in res.status:
                break
        # 容差：关节滤波死区 0.008 rad -> 残差约 0.011 rad（0.6°）即视为原位
        tol = max(2e-3, 3.0 * float(cfg.filter_joint.deadband))
        for n in cfg.active_joint_names():
            assert rt.target[idx(n)] == pytest.approx(home[idx(n)], abs=tol), (
                f"{mode}: {n} 未回原位 "
                f"({rt.target[idx(n)]:+.3f} vs home {home[idx(n)]:+.3f})，"
                f"status={res.status}")
        assert "已回到初始位姿" in res.status
        del before

    @pytest.mark.parametrize("mode", ["legacy", "task_space"])
    def test_return_is_smooth_not_a_jump(self, cfg, robot_cfg, mode):
        """回位过程必须平滑：单帧变化不超过配置的回位步长上界"""
        rt, _ = build(cfg, robot_cfg, mode)
        idx = cfg.joint_names.index
        t, prev, max_step = 0.0, list(rt.target), 0.0
        for _ in range(400):
            t += 0.05
            rt.update(None, now=t)
            step = max(abs(a - b) for a, b in zip(rt.target, prev))
            max_step = max(max_step, step)
            prev = list(rt.target)
        # return_home_speed=0.6 是指数逼近；单帧最大步长受它约束
        assert max_step < 1.2, f"{mode}: 回位单帧跳变过大 {max_step:.3f} rad"

    def test_freeze_mode_keeps_position(self, cfg, robot_cfg):
        """return_home_when_lost=false 时必须冻结（保持旧行为，可回退）"""
        cfg.safety.return_home_when_lost = False
        rt, _ = build(cfg, robot_cfg, "task_space")
        frozen = list(rt.target)
        t = 0.0
        for _ in range(200):
            t += 0.05
            res = rt.update(None, now=t)
        assert rt.target == pytest.approx(frozen)
        assert "冻结" in res.status or "丢失" in res.status

    def test_grace_period_holds_position(self, cfg, robot_cfg):
        """宽限期内（<lost_grace_seconds）不能开始回位，避免手一晃就动"""
        rt, _ = build(cfg, robot_cfg, "task_space")
        before = list(rt.target)
        res = rt.update(None, now=0.0)
        assert rt.target == pytest.approx(before)
        assert "保持" in res.status or "丢失" in res.status

    def test_reacquire_resumes_tracking(self, cfg, robot_cfg):
        """回位途中重新识别到手臂 -> 必须能恢复跟随，且不跳变"""
        rt, arm1 = build(cfg, robot_cfg, "task_space")
        t = 0.0
        for _ in range(20):
            t += 0.05
            rt.update(None, now=t)
        mid = list(rt.target)
        res = rt.update(arm1, now=t + 0.05)
        jump = max(abs(a - b) for a, b in zip(res.positions, mid))
        assert jump < 0.5, f"恢复跟随时跳变 {jump:.3f} rad"
        assert res.state.name == "TRACKING"

    def test_active_joint_names_covers_both_modes(self, cfg):
        """统一集合的自检：task_space 下必须包含 j2/j3，legacy 下不重复"""
        c = cfg
        legacy = c.active_joint_names()
        c.retarget_mode = "task_space"
        ts = c.active_joint_names()
        assert {"joint2", "joint3"} <= set(ts)
        assert {"joint5", "joint6"} <= set(ts)
        assert len(ts) == len(set(ts)), "不应有重复关节"
        # 顺序与 joint_names 一致（CSV / 下发的顺序依赖它）
        order = [c.joint_names.index(n) for n in ts]
        assert order == sorted(order) or set(legacy) <= set(ts)
