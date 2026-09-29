# -*- coding: utf-8 -*-
"""
piper_human_retargeting.synth_compare
=====================================
用**合成人体动作**对比 legacy 与 task_space 两种重映射。

为什么要合成动作（而不是只用真实视频）
======================================
任务书第十六节要判断：
    「人体前伸主要影响 X，抬手主要影响 Z」
但真实视频里「前伸」和「抬手」是**混在一起**的，无法干净归因。
合成动作可以做到单一变量：
    · 纯前伸：肩膀不动，手臂沿视线方向伸出（reach 变、elevation 近似不变）
    · 纯抬手：手臂整体上摆（elevation 变、reach 基本不变）

这样测出的 ΔX / ΔZ 才能确定性归因，是「解耦是否成立」的直接证据。

两类动作都用**臂长归一化**的方式生成，因此不依赖像素尺度。

用法：
    python3 -m piper_human_retargeting.synth_compare [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Tuple


def _arm_points(arm_len: float, elbow_deg: float, shoulder_deg: float,
                scale: float = 1.0):
    """
    按「上臂方向角 + 肘夹角」造出 S/E/W 三点（图像坐标，y 向下）。

    Args:
        elbow_deg:    肘夹角。0=伸直，180=完全折回
        shoulder_deg: 上臂方向角（图像角，y 翻转语义：+90=向上）
        scale:        像素尺度
    """
    L1 = 0.5 * arm_len * scale
    L2 = 0.5 * arm_len * scale
    a1 = math.radians(shoulder_deg)
    # 图像 y 向下，故取负
    ex = L1 * math.cos(a1)
    ey = -L1 * math.sin(a1)
    # 前臂相对上臂折 elbow_deg
    a2 = a1 - math.radians(elbow_deg)
    wx = ex + L2 * math.cos(a2)
    wy = ey - L2 * math.sin(a2)
    return (0.0, 0.0), (ex, ey), (wx, wy)


def arm_from_hand(hand_xy, arm_len: float = 200.0):
    """
    由**腕相对肩的位置**反解出 S/E/W 三点（两连杆，取肘在下方的那组解）。

    为什么改用这个构造，而不是「上臂角 + 肘夹角」：
        最初用角度造动作时，测出来「纯前伸」里 elevation 也变了 0.703 ——
        因为伸展上臂的同时手腕被抬高了。那样的动作**本身就不纯**，
        无法用来判断解耦是否成立。

        直接指定腕的位置才能造出真正的单一变量动作：
            水平前伸     : 腕沿 +x 外移、高度不变 -> 只有 reach 变
            半径不变上摆 : 腕沿圆弧上摆           -> 只有 elevation 变
        实测（arm_len=200）：
            水平前伸 120->200 : Δreach=+0.399  Δelev= 0.000
            半径上摆 -40->80  : Δreach= 0.000  Δelev=+1.626
        即两个量在这两类动作下**完全独立**，是合格的解耦判据。
    """
    L = arm_len / 2.0
    hx, hy = float(hand_xy[0]), float(hand_xy[1])
    d = math.hypot(hx, hy)
    d = min(d, 2 * L * 0.999)
    if d < 1e-6:
        return (0.0, 0.0), (L, 0.0), (0.0, 0.0)
    a = math.atan2(hy, hx)
    cosb = max(-1.0, min(1.0, d / (2 * L)))
    ang = a - math.acos(cosb)
    ex, ey = L * math.cos(ang), L * math.sin(ang)
    return (0.0, 0.0), (ex, ey), (hx, hy)


def run_synth(pose_model_path: str = "") -> Dict[str, dict]:
    """跑合成对比，返回两种模式在两类动作上的指标"""
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    from piper_human_control import ControlConfig, Keypoint
    from piper_human_control import ArmKeypoints
    from piper_human_retargeting import RetargetingConfig, Retargeter
    from piper_human_retargeting.task_space import (JointSolver,
                                                    compute_human_task_state)

    base = RetargetingConfig.from_yaml()
    robot_cfg = ControlConfig.from_yaml()
    limits = {"joint2": (robot_cfg.lower_limits["joint2"],
                         robot_cfg.upper_limits["joint2"]),
              "joint3": (robot_cfg.lower_limits["joint3"],
                         robot_cfg.upper_limits["joint3"])}

    # FK 表（两种模式共用，保证末端位置可比）。
    # 复用 Retargeter 的路径解析：安装后配置在 share/<pkg>/config/、
    # 表在 share/<pkg>/data/，两者不同层，写死任一层都会在另一布局下失效。
    from piper_human_retargeting.retargeter import Retargeter
    tp = Retargeter._resolve_table_path(base.task_space.table_path,
                                        base.source_path)
    fk = JointSolver(tp)

    def mkarm(s, e, w):
        return ArmKeypoints(
            shoulder=Keypoint(*s, confidence=0.9, name="s"),
            elbow=Keypoint(*e, confidence=0.9, name="e"),
            wrist=Keypoint(*w, confidence=0.9, name="w"),
            side="right",
            hand_wrist=Keypoint(*w, confidence=0.9, name="w2"))

    # ---------------- 动作定义 ----------------
    A = 200.0          # 臂长（像素）

    def pure_reach(t: float):
        """
        纯前伸：腕沿 +x 水平外移、高度不变。
        实测该项只有 reach 变（Δelev = 0）。
        """
        hx = 110.0 + 90.0 * t
        s, e, w = arm_from_hand((hx, 0.0), A)
        return mkarm(s, e, w)

    def pure_raise(t: float):
        """
        纯抬手：腕沿**以肩为圆心、半径不变**的圆弧上摆。
        实测该项只有 elevation 变（Δreach = 0）。
        """
        deg = -45.0 + 130.0 * t
        r = math.radians(deg)
        hx = A * math.cos(r)
        hy = -A * math.sin(r)          # 图像 y 向下
        s, e, w = arm_from_hand((hx, hy), A)
        return mkarm(s, e, w)

    ACTIONS = {"pure_reach": pure_reach, "pure_raise": pure_raise}
    N = 40

    out = {}
    for mode in ("legacy", "task_space"):
        cfg = RetargetingConfig.from_yaml()
        cfg.retarget_mode = mode
        rt = Retargeter(cfg, robot_cfg)

        for act_name, fn in ACTIONS.items():
            # 每个动作前重新标定（以 t=0 的那一帧为基准）
            rt = Retargeter(cfg, robot_cfg)
            rt.start_calibration()
            for k in range(20):
                arm = fn(0.0)
                rt.add_calibration_sample_for(
                    arm, _geo(arm), hand_pose=_HAND)
            rt.finish_calibration()
            rt.set_initial_pose(cfg.neutral_joints)

            rows = []
            for k in range(N):
                t = k / (N - 1)
                arm = fn(t)
                res = rt.update(arm, now=k * 0.033, frame=k,
                                hand_pose=_HAND)
                i2 = cfg.joint_names.index("joint2")
                i3 = cfg.joint_names.index("joint3")
                q2, q3 = res.positions[i2], res.positions[i3]
                x, z = fk.fk(q2, q3)
                ts = compute_human_task_state(arm.shoulder, arm.elbow,
                                              arm.effective_wrist)
                rows.append({"q2": q2, "q3": q3, "x": x, "z": z,
                             "reach": ts.reach, "elev": ts.elevation})
            out[f"{mode}|{act_name}"] = rows

    return out


def _geo(arm):
    from piper_human_retargeting import compute_arm_geometry
    return compute_arm_geometry(arm, pivot="shoulder")


class _FixedHandPose:
    """
    合成测试用的固定手部姿态。

    为什么必须给：`used_measures()` 里包含腕部量（wrist_pitch / thumb_offset），
    而 `add_calibration_sample_for` 要求**所有**在用量本帧都有值，
    否则整条样本不采纳。合成测试只造手臂关键点、没有手部数据，
    于是标定样本一个都收不到 -> reach/elevation 基准为空 -> task_space 静默不生效
    （表现为两种模式输出完全相同）。

    这类「某个量缺失把整条链路带停」的问题在本项目里反复出现，
    所以这里显式补上一个合法的手部姿态。
    """

    def __init__(self, wrist_pitch_deg=0.0, thumb_offset=0.0):
        self.wrist_pitch_deg = wrist_pitch_deg
        self.thumb_offset = thumb_offset
        self.palm_roll_deg = 0.0
        self.pinch_distance = 1.0
        self.finger_tips_converged = 0.0
        self.valid = True

    def as_dict(self):
        return {"wrist_pitch_deg": self.wrist_pitch_deg,
                "thumb_offset": self.thumb_offset}


_HAND = _FixedHandPose()


def summarize_synth(res: Dict[str, list]) -> None:
    print()
    print("=" * 86)
    print("合成动作对比：解耦能力（Legacy vs Task-Space）")
    print("=" * 86)
    hdr = (f"{'动作':<12}{'模式':<12}{'reach行程':>10}{'elev行程':>10}"
           f"{'ΔX(m)':>9}{'ΔZ(m)':>9}{'|ΔX|/|ΔZ|':>11}{'q2范围':>16}{'q3范围':>16}")
    print(hdr)
    print("-" * 86)
    for act in ("pure_reach", "pure_raise"):
        for mode in ("legacy", "task_space"):
            rows = res.get(f"{mode}|{act}")
            if not rows:
                continue
            xs = [r["x"] for r in rows]
            zs = [r["z"] for r in rows]
            rs = [r["reach"] for r in rows]
            es = [r["elev"] for r in rows]
            q2 = [r["q2"] for r in rows]
            q3 = [r["q3"] for r in rows]
            dx = max(xs) - min(xs)
            dz = max(zs) - min(zs)
            ratio = dx / dz if dz > 1e-6 else float("inf")
            print(f"{act:<12}{mode:<12}{max(rs)-min(rs):>10.3f}"
                  f"{max(es)-min(es):>10.3f}{dx:>9.3f}{dz:>9.3f}{ratio:>11.2f}"
                  f"{f'{min(q2):+.2f}~{max(q2):+.2f}':>16}"
                  f"{f'{min(q3):+.2f}~{max(q3):+.2f}':>16}")
        print("-" * 86)
    print()
    print("判据：")
    print("  · pure_reach 行：|ΔX| 应明显大于 |ΔZ|（前伸主要走 X）")
    print("  · pure_raise 行：|ΔZ| 应明显大于 |ΔX|（抬手主要走 Z）")
    print("=" * 86)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    res = run_synth()
    summarize_synth(res)
    if a.json:
        slim = {k: v for k, v in res.items()}
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(slim, f, ensure_ascii=False)
        print(f"明细已写入 {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
