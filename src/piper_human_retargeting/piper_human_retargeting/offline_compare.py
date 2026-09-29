# -*- coding: utf-8 -*-
"""
piper_human_retargeting.offline_compare
=======================================
离线对比 **legacy** 与 **task_space** 两种重映射。

任务书第十六节要求：用**同一段人体输入**比较
    joint2/joint3 饱和率
    TCP X / Z 行程
    人体前伸时的 ΔX/ΔZ
    人体抬手时的 ΔX/ΔZ
    J2/J3 平滑性、IK 跳变次数、tracking 连续性

为什么能在离线做：
    两种模式都只依赖「人体关键点 → 关节角」这条纯计算链路，
    不需要真的驱动机械臂。末端 X/Z 由 FK 表反查得到。

用法：
    python3 -m piper_human_retargeting.offline_compare \\
        --video <视频> [--frames N] [--side right] [--out /tmp/cmp.json]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st
import sys
from typing import Dict, List, Optional, Tuple


# ============================================================
# 指标计算
# ============================================================
def _saturation_rate(values: List[float], lo: float, hi: float,
                     eps: float = 1e-3) -> float:
    """贴到区间任一端（含裁剪后恰好等于端值）的比例"""
    if not values:
        return 0.0
    n = sum(1 for v in values if v <= lo + eps or v >= hi - eps)
    return 100.0 * n / len(values)


def _travel(values: List[float]) -> float:
    return (max(values) - min(values)) if values else 0.0


def _jerk(values: List[float]) -> float:
    """相邻帧变化量的中位数（抖动指标）"""
    d = sorted(abs(a - b) for a, b in zip(values[1:], values[:-1]))
    return d[len(d) // 2] if d else 0.0


def _corr(a: List[float], b: List[float]) -> float:
    n = min(len(a), len(b))
    if n < 3:
        return float("nan")
    a, b = a[:n], b[:n]
    ma, mb = st.mean(a), st.mean(b)
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b)) / n
    sa = (sum((x - ma) ** 2 for x in a) / n) ** 0.5
    sb = (sum((y - mb) ** 2 for y in b) / n) ** 0.5
    return cov / (sa * sb) if sa * sb > 1e-12 else float("nan")


def summarize(name: str, rows: List[dict],
              joint_limits: Dict[str, Tuple[float, float]]) -> dict:
    """
    把一串逐帧记录汇总成指标。

    rows 每项需含：reach_raw, elevation_raw, d_reach, d_elevation,
                   x, z, q2, q3, state, ik_jump(可选)
    """
    ok = [r for r in rows if r.get("state") == "TRACKING"]
    if not ok:
        return {"mode": name, "n": 0, "note": "无 TRACKING 帧"}

    xs = [r["x"] for r in ok]
    zs = [r["z"] for r in ok]
    q2 = [r["q2"] for r in ok]
    q3 = [r["q3"] for r in ok]
    dr = [r["d_reach"] for r in ok]
    de = [r["d_elevation"] for r in ok]

    j2lo, j2hi = joint_limits["joint2"]
    j3lo, j3hi = joint_limits["joint3"]

    xt, zt = _travel(xs), _travel(zs)

    # 前伸 / 抬手 的分离度：
    #   ΔX 应主要跟随 d_reach，ΔZ 应主要跟随 d_elevation
    c_x_reach = _corr(dr, xs)
    c_x_elev = _corr(de, xs)
    c_z_reach = _corr(dr, zs)
    c_z_elev = _corr(de, zs)

    # IK 稳定性用**比例**而不是「超过某阈值的次数」：
    # FK 表是 0.2x0.25 rad 的离散网格，解本来就在网格点之间步进，
    # 用固定阈值（如 0.25 rad）统计会把正常步进全判成跳变
    #（实测误报 507 次）。改用「相邻帧关节位移 > 0.3 rad 的帧占比」。
    jumps = [r.get("ik_jump", 0.0) for r in ok if "ik_jump" in r]
    n_jump = sum(1 for d in jumps if d > 0.3) if jumps else 0
    jump_pct = 100.0 * n_jump / max(len(jumps), 1)

    return {
        "mode": name,
        "n": len(ok),
        "n_total": len(rows),
        "tracking_pct": 100.0 * len(ok) / max(len(rows), 1),
        "x_travel": xt,
        "z_travel": zt,
        "xz_ratio": (xt / zt) if zt > 1e-6 else float("inf"),
        "j2_travel": _travel(q2),
        "j3_travel": _travel(q3),
        "j2_range": [min(q2), max(q2)],
        "j3_range": [min(q3), max(q3)],
        "j2_sat_pct": _saturation_rate(q2, j2lo, j2hi),
        "j3_sat_pct": _saturation_rate(q3, j3lo, j3hi),
        "j2_jerk": _jerk(q2),
        "j3_jerk": _jerk(q3),
        "corr_dReach_X": c_x_reach,
        "corr_dElev_X": c_x_elev,
        "corr_dReach_Z": c_z_reach,
        "corr_dElev_Z": c_z_elev,
        "ik_jumps": n_jump,
        "ik_jump_pct": jump_pct,
        "ik_jump_median": (st.median([d for d in jumps if d > 0])
                           if any(d > 0 for d in jumps) else 0.0),
        "reach_travel": _travel(dr),
        "elev_travel": _travel(de),
    }


def print_table(results: List[dict]) -> None:
    """控制台对比表"""
    def g(r, k, fmt="{:.3f}"):
        v = r.get(k)
        if v is None:
            return "  --  "
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return "  --  "
        return fmt.format(v)

    print()
    print("=" * 78)
    print("Legacy vs Task-Space 对比")
    print("=" * 78)
    head = f"{'指标':<26}" + "".join(f"{r['mode']:>24}" for r in results)
    print(head)
    print("-" * 78)
    rows = [
        ("TRACKING 帧数", lambda r: f"{r.get('n',0)}", None),
        ("X 行程 (m)", lambda r: g(r, "x_travel"), None),
        ("Z 行程 (m)", lambda r: g(r, "z_travel"), None),
        ("X/Z 比", lambda r: g(r, "xz_ratio", "{:.2f}"), None),
        ("joint2 行程 (rad)", lambda r: g(r, "j2_travel"), None),
        ("joint3 行程 (rad)", lambda r: g(r, "j3_travel"), None),
        ("joint2 饱和率 (%)", lambda r: g(r, "j2_sat_pct", "{:.1f}"), None),
        ("joint3 饱和率 (%)", lambda r: g(r, "j3_sat_pct", "{:.1f}"), None),
        ("joint2 抖动 (rad)", lambda r: g(r, "j2_jerk", "{:.4f}"), None),
        ("joint3 抖动 (rad)", lambda r: g(r, "j3_jerk", "{:.4f}"), None),
        ("IK 大跳变占比 (%)", lambda r: g(r, "ik_jump_pct", "{:.1f}"), None),
        ("IK 单帧位移中位", lambda r: g(r, "ik_jump_median", "{:.3f}"), None),
        ("corr(dReach, X)", lambda r: g(r, "corr_dReach_X", "{:+.3f}"), None),
        ("corr(dElev,  X)", lambda r: g(r, "corr_dElev_X", "{:+.3f}"), None),
        ("corr(dReach, Z)", lambda r: g(r, "corr_dReach_Z", "{:+.3f}"), None),
        ("corr(dElev,  Z)", lambda r: g(r, "corr_dElev_Z", "{:+.3f}"), None),
    ]
    for label, fn, _ in rows:
        print(f"{label:<26}" + "".join(f"{fn(r):>24}" for r in results))
    print("=" * 78)
    print()
    print("判据（任务书十六节）：")
    print("  · 前伸主要影响 X  ->  |corr(dReach,X)| 应明显大于 |corr(dReach,Z)|")
    print("  · 抬手主要影响 Z  ->  |corr(dElev,Z)|  应明显大于 |corr(dElev,X)|")
    print("  · X/Z 比 应比 legacy 更接近 1（legacy 实测约 0.55）")
    print("  · 饱和率与 IK 跳变应尽量低")


# ============================================================
# 主流程
# ============================================================
def run_compare(video: str, n_frames: int = 0, side: str = "right",
                device: Optional[str] = None, imgsz: int = 960,
                pose_model: Optional[str] = None) -> dict:
    """对同一段视频跑两种模式并返回对比结果"""
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    from piper_human_control import ControlConfig
    from piper_human_perception import (VideoFileSource, YOLOPoseProvider,
                                        compute_hand_pose, make_hand_provider,
                                        merge_hand_wrist)
    from piper_human_retargeting import RetargetingConfig, Retargeter
    from piper_human_retargeting.task_space import compute_human_task_state

    base = RetargetingConfig.from_yaml()
    robot_cfg = ControlConfig.from_yaml()
    limits = {"joint2": (robot_cfg.lower_limits["joint2"],
                         robot_cfg.upper_limits["joint2"]),
              "joint3": (robot_cfg.lower_limits["joint3"],
                         robot_cfg.upper_limits["joint3"])}

    # ---- 一次性抽帧（两种模式共用同一段输入，保证可比）----
    src = VideoFileSource(video, loop=False)
    if not src.open():
        raise RuntimeError(f"视频打不开: {video}")
    pm = pose_model or os.path.expanduser(
        "~/Yolo_pose+piper/models/yolo11n-pose.pt")
    if not os.path.isfile(pm):
        pm = "yolo11n-pose.pt"
    prov = YOLOPoseProvider(model_path=pm, conf_threshold=0.4,
                            person_conf=0.5, target_side=side,
                            device=device, imgsz=imgsz)
    prov.load()
    hand = make_hand_provider()

    frames = []
    i = 0
    while True:
        fr = src.read()
        if fr is None:
            break
        i += 1
        if n_frames and i > n_frames:
            break
        d = prov.detect(fr)
        if d.arm is None:
            continue
        hd = hand.detect(fr, d.arm.wrist, side)
        arm = merge_hand_wrist(d.arm, hd)
        hp = compute_hand_pose(arm, hd)
        g = compute_human_task_state(arm.shoulder, arm.elbow,
                                     arm.effective_wrist)
        frames.append((i, arm, hp, g))
    src.release()
    prov.close()
    hand.close()
    print(f"抽帧完成：{len(frames)} 有效帧（共 {i} 帧）")
    if len(frames) < 30:
        raise RuntimeError(f"有效帧太少({len(frames)})，无法对比")

    # ---- 两种模式各跑一遍 ----
    all_res = {}
    for mode in ("legacy", "task_space"):
        # ⚠️ 必须「先设模式、再构造 Retargeter」。
        #   used_measures() 里是否含 reach/elevation 是**按 retarget_mode 推导**的，
        #   若先构造再改 cfg.retarget_mode，滤波/标定/缺失检查都不知道这两个量的存在，
        #   task_space 会静默失效（两种模式输出完全相同）。
        #   这里通过环境变量让 from_yaml 一开始就按目标模式解析。
        cfg = _config_for_mode(mode)
        rt = Retargeter(cfg, robot_cfg)
        assert (rt.task_solver is not None) == (mode == "task_space"), \
            f"{mode} 模式求解器装配不正确"

        # 标定：用前 N 帧建立人体基准（两种模式用**同一个**标定窗口）
        cal_n = min(120, max(20, len(frames) // 4))
        rt.start_calibration()
        for _i, arm, hp, ts in frames[:cal_n]:
            rt.add_calibration_sample_for(arm, _geo(arm), hand_pose=hp)
        rt.finish_calibration()
        rt.set_initial_pose(cfg.neutral_joints)

        rows = []
        for idx, (fi, arm, hp, ts) in enumerate(frames):
            res = rt.update(arm, now=fi * 0.033, frame=fi, hand_pose=hp)
            i2 = cfg.joint_names.index("joint2")
            i3 = cfg.joint_names.index("joint3")
            q2, q3 = res.positions[i2], res.positions[i3]
            # 末端位置：由 FK 表反查（两种模式用同一张表，公平）
            if rt.task_solver is not None:
                x, z = rt.task_solver.fk(q2, q3)
            else:
                x, z = _fk_via_tmp(cfg, q2, q3)
            d = res.debug
            rows.append({
                "frame": fi,
                "state": str(res.state).split(".")[-1],
                "q2": q2, "q3": q3, "x": x, "z": z,
                "reach_raw": ts.reach if ts.valid else 0.0,
                "elevation_raw": ts.elevation if ts.valid else 0.0,
                "d_reach": getattr(d, "d_reach", 0.0),
                "d_elevation": getattr(d, "d_elevation", 0.0),
                "ik_jump": getattr(d, "ik_jump", 0.0),
            })
        all_res[mode] = summarize(mode, rows, limits)
        all_res[mode + "_rows"] = rows

    return all_res


def _config_for_mode(mode: str):
    """
    按目标模式构造配置。

    做法：读 YAML 文本 -> 覆盖顶层 retarget_mode -> 从字典加载。
    这样 used_measures()/校验都按新模式生效，
    而不是「先按默认模式加载、事后改属性」。
    """
    import yaml
    from piper_human_retargeting import RetargetingConfig
    path = RetargetingConfig.default_yaml_path()
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    raw["retarget_mode"] = mode
    cfg = RetargetingConfig._from_dict(raw)
    cfg.source_path = path
    cfg.validate()
    return cfg


def _geo(arm):
    from piper_human_retargeting import compute_arm_geometry
    return compute_arm_geometry(arm, pivot="shoulder")


_TMP_SOLVER = {}


def _fk_via_tmp(cfg, q2, q3):
    """
    legacy 模式下也要能算末端位置 —— 复用一个临时求解器（只读 FK）。

    路径解析复用 Retargeter._resolve_table_path：
    安装后配置在 share/<pkg>/config/、FK 表在 share/<pkg>/data/，
    两者不同层。先前自己拼路径时重复了一层 data/，直接报「表不存在」。
    """
    if "s" not in _TMP_SOLVER:
        from piper_human_retargeting.retargeter import Retargeter
        from piper_human_retargeting.task_space import JointSolver
        p = Retargeter._resolve_table_path(cfg.task_space.table_path,
                                           cfg.source_path)
        _TMP_SOLVER["s"] = JointSolver(p)
    return _TMP_SOLVER["s"].fk(q2, q3)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="legacy vs task_space 离线对比")
    ap.add_argument("--video", required=True)
    ap.add_argument("--frames", type=int, default=0, help="只用前 N 帧")
    ap.add_argument("--side", default="right", choices=("left", "right"))
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--device", default=None)
    ap.add_argument("--pose-model", default=None)
    ap.add_argument("--out", default=None, help="结果写入 JSON")
    a = ap.parse_args(argv)

    res = run_compare(a.video, a.frames, a.side, a.device, a.imgsz,
                      a.pose_model)
    both = [res["legacy"], res["task_space"]]
    print_table(both)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({"legacy": res["legacy"],
                       "task_space": res["task_space"]}, f,
                      ensure_ascii=False, indent=2)
        print(f"结果已写入 {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
