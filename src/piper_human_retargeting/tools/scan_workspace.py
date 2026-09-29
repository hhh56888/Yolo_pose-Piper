#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/scan_workspace.py
=======================
J2/J3 二维 FK workspace 扫描 + Neutral 自动搜索（V1.1 任务书 §4/§5）。

做四件事：
    1. 在给定的 (q2, q3) 盒子里做二维网格扫描，记录 TCP x/y/z、
       到区间上下界的余量、XZ 平面最小奇异值（可操作度）；
    2. 从可达点云里求**最大内接矩形**（x × z）—— 这是"TCP 目标区间"的
       物理上限：目标若超出它，IK 必然够不到；
    3. 用评分函数（工作区中心度 + 关节余量 + 可操作度）搜索 Neutral，
       输出 Top-5；
    4. 打印可直接粘进 retargeting.yaml 的建议配置。

用法：
    python3 tools/scan_workspace.py                       # 默认安全区间
    python3 tools/scan_workspace.py --n2 61 --n3 61 --out data/workspace.json
"""
import argparse
import json
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_HERE)
_SRC = os.path.dirname(_PKG_ROOT)
for _p in (_PKG_ROOT, os.path.join(_SRC, "piper_human_control")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from piper_human_retargeting.kinematics import load_default_chain   # noqa: E402
from piper_human_retargeting.limits import physical_limits, safe_limits  # noqa: E402
from piper_human_retargeting.task_space import (                    # noqa: E402
    neutral_candidates, scan_workspace)


def reachable_mask(ik, xs, zs, tol=None):
    """
    用**真实 IK** 判定每个 (x,z) 是否可达（含关节区间约束）。

    为什么不用点云占据图：两连杆的可达集是平面上的**厚环带**，
    用扫描点做栅格占据会把"点稀疏"误判成"不可达"，
    得出的内接矩形会小得离谱（实测 0.04×0.17 m，明显不合理）。
    直接用 IK 逐个试，既准确又和实际求解器一致。
    """
    tol = ik.pos_tol if tol is None else tol
    mask = np.zeros((len(xs), len(zs)), dtype=bool)
    for i, x in enumerate(xs):
        for j, z in enumerate(zs):
            s = ik.solve(float(x), float(z), record_stats=False,
                         q_neutral=None)
            mask[i, j] = math.hypot(s.err_x, s.err_z) <= tol
    return mask


def inscribed_rect(mask, xs, zs, min_fill=1.0):
    """
    在可达性掩码上求最大面积轴对齐矩形（积分图 + 枚举上下界）。
    """
    xs = np.asarray(xs, dtype=float)
    zs = np.asarray(zs, dtype=float)
    nx, nz = mask.shape
    S = np.zeros((nx + 1, nz + 1), dtype=np.int64)
    S[1:, 1:] = mask.astype(np.int64).cumsum(0).cumsum(1)
    best = None
    for ax in range(nx):
        for bx in range(ax + 1, nx + 1):
            for az in range(nz):
                for bz in range(az + 1, nz + 1):
                    cells = (bx - ax) * (bz - az)
                    got = S[bx, bz] - S[ax, bz] - S[bx, az] + S[ax, az]
                    if got < min_fill * cells:
                        continue
                    area = (xs[bx - 1] - xs[ax]) * (zs[bz - 1] - zs[az])
                    if best is None or area > best[0]:
                        best = (area, xs[ax], xs[bx - 1], zs[az], zs[bz - 1])
    if best is None:
        return None
    return {"x_min": float(best[1]), "x_max": float(best[2]),
            "z_min": float(best[3]), "z_max": float(best[4]),
            "area": float(best[0])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q2-min", type=float, default=None)
    ap.add_argument("--q2-max", type=float, default=None)
    ap.add_argument("--q3-min", type=float, default=None)
    ap.add_argument("--q3-max", type=float, default=None)
    ap.add_argument("--inset", type=float, default=0.15,
                    help="在 safe 限位基础上再向内收多少 rad（默认 0.15）")
    ap.add_argument("--n2", type=int, default=61)
    ap.add_argument("--n3", type=int, default=61)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--out", type=str, default="")
    a = ap.parse_args()

    phys, src = physical_limits()
    safe = safe_limits(phys)
    q2r = (a.q2_min if a.q2_min is not None else safe["joint2"][0] + a.inset,
           a.q2_max if a.q2_max is not None else safe["joint2"][1] - a.inset)
    q3r = (a.q3_min if a.q3_min is not None else safe["joint3"][0] + a.inset,
           a.q3_max if a.q3_max is not None else safe["joint3"][1] - a.inset)
    print("限位来源: %s" % src)
    print("physical joint2=%s joint3=%s" % (phys["joint2"], phys["joint3"]))
    print("safe     joint2=%s joint3=%s" % (safe["joint2"], safe["joint3"]))
    print("扫描区间 joint2=(%.3f, %.3f) joint3=(%.3f, %.3f) 网格 %dx%d"
          % (q2r[0], q2r[1], q3r[0], q3r[1], a.n2, a.n3))
    print("  → 该区间 ⊂ safe ⊂ physical: %s"
          % (safe["joint2"][0] <= q2r[0] and q2r[1] <= safe["joint2"][1]
             and safe["joint3"][0] <= q3r[0] and q3r[1] <= safe["joint3"][1]))

    pts = scan_workspace(q2_range=q2r, q3_range=q3r, n2=a.n2, n3=a.n3)
    xs = [p.x for p in pts]
    zs = [p.z for p in pts]
    sig = [p.sigma_min for p in pts]
    print("\n扫描点 %d  TCP x∈[%+.3f,%+.3f]  z∈[%+.3f,%+.3f]  "
          "sigma_min∈[%.4f,%.4f]"
          % (len(pts), min(xs), max(xs), min(zs), max(zs), min(sig), max(sig)))

    # ---- 求"最大可达矩形"：用真实 IK 在 (x,z) 网格上逐个验证 ----
    from piper_human_retargeting.task_space import TaskSpaceIk
    ik_chk = TaskSpaceIk(model=load_default_chain(),
                         q_min=(q2r[0], q3r[0]), q_max=(q2r[1], q3r[1]),
                         lambda_prev=0.0, lambda_neutral=0.0)
    gx = np.linspace(max(0.0, min(xs)), max(xs), 46)
    gz = np.linspace(max(0.05, min(zs)), max(zs), 46)
    print("\n用 IK 验证 %d 个 (x,z) 格点的可达性（这决定了 TCP 目标区间的上限）"
          % (len(gx) * len(gz)))
    mask = reachable_mask(ik_chk, gx, gz)
    print("  可达格点 %d / %d (%.1f%%)"
          % (int(mask.sum()), mask.size, 100.0 * mask.mean()))
    rect = inscribed_rect(mask, gx, gz)
    if rect:
        print("\n最大可达矩形（x × z，面积 %.4f m²）：" % rect["area"])
        print("  x ∈ [%+.3f, %+.3f]  (%.3f m)" % (rect["x_min"], rect["x_max"],
                                                 rect["x_max"] - rect["x_min"]))
        print("  z ∈ [%+.3f, %+.3f]  (%.3f m)" % (rect["z_min"], rect["z_max"],
                                                 rect["z_max"] - rect["z_min"]))

    # ---- neutral 搜索：只在矩形内的点里选，保证 neutral 处于可控区中央 ----
    pool = pts
    if rect:
        pool = [p for p in pts
                if rect["x_min"] <= p.x <= rect["x_max"]
                and rect["z_min"] <= p.z <= rect["z_max"]]
    x_span = (rect["x_min"], rect["x_max"]) if rect else None
    z_span = (rect["z_min"], rect["z_max"]) if rect else None
    cands = neutral_candidates(pool, top=a.top, x_span=x_span, z_span=z_span)
    print("\nNeutral 候选 Top%d（评分 = 工作区中心度 + 关节余量 + 可操作度）："
          % len(cands))
    for i, c in enumerate(cands, 1):
        print("  %d. %s" % (i, c.table_row()))

    if cands:
        best = cands[0]
        print("\n建议写入 config/retargeting.yaml 的 task_space 段：")
        print("    q2_min: %.2f" % q2r[0])
        print("    q2_max: %.2f" % q2r[1])
        print("    q3_min: %.2f" % q3r[0])
        print("    q3_max: %.2f" % q3r[1])
        nl = [0.0, round(best.q2, 3), round(best.q3, 3), 0.0, 0.0, 0.0]
        print("    neutral_joints: %s" % nl)
        if rect:
            print("    x_min: %.2f      # 由可达域内接矩形给出" %
                  (rect["x_min"] + 0.02))
            print("    x_max: %.2f" % (rect["x_max"] - 0.02))
            print("    z_min: %.2f" % (rect["z_min"] + 0.02))
            print("    z_max: %.2f" % (rect["z_max"] - 0.02))
        print("    # neutral 处 TCP=(%+.3f, %+.3f)，到区间余量 "
              "m2=(%+.3f,%+.3f) m3=(%+.3f,%+.3f)，sigma_min=%.4f"
              % (best.x, best.z,
                 best.margins["joint2"][0], best.margins["joint2"][1],
                 best.margins["joint3"][0], best.margins["joint3"][1],
                 best.sigma_min))

    if a.out:
        data = {"source": src, "q2_range": list(q2r), "q3_range": list(q3r),
                "n_points": len(pts),
                "workspace": [p.as_dict() for p in pts],
                "inscribed_rect": rect,
                "neutral_top": [c.as_dict() for c in cands]}
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        print("\n写出 %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
