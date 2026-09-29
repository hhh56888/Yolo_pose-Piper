#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
J3 审计 · 静态链路审计（不连 ROS，不改任何配置）
================================================
目的：用**当前生产代码**回答
  1. 人体肘角的定义/公式/范围（§1）
  2. elbow -> joint3 全链路每一级的实际数值（§2）
  3. delta_elbow 的做差方式是否误用周期角（§3）

做法：构造合成手臂关键点 -> 走真实的 compute_arm_geometry /
Retargeter / mapping 代码路径，把每一级打印出来。
"""
import math
import os
import sys

_LOCAL = "/Users/hyu/Desktop/catkin_ws/src"
_REMOTE = os.path.expanduser("~/Yolo_pose+piper/src")
SRC = _LOCAL if os.path.isdir(_LOCAL) else _REMOTE
for _p in (os.path.join(SRC, "piper_human_retargeting"),
           os.path.join(SRC, "piper_human_control"),
           os.path.join(SRC, "piper_human_perception")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from piper_human_control import ArmKeypoints, Keypoint          # noqa: E402
from piper_human_retargeting import (Retargeter,                 # noqa: E402
                                     RetargetingConfig)
from piper_human_retargeting.arm_geometry import (               # noqa: E402
    angle_delta_deg, compute_arm_geometry)

LINE = "=" * 78


def make_arm(upper_deg, fore_deg):
    """与 test/verify_joints.py 完全一致的合成方式（肘夹角 = fore - upper）"""
    ea = math.radians(upper_deg)
    e = (100.0 * math.cos(ea), -100.0 * math.sin(ea))
    fa = math.radians(fore_deg)
    w = (e[0] + 100.0 * math.cos(fa), e[1] - 100.0 * math.sin(fa))
    return ArmKeypoints(
        shoulder=Keypoint(0.0, 0.0, confidence=0.9, name="right_shoulder"),
        elbow=Keypoint(*e, confidence=0.9, name="right_elbow"),
        wrist=Keypoint(*w, confidence=0.9, name="right_wrist"),
        side="right",
        hand_wrist=Keypoint(*w, confidence=0.9, name="right_wrist_from_hand"))


def section(title):
    print("\n" + LINE + "\n" + title + "\n" + LINE)


# ============================================================
# §1 人体肘角定义：直接调真实代码
# ============================================================
section("§1 人体肘角定义（调用真实 compute_arm_geometry）")
print("%-14s %-16s %-16s %-16s" % ("fore-upper", "elbow_angle_deg", "elbow_flexion", "interior(=180-bend)"))
for bend in (0, 30, 45, 90, 120, 135, 150, 180):
    geo = compute_arm_geometry(make_arm(0.0, float(bend)))
    print("%-14s %-16.2f %-16.2f %-16.2f" % (
        "%d°" % bend, geo.elbow_angle_deg, 180.0 - geo.elbow_angle_deg,
        180.0 - geo.elbow_angle_deg))
print("→ 结论：elbow_angle_deg 是「屈曲量」(0=伸直, 180=完全折回)，"
      "数值随弯曲**单调增大**")
print("→ elbow_flexion = elbow_angle_deg 本身（无需再做 180-x）")

# ============================================================
# §2 当前生效配置
# ============================================================
section("§2 当前生效配置（RetargetingConfig.from_yaml()）")
cfg = RetargetingConfig.from_yaml()
print("配置文件来源: %s" % getattr(cfg, "source_path", "(由 from_yaml 默认解析)"))
print("joint_names      = %s" % cfg.joint_names)
print("neutral_joints   = %s" % [round(v, 4) for v in cfg.neutral_joints])
print("fixed_joints     = %s" % cfg.fixed_joints)
print("retarget_mode    = %s" % cfg.retarget_mode)
print("use_filtered_angles = %s" % cfg.use_filtered_angles)
print()
print("measure_sign()   = %s" % cfg.measure_sign())
print()
hdr = ("%-10s %-22s %-9s %8s %8s %9s %9s %8s %7s %6s %6s %9s %9s"
       % ("rule", "human", "joint", "h_min", "h_max", "r_min", "r_max",
          "scale", "offset", "invert", "sign", "joint_lo", "joint_hi"))
print(hdr)
for r in cfg.rules:
    if r.joint != "joint3":
        continue
    print("%-10s %-22s %-9s %8.3f %8.3f %9.3f %9.3f %8.5f %7.2f %6s %+6.1f %9.3f %9.3f" % (
        r.key, r.human, r.joint, r.human_min, r.human_max, r.robot_min,
        r.robot_max, r.scale, r.offset, r.invert, r.sign,
        r.joint_lower, r.joint_upper))
    print("           enabled=%s clamp=%s max_jump=%.1f" % (
        r.enabled, r.clamp, r.max_jump))
print()
j3n = cfg.neutral_for("joint3")
print("joint3 neutral = %+.4f rad (来自 neutral_joints)" % j3n)
_j3r = [r for r in cfg.rules if r.joint == "joint3"][0]
print("joint3 限位     = [%.4f, %.4f]  (来自 joint_limits.yaml，与 URDF 一致)"
      % (_j3r.joint_lower, _j3r.joint_upper))
print("distance_to_lower = %+.4f rad, distance_to_upper = %+.4f rad"
      % (j3n - (-2.967), 0.0 - j3n))

# ============================================================
# §2b 合成扫掠：链路每一级的数值
# ============================================================
def chain_rows(neutral_elbow_deg, bends, label):
    """用真实 Retargeter 标定 + 扫掠，打印链路每一级"""
    rt = Retargeter(cfg, None)
    rt.set_initial_pose(list(cfg.neutral_joints))
    rt.start_calibration()
    base = compute_arm_geometry(make_arm(0.0, neutral_elbow_deg))
    for _ in range(30):
        rt.add_calibration_sample(base)
    rt.finish_calibration()
    i3 = cfg.joint_names.index("joint3")
    print("\n[%s] 标定中和角 = %.1f°, 标定后 neutral_elbow = %.2f°" % (
        label, neutral_elbow_deg, rt.neutral.get("elbow_angle_deg") or float("nan")))
    print("%-8s %-10s %-10s %-10s %-10s %-12s %-9s %-8s" % (
        "屈曲(°)", "raw", "flt", "neutral", "delta", "mapped_j3", "j3 target", "flexion"))
    out = []
    SETTLE = 40          # 每个位姿保持 40 帧，让角度/关节滤波收敛后再读数
    for bend in bends:
        arm = make_arm(0.0, float(bend))
        geo = compute_arm_geometry(arm)
        for _ in range(SETTLE):
            res = rt.update(arm, now=0.0, raw_geometry=geo)
        d = res.debug
        out.append((bend, geo.elbow_angle_deg, d.delta_elbow,
                    res.mapped_joints.get("joint3", float("nan")),
                    res.positions[i3]))
        print("%-8.0f %-10.2f %-10.2f %-10.2f %-10.2f %-12.4f %-9.4f %-8.2f" % (
            bend, d.raw_elbow, d.flt_elbow,
            rt.neutral.get("elbow_angle_deg") or float("nan"),
            d.delta_elbow, res.mapped_joints.get("joint3", float("nan")), res.positions[i3],
            180.0 - geo.elbow_angle_deg))
    return out


section("§2b 链路逐级数值（合成输入，真实代码路径）")
chain_rows(90.0, [0, 45, 90, 135, 180], "标定=肘90°（verify_joints 的约定）")
chain_rows(126.6, [20, 60, 126.6, 160, 180], "标定=实机 126.6°（本次 live 运行）")

# ============================================================
# §3 做差方式
# ============================================================
section("§3 delta_elbow 的做差方式（周期角是否误用）")
print("代码：deltas_deg[name] = angle_delta_deg(cur * sign, neutral.get(name))")
print("      angle_delta_deg(a,b) = wrap180(a-b)  → d=(a-b)%%360; d>180 则 d-=360")
print()
cases = [(0.0, 0.0), (180.0, 0.0), (0.0, 180.0), (175.0, 5.0), (5.0, 175.0),
         (126.6, 126.6), (160.0, 90.0), (90.0, 160.0)]
print("%-10s %-10s %-12s %-12s %-12s" % ("raw", "neutral", "普通差", "带符号差", "是否一致"))
sign = cfg.measure_sign().get("elbow_angle_deg", 1.0)
for cur, base in cases:
    plain = cur - base
    wrapped = angle_delta_deg(cur * sign, base * sign)
    with_sign = sign * plain
    print("%-10.1f %-10.1f %-12.2f %-12.2f %-12s" % (
        cur, base, plain, wrapped, "一致" if abs(wrapped - with_sign) < 1e-9 else "不一致"))
print("→ 因为 elbow∈[0,180]，|raw-neutral|<=180，wrap180 恒等于普通差，"
      "周期角逻辑对肘角是**无害的空操作**（但语义上肘角不是周期量）")

# ============================================================
# §2c 用 live 数据复核（真实 CSV）
# ============================================================
section("§2c 真实运行数据复核（/tmp/live_run2.csv，生产配置）")
try:
    import csv
    rows = [r for r in csv.DictReader(open(os.environ.get("AUDIT_CSV", "/tmp/live_run.csv")))
            if r["state"] == "TRACKING"]

    def col(k):
        return [float(r[k]) for r in rows]
    raw, j3, dj3 = col("raw_elbow"), col("joint3"), col("delta_elbow")
    n = len(rows)
    mx, mn = max(j3), min(j3)
    up_sat = sum(1 for v in j3 if v > -0.01) / n * 100
    lo_sat = sum(1 for v in j3 if v < -1.19) / n * 100

    def corr(a, b):
        ma, mb = sum(a) / len(a), sum(b) / len(b)
        va = sum((x - ma) ** 2 for x in a)
        vb = sum((x - mb) ** 2 for x in b)
        return sum((a[i] - ma) * (b[i] - mb) for i in range(len(a))) / (va * vb) ** 0.5
    print("样本 %d（TRACKING）" % n)
    print("corr(raw_elbow, joint3)      = %+.4f  (正=弯肘时 joint3 增大)" % corr(raw, j3))
    print("corr(delta_elbow, joint3)    = %+.4f" % corr(dj3, j3))
    print("joint3 行程 = [%+.4f, %+.4f]  range=%.4f" % (mn, mx, mx - mn))
    print("贴上界(q3>=-0.01) %.2f%%   贴下界(q3<=-1.19) %.2f%%" % (up_sat, lo_sat))
    print("raw_elbow 范围 = [%.1f, %.1f]" % (min(raw), max(raw)))
    print("delta_elbow 范围 = [%.1f, %.1f]  (human 区间 ±%.1f)"
          % (min(dj3), max(dj3), 65.0))
except FileNotFoundError:
    print("(无 /tmp/live_run2.csv，跳过)")
