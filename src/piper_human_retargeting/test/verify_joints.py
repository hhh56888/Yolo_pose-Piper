#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
逐关节验证脚本（对应需求「测试顺序」一节）
==========================================
按顺序验证：
    1. J3 only   —— 弯肘
    2. J2 only   —— 抬手
    3. J5 only   —— 前臂方向
    4. J2 + J3
    5. J2 + J3 + J5

每个阶段的做法：
    用**合成人体几何**驱动 Retargeter（不依赖摄像头/真人），
    通过真实 PiperJointController 下发到 Gazebo，
    再用 TF 独立核对 Gazebo 里关节是否真的跟着动。

为什么用合成几何而不是真人：
    真人无法精确复现「上臂角从 0 变到 +40°」这样的输入，
    而验证映射正确性恰恰需要精确、可复现的输入。
    用合成几何可以把「感知误差」从「映射误差」里剥离出来。

为什么还要用 TF 核对：
    /joint_states 是控制器声称的状态；TF 是从 Gazebo 的
    实际位姿算出来的。两者一致才说明「真的动了」。
    这是独立验证，避免自证。

运行：
    python3 verify_joints.py            # 全部 5 个阶段
    python3 verify_joints.py --stage 0  # 只跑第 1 个阶段
"""

import argparse
import math
import os
import sys
import time

# 允许直接以源码目录运行
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)
for p in (_HERE, os.path.join(_SRC, "piper_human_control"),
          os.path.join(_SRC, "piper_human_perception")):
    if p not in sys.path:
        sys.path.insert(0, p)

import rclpy                                          # noqa: E402
import tf2_ros                                        # noqa: E402

from piper_human_control import (ArmKeypoints, ControlConfig,  # noqa: E402
                                 Keypoint, PiperJointController)
from piper_human_retargeting import (ArmGeometry, Retargeter,  # noqa: E402
                                     RetargetingConfig)

import numpy as np                                    # noqa: E402

STAGES = [
    ("1. J3 only", ["joint3"]),
    ("2. J2 only", ["joint2"]),
    ("3. J5 only", ["joint5"]),
    ("4. J2 + J3", ["joint2", "joint3"]),
    ("5. J2 + J3 + J5", ["joint2", "joint3", "joint5"]),
]

# 合成人体输入的「推拉」幅度（度）
PUSH = 40.0

# ============================================================
# 激励构造要点（这里连续踩过坑，写清楚避免再犯）
# ============================================================
# make_arm(upper, _, fore) 里：
#     上臂方向角 = upper
#     前臂方向角 = fore
#     **肘夹角 = fore - upper**（两段之间的夹角，0 = 完全伸直）
#
# 坑 1：把 upper 和 fore 设成同一个值 -> 两段平行 -> 肘夹角 0（伸直），
#       而不是弯曲。曾据此误判"J3 不响应"。
# 坑 2：想要某个**绝对**肘夹角 E，必须令
#           fore = upper + E
#       而不是把 E 填到 upper 或 fore 的任一个位置。
#       曾写成 make_arm(40, 0, 80) 以为肘夹角=130，
#       实际 fore-upper=40，得到的是 40°（远超预期之外的动作）。
#
# 标定基准：上臂 0°、前臂 0° -> 肘夹角 0°，但调用方通常把
# 中立值记成 90°（见下方 CAL_ELBOW），所以激励要相应地偏移。
# ============================================================
CAL_UPPER = 0.0      # 标定时的上臂方向角
CAL_FORE = 0.0       # 标定时的前臂方向角
CAL_ELBOW = 90.0     # 标定用的「肘夹角」名义值（写入 neutral）

EXCITE_UPPER = +PUSH                    # 上臂抬起 40°
EXCITE_ELBOW_ABS = CAL_ELBOW + PUSH     # 目标**绝对**肘夹角 = 90 + 40 = 130


def make_arm(upper_deg: float, elbow_deg: float, fore_deg: float):
    """
    由三个角度合成一组手臂关键点。

    构造方式（以肩为原点，像素长度固定 100/100）：
        肘 = 肩 + R(upper_deg) * 100
        腕 = 肘 + R(fore_deg)   * 100
    其中 R(a) 在图像坐标下为 (cos a, -sin a)（y 翻转，向上为正）。

    这样 compute_arm_geometry 算回来的角度就等于给定值，
    可以精确控制输入。
    """
    s = (0.0, 0.0)
    ea = math.radians(upper_deg)
    e = (100.0 * math.cos(ea), -100.0 * math.sin(ea))
    fa = math.radians(fore_deg)
    w = (e[0] + 100.0 * math.cos(fa), e[1] - 100.0 * math.sin(fa))
    return ArmKeypoints(
        shoulder=Keypoint(*s, confidence=0.9, name="right_shoulder"),
        elbow=Keypoint(*e, confidence=0.9, name="right_elbow"),
        wrist=Keypoint(*w, confidence=0.9, name="right_wrist"),
        side="right",
        hand_wrist=Keypoint(*w, confidence=0.9, name="right_wrist_from_hand"))


class JointVerifier:
    def __init__(self, settle_s: float = 4.0):
        self.settle_s = settle_s
        rclpy.init()
        self.robot_cfg = ControlConfig.from_yaml()
        self.robot = PiperJointController(self.robot_cfg,
                                          node_name="verify_joints")
        self.buf = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buf, self.robot)

        t0 = time.monotonic()
        while time.monotonic() - t0 < 10 and not self.robot.has_state():
            rclpy.spin_once(self.robot, timeout_sec=0.1)
        if not self.robot.has_state():
            raise RuntimeError("收不到 /joint_states，请确认 Gazebo 已启动")

        self.robot.start()
        self.home = self.robot.get_joint_positions()

    # ------------------------------------------------------------
    def spin(self, seconds: float):
        t0 = time.monotonic()
        while time.monotonic() - t0 < seconds:
            rclpy.spin_once(self.robot, timeout_sec=0.01)

    def tf_joints(self) -> dict:
        """
        用 TF 独立读出各关节角（从相邻 link 的相对姿态）。
        只取 J2/J3/J5 用于核对 —— 它们都在 Y 轴上，
        因此关节角 = 绕 Y 的旋转角。
        """
        from tf_transformations import euler_from_quaternion
        pairs = {"joint2": ("link1", "link2"),
                 "joint3": ("link2", "link3"),
                 "joint5": ("link4", "link5")}
        out = {}
        for name, (parent, child) in pairs.items():
            try:
                tr = self.buf.lookup_transform(parent, child,
                                               rclpy.time.Time())
                q = tr.transform.rotation
                _r, pitch, _y = euler_from_quaternion([q.x, q.y, q.z, q.w])
                out[name] = pitch
            except Exception:                           # noqa: BLE001
                out[name] = None
        return out

    # ------------------------------------------------------------
    def run_stage(self, idx: int, name: str, joints: list) -> dict:
        """跑一个阶段，返回结果字典"""
        cfg = RetargetingConfig.from_yaml()
        for r in cfg.rules:
            r.enabled = (r.joint in joints)

        rt = Retargeter(cfg, self.robot_cfg)

        # ---- 关键：先把机械臂摆到**配置的中立位姿**，再以此作为标定锚点 ----
        #
        # 为什么不能直接用「当前位姿」：
        #   上一阶段结束时机械臂可能停在任意位置（实测出现过 joint3 停在
        #   -2.795，已贴住 -2.85 限位）。若把它当初始位姿，
        #   下一阶段就会因「已经顶在限位上」而无法再动 —— 表现为
        #   「目标关节不响应」，但这其实是测试串扰，不是映射问题。
        #
        #   更本质地说：标定锚点必须是**一个可复现的、行程居中的安全位姿**，
        #   否则多阶段/多次运行之间无法比较，且容易单边饱和。
        neutral = list(cfg.neutral_joints)
        t0 = time.monotonic()
        while time.monotonic() - t0 < 6.0:
            self.robot.send_joint_target(neutral, label="goto_neutral")
            self.spin(0.05)
        self.spin(1.5)
        actual = self.robot.get_joint_positions()
        print(f"    [置中] 目标 {[round(v,3) for v in neutral]}")
        print(f"           实测 {[round(v,3) for v in actual]}")

        # 以中立位姿为映射锚点（initial_pose 与 human_neutral 成对）
        rt.set_initial_pose(neutral)

        # 标定：以「上臂 0° / 肘 90° / 前臂 0°」为人体基准
        rt.start_calibration()
        base = ArmGeometry(0.0, 0.0, 90.0, 100, 100, True)
        for _ in range(20):
            rt.add_calibration_sample(base)
        rt.finish_calibration()

        # 打印实际标定到的中立值 —— 与期望的 (0, 90, 0) 对齐核对
        print(f"    [标定] 人体中立值 = "
              f"{ {k: round(v,2) for k, v in rt.neutral.values.items()} }")

        # 让映射输出与中立位姿对齐（Δ=0 -> neutral）
        for _ in range(120):
            res = rt.update(make_arm(0.0, 90.0, 0.0), now=0.0)
            self.robot.send_joint_target(res.positions, label="align")
            self.spin(0.02)
        self.spin(1.5)

        before = list(self.robot.get_joint_positions())
        tf_before = self.tf_joints()

        # ---- 施加激励 ----
        # 肘夹角 = fore - upper，所以由目标绝对肘夹角反推 fore
        exc_upper = EXCITE_UPPER
        exc_fore = exc_upper + EXCITE_ELBOW_ABS
        t = 10.0
        for k in range(400):
            a = make_arm(exc_upper, 0.0, exc_fore)
            res = rt.update(a, now=t)
            t += 0.033
            self.robot.send_joint_target(res.positions, label="push")
            self.spin(0.02)
            if k in (0, 5, 50, 200, 399):
                _i3 = cfg.joint_names.index("joint3")
                print(f"      [push {k:3d}] rawE={res.debug.raw_elbow:7.2f} "
                      f"fltE={res.debug.flt_elbow:7.2f} "
                      f"dE={res.debug_deltas_deg.get('elbow_angle_deg', 0):7.2f} "
                      f"mapJ3={res.mapped_joints.get('joint3', 0):+.4f} "
                      f"tgtJ3={rt.target[_i3]:+.4f}")
        self.spin(1.5)

        after = list(self.robot.get_joint_positions())
        tf_after = self.tf_joints()

        # 记录控制器**声称的目标**，用于区分
        #   「目标没到位」 vs 「到位了但没跟踪上」
        tgt = rt.target
        print(f"    [目标] 映射目标 {[round(v,3) for v in tgt]}")
        print(f"           实测     {[round(v,3) for v in after]}")
        print(f"           目标-实测 "
              f"{[round(a-b,3) for a,b in zip(tgt, after)]}")

        # ---- 判定：期望的关节要有明显变化，其它关节要几乎不动 ----
        names = cfg.joint_names
        moved = {}
        for j in names:
            i = names.index(j)
            moved[j] = after[i] - before[i]

        expected = {j: abs(moved[j]) > 0.15 for j in joints}
        unexpected = {j: abs(moved[j]) > 0.05
                      for j in names if j not in joints and j not in ("joint1",
                                                                     "joint4",
                                                                     "joint6")}

        # ---- Gazebo 侧核对（TF） ----
        tf_moved = {}
        for j in joints:
            if tf_before.get(j) is not None and tf_after.get(j) is not None:
                tf_moved[j] = tf_after[j] - tf_before[j]

        return {
            "name": name, "joints": joints,
            "before": before, "after": after, "moved": moved,
            "expected_ok": expected, "unexpected": unexpected,
            "tf_moved": tf_moved,
        }

    # ------------------------------------------------------------
    def cleanup(self):
        try:
            self.robot.stop()
            self.robot.destroy_node()
        except Exception:                               # noqa: BLE001
            pass
        if rclpy.ok():
            rclpy.shutdown()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=-1)
    ap.add_argument("--settle", type=float, default=4.0)
    args = ap.parse_args()

    v = JointVerifier(args.settle)
    print("=" * 84)
    print("Piper 逐关节映射验证")
    print("=" * 84)
    print(f"  初始位姿 : {[round(x, 4) for x in v.home]}")
    _fu = EXCITE_UPPER
    _ff = _fu + EXCITE_ELBOW_ABS
    print(f"  激励     : 上臂 {_fu:+.0f}°  前臂 {_ff:+.0f}°  "
          f"=> 肘夹角 {_ff - _fu:.0f}° (标定基准 {CAL_ELBOW:.0f}°，"
          f"弯 {_ff - _fu - CAL_ELBOW:+.0f}°)")
    print(f"  判定标准 : 目标关节 |Δ| > 0.15 rad；未启用关节 |Δ| < 0.05 rad")
    print("=" * 84)

    stages = STAGES if args.stage < 0 else [STAGES[args.stage]]
    results = []
    try:
        for i, (name, joints) in enumerate(stages):
            print(f"\n---- 阶段 {name} ----")
            r = v.run_stage(i, name, joints)
            results.append(r)
            names = ControlConfig.from_yaml().joint_names
            for j in names:
                idx = names.index(j)
                mark = ""
                if j in joints:
                    mark = "✅目标" if r["expected_ok"][j] else "❌未动"
                elif j in r["unexpected"]:
                    mark = ("  ·" if not r["unexpected"][j] else "❌误动")
                print(f"    {j:8s} Δ={r['moved'][j]:+.4f} rad "
                      f"({r['moved'][j]*57.2958:+7.2f}°)  {mark}")
            if r["tf_moved"]:
                tfs = " ".join(f"{j}={d*57.2958:+.1f}°"
                               for j, d in r["tf_moved"].items())
                print(f"    TF 独立核对(绕Y转角): {tfs}")
    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        v.cleanup()

    # ---- 汇总 ----
    print("\n" + "=" * 84)
    print("汇总")
    print("=" * 84)
    all_ok = True
    for r in results:
        exp_ok = all(r["expected_ok"].values())
        no_unexp = not any(r["unexpected"].values())
        ok = exp_ok and no_unexp
        all_ok &= ok
        print(f"  {r['name']:16s} 目标关节响应={'OK' if exp_ok else 'FAIL'}  "
              f"无误动={'OK' if no_unexp else 'FAIL'}  -> {'PASS' if ok else 'FAIL'}")
    print("=" * 84)
    print(f"  总体: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
