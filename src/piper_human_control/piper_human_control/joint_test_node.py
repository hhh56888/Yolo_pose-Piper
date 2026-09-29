# -*- coding: utf-8 -*-
"""
阶段二验收测试节点
==================
测试序列（对应任务书要求）：
    HOME  ->  修改 J2  ->  修改 J3  ->  修改 J5  ->  回 HOME

除此之外还包含：
    * 限位测试      —— 故意下发越界值，验证被裁剪而非直接执行
    * 非法输入测试  —— 关节数错误 / NaN，验证被拒绝
    * 急停测试      —— 触发急停后验证新目标被拒绝

运行：
    ros2 run piper_human_control joint_test
    # 或
    python3 -m piper_human_control.joint_test_node
"""

import math
import sys
import time

import rclpy

from .config import ControlConfig
from .controller import PiperJointController


# ---------------- 输出格式 ----------------
LINE = "=" * 100
SUB = "-" * 100


def fmt(vals, nd=4):
    """格式化关节数组"""
    if vals is None:
        return "None"
    return "[" + ", ".join(f"{v:+.{nd}f}" for v in vals) + "]"


def print_state(robot, tag=""):
    cur = robot.get_joint_positions()
    tgt = robot.get_target()
    print(f"  {tag:<22} 当前={fmt(cur)}")
    if tgt is not None:
        print(f"  {'':<22} 目标={fmt(tgt)}")


class Stage2Test:
    """阶段二验收测试"""

    def __init__(self):
        self.cfg = ControlConfig.from_yaml()
        self.robot = PiperJointController(self.cfg)
        self.results = []          # (名称, 是否通过, 说明)

    # ------------------------------------------------------------
    def record(self, name, ok, detail=""):
        self.results.append((name, ok, detail))
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name}" + (f"  -- {detail}" if detail else ""))

    # ------------------------------------------------------------
    def wait_ready(self, timeout=15.0):
        """等待 /joint_states 有数据"""
        print("\n[准备] 等待 /joint_states ...")
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            rclpy.spin_once(self.robot, timeout_sec=0.1)
            if self.robot.has_state():
                print(f"  [OK] 已收到状态反馈，用时 {time.monotonic()-t0:.2f}s")
                print_state(self.robot, "初始位置")
                return True
        print("  [FAIL] 超时未收到 /joint_states，请确认 Gazebo 仿真已启动")
        return False

    # ------------------------------------------------------------
    def goto(self, target, name, tol=0.02, timeout=25.0):
        """
        下发目标并等待到位。

        注意：到位判断与误差统计都针对「下发时的期望目标」，
        而不是控制器内部的中间目标，否则会出现「刚起步就报到位」的假通过。
        """
        print(f"\n  >> 下发 {name}: {fmt(target)}")
        acc = self.robot.send_joint_target(target, label=name)
        if not acc:
            print("     目标被拒绝")
            return False, None, None

        t0 = time.monotonic()
        ok = self.robot.wait_until_reached(tol=tol, timeout=timeout)
        elapsed = time.monotonic() - t0

        cur = self.robot.get_joint_positions()
        desired = self.robot.get_desired_target()
        err = self.robot.target_error()
        print(f"     结果: {'到位' if ok else '超时'}  (耗时 {elapsed:.2f}s)")
        print(f"     期望: {fmt(desired)}")
        print(f"     实测: {fmt(cur)}")
        if err is not None:
            max_err = max(abs(e) for e in err)
            print(f"     最大误差: {max_err:.4f} rad ({max_err*57.2958:.2f} deg)")
            # 额外校验：实测位置必须贴近「下发时请求的值」，
            # 而不是控制器内部的限速中间值
            req_err = max(abs(c - t) for c, t in zip(cur, target))
            print(f"     对请求值误差: {req_err:.4f} rad ({req_err*57.2958:.2f} deg)")
            if req_err > tol:
                print(f"     [注意] 对请求值误差 {req_err:.4f} 超过容差 {tol}")
                ok = False
            return ok, max_err, cur
        return ok, None, cur

    # ------------------------------------------------------------
    def run(self):
        cfg = self.cfg
        print(LINE)
        print("阶段二 · Piper 独立关节控制接口 —— 验收测试")
        print(LINE)
        print(f"配置来源: {cfg.source_path}")
        print(f"关节顺序: {cfg.joint_names}")
        print(f"HOME    : {fmt(cfg.home)}")

        if not self.wait_ready():
            return False

        self.robot.start()
        print("  [OK] 控制循环已启动")

        # ============================================================
        # 步骤 1：回 HOME
        # ============================================================
        print(f"\n{LINE}\n[步骤 1] 前往 HOME\n{LINE}")
        ok, err, _ = self.goto(cfg.home, "HOME")
        self.record("回 HOME", ok, f"最大误差 {err:.4f} rad" if err else "")

        # ============================================================
        # 步骤 2：修改 J2
        # ============================================================
        print(f"\n{LINE}\n[步骤 2] 修改 J2 (joint2)\n{LINE}")
        j2 = list(cfg.home)
        j2[1] = 1.2
        ok, err, _ = self.goto(j2, "J2 -> 1.2")
        self.record("修改 J2", ok, f"最大误差 {err:.4f} rad" if err else "")

        # 回 HOME 便于下一步观察
        self.goto(cfg.home, "回 HOME")

        # ============================================================
        # 步骤 3：修改 J3
        # ============================================================
        print(f"\n{LINE}\n[步骤 3] 修改 J3 (joint3)\n{LINE}")
        j3 = list(cfg.home)
        j3[2] = -1.2
        ok, err, _ = self.goto(j3, "J3 -> -1.2")
        self.record("修改 J3", ok, f"最大误差 {err:.4f} rad" if err else "")

        self.goto(cfg.home, "回 HOME")

        # ============================================================
        # 步骤 4：修改 J5
        # ============================================================
        print(f"\n{LINE}\n[步骤 4] 修改 J5 (joint5)\n{LINE}")
        j5 = list(cfg.home)
        j5[4] = 0.9
        ok, err, _ = self.goto(j5, "J5 -> 0.9")
        self.record("修改 J5", ok, f"最大误差 {err:.4f} rad" if err else "")

        # ============================================================
        # 步骤 5：回 HOME
        # ============================================================
        print(f"\n{LINE}\n[步骤 5] 回 HOME\n{LINE}")
        ok, err, _ = self.goto(cfg.home, "回 HOME")
        self.record("最终回 HOME", ok, f"最大误差 {err:.4f} rad" if err else "")

        # ============================================================
        # 步骤 6：限位裁剪测试
        # ============================================================
        print(f"\n{LINE}\n[步骤 6] 安全测试 · 越界值应被裁剪\n{LINE}")
        # joint2 上限 3.14，故意给 99；joint3 上限 0，故意给 +5
        bad = list(cfg.home)
        bad[1] = 99.0
        bad[2] = 5.0
        print(f"  下发越界值: {fmt(bad)}")
        self.robot.send_joint_target(bad, label="越界测试")
        tgt = self.robot.get_target()
        print(f"  内部目标已裁剪为: {fmt(tgt)}")
        if tgt is not None:
            in_range = (tgt[1] <= cfg.upper_limits["joint2"] + 1e-6 and
                        tgt[2] <= cfg.upper_limits["joint3"] + 1e-6)
            self.record("越界值被裁剪到限位", in_range,
                        f"j2={tgt[1]:.4f}(上限{cfg.upper_limits['joint2']}), "
                        f"j3={tgt[2]:.4f}(上限{cfg.upper_limits['joint3']})")
        else:
            self.record("越界值被裁剪到限位", False, "目标为 None")

        self.goto(cfg.home, "回 HOME")

        # ============================================================
        # 步骤 7：非法输入测试
        # ============================================================
        print(f"\n{LINE}\n[步骤 7] 安全测试 · 非法输入应被拒绝\n{LINE}")
        r1 = self.robot.send_joint_target([0.0, 0.5], label="长度错误")
        print(f"  长度错误 (2 个值) -> 接受={r1}")
        self.record("关节数错误被拒绝", not r1)

        r2 = self.robot.send_joint_target(
            [0.0, float("nan"), 0.0, 0.0, 0.0, 0.0], label="NaN")
        print(f"  NaN 输入 -> 接受={r2}")
        self.record("NaN 输入被拒绝", not r2)

        r3 = self.robot.send_joint_target(
            [0.0, float("inf"), 0.0, 0.0, 0.0, 0.0], label="Inf")
        print(f"  Inf 输入 -> 接受={r3}")
        self.record("Inf 输入被拒绝", not r3)

        # ============================================================
        # 步骤 8：急停测试
        # ============================================================
        print(f"\n{LINE}\n[步骤 8] 安全测试 · 软件急停\n{LINE}")
        self.robot.emergency_stop("验收测试触发")
        r4 = self.robot.send_joint_target([0.0, 2.0, -2.0, 0.0, 1.0, 0.0], label="急停后")
        print(f"  急停后下发新目标 -> 接受={r4}")
        self.record("急停后拒绝新目标", not r4)

        # 急停期间位置应基本不变
        p_before = self.robot.get_joint_positions()
        t0 = time.monotonic()
        while time.monotonic() - t0 < 2.0:
            rclpy.spin_once(self.robot, timeout_sec=0.05)
        p_after = self.robot.get_joint_positions()
        drift = max(abs(a - b) for a, b in zip(p_before, p_after)) if p_before else 0.0
        print(f"  急停期间位置漂移: {drift:.5f} rad")
        self.record("急停期间保持位置", drift < 0.05, f"漂移 {drift:.5f} rad")

        self.robot.reset_emergency_stop()

        # ============================================================
        # 步骤 9：连续流式发送测试
        # ============================================================
        print(f"\n{LINE}\n[步骤 9] 流式发送测试 · 高频更新目标\n{LINE}")
        # 模拟阶段四「每帧更新目标」的场景：以 50 Hz 连续改目标
        print("  以 50 Hz 连续改变 J5 目标 (模拟视觉帧率驱动) ...")
        t0 = time.monotonic()
        n_sent = 0
        while time.monotonic() - t0 < 4.0:
            phase = (time.monotonic() - t0) / 4.0          # 0 -> 1
            q = list(cfg.home)
            q[4] = cfg.home[4] + 0.6 * math.sin(phase * math.pi)
            if self.robot.send_joint_target(q, label="stream"):
                n_sent += 1
            rclpy.spin_once(self.robot, timeout_sec=0.02)
        cur = self.robot.get_joint_positions()
        print(f"  发送 {n_sent} 次目标，当前: {fmt(cur)}")
        self.record("高频流式发送无异常", n_sent > 100, f"发送 {n_sent} 次")

        # 收敛回 HOME
        ok, err, _ = self.goto(cfg.home, "回 HOME", timeout=25.0)
        self.record("流式后仍能收敛回 HOME", ok,
                    f"最大误差 {err:.4f} rad" if err else "")

        return True

    # ------------------------------------------------------------
    def summary(self):
        print(f"\n{LINE}")
        print("[测试汇总]")
        print(LINE)
        passed = [r for r in self.results if r[1]]
        failed = [r for r in self.results if not r[1]]
        for name, ok, detail in self.results:
            mark = "PASS" if ok else "FAIL"
            print(f"  [{mark}] {name:<28} {detail}")
        print(SUB)
        print(f"  合计 {len(self.results)} 项 | 通过 {len(passed)} | 失败 {len(failed)}")
        if failed:
            print("\n  失败项:")
            for name, _, detail in failed:
                print(f"    - {name}  {detail}")
        print(f"\n  >>> 阶段二验收: {'通过' if not failed else '未通过'}")
        return not failed


def main(args=None):
    rclpy.init(args=args)
    test = Stage2Test()
    exit_code = 1
    try:
        if test.run():
            exit_code = 0 if test.summary() else 1
        else:
            print("\n  [FATAL] 前置条件不满足，测试中止")
            exit_code = 1
    except KeyboardInterrupt:
        print("\n  用户中断")
    finally:
        try:
            test.robot.stop()
            test.robot.destroy_node()
        except Exception:                              # noqa: BLE001
            pass
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
