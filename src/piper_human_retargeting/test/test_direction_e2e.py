# -*- coding: utf-8 -*-
"""
端到端方向语义测试（任务书 §2 的核心要求）
==========================================
旧测试只检查 `mapping.Rule.map()`，**绕过了 sign/invert**，
因此无法保护生产方向（见 docs/J3_Elbow_Mapping_Audit.md §10.2）。

本文件改为走**完整生产链**：

    Human elbow flexion
        ↓
    生产 Retargeter（含 sign / invert / 滤波 / 标定）
        ↓
    joint3 target
        ↓
    Piper FK（由实际 URDF 生成，且已用仿真 TF 校准）
        ↓
    TCP

并锁定两条语义：
    * 人弯肘 -> 机器人**收臂**（TCP 离肩更近、更低）
    * 人伸肘 -> 机器人**伸臂**（TCP 更远、更高）

另外覆盖 V1.1 task-space 的两条底线：
    * 人手抬高 -> TCP z 抬高（原来 corr(手高,TCP高) ≈ −0.353 的问题）
    * task_space 模式下肘角对 q2/q3 **没有权限**（任务书 §15）

运行：
    python3 -m pytest test/test_direction_e2e.py -q -p no:anyio
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

from piper_human_control import ArmKeypoints, Keypoint          # noqa: E402
from piper_human_retargeting import Retargeter, RetargetingConfig  # noqa: E402
from piper_human_retargeting.arm_geometry import compute_arm_geometry  # noqa: E402
from piper_human_retargeting.kinematics import load_default_chain   # noqa: E402

from conftest import disable_wrist_rules                        # noqa: E402

SHOULDER_XYZ = np.array([0.0, 0.0, 0.123])      # joint2 转轴在 base_link 下的位置


def make_arm(upper_deg, fore_deg, side="right"):
    """与 test/verify_joints.py 相同的构造：肘夹角 = fore - upper"""
    ea = math.radians(upper_deg)
    e = (100.0 * math.cos(ea), -100.0 * math.sin(ea))
    fa = math.radians(fore_deg)
    w = (e[0] + 100.0 * math.cos(fa), e[1] - 100.0 * math.sin(fa))
    return ArmKeypoints(
        shoulder=Keypoint(0.0, 0.0, confidence=0.9, name=f"{side}_shoulder"),
        elbow=Keypoint(*e, confidence=0.9, name=f"{side}_elbow"),
        wrist=Keypoint(*w, confidence=0.9, name=f"{side}_wrist"),
        side=side,
        hand_wrist=Keypoint(*w, confidence=0.9,
                            name=f"{side}_wrist_from_hand"))


def calibrate(rt, neutral_elbow_deg, upper_deg=0.0, fore_deg=None, n=30):
    """用合成姿势标定（neutral 处 u/v、肘角都在这里定零）"""
    fore = neutral_elbow_deg if fore_deg is None else fore_deg
    rt.set_initial_pose(list(rt.cfg.active_neutral_joints()))
    rt.start_calibration()
    geo = compute_arm_geometry(make_arm(upper_deg, fore))
    for _ in range(n):
        rt.add_calibration_sample_for(make_arm(upper_deg, fore), geo)
    assert rt.finish_calibration()


def settle(rt, arm, frames=40):
    res = None
    for _ in range(frames):
        res = rt.update(arm, now=0.0)
    return res


def tcp_and_reach(cfg, positions):
    model = load_default_chain()
    q = {n: float(v) for n, v in zip(cfg.joint_names, positions)}
    for n in ("joint1", "joint4", "joint5", "joint6"):
        q[n] = 0.0
    p = model.tcp(q)
    return p, float(np.linalg.norm(p - SHOULDER_XYZ))


# ============================================================
# ① legacy：人弯肘 -> 收臂
# ============================================================
class TestLegacyElbowDirection:
    @pytest.fixture
    def rt(self, cfg, robot_cfg):
        disable_wrist_rules(cfg)
        r = Retargeter(cfg, robot_cfg)
        calibrate(r, 90.0)
        return r

    def test_human_flexion_increases_joint3(self, rt, cfg):
        """人弯肘（flexion ↑）-> joint3 朝"收臂"方向变化（增大）"""
        i3 = cfg.joint_names.index("joint3")
        rt_0 = settle(rt, make_arm(0.0, 90.0)).positions[i3]
        bent = settle(rt, make_arm(0.0, 170.0)).positions[i3]
        assert bent > rt_0, f"弯肘后 joint3={bent:+.4f} 未朝收臂方向（{rt_0:+.4f}）"

    def test_human_flexion_reduces_tcp_radial_reach(self, rt, cfg):
        """
        任务书 §2 要求的那条语义（走完 Retargeter + FK 全链）：
            Human elbow flexion ↑  ->  Robot arm radial reach ↓
        """
        straight = settle(rt, make_arm(0.0, 15.0)).positions
        bent = settle(rt, make_arm(0.0, 165.0)).positions
        _, r_straight = tcp_and_reach(cfg, straight)
        _, r_bent = tcp_and_reach(cfg, bent)
        assert r_bent < r_straight, (
            f"弯肘时 TCP 半径应变小: 伸直 {r_straight:.4f} -> 弯 {r_bent:.4f}")

    def test_human_extension_increases_tcp_radial_reach(self, rt, cfg):
        bent = settle(rt, make_arm(0.0, 165.0)).positions
        straight = settle(rt, make_arm(0.0, 15.0)).positions
        _, r_bent = tcp_and_reach(cfg, bent)
        _, r_straight = tcp_and_reach(cfg, straight)
        assert r_straight > r_bent

    def test_direction_is_monotone_over_the_whole_range(self, rt, cfg):
        """单调性：连续弯肘 -> TCP 半径单调不增（挡"中途翻转"的配置错误）"""
        reaches = []
        for bend in (10, 40, 70, 100, 130, 160):
            pos = settle(rt, make_arm(0.0, float(bend))).positions
            reaches.append(tcp_and_reach(cfg, pos)[1])
        diffs = np.diff(reaches)
        assert np.all(diffs <= 1e-6), f"半径非单调: {reaches}"


# ============================================================
# ② task_space：人手抬高 -> TCP 抬高
# ============================================================
class TestWristPitchDirection:
    """
    腕部俯仰（J5）方向的**标定值锁定**。

    方向不是几何推导出来的：2026-09-29 实时识别测试中操作者反馈
    invert=true 时「手腕上翘/下压反了」，翻成 false 后确认正确。
    人手侧的角度符号（手往上翘时 wrist_pitch_deg 变大还是变小）取决于
    相机站位与手的姿态，单目 2D 下不可靠推导 —— 与 task_space.sign_h 同类。

    本测试把**当前标定值**对应的映射方向钉住：以后谁再动 invert，
    必须同步改这里并说明依据（配置注释里也写了实测来源）。
    """

    def test_wrist_pitch_direction_follows_calibration(self, cfg):
        r = cfg.rule_for("joint5")
        assert r.human == "wrist_pitch_deg"
        q_lo, _ = r.map(-45.0)          # 手往下压
        q_hi, _ = r.map(+45.0)          # 手往上翘
        dq = q_hi - q_lo
        if r.invert:
            # invert=true：上翘 -> joint5 减小（实测末端指向向上）
            assert dq < 0
        else:
            # invert=false（当前实测标定）：上翘 -> joint5 增大
            assert dq > 0
        assert abs(dq) > 0.3, "腕俯仰通道行程过小"

    def test_calibration_matches_live_observation(self, cfg):
        """把"实测标定值"写死在断言里：改动必须是有意识的"""
        assert cfg.rule_for("joint5").invert is False, (
            "j5_wrist.invert 的当前标定值来自 2026-09-29 实时测试"
            "（操作者确认 invert=true 时上下反了）。若确有新证据要改，"
            "请同步更新本测试与 config/retargeting.yaml 的注释。")


class TestTaskSpaceVerticalSemantics:
    @pytest.fixture
    def rt(self, cfg, robot_cfg):
        cfg.retarget_mode = "task_space"
        disable_wrist_rules(cfg)
        r = Retargeter(cfg, robot_cfg)
        # 标定姿势：手腕在肩水平线上（v0 = 0）
        calibrate(r, 90.0, upper_deg=0.0, fore_deg=90.0)
        return r

    def test_hand_up_raises_tcp(self, rt, cfg):
        """
        审计发现的老问题：corr(人体手高, TCP 高) ≈ −0.353。
        task-space 模式必须把它变成**正相关**：
        让手腕在图像里抬高，TCP 的 z 必须跟着升高。
        """
        # 注意构造：u = (cos ua + cos f)/2、v = (sin ua + sin f)/2（L1=L2=100）。
        # 要"只改高度、不改水平"，必须让 ua 与 f 同号同值：
        #   (+30°,+30°) -> (u,v) = (0.866,+0.5)
        #   (  0°,  0°) -> (1.000, 0.0)
        #   (-30°,-30°) -> (0.866,-0.5)
        # 前两者的 u 相同，只有 v 反号 —— 这才是"抬手/放手"。
        # （踩过的坑：写成 fore 55°/125° 只改了肘弯，手腕高度反而对称不变）
        arm_up = make_arm(30.0, 30.0)      # 手抬高
        arm_mid = make_arm(0.0, 0.0)       # 水平
        arm_dn = make_arm(-30.0, -30.0)    # 手放低

        z_mid = tcp_and_reach(cfg, settle(rt, arm_mid).positions)[0][2]
        z_up = tcp_and_reach(cfg, settle(rt, arm_up).positions)[0][2]
        z_dn = tcp_and_reach(cfg, settle(rt, arm_dn).positions)[0][2]
        assert z_up > z_mid > z_dn, (
            f"TCP z 未跟随手的高度: 上 {z_up:.4f} / 中 {z_mid:.4f} / 下 {z_dn:.4f}")

    def test_hand_horizontal_motion_follows_configured_sign(self, rt, cfg):
        """
        水平方向的语义由**配置的 sign_h** 决定 —— 它是"相机站位标定值"：

            sign_h = +1：侧视机位（图像水平 ≈ 机器人前后），手前伸(u↑) -> TCP 前伸
            sign_h = -1：正面机位（u 是投影量，手前伸时 u↓），手前伸 -> TCP 前伸

        实测依据见 docs/2D_TaskSpace_Retargeting_V1_1_Report.md §13.2。
        本测试不写死 +1，而是断言"实际运动方向与配置一致"——
        这样无论是哪种机位，配置改错都会被抓到。
        """
        sign = cfg.task_space.sign_h
        arm_near = make_arm(-40.0, -40.0)     # u = 0.766
        arm_far = make_arm(-20.0, -20.0)      # u = 0.940
        x_near = tcp_and_reach(cfg, settle(rt, arm_near).positions)[0][0]
        x_far = tcp_and_reach(cfg, settle(rt, arm_far).positions)[0][0]
        du = (0.940 - 0.766)
        dx = x_far - x_near
        assert abs(dx) > 0.01, f"水平通道无响应: Δx={dx:+.4f}"
        assert (du > 0) == (dx > 0) if sign > 0 else (du > 0) != (dx > 0), (
            f"sign_h={sign} 时方向不符: Δu={du:+.3f} -> ΔTCPx={dx:+.4f}")

    def test_u_v_depend_only_on_shoulder_and_wrist(self, rt, cfg):
        """
        任务书 §15/§16：task_space 模式下 q2/q3 只由**手腕相对肩的位置**决定。

        注意"肘"在这里的精确语义：真实手臂是**刚性两连杆**，
        |SE| 与 |EW| 是人体上臂/前臂长度，不随姿势变化，
        因此 L = |SE|+|EW| 恒定、u/v 只取决于 S 和 W。
        （构造反例时若把肘挪到别处而不保持 |SE|、|EW|，等于造了一条
          "会伸缩的胳膊"，那不是人体，L 会变、u/v 也会跟着变 —— 见下面的断言。）

        这里用**镜像肘**构造同一 S/W 下的两种构型（|SE|、|EW| 完全相同），
        要求 u/v 与最终 q2/q3 都一致。
        """
        arm_a = make_arm(0.0, 90.0)          # S=(0,0) E=(100,0) W=(100,-100)
        arm_b = ArmKeypoints(                # 镜像肘 E=(0,-100)：|SE|=|EW|=100
            shoulder=arm_a.shoulder,
            elbow=Keypoint(0.0, -100.0, confidence=0.9, name="right_elbow"),
            wrist=arm_a.wrist,
            side="right",
            hand_wrist=arm_a.hand_wrist)
        ga = compute_arm_geometry(arm_a)
        gb = compute_arm_geometry(arm_b)
        assert ga.upper_arm_len_px == pytest.approx(gb.upper_arm_len_px)
        assert ga.forearm_len_px == pytest.approx(gb.forearm_len_px)
        assert ga.human_u == pytest.approx(gb.human_u, abs=1e-9)
        assert ga.human_v == pytest.approx(gb.human_v, abs=1e-9)
        pa = settle(rt, arm_a).positions
        pb = settle(rt, arm_b).positions
        i2 = cfg.joint_names.index("joint2")
        i3 = cfg.joint_names.index("joint3")
        # u/v 完全一致（1e-9）；关节解允许 0.02 rad 的差异 ——
        # 来自 IK 的停止判据（max|Δq|<1e-4）与 warm start 历史，
        # 不是"肘角在起作用"。
        assert pa[i2] == pytest.approx(pb[i2], abs=0.02)
        assert pa[i3] == pytest.approx(pb[i3], abs=0.02)

    def test_nonrigid_arm_changes_scale_but_not_direction(self, rt, cfg):
        """
        记录一个边界事实：若肘的位置改变而 |SE|、|EW| 不守恒（非刚体构造），
        L 会变，u/v 随之被"尺度"影响 —— 这不是控制链的 bug，
        而是"归一化尺度依赖手臂长度"的必然结果。
        真实人体不会出现，A/B 数据里也不会。
        """
        a = make_arm(0.0, 90.0)
        b = ArmKeypoints(shoulder=a.shoulder,
                         elbow=Keypoint(50.0, 70.0, confidence=0.9,
                                        name="right_elbow"),
                         wrist=a.wrist, side="right",
                         hand_wrist=a.hand_wrist)
        ga = compute_arm_geometry(a)
        gb = compute_arm_geometry(b)
        assert ga.elbow_angle_deg != pytest.approx(gb.elbow_angle_deg)
        assert abs(ga.human_u - gb.human_u) > 0.05      # 尺度受影响（预期内）

    def test_elbow_data_still_recorded(self, rt, cfg):
        """任务书 §16：肘角不参与控制，但必须仍被记录（将来做姿态偏好）"""
        res = settle(rt, make_arm(0.0, 120.0))
        assert res.debug is not None
        assert res.debug.raw_elbow > 0
        row = res.debug.to_csv_row()
        assert "raw_elbow" in row and "human_v_flt" in row
        assert "ik_q2" in row and "ik_q3" in row and "ik_time_ms" in row

    def test_joint2_joint3_columns_are_real_in_task_space(self, rt, cfg):
        """
        回归（实时测试踩到）：task_space 模式下 joint2/joint3 由 IK 决定、
        不在 controlled_joints 里；若按 controlled_joints 记录，
        CSV 的 mapped_joint2/3 与 joint2/3 会**恒为 0** ——
        看起来像"这两个关节没被控制"，实际它们在动。
        """
        res = settle(rt, make_arm(0.0, 70.0))
        row = res.debug.to_csv_row()
        q2 = float(row["joint2"])
        q3 = float(row["joint3"])
        assert abs(q2) > 1e-6 and abs(q3) > 1e-6, f"j2/j3 列恒 0: {q2}, {q3}"
        assert abs(float(row["mapped_joint2"])) > 1e-6
        assert abs(float(row["ik_q2"]) - q2) < 0.05      # 与 IK 解一致

    def test_joint_columns_never_silently_zero_on_early_return(self, rt, cfg):
        """
        回归（实时测试踩到）：手臂丢失 / 标定中 等早退路径不填 final_joints，
        CSV 会用 .get(j, 0.0) 填成 0.0000 —— 看起来像"关节归零"，
        而机械臂其实保持不动。关节列必须永远等于**当前实际目标**。
        """
        rt.update(None, now=0.0)                  # 无手臂数据 -> 早退路径
        res = rt.update(None, now=0.0)
        row = res.debug.to_csv_row()
        tgt = rt.target
        j2 = float(row["joint2"])
        assert abs(j2 - tgt[cfg.joint_names.index("joint2")]) < 1e-9, (
            f"早退帧 joint2 记成 {j2}，实际目标 {tgt[cfg.joint_names.index('joint2')]}")

    def test_all_new_quantities_are_in_csv(self, rt, cfg):
        """任务书 §18：新增控制量必须全部进 CSV（raw/filtered/target/IK）"""
        res = settle(rt, make_arm(0.0, 70.0))
        cols = set(res.debug.to_csv_row())
        need = {"human_u_raw", "human_v_raw", "human_u_flt", "human_v_flt",
                "du", "dv", "du_deadzoned", "dv_deadzoned",
                "tcp_target_x", "tcp_target_z", "ik_q2", "ik_q3",
                "ik_err_x", "ik_err_z", "ik_jump", "ik_iters", "ik_time_ms"}
        assert need <= cols, f"缺少列: {need - cols}"
        hdr = set(type(res.debug).csv_header())
        assert need <= hdr, f"表头缺少: {need - hdr}"


class TestTaskSpaceTracking:
    def test_targets_are_hit_within_tolerance(self, cfg, robot_cfg):
        """目标 z 与 IK 解出的 TCP z 必须一致（否则"跟手"是假的）"""
        cfg.retarget_mode = "task_space"
        disable_wrist_rules(cfg)
        rt = Retargeter(cfg, robot_cfg)
        calibrate(rt, 90.0, upper_deg=0.0, fore_deg=90.0)
        model = load_default_chain()
        for fore in (60.0, 75.0, 90.0, 105.0, 120.0):
            res = settle(rt, make_arm(0.0, fore))
            q = {n: float(v) for n, v in zip(cfg.joint_names, res.positions)}
            for n in ("joint1", "joint4", "joint5", "joint6"):
                q[n] = 0.0
            p = model.tcp(q)
            assert abs(p[2] - res.debug.tcp_target_z) < 2e-3, (
                f"fore={fore}: TCP z={p[2]:.4f} 目标 z={res.debug.tcp_target_z:.4f}")

    def test_no_failure_status_in_normal_motion(self, cfg, robot_cfg):
        """正常动作里不允许出现 IK 未收敛（状态行会写明原因）"""
        cfg.retarget_mode = "task_space"
        disable_wrist_rules(cfg)
        rt = Retargeter(cfg, robot_cfg)
        calibrate(rt, 90.0, upper_deg=0.0, fore_deg=90.0)
        for fore in np.linspace(60, 120, 25):
            res = settle(rt, make_arm(0.0, float(fore)), frames=6)
            assert "未收敛" not in res.status, res.status
