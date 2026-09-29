# -*- coding: utf-8 -*-
"""
Retargeter 集成测试（新架构）
=============================
针对统一映射 + 三级滤波 + 状态机结构。

本文件聚焦「整条链路串起来之后是否还对」：
    * 标定（相对角零点）
    * 初始位姿与 neutral 必须成对
    * 区间映射在 Retargeter 里被正确调用
    * 固定关节（J1/J4/J6）永不参与映射
    * 结果始终落在机械臂真实限位内
    * 逐关节开关（用于 J3-only / J2-only 等验证阶段）
    * 调试快照（raw/filtered）内容完整

单点数学细节分别由 test_mapping.py 与 test_tracker_filter.py 覆盖，
本文件不重复。

运行：
    python3 -m pytest test/test_retargeting.py -q -p no:anyio
"""

import math
import os
import sys

import pytest

# 本模块为**纯手臂**测试：不构造手部姿态，因此需要关掉
# 依赖 wrist_pitch_deg / palm_roll_deg 的腕部规则（见 conftest）。
# 手臂映射/状态机/滤波/CSV 测试（纯手臂，不提供手部姿态）
ARM_ONLY_TESTS = True

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_HERE)
_SRC = os.path.dirname(_PKG_ROOT)
for p in (_PKG_ROOT, os.path.join(_SRC, "piper_human_control")):
    if p not in sys.path:
        sys.path.insert(0, p)

from piper_human_control import (ArmKeypoints, ControlConfig,   # noqa: E402
                                 Keypoint)
from piper_human_retargeting import (ArmGeometry, Retargeter,   # noqa: E402
                                     RetargetingConfig, TrackState,
                                     compute_arm_geometry)


# ============================================================
# 夹具
# ============================================================
# 注意：cfg / robot_cfg 夹具已统一放在 conftest.py。
# 曾经这里也定义过一份 cfg，结果**模块内的定义覆盖了 conftest 的版本**，
# 导致 conftest 里「按模块标记关掉腕部规则」的逻辑完全不生效 ——
# 排查了好一阵。夹具只留一处。
#
# 另外 cfg 必须是**函数级**：多个测试会就地改 cfg.rules / neutral_joints，
# 共享实例会互相污染（实际踩过：前一个测试关掉滤波后，后一个拿到的也是关的）。


def make_arm(s=(0, 0), e=(100, 0), w=(100, 100),
             cs=0.9, ce=0.9, cw=0.9):
    return ArmKeypoints(
        shoulder=Keypoint(*s, confidence=cs, name="right_shoulder"),
        elbow=Keypoint(*e, confidence=ce, name="right_elbow"),
        wrist=Keypoint(*w, confidence=cw, name="right_wrist"),
        side="right",
        hand_wrist=Keypoint(*w, confidence=cw, name="right_wrist_from_hand"))


def calibrated(cfg, robot_cfg, upper=0.0, elbow=90.0, forearm=0.0):
    """
    建一个已标定的 Retargeter。

    注意：三级滤波都是 EMA，需要若干帧才收敛。
    断言精确值时应先 warmup 或关闭对应滤波器（见各用例）。
    """
    rt = Retargeter(cfg, robot_cfg)
    rt.start_calibration()
    g = ArmGeometry(upper, forearm, elbow, 100, 100, True)
    for _ in range(20):
        rt.add_calibration_sample(g)
    assert rt.finish_calibration()
    rt.set_initial_pose(cfg.neutral_joints)
    return rt


# ============================================================
# 标定
# ============================================================
class TestCalibration:
    def test_not_calibrated_initially(self, cfg, robot_cfg):
        assert not Retargeter(cfg, robot_cfg).is_calibrated

    def test_holds_before_calibration(self, cfg, robot_cfg):
        rt = Retargeter(cfg, robot_cfg)
        before = rt.target
        res = rt.update(make_arm(), now=0.0)
        assert res.positions == pytest.approx(before)
        assert not res.changed
        assert "未标定" in res.status

    def test_requires_min_samples(self, cfg, robot_cfg):
        rt = Retargeter(cfg, robot_cfg)
        rt.start_calibration()
        for _ in range(2):
            rt.add_calibration_sample(ArmGeometry(0, 0, 90, 100, 100, True))
        assert not rt.finish_calibration()

    def test_median_robust_to_outlier(self, cfg, robot_cfg):
        rt = Retargeter(cfg, robot_cfg)
        rt.start_calibration()
        for _ in range(20):
            rt.add_calibration_sample(
                ArmGeometry(10.0, 20.0, 30.0, 100, 100, True))
        rt.add_calibration_sample(
            ArmGeometry(170.0, 20.0, 30.0, 100, 100, True))
        rt.finish_calibration()
        # 断言**原始存储值**：本用例测的是「中位数抗离群」，
        # 与符号修正（mapping.*.sign）无关。
        # neutral.get() 会施加符号，不要拿它来验证中位数。
        assert rt.neutral.values["upper_arm_angle_deg"] == pytest.approx(10.0)
        # 同时确认符号约定：get() = values × sign
        sign = cfg.measure_sign()["upper_arm_angle_deg"]
        assert rt.neutral.get("upper_arm_angle_deg") == pytest.approx(10.0 * sign)

    def test_invalid_samples_not_counted(self, cfg, robot_cfg):
        rt = Retargeter(cfg, robot_cfg)
        rt.start_calibration()
        for _ in range(50):
            rt.add_calibration_sample(
                ArmGeometry(0, 0, 0, 100, 100, False, "退化"))
        assert not rt.finish_calibration()


# ============================================================
# 初始位姿
# ============================================================
class TestInitialPose:
    def test_set_updates_neutral_too(self, cfg, robot_cfg):
        """initial_pose 与 neutral_joints 必须一致，否则映射零点错位"""
        rt = calibrated(cfg, robot_cfg)
        target = [0.0, 0.75, -0.55, 0.0, 0.1, 0.0]
        rt.set_initial_pose(target)
        assert rt.initial_pose == pytest.approx(target)
        assert cfg.neutral_joints == pytest.approx(target)
        assert rt.target == pytest.approx(target)

    def test_rejects_wrong_length(self, cfg, robot_cfg):
        rt = calibrated(cfg, robot_cfg)
        with pytest.raises(ValueError):
            rt.set_initial_pose([0.0, 0.5])

    def test_keeps_fixed_joints(self, cfg, robot_cfg):
        rt = calibrated(cfg, robot_cfg)
        rt.set_initial_pose([0.0, 0.8, -0.5, 0.5, 0.0, 0.5])
        for name, val in cfg.fixed_joints.items():
            idx = cfg.joint_names.index(name)
            assert rt.target[idx] == pytest.approx(val)

    def test_mapping_anchored_on_initial_pose(self, cfg, robot_cfg):
        """
        人保持标定姿势（Δ=0）时，映射结果应落在区间中点。
        这里关闭滤波器以断言精确值。
        """
        rt = calibrated(cfg, robot_cfg)
        rt.joint_filter.cfg.enabled = False
        rt.angle_filter.cfg.enabled = False

        res = rt.update(make_arm(e=(100, 0), w=(100, 100)), now=0.0)
        r3 = cfg.rule_for("joint3")
        expect = r3.robot_min + 0.5 * (r3.robot_max - r3.robot_min)
        if r3.invert:
            expect = r3.robot_min + r3.robot_max - expect
        expect += r3.offset
        expect = max(r3.joint_lower, min(r3.joint_upper, expect))
        assert res.mapped_joints["joint3"] == pytest.approx(expect, abs=1e-6)


# ============================================================
# 映射在 Retargeter 中的行为
# ============================================================
class TestMappingInRetargeter:
    def test_respects_robot_limits(self, cfg, robot_cfg):
        """输出必须始终落在机械臂真实限位内"""
        rt = calibrated(cfg, robot_cfg)
        cases = [((100, 0), (200, 0)), ((0, 100), (0, 200)),
                 ((100, 0), (0, 100)), ((-100, 0), (-200, 0))]
        for k, (e, w) in enumerate(cases):
            for j in range(60):
                res = rt.update(make_arm(e=e, w=w), now=k + j * 0.001)
            for i, name in enumerate(cfg.joint_names):
                lo = rt.robot_cfg.lower_limits[name]
                hi = rt.robot_cfg.upper_limits[name]
                assert lo - 1e-9 <= res.positions[i] <= hi + 1e-9, name

    def test_fixed_joints_never_change(self, cfg, robot_cfg):
        rt = calibrated(cfg, robot_cfg)
        for k in range(60):
            res = rt.update(make_arm(e=(100, 0), w=(0, 150)), now=k * 0.033)
        for name, val in cfg.fixed_joints.items():
            idx = cfg.joint_names.index(name)
            assert res.positions[idx] == pytest.approx(val)

    def test_elbow_bend_moves_joint3(self, cfg, robot_cfg):
        """弯肘必须显著改变 J3 —— 2D 模式下最可靠的量"""
        rt = calibrated(cfg, robot_cfg, elbow=90.0)
        idx3 = cfg.joint_names.index("joint3")

        for k in range(150):
            rt.update(make_arm(e=(100, 0), w=(100, 100)), now=k * 0.033)
        j3_straight = rt.target[idx3]

        # 大幅弯肘：腕靠近肩 -> 肘夹角增大
        for k in range(150):
            rt.update(make_arm(e=(100, 0), w=(20, 20)), now=6 + k * 0.033)
        j3_bent = rt.target[idx3]

        assert abs(j3_bent - j3_straight) > 0.2

    def test_invalid_geometry_holds_target(self, cfg, robot_cfg):
        """关键点重合（几何退化）时应保持不动，不放行错误值"""
        rt = calibrated(cfg, robot_cfg)
        for k in range(60):
            rt.update(make_arm(e=(100, 0), w=(0, 150)), now=k * 0.033)
        before = list(rt.target)
        res = rt.update(make_arm(e=(0, 0), w=(0, 0)), now=3.0)
        assert res.positions == pytest.approx(before)

    def test_angle_wraparound_handled(self, cfg, robot_cfg):
        """
        角度周期归一化：标定在 +170°、当前 -170° 时，
        真实差是 +20°（跨过 180° 射线），不是 -340°。
        若未归一化，结果会被 clamp 到区间端点。
        """
        rt = calibrated(cfg, robot_cfg, upper=170.0)
        rt.joint_filter.cfg.enabled = False
        rt.angle_filter.cfg.enabled = False

        a = math.radians(-170.0)
        e = (100 * math.cos(a), -100 * math.sin(a))
        res = rt.update(make_arm(e=e, w=(e[0], e[1] - 50)), now=0.0)
        got = res.mapped_joints["joint2"]

        r2 = cfg.rule_for("joint2")
        # 不应落在区间端点（那是 Δ=-340 被 clamp 的结果）
        assert abs(got - r2.robot_min) > 1e-3
        assert abs(got - r2.robot_max) > 1e-3


# ============================================================
# neutral 行程（防单边饱和）
# ============================================================
class TestMappingHeadroom:
    """
    回归背景（真实踩到的坑）：
        joint2 限位 [0, 3.14]，若 neutral 取 0.9，
        配合某些 invert 设置，人一抬臂就撞下限被裁剪，
        表现为「机械臂抬到某个角度就不动了」。
        这是配置问题而非代码缺陷，但不加约束会反复踩。
    """

    def test_neutral_leaves_headroom(self, cfg, robot_cfg):
        for r in cfg.enabled_rules():
            n = cfg.neutral_for(r.joint)
            lo, hi = r.joint_lower, r.joint_upper
            room = min(abs(r.robot_min - n), abs(r.robot_max - n),
                       n - lo, hi - n)
            assert room > 0.1, (
                f"{r.joint}: neutral={n} 行程不足 "
                f"(限位[{lo:+.2f},{hi:+.2f}] 映射[{r.robot_min:+.2f},{r.robot_max:+.2f}])")

    def test_report_headroom(self, cfg, robot_cfg):
        """打印各关节行程，便于调 neutral（永远通过）"""
        print("\n  各受控关节行程:")
        for r in cfg.enabled_rules():
            n = cfg.neutral_for(r.joint)
            print(f"    {r.joint:8s} neutral={n:+.3f} "
                  f"限位[{r.joint_lower:+.2f},{r.joint_upper:+.2f}] "
                  f"映射[{r.robot_min:+.2f},{r.robot_max:+.2f}]")


# ============================================================
# 逐关节开关（用于 J3-only / J2-only 等验证阶段）
# ============================================================
class TestPerJointEnable:
    def _build(self, cfg, robot_cfg, only: str):
        for r in cfg.rules:
            r.enabled = (r.joint == only)
        rt = Retargeter(cfg, robot_cfg)
        rt.start_calibration()
        for _ in range(20):
            rt.add_calibration_sample(ArmGeometry(0, 0, 90, 100, 100, True))
        rt.finish_calibration()
        rt.set_initial_pose(cfg.neutral_joints)
        return rt

    def test_j3_only_moves_only_j3(self, cfg, robot_cfg):
        rt = self._build(cfg, robot_cfg, "joint3")
        base = list(rt.target)
        for k in range(150):
            res = rt.update(make_arm(e=(100, 0), w=(20, 20)), now=k * 0.033)
        i2 = cfg.joint_names.index("joint2")
        i3 = cfg.joint_names.index("joint3")
        i5 = cfg.joint_names.index("joint5")
        assert res.positions[i2] == pytest.approx(base[i2])
        assert res.positions[i5] == pytest.approx(base[i5])
        assert abs(res.positions[i3] - base[i3]) > 0.15

    def test_controlled_joints_reflects_enable(self, cfg, robot_cfg):
        # 注意：这里只断言「未启用的关节不应出现在受控集合里」。
        # 不再写死完整集合 —— 配置新增了腕部关节 joint5/joint6
        # （由 hand_pose 驱动），写死集合会让每次加规则都改测试。
        assert "joint4" not in cfg.controlled_joints()
        assert "joint1" not in cfg.controlled_joints()
        assert {"joint2", "joint3"} <= set(cfg.controlled_joints())
        for r in cfg.rules:
            r.enabled = (r.joint == "joint2")
        assert cfg.controlled_joints() == ["joint2"]

    def test_disabled_joint_stays_at_neutral(self, cfg, robot_cfg):
        rt = self._build(cfg, robot_cfg, "joint2")
        base = list(rt.target)
        for k in range(100):
            rt.update(make_arm(e=(100, 0), w=(20, 20)), now=k * 0.033)
        i3 = cfg.joint_names.index("joint3")
        assert rt.target[i3] == pytest.approx(base[i3])


# ============================================================
# 调试快照
# ============================================================
class TestDebugSnapshot:
    def test_snapshot_has_raw_and_filtered(self, cfg, robot_cfg):
        rt = calibrated(cfg, robot_cfg)
        for k in range(60):
            res = rt.update(make_arm(e=(100, 0), w=(80, 60)),
                            openness=0.6, now=k * 0.033, frame=k)
        d = res.debug
        assert d is not None
        assert d.state == str(TrackState.TRACKING)
        assert d.openness == pytest.approx(0.6)

    def test_csv_header_and_row_match(self, cfg, robot_cfg):
        rt = calibrated(cfg, robot_cfg)
        for k in range(30):
            res = rt.update(make_arm(e=(100, 0), w=(80, 60)), now=k * 0.033)
        header = res.debug.csv_header()
        row = res.debug.to_csv_row()
        for col in header:
            assert col in row, f"CSV 表头 {col} 在行里缺失"

    def test_csv_contains_required_columns(self, cfg, robot_cfg):
        """需求文档明确要求的列必须在"""
        rt = calibrated(cfg, robot_cfg)
        res = rt.update(make_arm(), now=0.0)
        header = set(res.debug.csv_header())
        for col in ("timestamp", "shoulder_x", "shoulder_y", "elbow_x",
                    "elbow_y", "wrist_x", "wrist_y",
                    "raw_upper", "filtered_upper",
                    "raw_elbow", "filtered_elbow",
                    "joint2", "joint3", "joint5",
                    "mapped_joint2", "mapped_joint3", "mapped_joint5",
                    "openness", "joint7"):
            assert col in header, f"缺少列 {col}"

    def test_snapshot_records_keypoints(self, cfg, robot_cfg):
        rt = calibrated(cfg, robot_cfg)
        res = rt.update(make_arm(s=(11, 22), e=(33, 44), w=(55, 66)), now=0.0)
        assert res.debug.shoulder_xy == pytest.approx((11, 22))
        assert res.debug.elbow_xy == pytest.approx((33, 44))
        assert res.debug.wrist_xy == pytest.approx((55, 66))

    # --------------------------------------------------------
    # CSV 数值列「恒为 0」是一类反复出现的缺陷，不是单点 bug：
    #   * filtered_elbow / filtered_forearm 曾恒为 0（只填了 upper）
    #   * delta_* 三列曾恒为 0（deltas_deg 算好了却没转写进快照）
    #   * joint7 曾恒为 0（夹爪在 demo 层，快照没被回填）
    # 共同根因：量已经算出来了，却没有从 RetargetResult 同步到
    # DebugSnapshot。下面用「凡是动过的量，CSV 里就不能是常数 0」
    # 一次性兜住这一整类问题。
    # --------------------------------------------------------
    def test_csv_columns_are_not_silently_zero(self, cfg, robot_cfg):
        """
        让手臂做一段明显运动，每个「本应随动作变化」的列都必须出现
        非零值。恒为 0 说明该列没被填充，而不是"动作太小"。
        """
        rt = calibrated(cfg, robot_cfg)
        rows = []
        for k in range(120):
            t = k / 119.0
            e = (100, -int(60 * t))
            w = (int(90 - 60 * t), int(30 + 90 * t))
            res = rt.update(make_arm(e=e, w=w), openness=0.3 + 0.6 * t,
                            now=k * 0.033, frame=k)
            if res.debug is not None:
                rows.append(res.debug.to_csv_row())

        assert len(rows) > 50, "样本太少，无法判断"

        # 只要求**当前配置真正在用**的量对应列非零。
        # 不写死列名：配置加一条规则就要改测试，且会误报
        #（例如前臂角已不再驱动任何关节，它的列恒 0 是正常的，
        #  但 raw 列仍应反映观测）。
        used = set(cfg.used_measures())
        raw_of = {
            "upper_arm_angle_deg": "raw_upper",
            "elbow_angle_deg": "raw_elbow",
            "forearm_angle_deg": "raw_forearm",
        }
        flt_of = {
            "upper_arm_angle_deg": "filtered_upper",
            "elbow_angle_deg": "filtered_elbow",
            "forearm_angle_deg": "filtered_forearm",
        }
        delta_of = {
            "upper_arm_angle_deg": "delta_upper",
            "elbow_angle_deg": "delta_elbow",
            "forearm_angle_deg": "delta_forearm",
        }
        must_vary = ["openness"]
        for m in used:
            if m in raw_of:
                must_vary += [raw_of[m], flt_of[m], delta_of[m]]
        for j in cfg.controlled_joints():
            must_vary += [j, f"mapped_{j}"]
        # joint7 由 demo 层回填，Retargeter 不负责，故不在此断言

        for col in must_vary:
            vals = [float(r[col]) for r in rows]
            assert any(abs(v) > 1e-9 for v in vals), \
                f"CSV 列 {col} 恒为 0 —— 该列没有被填充 (在用测量量: {sorted(used)})"

    def test_delta_columns_match_result_deltas(self, cfg, robot_cfg):
        """delta_* 必须等于 RetargetResult.debug_deltas_deg（算的=记的）"""
        rt = calibrated(cfg, robot_cfg)
        checked = 0
        for k in range(60):
            res = rt.update(make_arm(e=(100, -20), w=(60, 60)),
                            now=k * 0.033, frame=k)
            if not res.debug_deltas_deg or res.debug is None:
                continue
            # 只断言**配置在用**的量。前臂角已不再驱动任何关节
            # （j5 改由手腕俯仰驱动），它的 delta 不在快照里是正确的。
            col_of = {"upper_arm_angle_deg": "delta_upper",
                      "elbow_angle_deg": "delta_elbow",
                      "forearm_angle_deg": "delta_forearm"}
            for m, col in col_of.items():
                if m not in res.debug_deltas_deg:
                    continue
                assert getattr(res.debug, col) == pytest.approx(
                    res.debug_deltas_deg[m], abs=1e-6), m
            checked += 1
        assert checked > 10, "没有取到足够的 TRACKING 样本"

    def test_filtered_columns_match_angle_filter(self, cfg, robot_cfg):
        """filtered_* 必须等于 AngleFilter.value()，不能是编出来的数"""
        rt = calibrated(cfg, robot_cfg)
        checked = 0
        for k in range(60):
            res = rt.update(make_arm(e=(100, -20), w=(60, 60)),
                            now=k * 0.033, frame=k)
            if res.debug is None:
                continue
            flt = rt.angle_filter.value()
            if not flt:
                continue
            col_of = {"upper_arm_angle_deg": "flt_upper",
                      "elbow_angle_deg": "flt_elbow",
                      "forearm_angle_deg": "flt_forearm"}
            for m, col in col_of.items():
                if m not in flt:
                    continue
                assert getattr(res.debug, col) == pytest.approx(
                    flt[m], abs=1e-6), m
            checked += 1
        assert checked > 10


# ============================================================
# 角度滤波是否真的作用于下发值
# ============================================================
class TestFilteredAngleSource:
    """
    映射阶段的输入源（raw / filtered）必须由配置显式控制。

    缺陷背景：map_all 原本吃的是**未滤波**的 geo_dict，
    于是角度滤波器只被写进 CSV、对下发值毫无影响 ——
    三级滤波退化成两级，且从 CSV 上看不出来（两列都在动）。
    实测噪声（相邻帧抖动中位数）：上臂 4.81° vs 滤波后 0.93°。
    """

    # 测试用基准姿势：上臂近水平、前臂朝下偏右。
    # 选这个姿势是因为要让三个受控量在标定后都落在各自区间[±60°,±65°]的
    # **中段**。曾经随手用 (0,0)->(100,0)->(100,100)，结果该几何的前臂角是
    # -90°，标定基准却是 forearm=0，于是 delta_forearm 恒为 -90°、
    # 直接饱和在区间下界，joint5 被钉死在上限 —— 看起来像"滤波没效果"，
    # 实际是测试姿势本身就饱和了。
    BASE_E = (100, 20)
    BASE_W = (110, 30)

    @classmethod
    def _neutral_geo(cls):
        """用真实几何函数算出基准姿势的几何量，保证与流水线一致"""
        g = compute_arm_geometry(
            make_arm(s=(0, 0), e=cls.BASE_E, w=cls.BASE_W),
            min_confidence=0.0, require_complete=False)
        assert g.valid, f"测试基准姿势几何无效: {g.reason}"
        return g

    @classmethod
    def _make_rt(cls, cfg, robot_cfg, use_filtered: bool):
        cfg.use_filtered_angles = use_filtered
        rt = Retargeter(cfg, robot_cfg)
        base = cls._neutral_geo()
        rt.start_calibration()
        for _ in range(20):
            rt.add_calibration_sample(base)
        assert rt.finish_calibration()
        rt.set_initial_pose(cfg.neutral_joints)
        return rt

    @staticmethod
    def _jerk(v):
        """相邻帧变化量的中位数 —— 抖动指标"""
        d = sorted(abs(a - b) for a, b in zip(v[1:], v[:-1]))
        return d[len(d) // 2] if d else 0.0

    @classmethod
    def _noisy_run(cls, cfg, robot_cfg, use_filtered: bool, joint="joint3"):
        """
        在基准姿势上叠加关键点噪声，返回该关节的指令序列。

        噪声轴不能拍脑袋选 —— 该几何下各轴激励能力实测差别极大：
            腕 y 抖动: 上臂 0.00°  肘 0.00°  前臂  0.00°（只改长度不改方向）
            腕 x 抖动: 上臂 0.00°  肘 33.4°  前臂 33.4°
            肘 y 抖动: 上臂 33.4°  肘 33.4°  前臂  0.00°
        所以驱动 joint3/joint5 用腕 x 抖动，驱动 joint2 用肘 y 抖动。
        """
        rt = cls._make_rt(cfg, robot_cfg, use_filtered)
        be, bw = cls.BASE_E, cls.BASE_W
        noise = [0, 25, -25, 18, -18, 25, -8, 8, -25, 18]
        targets = []
        for k in range(160):
            n = noise[k % len(noise)]
            if joint == "joint2":
                e = (be[0], be[1] + n)          # 上臂角随肘 y 变化
                w = (cls.BASE_W[0], cls.BASE_W[1])
            else:
                # joint3 由肘夹角驱动 -> 抖腕 x 改肘夹角
                e = be
                w = (cls.BASE_W[0] + n, cls.BASE_W[1])
            res = rt.update(make_arm(e=e, w=w), now=k * 0.033, frame=k)
            targets.append(res.positions[cfg.joint_names.index(joint)])
        return targets[80:]          # 丢掉 EMA 收敛段

    def test_flag_off_keeps_historical_behaviour(self, cfg, robot_cfg):
        """
        显式设为 False 时必须复现历史行为（映射吃原始角）。
        历史行为要保留，否则出问题时无法回退对比。
        """
        cfg.use_filtered_angles = False
        assert cfg.use_filtered_angles is False
        # 关掉后，纯噪声输入不应被角度滤波吸收
        raw_t = self._noisy_run(cfg, robot_cfg, use_filtered=False)
        assert self._jerk(raw_t) > 0, "关掉后应仍可见原始抖动"

    def test_flag_on_reduces_joint_jitter(self, cfg, robot_cfg):
        """
        开启后，同样的关键点噪声必须产生**更平滑**的关节指令。
        这正是「角度滤波要作用于下发值」的可观测证据。
        """
        raw_t = self._noisy_run(cfg, robot_cfg, use_filtered=False)
        flt_t = self._noisy_run(cfg, robot_cfg, use_filtered=True)

        j_raw, j_flt = self._jerk(raw_t), self._jerk(flt_t)
        assert j_raw > 0, "测试自身无效：raw 路径没有产生任何抖动"
        assert j_flt < j_raw, (
            f"开启 use_filtered_angles 后抖动没有下降: "
            f"raw={j_raw:.5f} filtered={j_flt:.5f}")

    def test_raw_jitter_is_materially_larger(self, cfg, robot_cfg):
        """
        不只是「更小」，而要有量级差异 ——
        否则说明噪声本来就很小，这个开关无关紧要（结论就不成立）。
        """
        raw_t = self._noisy_run(cfg, robot_cfg, use_filtered=False)
        flt_t = self._noisy_run(cfg, robot_cfg, use_filtered=True)
        j_raw, j_flt = self._jerk(raw_t), self._jerk(flt_t)
        assert j_flt < 0.5 * j_raw, (
            f"滤波收益不足 50%: raw={j_raw:.5f} -> filtered={j_flt:.5f}")

    def test_filter_hurts_nothing_when_angles_are_clean(self, cfg, robot_cfg):
        """
        反向保障：输入本身干净时，开启滤波不应把运动也抹掉
        （若 alpha 太小会把真实动作也滤没，那就是过度滤波）。
        """
        def run(use_filtered):
            cfg.use_filtered_angles = use_filtered
            rt = calibrated(cfg, robot_cfg)
            out = []
            for k in range(200):
                t = k / 199.0
                arm = make_arm(e=(100, -int(70 * t)),
                               w=(int(100 + 80 * t), 100))
                res = rt.update(arm, now=k * 0.033, frame=k)
                out.append(res.positions[cfg.joint_names.index("joint5")])
            return out[-1]

        cfg.use_filtered_angles = False
        v_raw = run(False)
        v_flt = run(True)
        # 慢速大范围运动，两者都应到达相近的终值
        assert abs(v_flt - v_raw) < 0.15, (
            f"滤波把真实运动也削弱了: raw={v_raw:.4f} filtered={v_flt:.4f}")

    def test_flag_read_from_yaml(self, tmp_path):
        """
        配置项必须能真正从 YAML 读出来（不能只存在于 dataclass）。
        同时验证 mapping 段里的开关项不会被当成映射规则。
        """
        import yaml
        base = yaml.safe_load(open(
            os.path.join(_PKG_ROOT, "config", "retargeting.yaml"),
            encoding="utf-8"))
        base.setdefault("mapping", {})["use_filtered_angles"] = False
        p = tmp_path / "rt.yaml"
        with open(p, "w", encoding="utf-8") as fh:
            yaml.safe_dump(base, fh, allow_unicode=True)
        c = RetargetingConfig.from_yaml(str(p))
        assert c.use_filtered_angles is False
        # 三关节规则仍应完整解析出来（开关项不能被当成规则）
        assert {r.joint for r in c.rules} >= {"joint2", "joint3", "joint5"}
        assert len(c.rules) >= 3, f"开关项被误当成映射规则: {len(c.rules)} 条"

    def test_side_flag_is_wired_to_config(self):
        """
        --side 必须存在且接到 cfg.human.side。

        这是实机踩出来的需求：摄像头只拍到一侧手臂时，
        另一侧肘部会一直在画面外（elbow confidence=0），
        几何永远无效 —— 没有这个开关就只能改 YAML 才能换边。
        """
        src = open(os.path.join(
            _PKG_ROOT, "piper_human_retargeting", "retarget_demo.py"),
            encoding="utf-8").read()
        assert '"--side"' in src or "'--side'" in src, "CLI 里没有 --side 参数"
        assert "self.cfg.human.side = args.side" in src, \
            "--side 没有接到 cfg.human.side"

    def test_repo_config_enables_filtered_angles(self, cfg):
        """
        仓库自带配置默认开启（有实机 A/B 证据支撑）。
        若有人把它改回 false，这个测试会提醒他先看 YAML 里的实测数据。
        """
        assert cfg.use_filtered_angles is True

    def test_cli_flag_is_wired_to_config(self):
        """
        --filtered-angles 必须真的接到 cfg.use_filtered_angles 上。

        这里不 import retarget_demo（它需要 rclpy），改为直接读源码，
        断言「参数存在」且「参数被赋给配置」两件事都在。
        纯字符串检查虽然笨，但能挡住"加了 CLI 参数却忘了接线"这类漏洞。
        """
        src = open(os.path.join(
            _PKG_ROOT, "piper_human_retargeting", "retarget_demo.py"),
            encoding="utf-8").read()
        assert '"--filtered-angles"' in src or "'--filtered-angles'" in src, \
            "CLI 里没有 --filtered-angles 参数"
        assert "self.cfg.use_filtered_angles = True" in src, \
            "--filtered-angles 没有接到 cfg.use_filtered_angles"


# ============================================================
# 结果字段
# ============================================================
class TestResultFields:
    def test_mapped_joints_vs_final(self, cfg, robot_cfg):
        """
        mapped_joints 是滤波前、positions 是滤波后。
        两者都要能取到，用于区分抖动来源。
        """
        rt = calibrated(cfg, robot_cfg)
        res = rt.update(make_arm(e=(100, 0), w=(20, 20)), now=0.0)
        assert res.mapped_joints
        assert res.positions
        for j in cfg.controlled_joints():
            assert j in res.mapped_joints

    def test_deltas_reported(self, cfg, robot_cfg):
        rt = calibrated(cfg, robot_cfg)
        for k in range(80):
            res = rt.update(make_arm(e=(100, 0), w=(20, 20)), now=k * 0.033)
        assert set(res.deltas.keys()) == set(cfg.controlled_joints())

    def test_stats_available(self, cfg, robot_cfg):
        rt = calibrated(cfg, robot_cfg)
        for k in range(10):
            rt.update(make_arm(), now=k * 0.033)
        s = rt.stats
        for k in ("updates", "skips", "tracking", "lost_short",
                  "lost_long", "transitions"):
            assert k in s
