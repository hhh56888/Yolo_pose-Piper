# -*- coding: utf-8 -*-
"""
FK 运动学测试（kinematics.py）
==============================
关键点：本测试**不是**用 URDF 自证，而是用**仿真 Gazebo TF 实测**的位姿点核对：

    数据来源：docs/J3_Elbow_Mapping_Audit.md §5 的单关节实测
    （joint2=0.6 固定，只改 joint3，读 base_link -> gripper_base 的 TF）

运行：
    python3 -m pytest test/test_kinematics.py -q -p no:anyio
"""
import math
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_HERE)
_SRC = os.path.dirname(_PKG_ROOT)
for _p in (_PKG_ROOT, os.path.join(_SRC, "piper_human_control")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from piper_human_retargeting.kinematics import (          # noqa: E402
    ChainModel, default_chain_path, load_default_chain, rpy_to_matrix)


# ---- 仿真 TF 实测锚点：(q3, TCP x, y, z)，joint2=0.6，其余为 0 ----
TF_ANCHORS = [
    (-0.5985, 0.1272, 0.0000, 0.3655),
    (-0.7485, 0.1157, 0.0000, 0.4156),
    (-0.4485, 0.1311, 0.0000, 0.3144),
    (-0.8984, 0.0969, 0.0000, 0.4633),
    (-0.2986, 0.1273, 0.0000, 0.2632),
    (-1.1984, 0.0390, 0.0000, 0.5477),
    (-1.4983, -0.0412, 0.0000, 0.6113),
    (-1.7983, -0.1366, 0.0000, 0.6483),
    (-2.0985, -0.2388, 0.0000, 0.6555),
]


@pytest.fixture(scope="module")
def model():
    return load_default_chain()


class TestChainModel:
    def test_loads_with_auditable_source(self, model):
        """链路文件必须记录来源（哪份 URDF、md5），否则无法审计"""
        assert model.source.get("urdf_md5")
        assert model.source.get("root_link") == "base_link"
        assert model.source.get("tip_link") == "gripper_base"

    def test_dof_and_limits(self, model):
        assert model.dof_names == ["joint1", "joint2", "joint3",
                                   "joint4", "joint5", "joint6"]
        lim = model.limits()
        assert lim["joint2"] == (0.0, 3.14)
        assert lim["joint3"] == (-2.967, 0.0)

    def test_urdf_limits_match_control_config(self):
        """三层限位的第一层：URDF 与控制侧 joint_limits.yaml 必须一致"""
        from piper_human_retargeting.limits import physical_limits
        phys, src = physical_limits()
        assert "一致" in src or "不可用" in src
        assert phys["joint3"] == (-2.967, 0.0)

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            ChainModel.load("/tmp/definitely_not_here_12345.json")


class TestForwardKinematics:
    @pytest.mark.parametrize("q3,x,y,z", TF_ANCHORS)
    def test_matches_gazebo_tf_measurements(self, model, q3, x, y, z):
        p = model.tcp({"joint1": 0.0, "joint2": 0.6, "joint3": q3,
                       "joint4": 0.0, "joint5": 0.0, "joint6": 0.0})
        err = math.dist(p, (x, y, z))
        assert err < 2e-3, f"q3={q3}: FK {p} vs Gazebo ({x},{y},{z}) 差 {err * 1000:.2f} mm"

    def test_max_error_over_all_anchors(self, model):
        worst = 0.0
        for q3, x, y, z in TF_ANCHORS:
            p = model.tcp({"joint1": 0.0, "joint2": 0.6, "joint3": q3,
                           "joint4": 0.0, "joint5": 0.0, "joint6": 0.0})
            worst = max(worst, math.dist(p, (x, y, z)))
        assert worst < 1e-3, f"最大误差 {worst * 1000:.3f} mm"

    def test_fk_is_continuous(self, model):
        """相邻 q 的 TCP 不能跳变（数值 IK 的前提）"""
        prev = None
        for q3 in np.linspace(-2.9, -0.05, 200):
            p = model.tcp({"joint1": 0.0, "joint2": 1.0, "joint3": float(q3),
                           "joint4": 0.0, "joint5": 0.0, "joint6": 0.0})
            if prev is not None:
                assert math.dist(p, prev) < 0.02
            prev = p

    def test_joint3_increasing_shortens_reach(self, model):
        """
        用 FK 复核审计结论：q3 增大 = 收臂（TCP 离肩更近、更低）。
        这条断言是「同向映射」判定的量化依据之一。
        """
        def tcp(q3):
            return model.tcp({"joint1": 0.0, "joint2": 1.0, "joint3": q3,
                              "joint4": 0.0, "joint5": 0.0, "joint6": 0.0})
        shoulder = np.array([0.0, 0.0, 0.123])
        r_lo = np.linalg.norm(tcp(-2.0) - shoulder)
        r_hi = np.linalg.norm(tcp(-0.2) - shoulder)
        assert r_hi < r_lo
        assert tcp(-0.2)[2] < tcp(-2.0)[2]


class TestJacobian:
    @pytest.mark.parametrize("q2,q3", [(0.6, -0.6), (1.2, -1.2), (2.0, -2.0)])
    def test_matches_finite_difference(self, model, q2, q3):
        q = {"joint1": 0.0, "joint2": q2, "joint3": q3,
             "joint4": 0.0, "joint5": 0.0, "joint6": 0.0}
        J = model.jacobian(q, ["joint2", "joint3"])
        h = 1e-7
        cols = []
        for n in ("joint2", "joint3"):
            qp, qm = dict(q), dict(q)
            qp[n] += h
            qm[n] -= h
            cols.append((model.tcp(qp) - model.tcp(qm)) / (2 * h))
        FD = np.column_stack(cols)
        assert np.abs(J - FD).max() < 1e-6

    def test_xz_slice_and_singular_values(self, model):
        q = {"joint1": 0.0, "joint2": 1.5, "joint3": -1.0,
             "joint4": 0.0, "joint5": 0.0, "joint6": 0.0}
        J = model.jacobian_xz(q, ["joint2", "joint3"])
        assert J.shape == (2, 2)
        smin = model.sigma_min_xz(q, ["joint2", "joint3"])
        assert smin > 0.05                     # neutral 附近不该接近奇异
        assert model.manipulability_xz(q, ["joint2", "joint3"]) > 0

    def test_unknown_joint_raises(self, model):
        with pytest.raises(KeyError):
            model.jacobian({"joint2": 0.5}, ["joint9"])


class TestRpy:
    def test_identity(self):
        assert np.allclose(rpy_to_matrix((0, 0, 0)), np.eye(3))

    def test_rotation_order_is_zyx(self):
        """URDF 固定轴 RPY = Rz(y)Ry(p)Rx(r)；这里用 90° 单轴逐个核对"""
        rz = rpy_to_matrix((0, 0, math.pi / 2))
        assert np.allclose(rz @ np.array([1, 0, 0]), [0, 1, 0], atol=1e-9)
        rx = rpy_to_matrix((math.pi / 2, 0, 0))
        assert np.allclose(rx @ np.array([0, 1, 0]), [0, 0, 1], atol=1e-9)
