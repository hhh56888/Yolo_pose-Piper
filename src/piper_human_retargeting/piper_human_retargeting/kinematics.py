# -*- coding: utf-8 -*-
"""
piper_human_retargeting.kinematics
==================================
Piper 运动学（FK / Jacobian）—— 由**实际加载的** URDF 生成的数据驱动。

数据来源（可审计）：
    `data/fk_chain.json` 由 `tools/build_fk_chain.py` 从
    `ros2 param get /robot_state_publisher robot_description` 得到的 URDF 生成，
    文件里记录了 URDF 的 md5 与 root/tip link 名。
    URDF 出自 piper_description_gazebo.xacro（piper_gazebo.launch.py 用 xacro 解析）。

为什么要自己写 FK：
    - 数值 IK 需要**快**（30 Hz 主循环）且需要 **Jacobian**，
      网格查表只能给到网格精度，且无法求导；
    - 需要严格把关节限位放进优化迭代（见 task_space.TaskSpaceIk）；
    - 运行时零额外依赖（ROS 环境只有 numpy）。

坐标约定：
    FK 结果在 **root_link（base_link）** 坐标系下，与 Gazebo/ TF 一致：
        +x 前方、+y 左、+z 上。
    tip 默认 `gripper_base`（与 link6 原点重合，joint6_to_gripper_base 平移为 0）。

正确性保证：
    `test/test_kinematics.py` 用**仿真 TF 实测**的 9 个位姿点核对本模块
    （不是与 URDF 自证）。
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# 参与 FK 的关节类型（fixed 关节只贡献常量变换）
_MOVABLE = ("revolute", "continuous", "prismatic")


def rpy_to_matrix(rpy: Sequence[float]) -> np.ndarray:
    """URDF 固定轴 RPY：R = Rz(yaw) @ Ry(pitch) @ Rx(roll)"""
    r, p, y = (float(v) for v in rpy)
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return rz @ ry @ rx


def axis_angle_to_matrix(axis: Sequence[float], angle: float) -> np.ndarray:
    """Rodrigues 公式（轴已在关节坐标系内，无需归一化前提）"""
    a = np.asarray(axis, dtype=float)
    n = float(np.linalg.norm(a))
    if n < 1e-12:
        return np.eye(3)
    a = a / n
    K = np.array([[0.0, -a[2], a[1]],
                  [a[2], 0.0, -a[0]],
                  [-a[1], a[0], 0.0]], dtype=float)
    return np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


def _T(R: np.ndarray, t: Sequence[float]) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = R
    out[:3, 3] = np.asarray(t, dtype=float)
    return out


@dataclass(frozen=True)
class JointSpec:
    name: str
    type: str
    parent: str
    child: str
    origin_xyz: Tuple[float, float, float]
    origin_rpy: Tuple[float, float, float]
    axis: Tuple[float, float, float]
    lower: Optional[float] = None
    upper: Optional[float] = None
    velocity: Optional[float] = None
    effort: Optional[float] = None

    @property
    def movable(self) -> bool:
        return self.type in _MOVABLE


class ChainModel:
    """URDF 链路 FK 模型（只含 root→tip 之间的关节）"""

    def __init__(self, chain: List[JointSpec], root_link: str, tip_link: str,
                 source: Optional[dict] = None):
        self.chain = list(chain)
        self.root_link = root_link
        self.tip_link = tip_link
        self.source = dict(source or {})
        self.dof_names = [j.name for j in self.chain if j.movable]
        # 预计算每个关节的常量部分：R_origin、t_origin、axis、是否可动
        self._pre = []
        for j in self.chain:
            self._pre.append((rpy_to_matrix(j.origin_rpy),
                              np.asarray(j.origin_xyz, dtype=float),
                              np.asarray(j.axis, dtype=float), j.movable))

    # ------------------------------------------------------------
    @classmethod
    def load(cls, path: str) -> "ChainModel":
        if not os.path.isfile(path):
            raise FileNotFoundError(f"FK 链路文件不存在: {path}")
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        chain = []
        for d in raw["chain"]:
            lim = d.get("limit") or {}
            chain.append(JointSpec(
                name=d["name"], type=d["type"], parent=d["parent"],
                child=d["child"],
                origin_xyz=tuple(d["origin_xyz"]), origin_rpy=tuple(d["origin_rpy"]),
                axis=tuple(d["axis"]),
                lower=lim.get("lower"), upper=lim.get("upper"),
                velocity=lim.get("velocity"), effort=lim.get("effort")))
        src = raw.get("source", {})
        return cls(chain, src.get("root_link", "base_link"),
                   src.get("tip_link", "gripper_base"), src)

    # ------------------------------------------------------------
    def limits(self) -> Dict[str, Tuple[float, float]]:
        out = {}
        for j in self.chain:
            if not j.movable:
                continue
            if j.lower is None or j.upper is None:
                raise ValueError(f"关节 {j.name} 在 URDF 里没有位置限位")
            out[j.name] = (float(j.lower), float(j.upper))
        return out

    def fk_frames(self, q: Mapping[str, float]) -> Dict[str, np.ndarray]:
        """返回链路上每个 link 的位姿（root 坐标系）"""
        T = np.eye(4)
        out = {self.root_link: T.copy()}
        for j, (R0, t0, axis, movable) in zip(self.chain, self._pre):
            T = T @ _T(R0, t0)
            if movable:
                ang = float(q.get(j.name, 0.0))
                if j.type == "prismatic":
                    T = T @ _T(np.eye(3), axis * ang)
                else:
                    T = T @ _T(axis_angle_to_matrix(axis, ang), (0.0, 0.0, 0.0))
            out[j.child] = T.copy()
        return out

    def fk(self, q: Mapping[str, float]) -> np.ndarray:
        return self.fk_frames(q)[self.tip_link]

    def tcp(self, q: Mapping[str, float]) -> np.ndarray:
        return self.fk(q)[:3, 3].copy()

    # ------------------------------------------------------------
    def jacobian(self, q: Mapping[str, float],
                 names: Optional[Iterable[str]] = None) -> np.ndarray:
        """
        位置 Jacobian（3 x N），列与 names 对应。

        旋转关节：dp/dq = axis_world × (p_tip - p_joint)
        （p_joint = 该关节**旋转后**的原点，即子 link 的原点）
        """
        names = list(names) if names is not None else list(self.dof_names)
        frames = self.fk_frames(q)
        p_tip = frames[self.tip_link][:3, 3]
        by_name = {j.name: (j, pre) for j, pre in zip(self.chain, self._pre)}
        cols = []
        for name in names:
            if name not in by_name:
                raise KeyError(f"链路里没有关节 {name}")
            j, (R0, t0, axis, movable) = by_name[name]
            if not movable:
                cols.append(np.zeros(3))
                continue
            T_joint = frames[j.parent] @ _T(R0, t0)   # 关节坐标系（q 不影响轴）
            if j.type == "prismatic":
                cols.append(T_joint[:3, :3] @ axis)
            else:
                a_world = T_joint[:3, :3] @ axis
                p_joint = T_joint[:3, 3]
                cols.append(np.cross(a_world, p_tip - p_joint))
        return np.column_stack(cols) if cols else np.zeros((3, 0))

    # ---- XZ 平面（2D task-space 用）----
    def jacobian_xz(self, q: Mapping[str, float],
                    names: Sequence[str]) -> np.ndarray:
        J = self.jacobian(q, names)
        return J[[0, 2], :]

    def sigma_min_xz(self, q: Mapping[str, float],
                     names: Sequence[str]) -> float:
        """XZ 平面 2x2 Jacobian 的最小奇异值（可操作度指标）"""
        J = self.jacobian_xz(q, names)
        if J.shape[1] < 2:
            return 0.0
        s = np.linalg.svd(J, compute_uv=False)
        return float(s[-1])

    def manipulability_xz(self, q: Mapping[str, float],
                          names: Sequence[str]) -> float:
        J = self.jacobian_xz(q, names)
        return float(abs(np.linalg.det(J))) if J.shape[1] == 2 else 0.0


def _R0_of(j: JointSpec) -> np.ndarray:      # 供 jacobian 复用
    return rpy_to_matrix(j.origin_rpy)


class TwoDofFk:
    """
    (joint_a, joint_b) 双关节解算专用快速 FK / Jacobian —— 其余关节固定。

    为什么需要：
        数值 IK 每帧要算几十次 FK，通用 `ChainModel.fk()` 每次都要重建
        7 个 4x4 变换（~50 µs），在 30 Hz 控制环里会吃掉可观的时间
        （实测 P95 2.1 ms）。把链路按"关节 a 之前 / a 与 b 之间 / b 之后"
        切成三块常量矩阵后，每次 FK 只剩 2 次旋转 + 3 次矩阵乘（~5 µs），
        结果与通用 FK **逐位一致**（test_kinematics.py 有对照测试）。

    只支持 2 个**转动**关节，且 a 在链路上位于 b 之前。
    """

    def __init__(self, model: ChainModel, joint_a: str, joint_b: str,
                 fixed: Optional[Dict[str, float]] = None):
        self.model = model
        self.ja, self.jb = joint_a, joint_b
        fixed = dict(fixed or {})
        names = [j.name for j in model.chain]
        ia, ib = names.index(joint_a), names.index(joint_b)
        if ia >= ib:
            raise ValueError("joint_a 必须在链路上位于 joint_b 之前")

        def seg(lo, hi):
            """[lo, hi) 区间内的常量变换（可动关节取 fixed 值）"""
            T = np.eye(4)
            for j, (R0, t0, axis, movable) in zip(model.chain[lo:hi],
                                                  model._pre[lo:hi]):
                T = T @ _T(R0, t0)
                if movable:
                    v = float(fixed.get(j.name, 0.0))
                    if j.type == "prismatic":
                        T = T @ _T(np.eye(3), np.asarray(axis) * v)
                    else:
                        T = T @ _T(axis_angle_to_matrix(axis, v),
                                   (0.0, 0.0, 0.0))
            return T

        self.T_pre = seg(0, ia) @ _T(*model._pre[ia][:2])      # -> joint_a 坐标系
        self.T_mid = seg(ia + 1, ib) @ _T(*model._pre[ib][:2])  # a 子链 -> joint_b 坐标系
        self.T_post = seg(ib + 1, len(model.chain))             # b 子链 -> tip
        self.axis_a = np.asarray(model._pre[ia][2], dtype=float)
        self.axis_b = np.asarray(model._pre[ib][2], dtype=float)
        if model.chain[ia].type != "revolute" or model.chain[ib].type != "revolute":
            raise ValueError("TwoDofFk 只支持两个转动关节")

    # ------------------------------------------------------------
    def frames(self, qa: float, qb: float):
        """返回 (T2, T3, T_tip)：joint_a 坐标系 / joint_b 坐标系 / 末端位姿"""
        T2 = self.T_pre @ _T(axis_angle_to_matrix(self.axis_a, qa),
                             (0.0, 0.0, 0.0))
        T3 = T2 @ self.T_mid @ _T(axis_angle_to_matrix(self.axis_b, qb),
                                  (0.0, 0.0, 0.0))
        return T2, T3, T3 @ self.T_post

    def fk(self, qa: float, qb: float) -> np.ndarray:
        return self.frames(qa, qb)[2]

    def pos(self, qa: float, qb: float) -> Tuple[float, float]:
        p = self.fk(qa, qb)[:3, 3]
        return float(p[0]), float(p[2])

    def jac_xz(self, qa: float, qb: float) -> np.ndarray:
        """2x2：d(x,z)/d(qa,qb)"""
        T2, T3, T = self.frames(qa, qb)
        p = T[:3, 3]
        pa = T2[:3, 3]
        pb = (T2 @ self.T_mid)[:3, 3]
        aw = T2[:3, :3] @ self.axis_a
        bw = (T2 @ self.T_mid)[:3, :3] @ self.axis_b
        ca = np.cross(aw, p - pa)
        cb = np.cross(bw, p - pb)
        return np.array([[ca[0], cb[0]], [ca[2], cb[2]]], dtype=float)


def default_chain_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "data", "fk_chain.json")


def load_default_chain() -> ChainModel:
    return ChainModel.load(default_chain_path())


def tcp_xz(model: ChainModel, q: Mapping[str, float],
           tip: Optional[str] = None) -> Tuple[float, float]:
    p = model.tcp(q) if tip is None else model.fk_frames(q)[tip][:3, 3]
    return float(p[0]), float(p[2])
