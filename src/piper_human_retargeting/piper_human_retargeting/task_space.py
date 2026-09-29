# -*- coding: utf-8 -*-
"""
piper_human_retargeting.task_space  (V1.1)
==========================================
2D Task-Space Retargeting：**人的手往哪走，机器人 TCP 就往哪走**。

数据流（与 V1.1 任务书一致）：

    Human Shoulder / Elbow / Wrist        （arm_geometry 给出 human_u / human_v）
              ↓
      2D Human Geometry                  u = (W.x-S.x)/L , v = -(W.y-S.y)/L , L=|SE|+|EW|
              ↓
      Wrist Relative Motion              du = u - u0 , dv = v - v0（相对标定姿势）
              ↓
      Robot TCP X/Z Target               x = x0 + kx*du , z = z0 + kz*dv（轴/符号可配置）
              ↓
      J2 + J3 联合求解（数值 IK，约束在迭代内）   TaskSpaceIk
              ↓
      Piper

设计要点（都是踩过坑之后定下来的）：

1. **不再用肘角决定 J3**：肘角与 J2/J3 耦合，逐关节映射无法保证"手去哪、TCP 去哪"。
   task_space 模式下 q2/q3 只由 TCP 目标决定（任务书 §15）；肘角仍记录，
   作为将来的姿态偏好项（§16）。
2. **关节限位进入优化迭代**，而不是"先解再 clamp"（§13）：
   每步都做 box 投影 + 回溯线搜索，迭代全程可行。
3. **上一帧解 warm start + lambda_prev 正则**，避免相邻帧 IK 跳解（§12）。
4. **轴关系配置化**（§8）：不做"相机一定在侧面"的假设。
5. **所有量可观测**（§18）：u/v、du/dv、target、ik 解、求解耗时全部进 debug/CSV；
   缺量时**跳过并报原因**，不静默继续。

旧版（reach/elevation + 网格最近邻 IK）已整体替换：
    * 网格最近邻精度受网格步长限制（0.2×0.25 rad），实测产生大量跳解；
    * reach/elevation 与"手在图像里的位置"不是线性关系，难以直觉调参。
备份见 /tmp/task_space_v1_backup.py（md5 17f869b64a91aa067d5a9c7d5fb6f654）。
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .kinematics import ChainModel, TwoDofFk, load_default_chain

# ============================================================
# ① 人体侧：手腕的 2D 归一化位置
# ============================================================
@dataclass
class HumanWristTask:
    """人体手腕相对肩的归一化位置（尺度无关）"""
    u: float = 0.0          # 水平：+ 指向图像右（正负语义由配置 sign_h 决定）
    v: float = 0.0          # 垂直：**向上为正**（= -(W.y-S.y)/L）
    reach: float = 1.0      # |S-W| / L：伸展程度（诊断量）
    arm_scale: float = 0.0  # L = |S-E| + |E-W|（像素）
    valid: bool = False
    reason: str = ""

    def as_dict(self) -> Dict[str, float]:
        return {"human_u": self.u, "human_v": self.v,
                "reach": self.reach, "arm_scale": self.arm_scale}


def compute_human_wrist_task(shoulder, elbow, wrist,
                             min_seg_px: float = 8.0) -> HumanWristTask:
    """
    由肩/肘/腕三点算 u / v。

    图像坐标约定：+x 右、+y **下**（OpenCV）。因此垂直量取负号，
    使 `v > 0` 表示**手腕在肩上方** —— 即"人手抬高"的物理语义。

    Args:
        shoulder/elbow/wrist: 带 .x/.y/.is_valid 的关键点
        min_seg_px: 上/下臂过短判定（与 arm_geometry 一致）
    """
    if shoulder is None or not getattr(shoulder, "is_valid", False):
        return HumanWristTask(valid=False, reason="缺肩点")
    if elbow is None or not getattr(elbow, "is_valid", False):
        return HumanWristTask(valid=False, reason="缺肘点")
    if wrist is None or not getattr(wrist, "is_valid", False):
        return HumanWristTask(valid=False, reason="缺腕点")

    l1 = math.hypot(elbow.x - shoulder.x, elbow.y - shoulder.y)
    l2 = math.hypot(wrist.x - elbow.x, wrist.y - elbow.y)
    if l1 < min_seg_px or l2 < min_seg_px:
        return HumanWristTask(valid=False,
                              reason=f"手臂过短({l1:.1f},{l2:.1f}px)")
    scale = l1 + l2
    dx = wrist.x - shoulder.x
    dy = wrist.y - shoulder.y
    return HumanWristTask(u=dx / scale,
                          v=-dy / scale,
                          reach=math.hypot(dx, dy) / scale,
                          arm_scale=scale,
                          valid=True)


# 兼容旧名（V1.1 早期版本用 reach/elevation 作为主控量，现已改为 u/v）
HumanTaskState = HumanWristTask


# ============================================================
# ② 人体 -> 机器人 TCP 目标
# ============================================================
@dataclass
class TaskMappingConfig:
    """
    人体 (du, dv) -> 机器人 (x, z) 目标。

    轴关系是**配置项**（任务书 §8）：单目 2D 下"图像水平"代表哪个机器人轴
    取决于相机站位，不能硬编码。
    """
    horizontal_axis: str = "x"
    vertical_axis: str = "z"
    sign_h: float = 1.0
    sign_v: float = 1.0

    # 增益：归一化人体位移 -> 米
    kx: float = 0.55
    kz: float = 0.55

    # 目标行程限幅（base_link 坐标，米）
    x_min: float = -0.05
    x_max: float = 0.50
    z_min: float = 0.08
    z_max: float = 0.62

    # 死区（归一化人体位移），抑制关键点微抖
    deadzone_u: float = 0.02
    deadzone_v: float = 0.02
    clamp: bool = True

    def validate(self) -> None:
        for name, ax in (("horizontal_axis", self.horizontal_axis),
                         ("vertical_axis", self.vertical_axis)):
            if ax not in ("x", "y", "z"):
                raise ValueError(f"配置错误: task_space.{name} 必须是 x/y/z，"
                                 f"收到 {ax!r}")
        if self.horizontal_axis == self.vertical_axis:
            raise ValueError("配置错误: horizontal_axis 与 vertical_axis 不能相同")
        if self.sign_h not in (1.0, -1.0) or self.sign_v not in (1.0, -1.0):
            raise ValueError("配置错误: sign_h / sign_v 只能是 +1 / -1")
        if self.x_min >= self.x_max:
            raise ValueError("配置错误: task_space.x_min 必须小于 x_max")
        if self.z_min >= self.z_max:
            raise ValueError("配置错误: task_space.z_min 必须小于 z_max")
        if self.kx <= 0 or self.kz <= 0:
            raise ValueError("配置错误: task_space.kx / kz 必须为正")
        if self.deadzone_u < 0 or self.deadzone_v < 0:
            raise ValueError("配置错误: task_space.deadzone_u/v 不能为负")


def apply_deadzone(delta: float, deadzone: float) -> float:
    """
    死区：|delta| < deadzone 归零；否则**减去死区**再输出。

    为什么不"超界就保留原值"：那样在边界处有 0 -> deadzone 的台阶，
    机械臂会看到一次小阶跃。减去死区后输出是连续的。
    """
    if deadzone <= 0:
        return float(delta)
    if abs(delta) < deadzone:
        return 0.0
    return float(math.copysign(abs(delta) - deadzone, delta))


def map_human_to_task(du: float, dv: float,
                      x0: float, z0: float,
                      cfg: TaskMappingConfig
                      ) -> Tuple[float, float, Dict[str, float]]:
    """
    人体相对位移 -> 机器人 TCP 目标 (x, z)（其余轴保持基线）。

    Returns:
        (x_target, z_target, info)  info 供 debug/CSV 记录
    """
    du_dz = apply_deadzone(du, cfg.deadzone_u)
    dv_dz = apply_deadzone(dv, cfg.deadzone_v)

    x_t = x0 + cfg.sign_h * cfg.kx * du_dz
    z_t = z0 + cfg.sign_v * cfg.kz * dv_dz
    x_raw, z_raw = x_t, z_t
    if cfg.clamp:
        x_t = min(cfg.x_max, max(cfg.x_min, x_t))
        z_t = min(cfg.z_max, max(cfg.z_min, z_t))

    info = {
        "du": float(du), "dv": float(dv),
        "du_deadzoned": du_dz, "dv_deadzoned": dv_dz,
        "x_target_raw": float(x_raw), "z_target_raw": float(z_raw),
        "x_target": float(x_t), "z_target": float(z_t),
        "x_clamped": float(abs(x_t - x_raw) > 1e-12),
        "z_clamped": float(abs(z_t - z_raw) > 1e-12),
    }
    return float(x_t), float(z_t), info


# ============================================================
# ③ J2/J3 数值 IK（限位在迭代内、warm start、耗时统计）
# ============================================================
@dataclass
class IkSolution:
    q2: float = 0.0
    q3: float = 0.0
    x: float = 0.0
    z: float = 0.0
    err_x: float = 0.0
    err_z: float = 0.0
    cost: float = 0.0
    iters: int = 0
    time_ms: float = 0.0
    jump: float = 0.0            # ||q - q_prev||（跳解检测）
    at_lower: Tuple[bool, bool] = (False, False)
    at_upper: Tuple[bool, bool] = (False, False)
    valid: bool = False
    reason: str = ""

    def as_dict(self) -> Dict[str, float]:
        return {"ik_q2": self.q2, "ik_q3": self.q3,
                "ik_x": self.x, "ik_z": self.z,
                "ik_err_x": self.err_x, "ik_err_z": self.err_z,
                "ik_cost": self.cost, "ik_iters": float(self.iters),
                "ik_time_ms": self.time_ms, "ik_jump": self.jump}


@dataclass
class IkStats:
    """求解耗时统计（任务书 §14：必须记录 mean/median/P95/max）"""
    times_ms: List[float] = field(default_factory=list)
    iters: List[int] = field(default_factory=list)
    failures: int = 0

    def add(self, ms: float, iters: int, ok: bool) -> None:
        self.times_ms.append(float(ms))
        self.iters.append(int(iters))
        if not ok:
            self.failures += 1

    def summary(self) -> Dict[str, float]:
        if not self.times_ms:
            return {"n": 0}
        a = np.asarray(self.times_ms, dtype=float)
        return {
            "n": int(a.size),
            "mean_ms": float(a.mean()),
            "median_ms": float(np.median(a)),
            "p95_ms": float(np.percentile(a, 95)),
            "max_ms": float(a.max()),
            "mean_iters": float(np.mean(self.iters)) if self.iters else 0.0,
            "failures": int(self.failures),
        }

    def reset(self) -> None:
        self.times_ms.clear()
        self.iters.clear()
        self.failures = 0


class TaskSpaceIk:
    """
    (x, z) -> (q2, q3) 的数值 IK。

    目标函数（任务书 §12）：

        cost = wx*(x(q)-x_t)^2 + wz*(z(q)-z_t)^2
             + lambda_prev    * ||q - q_prev||^2
             + lambda_neutral * ||q - q_neutral||^2

    ⚠️ 权重必须与"位置误差"同量纲可比（实测教训）：
        位置项是**米²**，正则项是**弧度²**。若 lambda_prev 取 0.35，
        偏离上一帧 0.5 rad 的代价 = 0.0875，等价于"允许 0.3 米跟踪误差" ——
        IK 会为了贴着上一帧而**拒绝走向目标**，实测大面积不收敛。
        正确量级：lambda_prev ≈ 0.02（0.5 rad 偏差 ≈ 5 mm 位置误差当量），
        lambda_neutral ≈ 0.002（多解时只作轻微偏好）。
        由 test_task_space.py 的 test_hits_reachable_targets 与
        test_lambda_prev_reduces_jump 共同锁定。

    求解：投影 Gauss-Newton（LM 阻尼）+ 回溯线搜索。
    **每一步都做 box 投影**，迭代全程满足关节约束（不是最后再 clamp）。

    为什么不用通用优化器（scipy 等）：
        * 30 Hz 主循环不引入重依赖与不确定耗时；
        * 本问题只有 2 维、有解析 Jacobian，实测 3~8 次迭代收敛（< 0.2 ms）。
    """

    def __init__(self, model: Optional[ChainModel] = None,
                 joint_names: Sequence[str] = ("joint2", "joint3"),
                 q_min: Sequence[float] = (0.20, -2.30),
                 q_max: Sequence[float] = (2.40, -0.40),
                 fixed_q: Optional[Dict[str, float]] = None,
                 wx: float = 1.0, wz: float = 1.0,
                 lambda_prev: float = 0.02, lambda_neutral: float = 0.002,
                 max_iters: int = 12, tol: float = 1e-4,
                 damping: float = 1e-3, pos_tol: float = 5e-4,
                 retry_seeds: bool = True):
        self.model = model if model is not None else load_default_chain()
        self.joint_names = list(joint_names)
        if len(self.joint_names) != 2:
            raise ValueError("TaskSpaceIk 目前只解 2 个关节（joint2/joint3）")
        self.q_min = np.asarray(q_min, dtype=float)
        self.q_max = np.asarray(q_max, dtype=float)
        if np.any(self.q_min >= self.q_max):
            raise ValueError("IK 的 q_min 必须严格小于 q_max")
        self.fixed_q = dict(fixed_q or {"joint1": 0.0, "joint4": 0.0,
                                        "joint5": 0.0, "joint6": 0.0})
        for n in self.joint_names:
            self.fixed_q.pop(n, None)
        self.wx, self.wz = float(wx), float(wz)
        self.lambda_prev = float(lambda_prev)
        self.lambda_neutral = float(lambda_neutral)
        self.max_iters = int(max_iters)
        self.tol = float(tol)
        self.damping = float(damping)
        self.pos_tol = float(pos_tol)
        # 解算专用快速 FK：把链路切成三段常量矩阵，每次 FK 从 ~39 µs 降到 ~10 µs
        # （结果与 ChainModel.fk 逐位一致，见 test_kinematics.py）。
        # 这是 30 Hz 主循环里 IK 耗时的主要来源。
        self._fast = None
        try:
            if self.joint_names == ["joint2", "joint3"]:
                self._fast = TwoDofFk(self.model, "joint2", "joint3",
                                      self.fixed_q)
        except Exception:                                  # noqa: BLE001
            self._fast = None
        # 主解未收敛时是否换初值重试（两连杆的肘上/肘下双解）
        self.retry_seeds = bool(retry_seeds)
        self.retries = 0
        self.stats = IkStats()

    # ------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg, model: Optional[ChainModel] = None) -> "TaskSpaceIk":
        """由 RetargetingConfig.task_space 构造（配置唯一来源）"""
        ts = cfg.task_space
        return cls(model=model,
                   q_min=(ts.q2_min, ts.q3_min),
                   q_max=(ts.q2_max, ts.q3_max),
                   fixed_q={"joint1": cfg.fixed_joints.get("joint1", 0.0),
                            "joint4": cfg.fixed_joints.get("joint4", 0.0),
                            "joint5": 0.0, "joint6": 0.0},
                   wx=ts.wx, wz=ts.wz,
                   lambda_prev=ts.lambda_prev,
                   lambda_neutral=ts.lambda_neutral,
                   max_iters=ts.max_iters, tol=ts.tol)

    # ------------------------------------------------------------
    def _q_dict(self, q2: float, q3: float) -> Dict[str, float]:
        q = dict(self.fixed_q)
        q[self.joint_names[0]] = float(q2)
        q[self.joint_names[1]] = float(q3)
        return q

    def fk_xz(self, q2: float, q3: float) -> Tuple[float, float]:
        if self._fast is not None:
            return self._fast.pos(q2, q3)
        p = self.model.tcp(self._q_dict(q2, q3))
        return float(p[0]), float(p[2])

    def _jac_xz(self, q2: float, q3: float) -> np.ndarray:
        """2x2：d(x,z)/d(q2,q3)"""
        if self._fast is not None:
            return self._fast.jac_xz(q2, q3)
        J = self.model.jacobian(self._q_dict(q2, q3), self.joint_names)
        return J[[0, 2], :]

    def cost(self, q: np.ndarray, x_t: float, z_t: float,
             q_prev: Optional[np.ndarray],
             q_neutral: Optional[np.ndarray]) -> float:
        x, z = self.fk_xz(q[0], q[1])
        c = self.wx * (x - x_t) ** 2 + self.wz * (z - z_t) ** 2
        if q_prev is not None and self.lambda_prev > 0:
            c += self.lambda_prev * float(np.sum((q - q_prev) ** 2))
        if q_neutral is not None and self.lambda_neutral > 0:
            c += self.lambda_neutral * float(np.sum((q - q_neutral) ** 2))
        return float(c)

    # ------------------------------------------------------------
    def solve(self, x_target: float, z_target: float,
              q_prev: Optional[Tuple[float, float]] = None,
              q_neutral: Optional[Tuple[float, float]] = None,
              record_stats: bool = True) -> IkSolution:
        """
        求解 (q2, q3) 使 TCP 到达 (x_target, z_target)。

        约束在**迭代内**满足：每步先算无约束 GN 步，再 box 投影 + 回溯线搜索。
        主解未收敛时用备用初值重试（见 `retry_seeds`）。
        """
        t0 = time.perf_counter()
        qp = None if q_prev is None else np.asarray(q_prev, dtype=float)
        qn = None if q_neutral is None else np.asarray(q_neutral, dtype=float)
        seed = (q_prev if q_prev is not None
                else tuple((self.q_min + self.q_max) / 2.0))
        sol = self._iterate(x_target, z_target, seed, qp, qn, t0)

        # ---- 备用初值重试 ----
        # 两连杆对同一 (x,z) 常有两个解（肘上/肘下）。warm start 落在错分支时
        # GN 会在局部极小停下。此时换初值重试：① 区间中点 ② q3 方向镜像。
        # 只在主解未收敛时触发，正常跟踪几乎不发生。
        if self.retry_seeds and not sol.valid:
            seeds = [tuple((self.q_min + self.q_max) / 2.0)]
            if qp is not None:
                mir = self.q_min[1] + self.q_max[1] - float(qp[1])
                seeds.append((float(qp[0]),
                              float(np.clip(mir, self.q_min[1], self.q_max[1]))))
            keep_p, keep_n = self.lambda_prev, self.lambda_neutral
            self.lambda_prev, self.lambda_neutral = 0.0, 0.0
            best = sol
            for sd in seeds:
                self.retries += 1
                alt = self._iterate(x_target, z_target, sd, qp, qn,
                                    time.perf_counter())
                if (math.hypot(alt.err_x, alt.err_z)
                        < math.hypot(best.err_x, best.err_z)):
                    best = alt
            self.lambda_prev, self.lambda_neutral = keep_p, keep_n
            sol = best

        if record_stats:
            self.stats.add(sol.time_ms, sol.iters, sol.valid)
        return sol

    # ------------------------------------------------------------
    def _iterate(self, x_target: float, z_target: float, seed,
                 qp: Optional[np.ndarray], qn: Optional[np.ndarray],
                 t0: float) -> IkSolution:
        """投影 Gauss-Newton 主循环（不做重试、不记统计）"""
        q = np.clip(np.asarray(seed, dtype=float), self.q_min, self.q_max)

        def total(qq):
            return self.cost(qq, x_target, z_target, qp, qn)

        cur = total(q)
        it = 0
        for it in range(1, self.max_iters + 1):
            x, z = self.fk_xz(q[0], q[1])
            ex, ez = x - x_target, z - z_target
            J = self._jac_xz(q[0], q[1])
            g = (np.array([2 * self.wx * ex, 2 * self.wz * ez]) @ J).astype(float)
            if qp is not None and self.lambda_prev > 0:
                g = g + 2 * self.lambda_prev * (q - qp)
            if qn is not None and self.lambda_neutral > 0:
                g = g + 2 * self.lambda_neutral * (q - qn)

            H = (2 * self.wx * np.outer(J[0], J[0])
                 + 2 * self.wz * np.outer(J[1], J[1])
                 + 2 * (self.lambda_prev + self.lambda_neutral) * np.eye(2)
                 + self.damping * np.eye(2))
            try:
                step = -np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                return self._fail("Hessian 奇异", q, t0, it, x_target,
                                  z_target, qp)
            if not np.all(np.isfinite(step)):
                return self._fail("步长非有限", q, t0, it, x_target,
                                  z_target, qp)

            # ---- 回溯线搜索（先投影再比较代价）----
            alpha, improved = 1.0, False
            for _ in range(8):
                cand = np.clip(q + alpha * step, self.q_min, self.q_max)
                c_new = total(cand)
                if c_new <= cur - 1e-15:
                    q, cur, improved = cand, c_new, True
                    break
                alpha *= 0.5
            if not improved:
                break
            if float(np.max(np.abs(alpha * step))) < self.tol:
                break

        x, z = self.fk_xz(q[0], q[1])
        ex, ez = x - x_target, z - z_target
        sol = IkSolution(
            q2=float(q[0]), q3=float(q[1]), x=float(x), z=float(z),
            err_x=float(ex), err_z=float(ez), cost=float(cur), iters=int(it),
            time_ms=(time.perf_counter() - t0) * 1e3,
            jump=0.0 if qp is None else float(np.linalg.norm(q - qp)),
            at_lower=(bool(q[0] <= self.q_min[0] + 1e-9),
                      bool(q[1] <= self.q_min[1] + 1e-9)),
            at_upper=(bool(q[0] >= self.q_max[0] - 1e-9),
                      bool(q[1] >= self.q_max[1] - 1e-9)),
            valid=True)
        if math.hypot(ex, ez) > self.pos_tol:
            sol.valid = False
            sol.reason = ("未收敛: |err|=%.1fmm > %.1fmm"
                          % (math.hypot(ex, ez) * 1000, self.pos_tol * 1000))
        return sol

    def _fail(self, reason, q, t0, it, x_t, z_t, qp) -> IkSolution:
        x, z = self.fk_xz(q[0], q[1])
        sol = IkSolution(q2=float(q[0]), q3=float(q[1]), x=x, z=z,
                         err_x=x - x_t, err_z=z - z_t, iters=it,
                         time_ms=(time.perf_counter() - t0) * 1e3,
                         jump=0.0 if qp is None else float(np.linalg.norm(q - qp)),
                         valid=False, reason=reason)
        self.stats.add(sol.time_ms, it, False)
        return sol


# ============================================================
# ④ Workspace 扫描 + Neutral 自动搜索
# ============================================================
@dataclass
class WorkspacePoint:
    q2: float
    q3: float
    x: float
    y: float
    z: float
    m2_lo: float
    m2_hi: float
    m3_lo: float
    m3_hi: float
    sigma_min: float
    manipulability: float

    def as_dict(self) -> dict:
        return {"q2": self.q2, "q3": self.q3, "x": self.x, "y": self.y,
                "z": self.z, "m2_lo": self.m2_lo, "m2_hi": self.m2_hi,
                "m3_lo": self.m3_lo, "m3_hi": self.m3_hi,
                "sigma_min": self.sigma_min,
                "manipulability": self.manipulability}


def scan_workspace(model: Optional[ChainModel] = None,
                   q2_range: Tuple[float, float] = (0.20, 2.40),
                   q3_range: Tuple[float, float] = (-2.30, -0.40),
                   n2: int = 31, n3: int = 31,
                   fixed_q: Optional[Dict[str, float]] = None,
                   ) -> List[WorkspacePoint]:
    """
    q2 × q3 二维网格扫描（任务书 §4）。

    记录 TCP x/y/z、到给定区间上下界的余量、XZ 平面最小奇异值（可操作度）。
    """
    model = model if model is not None else load_default_chain()
    fixed = dict(fixed_q or {"joint1": 0.0, "joint4": 0.0,
                             "joint5": 0.0, "joint6": 0.0})
    names = ("joint2", "joint3")
    pts: List[WorkspacePoint] = []
    for q2 in np.linspace(q2_range[0], q2_range[1], n2):
        for q3 in np.linspace(q3_range[0], q3_range[1], n3):
            q = dict(fixed)
            q["joint2"], q["joint3"] = float(q2), float(q3)
            p = model.tcp(q)
            J = model.jacobian_xz(q, names)
            sv = np.linalg.svd(J, compute_uv=False)
            pts.append(WorkspacePoint(
                q2=float(q2), q3=float(q3),
                x=float(p[0]), y=float(p[1]), z=float(p[2]),
                m2_lo=float(q2 - q2_range[0]), m2_hi=float(q2_range[1] - q2),
                m3_lo=float(q3 - q3_range[0]), m3_hi=float(q3_range[1] - q3),
                sigma_min=float(sv[-1]),
                manipulability=float(abs(np.linalg.det(J)))))
    return pts


def _norm01(v: float, lo: float, hi: float) -> float:
    if hi - lo < 1e-12:
        return 0.0
    return max(0.0, min(1.0, (v - lo) / (hi - lo)))


@dataclass
class NeutralCandidate:
    q2: float
    q3: float
    x: float
    z: float
    margins: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    sigma_min: float = 0.0
    manipulability: float = 0.0
    score: float = 0.0
    parts: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"q2": self.q2, "q3": self.q3, "x": self.x, "z": self.z,
                "score": self.score, "sigma_min": self.sigma_min,
                "manipulability": self.manipulability,
                "margins": {k: list(v) for k, v in self.margins.items()},
                "parts": dict(self.parts)}

    def table_row(self) -> str:
        m2 = self.margins.get("joint2", (0.0, 0.0))
        m3 = self.margins.get("joint3", (0.0, 0.0))
        return ("q2=%+.3f q3=%+.3f | TCP=(%+.3f, %+.3f) | "
                "m2=(%+.3f,%+.3f) m3=(%+.3f,%+.3f) | sigma=%+.4f | "
                "score=%.4f (ws %.3f lim %.3f manip %.3f)"
                % (self.q2, self.q3, self.x, self.z, m2[0], m2[1],
                   m3[0], m3[1], self.sigma_min, self.score,
                   self.parts.get("workspace", 0.0),
                   self.parts.get("limit", 0.0),
                   self.parts.get("manip", 0.0)))


def neutral_candidates(points: Sequence[WorkspacePoint],
                       x_center: Optional[float] = None,
                       z_center: Optional[float] = None,
                       w_workspace: float = 1.0, w_limit: float = 1.0,
                       w_manip: float = 0.6,
                       top: int = 5,
                       x_span: Optional[Tuple[float, float]] = None,
                       z_span: Optional[Tuple[float, float]] = None
                       ) -> List[NeutralCandidate]:
    """
    Neutral 自动搜索（任务书 §5）。

        score = w_workspace * workspace_center_score
              + w_limit     * joint_margin_score
              + w_manip     * manipulability_score

    * workspace_center_score：TCP 离工作区中心的归一化距离（越近越好）。
      中心默认取扫描点云 x/z 区间的中点。
    * joint_margin_score：q2/q3 到区间上下界余量的**最小值**归一化
      （最短板决定分数 —— 保证两个关节双向都有余量）。
    * manipulability_score：XZ 平面最小奇异值 sigma_min 归一化，避免接近奇异。
    """
    if not points:
        return []
    xs = np.array([p.x for p in points])
    zs = np.array([p.z for p in points])
    if x_span is None:
        x_span = (float(xs.min()), float(xs.max()))
    if z_span is None:
        z_span = (float(zs.min()), float(zs.max()))
    if x_center is None:
        x_center = 0.5 * (x_span[0] + x_span[1])
    if z_center is None:
        z_center = 0.5 * (z_span[0] + z_span[1])

    half_x = max(1e-9, 0.5 * (x_span[1] - x_span[0]))
    half_z = max(1e-9, 0.5 * (z_span[1] - z_span[0]))
    sig = np.array([p.sigma_min for p in points])
    sig_lo, sig_hi = float(sig.min()), float(sig.max())
    m_all = [min(p.m2_lo, p.m2_hi, p.m3_lo, p.m3_hi) for p in points]
    m_lo, m_hi = float(min(m_all)), float(max(m_all))

    out: List[NeutralCandidate] = []
    for p in points:
        d_center = math.hypot((p.x - x_center) / half_x,
                              (p.z - z_center) / half_z)
        ws_score = max(0.0, 1.0 - d_center)
        limit_score = _norm01(min(p.m2_lo, p.m2_hi, p.m3_lo, p.m3_hi),
                              m_lo, m_hi)
        manip_score = _norm01(p.sigma_min, sig_lo, sig_hi)
        score = (w_workspace * ws_score + w_limit * limit_score
                 + w_manip * manip_score)
        out.append(NeutralCandidate(
            q2=p.q2, q3=p.q3, x=p.x, z=p.z,
            margins={"joint2": (p.m2_lo, p.m2_hi),
                     "joint3": (p.m3_lo, p.m3_hi)},
            sigma_min=p.sigma_min, manipulability=p.manipulability,
            score=float(score),
            parts={"workspace": float(ws_score), "limit": float(limit_score),
                   "manip": float(manip_score)}))
    out.sort(key=lambda c: -c.score)
    return out[:top]
