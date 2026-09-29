# -*- coding: utf-8 -*-
"""
piper_human_retargeting.arm_geometry
====================================
从人体手臂关键点计算用于映射的几何量。

本模块是**纯几何**，不含任何机械臂知识、不含标定、不做映射：
    输入：ArmKeypoints（肩 / 肘 / 有效腕）
    输出：ArmGeometry（上臂方向角 / 肘夹角 / 前臂方向角）

这样切分的目的：
    * 几何定义与「怎么映射到 Piper」解耦，换机器人不用改几何；
    * 可以脱离摄像头与机械臂做单元测试；
    * 阶段八升级到 3D 时，只需新增一个 3D 版几何函数，
      映射层接口保持不变。

坐标约定（**务必看清，这是最容易出错的地方**）：
    图像坐标系 x 向右、y 向下。
    本模块输出的角度统一做 **y 翻转**，即：
        0°   = 指向图像右方
        +90° = 指向图像上方
        -90° = 指向图像下方
        ±180°= 指向图像左方
    这样「角度变大 = 手臂抬高」，与人的直觉一致，也便于写 sign 配置。

关于镜像：
    若显示端做了水平镜像，**不影响本模块** —— 本模块只消费
    `ArmKeypoints` 中的原始图像坐标，与显示无关
    （visualization 里的镜像换算只作用于绘制）。
"""

from dataclasses import dataclass
import math
from typing import List, Optional, Tuple

import numpy as np

from piper_human_control import ArmKeypoints, Keypoint


@dataclass
class ArmGeometry:
    """
    一帧的人体手臂几何量。

    Attributes:
        upper_arm_angle_deg: 上臂方向角。pivot=shoulder 时为「肩→肘」，
                             pivot=anchor 时为「画面锚点→肘」
        forearm_angle_deg:   前臂方向角。pivot=shoulder 时为「肘→腕」，
                             pivot=anchor 时为「画面锚点→腕」
        elbow_angle_deg:     肘夹角。0 = 完全伸直，180 = 完全折回。
                             这一项**始终**由「肩-肘-腕」三点构成，
                             与 pivot 选择无关（它是关节角，不是方向角）
        upper_arm_len_px:    上臂像素长度
        forearm_len_px:      前臂像素长度
        valid:               本帧几何是否可用
        reason:              valid=False 时说明原因，便于排查
        pivot:               本帧实际使用的枢轴（"shoulder" / "anchor"）
        anchor_xy:           使用锚点时的锚点像素坐标，便于可视化
    """
    upper_arm_angle_deg: float = 0.0
    forearm_angle_deg: float = 0.0
    elbow_angle_deg: float = 0.0
    upper_arm_len_px: float = 0.0
    forearm_len_px: float = 0.0
    valid: bool = False
    reason: str = ""
    pivot: str = "shoulder"
    anchor_xy: Optional[Tuple[float, float]] = None
    # ---- 2D task-space（V1.1）----
    # 手腕相对肩的**归一化**位置，用于「人手去哪、机器人 TCP 去哪」的重映射。
    #   u = (W.x - S.x) / L      + 表示图像右
    #   v = -(W.y - S.y) / L     + 表示**手腕在肩上方**（图像 y 向下，故取负）
    #   L = |S-E| + |E-W|        手臂总长（像素），使量纲与人机距离无关
    #
    # ⚠️ 必须放在**最后**：本 dataclass 存在按位置构造的调用
    #    （ArmGeometry(upper, forearm, elbow, l1, l2, True)），
    #    新字段插在中间会把 `valid` 挤到 human_u 上 —— 实测导致
    #    "标定失败: 无样本"（样本全被判为无效）。
    human_u: float = 0.0
    human_v: float = 0.0

    def as_dict(self) -> dict:
        """便于日志与映射层按键取用"""
        return {
            "upper_arm_angle_deg": self.upper_arm_angle_deg,
            "forearm_angle_deg": self.forearm_angle_deg,
            "elbow_angle_deg": self.elbow_angle_deg,
            "human_u": self.human_u,
            "human_v": self.human_v,
        }

    def summary(self) -> str:
        if not self.valid:
            return f"无效({self.reason})"
        return (f"上臂{self.upper_arm_angle_deg:+7.1f}° "
                f"前臂{self.forearm_angle_deg:+7.1f}° "
                f"肘{self.elbow_angle_deg:6.1f}° "
                f"[{self.pivot}] "
                f"臂长{self.upper_arm_len_px:.0f}/{self.forearm_len_px:.0f}px")


def _angle_deg(dx: float, dy: float) -> float:
    """
    图像向量 -> 方向角（度），已做 y 翻转使「向上为正」。

    返回范围 (-180, 180]。
    """
    return math.degrees(math.atan2(-dy, dx))


def compute_arm_geometry(arm: Optional[ArmKeypoints],
                         min_confidence: float = 0.0,
                         require_complete: bool = True,
                         pivot: str = "shoulder",
                         image_size: Optional[Tuple[int, int]] = None,
                         anchor: Tuple[float, float] = (0.5, 1.0)) -> ArmGeometry:
    """
    计算人体手臂几何量。

    Args:
        arm:               手臂关键点（**会用 effective_wrist**，
                           即优先手部识别的腕关节，见 types.ArmKeypoints）
        min_confidence:    关键点置信度门限，低于则判定无效
        require_complete:  是否要求三点齐全
        pivot:             "shoulder" 以人体肩点为枢轴（原始做法）；
                           "anchor"   以画面锚点为枢轴（见下）
        image_size:        (width, height)。pivot="anchor" 时**必须**给，
                           否则退化为 shoulder 并在 reason 里说明
        anchor:            锚点归一化位置，默认 (0.5, 1.0) = 底边中点

    pivot="anchor" 的动机（实测反馈）：
        以人体肩点为枢轴时，上臂角/前臂角是**肢体自身的朝向**，
        而显示端是镜像的，于是「角度变大」在观感上对应手臂往下，
        映射到机械臂就成了反直觉的**反向弯曲**（人伸手、机器人缩回）。

        改用画面底边中点作枢轴后，角度变成「手相对画面中心的方位」：
            手举到画面中上方 -> 前臂角 ≈ +90°
            手放低到画面下方 -> 前臂角 ≈ -90°
        「越伸展/越抬高、角度越大」是直接读出来的，不再绕一层肢体朝向。

        附带好处：肩关键点不再参与上臂/前臂角，
        因此肩部遮挡、出画时的抖动不会再同时污染三个量。
        （肘夹角仍用三点计算 —— 它是关节角，与枢轴选择无关。）

    Returns:
        ArmGeometry；不可用时 valid=False 并给出 reason
    """
    if arm is None:
        return ArmGeometry(valid=False, reason="无手臂数据")

    # 注意：这里取 effective_wrist（手部识别的腕关节），
    # 而不是 arm.wrist（人体姿态腕部）—— 遵循项目核心规则。
    s, e, w = arm.shoulder, arm.elbow, arm.effective_wrist

    if e is None or w is None:
        return ArmGeometry(valid=False, reason="关键点缺失")

    # ---- 决定枢轴 ----
    # 注意这里的**降级**策略：
    # config 默认 anchor（对用户更直观），但 anchor 需要知道画面尺寸。
    # 若调用方没给 image_size（离线单测、纯几何工具），
    # 就退回 shoulder，而不是判定无效 —— 无效会让所有不传尺寸的调用方
    # 静默失效，那种问题很难查。
    # 实际用的枢轴记录在 geo.pivot 里，所以降级是**可见**的。
    use_anchor = (pivot == "anchor" and image_size is not None)
    ask_shoulder = not use_anchor
    if ask_shoulder and s is None:
        return ArmGeometry(valid=False, reason="关键点缺失")

    if ask_shoulder:
        anchor_pt: Optional[Tuple[float, float]] = None
        p_upper = (s.x, s.y)                      # 肩 -> 肘
        p_fore = (e.x, e.y)                       # 肘 -> 腕
    else:
        W, H = image_size
        ax, ay = anchor[0] * W, anchor[1] * H
        anchor_pt = (ax, ay)
        p_upper = (ax, ay)                        # 锚点 -> 肘
        p_fore = (ax, ay)                         # 锚点 -> 腕

    # ---- 置信度检查 ----
    # 无论哪种枢轴，参与「方向角/关节角」的点都必须有效。
    # pivot=anchor 时肩点不参与几何，因此**不再要求肩点有效** ——
    # 这正是这个模式对肩部遮挡更鲁棒的原因。
    need = [("肘", e), ("腕", w)]
    if ask_shoulder:
        need.insert(0, ("肩", s))
    if require_complete:
        for name, kp in need:
            if not kp.is_valid:
                return ArmGeometry(
                    valid=False, reason=f"{name}关键点无效(置信度为0)")
    if min_confidence > 0.0:
        for name, kp in need:
            if kp.confidence < min_confidence:
                return ArmGeometry(
                    valid=False,
                    reason=f"{name}置信度{kp.confidence:.2f}<{min_confidence:.2f}")

    # ---- 向量 ----
    v_upper = np.array([e.x - p_upper[0], e.y - p_upper[1]], dtype=np.float64)
    v_fore = np.array([w.x - p_fore[0], w.y - p_fore[1]], dtype=np.float64)

    len_upper = float(np.linalg.norm(v_upper))
    len_fore = float(np.linalg.norm(v_fore))

    # 长度过小说明关键点重合（YOLO 在遮挡/出画时会出现），角度不可信
    MIN_LEN = 8.0
    if len_upper < MIN_LEN:
        return ArmGeometry(valid=False,
                           reason=f"上臂过短({len_upper:.1f}px)")
    if len_fore < MIN_LEN:
        return ArmGeometry(valid=False,
                           reason=f"前臂过短({len_fore:.1f}px)")

    # ---- 上臂 / 前臂方向角 ----
    upper_ang = _angle_deg(v_upper[0], v_upper[1])
    fore_ang = _angle_deg(v_fore[0], v_fore[1])

    # ---- 肘夹角（始终用肩-肘-腕三点；缺肩点时无法计算）----
    # 用「肩->肘」与「腕->肘」两个向量的夹角：
    #   伸直时两向量反向 -> 夹角 180°
    #   完全折回时两向量同向 -> 夹角 0°
    # 为了让「0 = 伸直、越大越弯」更符合直觉，这里返回 180 - 夹角，
    # 即：0 = 完全伸直，180 = 完全折回。
    if s is None:
        elbow_bend = 0.0        # 无可用的肩点，关节角无法定义（不伪造 180）
    else:
        a = np.array([s.x - e.x, s.y - e.y], dtype=np.float64)
        b = np.array([w.x - e.x, w.y - e.y], dtype=np.float64)
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        if na < 1e-6 or nb < 1e-6:
            return ArmGeometry(valid=False, reason="肘部向量退化")
        cosang = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
        interior = math.degrees(math.acos(cosang))      # 180=伸直, 0=折回
        elbow_bend = 180.0 - interior                   # 0=伸直, 180=折回

    return ArmGeometry(
        upper_arm_angle_deg=upper_ang,
        forearm_angle_deg=fore_ang,
        elbow_angle_deg=elbow_bend,
        upper_arm_len_px=len_upper,
        forearm_len_px=len_fore,
        # ---- 2D task-space（V1.1）：手腕相对肩的归一化位置 ----
        # L 取「上臂+前臂」像素长度之和，因此 u/v 与体型、与相机距离无关。
        # 这是「人的手往哪走，机器人 TCP 就往哪走」所需的量（任务书 §7）。
        #   u > 0：手腕在肩的**图像右侧**
        #   v > 0：手腕在肩的**上方**（图像 y 向下，故取负号）
        human_u=float((w.x - p_upper[0]) / (len_upper + len_fore)),
        human_v=float(-(w.y - p_upper[1]) / (len_upper + len_fore)),
        valid=True,
        reason="",
        pivot=pivot if not ask_shoulder else "shoulder",
        anchor_xy=anchor_pt,
    )


def angle_delta_deg(a: float, b: float) -> float:
    """
    两个角度的差值，结果归一化到 (-180, 180]。

    必要性：角度是周期量。若不做归一化，170° 与 -170° 的差会算成 340°，
    导致机械臂朝完全相反的方向猛动。映射时必须用它。
    """
    d = (a - b) % 360.0
    if d > 180.0:
        d -= 360.0
    return d
