# -*- coding: utf-8 -*-
"""
piper_human_perception.visualization
====================================
实时可视化层。

显示内容（对应任务书阶段三要求）：
    * YOLO 检测图像
    * shoulder / elbow / wrist 三个关键点
    * 手臂骨架连线（上臂 + 前臂）
    * 每个关键点的 confidence 数值
    * 上臂方向与肘部夹角的实时数值（阶段四的输入，先显示出来便于观察）

全部使用 OpenCV 绘制，不依赖 matplotlib，保证实时性。
"""

import math
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from piper_human_control import ArmKeypoints, Keypoint

from .hand import (FINGER_NAMES, HAND_CONNECTIONS, HandDetection,
                   HandGeometry)
from .perception import Frame
from .pose import PoseDetection
from .textdraw import TextRenderer


# ============================================================
# 配色（BGR）
# ============================================================
COLOR_SHOULDER = (0, 165, 255)      # 橙
COLOR_ELBOW = (0, 255, 255)         # 黄
COLOR_WRIST = (0, 255, 0)           # 绿
COLOR_SKELETON = (255, 128, 0)      # 蓝
COLOR_INVALID = (128, 128, 128)     # 灰（低置信度）
COLOR_TEXT = (255, 255, 255)
COLOR_OK = (0, 255, 0)
COLOR_WARN = (0, 200, 255)
COLOR_BAD = (0, 0, 255)

# COCO-17 骨架连接（用于画完整人体，辅助判断检测质量）
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),                 # 头
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),        # 双臂
    (5, 11), (6, 12), (11, 12),                     # 躯干
    (11, 13), (13, 15), (12, 14), (14, 16),         # 双腿
]

# 关注的三个关键点 -> 颜色
KP_COLORS = {
    "right_shoulder": COLOR_SHOULDER, "left_shoulder": COLOR_SHOULDER,
    "right_elbow": COLOR_ELBOW, "left_elbow": COLOR_ELBOW,
    "right_wrist": COLOR_WRIST, "left_wrist": COLOR_WRIST,
}

RADIUS = 6
THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX

# ---------------- 手部配色 ----------------
COLOR_HAND_BONE = (255, 160, 60)      # 手部骨架（浅蓝）
COLOR_FINGERTIP = (0, 128, 255)       # 指尖（橙）
COLOR_PALM = (255, 0, 255)            # 掌心（品红）
COLOR_HAND_ROOT = (0, 255, 128)       # 手根（青绿）
COLOR_ORIENT_ARROW = (0, 255, 255)    # 朝向箭头（黄）
COLOR_ANCHOR = (255, 0, 255)          # 画面锚点（品红）

HAND_BONE_THICKNESS = 2
FINGERTIP_RADIUS = 4


class PoseVisualizer:
    """
    姿态可视化器。

    参数：
        show_all_skeleton: 是否绘制完整 17 点骨架。
                           调姿势时开着更有用（能看出是不是整体检测歪了）；
                           正式跑映射时可以关掉只看手臂。
        show_metrics:      是否显示上臂角 / 肘夹角数值
        mirror:            是否水平镜像显示。
                           摄像头对着人时镜像更符合直觉（像照镜子），
                           但注意：**镜像只影响显示，不影响传给下游的坐标**。
        pivot:             几何枢轴，与 retargeting 配置保持一致。
                           "anchor" 时会在画面上**画出锚点**及
                           「锚点->肘 / 锚点->腕」两条参考线 ——
                           否则用户看不到几何真正用的是哪个基准点。
        anchor:            锚点归一化位置（默认 0.5, 1.0 = 底边中点）
    """

    def __init__(self, show_all_skeleton: bool = True,
                 show_metrics: bool = True,
                 mirror: bool = True,
                 pivot: str = "shoulder",
                 anchor: tuple = (0.5, 1.0)):
        self.show_all_skeleton = show_all_skeleton
        self.show_metrics = show_metrics
        self.mirror = mirror
        self.pivot = pivot
        self.anchor = anchor
        # 中英文混排绘制器（cv2.putText 不支持中文）
        self.text = TextRenderer(font_size=16)

    # ------------------------------------------------------------
    def render(self, frame: Frame, det: PoseDetection,
               hand_det: Optional[HandDetection] = None,
               extra_lines: Optional[list] = None) -> np.ndarray:
        """
        绘制一帧。

        Args:
            frame:       原始帧
            det:         人体姿态检测结果
            hand_det:    手部检测结果（可选；为 None 则不画手）
            extra_lines: 额外要显示的文字行

        Returns:
            绘制后的图像（新数组，不修改原帧）
        """
        canvas = frame.rgb.copy()

        # 镜像必须在「画关键点之前」做：
        #   反转的是图像内容（像照镜子），
        #   如果在画完 HUD 之后再整体 flip，文字会变成镜像反字。
        if self.mirror:
            canvas = cv2.flip(canvas, 1)

        if self.show_all_skeleton and det.all_keypoints:
            self._draw_full_skeleton(canvas, det.all_keypoints)

        if det.bbox is not None:
            self._draw_bbox(canvas, det.bbox, det.score)

        if det.arm is not None:
            self._draw_arm(canvas, det.arm)

        # 手部画在手臂之后，保证手部关键点在最上层（手通常压在手腕附近）
        if hand_det is not None and hand_det.hand is not None:
            self._draw_hand(canvas, hand_det)

        # 几何锚点（若启用了 anchor 枢轴）
        if self.pivot == "anchor":
            self._draw_anchor(canvas, frame, det)

        # HUD 最后画，永不翻转，保证文字可读
        self._draw_hud(canvas, frame, det, extra_lines, hand_det)

        return canvas

    # ------------------------------------------------------------
    # 坐标镜像
    # ------------------------------------------------------------
    def _mx(self, x: float, width: int) -> float:
        """
        把图像坐标 x 映射到「镜像后的画布」坐标。

        因为镜像（cv2.flip）只翻转了画面，而关键点坐标仍来自原图，
        所以显示时必须同步换算：x' = (W - 1) - x。
        注意：**这只是显示用换算，不会改变传给下游的坐标**。
        """
        if not self.mirror:
            return x
        return (width - 1) - x

    def _pt(self, kp: Optional[Keypoint], width: int) -> Optional[Tuple[int, int]]:
        """把关键点转成整数像素坐标（含必要的镜像换算）；无效点返回 None"""
        if kp is None or not kp.is_valid:
            return None
        return (int(round(self._mx(kp.x, width))), int(round(kp.y)))

    def _draw_full_skeleton(self, img: np.ndarray,
                            kps: Dict[str, Keypoint]) -> None:
        """绘制完整 17 点骨架（淡色，作为背景参考）"""
        from .pose import COCO_KEYPOINT_NAMES

        w = img.shape[1]

        # 连线
        for i, j in COCO_SKELETON:
            a = kps.get(COCO_KEYPOINT_NAMES[i])
            b = kps.get(COCO_KEYPOINT_NAMES[j])
            pa, pb = self._pt(a, w), self._pt(b, w)
            if pa and pb:
                cv2.line(img, pa, pb, (180, 180, 180), 1, cv2.LINE_AA)

        # 点
        for name, kp in kps.items():
            if not kp.is_valid or name in KP_COLORS:
                continue                 # 手臂三点单独高亮绘制
            p = self._pt(kp, w)
            if p:
                cv2.circle(img, p, 2, (150, 150, 150), -1, cv2.LINE_AA)

    def _draw_bbox(self, img: np.ndarray, bbox, score: float) -> None:
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        if self.mirror:
            w = img.shape[1]
            x1, x2 = int(round(self._mx(x1, w))), int(round(self._mx(x2, w)))
            x1, x2 = min(x1, x2), max(x1, x2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 200, 0), 1, cv2.LINE_AA)
        if score > 0:
            self.text.draw(img, f"person {score:.2f}",
                           (x1, max(14, y1 - 5)), (255, 200, 0))

    def _draw_arm(self, img: np.ndarray, arm: ArmKeypoints) -> None:
        """绘制目标手臂：骨架 + 三点 + 置信度"""
        w = img.shape[1]
        # 腕部取 effective_wrist：优先手部识别的腕关节，其次人体姿态
        s, e, wr = arm.shoulder, arm.elbow, arm.effective_wrist
        ps, pe, pw = self._pt(s, w), self._pt(e, w), self._pt(wr, w)

        # 上臂 S-E
        if ps and pe:
            cv2.line(img, ps, pe, COLOR_SKELETON, THICKNESS + 1, cv2.LINE_AA)
        # 前臂 E-W
        if pe and pw:
            cv2.line(img, pe, pw, COLOR_SKELETON, THICKNESS + 1, cv2.LINE_AA)

        for kp, color, label in ((s, COLOR_SHOULDER, "S"),
                                 (e, COLOR_ELBOW, "E"),
                                 (wr, COLOR_WRIST, "W")):
            self._draw_keypoint(img, kp, color, label, w)

    def _draw_keypoint(self, img: np.ndarray, kp: Optional[Keypoint],
                       color, label: str, width: int) -> None:
        """绘制单个关键点 + 置信度"""
        if kp is None:
            return

        valid = kp.is_valid
        draw_color = color if valid else COLOR_INVALID
        p = (int(round(self._mx(kp.x, width))), int(round(kp.y)))

        # 十字 + 圆
        cv2.circle(img, p, RADIUS, draw_color, -1, cv2.LINE_AA)
        cv2.circle(img, p, RADIUS + 2, (0, 0, 0), 1, cv2.LINE_AA)

        # 标签：S/E/W + 置信度
        text = f"{label} {kp.confidence:.2f}"
        tx, ty = p[0] + RADIUS + 4, p[1] - RADIUS
        self.text.draw(img, text, (tx, ty), draw_color)

    def _draw_anchor(self, img: np.ndarray, frame: Frame,
                     det: PoseDetection) -> None:
        """
        画出几何锚点，以及「锚点 -> 肘 / 锚点 -> 腕」两条参考线。

        为什么要画：pivot=anchor 时肩关键点不参与几何，
        如果画面上不显示锚点，用户会以为程序还在用人体肩点，
        从而无法理解角度的来源。
        """
        h, w = img.shape[:2]
        ax = int(round(self._mx(self.anchor[0] * w, w)))
        ay = int(round(self.anchor[1] * h))

        # 参考线（先画线，点压在上面）
        if det.arm is not None:
            for kp, color in ((det.arm.elbow, COLOR_ELBOW),
                              (det.arm.effective_wrist, COLOR_WRIST)):
                if kp is None or not kp.is_valid:
                    continue
                p = (int(round(self._mx(kp.x, w))), int(round(kp.y)))
                cv2.line(img, (ax, ay), p, color, 1, cv2.LINE_AA)

        cv2.drawMarker(img, (ax, ay), COLOR_ANCHOR, cv2.MARKER_TILTED_CROSS,
                       22, 2, cv2.LINE_AA)
        self.text.draw(img, "PIVOT", (ax + 16, ay - 8), COLOR_ANCHOR)

    def _draw_hand(self, img: np.ndarray, hand_det: HandDetection) -> None:
        """
        绘制手部：21 点骨架 + 指尖 + 掌心 + 手根 + 朝向箭头。

        注意：手部模型的坐标是**原图坐标**，镜像显示时需要同步换算，
        与手臂关键点一致（_mx）。
        """
        hand = hand_det.hand
        if hand is None:
            return

        w = img.shape[1]

        def P(kp):
            if kp is None or not kp.is_valid:
                return None
            return (int(round(self._mx(kp.x, w))), int(round(kp.y)))

        # ---- 骨架连线 ----
        lm = hand.landmarks
        for a_name, b_name in HAND_CONNECTIONS:
            pa, pb = P(lm.get(a_name)), P(lm.get(b_name))
            if pa and pb:
                cv2.line(img, pa, pb, COLOR_HAND_BONE,
                         HAND_BONE_THICKNESS, cv2.LINE_AA)

        # ---- 21 个关键点 ----
        for name, kp in lm.items():
            p = P(kp)
            if p is None:
                continue
            if name.endswith("_tip"):
                cv2.circle(img, p, FINGERTIP_RADIUS, COLOR_FINGERTIP, -1, cv2.LINE_AA)
                cv2.circle(img, p, FINGERTIP_RADIUS + 1, (0, 0, 0), 1, cv2.LINE_AA)
            elif name == "wrist":
                cv2.circle(img, p, 5, COLOR_HAND_ROOT, -1, cv2.LINE_AA)
            else:
                cv2.circle(img, p, 2, COLOR_HAND_BONE, -1, cv2.LINE_AA)

        # ---- 掌心 ----
        pp = P(hand.palm_center)
        if pp:
            cv2.circle(img, pp, 5, COLOR_PALM, -1, cv2.LINE_AA)
            cv2.circle(img, pp, 6, (0, 0, 0), 1, cv2.LINE_AA)

        # ---- 朝向箭头：手腕 -> 掌心方向 ----
        geom = hand_det.geometry
        if geom is not None and pp is not None:
            pr = P(hand.hand_root)
            if pr:
                # 箭头长度取掌宽的量级，避免遮挡画面
                length = max(18.0, min(60.0, geom.palm_width * 1.2))
                ang = math.radians(geom.orientation_deg)
                # 注意：orientation_deg 是「y 向上为正」的角度，
                # 转回图像坐标（y 向下）需要取负
                tip = (int(pr[0] + length * math.cos(ang)),
                       int(pr[1] - length * math.sin(ang)))
                cv2.arrowedLine(img, pr, tip, COLOR_ORIENT_ARROW,
                                2, cv2.LINE_AA, tipLength=0.25)

    def _draw_hud(self, img: np.ndarray, frame: Frame,
                  det: PoseDetection,
                  extra_lines: Optional[list],
                  hand_det: Optional[HandDetection] = None) -> None:
        """左上角状态面板"""
        lines = []

        # 人体检测状态
        if det.num_persons == 0:
            lines.append(("未检测到人体", COLOR_BAD))
        else:
            arm_ok = det.arm_complete
            txt = f"人数={det.num_persons} 手臂={'完整' if arm_ok else '不完整'}"
            lines.append((txt, COLOR_OK if arm_ok else COLOR_WARN))

        # 三个关键点置信度
        if det.arm is not None:
            for name, kp in (("肩", det.arm.shoulder),
                             ("肘", det.arm.elbow),
                             ("腕", det.arm.effective_wrist)):
                if kp is None:
                    lines.append((f"{name}: 无", COLOR_BAD))
                else:
                    c = kp.confidence
                    col = COLOR_OK if c >= 0.5 else (COLOR_WARN if c > 0 else COLOR_BAD)
                    lines.append((f"{name}: conf={c:.2f}", col))
            # 明确标出腕部来源，便于确认「已抛弃人体腕部」
            src = det.arm.wrist_source
            src_txt = {"hand": "手部(权威)", "pose": "人体(兜底)", "none": "无"}[src]
            lines.append((f"腕来源: {src_txt}",
                          COLOR_OK if src == "hand" else COLOR_WARN))

        # 几何量（阶段四的输入，先显示便于观察）
        if self.show_metrics and det.arm_complete:
            metrics = compute_arm_metrics(det.arm)
            if metrics:
                for k, v in metrics.items():
                    lines.append((f"{k}: {v}", COLOR_TEXT))

        # ---------------- 手部信息 ----------------
        if hand_det is not None:
            if hand_det.has_hand:
                n_lm = hand_det.hand.num_landmarks
                lines.append((f"手: 已检出 ({n_lm}点)", COLOR_OK))
                geom = hand_det.geometry
                if geom is not None:
                    lines.append((f"掌朝向: {geom.orientation_deg:+.1f}°", COLOR_TEXT))
                    lines.append((f"张开度: {geom.openness:.2f}"
                                  f"{' (握拳)' if geom.is_fist else ''}", COLOR_TEXT))
                    lines.append((f"分散度: {geom.spread:.2f}", COLOR_TEXT))
                    lines.append((f"掌宽: {geom.palm_width:.0f}px", COLOR_TEXT))
            elif (hand_det.hand is not None
                  and hand_det.hand.wrist is not None):
                # 只有腕部基线（手部模型未检出）
                # 注意：hand_det.hand 可能为 None（例如未传入 arm_wrist 且
                # MediaPipe 也没检出），必须先判空再取 .wrist
                lines.append(("手: 仅腕点", COLOR_WARN))
            else:
                lines.append(("手: 未检出", COLOR_BAD))

        lines.append((f"推理 {det.latency_ms:.1f} ms", COLOR_TEXT))

        if extra_lines:
            for t in extra_lines:
                lines.append((t, COLOR_TEXT))

        # 面板背景
        pad, lh = 8, 20
        panel_h = pad * 2 + lh * len(lines)
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (300, panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)

        for i, (text, color) in enumerate(lines):
            y = pad + lh * (i + 1) - 5
            self.text.draw(img, text, (pad, y), color)

        # 帧信息
        info = f"{frame.width}x{frame.height}  frame={frame.frame_id}"
        self.text.draw(img, info, (pad, img.shape[0] - 10), (200, 200, 200))


# ============================================================
# 几何量计算（阶段三只用于显示，阶段四会正式使用）
# ============================================================
def compute_arm_metrics(arm: ArmKeypoints) -> Dict[str, str]:
    """
    计算并格式化手臂的直观几何量。

    注意：这里只做「便于观察」的粗略计算，
    正式的映射几何属于阶段四（retargeting 模块）。
    """
    import math

    if not arm.is_complete:
        return {}

    s, e, w = arm.shoulder, arm.elbow, arm.effective_wrist
    out: Dict[str, str] = {}

    # 上臂方向（图像坐标系，y 向下）
    dx1, dy1 = e.x - s.x, e.y - s.y
    upper_ang = math.degrees(math.atan2(-dy1, dx1))     # 取负号使「向上为正」
    out["上臂角"] = f"{upper_ang:+.1f}°"

    # 前臂方向
    dx2, dy2 = w.x - e.x, w.y - e.y
    fore_ang = math.degrees(math.atan2(-dy2, dx2))
    out["前臂角"] = f"{fore_ang:+.1f}°"

    # 肘部夹角（上臂向量与前臂向量的夹角）
    v1 = np.array([s.x - e.x, s.y - e.y], dtype=np.float64)
    v2 = np.array([w.x - e.x, w.y - e.y], dtype=np.float64)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 > 1e-6 and n2 > 1e-6:
        cosang = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        out["肘夹角"] = f"{math.degrees(math.acos(cosang)):.1f}°"

    # 上臂/前臂像素长度（粗略尺度参考）
    out["上臂长"] = f"{n1:.0f}px"
    out["前臂长"] = f"{n2:.0f}px"

    return out
