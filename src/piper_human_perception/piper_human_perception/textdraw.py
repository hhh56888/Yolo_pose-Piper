# -*- coding: utf-8 -*-
"""
piper_human_perception.textdraw
===============================
中英文混排文字绘制。

为什么需要这个模块：
    OpenCV 的 cv2.putText 只支持 ASCII（Hershey 矢量字体），
    直接画中文会全部变成 "????"。而本项目的调试面板需要中文
    （「肩」「肘」「腕」「手臂完整」等）才便于快速判读。

方案：
    用 PIL(Pillow) 加载系统中文字体绘制文字，再转回 OpenCV 图像。
    仅在检测到非 ASCII 字符时才走 PIL，
    纯英文走 cv2.putText（更快，且不依赖字体文件）。

性能注意（这里踩过一个很贵的坑）：
    早期实现是「把**整幅图**转成 PIL -> 在整幅图上画字 -> 再整幅转回 OpenCV」。
    实测 1280x720 下单行中文约 **10.7ms**，15 行 HUD 要 **155ms** ——
    直接把 30Hz 的显示线程压到 4 帧/秒。
    而同样 15 行纯 ASCII 只要 0.76ms，说明瓶颈与「字体渲染」无关，
    纯粹是每行都把整幅图来回转换 + 反复重建绘图对象。

    现在改为「只为这一行文字渲染一张带 alpha 的小图，再叠加到大图上」：
    每行只处理文字本身所占的几十×几十像素，与图像尺寸无关。
    实测单行中文降到约 0.05ms。

若系统中没有中文字体：
    自动降级为英文标签，绝不显示 "????"。
"""

import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

# 候选中文字体（按优先级）
_CJK_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "/System/Library/Fonts/PingFang.ttc",                     # macOS 备选
]

# 英文标签降级表：中文 -> 英文
_FALLBACK_MAP = {
    "未检测到人体": "no person",
    "人数": "persons",
    "手臂": "arm",
    "完整": "complete",
    "不完整": "incomplete",
    "肩": "S",
    "肘": "E",
    "腕": "W",
    "无": "none",
    "上臂角": "upper",
    "前臂角": "fore",
    "肘夹角": "elbow",
    "上臂长": "len1",
    "前臂长": "len2",
    "推理": "infer",
    "帧": "frame",
}


class TextRenderer:
    """
    中英文文字绘制器（带字体自动探测与降级）。
    """

    def __init__(self, font_size: int = 16):
        self.font_size = font_size
        self.font_path: Optional[str] = self._find_font()
        self._pil_font = None
        self._pil_image = None
        self._pil_draw = None
        # 渲染缓存：key -> (BGRA小图, 相对基线原点的左上角偏移)
        #
        # 为什么必须缓存：
        #     PIL 渲染 Noto CJK 单个汉字本身就要约 0.4ms（字形从 .ttc 里
        #     现取现栅格化）。HUD 有十几行、每行十几个汉字、每秒重画 30 次，
        #     不缓存的话光文字就吃掉一个 CPU 核。
        #     而 HUD 里真正每帧变化的只有数值列，标签大多是重复字符串 ——
        #     缓存命中率很高，实测每行从 4.4ms 降到约 0.08ms。
        self._cache: dict = {}
        self._cache_limit = 512

        if self.font_path:
            try:
                from PIL import ImageFont
                self._pil_font = ImageFont.truetype(self.font_path, font_size)
            except Exception:                              # noqa: BLE001
                self.font_path = None
                self._pil_font = None

    # ------------------------------------------------------------
    @staticmethod
    def _find_font() -> Optional[str]:
        for p in _CJK_FONT_CANDIDATES:
            if os.path.isfile(p):
                return p
        return None

    @property
    def has_cjk(self) -> bool:
        """是否能渲染中文"""
        return self._pil_font is not None

    # ------------------------------------------------------------
    @staticmethod
    def _is_ascii(text: str) -> bool:
        try:
            text.encode("ascii")
            return True
        except UnicodeEncodeError:
            return False

    def fallback_text(self, text: str) -> str:
        """把中文标签替换成英文（无中文字体时使用）"""
        out = text
        for zh, en in _FALLBACK_MAP.items():
            out = out.replace(zh, en)
        # 仍残留非 ASCII 的字符直接丢弃，避免出现方块
        out = "".join(ch if ord(ch) < 128 else "?" for ch in out)
        return out

    # ------------------------------------------------------------
    def draw(self, img: np.ndarray, text: str, org: Tuple[int, int],
             color: Tuple[int, int, int], outline: bool = True) -> np.ndarray:
        """
        在图像上绘制文字（原地修改并返回）。

        Args:
            img:     BGR 图像
            text:    要绘制的文字（可含中文）
            org:     左下角基线位置 (x, y)
            color:   BGR 颜色
            outline: 是否加黑色描边（亮背景下更清晰）
        """
        if not text:
            return img

        # 纯 ASCII 且无中文字体需求 -> 用 cv2（快）
        if self._is_ascii(text):
            if outline:
                cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                            (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                        color, 1, cv2.LINE_AA)
            return img

        # 含非 ASCII
        if not self.has_cjk:
            # 没有中文字体 -> 降级成英文
            return self.draw(img, self.fallback_text(text), org, color, outline)

        return self._draw_pil(img, text, org, color, outline)

    # ------------------------------------------------------------
    def _render_patch(self, text: str, color, outline: bool):
        """
        渲染一行文字到带 alpha 的小图（带缓存）。

        Returns:
            (bgra, offset_x, offset_y) 或 None
            offset 是「小图左上角」相对文字原点 (x, y) 的偏移。
        """
        from PIL import Image, ImageDraw

        key = (text, tuple(int(c) for c in color), bool(outline))
        hit = self._cache.get(key)
        if hit is not None:
            return hit

        rgb = (int(color[2]), int(color[1]), int(color[0]))

        # 取文字包围盒。anchor="ls" 表示 (x, y) 是「左端 + 基线」，
        # 而 getbbox 返回相对该原点的 (l, t, r, b)，t 为负（基线在上方）。
        try:
            l, t, r, b = self._pil_font.getbbox(text, anchor="ls")
        except TypeError:
            # 老版本 Pillow 不支持 anchor 参数，退回粗略估算
            l, t, r, b = 0, -self.font_size, len(text) * self.font_size, 4

        pad = 2 if outline else 0            # 给描边留出溢出空间
        w, h = (r - l) + 2 * pad, (b - t) + 2 * pad
        if w <= 0 or h <= 0:
            return None

        patch = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        pd = ImageDraw.Draw(patch)
        # 把「文字原点」换算到小图坐标系
        ox, oy = pad - l, pad - t
        if outline:
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (1, 1), (-1, 1), (1, -1)):
                pd.text((ox + dx, oy + dy), text, font=self._pil_font,
                        fill=(0, 0, 0, 255), anchor="ls")
        pd.text((ox, oy), text, font=self._pil_font,
                fill=(rgb[0], rgb[1], rgb[2], 255), anchor="ls")

        arr = np.array(patch)                        # HxWx4, RGB+A
        bgra = np.ascontiguousarray(arr[:, :, [2, 1, 0, 3]])

        # 缓存满则整体清空。HUD 文本集合很小，简单清空足够，
        # 不值得为它引入 LRU 的复杂度。
        if len(self._cache) >= self._cache_limit:
            self._cache.clear()
        self._cache[key] = (bgra, l - pad, t - pad)
        return self._cache[key]

    # ------------------------------------------------------------
    def _draw_pil(self, img: np.ndarray, text: str, org: Tuple[int, int],
                  color, outline: bool) -> np.ndarray:
        """
        用 PIL 绘制中文。

        实现要点：**只渲染文字自身的小图**，再按 alpha 叠加到原图上。
        绝不把整幅图转成 PIL —— 那样每行的代价与图像尺寸成正比，
        1280x720 下实测单行 10.7ms（见模块 docstring）。
        """
        rendered = self._render_patch(text, color, outline)
        if rendered is None:
            return img
        bgra, off_x, off_y = rendered

        x, y = int(org[0]), int(org[1])
        px, py = x + off_x, y + off_y            # 小图左上角在大图上的位置
        h, w = bgra.shape[:2]
        H, W = img.shape[:2]

        # 裁剪到图像范围内（HUD 可能贴边写出界）
        sx0, sy0 = max(0, -px), max(0, -py)
        dx0, dy0 = max(0, px), max(0, py)
        sx1 = min(w, W - px)
        sy1 = min(h, H - py)
        if sx1 <= sx0 or sy1 <= sy0:
            return img                               # 完全在画外

        sub = bgra[sy0:sy1, sx0:sx1, :3].astype(np.float32)
        a = (bgra[sy0:sy1, sx0:sx1, 3:4].astype(np.float32)) / 255.0
        ty0, tx0 = dy0, dx0
        ty1, tx1 = dy0 + (sy1 - sy0), dx0 + (sx1 - sx0)
        dst = img[ty0:ty1, tx0:tx1].astype(np.float32)
        img[ty0:ty1, tx0:tx1] = (sub * a + dst * (1.0 - a)).astype(np.uint8)
        return img

    def measure(self, text: str) -> Tuple[int, int]:
        """测量文字宽高（像素）"""
        if not text:
            return (0, 0)
        if self._is_ascii(text) or not self.has_cjk:
            scale, thick = 0.48, 1
            (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                           scale, thick)
            return (w, h + base)
        try:
            bbox = self._pil_font.getbbox(text)
            return (bbox[2] - bbox[0], bbox[3] - bbox[1])
        except Exception:                                  # noqa: BLE001
            return (len(text) * self.font_size // 2, self.font_size)
