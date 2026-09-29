# -*- coding: utf-8 -*-
"""
中英文文字绘制器测试
====================
背景（一次很贵的性能事故）：
    早期 `_draw_pil` 把**整幅图**转成 PIL、在整幅图上画字、再整幅转回 OpenCV。
    1280x720 下单行中文约 10.7ms，15 行 HUD 要 155ms ——
    足以把 30Hz 的显示线程压到 4 帧/秒。
    而同样 15 行纯 ASCII 只要 0.76ms，证明瓶颈不在「字体渲染」，
    而在每行都把整幅图来回转换。

修复：
    1) 只为**这一行文字**渲染一张带 alpha 的小图，再叠加到大图；
    2) 对渲染结果做缓存（HUD 里重复字符串很多）。

本测试锁定：
    * 绘制结果正确（中文真被画上去、位置随 org 移动、不越界崩溃）
    * 代价与**图像尺寸无关**（这是修复的核心性质）
    * 代价与**行数近似无关**（缓存生效）

运行：
    python3 -m pytest test/test_textdraw.py -q -p no:anyio
"""

import os
import sys
import time

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_HERE)
_SRC = os.path.dirname(_PKG_ROOT)
for p in (_PKG_ROOT, os.path.join(_SRC, "piper_human_control")):
    if p not in sys.path:
        sys.path.insert(0, p)

from piper_human_perception.textdraw import TextRenderer   # noqa: E402

CJK = "人体角 raw -> filtered:"
ASCII = "RAW upper +156.8 elbow +106.1"


@pytest.fixture
def tr():
    return TextRenderer(font_size=16)


def _canvas(h=720, w=1280):
    return np.full((h, w, 3), 40, np.uint8)


# ============================================================
# 正确性
# ============================================================
def test_cjk_text_is_actually_drawn(tr):
    """中文必须真的被画上去（不是静默失败）"""
    if not tr.has_cjk:
        pytest.skip("系统无中文字体")
    img = _canvas()
    before = img.copy()
    tr.draw(img, CJK, (10, 40), (0, 255, 0))
    assert not np.array_equal(img, before), "画完图像没有任何变化"
    # 应出现明显的绿色像素
    green = (img[:, :, 1].astype(int) - img[:, :, 2].astype(int)) > 60
    assert green.sum() > 50, f"绿色文字像素过少: {green.sum()}"


def test_org_moves_the_text(tr):
    """同一段文字画在不同位置，结果应不同且各自局部有内容"""
    if not tr.has_cjk:
        pytest.skip("系统无中文字体")
    a, b = _canvas(200, 400), _canvas(200, 400)
    tr.draw(a, CJK, (10, 40), (0, 255, 0))
    tr.draw(b, CJK, (10, 140), (0, 255, 0))
    assert not np.array_equal(a, b), "改变 org 后图像完全相同，说明位置没生效"
    top_a = a[:80].sum()
    top_b = b[:80].sum()
    assert top_a != top_b


def test_drawing_off_canvas_does_not_crash(tr):
    """
    HUD 可能被写出边界（负坐标 / 超出右下角）。
    越界必须是安全的裁剪，不能抛异常、不能整幅乱涂。
    """
    if not tr.has_cjk:
        pytest.skip("系统无中文字体")
    img = _canvas(100, 200)
    for org in ((-500, -500), (190, 90), (10_000, 10_000), (-5, 50), (195, 50)):
        tr.draw(img, CJK, org, (0, 255, 0))       # 不抛异常即通过


def test_empty_text_is_noop(tr):
    img = _canvas(50, 100)
    before = img.copy()
    tr.draw(img, "", (10, 20), (0, 255, 0))
    assert np.array_equal(img, before)


def test_ascii_uses_cv2_path(tr):
    """纯 ASCII 应走 cv2.putText（不依赖字体文件）"""
    img = _canvas(60, 400)
    before = img.copy()
    tr.draw(img, ASCII, (10, 30), (255, 255, 255))
    assert not np.array_equal(img, before)


def test_fallback_text_has_no_non_ascii(tr):
    """无中文字体时的降级路径：产出必须全是 ASCII"""
    out = tr.fallback_text("状态: 追踪中 上臂")
    assert all(ord(c) < 128 for c in out), f"降级后仍有非 ASCII: {out!r}"


# ============================================================
# 性能契约 —— 本模块最关键的性质
# ============================================================
def _time_draw(tr, img, lines, n=30):
    tr.draw(img.copy(), lines[0], (10, 30), (0, 255, 0))   # 预热/填充缓存
    t0 = time.monotonic()
    for _ in range(n):
        im = img.copy()
        for i, ln in enumerate(lines):
            tr.draw(im, ln, (10, 30 + i * 20), (0, 255, 0))
    return (time.monotonic() - t0) / n * 1000.0


def test_cost_is_independent_of_image_size(tr):
    """
    修复的核心性质：绘制一行的代价必须与**图像尺寸**无关。

    旧实现把整幅图转 PIL 再转回来，代价与像素数成正比：
        1280x720 -> 约 10.7ms/行
        3840x2160 -> 约 16 倍
    新实现只处理文字小图，大图小图应当同量级。
    """
    if not tr.has_cjk:
        pytest.skip("系统无中文字体")
    small = _time_draw(tr, _canvas(720, 1280), [CJK])
    large = _time_draw(tr, _canvas(2160, 3840), [CJK])
    # 允许 4 倍余量；旧实现这里会是 9 倍以上
    assert large < small * 4 + 1.0, (
        f"绘制代价随图像尺寸显著增长: 1280x720={small:.3f}ms "
        f"3840x2160={large:.3f}ms")


def test_many_lines_are_cheap_thanks_to_cache(tr):
    """
    15 行 HUD 的总代价必须远低于「每行都重新渲染字形」的线性代价。

    实测：修复前 15 行 155ms；缓存后约 1ms。
    这里给到 25ms 的宽松上限 —— 只要能满足 30Hz（33ms 预算）即可，
    不把测试绑死在某一台机器的绝对性能上。
    """
    if not tr.has_cjk:
        pytest.skip("系统无中文字体")
    ms = _time_draw(tr, _canvas(720, 1280), [CJK] * 15)
    assert ms < 25.0, f"15 行 HUD 耗时 {ms:.1f}ms，30Hz 预算(33ms)会被吃掉"


def test_cache_is_bounded(tr):
    """缓存不能无限增长（HUD 数值每帧都变，唯一字符串会很多）"""
    if not tr.has_cjk:
        pytest.skip("系统无中文字体")
    img = _canvas(200, 800)
    for i in range(tr._cache_limit + 200):
        tr.draw(img, f"编号{i}", (10, 30), (0, 255, 0))
    assert len(tr._cache) <= tr._cache_limit, \
        f"缓存超出上限: {len(tr._cache)}"


def test_cache_is_keyed_by_colour(tr):
    """不同颜色必须各自缓存，不能串色"""
    if not tr.has_cjk:
        pytest.skip("系统无中文字体")
    a, b = _canvas(60, 400), _canvas(60, 400)
    tr.draw(a, CJK, (10, 30), (0, 255, 0))
    tr.draw(b, CJK, (10, 30), (0, 0, 255))
    assert not np.array_equal(a, b), "换颜色后渲染结果相同，缓存键漏了颜色"
