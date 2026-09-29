# -*- coding: utf-8 -*-
"""
关节限位三层结构测试（limits.py）
=================================
    physical（URDF/实际加载） ⊇ safe（留 margin） ⊇ retarget（重映射输出）

动机（审计实测）：
    * joint3 的映射上端 robot_max=0.00 **正好等于**物理上限 -> live 15.25% 骑限位；
    * joint2 的映射下端 robot_min=−1.00 **小于**物理下限 0.00 -> 29.67% 被削平。
三层结构要在**配置校验期**把这类问题顶出来，而不是让它变成"到某个方向就顶住"。

运行：
    python3 -m pytest test/test_limits.py -q -p no:anyio
"""
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_HERE)
_SRC = os.path.dirname(_PKG_ROOT)
for _p in (_PKG_ROOT, os.path.join(_SRC, "piper_human_control")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from piper_human_retargeting.limits import (              # noqa: E402
    DEFAULT_MARGIN_RAD, LimitLayer, build_layers, physical_limits,
    safe_limits)


class TestPhysical:
    def test_from_urdf_with_cross_check(self):
        phys, src = physical_limits()
        assert phys["joint2"] == (0.0, 3.14)
        assert phys["joint3"] == (-2.967, 0.0)
        assert "URDF" in src
        assert "一致" in src or "不可用" in src

    def test_physical_limits_are_ordered(self):
        phys, _ = physical_limits()
        for j, (lo, hi) in phys.items():
            assert lo < hi, j


class TestSafe:
    def test_margin_applied_on_both_ends(self):
        phys = {"joint3": (-2.967, 0.0)}
        safe = safe_limits(phys, 0.10)
        assert safe["joint3"] == pytest.approx((-2.867, -0.10))

    def test_safe_is_inside_physical(self):
        phys, _ = physical_limits()
        safe = safe_limits(phys, DEFAULT_MARGIN_RAD)
        for j in safe:
            assert phys[j][0] < safe[j][0] < safe[j][1] < phys[j][1], j

    def test_too_narrow_range_rejected(self):
        with pytest.raises(ValueError):
            safe_limits({"jointX": (0.0, 0.1)}, 0.10)


class TestNesting:
    def test_valid_nesting_has_no_violation(self):
        layer = build_layers({"joint2": (0.25, 2.89), "joint3": (-2.72, -0.25)})
        assert layer.violations == []

    def test_retarget_exceeding_physical_is_reported(self):
        """joint2 映射下端 −1.00 小于物理下限 0.0（legacy 的真实问题）"""
        layer = build_layers({"joint2": (-1.00, 2.20), "joint3": (-1.20, 0.00)})
        assert layer.violations
        assert any("joint2" in v for v in layer.violations)
        # joint3 的 robot_max 正好压在物理上限上，同样算越界（safe 更窄）
        assert any("joint3" in v for v in layer.violations)

    def test_retarget_inside_physical_but_outside_safe_is_reported(self):
        """即使没越 physical，越出 safe 也要报（safe 才是允许的输出范围）"""
        layer = build_layers({"joint2": (0.05, 2.0)}, margin=0.10)
        assert layer.violations and "safe" in layer.violations[0]

    def test_margins_helper(self):
        layer = build_layers({"joint2": (0.25, 2.89), "joint3": (-2.72, -0.25)})
        m = layer.margins({"joint2": 1.5, "joint3": -1.0})
        # margins() 给的是到 **safe** 的余量：joint2 [0.10,3.04] joint3 [-2.867,-0.10]
        assert m["joint2"] == pytest.approx((1.40, 1.54))
        assert m["joint3"] == pytest.approx((1.867, 0.90))

    def test_table_and_dict_render(self):
        layer = build_layers({"joint3": (-2.72, -0.25)})
        assert "physical" in layer.table()
        d = layer.as_dict()
        assert d["retarget"]["joint3"] == [-2.72, -0.25]

    def test_legacy_ranges_audit_is_visible(self):
        """
        legacy 的映射区间（配置原值）必须能被审计出来 ——
        本轮不修改 legacy 行为，但问题不能是"看不见的"。
        """
        from piper_human_retargeting import RetargetingConfig
        c = RetargetingConfig.from_yaml()
        ranges = {}
        for r in c.rules:
            if r.joint in ("joint2", "joint3"):
                ranges[r.joint] = (r.robot_min, r.robot_max)
        layer = build_layers(ranges)
        assert layer.violations, "legacy 的越界必须被报出来"
