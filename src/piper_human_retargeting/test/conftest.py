# -*- coding: utf-8 -*-
"""
pytest 共享夹具
===============
`cfg` / `robot_cfg` 原本只在 test_retargeting.py 里定义，
其它测试文件（如 test_geometry_pivot.py）引用时拿不到。

放在 conftest.py 里由 pytest 自动发现，各文件共享，
避免「同一个夹具复制三份、改一处漏两处」。

⚠️ 必须是**函数级** fixture：
   多个测试会就地修改 cfg.rules / cfg.neutral_joints
   （例如关掉某条映射、改 sign）。若共享同一个实例，
   前一个测试的改动会污染后一个 —— 这个坑实际踩过。
"""

import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_HERE)
_SRC = os.path.dirname(_PKG_ROOT)
for p in (_PKG_ROOT, os.path.join(_SRC, "piper_human_control")):
    if p not in sys.path:
        sys.path.insert(0, p)

from piper_human_control import ControlConfig              # noqa: E402
from piper_human_retargeting import RetargetingConfig      # noqa: E402


# 腕部规则名（依赖手部姿态，手臂类测试通常不提供）
WRIST_RULE_KEYS = ("j5_wrist", "j6_roll")


def disable_wrist_rules(cfg):
    """
    关掉依赖手部姿态的映射规则。

    为什么需要：j5/j6 现在由 wrist_pitch_deg / palm_roll_deg 驱动，
    而这两个量来自手部 21 点。手臂类测试只造 ArmKeypoints、不给手部数据，
    于是腕部量缺失 -> 那些规则走「缺少测量量」分支 -> 整条映射链路被阻塞
    （表现为 deltas 全空、关节不动）。

    手臂测试想验证的是手臂映射，不该被腕部缺失连坐，
    所以显式关掉腕部规则 —— 关掉是**显式**的，不是静默忽略。
    """
    for r in cfg.rules:
        if r.key in WRIST_RULE_KEYS:
            r.enabled = False
    return cfg


@pytest.fixture
def cfg(request):
    """
    默认：完整配置（含腕部规则）。

    用法：`@pytest.mark.parametrize("cfg", [False], indirect=True)`
         传入 False 会关掉腕部规则，供手臂类测试使用。
    """
    c = RetargetingConfig.from_yaml(
        os.path.join(_PKG_ROOT, "config", "retargeting.yaml"))
    # 模块级标记：只有**明确声明**是纯手臂测试的模块才关掉腕部规则。
    # 用标记而不是「自动探测有没有用到 hand_pose」——
    # 后者会在测试重构时静默改变行为，很难查。
    import sys as _sys
    _mod = _sys.modules.get(request.module.__name__)
    if getattr(_mod, "ARM_ONLY_TESTS", False):
        disable_wrist_rules(c)
    return c


@pytest.fixture
def robot_cfg():
    return ControlConfig.from_yaml()

