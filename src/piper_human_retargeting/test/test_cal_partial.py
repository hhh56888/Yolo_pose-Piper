# -*- coding: utf-8 -*-
"""标定采样必须逐量：缺一个量不能连累其它量"""
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_HERE)
_SRC = os.path.dirname(_PKG_ROOT)
for _p in (_PKG_ROOT, os.path.join(_SRC, "piper_human_control")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from piper_human_retargeting import Retargeter            # noqa: E402
from piper_human_retargeting.arm_geometry import ArmGeometry  # noqa: E402


class TestPartialCalibrationSampling:
    def test_missing_hand_measures_do_not_block_arm_calibration(self, cfg, robot_cfg):
        """
        实测回归：task_space 模式下 used_measures 含人手量，
        用合成手臂（无手部数据）标定时，若"缺一个就整条丢弃"，
        样本数恒为 0 -> 标定失败 -> task_space 完全跑不起来。
        """
        cfg.retarget_mode = "task_space"
        rt = Retargeter(cfg, robot_cfg)
        rt.start_calibration()
        g = ArmGeometry(0.0, 0.0, 90.0, 100, 100, True)
        for _ in range(20):
            assert rt.add_calibration_sample(g)
        assert rt.finish_calibration()
        for m in ("human_u", "human_v", "elbow_angle_deg"):
            assert rt.neutral.get(m) is not None

    def test_invalid_geometry_is_rejected(self, cfg, robot_cfg):
        """几何无效的帧必须拒绝（不能采到"空样本"）"""
        rt = Retargeter(cfg, robot_cfg)
        rt.start_calibration()
        assert not rt.add_calibration_sample(ArmGeometry(valid=False))
        assert not rt.finish_calibration()          # 无样本 -> 标定失败
