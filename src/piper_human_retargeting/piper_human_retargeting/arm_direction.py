# -*- coding: utf-8 -*-
"""
piper_human_retargeting.arm_direction
=====================================
「手臂方向」驱动的映射 —— 替代逐关节角度映射。

为什么需要它
============
逐关节映射（人上臂角->joint2、肘夹角->joint3）有个根本问题：
**它不保证机械臂朝人手臂的方向伸出去**。

原因是 Piper 的关节是耦合的（实测 FK）：
    joint2 同时抬高与前推末端
    joint3 同时改变肘弯与伸展
于是「joint3 伸得更直」不等于「机械臂往前伸」——
用户反馈「手臂伸直时机械臂没有向前伸直」正是这个耦合造成的。

用户的要求是**方向性**的：
    手臂向前伸直 -> 机械臂向前伸直
    手臂向上伸直 -> 机械臂向上伸直
    「更注重腕部相对于肩部的相对角度」

所以本模块换一种控制思路：
    1. 从人体取「肩 -> 腕」的**方向角**作为主控量；
    2. 从 FK 表里反解出能让机械臂「肩 -> 腕」指向同一方向的 joint2/joint3。

这样方向是**直接对齐**的，不受关节耦合影响。

FK 表
=====
`fk_table.json` 由 `tools/build_fk_table.py` 在仿真里扫描生成，
每项为 {"j2","j3","dir","reach"}：
    dir   机械臂「肩(link2) -> 腕(link4)」在矢状面内的方向角（度）
          0° = 正前方，+90° = 正上方
    reach 同一向量的长度（米）
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class FkEntry:
    j2: float
    j3: float
    dir_deg: float
    reach: float


class ArmDirectionMapper:
    """
    用 FK 表把「目标方向角 + 目标伸展」反解成 (joint2, joint3)。

    Args:
        table_path: FK 表 JSON 路径
        dir_weight: 方向误差权重（度）；伸展误差权重固定为 1 米
        reach_weight: 伸展误差权重（米）
    """

    def __init__(self, table_path: str,
                 dir_weight: float = 1.0,
                 reach_weight: float = 60.0):
        self.table_path = table_path
        self.dir_weight = float(dir_weight)
        self.reach_weight = float(reach_weight)
        self.table: List[FkEntry] = []
        self._load()

    # ------------------------------------------------------------
    def _load(self) -> None:
        if not os.path.isfile(self.table_path):
            raise FileNotFoundError(f"FK 表不存在: {self.table_path}")
        with open(self.table_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.table = [FkEntry(float(d["j2"]), float(d["j3"]),
                              float(d["dir"]), float(d["reach"]))
                      for d in raw]
        if not self.table:
            raise ValueError(f"FK 表为空: {self.table_path}")

    # ------------------------------------------------------------
    @property
    def dir_range(self) -> Tuple[float, float]:
        d = [e.dir_deg for e in self.table]
        return (min(d), max(d))

    @property
    def reach_range(self) -> Tuple[float, float]:
        r = [e.reach for e in self.table]
        return (min(r), max(r))

    # ------------------------------------------------------------
    def solve(self, dir_deg: float, reach: Optional[float] = None
              ) -> Optional[Tuple[float, float]]:
        """
        反解 (joint2, joint3)。

        Args:
            dir_deg: 目标方向角（度）
            reach:   目标伸展（米）。None 表示只对齐方向，
                     此时在同方向里取**伸展最大**的解
                     （即「朝着那个方向尽量伸直」）。

        Returns:
            (joint2, joint3)；表为空时 None。
        """
        if not self.table:
            return None

        best = None
        for e in self.table:
            # 方向误差本身是周期量，但本表范围在 (-180,180] 内连续
            d_err = abs(e.dir_deg - dir_deg)
            if reach is None:
                # 只对齐方向：同方向里优先伸展更大者，
                # 用一个很小的惩罚把「更伸展」排在前面
                cost = d_err * self.dir_weight - e.reach * 1e-3
            else:
                cost = (d_err * self.dir_weight
                        + abs(e.reach - reach) * self.reach_weight)
            if best is None or cost < best[0]:
                best = (cost, e)
        if best is None:
            return None
        return (best[1].j2, best[1].j3)

    # ------------------------------------------------------------
    def nearest_dir(self, dir_deg: float) -> float:
        """表里最接近该方向的角度（用于判断是否超出可达范围）"""
        return min((e.dir_deg for e in self.table),
                   key=lambda d: abs(d - dir_deg))


def wrap180(a: float) -> float:
    """角度归一化到 (-180, 180]"""
    d = (a + 180.0) % 360.0 - 180.0
    return d


def arm_direction_deg(shoulder, wrist) -> Optional[float]:
    """
    人体「肩 -> 腕」方向角（度）。

    坐标约定与 arm_geometry 一致：y 翻转，「向上为正」。
        0°   = 正右（画面里手指向右侧）
        +90° = 正上
    """
    if shoulder is None or wrist is None:
        return None
    if not (shoulder.is_valid and wrist.is_valid):
        return None
    dx = wrist.x - shoulder.x
    dy = wrist.y - shoulder.y
    if math.hypot(dx, dy) < 1e-6:
        return None
    return math.degrees(math.atan2(-dy, dx))
