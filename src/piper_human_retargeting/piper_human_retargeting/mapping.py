# -*- coding: utf-8 -*-
"""
piper_human_retargeting.mapping
================================
人体角度 -> Piper 关节 的**统一映射函数**。

本模块存在的意义：
    映射是整个遥操作里最容易出问题、也最需要反复调参的一环。
    把「公式」收进一个纯函数，好处是：
      * 只有一处实现，不会出现「A 模块这么算、B 模块那么算」；
      * 可以完全脱机单元测试（本模块不依赖 ROS2 / 相机 / 机械臂）；
      * 调参只改 YAML，不动代码。

============================================================
映射公式（区间归一化）
============================================================

    # 1) 相对标定姿态的角度差（消除摄像头安装角、人的位置/身高影响）
    delta = human_angle - human_neutral

    # 2) 归一化到 [0, 1]
    ratio = (delta - human_min) / (human_max - human_min)
    ratio = clamp(ratio, 0, 1)          # clamp=true 时

    # 3) 映射到机械臂行程
    robot = robot_min + ratio * (robot_max - robot_min)

    # 4) 方向翻转（不要在主程序里写 angle = -angle）
    if invert:
        robot = robot_min + robot_max - robot

    # 5) 偏置
    robot = robot + offset

    # 6) 机械臂真实限位（最后一道保险）
    robot = clamp(robot, joint_lower, joint_upper)

============================================================
为什么不是「human_angle = robot_joint」
============================================================
    人体与 Piper 的运动学结构不同（连杆长度、关节顺序、零位定义都不同），
    直接赋值在物理上没有意义，且人体绝对角度受摄像头安装角度影响极大。
    本方案映射的是**相对标定姿态的角度差**，因此：
      * 摄像头装歪了 -> 标定会吸收掉；
      * 人站得远近高低不同 -> 标定会吸收掉；
      * 不同身高的人 -> 标定会吸收掉。

============================================================
与 3D 升级的关系
============================================================
    本模块只负责「一个标量 -> 一个标量」的映射，不关心输入是 2D 角度
    还是 3D 姿态解算出来的角度。将来升级 RGB-D / 3D Retargeting 时，
    只要新的 retargeting 层产出的仍是「人体角度」，本函数可直接复用；
    若升级为「位姿 -> IK」，则替换的是 retargeting 层，
    本模块可以整体退役 —— 这就是把映射独立成一层的目的。
"""

from dataclasses import dataclass
import math
from typing import Dict, Optional, Tuple


# ============================================================
# 单条映射规则
# ============================================================
@dataclass
class RangeMapping:
    """
    一条「人体角度 -> 机械臂关节」的区间映射规则。

    Attributes:
        key:        规则标识（配置里的 j2/j3/j5）
        human:      人体测量量名称，如 "upper_arm_angle_deg"
        joint:      机械臂关节名，如 "joint2"

        human_min:  相对标定姿态的人体角度**下界**（度）
        human_max:  相对标定姿态的人体角度**上界**（度）
        robot_min:  该关节在下界处对应的角度（弧度）
        robot_max:  该关节在上界处对应的角度（弧度）

        invert:     方向翻转。等价于过去的 sign=-1，
                    但**统一在这里处理**，不在主程序里散落负号。
        sign:       人体测量量本身的**符号修正**（+1 / -1），在做差之前施加。
                    与 invert 的区别很重要：
                      sign  修正「这个人体量增大到底代表什么」——例如
                            「手往下移、前臂角反而变大」时，取 -1 让
                            「手抬高 = 数值变大」成立；
                      invert 修正「机械臂关节该往哪转」。
                    两者作用在不同环节，混在一起会很难调。
        offset:     额外偏置（弧度）
        clamp:      是否把 ratio 夹到 [0,1]（推荐 true，防止超出行程外推）
        enabled:    是否启用

        joint_lower / joint_upper: 机械臂真实限位（弧度），最后一道保险。
                    由调用方从 joint_limits.yaml 注入，
                    避免本模块自己再维护一份限位表（会不一致）。
    """
    key: str
    human: str
    joint: str

    human_min: float = -180.0
    human_max: float = 180.0
    robot_min: float = 0.0
    robot_max: float = 0.0

    invert: bool = False
    sign: float = 1.0
    offset: float = 0.0

    # 该测量量的单帧突变门限（度）。0 = 关闭。
    # 必须**按量分别设置**：腕俯仰的假跳变可达 353°（关键点误估到对侧），
    # 而掌心滚转本身就是个会跨 ±180° 的量、真实单帧变化就能到 100°+。
    # 用一个全局门限必然误伤其中之一。
    max_jump: float = 0.0
    clamp: bool = True
    enabled: bool = True

    joint_lower: float = -math.pi
    joint_upper: float = math.pi

    # ------------------------------------------------------------
    @property
    def human_span(self) -> float:
        return self.human_max - self.human_min

    @property
    def robot_span(self) -> float:
        return self.robot_max - self.robot_min

    @property
    def scale(self) -> float:
        """
        等效比例系数（rad per degree）。

        由区间自动推导：robot_span / human_span。
        之所以保留这个属性：调试与文档里用单一数字描述「灵敏度」更直观。
        """
        if abs(self.human_span) < 1e-12:
            return 0.0
        return self.robot_span / self.human_span

    # ------------------------------------------------------------
    def map(self, human_delta_deg: float) -> Tuple[float, Dict[str, float]]:
        """
        把「相对标定姿态的人体角度差」映射为关节角。

        Args:
            human_delta_deg: human_angle - human_neutral，单位**度**

        Returns:
            (关节角(rad), 中间量字典)
            中间量便于调试与 CSV 记录：ratio / raw / after_invert /
            after_offset / before_limit
        """
        info: Dict[str, float] = {}

        # 1) 归一化
        if abs(self.human_span) < 1e-12:
            ratio = 0.5
        else:
            ratio = (human_delta_deg - self.human_min) / self.human_span
        info["ratio_raw"] = ratio

        if self.clamp:
            ratio = max(0.0, min(1.0, ratio))
        info["ratio"] = ratio

        # 2) 映射到机械臂行程
        raw = self.robot_min + ratio * self.robot_span
        info["after_map"] = raw

        # 3) 方向翻转（统一在这里，不在主程序散落负号）
        if self.invert:
            raw = self.robot_min + self.robot_max - raw
        info["after_invert"] = raw

        # 4) 偏置
        raw = raw + self.offset
        info["after_offset"] = raw

        # 5) 机械臂真实限位（最后一道保险）
        info["before_limit"] = raw
        out = max(self.joint_lower, min(self.joint_upper, raw))
        info["limited"] = (out != raw)

        return out, info

    # ------------------------------------------------------------
    def validate(self) -> None:
        """配置合法性校验，启动即失败优于运行中出错"""
        if self.human_max <= self.human_min:
            raise ValueError(
                f"配置错误: {self.key}.human_max({self.human_max}) "
                f"必须大于 human_min({self.human_min})")
        if self.robot_max == self.robot_min and abs(self.offset) < 1e-12:
            raise ValueError(
                f"配置错误: {self.key} 的 robot_min == robot_max 且无 offset，"
                f"该关节永远不会动")
        if self.sign not in (1.0, -1.0):
            raise ValueError(
                f"配置错误: {self.key}.sign 只能是 +1 或 -1，收到 {self.sign}")
        if self.joint_lower >= self.joint_upper:
            raise ValueError(
                f"配置错误: {self.key} 的关节限位区间非法 "
                f"[{self.joint_lower}, {self.joint_upper}]")
        # 映射结果若完全落在限位外，说明配置写错了
        lo = min(self.robot_min, self.robot_max) + self.offset
        hi = max(self.robot_min, self.robot_max) + self.offset
        if hi < self.joint_lower or lo > self.joint_upper:
            raise ValueError(
                f"配置错误: {self.key} 的映射区间 [{self.robot_min}, "
                f"{self.robot_max}] + offset({self.offset}) 完全落在关节限位 "
                f"[{self.joint_lower}, {self.joint_upper}] 之外")


# ============================================================
# 统一入口
# ============================================================
def map_human_to_robot(
    human_delta_deg: float,
    rule: RangeMapping,
) -> float:
    """
    **统一映射入口**：人体相对角度 -> 机械臂关节角。

    这是全项目唯一的「人体角度 -> 关节」转换函数。
    不要在别处再写 `joint = a * angle + b`。

    Args:
        human_delta_deg: human_angle - human_neutral（度）
        rule:            该关节的映射规则（来自 YAML）

    Returns:
        关节角（弧度），已 clamp 到机械臂真实限位
    """
    out, _ = rule.map(human_delta_deg)
    return out


def map_all(
    human_deltas_deg: Dict[str, float],
    rules,
    base_joints: Dict[str, float],
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """
    批量映射多个关节。

    Args:
        human_deltas_deg: {"upper_arm_angle_deg": 12.3, ...}  相对角(度)
        rules:            可迭代的 RangeMapping
        base_joints:      未受控关节的固定值 {"joint1": 0.0, ...}

    Returns:
        (关节角字典, 每个关节的中间量)
        关节角字典已包含 base_joints 里那些固定关节。
    """
    out = dict(base_joints)
    infos: Dict[str, Dict[str, float]] = {}

    for rule in rules:
        if not rule.enabled:
            continue
        delta = human_deltas_deg.get(rule.human)
        if delta is None:
            # 缺少该测量量：保守处理 —— 该关节保持不动
            # （不猜、不填 0，避免把错误值当真值下发）
            continue
        value, info = rule.map(delta)
        out[rule.joint] = value
        infos[rule.joint] = info

    return out, infos
