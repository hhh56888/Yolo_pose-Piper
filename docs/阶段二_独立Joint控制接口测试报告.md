# 阶段二测试报告：独立 Joint 控制接口

> 项目：基于 YOLO Pose 的人体手臂动作识别与 Piper 机械臂仿真控制
> 远程主机：`zxmy@100.89.121.96`（Tailscale）
> 工作空间：`~/Yolo_pose+piper`
> 新增包：`piper_human_control`（独立包，**未修改 piper_ros 任何文件**）
> 报告日期：2026-09-23

---

## 0. 结论摘要

```
阶段二：通过
```

新增独立控制包 `piper_human_control`，对外提供
`robot.send_joint_target([q1..q6])` 接口，具备限位、速度/加速度限制、
合法性检查、平滑连续发送、状态读取与软件急停能力。

验收测试 **13/13 通过**，位置精度 **0.0013 rad (0.07°)**，
单元测试 **25/25 通过**。

---

## 1. 交付物

### 1.1 新增包（完全独立，不侵入官方项目）

```
~/Yolo_pose+piper/src/
├── piper_ros/                    ← AgileX 官方，零改动
└── piper_human_control/          ← 本阶段新增
    ├── package.xml
    ├── setup.py
    ├── config/
    │   └── joint_limits.yaml     ★ 所有控制参数的唯一来源
    ├── resource/piper_human_control
    ├── scripts/
    │   ├── joint_test            ros2 run 入口启动器
    │   └── joint_teleop
    ├── test/
    │   └── test_safety.py        25 个单元测试（不依赖 ROS2）
    └── piper_human_control/
        ├── __init__.py           对外 API 与惰性导入
        ├── types.py              数据结构（Keypoint / JointTarget / JointState）
        ├── config.py             配置加载与校验
        ├── safety.py             限位 / 速率限制 / 合法性 / 急停
        ├── streaming.py          轨迹流式发送
        ├── controller.py         ★ 编排层，对外主接口
        ├── joint_test_node.py    验收测试程序
        └── joint_teleop_node.py  键盘手动控制
```

### 1.2 对外接口

```python
from piper_human_control import ControlConfig, PiperJointController

cfg   = ControlConfig.from_yaml()
robot = PiperJointController(cfg)
robot.start()

robot.send_joint_target([q1, q2, q3, q4, q5, q6])   # 主接口
robot.move_to_home()                                 # 回 HOME
robot.hold_current()                                 # 保持当前位置
robot.get_joint_positions()                          # 读当前关节角
robot.get_desired_target()                           # 读期望目标
robot.max_target_error()                             # 读到位误差
robot.wait_until_reached(tol=0.02, timeout=15.0)     # 阻塞等待到位
robot.emergency_stop("原因")                          # 软件急停
robot.reset_emergency_stop()                         # 解除（需显式调用）
```

### 1.3 运行方式

```bash
cd ~/Yolo_pose+piper
source /opt/ros/humble/setup.bash && source install/setup.bash

ros2 run piper_human_control joint_test      # 验收测试
ros2 run piper_human_control joint_teleop    # 键盘手动控制
```

`joint_teleop` 按键：`q/a` 选关节、`w/s` 增减、`h` 回 HOME、
`space` 保持、`e` 急停/解除、`x` 退出。

---

## 2. 架构与解耦

严格按任务书「感知 / 姿态计算 / 运动映射 / 机器人控制相互解耦」设计：

| 模块 | 职责 | 依赖 ROS2 |
|---|---|---|
| `types.py` | 数据结构定义 | ❌ 无 |
| `config.py` | 参数加载与校验 | ❌ 无（可选 ament_index） |
| `safety.py` | 限位、速率限制、急停（纯函数式） | ❌ 无 |
| `streaming.py` | 轨迹编码与发布 | ✅ |
| `controller.py` | 编排：状态缓存 + 时钟 + 调用上述模块 | ✅ |

好处：
- `safety.py` / `config.py` / `types.py` **可脱离 ROS2 单元测试**（已实现，25 项）；
- 阶段四的映射模块只需产出 `[q1..q6]`，调用 `send_joint_target()` 即可，
  完全不接触 ROS2 细节；
- `types.py` 中的 `Keypoint` 已预留 `z=None` 字段，阶段三/七无需改动数据结构。

---

## 3. 配置参数（config/joint_limits.yaml）

所有参数集中在配置文件，**主程序零硬编码**。

| 分组 | 参数 | 值 | 说明 |
|---|---|---|---|
| joints | names | joint1~joint6 | 顺序须与 arm_controller 一致 |
| | limits | 见阶段一报告 | 官方 piper_sdk 限位 |
| | velocity_limits | 1.0~1.5 rad/s | 控制用（URDF 标称 5，取保守值） |
| | acceleration_limits | 2.0~3.0 rad/s² | 用于速率限制 |
| home | — | `[0, 0.6, -0.6, 0, 0, 0]` | 满足 j2≥0、j3≤0 |
| controller | publish_rate_hz | 20.0 | 轨迹重发频率 |
| | trajectory_horizon_s | **0.1** | 关键参数，见 §5 |
| | state_timeout_s | 0.5 | 状态丢失判定 |
| safety | max_step_per_cycle_rad | 0.15 | 单周期硬上限（兜底） |
| | max_publish_failures | 10 | 连续失败进入急停 |
| | lost_tracking_hold_s | 1.0 | 丢失后保持（**不回 HOME**） |

配置在加载时即校验（长度、上下限顺序、HOME 是否越界），
错误在启动阶段暴露，而不是运行到一半才崩。

---

## 4. 安全机制

| 机制 | 实现 | 验证结果 |
|---|---|---|
| 位置限位 | clip 到官方限位 | ✅ j2=99→3.14, j3=5→0.0 |
| 速度限制 | `v_max * dt` | ✅ |
| 加速度限制 | `a_max * dt` | ✅ |
| 单周期增量硬上限 | `min(a·dt, v·dt, 0.15)` | ✅ 单元测试覆盖 |
| 关节数校验 | ≠6 直接拒绝 | ✅ |
| NaN / Inf 校验 | 拒绝 | ✅ |
| 越界不报错但记录 | `SafetyReport.clamped_joints` | ✅ 日志可见 |
| 软件急停 | 锁定目标、拒绝新指令 | ✅ 漂移 0.00000 rad |
| 连续发布失败保护 | ≥10 次进入急停 | 已实现 |
| 状态丢失 | 保持最后目标，**不自动回 HOME** | 已实现 |
| 急停解除 | 需显式 reset，**不自动恢复运动** | ✅ |

设计取舍说明：
- **越界裁剪 vs 拒绝**：位置越界选择「裁剪 + 明确日志」，
  因为阶段四的人体映射难免短暂越界，直接拒绝会导致机械臂卡住不动；
  而 NaN/Inf/长度错误属于程序缺陷，选择「拒绝」以便暴露问题。
- **丢失跟踪不自动回 HOME**：自动回 HOME 是一次大幅运动，
  在人体快速移动场景下可能造成危险，因此只保持当前位姿。

---

## 5. 关键技术决策：轨迹时间窗

`trajectory_horizon_s` 是最影响控制质量的参数。部署前做了
(horizon × rate) 参数扫描，实测稳态误差：

| horizon | rate | j2 @3s | j2 @10s | 稳态误差 |
|---|---|---|---|---|
| **0.1 s** | **10 Hz** | 1.5000 | 1.5000 | **0.0000** |
| 0.1 s | 20 Hz | 1.5000 | 1.5000 | 0.0001 |
| 0.2 s | 20 Hz | 1.5000 | 1.5000 | 0.0001 |
| 0.5 s | 20 Hz | 1.4963 | 1.5000 | 0.0003 |
| 1.0 s | 10 Hz | 1.4341 | 1.4999 | 0.0003 |

结论：**时间窗取小值**（0.1 s），配合 20 Hz 重发。
时间窗取 1.0 s 时 3 秒才走到 1.434（目标 1.5），滞后明显。

最终配置：`horizon=0.1s` + `publish_rate=20Hz`。

---

## 6. 验收测试结果

### 6.1 测试序列（任务书要求）

`HOME → 修改 J2 → 修改 J3 → 修改 J5 → 回 HOME`

### 6.2 测试汇总（13/13 通过）

| # | 测试项 | 结果 | 最大误差 |
|---|---|---|---|
| 1 | 回 HOME | ✅ PASS | 0.0013 rad (0.07°) |
| 2 | 修改 J2 (→1.2) | ✅ PASS | 0.0013 rad |
| 3 | 修改 J3 (→−1.2) | ✅ PASS | 0.0017 rad |
| 4 | 修改 J5 (→0.9) | ✅ PASS | 0.0013 rad |
| 5 | 最终回 HOME | ✅ PASS | 0.0016 rad |
| 6 | 越界值被裁剪到限位 | ✅ PASS | j2=3.1400, j3=0.0000 |
| 7 | 关节数错误被拒绝 | ✅ PASS | — |
| 8 | NaN 输入被拒绝 | ✅ PASS | — |
| 9 | Inf 输入被拒绝 | ✅ PASS | — |
| 10 | 急停后拒绝新目标 | ✅ PASS | — |
| 11 | 急停期间保持位置 | ✅ PASS | 漂移 **0.00000 rad** |
| 12 | 高频流式发送无异常 | ✅ PASS | 发送 2074 次 |
| 13 | 流式后仍能收敛回 HOME | ✅ PASS | 0.0016 rad |

**位置精度 0.0013 rad = 0.07°**，步进收敛时间约 1.3 s。

### 6.3 单元测试（25/25 通过）

`test/test_safety.py`，不依赖 ROS2：

```
TestConfig            4 项   限位与官方 SDK 一致性（回归保护）
TestValidation        4 项   长度/NaN/Inf/None 拒绝
TestPositionLimits    5 项   上下限裁剪、越界描述
TestRateLimit         5 项   速率限制、单周期硬上限、多周期收敛
TestEmergencyStop     2 项   急停状态机
TestTypes             5 项   Keypoint 的 z 字段约定（阶段三/七兼容）
```

---

## 7. 过程中发现并修复的问题

### 7.1 【严重，已修复】单次调用无法收敛 + 假通过

**现象**：首轮验收测试 13/13「全部通过」，但实测位置明显不符：

| 测试 | 指令目标 | 首轮实测 |
|---|---|---|
| J2 | 1.2 | 0.6302 |
| J5 | 0.9 | 0.0505 |

**根因**（两个耦合缺陷）：
1. `send_joint_target()` 内部**只做一次**速率限制。
   而 `SafetyLimiter` 是按「本周期允许的增量」计算的，
   因此一次调用最多让目标前进 0.1 rad，机械臂永远到不了最终目标。
2. `reached_target()` 拿 **streamer 的内部（被限速的）目标** 做比较，
   而不是上层期望目标，于是「刚开始动」就判定「已到位」。

两者叠加产生**假通过**——测试全绿但功能实际不工作。

**修复**：
- 拆分两个量：`_desired`（上层期望，做限位）与
  streamer 目标（经速率限制、本周期实际下发）；
- 把速率限制移到控制循环 `_advance_target()`，**每周期推进一次**；
- `reached_target()` / `target_error()` 改为针对 `_desired`；
- 验收测试增加「对请求值误差」的独立校验，防止再次出现假通过。

**修复后实测**（单次调用即收敛）：

```
只调用一次 send_joint_target([0, 1.5, -0.6, 0, 0.8, 0])
   t=0.0s  joint2=0.6000  joint5=-0.0000   err=0.9000
   t=0.5s  joint2=0.8031  joint5= 0.3031   err=0.6969
   t=2.0s  joint2=1.4744  joint5= 0.7979   err=0.0256
   t=2.5s  joint2=1.5003  joint5= 0.7975   err=0.0025
最终 joint2=1.5003 (误差 +0.0003) / joint5=0.7976 (误差 -0.0024)
收敛耗时 2.03s
```

**教训**：测试通过不等于功能正确。到位判据必须独立于控制器内部状态。

### 7.2 【已修复】到位判据导致精度虚低

`wait_until_reached()` 原来一进入容差就立即返回，
导致返回时误差刚好卡在容差边缘（0.0199 rad）。

修复：增加 `settle` 参数（默认 0.8 s），进入容差后再稳定观察一段时间。
效果：

| 指标 | 修复前 | 修复后 |
|---|---|---|
| 位置精度 | 0.0199 rad (1.14°) | **0.0013 rad (0.07°)** |
| 急停漂移 | 0.020 rad | **0.00000 rad** |

### 7.3 【已解决】`ros2 run` 找不到可执行文件

**现象**：`ros2 run piper_human_control joint_test` 报 `No executable found`。

**根因**：`ros2 run` 与 `ros2 pkg executables` 只在
`<prefix>/lib/<pkg>/` 下查找可执行文件；
而 ament_python 经 setuptools `easy_install` 生成的 entry_points 脚本，
在本环境（setuptools 59.6 + `colcon --symlink-install`）下被装到
`<prefix>/bin/`。同工作空间的官方 `piper` 包脚本位于 `lib/piper/`，
故其 `ros2 run` 正常——差异就在 easy_install 的 `script_dir`。

**尝试过但无效的方案**（记录备查，避免重复踩坑）：
1. `setup.py` 里追加 `--install-scripts` 参数 —— 无效
2. 自定义 `build_py.install_scripts` —— `develop` 模式不调用 `build_py`
3. 覆盖 `develop` / `install` 命令 —— 不改变 `easy_install` 行为
4. 覆盖 `easy_install.finalize_options` —— `script_dir` 随后被再次改写

**最终方案**：用 `data_files` 把 `scripts/<命令名>` 直接投放到
`lib/<pkg>/<命令名>`。因为 `data_files` 不保留文件名别名，
所以每个命令一个同名文件；启动器按自身文件名分发到对应入口。

结果：
```
$ ros2 pkg executables piper_human_control
piper_human_control joint_teleop
piper_human_control joint_test
```

### 7.4 【环境问题，已绕开】pytest 插件冲突

远程 `~/.local/lib/python3.10/site-packages` 下的旧版 `anyio`
的 pytest 插件与系统 pytest 不兼容（`ModuleNotFoundError: _pytest.scope`）。

绕开方式：运行时加 `-p no:anyio`。
```
python3 -m pytest test/test_safety.py -q -p no:anyio
```

---

## 8. 对阶段三/四的接口交接

### 8.1 阶段三（YOLO Pose）需要产出

```python
from piper_human_control import Keypoint, ArmKeypoints

kp = Keypoint(x=320.0, y=240.0, z=None, confidence=0.92,
              name="right_shoulder")
arm = ArmKeypoints(shoulder=..., elbow=..., wrist=..., side="right")
```

`z=None` 是当前约定；阶段七接入 RGB-D 后填真实深度，**接口不变**。

### 8.2 阶段四（映射）只需调用

```python
robot.send_joint_target([q1, q2, q3, q4, q5, q6])
```

限位、平滑、安全全部由本层负责，映射层不需要关心。

### 8.3 阶段五可直接复用的钩子

- `robot.emergency_stop(reason)` / `reset_emergency_stop()` —— 急停接口已就绪
- `robot.hold_current()` —— 「丢失跟踪时保持」已就绪
- `SafetyLimiter` 的 `clamped_joints` / `rate_limited_joints` 报告 —— 可观测性
- 待阶段五补充：EMA 滤波、Deadband、置信度门限（应加在**映射层**，
  而非控制层，以保持解耦）

### 8.4 关键约束提醒

- 关节顺序必须是 `["joint1".."joint6"]`
- `joint2 ≥ 0`、`joint3 ≤ 0`（阶段四映射必须遵守，否则会被裁剪）
- 控制频率建议 20~50 Hz 重发绝对目标；**不要**用 50 Hz 单点轰炸 + 大时间窗

---

## 9. 阶段二判定

```
阶段二：通过
```

**依据**：
1. 新增独立包 `piper_human_control`，**未修改 AgileX 官方 piper_ros 任何文件**；
2. 对外提供 `send_joint_target([q1..q6])` 接口，满足任务书要求的形式；
3. 限位、速度限制、加速度限制、合法性检查、平滑连续发送、
   状态读取、软件急停全部实现并验证；
4. 验收测试 **13/13 通过**，`HOME→J2→J3→J5→HOME` 序列正常，
   位置精度 **0.0013 rad (0.07°)**；
5. 单元测试 **25/25 通过**；
6. 所有映射/控制参数集中在 `config/joint_limits.yaml`，主程序无硬编码；
7. 过程中发现并修复了「单次调用无法收敛 + 到位判据假通过」的严重缺陷，
   并补充了独立校验防止复发。

**可进入阶段三。**
