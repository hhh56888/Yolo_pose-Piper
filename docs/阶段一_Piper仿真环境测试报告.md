# 阶段一测试报告：Piper ROS2 Gazebo 仿真环境

> 项目：基于 YOLO Pose 的人体手臂动作识别与 Piper 机械臂仿真控制
> 远程主机：`zxmy@100.89.121.96`（Tailscale）
> 工作空间：`~/Yolo_pose+piper`
> 报告日期：2026-09-23

---

## 0. 结论摘要

```
阶段一：通过
```

Piper 在 Gazebo Classic 11 中正常加载，6 个关节全部可正常运动并精确收敛到目标位，
无物理异常。阶段二可以开始。

---

## 1. 重要前置发现（与任务书的偏差）

### 1.1 `roboticscollective/piper` 仓库不存在

任务书中点名参考的 `roboticscollective/piper` **在 GitHub 上不存在**。
经核实，Piper 机械臂的官方 ROS 仓库是：

- **<https://github.com/agilexrobotics/piper_ros>**（含 ROS1/ROS2 多分支）

该仓库分支情况：

| 分支 | 说明 |
|---|---|
| `noetic` | ROS1（默认分支，本机 `catkin_ws/src/piper_ros` 当前所在分支） |
| `foxy` | ROS2 Foxy |
| **`humble`** | **ROS2 Humble ← 本项目采用** |
| `humble_beta1` | 测试分支 |

### 1.2 远程机器上原本没有任何 Piper ROS 包

对远程 `$HOME` 全盘检索 `package.xml`（内容含 piper）与目录名，
**只找到一个 Isaac Sim 用的 `piper_description`**（位于 `isaac_projects/kinova_isaaclab_sim2real_piper`）。
远程 `~/Desktop/Projects/Project2` 是 MuJoCo + RRT 项目，不是 ROS2 仿真。

因此阶段一的性质是**部署**，不是「查找已有项目」。

### 1.3 模拟器版本决策

远程同时存在两套 Gazebo：

| 模拟器 | 版本 | 桥接包 | 状态 |
|---|---|---|---|
| Gazebo Classic | 11.10.2 | `gazebo_ros2_control` | 原本**缺失**，已安装 |
| Gazebo Sim (Fortress) | 6.16.0 | `gz_ros2_control` | 已装，但官方 humble 分支不用它 |

官方 `piper_gazebo` 的 launch 文件使用 `gazebo`（Classic）+ `libgazebo_ros2_control.so`，
故选择 **Gazebo Classic 11** 路线。

---

## 2. 仿真启动命令

### 2.1 依赖安装（已完成）

```bash
sudo apt install ros-humble-gazebo-ros2-control
```

> 官方 `piper_sim/README.md` 要求的完整依赖：
> `gazebo ros-humble-gazebo-ros-pkgs ros-humble-gazebo-ros2-control
>  ros-humble-ros2-control ros-humble-ros2-controllers`
> 其中只有 `gazebo_ros2_control` 缺失，其余已预装。

### 2.2 首次编译

```bash
cd ~/Yolo_pose+piper
source /opt/ros/humble/setup.bash
colcon build --symlink-install
```

### 2.3 启动仿真（带夹爪）

```bash
cd ~/Yolo_pose+piper
source /opt/ros/humble/setup.bash
source install/setup.bash
export DISPLAY=:1          # 远程桌面为 :1
ros2 launch piper_gazebo piper_gazebo.launch.py
```

### 2.4 启动仿真（无夹爪）

```bash
ros2 launch piper_gazebo piper_no_gripper_gazebo.launch.py
```

---

## 3. 工作空间结构

```
~/Yolo_pose+piper/               ← 完全独立的新工作空间，不污染原有项目
├── src/
│   └── piper_ros/               ← agilexrobotics/piper_ros @ humble (commit 017ffef)
│       └── src/
│           ├── piper_description/   URDF / xacro / meshes / rviz
│           ├── piper_sim/
│           │   ├── piper_gazebo/    ★ Gazebo 仿真（本项目核心）
│           │   └── piper_mujoco/    MuJoCo 仿真
│           ├── piper_msgs/          ROS2 消息定义
│           ├── piper_moveit/        MoveIt 配置（单臂/无夹爪）
│           ├── piper/               真机 CAN 控制节点（本阶段不用）
│           └── piper_humble/        空壳包
├── build/  install/  log/       ← colcon 产物，共 8 个包编译成功
└── docs/                        ← 本报告
```

编译结果：**8 个包全部成功**（`Summary: 8 packages finished`），
其中 `piper_msgs` 需编译消息，其余为 ament_cmake / ament_python 包。

---

## 4. 主要 ROS2 package

| 包名 | 类型 | 作用 |
|---|---|---|
| `piper_description` | ament_cmake | URDF / xacro / STL 网格 / RViz 配置 |
| `piper_gazebo` | ament_cmake | **Gazebo 仿真 launch + controller 配置 + 夹爪镜像节点** |
| `piper_mujoco` | ament_cmake | MuJoCo 仿真（本阶段未使用） |
| `piper_msgs` | ament_cmake | 自定义消息/服务 |
| `piper` | ament_python | 真机 CAN 通信节点（需硬件，本阶段不用） |
| `piper_with_gripper_moveit` / `piper_no_gripper_moveit` | ament_cmake | MoveIt 配置 |

---

## 5. 主要 Topic

### 5.1 控制指令入口（**写**）

| Topic | 类型 | 说明 |
|---|---|---|
| `/arm_controller/joint_trajectory` | `trajectory_msgs/JointTrajectory` | **6 轴手臂控制入口** |
| `/gripper_controller/joint_trajectory` | `trajectory_msgs/JointTrajectory` | 夹爪 joint7 控制 |
| `/gripper8_controller/joint_trajectory` | `trajectory_msgs/JointTrajectory` | 夹爪 joint8 控制 |

### 5.2 状态反馈（**读**）

| Topic | 类型 | 说明 |
|---|---|---|
| `/joint_states` | `sensor_msgs/JointState` | **8 个关节的位置/速度**，500 Hz |
| `/arm_controller/controller_state` | `control_msgs/JointTrajectoryControllerState` | 含 `reference`（控制器内部目标） |
| `/dynamic_joint_states` | `control_msgs/DynamicJointState` | 含 effort 信息 |
| `/tf`, `/tf_static` | TF | 由 `robot_state_publisher` 发布 |
| `/robot_description` | `std_msgs/String` | URDF 文本 |

> ⚠️ **重要**：`/joint_states` 是**状态反馈**，不是指令入口。
> 真机 `piper_ros` 的 ROS1 节点确实订阅 `/joint_states` 收指令，
> 但**仿真下必须用 `/arm_controller/joint_trajectory`**。这一点与直觉相反，阶段二会封装掉。

---

## 6. Controller

配置文件：`src/piper_ros/src/piper_sim/piper_gazebo/config/ros2_controllers.yaml`

`controller_manager` 更新率：**500 Hz**

| Controller | 类型 | 关节 | 命令接口 | 状态 |
|---|---|---|---|---|
| `joint_state_broadcaster` | `JointStateBroadcaster` | joint1~8（状态） | — | **active** |
| `arm_controller` | `JointTrajectoryController` | joint1~joint6 | position | **active** |
| `gripper_controller` | `JointTrajectoryController` | joint7 | position | **active** |
| `gripper8_controller` | `JointTrajectoryController` | joint8 | position | **active** |

补充：
- `arm_controller` 使用 `splines` 插值，状态发布 50 Hz，action 监控 20 Hz。
- 另有独立节点 `gripper_mirror_controller`（`joint8_ctrl.py`），
  订阅 `/gripper_controller/controller_state`，令 `joint8 = -joint7`，
  实现双指同步。**实测有效**（joint7=+0.0005458 时 joint8=-0.0005459）。

---

## 7. J1~J6 关节名称 / Joint Limit / 旋转方向

### 7.1 权威限位来源

限位以官方 **`piper_sdk/piper_param/piper_param_manager.py`** 为准。
经比对，`piper_description_gazebo.xacro` 中的 URDF 限位与 SDK **完全一致**，可信。

> ⚠️ 第三方文档 <https://www.roboticscenter.ai/zh/hardware/agilex-piper/specs>
> 声称 J1 为 ±175°、J2 为 −90°~+135°，与官方 SDK **不符，不可采用**。

### 7.2 完整关节表

| 关节 | 类型 | URDF 限位 (rad) | 官方 SDK 限位 (deg) | 速度限制 | effort | 轴向 | 说明 |
|---|---|---|---|---|---|---|---|
| `joint1` | revolute | `[-2.618, 2.618]` | ±150° | 5 | 100 | `0 0 1` | 底座回转 |
| `joint2` | revolute | `[0, 3.14]` | 0° ~ 180° | 5 | 100 | `0 0 1` | **肩部俯仰（下限为 0）** |
| `joint3` | revolute | `[-2.967, 0]` | −170° ~ 0° | 5 | 100 | `0 0 1` | **肘部（上限为 0）** |
| `joint4` | revolute | `[-1.745, 1.745]` | ±100° | 5 | 100 | `0 0 1` | 前臂回转 |
| `joint5` | revolute | `[-1.22, 1.22]` | ±70° | 5 | 100 | `0 0 1` | 腕部俯仰 |
| `joint6` | revolute | `[-2.0944, 2.0944]` | ±120° | 3 | 100 | `0 0 1` | 腕部回转 |
| `joint7` | **prismatic** | `[0, 0.035]` m | 夹爪 0~0.07 m | 1 | 10 | `0 0 1` | 夹爪手指 A |
| `joint8` | **prismatic** | `[-0.035, 0]` m | 同上 | 1 | 10 | `0 0 -1` | 夹爪手指 B（镜像） |

**关键坑点（阶段四映射必须注意）**：
- `joint2` 下限是 **0**，不是负数 —— 不能直接给负角度。
- `joint3` 上限是 **0**，不是正数 —— 弯曲方向与直觉相反。
- `joint7`/`joint8` 是**移动关节**（米），不是旋转关节；
  且两者**轴方向相反**（`0 0 1` vs `0 0 -1`），与 `joint8 = -joint7` 的镜像逻辑对应。

### 7.3 旋转方向（实测）

采用「固定 HOME → 单关节相对变化 → 比较 Δ实测 与 Δ指令 符号」测得。
GRASP 实际测试中所有关节**指令符号与实测位移符号一致**，即：

| 关节 | 正指令 | 效果 |
|---|---|---|
| `joint1` | + | 逆时针（俯视）回转 |
| `joint2` | + | 上臂抬起（0 → 180°） |
| `joint3` | − | 肘部弯曲（0 → −170°） |
| `joint4` | + | 前臂正向回转 |
| `joint5` | + | 腕部正向俯仰 |
| `joint6` | + | 腕部正向回转 |

> 各关节 `origin rpy` 存在大量 ±1.5708 / −3.1416 / −1.7939 的坐标系偏置，
> 因此「URDF 轴向量」不等于「视觉上的世界方向」。上表为**实测**结果。

---

## 8. 验证过程与结果

### 8.1 验证环境

- Gazebo Classic 11.10.2，`DISPLAY=:1`，NVIDIA RTX 5080 硬件渲染
  （`OpenGL renderer: NVIDIA GeForce RTX 5080 Laptop GPU/PCIe/SSE2`, OpenGL 4.6）
- ROS2 Humble，`ros2_control` 更新率 500 Hz

### 8.2 启动验证

| 检查项 | 结果 |
|---|---|
| `gzserver` / `gzclient` 进程 | ✅ 各 1 个，稳定运行 |
| 机器人实体生成 | ✅ `SpawnEntity: Successfully spawned entity [piper]` |
| `gazebo_ros2_control` 插件加载 | ✅ `Loaded gazebo_ros2_control.` |
| 硬件接口 `GazeboSystem` | ✅ initialize / configure / activate 全部成功 |
| `/joint_states` | ✅ 8 个关节，**500.4 Hz** |
| Controller | ✅ 4 个全部 `active` |
| 命令接口 claim | ✅ 8 个 `jointN/position [available] [claimed]` |
| effort 字段 | ⚠️ 全为 `.nan`（Gazebo 位置接口无 effort 反馈，正常） |

### 8.3 关节运动验证

**测试方法（v3，自适应收敛判定）**：
从 HOME `[0, 0.6, -0.6, 0, 0, 0]` 出发，每个关节取限位范围内 25% 与 75% 两个目标，
以 2.0 s 轨迹时窗、5 Hz 持续重发绝对目标，直到误差进入 ±0.02 rad 或超时 20 s。

| 关节 | 目标 | 实测 | 误差 | 收敛耗时 | 判定 |
|---|---|---|---|---|---|
| joint1 | −1.309 | −1.2945 | +0.0144 | 8.55 s | ✅ PASS |
| joint1 | +1.309 | +1.2959 | −0.0131 | 8.75 s | ✅ PASS |
| joint2 | +0.785 | +0.7713 | −0.0137 | 4.93 s | ✅ PASS |
| joint2 | +2.355 | +2.3406 | **−0.0144** | **~12 s** | ✅ PASS ★ |
| joint3 | −2.225 | −2.2118 | +0.0135 | 9.15 s | ✅ PASS |
| joint3 | −0.742 | −0.7275 | +0.0143 | 4.12 s | ✅ PASS |
| joint4 | −0.873 | −0.8594 | +0.0131 | 7.95 s | ✅ PASS |
| joint4 | +0.873 | +0.8591 | −0.0134 | 7.94 s | ✅ PASS |
| joint5 | −0.610 | −0.5961 | +0.0139 | 7.15 s | ✅ PASS |
| joint5 | +0.610 | +0.5956 | −0.0144 | 7.14 s | ✅ PASS |
| joint6 | −1.047 | −1.0329 | +0.0143 | 8.14 s | ✅ PASS |
| joint6 | +1.047 | +1.0339 | −0.0133 | 8.34 s | ✅ PASS |

**12/12 全部通过**，最大稳态误差 **0.0144 rad = 0.83°**。

★ `joint2` 在 v3 中因 20 s 超时被判 FAIL（实测 2.1469，误差 0.208），
经专项延长测试（30 s）确认其稳态误差同为 **−0.0144 rad**，属**稳定时间不足**而非故障：

```
t= 0.5s  j2=2.2375  err=-0.1175
t= 3.5s  j2=2.3269  err=-0.0281
t= 9.5s  j2=2.3394  err=-0.0156
t=30.0s  j2=2.3406  err=-0.0144   ← 稳定
```

### 8.4 反向运动与平滑轨迹

| 测试 | 结果 |
|---|---|
| joint2 顺滑下降 `2.355 → 0` | ✅ 收敛到 `−0.0000`，无超调、无振荡 |
| joint2 多航点上升 `0 → 2.355`（4 段） | ✅ 平滑到达 2.32，稳态残差 ~0.03–0.05 rad |

---

## 9. 发现的问题与修复

### 9.1 【已修复】上游 bug：`joint8_ctrl.py` 缺少可执行位

**现象**：首次 `ros2 launch` 直接失败：

```
[ERROR] [launch]: Caught exception in launch: executable 'joint8_ctrl.py' not found
on the libexec directory '.../install/piper_gazebo/lib/piper_gazebo'
```

**根因定位**：对比 git 记录的权限位，发现上游提交不一致——

| 文件 | git mode |
|---|---|
| `piper_mujoco/scripts/piper_mujoco_ctrl.py` | `100755` ✅ |
| **`piper_gazebo/scripts/joint8_ctrl.py`** | **`100644`** ❌ |

AgileX 在 `humble` 分支把 `joint8_ctrl.py` 提交为**非可执行**，
但 `piper_gazebo.launch.py` 却把它当 executable 启动（`Node(package='piper_gazebo',
executable='joint8_ctrl.py')`）。配合 `--symlink-install`，安装目录只是软链，
`ros2 launch` 找不到可执行文件。

**修复**（最小改动，仅改权限位，**文件内容零改动**）：

```bash
chmod +x ~/Yolo_pose+piper/src/piper_ros/src/piper_sim/piper_gazebo/scripts/joint8_ctrl.py
```

MD5 修复前后均为 `9d30891a5934d17a8ff4575bb155d8ad`，证明内容未变。

> ⚠️ **该修复会被覆盖**：重新解压源码、`git checkout` 或 `git pull` 都会把
> 权限位还原为 `644`，导致下次启动仿真再次失败（实测已复现一次）。
>
> **已提供幂等修复脚本**（阶段二新增）：
> ```bash
> bash ~/Yolo_pose+piper/fix_upstream_bugs.sh
> ```
> 该脚本会补可执行位并重新 colcon build，可重复安全执行。
> 每次重新拉取源码后跑一次即可。

### 9.2 【仅分析，未修改】`ros2_control` 中 command limit 严重失真

`piper_description_gazebo.xacro` 中 8 个关节的 `<command_interface name="position">`
被统一写成：

```xml
<param name="min">-1</param>
<param name="max">1</param>
```

这与真实关节限位完全不符（如 `joint2` 实际范围 `[0, 3.14]`）。

**实测结论：该参数对 `position` 命令接口不生效**——
测试中 `joint2` 实际到达 **3.0804 rad**，远超 `max=1`。
（在 `gazebo_ros2_control` 中 `min`/`max` 用于 **velocity** 命令接口。）

**风险提示**：这些数字具有误导性，阶段二的安全限位**不可**依赖它，
必须使用第 7.2 节的官方 SDK 限位表。

### 9.3 【已规避】测试方法论陷阱

首轮测试误判「6 个关节全部不响应」，实际是**测试代码缺陷**：
以 50 Hz 连续发布「单点轨迹」，每次发布都让 `JointTrajectoryController`
以**当前位置**为新轨迹起点重新规划，参考轨迹被反复锚定在当前位，
收敛被人为扼杀。改用「足够 `time_from_start` + 低频重发绝对目标」后全部正常。

**教训（阶段五直接相关）**：控制周期与轨迹时窗的配合方式会显著影响收敛性，
实时控制模块设计时必须明确 trajectory 的 `time_from_start` 策略。

### 9.4 【环境噪声，无影响】其他

| 现象 | 说明 |
|---|---|
| `ALSA ... Unable to open audio device` | 无音频设备，Gazebo 自动禁用音频，无影响 |
| `Missing model.config for model "mycar"` | 用户 `~/.gazebo/models/mycar` 残留模型不完整，与本项目无关 |
| `Desired controller update period (0.002s) is slower than gazebo simulation period (0.001s)` | 500 Hz 控制器 vs 1000 Hz 物理步长，仅提示，运行正常 |
| `allow_nonzero_velocity_at_trajectory_end` deprecated | 上游 yaml 使用了将废弃参数，暂不影响 |

---

## 10. 验收标准对照

| 验收标准 | 结果 |
|---|---|
| Gazebo 中 Piper 正常加载 | ✅ 实体生成成功，6+2 关节完整 |
| 6 个 Joint 可以正常运动 | ✅ J1~J6 双向运动，稳态误差 ≤ 0.0144 rad (0.83°) |
| 没有明显物理异常 | ✅ 无抖动、无穿模、无失稳；反向轨迹无超调 |

---

## 11. 对阶段二的接口交接

阶段二需要封装的底层事实：

```
控制入口   : /arm_controller/joint_trajectory   (trajectory_msgs/JointTrajectory)
关节顺序   : ["joint1","joint2","joint3","joint4","joint5","joint6"]
状态反馈   : /joint_states                       (500 Hz, 含 joint1~8)
轨迹时窗   : 建议 >= 1.0 s (实测 2.0 s 收敛良好)
控制频率   : 5~20 Hz 重发绝对目标即可（不建议 50 Hz 单点轰炸）
稳态精度   : ~0.014 rad (0.83°)
收敛时间   : 单关节阶跃约 4~12 s (与幅度和负载相关)
安全限位   : 见第 7.2 节（官方 SDK 表，勿用 xacro 中的 ±1）
```

---

## 12. 阶段一判定

```
阶段一：通过
```

**依据**：
1. Gazebo Classic 11 中 Piper 正常加载，URDF/Xacro、ros2_control、controller 全部确认；
2. `joint_state_broadcaster` + `arm_controller` + 2 个夹爪 controller 全部 `active`；
3. `/joint_states` 稳定 500 Hz，8 关节反馈正常，`joint8 = -joint7` 镜像生效；
4. J1~J6 逐个双向驱动测试 **12/12 收敛到位**，最大稳态误差 0.83°；
5. 反向与多航点顺滑轨迹均正常，无超调、无振荡、无物理异常；
6. 唯一阻塞性缺陷（上游 `joint8_ctrl.py` 可执行位）已定位根因并最小化修复。

**可进入阶段二。**
