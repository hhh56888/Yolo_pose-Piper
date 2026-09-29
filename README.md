# Yolo_pose-Piper

单目 2D 人体姿态 → Piper 机械臂**实时遥操作**（ROS 2 Humble + Gazebo Classic）。

用普通 USB 摄像头识别操作者的手臂与手部，把动作实时重映射为 Piper 六轴机械臂的关节指令，
经 `JointTrajectoryController` 平滑下发。核心思路是"**人的手往哪走，机器人的手（TCP）就往哪走**"，
机器人内部的 J2/J3 由运动学求解，而不是"人体肘转多少、机械臂肘就转多少"。

---

## 功能与状态

| 模块 | 内容 | 状态 |
|---|---|---|
| 感知 | YOLO Pose（人体 17 点）+ MediaPipe（手部 21 点）| ✅ 实测 30 Hz |
| 几何 | 上臂/前臂方向角、肘夹角、腕部姿态、手腕归一化位置 u/v | ✅ |
| 重映射 A（`legacy`）| 上臂角→J2、肘夹角→J3（逐关节，保留可回退）| ✅ 稳定版 |
| 重映射 B（`task_space`）| 人体手腕相对位移 → 机器人 TCP (x,z) → J2/J3 数值 IK | ✅ V1.1 PASS |
| 腕部 | 腕俯仰→J5、拇指相对四指→J6（末端自转）| ✅ 实机标定 |
| 夹爪 | 手掌张开度 → joint7 开合 | ✅ |
| 安全 | 关节限位三层（physical ⊇ safe ⊇ retarget）、速度/加速度限速、急停、丢失回归原位 | ✅ |
| 数据可信 | 每帧 raw/filtered/mapped/final 全链路进 CSV；缺量显式报错不静默 | ✅ |

一行配置切换两套重映射：

```yaml
# src/piper_human_retargeting/config/retargeting.yaml
retarget_mode: "legacy"      # 逐关节映射（默认，稳定可回退）
retarget_mode: "task_space"  # V1.1：人的手 → TCP → J2/J3 IK
```

---

## 目录结构

```text
├── docs/                        文档与实测证据（先看 docs/README.md 索引）
├── models/                      YOLO 权重
├── testdata/                    测试视频
├── src/piper_human_perception/  感知：YOLO Pose + MediaPipe
├── src/piper_human_control/     控制：安全限速 + 流式轨迹下发
├── src/piper_human_retargeting/ 重映射：几何 → legacy / task_space
│   ├── config/retargeting.yaml      唯一配置源
│   ├── piper_human_retargeting/     模块（含 data/fk_chain.json）
│   ├── tools/                       离线工具（FK 生成 / workspace 扫描 / 审计 / A-B）
│   └── test/                        315 条 pytest 用例
└── fix_upstream_bugs.sh         官方 piper_ros 的补丁脚本
```

**详细约定见 [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md)**（哪些目录属于本项目、新文件放哪、
数据文件来源与再生成、维护与备份命令）。

> 本仓库**不包含**官方 `piper_ros` / `piper_sdk`（第三方上游，按需单独克隆，
> 本项目对其的唯一改动由 `fix_upstream_bugs.sh` 记录并可复现）。

### 仓库内容来源

本仓库内容 = 机械臂主机上的工程目录 `~/Yolo_pose+piper`（本项目实际运行的那一份），
**逐文件核对一致**（比对方式：两侧 `md5sum` 全量清单 diff，代码/配置/脚本 0 处不同）。
不含该目录下的 colcon 产物 `build/` `install/` `log/`（可重建）与官方 `piper_ros`。
`docs/evidence/` 取两侧的并集（比远端多保留早期阶段的截图/CSV，报告里按此引用）。

---

## 快速开始

### 依赖

```text
ROS 2 Humble（rclpy / sensor_msgs / trajectory_msgs / tf2_ros）
Python: numpy, opencv-contrib-python(4.x), ultralytics(YOLO Pose), mediapipe, pyyaml
仿真:   gazebo_ros2_control + JointTrajectoryController（/arm_controller）
```

### 编译与运行

```bash
# 工作区（把本仓库放进 catkin_ws/src，或直接作为工程根）
colcon build --packages-select piper_human_control piper_human_perception piper_human_retargeting
source install/setup.bash

# 摄像头实时遥操作
ros2 run piper_human_retargeting retarget_demo \
  --camera 0 --side left --imgsz 960 --auto-calibrate --csv /tmp/run.csv

# 离线回放（不接机械臂也能验证识别与映射）
ros2 run piper_human_retargeting retarget_demo \
  --video testdata/arm_fwd_up.mp4 --side right --auto-calibrate --no-display
```

运行中按键：`c` 标定 · `r` 重标定 · `n` 回位 · `k` 切换测试阶段 · `g` 切换夹爪 ·
`o` 记张开 · `p` 记握拳 · `e` 急停 · `q` 退出。

### 测试

```bash
cd src/piper_human_retargeting && python3 -m pytest test/ -q -p no:anyio   # 315 passed
```

---

## 关键实测结果（V1.1 task-space）

同一段人体输入、只改 `retarget_mode` 一行：

| 指标 | legacy | task_space |
|---|---|---|
| `corr(人体水平 u, TCP x)` | −0.407 | **+0.778** |
| `corr(人体垂直 v, TCP z)` | +0.259 | **+0.512** |
| 串扰 z→x（单轴输入）| −0.436 | **+0.008** |
| J2 关节限位饱和 | 9.26% | **0.00%** |
| IK 求解耗时 | — | 中位 0.51 ms / P95 0.63 ms |
| 控制环 | 30 Hz | 30 Hz，**超时 0** |

实时识别测试（真摄像头 + 人在环）：TRACKING 99%，`corr(手抬高, TCP 抬高) = +0.90`，
识别丢失后 1 s 宽限 → 平滑回归原位。详见
[`docs/2D_TaskSpace_Retargeting_V1_1_Report.md`](docs/2D_TaskSpace_Retargeting_V1_1_Report.md)。

---

## 已知限制

* **单目 2D 无法观测深度**：正面机位下"手前伸"只能用投影变化近似（实测标定 `sign_h`）；
  想要几何精确的前后控制，应把相机放在人体侧面。RGB-D / 3D 未实现。
* 人手侧的**角度符号依赖相机站位**，属于"必须实测标定"的配置项
  （`task_space.sign_h`、`j5_wrist.invert` 已在配置里写明标定依据）。
* J1/J4 固定不控；夹爪开合阈值需按人手标定。
* `docs/evidence/` 保留了各阶段实测截图与 CSV（约 14 MB），便于追溯；如需精简可单独裁剪。

---

## 许可与第三方

* 本项目代码：未声明许可证（默认保留所有权利），如需开源请自行补 `LICENSE`。
* `models/yolo11n-pose.pt` 为 Ultralytics YOLO 权重（**AGPL-3.0**），商用前请确认其许可。
* 依赖 MediaPipe、OpenCV、ROS 2 等第三方组件，各自遵循其原始许可。
* 官方 `piper_ros` / `piper_sdk` 不随本仓库分发。
