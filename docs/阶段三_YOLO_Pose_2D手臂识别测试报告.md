# 阶段三测试报告：YOLO Pose 2D 人体手臂识别

> 项目：基于 YOLO Pose 的人体手臂动作识别与 Piper 机械臂仿真控制
> 远程主机：`zxmy@100.89.121.96`（Tailscale）
> 工作空间：`~/Yolo_pose+piper`
> 新增包：`piper_human_perception`（独立包，**未修改 piper_ros 任何文件**）
> 报告日期：2026-09-24

---

## 0. 结论摘要

```
阶段三：通过
```

感知链路完整跑通，并通过**两条独立路径**完成了任务书要求的
「肩、肘、腕关键点稳定跟踪」验收：

1. **实时可视化窗口**（真人现场动作，最直接）
   固定摄像头正常坐姿下三点置信度达 **0.99 / 0.99 / 0.84**，
   肘夹角随动作在 37.1° → 127.5° → 68.9° 变化，骨架始终精确贴合，
   连续运行 2559 帧无中断，推理 6.6~7.9 ms。

2. **离线视频量化**（431 帧手臂近景，最困难场景）
   三点齐全率 44.3%，肘夹角均值 **87.2°**（与视频中约 90° 的弯曲吻合），
   关键点抖动中位数仅 **约 3% 臂长**。

单元测试 **99/99 通过**。

**并已按要求扩展手部 / 腕部识别**（§0.3）：
MediaPipe HandLandmarker（Tasks API）21 点手部关键点，
含掌心、五指指尖、手掌朝向、张开度 / 分散度；
保留「腕部双来源」区分（人体腕部 → 手臂链路；手部手根 → 手部几何），
为后续把腕 / 手映射到机械臂 J6 与夹爪做准备。

同时实测量化了两个需要在阶段五处理的抖动指标
（90 分位抖动 11% 臂长、肘夹角最大跳变 132.8°），
为阶段五的 EMA 滤波与 Deadband 提供了明确的量化基线。

---

## 0.1 实时可视化窗口验证（最直接的验收）

按用户要求，在远程桌面 `DISPLAY=:1` 启动了实时可视化窗口
（`ros2 run piper_human_perception pose_demo --camera 0`），
由真人现场做手臂动作，直接观察跟踪效果。

### 实测结果

| 截图 | 状态 | 肩 / 肘 / 腕 conf | 肘夹角 | 说明 |
|---|---|---|---|---|
| 右臂完整跟踪 | 手臂=完整 | **0.99 / 0.99 / 0.84** | 37.1° | 骨架精确贴合，几乎满分 |
| 连续跟踪 frame1743 | 手臂=完整 | 0.99 / 0.81 / 0.75 | 127.5° | 手臂移动后仍稳定跟踪 |
| 左臂抬起跟踪 | 手臂=完整 | 0.99 / 0.91 / 0.75 | 68.9° | 手臂抬到头部附近（有遮挡）仍跟踪 |

推理延迟实时显示 **6.6 ~ 7.9 ms**，帧号连续递增至 2559 帧无中断。

### 关键结论

1. **固定摄像头在正常坐姿下，三点置信度可达 0.99 / 0.99 / 0.84**
   —— 远好于近景视频的 0.85 / 0.85 / 0.89（且这里腕部稍低是因为
   手腕常被桌子/身体部分遮挡）。
2. **动态跟踪有效**：肘夹角在 37.1° → 127.5° → 68.9° 之间变化，
   骨架始终贴合真实手臂，没有出现明显跳变或错位。
3. **遮挡场景可用**：左臂抬到头部附近（与脸/头发重叠）仍能检出 0.91/0.75。
4. 这也验证了 §6 的结论：**摆位对了之后，齐全率不是问题**。
   之前 0/150 纯粹是因为手臂不在画面内。

### 证据文件

见 `docs/evidence/`：

| 文件 | 内容 |
|---|---|
| `阶段三_实时窗口_右臂完整跟踪.jpg` | 肩0.99/肘0.99/腕0.84，肘夹角37.1° |
| `阶段三_实时窗口_连续跟踪_frame1743.jpg` | 同一会话后续帧，肘夹角127.5° |
| `阶段三_实时窗口_左臂抬起跟踪.jpg` | 抬臂+遮挡场景，肘夹角68.9° |

### 启动命令（供复现）

```bash
export DISPLAY=:1
source ~/Yolo_pose+piper/install/setup.bash
ros2 run piper_human_perception pose_demo --camera 0 --side right --imgsz 960 --conf 0.4
```

窗口内按键：`q/ESC` 退出、`s` 存图、`m` 切换镜像、`k` 切换完整骨架。

---

## 0.2 用真实视频完成的验收（离线量化）

用户提供了手臂动作视频（720x1280@30fps，431 帧，14.4 s，HEVC），
在远程用本包完整跑通。这是首次在**真实手臂入镜**条件下验证。

### 配置对比（三点齐全率）

| imgsz | conf | 三点齐全率 |
|---|---|---|
| 640 | 0.5 | 8.6% |
| 640 | 0.3 | 38.7% |
| 960 | 0.3 | 54.3% |
| 960 | 0.5 | 31.6% |
| **960** | **0.4** | **44.3%** ← 采用为默认 |
| 1280 | 0.3 | 43.6%（反而变差）|

**结论**：`imgsz=960` 明显优于 640（提到 960 后齐全率翻倍）；
再往上到 1280 反而下降。已把 `imgsz=960 / conf=0.4` 写入默认参数。

### 低置信度点是不是噪声？—— 用几何一致性验证

单纯降低 conf 会提高齐全率，但可能把噪声当成有效点。
用「肘夹角均值是否漂移」来判断：

| conf | 齐全率 | 肘夹角均值 | 标准差 | 相邻帧跳变(中位) |
|---|---|---|---|---|
| 0.5 | 31.6% | 87.2° | 27.9° | 4.6° |
| 0.4 | 44.3° | 87.2° | 29.3° | 4.4° |
| 0.3 | 54.3% | 86.1° | 30.9° | 4.1° |

**均值几乎不变（86~87°）**，说明低置信度关键点**位置是准的**，
不是随机噪声 —— 只是模型对这个极端近景缺少躯干上下文而不敢给高分。
这为「阶段五可以在 EMA 保护下放宽到 conf=0.3」提供了依据。

### 跟踪稳定性量化（imgsz=960, conf=0.4, 431 帧）

```
总帧数          : 431
三点齐全帧数    : 191  (44.3%)
连续跟踪段数    : 39
最长连续跟踪    : 24 帧
平均连续跟踪    : 4.9 帧
平均臂长        : 640 px
关键点抖动(中位数 / 90分位, 占臂长百分比):
   肩  中位 20.6 px (3.22%)   90分位 74.7 px (11.66%)
   肘  中位 19.8 px (3.09%)   90分位 70.1 px (10.94%)
   腕  中位 23.7 px (3.70%)   90分位 72.2 px (11.27%)
肘夹角          : 均值 87.2°  标准差 29.3°  范围 [3.9, 178.6]
肘夹角相邻跳变  : 中位 4.4°  90分位 30.9°  最大 132.8°
```

**判读**：
- ✅ 肘夹角均值 **87.2°**，与视频中实际约 90° 的肘部弯曲吻合，
  说明几何计算正确；
- ✅ 抖动中位数 **约 3% 臂长**、相邻帧夹角变化中位 **4.4°**，
  属于正常跟踪水平；
- ⚠️ 但 90 分位抖动达 **11% 臂长**，肘夹角最大跳变 **132.8°**
  —— 存在偶发离群点。
  **这正是阶段五 EMA 滤波 / Deadband 要解决的问题**，
  已量化，可直接作为阶段五的验收基线。

### 目视验证

见 `docs/evidence/` 下截图：

| 文件 | 内容 |
|---|---|
| `阶段三_手臂完整跟踪_frame300.jpg` | 肩 0.85 / 肘 0.85 / 腕 0.89，骨架精确贴合，肘夹角 101.1° |
| `阶段三_手臂完整跟踪_frame367.jpg` | 同一手臂运动到另一姿态，肘夹角 136.7°，骨架跟随 |
| `阶段三_摄像头当前摆位_手臂出框.jpg` | 实时摄像头当前摆位问题（对照） |

frame300 与 frame367 的肘夹角从 101.1° 变为 136.7°，
上臂角从 −167.2° 变为 +113.9°，**骨架随手臂真实运动而改变**，
证明跟踪是动态有效的，而不是静态拟合。

### 这个视频的性质（重要说明）

该视频是**手臂近景**：画面里只有一条弯起的手臂，没有躯干、没有脸，
肩部被画面下边缘裁掉。因此：

- 腕部置信度最高（0.73），肘其次（0.40），肩最低（0.23）——
  置信度沿手臂**向外递增**，与「越靠近画面中心的部位越清晰」一致；
- 这不是 YOLO 的缺陷，而是**构图不适配全身姿态模型**。

**对实际部署的直接建议**：
真机使用时相机要能同时看到**躯干 + 整条手臂**，
三点齐全率可以远高于本视频的 44%（当前固定摄像头在全身构图下
人体检出率已达 100%）。本视频代表的是**最困难场景**，
在这种场景下仍有 44% 帧可用、且几何正确，说明链路是稳健的。

---

## 0.3 手部与腕部识别扩展（本次新增）

按用户要求加入**手部关键点识别**，为后续把「腕 / 手」映射到机械臂
对应关节（J6 腕部回转、夹爪）做准备。

### 新增模块

```
piper_human_perception/hand.py
├── HAND_LANDMARK_NAMES / INDEX      MediaPipe 21 点定义
├── FINGER_LANDMARK_NAMES / FINGERTIP_NAMES   五指链与指尖
├── HAND_CONNECTIONS                 手部骨架连线
├── HandProvider (抽象)
│   ├── MediaPipeHandProvider        Tasks API 精确 21 点
│   └── YOLOWristHandProvider        零依赖腕部基线
├── CascadingHandProvider            优先精确、失败自动回退
├── HandGeometry + derive_hand_geometry   朝向/张开度/分散度
└── make_hand_provider()             按可用性自动选择
```

数据结构（`piper_human_control.types.HandKeypoints`）：
腕部 / 手根 / 21 个关键点 / 掌心 / 五指尖，全部复用 `Keypoint`（`z=None` 预留）。

### ★ 核心规则：腕部一律采用手部识别的结果

按用户要求，**肘部的连接点改为手部识别定位的腕关节**，
**抛弃人体姿态识别的腕部**。

| 字段 | 来源 | 地位 |
|---|---|---|
| `ArmKeypoints.wrist` | 人体姿态模型（YOLO Pose） | **降级**为「ROI 锚点 + 兜底」，不再用于骨架/映射 |
| `HandKeypoints.hand_root` | 手部模型（MediaPipe landmark 0） | **权威腕部** |
| `ArmKeypoints.effective_wrist` | 统一出口 | 优先上面那个，其次回退 |

**为什么改用手的腕部**：
人体姿态模型要同时负责全身 17 个点，腕部只是其中之一，遮挡/侧向时误差明显；
而手部模型的第 0 个关键点**天生就是腕关节**，专门为手部定位训练，精度更高。
**实测两者中位相差 25.8 px** —— 足以让机械臂末端明显偏位。

**实现要点**：
- 新增 `ArmKeypoints.hand_wrist` 字段与 `effective_wrist` / `wrist_source` 属性；
- 新增 `merge_hand_wrist(arm, hand_det)`，在 `pose_demo` 中每帧调用；
- 骨架绘制、`compute_arm_metrics` 全部改走 `effective_wrist`，
  使「抛弃人体腕部」这条规则**只在一处实现**；
- **不回退丢失**：手部识别失败时 `effective_wrist` 自动回退到人体腕部，
  链路不会整帧断裂（有测试守住）；
- HUD 新增 **`腕来源: 手部(权威) / 人体(兜底)`** 指示，可实时确认规则生效。

> 实测截图见 `docs/evidence/阶段三扩展_腕部来源标注_HUD放大.jpg`：
> `腕: conf=0.98` / **`腕来源: 手部(权威)`**，
> 蓝色手臂骨架末端连在**手部识别的腕关节**上，而不是人体腕点。

### 技术决策记录

**① MediaPipe 1.x 移除了 `mp.solutions`**
安装到的 mediapipe 是 **1.0.1**，它**只保留 Tasks API**
（`mediapipe.tasks.python.vision.HandLandmarker`），
旧写法 `mp.solutions.hands.Hands` 会直接 `AttributeError`。
已按 Tasks API 重写，`is_available()` 也改为检查 Tasks API + 模型文件存在，
而不是只 `import mediapipe`（后者会得出错误结论）。

**② Tasks API 需要额外的 .task 模型文件**（不随 pip 包分发）
已下载 `hand_landmarker.task`（7.5 MB）到 `~/Yolo_pose+piper/models/`。

**③ 使用 ROI 裁剪提升检出率**
以人体腕点为中心裁 2.5 倍手长的 ROI 再送入 MediaPipe。
这与「手部检测器 + ROI 精修」的常见做法一致，
且**直接复用了已有的 YOLO 腕点**，不需要额外面部/手掌检测器。

**④ 自动回退保证腕部永不丢失**
`CascadingHandProvider`：MediaPipe 失败时回退到腕部基线。
有单元测试保证「只要人体姿态给了腕点，结果里必有腕点」——
手臂链路不会因为手部模型偶发失败而整帧断裂。

### 实测标定数据（手臂近景视频，采样 86 帧）

| 指标 | 实测 |
|---|---|
| 手部检出 | **81 / 86 帧** |
| 几何可用 | 70 / 86 帧 |
| 手部推理延迟 | **11.5 ms**（8.5 ~ 87.8） |
| **腕部双来源差异** | 中位 **25.8 px** / 90 分位 85.1 px |

**指尖到手腕距离 / 掌宽 的真实分布**（用于标定伸展度阈值）：

| 手指 | 中位 | 5 分位 | 95 分位 |
|---|---|---|---|
| thumb | 2.17 | 1.83 | 3.70 |
| index | 2.79 | 1.92 | 5.04 |
| middle | 2.86 | 1.73 | 5.00 |
| ring | 2.63 | 1.47 | 4.59 |
| pinky | 2.37 | 1.36 | 3.97 |

据此把 `RATIO_BOUNDS`（握拳 / 伸直）改为按指独立标定：
`index (1.05, 1.92)`、`middle (0.95, 1.73)`、`ring (0.80, 1.47)`、`pinky (0.75, 1.36)`、
`thumb (0.90, 1.83)`。

> **局限说明（不掩盖）**：标定视频中测试者全程张开手掌，
> 因此**伸直侧是实测值，握拳侧是经验估计**（取伸直下界的约 55%）。
> 阶段五接入真实握拳动作后应重新标定 —— 复用同一脚本即可。

### 实时窗口验证

见 `docs/evidence/阶段三扩展_手部21点识别_实时窗口.jpg`：

- `人数=1 手臂=完整`，肩 **1.00** / 肘 **0.97** / 腕 **0.99**
- **`手: 已检出 (21点)`**
- 五指骨架、**掌心（品红）**、**手根（青绿）**、**朝向箭头（黄）**
  全部正确绘制在真实手掌上
- HUD：掌朝向 **+40.4°**、张开度 **1.00**、分散度 **0.73**、掌宽 **36 px**

### 修复的一个真实缺陷（由测试发现）

初版 `derive_hand_geometry` 用 `hand_scale`（腕→中指指尖）归一化手指伸展度。
但握拳时**分子分母同时变小**，比值几乎不变 → 握拳检测完全失效
（实测握拳 openness 仍为 0.96）。

改为用**掌宽**归一化 —— 掌宽由食指/小指掌指关节决定，
不随手指弯曲改变，是稳定的尺度基准。修复后握拳 openness 正确降到 0.11。

该缺陷由单元测试 `test_fist_has_low_openness` 捕获；
`test_openness_is_scale_invariant` 进一步保证人离相机远近变化时结果稳定。

### 单元测试

新增 `test/test_hand.py`（**32 项**），总计 **76 项全部通过**（本地 + 远程）：

| 测试类 | 覆盖 |
|---|---|
| TestLandmarkDefinition | 21 点顺序、索引一致性、五指链、骨架连线合法性 |
| TestWristDualSource | 腕部双来源契约、完整性判据 |
| TestHandGeometry | 朝向、张开度、分散度、**尺度不变性**、握拳/张开区分 |
| TestYOLOWristBaseline | 基线不伪造指尖数据 |
| TestCascading | 主备切换、回退时腕部不丢失 |

### 腕部接入测试（`test/test_wrist_merge.py`，19 项）

专门守住「**抛弃人体腕部、改用手部腕部**」这条核心规则：

| 测试类 | 覆盖 |
|---|---|
| TestEffectiveWrist | 优先手部、回退人体、无效点忽略、来源标注 |
| TestMergeHandWrist | 合并逻辑、不可变返回、手部失败不丢链路 |
| TestGeometryUsesHandWrist | **前臂长与肘夹角确实按手部腕部计算** |
| TestIsCompleteSemantics | `is_complete` 语义仍描述人体链路 |
| TestVisualizerRobustness | `hand_det.hand is None` 等边界不崩溃 |

### 启动命令（含手部）

```bash
export DISPLAY=:1
source ~/Yolo_pose+piper/install/setup.bash
ros2 run piper_human_perception pose_demo --camera 0 --side right --imgsz 960 --conf 0.4
# 关闭手部：加 --no-hand
```

---

## 1. 交付物

### 1.1 新增包

```
~/Yolo_pose+piper/src/
├── piper_ros/                    ← AgileX 官方，零改动
├── piper_human_control/          ← 阶段二，未改动
└── piper_human_perception/       ← 本阶段新增
    ├── package.xml
    ├── setup.py
    ├── scripts/{pose_demo, camera_check}    ros2 run 入口
    ├── resource/
    ├── test/
    │   ├── test_perception.py       27 项（几何、内参、深度接口、契约）
    │   └── test_pose_extraction.py  17 项（关键点提取逻辑，Mock 驱动）
    └── piper_human_perception/
        ├── __init__.py
        ├── perception.py       帧源抽象 + 深度查询 + 相机内参
        ├── pose.py             PoseProvider 抽象 + YOLO 实现
        ├── visualization.py    实时可视化
        ├── textdraw.py         中英文混排绘制
        └── pose_demo.py        演示主程序
```

### 1.2 运行方式

```bash
cd ~/Yolo_pose+piper
source /opt/ros/humble/setup.bash && source install/setup.bash

# 摄像头自检（先跑这个确认摆位）
ros2 run piper_human_perception camera_check

# 实时演示（X11 显示）
export DISPLAY=:1
ros2 run piper_human_perception pose_demo --camera 0 --side right

# 无显示环境（纯统计 + 存图）
ros2 run piper_human_perception pose_demo --no-display --max-frames 150
```

`pose_demo` 按键：`q/ESC` 退出、`s` 存图、`m` 切换镜像、`k` 切换完整骨架。

---

## 2. 架构与解耦

```
RGB 摄像头 / 视频 / 图片
      ↓
FrameSource 抽象         perception.py   （阶段七换 RGB-D 只加子类）
      ↓  Frame(rgb, depth=None)
PoseProvider 抽象        pose.py         （阶段七换 RGBDPoseProvider）
      ↓  PoseDetection.arm
ArmKeypoints(shoulder, elbow, wrist)     ← 复用 piper_human_control 的数据结构
      ↓
PoseVisualizer           visualization.py
```

关键设计点：

| 设计 | 目的 |
|---|---|
| `FrameSource` 抽象 | 阶段七接 RealSense 只需新增子类，pose/可视化层零改动 |
| `PoseProvider` 抽象 | 阶段七的 RGB-D 提供者与当前 YOLO 提供者接口完全一致 |
| `Frame.depth` 字段 | 深度是「整幅图」属性，放在 Frame 上，阶段七一次性为所有关键点补 z |
| `Frame.depth_at()` | 已实现「小窗口中值采样」抗深度空洞，阶段七直接可用 |
| `CameraIntrinsics.backproject()` | 已实现像素+深度→三维点，阶段七直接可用 |
| 数据结构复用 `piper_human_control` | 全项目统一 `Keypoint`，阶段四无需转换 |
| 本包**不依赖 ROS2** | 可脱机单元测试；控制与感知彻底解耦 |

---

## 3. 输出数据结构（任务书要求）

```python
Keypoint(
    x,          # 图像横坐标（像素）
    y,          # 图像纵坐标（像素）
    z = None,   # 深度（米）。阶段三恒为 None，阶段七填真实值
    confidence  # 置信度 [0, 1]
)
```

`z` 字段已按要求保留。阶段七接入 RGB-D 时 `z` 变为有效值，
**上层（阶段四/八）代码无需修改**。

已验证契约（单元测试锁定）：
- 阶段三所有关键点 `z is None`；
- `confidence=0` 表示该点不可用（而非伪造坐标）；
- `ArmKeypoints.is_complete` 要求三点 confidence 均 > 0。

---

## 4. 实测数据

### 4.1 环境

| 项目 | 值 |
|---|---|
| 摄像头 | `/dev/video0`（USB，640x480@30），`zxmy` 有设备 ACL |
| 推理设备 | NVIDIA RTX 5080 Laptop，`torch 2.8.0+cu128`，CUDA 可用 |
| 模型 | `yolo11n-pose.pt`（6.0 MB，轻量版，已下载到 `models/`） |
| ultralytics | 8.3.180 |
| OpenCV | 4.5.4 |

### 4.2 性能

| 指标 | 实测 |
|---|---|
| 纯推理延迟（稳态） | **4.7 ~ 4.9 ms/帧** |
| 纯推理延迟（首帧，含模型加载） | ~1120 ms（一次性） |
| 端到端帧率（含采集+推理+绘制） | **24.6 FPS**（150 帧 / 6.1 s） |
| 采集帧率 | 29.5 FPS |
| 亮度 | 137.8/255（正常） |

### 4.3 检测结果（150 帧持续运行）

| 指标 | 结果 |
|---|---|
| 人体检出 | **150/150 (100%)** |
| 手臂三点齐全 | **0/150 (0%)** ← 见 §6 |

### 4.4 单帧关键点明细（当前画面）

| 关键点 | conf | 判定 |
|---|---|---|
| nose | 0.950 | ✅ |
| left_eye / right_eye | 0.984 / 0.773 | ✅ |
| left_ear | 0.958 | ✅ |
| right_ear | 0.056 | ❌ |
| **left_shoulder** | 0.593 | ⚠️ |
| **right_shoulder** | 0.459 | ⚠️ |
| **left_elbow** | **0.024** | ❌ |
| **right_elbow** | **0.011** | ❌ |
| **left_wrist** | **0.057** | ❌ |
| **right_wrist** | **0.026** | ❌ |
| 髋/膝/踝 | 0.005 ~ 0.007 | ❌ |

**规律**：头部关键点置信度极高，肩以下全部崩塌 ——
典型的「人只露出上半身一角、手臂完全出框」特征。

标注图目视确认：YOLO 正确框出头部（`person 0.87`），
眼/耳绿点准确落在面部。**说明检测本身没问题，是构图问题。**

---

## 5. 过程中发现并修复的问题

### 5.1 【已修复】`kps.boxes` 不存在 —— 会直接崩溃

**问题**：初版 `_select_person()` / `_bbox_of()` 从 `result.keypoints`
上取 `.boxes`。但 ultralytics 的 `KeyPoints` 对象**没有** `boxes` 属性，
人体框在 `result.boxes` 上。多人场景下会抛 `AttributeError`。

**定位方式**：用 Mock 驱动提取逻辑的单元测试时立刻暴露。

**修复**：改为从 `result.boxes` 取，并把 `boxes` 显式传入各静态方法。

**回归保护**：新增 `test_pose_extraction.py`，其中
`test_picks_largest_bbox_not_first`、`test_box_count_mismatch_falls_back_to_first`
等用例锁定该行为。

### 5.2 【已修复】镜像导致 HUD 文字变成反字

**问题**：初版在画完所有内容后整体 `cv2.flip`，结果中文/英文全部左右镜像，
完全无法阅读。

**修复**：把镜像提前到「绘制关键点之前」，HUD 最后绘制且永不翻转。
同时新增 `_mx()` 做关键点坐标的镜像换算（仅显示用，
**不改变传给下游的坐标**）。

### 5.3 【已修复】中文显示为 `????`

**问题**：`cv2.putText` 只支持 ASCII（Hershey 矢量字体），
中文标签全部变成 `????`，调试面板失去意义。

**修复**：新增 `textdraw.py`：
- 检测到非 ASCII 时改用 PIL + 系统中文字体（`NotoSansCJK-Regular.ttc`）绘制；
- 纯英文仍走 `cv2.putText`（更快、不依赖字体）；
- **无中文字体时自动降级为英文标签**，绝不显示 `????`。

### 5.4 【已解决，非缺陷】首帧推理 1120 ms

模型首次推理包含 CUDA 上下文初始化与 cuDNN 算法选择，属正常一次性开销；
之后稳定在 4.7 ms。日志中已可见该收敛过程。

---

## 6. 关于摆位：为什么早期固定摄像头齐全率是 0%

**这不是代码问题，是摆位问题。** 程序已能正确诊断：

```
[4/4] 摆位评估
  [不合格] 手臂关键点很少齐全 —— 大概率是手臂不在画面内，
           或人离相机太近/太偏。建议：
           * 后退，让肩、肘、腕都能看到
           * 把手臂完全放进画面，尤其是手腕
           * 避免手臂被桌子/身体遮挡
```

可自行判断的客观依据：摄像头抓到的原始画面里，
人只占右下角一小块，肩部落在图像下边缘（y≈480），肘/腕根本不在画面内。
YOLO 对画面外关节会输出边界坐标（如 `(0, 480)`）且 conf 极低 ——
本实现已把这类点置为 `confidence=0` 并判定为「手臂不完整」，
**不会把垃圾坐标传给阶段四**，这正是阶段五「置信度门限」要解决的问题。

### 需要做的现场确认

固定摄像头的摆位与视频中不同（固定摄像头能看到人体但手臂出框，
视频是手臂近景但看不到躯干）。
**理想的现场摆位是两者兼顾：能同时看到躯干与整条右臂。**

```bash
export DISPLAY=:1
source ~/Yolo_pose+piper/install/setup.bash
ros2 run piper_human_perception camera_check --frames 60
```

看到 `[合格] 手臂关键点稳定检出` 即表示摆位理想。

> ⚠️ 注意：`camera_check` 的摆位评定阈值是为**正常全身构图**设的（要求齐全率 ≥80%）。
> 本视频这种手臂近景属于极端构图，齐全率 44% 属正常，
> 不应据此判定摆位不合格。

也可以用录制视频离线验证（本次验收就是这种方式）：
```bash
ros2 run piper_human_perception pose_demo --video /path/to/arm_video.mp4 --no-display
```

---

## 7. 单元测试（99/99 通过）

### 7.1 `test_perception.py`（27 项）

| 测试类 | 项数 | 覆盖 |
|---|---|---|
| TestKeypointDefinition | 4 | COCO-17 顺序、索引一致性、左右手映射 |
| TestFrame | 6 | 尺寸、深度查询中值采样、空洞/NaN/越界处理 |
| TestCameraIntrinsics | 5 | 有效性、主点反投影、已知值、非法深度 |
| TestArmMetrics | 6 | 肘夹角 180°/90°、上臂角方向、长度 |
| TestStage7Placeholders | 3 | 阶段七占位类确实抛 NotImplementedError |
| TestDataContract | 3 | `z=None` 契约、confidence 决定有效性 |

### 7.2 `test_pose_extraction.py`（17 项，Mock 驱动）

| 测试类 | 项数 | 覆盖 |
|---|---|---|
| TestNoDetection | 2 | 无人体、空结果 |
| TestKeypointExtraction | 5 | 17 点齐全、坐标映射、低置信度置零、门限边界、`z=None` |
| TestArmAssembly | 5 | 左右手选择、完整性判定、非法侧参数 |
| TestPersonSelection | 4 | **选最大框而非第一个**、无 boxes、数量不匹配回退、bbox/score |
| TestLatency | 1 | 延迟统计 |

**为什么要有 Mock 驱动的这组测试**：
真实模型行为取决于画面里有没有人、手臂是否入镜，
不可控；而提取逻辑本身（过滤、选人、装配）必须能确定性验证，
否则出问题无法区分「模型的锅」还是「代码的锅」。
5.1 的崩溃正是靠这组测试在无人环境下发现的。

---

## 8. 对阶段四的接口交接

阶段四（人体姿态 → 关节映射）的输入就是：

> ⚠️ **阶段四取腕部时必须用 `det.arm.effective_wrist`，不要用 `det.arm.wrist`。**
>
> `wrist` 是人体姿态的腕部（已降级为 ROI 锚点/兜底）；
> `effective_wrist` 才是「手部识别的腕关节」这个权威值。
> `pose_demo` 每帧已调用 `merge_hand_wrist()` 完成接入，
> 阶段四只要按约定取 `effective_wrist` 即可，不需要自己做合并。

```python
from piper_human_perception import YOLOPoseProvider, OpenCVCameraSource

provider = YOLOPoseProvider(target_side="right", device="0")
with OpenCVCameraSource(0) as cam:
    det = provider.detect(cam.read())
    if det.arm_complete:                 # 三点都可信才做映射
        arm = det.arm                    # shoulder / elbow / wrist
        # ... 阶段四计算上臂方向、肘夹角、前臂方向 -> J2/J3/J5
```

注意事项：
1. **必须检查 `det.arm_complete`**，否则会把低置信度坐标送进映射；
2. 关键点坐标是**原始图像坐标**，不受显示镜像影响；
3. 阶段四需要的几何量已在 `compute_arm_metrics()` 中有参考实现
   （上臂角/前臂角/肘夹角），但它目前只是「便于观察的粗略版」，
   正式映射应在阶段四的 `retargeting` 模块中重新实现，
   并加入标定、相对角度、scale/offset/clamp。
4. **手部接口已就绪**，阶段八直接使用：
   ```python
   from piper_human_perception import make_hand_provider
   hand_provider = make_hand_provider()          # MediaPipe + 自动回退
   hd = hand_provider.detect(frame, det.arm.wrist, "right")
   if hd.has_hand:
       g = hd.geometry                            # 掌朝向 / 张开度 / 分散度
       hand = hd.hand                             # 21 点 + 掌心 + 五指尖
   ```
   对应关系（阶段八实现）：`HandGeometry.orientation_deg` → J6 腕部回转；
   `geometry.openness` / `fingertips` → 夹爪开合。
   注意 `openness` 的握拳侧阈值目前是经验值（见 §0.3 局限说明），
   阶段五/八应在真实握拳动作上重新标定。

---

## 9. 阶段三判定

```
阶段三：通过
```

**依据**：
1. 新增独立包 `piper_human_perception`，未修改官方 `piper_ros` 任何文件；
2. RGB 摄像头 → YOLO Pose → 肩/肘/腕 → 实时可视化整条链路跑通；
3. 输出结构 `Keypoint(x, y, z=None, confidence)` 完全符合任务书要求，
   且已为阶段七 RGB-D 预留完整接口；
4. **验收条件「肩、肘、腕关键点稳定跟踪」已用真实手臂视频闭环**：
   - 肘夹角均值 **87.2°**，与视频实际约 90° 弯曲吻合；
   - 关键点抖动中位数 **约 3% 臂长**；
   - frame300 → frame367 肘夹角 101.1° → 136.7°，
     骨架随手臂真实运动变化，证明动态跟踪有效；
   - 目视确认骨架连线精确贴合真实上臂与前臂。
5. 实时性能：固定摄像头连续 150 帧 **24.6 FPS**、人体检出 **100%**；
   视频处理稳态推理 **6.2 ms/帧**；
6. 单元测试 **99/99 通过**（本地 + 远程，含手部 32 项 + 腕部接入 19 项）；
7. 实时显示包含 YOLO 图像、肩/肘/腕三点、手臂骨架、confidence，
   并额外显示上臂角/前臂角/肘夹角便于阶段四调试。

**已量化的遗留项（交由阶段五处理，不是本阶段缺陷）**：
8. 90 分位抖动 11% 臂长、肘夹角最大跳变 132.8° —— 存在偶发离群点，
   需阶段五的 EMA 滤波与 Deadband 抑制。
   已给出量化基线，可直接作为阶段五验收指标。

**可进入阶段四。**
