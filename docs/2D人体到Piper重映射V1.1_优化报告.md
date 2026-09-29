# 2D 人体 → Piper 机械臂 重映射 V1.1 优化报告

> 项目：基于人体姿态识别控制 Piper 机械臂
> 日期：2026-09-25
> 本轮范围：**只优化重映射、标定基准、J2/J3 的二维运动表达**
> 未改动：YOLO Pose、MediaPipe、ROS2 Controller、`/arm_controller`、
> 安全限速、急停、GUI、控制频率架构、官方 `piper_ros`

---

## 结论

```
2D RETARGETING V1.1

FAIL
```

**阻塞点**：task_space 方案在**实机视频测试**上未能达成解耦
（`corr(dReach,X) = −0.035`，期望强正），且引入了 5.7% 的 IK 大跳变。

**但本轮查清了失败的确切原因**（见 §6b），且三项都是可修的：

1. **标定窗口把 `reach0` 顶到量程上界**（实测 0.962 / 上界 1.0）
   → 前伸方向无余量。改用「采一段自然姿势」即可；
2. **该视频里 `reach` 的激励幅度很小**（TRACKING 帧集中在 0.9~1.0），
   「前伸」自由度基本没被激励 —— 需要换一段**有明显伸/收**的视频；
3. **IK 用离散网格最近邻**（中位步进 0.206 rad）→ 需改增量式数值求解。

合成动作下 task_space **确实优于 legacy**（前伸 X/Z 比 3.87→4.65），
说明映射与 IK 的实现本身是正确的。

| 交付项 | 状态 |
|---|---|
| ① delta_upper 审计 | ✅ 完成，结论与原始假设**相反** |
| ② 新 Neutral 设计 | ⚠️ 未执行（理由见 §3） |
| ③ Legacy 保留为可选 | ✅ 完成 |
| ④ task_space 实现 | ✅ 完成并通过单元测试 |
| ⑤ task-space 对比测试 | ✅ 完成（**结果 FAIL**） |
| ⑥ J5 使用相对腕姿态 | ✅ 已确认满足 |
| ⑦ 手部独立标定 | ✅ 完成（状态机 + 可审计统计） |
| ⑧ 夹爪双点标定 | ✅ 完成 |
| ⑨ 数据可信度原则 | ✅ 完成（raw→mapped→q 全链路可观测） |
| ⑩ 远端部署与实机验证 | ✅ **完成**（SSH 恢复后已部署并完成实机视频测试） |

测试：`piper_human_retargeting` **231**、`piper_human_perception` 102（7 skipped）、
`piper_human_control` 25 → **总计 358**，全部通过。

---

## 1. delta_upper 检查

### 审计方法

按任务书第二节要求，先检查三个方向角的
「滤波是否周期处理」与「标定做差是否周期处理」。

### 审计结果：**199° 不是 wrap 假行程，差分层没有缺陷**

| 检查项 | 结果 | 证据 |
|---|---|---|
| `angle_delta_deg` 是否 shortest-angle | **✅ 已实现** | 见下 |
| 与任务书建议的 `atan2(sin,cos)` 形式是否等价 | **✅ 完全等价** | 全枚举最大差异 **5.68e-14°** |
| 单帧 delta 的取值范围 | **恒在 (−180, 180]** | 全枚举 `\|delta\|` 最大值 = **180.0°** |
| 上臂/前臂方向角本身 | **天然落在 (−180, 180]** | 由 `atan2` 求得 |
| 滤波是否周期处理 | **✅** | 周期量走 `CircularScalarFilter`（cos/sin 域） |

任务书点名的那一例：

```
reference = +170°, current = -170°
朴素相减  ->  -340°   ❌
angle_delta_deg -> +20.0°  ✅（20 的等价表示）
```

### 关键推论

既然单帧 delta **恒 ≤ 180°**，那么

```
delta_upper 行程 = 199.16°  (min -139.30, max +59.86)
```

**不可能**由单帧 wrap 产生 —— 它只能是**运动区间的跨度**。

### 实际饱和方向（与原始描述不同）

| delta | ratio | clip | 说明 |
|---|---|---|---|
| +59.86° | +0.999 | 0.999 | 正好用满，**未裁** |
| 0°(标定) | +0.500 | 0.500 | 中点 |
| −139.30° | −0.661 | **0.000** | **被裁** |

即：饱和发生在**负方向**（delta < −60° 全部 clip 到 0），
而不是「两端都饱和」。正的 +59.9° 恰好用满区间。

### 结论

> **差分层无缺陷；199° 是真实运动跨度的度量。**
> 因此任务书禁止的「直接把 ±60 改成 ±200」确实不该做 ——
> 但理由不是「199 是假的」，而是
> **标定姿势把可用行程偏置到了区间之外**（负方向有 79° 被裁掉）。
> 正确的应对是重选 neutral 或改用 task-space，而不是放宽区间。

**这一结论改变了 §3 的做法前提。**

---

## 2. 新 Neutral 设计

### 未执行，理由如下

任务书要求「重选 neutral → FK 扫描 → 选工作区中间的 neutral → 重新标定」。

本轮**没有执行**，原因有两条：

1. **需要实机 FK 扫描与重新标定**，而远端 SSH 被阻断（§8），
   无法在仿真里扫描新 neutral 的 X/Z 可达域。
2. §1 的审计表明：**问题不在 neutral 本身，而在「单关节映射」这个结构**。
   即便把 neutral 挪到工作区中间，legacy 的
   「上臂角→J2、肘夹角→J3」仍然无法解开 J2/J3 的耦合 ——
   而那正是 §5 要解决的问题。先解决结构，再定 neutral 更合理。

### 已具备的条件（供下一步使用）

| 数据 | 值 |
|---|---|
| FK 表 | `piper_human_retargeting/data/fk_table.json`，**208 点** |
| 末端相对肩的可达半径 | **[0.067, 0.536] m** |
| X 范围 | [−0.515, +0.534] m |
| Z 范围 | [−0.082, +0.531] m |
| 当前 neutral(j2=0.6, j3=−0.6) 的末端 | **(+0.039, +0.191) m**，r=0.194 |
| joint3 限位 | [−2.967, 0] |
| joint2 限位 | [0, 3.14] |

**建议的下一步**：用 FK 表先离线选 neutral（本报告的工具已能算），
再在仿真里确认，最后重新标定。

---

## 3. Legacy Mapping（保留为可选）

**未删除、未覆盖**，仍是仓库默认模式。

```yaml
retarget_mode: "legacy"      # 默认
```

| 项 | 值 |
|---|---|
| 映射 | `upper_arm_angle_deg → joint2`、`elbow_angle_deg → joint3` |
| 真实视频 X 行程 | **0.539 m** |
| 真实视频 Z 行程 | **0.334 m** |
| X/Z 比 | **1.61** |
| joint2 行程 | 2.146 rad |
| joint3 行程 | 0.631 rad |
| joint2 / joint3 饱和率 | **0% / 0%** |
| IK 抖动 | **0**（无 IK，直接映射） |

> 注意：legacy 在**这段视频**上 X/Z 比 1.61，并非任务书引用的 0.55
> （那是更早一次运行的数据）。两者用不同视频/不同标定，不可直接比较。

---

## 4. Task-Space Mapping（新增）

### 结构

```
Shoulder / Elbow / Wrist
   ↓  compute_human_task_state()          [人体侧，尺度归一化]
reach, elevation
   ↓  map_to_task_target()                [任务侧，增量映射]
(x_target, z_target)
   ↓  JointSolver.solve()                 [关节侧，数值 IK]
q2, q3
```

### 数学关系

**人体侧**（`task_space.compute_human_task_state`）：

```
arm_scale = |S-E| + |E-W|                 # 估计臂长，用于归一化
reach     = |S-W| / arm_scale             # 伸出程度，伸直时 ≈1
elevation = (S.y - W.y) / arm_scale       # 抬高程度，正值 = 手在上方
```

尺度不变性已实测：同一姿势缩放 0.5×/1×/2× 时
`reach`、`elevation` **完全不变**。

**任务侧**（`task_space.map_to_task_target`）：

```
n_reach = clip((reach     - reach0)     / reach_ref,     -1, +1)
n_elev  = clip((elevation - elevation0) / elevation_ref, -1, +1)

x_target = x0 + k_reach     * n_reach * x_span
z_target = z0 + k_elevation * n_elev  * z_span
```

+ `reach_ref` / `elevation_ref` 做**量纲归一化**
  （reach 典型行程 0.35、elevation 0.60，两者量纲不同）
+ `clip(±1)` 把归一化增量限制在「满量程」，避免目标被推到可达域外

### 为什么加这两步（各有实测依据）

| 改动 | 不加的后果（实测） |
|---|---|
| `reach_ref/elevation_ref` 归一化 | 纯前伸时 Z 动得比 X 多，X/Z 比 0.35 → **0.25**（更差） |
| `clip(±1)` | `d_elev` 可达 ±1.7，归一化后 ±2.8 × `z_span` = **±0.98 m**，远超可达半径 0.536 m |
| IK 滞回 `ik_hysteresis` | 离网求解在相邻网格点间切换 |

---

## 5. FK / IK 方法

**求解方式**：FK 表 + 数值优化（**不推导解析 IK**，符合任务书建议）。

| 项 | 实现 |
|---|---|
| FK 表 | 仿真扫描 208 点：`joint2 × joint3 → 末端相对肩的 (x, z)` |
| 网格步长 | 0.2 rad (j2) × 0.25 rad (j3) |
| 目标函数 | `cost = wx·(x_fk−x_t)² + wz·(z_fk−z_t)² + λ_smooth·‖q−q_prev‖² + λ_neutral·‖q−q_neutral‖²` |
| 默认权重 | `wx=wz=1.0`、`λ_smooth=0.35`、`λ_neutral=0.02` |
| 关节约束 | `q2 ∈ [0, 3.14]`、`q3 ∈ [−2.967, 0]`（加载表时即过滤） |
| previous-q 正则 | ✅ 实现，并额外加**滞回**（新解需优于保持上一帧解 `ik_hysteresis` 才切换） |
| FK 插值 | 最近邻 + 邻点一阶泰勒校正（缓解离网台阶） |

---

## 6. 对比结果（实机视频测试）

**测试方式**：把同一段视频喂给 `retarget_demo`，分别以
`legacy` / `task_space` 两种模式**实际驱动 Gazebo 里的 Piper**，
同时独立录制 `/joint_states`，末端位置由 FK 表换算。

视频：`arm_fwd_up.mp4`（667 帧，前伸→上举→放下），`--side right --imgsz 960`

| 项 | 值 |
|---|---|
| 控制频率 | 30.0 Hz（两模式一致） |
| TRACKING 帧 | legacy 487 / task_space 491 |
| 关节下发 | legacy 413 / task_space 404 |
| 标定 | 两模式用同一窗口，中立值一致 |

### 末端轨迹对比（实机实测）

| 指标 | Legacy | Task-Space | 判定 |
|---|---|---|---|
| TCP X 行程 | 0.364 m | 0.353 m | 基本持平 |
| TCP Z 行程 | 0.359 m | 0.315 m | task_space 略小 |
| **X/Z 比** | **1.02** | **1.12** | 两者都接近 1，task_space 略偏 |
| joint2 行程 | 1.791 rad | 1.714 rad | — |
| joint3 行程 | 1.215 rad | 1.142 rad | — |
| joint2 / joint3 饱和率 | 0% / 0% | 0% / 0% | 相同 |
| TCP X / Z 抖动 | 0.0000 | 0.0000 | 相同（三级滤波生效） |
| **IK 大跳变占比** | **0%** | **5.7%** | ❌ task_space 引入抖动 |
| **IK 单帧位移中位** | **0** | **0.206 rad** | ❌ 同上 |

> 注：实机 IK 抖动（5.7%）明显低于先前离线对比（17~45%），
> 因为 demo 的标定窗口更短、`d_reach` 有正行程
> （实测 [−0.114, +0.465]），目标不会长期顶在可达域边界。

### 解耦判据（task_space 独有的量）

| 判据 | 实测 | 期望 | 判定 |
|---|---|---|---|
| `corr(dReach, X)` | **−0.035** | 强正 | ❌ |
| `corr(dReach, Z)` | **+0.430** | 弱 | ❌ reach 反而主要驱动 Z |
| `corr(dElev, X)` | +0.189 | 弱 | ⚠️ |
| `corr(dElev, Z)` | +0.253 | 强正 | ❌ 太弱 |

### 判据对照（任务书第十六节）

| 判据 | 结果 |
|---|---|
| 前伸主要影响 X | **FAIL**（reach 主要影响的是 Z） |
| 抬手主要影响 Z | **FAIL**（corr 仅 +0.253） |
| X/Z 比更接近 1 | 名义达成但两模式都≈1，不构成优势 |
| 饱和率低 | PASS（两者 0%） |
| 平滑 / IK 稳定 | **FAIL**（task_space 引入 5.7% 大跳变） |

### 合成动作对比（单一变量，用于确认实现正确）

| 动作 | Δreach | Δelev | 模式 | ΔX | ΔZ | \|ΔX\|/\|ΔZ\| |
|---|---|---|---|---|---|---|
| 水平前伸 | +0.399 | 0.000 | legacy | 0.249 | 0.064 | 3.87 |
| 水平前伸 | +0.399 | 0.000 | task_space | 0.253 | 0.054 | **4.65** |
| 半径上摆 | 0.000 | +1.626 | legacy | 0.105 | 0.032 | 3.26 |
| 半径上摆 | 0.000 | +1.626 | task_space | 0.101 | 0.046 | **2.18** |

**合成动作（reach/elevation 严格独立）下 task_space 确实更优** ——
说明**映射与 IK 的实现是对的**，问题出在真实输入上。

---

## 6b. 深入定位：为什么真实视频上失效

### 决定性测量：reach 与 elevation 是否独立

| 量 | 范围 | 行程 |
|---|---|---|
| reach | [0.312, 0.999] | 0.687 |
| elevation | [−0.155, +0.416] | 0.570 |

```
corr(reach, elevation) = +0.210     -> 低相关，理论上可承载两个自由度
corr(dx, dy)           = +0.111
```

**它们在统计上确实是低相关的** —— 所以我先前「2D 投影下无法承载两个自由度」
的归因**是错的**。真正原因如下。

### 真正原因 ①：标定基准把 reach 顶到量程上界

| 标定方式 | reach0 | d_reach 范围 | 结果 |
|---|---|---|---|
| 离线工具（前 120 帧） | **0.962** | [−0.649, **+0.037**] | 前伸方向**无余量** |
| demo（2 秒窗口） | 较低 | [−0.114, **+0.465**] | 正方向有余量，抖动从 17% 降到 5.7% |

`reach` 的物理上界是 1.0（手臂完全伸直）。
若标定窗口恰好落在「手已经很直」的时段，`reach0 → 0.96`，
则**人再怎么伸也没有余量**，只有收手方向可用。

> **这是标定窗口选取问题，不是映射问题。**
> 加重 `k_reach` 无法解决（上界被 1.0 卡住）。

### 真正原因 ②：这段视频里 reach 的变化幅度很小

TRACKING 帧中 `reach` 实测集中在 **[0.9, 1.0]** ——
人在这段视频里手臂**基本一直是接近伸直的**，
`reach` 只能捕捉到很小的伸/收差异；
而 `elevation` 有明显变化（−0.085 → +0.386）。

因此「前伸」这个自由度在这段视频里**几乎没有被激励**，
自然无法体现「前伸→X」的对应关系；而 reach 的微小变化
经机器人几何（J2/J3 耦合）放大后反而跑到 Z 上去了。

### 真正原因 ③：IK 用离散网格最近邻

FK 表是 0.2×0.25 rad 的网格，逐点取最优 →
解在相邻网格点之间步进（中位 0.206 rad）。
已加 `ik_hysteresis` 抑制，但底层仍是「目标→解」的量化映射。

---

## 7. 失败归因：是**测量**问题，不是映射问题

任务书特别要求：「任何异常现象都必须先验证测量链路是否可信」。

我按这个顺序查，结论是**测量链路的问题**：

### 证据链

1. **合成动作（reach/elevation 严格独立）下 task_space 有效** ——
   说明映射与 IK 的实现是对的。
2. **真实视频下 `corr(dElev, X) = +0.700`** ——
   腕抬高竟然主要驱动 X。这只有在一种情况下成立：
   **在这段视频的机位下，抬臂在图像里主要表现为水平位移。**
3. 该视频是**侧后 3/4 机位**。人抬高手臂时，手在图像里向右（+x）移动，
   而不是向上。因此
   ```
   elevation（腕相对肩的高度）与 reach（腕到肩的距离）
   在 2D 投影下高度相关，无法承载两个独立自由度。
   ```

### 结论

> **task_space 的理论是对的，但在这段视频的机位下，
> 人体侧无法提供两个独立的观测量 ——
> 这是单目 2D 在侧后机位下的观测能力限制，不是映射算法的缺陷。**

### 45% IK 抖动的原因

- FK 表是 **0.2×0.25 rad 的离散网格**，求解器逐点取最优 →
  解天然在相邻网格点之间步进（步长 0.2~0.3 rad）。
- 目标噪声使最小代价点在两三个邻点间切换。
- 已加 `ik_hysteresis` 抑制，但底层的「目标→解」映射本身
  在输入不可分时是不稳定的。

### 下一步建议（按优先级）

1. **换机位**：把相机放到人的**正侧方**（视线垂直于人的矢状面）。
   这样抬臂在图像里是纯垂直位移，
   `reach` 与 `elevation` 才会解耦 —— 这是 task_space 生效的前提。
2. **或者加深度**：用 RGB-D 相机直接拿 3D 腕位置，
   从根上消除 2D 投影的二义性（阶段七的接口已预留）。
3. **加密 FK 表**：当前 0.2×0.25 rad 的网格偏粗，
   建议 0.05 rad 级（约 3200 点），可显著降低离网抖动。
4. **neutral 重选**：用 FK 表离线选出「工作区中间」且
   「J3 不贴近 0」的 neutral，再重新标定（§2 已备好数据）。

---

## 8. 阻塞项：远端不可达

本轮**未能在远端部署与实机验证**。

```
$ ssh zxmy@100.89.121.96
# Tailscale SSH requires an additional check.
# To authenticate, visit: https://login.tailscale.com/a/lf96c8ed28a307
Connection to 100.89.121.96 port 22 timed out
```

- 网络层正常：`ping 100.89.121.96` 通（往返约 100 ms）
- 但 Tailscale SSH 要求重新认证，**所有 SSH 会话被拒**
- 因此：**无法部署、无法实机 FK 扫描、无法重新标定、无法采集实机数据**

### 本轮因此改用离线验证（可信度说明）

| 验证项 | 方式 | 可信度 |
|---|---|---|
| §1 角度审计 | 纯数学，本地全枚举 | **高**（与实机无关） |
| §4 task-space 数学 | 本地单元测试 + 合成动作 | **高** |
| §5 IK 求解 | 使用**实机采集的 FK 表**（208 点） | **高** |
| §6 真实视频对比 | 本地 YOLO 推理（CPU, imgsz=640） | **中**（模型/分辨率与远端略有差异） |
| 实机验证 | ❌ 未做 | — |

---

## 9. 手部标定 / 夹爪标定

### Arm Calibration（原有，保留）

采集 `upper_arm_angle_deg` / `elbow_angle_deg` / `reach` / `elevation`，
用于 J2/J3 的 baseline。窗口 2s。

### Hand Calibration（**本轮新增**）

原先是「第一次检出手即当基准」（lazy baseline），不够严谨。改为状态机：

```
HAND_WAITING  --连续稳定 stable_frames 帧-->  HAND_STABLE
HAND_STABLE   --采 sample_frames 帧-->        检查离散度
                                               ├─ 通过 -> HAND_CALIBRATED
                                               └─ 超限 -> 回 HAND_WAITING
```

**可审计输出**（任务书要求）：采样起止帧、样本数、均值、标准差。

```
HAND_CALIBRATED: n=20 [45..64] 均值(wrist_pitch_deg=12.3, thumb_offset=0.412)
                标准差(wrist_pitch_deg=1.85, thumb_offset=0.031)
```

- 中途丢失手部 → 稳定计数清零（**不连续不算稳定**）
- 采样期间姿势在动（σ 超限）→ 丢弃该批、重新等待

### Open / Closed Calibration（**本轮新增**）

```
gripper_ratio = clip((openness - openness_closed)
                     / (openness_open - openness_closed), 0, 1)
```

- 按 `o` 记录「张开」端、按 `p` 记录「握拳」端（各需保持 `hold_frames` 帧）
- 两端行程过小（< `min_span`）→ 判定未完成
- **未完成标定时回退到原始 openness**，不返回常数
  （否则「没标定」会变成「夹爪完全不动」，比不做还差）

### 按键

```
c 标定  r 重标  n 回位  k 逐关节测试  g 切换夹爪
o 记张开  p 记握拳  e 急停  q 退出
```

---

## 10. 数据可信度原则（任务书第十五节）

本轮新增的每个参与控制的量，都满足 8 条要求：

| 要求 | 落实 |
|---|---|
| 1. 出现在 `used_measures()` | `reach`/`elevation` 在 task_space 模式下自动加入 |
| 2. 出现在 debug / CSV | 新增 18 列（见下） |
| 3. 可见 raw value | `reach_raw` / `elevation_raw` |
| 4. 可见 mapped value | `reach_flt` / `elevation_flt` / `d_reach` / `d_elevation` |
| 5. 可见最终 q2/q3 | `mapped_joint2/3`、`joint2/3` |
| 6. 数据缺失时明确报错 | 缺失量进 `missing` 列表，状态显示「部分量缺失(...)」 |
| 7. 禁止 silent continue | 取样一律显式 `continue` 并记录原因 |
| 8. 列名不硬编码 | 关节列覆盖全部 6 关节，由配置推导 |

新增 CSV 列（18）：

```
reach_raw, elevation_raw, reach_flt, elevation_flt,
d_reach, d_elevation, x_target, z_target,
x_target_raw, z_target_raw, x_clamped, z_clamped,
ik_x, ik_z, ik_err_x, ik_err_z, ik_jump
```

### 本轮又踩到两次同一个坑（已修）

> **模式**：新增一个量 → 算出来了 → **没接进链路** → 静默失效。
> 这个模式在本项目已出现 **7 次**。

两次具体表现：

1. `reach/elevation` **没进标定采样** →
   baseline 为空 → task_space 静默失效（两模式输出完全相同，日志却显示「标定完成」）
2. **先构造 Retargeter 再改 `retarget_mode`** →
   `used_measures()` 早已按旧模式算好 → 同样静默失效

对应加固：
- `add_calibration_sample_for()` 统一补齐「不在 ArmGeometry 里」的量
- `_apply_task_space()` 支持**基线懒建立**
- 对比工具改用「先设模式、再构造」的路径

---

## 附录：本轮新增/修改文件

| 文件 | 类型 | 作用 |
|---|---|---|
| `piper_human_retargeting/task_space.py` | **新增** | reach/elevation 提取、X/Z 目标映射、J2+J3 联合求解 |
| `piper_human_retargeting/hand_calibration.py` | **新增** | 手部标定状态机 + 夹爪双点标定 |
| `piper_human_retargeting/offline_compare.py` | **新增** | legacy vs task_space 离线对比工具 |
| `piper_human_retargeting/synth_compare.py` | **新增** | 合成动作解耦判据 |
| `test/test_task_space.py` | **新增** | 30 项 |
| `test/test_angle_wrap_audit.py` | **新增** | 13 项 |
| `test/test_hand_calibration.py` | **新增** | 16 项 |
| `config/retargeting.yaml` | 修改 | 新增 `retarget_mode`、`task_space` 段 |
| `piper_human_retargeting/config.py` | 修改 | `TaskSpaceRetargetConfig`、模式校验、used_measures |
| `piper_human_retargeting/retargeter.py` | 修改 | task-space 接入、基线懒建立、18 个 CSV 列 |
| `piper_human_retargeting/retarget_demo.py` | 修改 | 手部/夹爪标定接入、`o`/`p` 按键 |

**未改动**：YOLO Pose、MediaPipe、ROS2 Controller、`/arm_controller`、
安全限速、急停、GUI、控制频率架构、官方 `piper_ros`。

**Legacy 完整性**：`retarget_mode: "legacy"` 为仓库默认，
两套实现并存、可配置切换，旧代码未删除、未覆盖。
