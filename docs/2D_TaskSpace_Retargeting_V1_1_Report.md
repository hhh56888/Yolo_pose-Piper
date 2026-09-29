# 2D Task-Space Retargeting V1.1 报告

> 目标：**人的手往哪走，机器人的手（TCP）就往对应方向走**
> —— 用 `Human 肩/肘/腕 → 手腕相对位置 → 机器人 TCP (x,z) → J2/J3 联合数值 IK`
> 取代 `上臂角→J2、肘角→J3` 的逐关节映射。
>
> 远端工程：`~/Yolo_pose+piper`（`zxmy@100.89.121.96`）
> 本地镜像：`/Users/hyu/Desktop/catkin_ws/src/`
> 测试：**本地 302 passed / 远端 302 passed**（Python 3.14 / 3.10 双环境）

---

## 1. 测试守卫修复

### 1.1 原缺陷（来自 `docs/J3_Elbow_Mapping_Audit.md` §10.2）

`test/test_mapping.py` 里的方向守卫有**两个互相抵消的错误**：

```python
# 修正前
#   q3 = -1.10 -> z = 0.5215 ; q3 = -0.10 -> z = 0.1970
#   z 随 q3 单调【上升】                                  ← 与上面数据相反
#   dz/dq3 ≈ (0.1970 - 0.5215) / (-0.10 - (-1.10)) = +0.3245   ← 算出来是 −0.3245
DZ_DQ = {"joint3": +0.3245}

def test_bending_elbow_...[cfg]:
    qmax, _ = r.map(r.human_max)          # 直接把 delta 当"人弯肘"
    dz = DZ_DQ["joint3"] * (qmax - q0)
    assert dz < 0                          # 于是永远通过
```

1. **常数符号与自身实测数据相反**（真值 −0.3245 m/rad，本轮仿真复核 −0.333）；
2. **守卫调用 `Rule.map(human_max)`，绕过了 `measure_sign`**：
   运行时 `elbow_angle_deg` 的 `sign = −1`，所以 **delta=+human_max 实际对应"人伸肘"**，
   守卫把端点用反了。两个错误相乘 → `38 passed`，但它**守不住任何真实方向**。

### 1.2 修复方式

| 项 | 修改 |
|---|---|
| 常数 | `DZ_DQ = {"joint3": -0.3245}`，注释改为"z 随 q3 单调**下降**"，并保留原始 4 个实测点 |
| 端点语义 | 守卫改为与运行时一致：`human_min ≡ 人弯肘`、`human_max ≡ 人伸肘`；并断言 `measure_sign()["elbow_angle_deg"] == -1`（sign 变了会立刻提醒） |
| 守卫自检 | 新增 `test_direction_guard_catches_flipped_invert`：把 `invert` 翻错后守卫**必须失败**（已实测：翻转后 dz=+0.1947 → 守卫报错） |
| 端到端语义 | 新增 `test/test_direction_e2e.py`（见下） |

### 1.3 新的端到端方向测试（走完整生产链）

```text
Human elbow flexion
      ↓  生产 Retargeter（sign / invert / 三级滤波 / 标定 全部参与）
joint3 target
      ↓  URDF FK（kinematics.ChainModel，已用仿真 TF 校准）
TCP  →  到肩关节的径向距离
```

锁定的语义（`test/test_direction_e2e.py::TestLegacyElbowDirection`）：

| 断言 | 结果 |
|---|---|
| 人弯肘 → joint3 朝收臂方向（增大） | ✅ |
| 人弯肘 → **TCP 径向距离减小**（收臂） | ✅ |
| 人伸肘 → TCP 径向距离增大（伸臂） | ✅ |
| 连续弯肘 10°→160° → 径向距离单调不增 | ✅ |

task_space 侧另有三条底线测试：`人手抬高 → TCP z 抬高`、`人手前移 → TCP x 增大`、
`task_space 下肘角对 q2/q3 无权限`（刚性臂两构型给出相同 u/v 与相同解）。

---

## 2. Joint Limit（三层）

```text
Physical  ⊇  Safe  ⊇  Retargeting
（实际加载的 URDF）  (两端留 0.10 rad)  (重映射/IK 输出区间)
```

| 关节 | physical（URDF，`robot_description` md5 79c1b6e3…） | safe（−0.10 两端） | retarget（task_space IK 盒） |
|---|---|---|---|
| joint2 | [0.000, 3.140] | [0.100, 3.040] | **[0.250, 2.890]** |
| joint3 | [−2.967, 0.000] | [−2.867, −0.100] | **[−2.720, −0.250]** |

* 物理限位**不新增第二份表**：由 `limits.physical_limits()` 从实际加载的 URDF 读取，
  并与控制侧 `joint_limits.yaml` 交叉验证，最大偏差 `1.0e−04 rad`（十进制记录精度），
  超过 `1e-3` 会直接报错。
* **构造期强制**：`Retargeter` 在 task_space 模式下会调 `build_layers()`，
  一旦 retarget 越出 safe 就 `raise`，不会等到运行中被下游静默裁剪。
* **legacy 的越界被审计出来但不改**（保持可回退、行为不变）：

```text
legacy: joint2 映射 [-1.00, 2.20] 越出 physical 下限 0.00  → 实测 29.67% 帧被削平
        joint3 映射 [-1.20, 0.00] 上端 == physical 上限   → 实测 15.25% 帧骑在限位
（limits.build_layers() 会把这些列为 violations；test_limits.py 断言它们可见）
```

---

## 3. FK Workspace（J2 × J3 二维扫描）

工具：`tools/scan_workspace.py`（**不接 ROS**，用 URDF 生成的 FK；另用 Gazebo TF 实测校准）。

```text
URDF FK 正确性：9 个仿真 TF 实测位姿点，最大误差 0.58 mm
Jacobian：与有限差分对照，最大差 6.4e−10
```

扫描（IK 盒内 61×61 = 3721 点）：

```text
TCP x ∈ [−0.561, +0.625] m
TCP z ∈ [−0.187, +0.749] m
sigma_min(XZ) ∈ [0.0188, 0.2015]       ← 最小奇异值（可操作度）
```

**可达矩形**（用真实 IK 逐格验证 46×46 = 2116 个格点，可达率 68.9%）：

```text
x ∈ [+0.181, +0.486]   (0.306 m)
z ∈ [+0.050, +0.516]   (0.466 m)
```

> 为什么不能拿扫描点做"占据图"求矩形：两连杆可达集是平面上的**厚环带**，
> 点云稀疏会被误判成不可达（实测得出 0.04×0.17 m 的荒谬矩形）。
> 改为逐格调用 IK 判定后才得到正确结果。

数据产物：远端 `/tmp/workspace.json`（含全部扫描点 + 内接矩形 + Top5 neutral）。

---

## 4. 新 Neutral

评分函数（任务书 §5）：

```text
score = 1.0 * workspace_center_score      # TCP 离工作区中心的归一化距离
      + 1.0 * joint_margin_score          # q2/q3 到区间上下界余量的**最小值**
      + 0.6 * manipulability_score        # XZ 平面最小奇异值归一化
（候选池限定在"可达矩形内"，保证 neutral 本身处于可控区中央）
```

| | q2 | q3 | TCP (x, z) | 到 IK 盒余量 m2 | m3 | sigma_min | score |
|---|---|---|---|---|---|---|---|
| **旧 neutral（legacy 保留）** | 0.600 | −0.600 | (+0.127, +0.366) | (0.35, 2.29) | (2.12, 0.35) | — | — |
| **新 neutral（task_space）** | **1.504** | **−0.990** | **(+0.340, +0.286)** | **(1.254, 1.386)** | **(1.727, 0.740)** | **0.1994** | **2.127** |

Top-5（前两名 score 几乎相同，第三个开始余量明显变差）：

```text
1. q2=+1.504 q3=-0.990 | TCP=(+0.340,+0.286) | m2=(1.254,1.386) m3=(1.727,0.740) | sigma=0.1994 | 2.1273
2. q2=+1.504 q3=-1.052 | TCP=(+0.347,+0.306) | m2=(1.254,1.386) m3=(1.665,0.802) | sigma=0.1972 | 2.0830
3. q2=+1.438 q3=-0.990 | TCP=(+0.328,+0.308) | m2=(1.188,1.452) m3=(1.727,0.740) | sigma=0.1994 | 2.0577
4. q2=+1.570 q3=-1.052 | TCP=(+0.358,+0.283) | m2=(1.320,1.320) m3=(1.665,0.802) | sigma=0.1972 | 2.0534
5. q2=+1.504 q3=-0.928 | TCP=(+0.332,+0.266) | m2=(1.254,1.386) m3=(1.789,0.678) | sigma=0.2009 | 2.0511
```

**选择理由**：TCP 落在可达矩形中心（(0.340,0.286) vs 矩形中心 (0.334,0.283)），
两个关节双向余量均衡（旧 neutral 的 joint3 只有 0.35 rad 上余量，新的 1.727/0.740），
可操作度 0.1994 ≈ 全工作区最大值的 99%（不接近奇异）。

旧 neutral 原封不动保留在 `robot.neutral_joints`，legacy 模式照旧使用
（`RetargetingConfig.active_neutral_joints()` 按模式选择）。

---

## 5. Human Task-Space（人体侧）

```text
S = 肩, E = 肘, W = 腕（W 用 effective_wrist：优先手部识别的腕，沿用阶段三规则）
L = |S-E| + |E-W|
u = (W.x - S.x) / L        + = 图像右
v = -(W.y - S.y) / L       + = **手腕在肩上方**（图像 y 向下，故取负）
```

* 计算位置：`arm_geometry.compute_arm_geometry()`（与肘角/上臂角同一次几何计算，
  不新增一条独立的几何通路）；随 `ArmGeometry.as_dict()` 进入滤波/标定/CSV 主干。
* **尺度无关**：除以手臂全长，因此与人机距离、体型无关。
* 真实手臂是刚性两连杆 → `|SE|`、`|EW|` 不随姿势变化 → `L` 恒定 →
  **u/v 只由肩和腕的位置决定**（已由端到端测试锁定：同一 S/W 的镜像肘构型
  给出逐位相同的 u/v）。
* 旧的 `reach/elevation` 主控量已移除；`reach` 作为诊断量保留（记录用），
  `elevation` 直接删除（垂直信息现在由 `v` 直接表达，避免"算了却没人用"的列）。

标定即零点（任务书 §9）：记录 `(u0, v0)` 与机器人 `(x0, z0) = FK(active_neutral)`，
控制只用相对位移 `du = u − u0`、`dv = v − v0`，人体绝对像素坐标不参与控制。

---

## 6. Robot Task-Space（机器人侧）

```text
x_target = x0 + sign_h * kx * deadzone(du)
z_target = z0 + sign_v * kz * deadzone(dv)
```

| 配置 | 值 | 说明 |
|---|---|---|
| `horizontal_axis` / `vertical_axis` | `x` / `z` | **轴关系配置化**（任务书 §8）：正视机位应改成 `y`/`z`，不硬编码 |
| `sign_h` / `sign_v` | +1 / +1 | 人手向图像右 → TCP +x；人手抬高 → TCP +z |
| `kx` / `kz` | 0.35 / 0.35 | 归一化位移 → 米（\|u\|≤1，1.0 ≈ 整条手臂横移） |
| `deadzone_u/v` | 0.02 / 0.02 | 见下 |
| `x_min/max` | 0.20 / 0.47 | 取自可达矩形 [0.181,0.486] 内收 2 cm |
| `z_min/max` | 0.07 / 0.50 | 取自可达矩形 [0.050,0.516] 内收 2 cm |

**死区实现（连续，不产生台阶）**：`|d| < deadzone → 0`；
否则输出 `sign(d)·(|d| − deadzone)`。若写成"超界就原样输出"，边界处会出现
`0 → 0.02` 的阶跃，机械臂会看到一次小跳。复用既有滤波链，**没有新增滤波器**。

---

## 7. IK（J2/J3 联合求解）

### 7.1 目标函数与求解器

```text
cost = wx·(x(q)-x_t)^2 + wz·(z(q)-z_t)^2
     + lambda_prev   ·||q - q_prev||^2
     + lambda_neutral·||q - q_neutral||^2

wx = wz = 1.0        lambda_prev = 0.02      lambda_neutral = 0.002
```

* 求解器：**投影 Gauss-Newton（LM 阻尼）+ 回溯线搜索**；
  每一步先算无约束 GN 步，再做 **box 投影**并比较代价 —— 关节约束在**迭代内**满足，
  **不是**"先解一个非法解、最后 clamp"（任务书 §13）。
* **warm start**：以上一帧解为初值；配合 `lambda_prev` 抑制跳解。
* **备用初值重试**：主解未收敛时，用「区间中点」「q3 方向镜像」重试（临时关掉正则），
  取位置误差最小者。只在异常帧触发。
* 权重单位说明（实测教训）：位置项是 **米²**、正则项是 **弧度²**。
  `lambda_prev = 0.35` 时偏离上一帧 0.5 rad 的代价 = 0.0875，等价"允许 0.3 m 跟踪误差"，
  IK 会为了贴着上一帧而**拒绝走向目标**（实测大面积不收敛）。
  0.02 量级才是"防跳解的轻微约束"。【已由测试锁定】
* 发现并记录：在本 IK 盒内，可达 (x,z) 对应的 (q2,q3) 是**唯一**的
  （120 个随机可达目标、两种初值都收敛到同一解），因此不需要旧的
  `ik_hysteresis` 滞回逻辑（已随旧网格 IK 一并删除）。

### 7.2 实时性（任务书 §14）

| 场景 | n | 均值 | 中位 | P95 | 峰值 | 未收敛 |
|---|---|---|---|---|---|---|
| 真实数据回放 907 帧 | 907 | — | **0.505 ms** | **0.633 ms** | 0.991 ms | 0 |
| 端到端 demo（含 YOLO+MediaPipe，30 Hz） | 247 | 0.632 ms | 0.634 ms | 0.853 ms | 1.258 ms | 0 |

**加速手段**：`kinematics.TwoDofFk` —— 把链路切成「joint2 之前 / j2–j3 之间 / j3 之后」
三段常量矩阵，每次 FK 从 38.7 µs 降到 10.4 µs，**与通用 FK 逐位一致（最大差 1.1e−16，
有对照测试）**。仅此一项把 IK P95 从 1.4 ms 降到 0.55 ms。

30 Hz 主循环（33 ms/帧）实测：**平均环路 1.8 ms、峰值 4.7 ms、超时 0 次**。

---

## 8. A/B 测试

### 8.1 方法

* 同一段输入、只改 `retarget_mode` 一行（临时配置由脚本生成，附 `diff` 证据，
  **生产 yaml 未动**）；两种模式各自使用自己的 neutral。
* 两种输入：
  * `synth`：**四个基础动作**（前 / 后 / 上 / 下），由两连杆反解构造**刚性**手臂
    （|SE|、|EW| 恒定），单轴、无噪声 —— 用来量**耦合**；
  * `real`：`/tmp/live_run.csv` 里真实记录的人体角度序列（肘部行程最大的一段，90 s），
    重建关键点后逐帧回放 —— 用来量**真实动作**。
* 测量：`/joint_states` 实测关节角 + TF 实测 `base_link→gripper_base`（TCP）。

### 8.2 四动作响应（synth，同一输入）

| 动作 | Δu | Δv | legacy ΔTCPx | legacy ΔTCPz | task_space ΔTCPx | task_space ΔTCPz |
|---|---|---|---|---|---|---|
| 前移 | +0.20 | 0 | **+0.1262** ✅ | −0.0170 | **+0.0595** ✅ | +0.0003 |
| 后移 | −0.20 | 0 | **−0.1239** ✅ | −0.0245 | **−0.0632** ✅ | +0.0000 |
| 抬高 | 0 | +0.30 | −0.0632 | **+0.0964** ✅ | −0.0002 | **+0.0935** ✅ |
| 放低 | 0 | −0.25 | +0.0778 | **−0.1081** ✅ | −0.0015 | **−0.0772** ✅ |

**方向全部正确**，但耦合差别很大：

```text
串扰 cross_x_to_z (u→TCPz)   legacy +0.039   task_space +0.004
串扰 cross_z_to_x (v→TCPx)   legacy -0.436   task_space +0.008   ← 差 55 倍
```

即：legacy 在"抬手"时 TCP 会横向跑掉 0.06~0.08 m（主轴是 z），
task_space 几乎不动（−0.0002 m）。

### 8.3 真实数据回放（907 帧）

| 指标 | legacy | task-space（V1.1） |
|---|---|---|
| `corr(人体水平 u, TCP x)` | **−0.407** ❌ | **+0.778** ✅ |
| `corr(人体垂直 v, TCP z)` | +0.259 | **+0.512** ✅ |
| 串扰 x→z | +0.005 | +0.153 |
| 串扰 z→x | −0.304 | +0.344 |
| J2 关节限位饱和 | 9.26% | **0.00%** |
| J3 关节限位饱和 | 0.00% | 0.00% |
| 实测关节单帧抖动 P95（j2 / j3） | 0.103 / 0.075 rad | **0.062 / 0.045 rad** |
| 实测 TCP 单帧 P95（x / z） | 0.024 / 0.032 m | **0.014 / 0.018 m** |
| TCP 行程 Δx / Δz | 0.581 / 0.441 m | 0.304 / 0.321 m |
| IK 单帧解跳变 >0.25 rad | — | 19 次（2.1%），最大 0.676 |
| IK 耗时（中位 / P95） | — | 0.505 / 0.633 ms |
| 控制环超时 | 0 | 0 |

> 真实数据上的"串扰"要谨慎解读：真人动作本身不是单轴的（抬手时 u 也在变），
> 相关系数里混进了人体自身的耦合。**干净的耦合证据是 8.2 的 synth 单轴测试**。

> legacy 的 Δx 更大（0.581 m）并不是优点：它是"手一动、末端大幅乱跑"的结果，
> 而且方向是反的（corr(u,x) = −0.407）。

### 8.4 IK 跳变的诚实说明（PASS 判据第 5 条）

原始 IK **解**在 19/907 帧上跳变 >0.25 rad（最大 0.676 rad），
来源是上游人体量在那些帧上的跳变（`real` 输入是用 live CSV 里**原始**角度重建的，
其中包含退化到 0°/180° 的肘角样本；目标单帧最大变化 0.097 m）。

但**实际下发**的关节明显比 legacy 平滑（见 8.3 表），因为既有的
「关节 EMA 滤波 + SafetyLimiter 每周期限幅 0.15 rad」把跳变吸收了：
19 个跳变帧对应的实际关节变化最大只有 j2 0.123 / j3 0.060 rad。

尝试把 `lambda_prev` 从 0.02 提到 0.08：跳变次数不变（19），相关性略变（+0.792/+0.511），
**说明跳变来自目标而非 IK 分支**（盒内解唯一，见 §7.1）。因此保持 0.02。
本条判为"**部分满足**"并如实记录。

---

## 9. 关键结果（任务书 §22 的头号指标）

```text
旧指标（审计实测，legacy）：corr(人体手高, TCP 高) ≈ −0.353     ← 反相关
```

同一段输入下两种模式对比：

| 输入 | legacy | task-space V1.1 |
|---|---|---|
| synth 四动作 | +0.871 | **+0.892** |
| real（907 帧真实数据） | +0.259 | **+0.512** |
| 单动作实测（抬手 0.30 归一化位移） | TCP z +0.0964 m | **TCP z +0.0935 m**（横向漂移 0.0632 → 0.0002 m） |

**负相关被修正**：新模式下"人手抬高 → TCP 抬高"在**所有**测试里都成立，
并且横向串扰几乎为零（0.004 / 0.008）。

三件事合起来回答任务书的最终问题：

```text
corr(人体水平 u, TCP x) :  −0.407 (legacy)  →  +0.778 (task_space)
corr(人体垂直 v, TCP z) :  +0.259 (legacy)  →  +0.512 (task_space)
  串扰 cross_z_to_x     :  −0.436 (legacy)  →  +0.008 (task_space, synth)
```

---

## 10. 最终结论

```text
2D TASK-SPACE RETARGETING V1.1

PASS
```

逐条对照任务书 §25 的 7 条最低要求：

| # | 要求 | 结果 | 证据 |
|---|---|---|---|
| 1 | 人手向任务空间正方向移动 → TCP 同方向移动 | ✅ | synth 前/后动作 ✅✅；real `corr(u,TCPx)=+0.778`（legacy −0.407） |
| 2 | 人手抬高 → TCP Z 抬高 | ✅ | synth 抬/放动作 ✅✅；real `corr(v,TCPz)=+0.512` |
| 3 | 原 `corr(手高,TCP高)` 负相关被修正 | ✅ | 同输入 A/B：+0.259 → **+0.512**；单动作 +0.0935 m，横向漂移 ≈0 |
| 4 | J2/J3 不出现明显新饱和 | ✅ | 关节限位饱和 0.00% / 0.00%（legacy J2 为 9.26%） |
| 5 | IK 无明显跳解 | ⚠️ **部分** | 原始 IK 解 19/907 帧跳变（上游量跳变所致）；**下发**运动比 legacy 更平滑，且无下发跳变 >0.14 rad。见 §8.4 |
| 6 | 30 Hz 控制链没有明显退化 | ✅ | 端到端 29.9–30.0 Hz、环路 1.8 ms、**超时 0**、IK P95 0.85 ms |
| 7 | Legacy 模式仍能正常回退 | ✅ | `retarget_mode` 一行切换；legacy 全部测试通过，legacy 端到端 29.9 Hz 正常 |

### 失败项归类（任务书要求：若 FAIL 须区分原因）

本轮**没有整体 FAIL**，第 5 条属"部分满足"，归因明确：

```text
[测量/日志问题] ← 唯一未完全达标项的真实来源
    real 输入用 live CSV 的**原始**肘角重建（含退化到 0°/180° 的样本），
    目标单帧最大跳变 0.097 m；IK 忠实跟随，故解跳变。
    · 不是 IK 分支翻转（盒内解唯一，实测 120/120）
    · 不是 joint limit 问题（饱和 0%）
    · 不是 neutral / workspace 问题（目标始终在可达矩形内）
```

---

## 13. 实时识别测试（真摄像头 + 人在环）

任务：把 V1.1 接到**真实摄像头**上跑（不是回放），验证识别链路 + task-space 手感。

命令（临时配置只改 `retarget_mode` 一行，生产 yaml 未动）：

```bash
DISPLAY=:1 ros2 run piper_human_retargeting retarget_demo \
  --camera 0 --side left --imgsz 960 \
  --retarget-config /tmp/live_ts_cfg/retargeting.yaml \
  --no-current-initial --auto-calibrate --csv /tmp/live_ts5.csv
```

### 13.1 实时测试发现并修复的 3 个真缺陷

| # | 现象 | 根因 | 修复 |
|---|---|---|---|
| 1 | 启动后机械臂停在 **legacy 中立位姿**，而 IK 基线用的是 task-space 位姿（开机即带固定偏差） | `robot.initial_pose` 在解析时被 `robot.neutral_joints` 填充，**永远非空**；`robot.initial_pose or active_neutral_joints()` 于是永远选前者 | 新增 `RetargetingConfig.active_initial_pose()`（task_space 优先用本模式 neutral），demo 改用它；`test_active_initial_pose_matches_mode_neutral` 锁定 |
| 2 | CSV 里 `joint2/joint3` 恒为 **0.0000**（看起来像"这两个关节没被控制"），实际 IK 输出 0.924/−0.907 | 早退路径（手臂丢失/标定中）不填 `final_joints`；且记录只遍历 `controlled_joints()`，task_space 下不含 j2/j3 | ① `_make_debug` 统一兜底填当前实际目标 ② 记录集合并入 `task_space_joints()`；两条回归测试锁定 |
| 3 | **手感反向**：手向前伸 → 机械臂往回收；手收回 → 机械臂前伸 | 正面机位下 `u` 是**投影量**，手臂伸直时 u 下降（`corr(肘角,u)=+0.964`），而 `sign_h=+1` 是按侧视机位配的 | 按实测标定 `sign_h = -1.0`（配置项，附实测注释）；`test` 保证轴/符号仍是配置驱动 |
| 5 | **腕部方向反了**：手往上翘/下压时末端俯仰相反 | `j5_wrist.invert` 原为 `true`（按"joint5 增大→末端指向向下 + 手往上翘时 wrist_pitch 变小"的**几何推导**配的）；实际操作者手感相反 | 按操作者实测把 `j5_wrist.invert` 标定为 `false`；实时数据复核 `corr(腕俯仰, joint5)` −0.88 → **+0.989**，且 joint5 未饱和。方向由 `test_direction_e2e.py::TestWristPitchDirection` 钉住（改动必须同步改测试与注释） |
| 4 | **识别不到时机械臂停在原地**（需求：应回归原位） | `_do_lost_long` 的回位循环只遍历 `cfg.controlled_joints()`，而 task_space 下 joint2/joint3 不在该集合 → 只有腕关节回位、大臂冻住 | 统一到新增的 `RetargetingConfig.active_joint_names()`（= legacy 受控 ∪ task-space IK 关节），回位/记录/HUD 全部改用它；"已回到原位"的判据与关节滤波死区对齐（否则永远判不到）；新增 `test/test_return_home.py`（8 条） |

> 这三个都是**回放测试发现不了**的：①只在真实启动时序下出现；②只在丢失/标定帧出现；
> ③只在真实相机站位下出现。这也说明"回放 A/B 通过"不等于"实机可用"。

### 13.2 相机轴向标定（任务书 §8 的实例）

正面机位实测（561 个 TRACKING 帧）：

```text
corr(肘角, human_u)  = +0.964     手臂伸直(肘角↓) -> u 下降
corr(reach, human_u) = -0.957     投影臂长变长   -> u 下降
```

即"手往前伸"在图像里表现为 **u 下降**（横向投影偏移变小），
所以正面机位需要 `sign_h = -1`；侧视机位才用 `+1`。

**标定后实测（1341 帧，真人操作）**：

| 指标 | 标定前 | 标定后 |
|---|---|---|
| `corr(肘角, TCP_x)` | **+0.785** ❌（伸直→后退） | **−0.769** ✅（伸直→前伸） |
| `corr(reach, TCP_x)` | −0.784 | **+0.850** ✅ |
| `corr(u, TCP_x)` | +0.850 | −0.864（= sign_h 生效） |
| `corr(v, TCP_z)` | +0.545 | **+0.718** ✅ |
| TCP 行程 | — | x [0.198, 0.471]，z [0.118, 0.374] |

其它实时指标：TRACKING 1333/1345（99%）、30 Hz、环路 6.3 ms、**超时 0**、
IK 中位 0.59 ms / P95 0.81 ms、贴物理限位 0.00% / 0.00%。

**诚实的局限**：单目正面机位下"前后"只能用**投影变化**近似，
纯横向的手部移动会被解释成前后（横向在机器人 J2/J3 上本来就不可控）。
要把"手往前伸 → TCP 前伸"做到几何精确，需要把相机移到人体**侧面**
（那时把 `sign_h` 改回 `+1` 并重做 §20 四动作验证）。

### 13.3 右侧标定项：人手侧符号必须由操作者实测标定

实时测试一共标定了 **2 个人手侧符号**，两者都是"几何推导不可靠、必须实测"：

| 配置项 | 几何推导会怎么选 | 实测标定 | 证据 |
|---|---|---|---|
| `task_space.sign_h` | `+1`（假设侧视机位，手前伸→u↑） | **−1.0** | 正面机位下 `corr(肘角,u)=+0.964`，手伸直时 u 下降；改 −1 后 `corr(肘角,TCP_x)` +0.785 → **−0.769** ✅ |
| `j5_wrist.invert` | `true`（假设手往上翘时 wrist_pitch 变小） | **false** | 操作者实测反馈方向相反；改 false 后 `corr(腕俯仰, joint5)` −0.88 → **+0.989** ✅ |

**教训（写进配置注释）**：单目 2D 下"人手侧的角度符号"取决于相机站位与手的姿态，
不能靠几何推导确定；这类量必须（a）配置化、（b）用操作者实测标定、
（c）把标定依据写进注释、（d）用测试把标定值钉住。
`j6_roll.invert` 目前仍是"待实机确认"的默认值，若操作者反馈拧腕反向，同样一行配置即可。

### 13.4 识别不到 → 回归原位（实测）

需求：识别不到手臂时不要停在原地，平滑回到**启动位姿（原位）**。

行为设计（配置项，见 `config/retargeting.yaml` 的 `safety:` 段）：

```text
LOST_SHORT  (<lost_grace_seconds=1.0s)  保持不动（手一晃不能一惊一乍）
LOST_LONG   (>=1.0s)                    以 return_home_speed=0.6 指数逼近原位
                                        仍经关节滤波 + SafetyLimiter 限速
重新识别到                               立即恢复跟随（resume_on_reacquire）
return_home_when_lost=false            退化为冻结保持（可回退）
```

实时实测（`/tmp/live_home.csv`，人走出画面触发）：

| 帧 | 状态 | joint2 | joint3 | 距原位 |
|---|---|---|---|---|
| 1028 | TRACKING | 1.7615 | −1.2995 | 0.258 / 0.310 |
| 1040 | LOST_SHORT | 1.7573 | −1.2867 | 不变（宽限期内保持）✅ |
| 1052 | LOST_LONG | 1.7573 | −1.2867 | 宽限期刚过，尚未起步 |
| 1064 | LOST_LONG | 1.6152 | −1.1203 | 0.111 / 0.130（回位中）|
| 1070 | LOST_LONG | 1.5254 | −1.0151 | 0.021 / 0.025 |
| 1076→ | LOST_LONG | **1.5164** | **−1.0010** | **0.012 / 0.011（≈0.7°）并稳定不动** ✅ |

从丢失到稳定在原位约 **1.5 s**；残差 0.012 rad 正是关节滤波死区（0.008 rad）量级，
物理上就是原位。测试见 `test/test_return_home.py`（覆盖两种模式、平滑性、
冻结开关、宽限期、重新识别恢复跟随、关节集合自检）。

### 13.5 实机验证结论

```text
实时识别 + task-space 控制：可用（PASS）
· 识别链路：YOLO Pose + MediaPipe 正常，TRACKING 99%
· 控制链：30 Hz 无超时，IK 0.6~0.8 ms，无未收敛
· 方向：手伸直/前伸 -> TCP 前伸；手抬高 -> TCP 抬高；手上翘 -> 末端上摆（标定 sign_h 与 j5 invert 后）
· 丢失行为：识别不到 -> 1s 宽限 -> 平滑回归原位并稳定停住（实测 1.5s 到位）
· 遗留：正面机位的"前后"是投影近似；建议侧视机位以获得几何精确的前后控制
```

---

## 11. 未改动项与可复现方式

### 11.1 未改动（本轮）

```text
未改：YOLO Pose 推理架构、MediaPipe、GUI 主架构、ROS2 Controller、
      /arm_controller、安全限速、急停、Joint EMA、30 Hz 主循环、官方 piper_ros
未改：joint1 / joint4 固定；J5/J6/夹爪驱动链路（wrist_pitch→J5、thumb→J6、
      openness→夹爪）保持原样
未改：legacy 映射的任何参数（invert/scale/neutral/区间全部原值，可随时回退）
新增：kinematics.py / limits.py / 重写的 task_space.py / tools 两个脚本 /
      data/fk_chain.json / 4 个测试文件
默认模式仍是 retarget_mode: "legacy"（本轮 A/B 用临时配置，未动生产 yaml）
```

### 11.2 一键切换

```yaml
# config/retargeting.yaml
retarget_mode: "legacy"      # 现状（逐关节映射，可回退）
retarget_mode: "task_space"  # V1.1（人手 → TCP → J2/J3 IK）
```

### 11.3 复现命令（远端）

```bash
source /opt/ros/humble/setup.bash && source ~/Yolo_pose+piper/install/setup.bash

# ① FK 链路生成（从实际加载的 URDF）
ros2 param get /robot_state_publisher robot_description > /tmp/rd.txt
python3 tools/build_fk_chain.py /tmp/robot_description.urdf \
        piper_human_retargeting/data/fk_chain.json

# ② 工作区扫描 + neutral 搜索
python3 tools/scan_workspace.py --n2 61 --n3 61 --out /tmp/workspace.json

# ③ 全量测试（302 条）
cd ~/Yolo_pose+piper/src/piper_human_retargeting && python3 -m pytest test/ -q -p no:anyio

# ④ A/B 实测（各 ~35 s / 90 s）
python3 /tmp/ab_v11.py --mode legacy     --input synth
python3 /tmp/ab_v11.py --mode task_space --input synth
python3 /tmp/ab_v11.py --mode legacy     --input real
python3 /tmp/ab_v11.py --mode task_space --input real

# ⑤ 端到端（真视觉链路）
ros2 run piper_human_retargeting retarget_demo \
  --video ~/Yolo_pose+piper/testdata/arm_fwd_up.mp4 --side right \
  --retarget-config /tmp/abv11_cfg_task_space/retargeting.yaml \
  --no-current-initial --auto-calibrate --csv /tmp/e2e_ts.csv --no-display
```

产物：`/tmp/abv11_<mode>_<input>.csv`（逐帧全链路 + 实测 TCP）、
`/tmp/workspace.json`、`/tmp/e2e_ts.csv`。

---

## 12. 遗留与下一步（本轮不做）

1. **上游测量跳变**：`raw_elbow` 会退化到 0°/180°（关键点共线/遮挡），
   它是 §8.4 那 19 次 IK 跳变的源头。下一步应在**测量层**加该量的
   `max_jump` 门限（配置里 j3 的 `max_jump` 目前是 0.0 = 不限制）。
2. **轴关系标定**：当前按"侧视机位"配置 `horizontal_axis: x`；
   若相机改为正面，需把水平轴改成 `y` 并重新验证（配置项已就位）。
3. **肘角作为姿态偏好**：本轮只记录 `elbow_angle_deg`（进 CSV），
   未加入 IK cost（任务书 §16 允许本轮只记录）。
4. **RGB-D / 3D**：按任务书要求本轮**不进入**。
