# J3 / 人体肘部 → Piper joint3 映射关系专项审计

> 审计对象：`人体肘部屈伸 -> Piper joint3` 这一段映射**应该是同向还是反向**
> 以及当前异常究竟来自 **符号 / neutral / scale / joint limit / 人体-机械臂几何结构差异** 中的哪一类。
>
> 本轮原则：**只审计 + 临时 A/B，不修改任何正式参数**
> （`invert` / `scale` / `neutral` / `mapping` / 滤波 / 安全限速 / GUI / YOLO / MediaPipe /
> ROS2 Controller / 官方 `piper_ros` 全部未改动，见 §12 未改动声明）。
>
> 所有结论均来自**当前实际运行代码**、**实际加载的 URDF**、**Gazebo 仿真实测**与**真实 live 运行数据**，
> 不包含视觉猜测。运行的机器：`zxmy@100.89.121.96`，工作区 `~/Yolo_pose+piper`。

---

## 0. 结论先行（详细证据见后文）

| 问题 | 结论 | 依据 |
|---|---|---|
| 应同向还是反向？ | **同向**（当前的 `invert=true` 已经是同向，**不应改符号**） | §5 单关节实测 + §7 A/B 仿真实测 |
| 异常来自符号错误吗？ | **不是** | §7：`invert=true` 时「人弯肘→收臂」成立；`invert=false` 才反 |
| 真正的问题在哪 | **neutral（偏向限位）+ scale（人体窗口过窄）+ joint limit（映射端点压在关节限位上）+ 几何结构（人蜷臂抬手 vs 机械臂屈肘收回）** | §9 / §10 |

一句话：**joint3 的符号是对的，错的是它周围的四个东西。**

---

## 1. Human Elbow Definition（人体肘角定义）

### 1.1 代码位置与公式

| 项 | 内容 |
|---|---|
| 文件 | `src/piper_human_retargeting/piper_human_retargeting/arm_geometry.py` |
| 函数 | `compute_arm_geometry(arm, min_confidence, require_complete, pivot, image_size, anchor)` |
| 行号 | 肘角计算在 **207–223** 行 |
| 输入关键点 | `shoulder` / `elbow` / `wrist`（腕点用 `arm.effective_wrist`，即**手部识别给出的腕**，符合阶段三的既定规则） |
| 输出单位 | 度（float） |
| 输出范围 | **[0, 180]** |

实际公式（原样摘录自代码语义）：

```text
a = S - E                      # 肩-肘 向量（像素）
b = W - E                      # 腕-肘 向量（像素）
interior = deg( acos( clip( a·b / (|a||b|), -1, +1 ) ) )   # 180° = 伸直, 0° = 折回
elbow_angle_deg = 180 - interior                           # 0 = 伸直, 180 = 完全折回
```

退化保护：`|SE| < 8px` 或 `|EW| < 8px` 判无效；缺肩点时 `elbow_bend = 0`（**不伪造 180**）。

### 1.2 实测验证（调用真实函数，合成关键点精确控制输入）

用 `fore - upper = X` 构造关键点（与 `test/verify_joints.py` 同一构造），再调真实 `compute_arm_geometry`：

| 构造的 `fore-upper` | 返回 `elbow_angle_deg` |
|---|---|
| 0° | 0.00 |
| 30° | 30.00 |
| 45° | 45.00 |
| 90° | 90.00 |
| 120° | 120.00 |
| 135° | 135.00 |
| 150° | 150.00 |
| 180° | 180.00 |

### 1.3 三个必答问题

```text
人体手臂完全伸直时：      raw_elbow_angle ≈ 0°      （interior=180°）
人体肘部弯曲约 90° 时：   raw_elbow_angle ≈ 90°     （interior=90°）
人体继续弯曲时：          raw_elbow_angle 单调【增大】（趋近 180°）
```

### 1.4 `elbow_flexion` 检查

```text
grep -rn "elbow_flexion|flexion"  →  无任何结果
```

工程中**没有** `elbow_flexion` 变量。并且必须指出一个容易搞反的点：

> 本工程的 `elbow_angle_deg` **本身就已经是屈曲量**（0=伸直、越大越弯），
> 它等价于常见写法的 `elbow_flexion`。
> 若按任务书里的公式再做一次 `flexion = 180 − elbow_angle_deg`，
> 得到的其实是「伸直度」，**方向会反过来**（0=完全折回、180=伸直）。
>
> 因此本报告的 `elbow_flexion` 一律 = `elbow_angle_deg`，且**不接入控制链**，
> 仅用于语义表述：`flexion ↑ = 更弯`。

---

## 2. Current Mapping（当前映射链与实际生效参数）

### 2.1 真实运行路径（逐级代码位置）

```text
人体关键点(YOLO) → 手部关键点(MediaPipe)
   ↓ ① compute_arm_geometry                      arm_geometry.py:207-223
raw_elbow_angle（0..180）
   ↓ ② 关键点滤波 KeypointFilter                 retargeter.py:314
   ↓ ③ 角度滤波 AngleFilter(circular=...)        retargeter.py:338-341
      ※ elbow_angle_deg 在 CIRCULAR_HUMAN_MEASURES 里（config.py:39-46）
      ※ use_filtered_angles=true → 映射吃的是【滤波后】的值（retargeter.py:937）
filtered_elbow_angle
   ↓ ④ 标定 finish_calibration（取中位数）        retargeter.py:489-545
      neutral 存原始值 + signs=cfg.measure_sign()；HumanNeutral.get() 读时乘符号
   ↓ ⑤ delta = angle_delta_deg(cur*sign, base)    retargeter.py:974-976
      angle_delta_deg = wrap180                    arm_geometry.py:238-248
delta_elbow
   ↓ ⑥ mapping.Rule.map()                        mapping.py:144-187
      ratio=(delta-h_min)/(h_max-h_min) → clamp → raw=r_min+ratio*(r_max-r_min)
      → invert → +offset → 关节限位夹紧
mapped_joint3
   ↓ ⑦ 关节滤波 JointFilter                       retargeter.py:1041
   ↓ ⑧ SafetyLimiter（速度/加速度限速）            piper_human_control/safety.py
joint3 target
   ↓ ⑨ 流式轨迹发布（header.stamp 留 0）           piper_human_control/streaming.py
/arm_controller/joint_trajectory → Gazebo / 真机
```

### 2.2 当前实际生效参数（用生产配置加载器读出，非人工抄写）

配置文件解析路径（两个路径内容 md5 完全一致 `45816d5211f857f4501336f0bdafc52e`）：

```text
运行时实际加载：~/Yolo_pose+piper/install/piper_human_retargeting/share/piper_human_retargeting/config/retargeting.yaml
源文件（审计用）：~/Yolo_pose+piper/src/piper_human_retargeting/config/retargeting.yaml
```

| 项 | 生效值 |
|---|---|
| `rule.key` | `j3` |
| `human` | `elbow_angle_deg` |
| `joint` | `joint3` |
| `human_min` / `human_max` | **−65.0 / +65.0** （度，作用在 `delta` 上） |
| `robot_min` / `robot_max` | **−1.20 / 0.00** （rad） |
| `scale`（自动推导 = (r_max−r_min)/(h_max−h_min)） | **0.0092308 rad/deg** |
| `offset` | 0.0 |
| `invert` | **true** |
| `sign`（做差前乘在人体量上） | **−1.0** |
| `clamp` | true |
| `max_jump` | 0.0（肘角未设突变门限） |
| `joint_lower` / `joint_upper`（来自 joint_limits.yaml） | **−2.967 / 0.0** |
| `neutral_joints`（全部 6 轴） | `[0.0, 0.6, -0.6, 0.0, 0.0, 0.0]` → **joint3 neutral = −0.600** |
| `fixed_joints` | `joint1=0, joint4=0` |
| `retarget_mode` | `legacy`（`task_space` 未启用） |
| `use_filtered_angles` | **true** |

`measure_sign()` 实测输出：`{upper_arm_angle_deg: +1.0, elbow_angle_deg: −1.0, wrist_pitch_deg: +1.0, thumb_offset: +1.0}`

### 2.3 链路逐级实际数值（真实代码路径，滤波收敛后读数）

标定姿势 = 肘 90°（`verify_joints.py` 的约定）：

| 人体屈曲 | raw | flt | neutral(读时带符号) | delta_elbow | mapped_joint3 | joint3 target |
|---|---|---|---|---|---|---|
| 0° | 0.00 | 0.00 | −90.00 | **+90.00** | −1.2000 | −1.1940 |
| 45° | 45.00 | 44.05 | −90.00 | +45.95 | −1.0242 | −1.0292 |
| 90° | 90.00 | 89.04 | −90.00 | +0.96 | −0.6088 | −0.6146 |
| 135° | 135.00 | 134.04 | −90.00 | −44.04 | −0.1935 | −0.1992 |
| 180° | 180.00 | 179.04 | −90.00 | −89.04 | **0.0000** | −0.0056 |

标定姿势 = 实机 live 值 126.6°：

| 人体屈曲 | raw | delta_elbow | mapped_joint3 | joint3 target |
|---|---|---|---|---|
| 20° | 20.00 | +106.60 | −1.2000（夹紧） | −1.1940 |
| 60° | 60.00 | +67.46 | −1.2000（夹紧） | −1.1940 |
| 126.6° | 126.60 | +0.84 | −0.6078 | −0.6135 |
| 160° | 160.00 | −32.29 | −0.3020 | −0.3080 |
| 180° | 180.00 | −52.36 | −0.1167 | −0.1217 |

### 2.4 必答：人体 elbow 数值增加时，joint3 target 是增加还是减小？

```text
人体 elbow 数值【增加】（更弯）
   → delta_elbow 【减小】（因为 sign = −1）
   → mapped_joint3 / joint3 target 【增大】（从 −1.2 向 0 方向走）
```

实测复核（真实 live 数据 48082 TRACKING 帧）：

```text
corr(raw_elbow, joint3)   = +0.8499     ← 弯肘 → joint3 增大
corr(delta_elbow, joint3) = −0.9803     ← 与上一行是同一件事（delta 被 sign 取反）
```

净关系式（与上表一致，等价写法）：

```text
joint3 = joint3_neutral + 0.00923 * (elbow_flexion - elbow_flexion_neutral)
        = -0.600        + 0.00923 * (elbow_angle_deg - 126.6)      [实机标定姿势]
超出 ±65° 的 delta 会被 ratio 夹紧 → joint3 停在 −1.2 或 0.0
```

---

## 3. 周期角 / 做差方式检查

代码：

```python
# retargeter.py:974-976
cur = cur * signs.get(name, 1.0)
deltas_deg[name] = angle_delta_deg(cur, base)      # base = neutral.get(name)（同样带符号）

# arm_geometry.py:238-248
def angle_delta_deg(a, b):
    d = (a - b) % 360.0
    if d > 180.0:
        d -= 360.0
    return d
```

检查结论：

1. 肘角本质**不是**周期量（它是 [0,180] 的屈曲量）。
2. 但 `sign` 在做差**两侧同时施加**（`cur*sign` 与 `neutral.get()` 都带符号），
   所以 `wrap180(±(raw−neutral))` 与普通差值**恒等**——只有恰好相隔 180° 时例外：

| raw | neutral | 普通差 | wrap180 结果 | 说明 |
|---|---|---|---|---|
| 175 | 5 | +170 | +170 | 一致 |
| 5 | 175 | −170 | −170 | 一致 |
| 180 | 0 | −180 | **+180** | 边界处符号落到 +180（等价角，但符号歧义） |

3. 真实数据里 `raw_elbow` 的范围恰好是 **[0.0, 180.0]**（两个端点都出现过），
   即退化姿态（完全伸直/完全折回，或关键点共线）正好落在边界上。
   `delta_elbow` 实测范围 **[−130.0, +136.1]**，而 human 窗口只有 ±65°：
   **约 29% 的帧处在夹紧区**（见 §9）。

> 本轮**不修改**：周期角逻辑在 (0,180) 内部是无害的空操作；
> 真正需要处理的是「0/180 端点退化」与「窗口过窄」，那是 scale/窗口问题，不是做差方式问题。

---

## 4. Piper Joint3 Definition（当前实际加载的模型）

### 4.1 确认 Gazebo / RViz 真正加载的是哪个文件

```text
启动命令：ros2 launch piper_gazebo piper_gazebo.launch.py     （PID 1088284，Sep23 起持续运行）
launch 文件（install 里是符号链接）：
  ~/Yolo_pose+piper/install/piper_gazebo/share/piper_gazebo/launch/piper_with_gripper/piper_gazebo.launch.py
    -> ~/Yolo_pose+piper/src/piper_ros/src/piper_sim/piper_gazebo/launch/piper_with_gripper/piper_gazebo.launch.py
其中：
  urdf_name = "piper_description_gazebo.xacro"
  xacro.parse(open(urdf_model_path)) → params={'robot_description': ...}
  robot_state_publisher 发布该 robot_description
  spawn_entity.py 用 '-topic robot_description' 生成实体
```

模型文件（已核对 md5）：

```text
~/Yolo_pose+piper/src/piper_ros/src/piper_description/urdf/piper_description_gazebo.xacro
md5 = 8bd795278ce46b5e0592ff8e1f9fc671
```

因为实体是**从 `robot_description` 话题**生成的，所以「运行时真正加载的模型」
= `/robot_state_publisher` 的参数值。本审计直接解析了这个在线值
（`ros2 param get /robot_state_publisher robot_description` → `/tmp/robot_description.urdf`），
而不是仓库里任意一个 urdf。

### 4.2 joint3 定义（来自在线 robot_description）

| 项 | 值 |
|---|---|
| joint name | `joint3` |
| type | `revolute` |
| parent link | `link2` |
| child link | `link3` |
| origin xyz | `0.28503 0 0` |
| origin rpy | `0 0 -1.7939` |
| axis xyz（关节坐标系） | `0 0 1` |
| limit lower / upper | **−2.967 / 0**（rad） |
| velocity limit | 5 |
| effort limit | 100 |

原始 XML：

```xml
<joint name="joint3" type="revolute">
  <origin rpy="0 0 -1.7939" xyz="0.28503 0 0" />
  <parent link="link2" /><child link="link3" />
  <axis xyz="0 0 1" />
  <limit effort="100" lower="-2.967" upper="0" velocity="5" />
</joint>
```

物理转轴：关节坐标系的 Z 轴（`rpy` 只是绕 Z 旋转，不改变 Z 方向），
在 `base_link` 下对应 **±Y**（本审计 TF 探针与阶段四报告一致：
joint2/joint3 都在基底 XZ 竖直平面内俯仰）。

配置侧一致性：`piper_human_control/config/joint_limits.yaml` 中
`joint3: {lower: -2.967, upper: 0.0}` 与 URDF **逐位一致**（该文件的限位就来自官方 `piper_sdk` 参数）。

---

## 5. Joint3 Physical Direction Test（单关节仿真实测）

方法：joint1/2/4/5/6 固定在 neutral `[0, 0.6, -0.6, 0, 0, 0]`，
只改 joint3（`neutral ± 0.15 / ± 0.30`，远离限位），
经既有 `PiperJointController` 下发，稳定后读 `/joint_states` 与 TF（`base_link → link6/gripper_base`）。
脚本：`/tmp/j3_probe.py`（只读探针 + 既有控制器，不改任何配置）。

| dq3 | q3_cmd | q3_actual | TCP x | TCP y | TCP z | \|TCP\| | reach(link2→link4) |
|---|---|---|---|---|---|---|---|
| 0.00 | −0.6000 | −0.5985 | +0.1272 | 0.0000 | 0.3655 | 0.3871 | 0.2375 |
| +0.15 | −0.4500 | −0.4485 | +0.1311 | 0.0000 | 0.3144 | 0.3406 | 0.2013 |
| +0.30 | −0.3000 | −0.2986 | +0.1273 | 0.0000 | 0.2632 | 0.2924 | 0.1641 |
| −0.15 | −0.7500 | −0.7485 | +0.1157 | 0.0000 | 0.4156 | 0.4314 | 0.2727 |
| −0.30 | −0.9000 | −0.8984 | +0.0969 | 0.0000 | 0.4633 | 0.4733 | 0.3063 |

指令跟踪误差 ≤ 0.0016 rad（实测 q3 与指令一致 ✓）。

扩展曲线（同一方法，扫到接近限位，脚本 `/tmp/j3_curve.py`）：

| q3_actual | TCP x | TCP z | \|TCP\| | reach |
|---|---|---|---|---|
| 0.0000 | +0.0973 | 0.1658 | 0.1923 | 0.0893 |
| −0.2986 | +0.1273 | 0.2632 | 0.2924 | 0.1641 |
| −0.5985 | +0.1272 | 0.3656 | 0.3871 | 0.2376 |
| −0.8985 | +0.0969 | 0.4633 | 0.4733 | 0.3063 |
| −1.1984 | +0.0390 | 0.5477 | 0.5491 | 0.3684 |
| −1.4983 | −0.0412 | 0.6113 | 0.6127 | 0.4224 |
| −1.7983 | −0.1366 | 0.6483 | 0.6626 | 0.4670 |
| −2.0985 | −0.2388 | 0.6555 | 0.6976 | 0.5013 |

### 必答（基于仿真实测，非 URDF 推断）

```text
joint3 数值【增加】（→ 0）
   → |TCP| 减小、TCP z 降低、TCP x 回缩
   → Piper 【收臂】（末端向基座/身体方向收回，同时下垂）

joint3 数值【减小】（→ −2.967）
   → |TCP| 增大、TCP z 升高
   → Piper 【伸臂】（末端离开基座并向上升）

局部灵敏度（生产工作区间 q3 ∈ [−1.2, 0]）：
   d|TCP|/dq3 ≈ −0.28 ~ −0.30 m/rad      d(TCP_z)/dq3 ≈ −0.33 m/rad
```

注意：`|TCP|` 在 q3 ∈ [−2.1, 0] 全区间单调（q3↓ → 更远），
但 **q3 < −1.2 之后 TCP x 变成负值（末端转到身后）**，
所以「可用于遥操作的前方工作区」实际就是映射用的 [−1.2, 0]。

---

## 6. Human Motion Test（人体肘部动作测试）

### 6.1 数据来源说明（与任务书 §6 的差异，必须交代）

任务书 §6 要求「人做 A 伸直 / B 45° / C 90° / D 伸直，每个阶段 1–2 秒」。
本审计用的是**真实运行数据**而非新录一段，原因与做法：

1. **live 真实数据**（现场摄像头 + 生产配置）：
   `/tmp/live_run.csv`，105995 帧、48082 帧 TRACKING，已含真实肘部屈伸；
   配合同时段运行**只读 TF 记录器**（不新增日志系统，不改生产代码）取得 TCP。
2. **同一段真实人体输入的仿真回放**（§7 A/B 用）：
   从 live CSV 取 `raw_upper/raw_forearm`，按 `test/verify_joints.py` 的同一构造
   重建关键点（重建后肘夹角 = |fore−upper|，与记录同源），
   再走**生产 Retargeter + 生产 PiperJointController** 下发到 Gazebo。
   → 这样 A/B 的**人体输入逐帧完全相同**（任务书 §7 的硬要求），
     比「人做两遍」更严格。
   （`--video` 回放 + `record_clip.py` 的真人录像方案也已部署好，
     见 §12，需要时可直接升级为「真人在环」版本。）

### 6.2 live 真实数据：按人体肘屈曲分档（生产配置，invert=true）

窗口：与 TF 记录器重叠的 50 秒（1384 个 TRACKING 样本，人体屈曲 47.6°–177.1°）。

| 屈曲档 | 样本 | raw | delta_elbow | joint3 实测 | TCP x | TCP z |
|---|---|---|---|---|---|---|
| 25–65° | 69 | 58.2 | +43.7 | **−0.9842** | **+0.3219** | 0.2391 |
| 65–115° | 492 | 86.0 | +20.1 | −0.7837 | +0.2350 | 0.2799 |
| 115–180° | 823 | 142.3 | −35.5 | **−0.2750** | **+0.0696** | 0.2793 |

```text
corr(raw_elbow, joint3_实测) = +0.969
corr(raw_elbow, |TCP-base|)  = −0.816      ← 越弯 → 离基座越近
corr(raw_elbow, TCP x)       = −0.752      ← 越弯 → 越往回收
回归：每弯 10° → TCP 前伸 −2.8 mm、离基座 −1.6 mm、高度 +0.0 mm
```

**即：人弯肘 → 机器人收臂；人伸肘 → 机器人伸臂。**（TCP x 0.32 → 0.07 m）

### 6.3 同一段输入的仿真回放：分档（wide 窗口，屈曲覆盖 12°–148°）

| 屈曲档 | 样本 | joint3 目标 | joint3 实测 | TCP x | TCP z | \|TCP\| |
|---|---|---|---|---|---|---|
| 0–25°（伸直） | 263 | −1.1859 | −1.1107 | +0.3207 | 0.2211 | 0.4482 |
| 25–65° | 307 | −1.1858 | −1.1467 | +0.1024 | 0.3414 | 0.4845 |
| 65–115° | 122 | −0.9650 | −0.8730 | +0.0205 | 0.3553 | 0.4273 |
| 115–181°（深弯） | 212 | −0.3470 | −0.6037 | +0.1674 | 0.2544 | 0.3825 |

阶段式（A 伸直 12° → D 深弯 148°）：**|TCP| −0.0657 m、x −0.1533 m、z +0.0333 m**
→ 弯肘时末端**向基座收回**。

---

## 7. A/B 测试（同一段人体输入，仿真实测）

### 7.1 唯一变量与证据

只改一行（脚本自动生成临时配置，生产文件不写；附 `diff` 证据）：

```diff
--- src/piper_human_retargeting/config/retargeting.yaml
+++ /tmp/ab_cfg_false/retargeting.yaml
@@  j3: 段 @@
-    invert: true
+    invert: false
```

其余全部不动：`scale(0.00923)`、`neutral(−0.6)`、`human_min/max(±65)`、
`robot_min/max(−1.2/0)`、`offset(0)`、`sign(−1)`、`clamp(true)`、
三级滤波参数、`SafetyLimiter` 限速、人体中立值（126.6°，取自 live 日志）。
输入：同一段真实人体角度序列（同一 CSV 窗口、逐帧相同、30 Hz、同一初始位姿）。

### 7.2 Test A — `invert = false`

| 屈曲档 | 样本 | joint3 实测 | TCP x | TCP z | \|TCP\| |
|---|---|---|---|---|---|
| 0–25° | 263 | −0.3151 | +0.1216 | 0.1542 | 0.2137 |
| 25–65° | 307 | −0.2829 | +0.1010 | 0.1861 | 0.2323 |
| 65–115° | 120 | −0.3864 | +0.0263 | 0.2999 | 0.3229 |
| 115–181° | 212 | −0.7732 | +0.1381 | 0.2983 | 0.3659 |

```text
corr(屈曲, joint3)   = −0.465      ← 弯肘 → joint3 减小
corr(屈曲, |TCP|)    = +0.542      ← 弯肘 → 离基座更远
d|TCP|/d屈曲         = +1.179 mm/°
A(12°) → D(148°)：|TCP| +0.1522 m、x +0.0166 m、z +0.1442 m
→ 人弯肘时 Piper 【向外伸展】
```

### 7.3 Test B — `invert = true`（= 当前生产配置）

| 屈曲档 | 样本 | joint3 实测 | TCP x | TCP z | \|TCP\| |
|---|---|---|---|---|---|
| 0–25° | 263 | −1.1107 | +0.3207 | 0.2211 | 0.4482 |
| 25–65° | 307 | −1.1467 | +0.1024 | 0.3414 | 0.4845 |
| 65–115° | 122 | −0.8730 | +0.0205 | 0.3553 | 0.4273 |
| 115–181° | 212 | −0.6037 | +0.1674 | 0.2544 | 0.3825 |

```text
corr(屈曲, joint3)   = +0.714      ← 弯肘 → joint3 增大
corr(屈曲, |TCP|)    = −0.449      ← 弯肘 → 离基座更近
d|TCP|/d屈曲         = −0.644 mm/°
A(12°) → D(148°)：|TCP| −0.0657 m、x −0.1533 m、z +0.0333 m
→ 人弯肘时 Piper 【向基座收回】
```

### 7.4 一致性复核（窄窗口 1685 帧，屈曲 47°–177°）

| 指标 | invert=false | invert=true |
|---|---|---|
| corr(屈曲, joint3) | −0.839 | **+0.866** |
| corr(屈曲, \|TCP\|) | +0.702 | **−0.688** |
| d\|TCP\|/d屈曲 | +2.156 mm/° | **−1.378 mm/°** |

两次不同窗口给出**同号同结论**，说明结果不是窗口挑选出来的。

---

## 8. A/B Comparison（对比表）

| 指标 | invert=false | invert=true（生产） | 语义要求 |
|---|---|---|---|
| 人弯肘时机器人动作 | **向外伸展** ✗ | **向基座收回** ✓ | 收臂 |
| 人伸肘时机器人动作 | 收回 ✗ | **伸展** ✓ | 伸臂 |
| joint3 行程（wide / narrow） | 1.5491 / 1.0322 rad | 1.0863 / 1.0434 rad | — |
| joint3 上界饱和（q3≥0） | 3.99% / 0.36% | **0.00% / 0.00%** | 越低越好 |
| joint3 下界饱和（q3≤−1.2） | 5.65% / 0.00% | 0.66% / 0.00% | 越低越好 |
| TCP ΔX | 0.5108 m | 0.6585 m | — |
| TCP ΔZ | 0.4794 m | 0.4518 m | — |
| TCP Δ\|TCP\|（A→D） | +0.1522 m（外伸） | −0.0657 m（收回） | 收臂 |
| corr(屈曲, joint3) | −0.465 ~ −0.839 | **+0.714 ~ +0.866** | 应为正 |
| corr(屈曲, \|TCP\|) | +0.542 ~ +0.702 | **−0.449 ~ −0.688** | 应为负 |
| 运动语义一致性 | **不一致** | **一致** | — |

> 「控制直觉」不是靠"看起来更好"判断，而是按 §11 的语义判据：
> **人弯肘 → 机器人收臂 / 人伸肘 → 机器人伸臂**。
> `invert=true` 满足，`invert=false` 违反。

---

## 9. 当前 neutral 与限位余量

| 关节 | neutral | lower | upper | distance_to_lower | distance_to_upper | 判定 |
|---|---|---|---|---|---|---|
| joint2 | +0.600 | 0.000 | 3.140 | **0.600** | 2.540 | 明显偏向**下限** |
| **joint3** | **−0.600** | **−2.967** | **0.000** | **2.367** | **0.600** | **明显偏向 0 rad 上限** ✔（任务书怀疑成立） |
| joint5 | 0.000 | −1.220 | +1.220 | 1.220 | 1.220 | 对称，无偏 |

人体侧 neutral（live 标定日志实测）：

```text
upper_arm_angle_deg = -82.1     elbow_angle_deg = 126.6
wrist_pitch_deg     = -6.1      thumb_offset    = 0.0
（标定样本：4 个量；受控关节最终停在 [0, 0.005, -0.493, 0, -0.03, -1.495]）
```

**joint3 确实靠近 0 rad 上限（0.6 rad vs 2.367 rad，约 4:1）**，而且更要紧的是：

```text
j3 映射的 robot_max = 0.00  ==  关节自身上限 0.00
→ 映射区间的一端【压在】关节限位上，必然经常骑在限位上
实测（live 全量 48082 TRACKING 帧）：
   q3 贴上界(≥ −0.01)  15.25%
   q3 贴下界(≤ −1.19)  13.68%
   q3 范围 [−1.195, 0.000]  ← 两端都到（上端就是关节上限）
```

对照 joint2（同类问题，方向相反）：

```text
joint2 映射区间 robot_min = −1.00 < 关节下限 0.00
→ 该方向 1.0 rad 行程不可达，被限位削成 0（现象："抬到头就到顶"）
实测（live 全量 48082 TRACKING 帧）：
   joint2 贴下限(≤0.02) 29.67%，贴上界(≥2.18) 10.75%，范围 [0.000, 2.195]
   （29.67% 贴下限 = 抬臂方向被关节下限削平；某段运行里 joint2 长时间钉在 0.01 不动）
```

---

## 10. 「反向」到底应该加在哪一层

### 10.1 三种方案分析

**方案 A：`joint3 = -joint3`（在关节值上直接取负）**

数学上**不等价**于翻转映射方向。它把关节角围绕 **0** 镜像，而不是围绕
**neutral（−0.6）** 镜像：标定姿势会从 −0.6 跳到 +0.6，
整体偏移 `2 × neutral = 1.2 rad`，且会撞上关节上限 0。
→ **不可用**。

**方案 B：`invert = true/false`（映射区间翻转，当前旋钮）**

`raw' = robot_min + robot_max − raw`。它保持
「Δ=0 ↔ neutral」的锚点不变（`test_invert_keeps_center` 已固化），
且与「交换 robot_min/robot_max」严格等价（`test_invert_equals_swapped_range`）。
→ 语义清晰、可测试：**这是当前唯一正确的方向旋钮**。
本轮实测证明**它现在的取值（true）就是同向**，不需要改。

**方案 C：`q3 = q3_neutral + sign * scale * (elbow_flexion − elbow_flexion_neutral)`**

这是任务书推荐的表达。要注意两点：

1. 本工程的 `elbow_angle_deg` **已经是屈曲量**，所以
   `elbow_flexion = 180 − elbow_angle_deg` 会把它**变成伸直度**，
   等于引入一次多余取反（还得再用 `sign=−1` 补回来）——语义更绕，不是更清晰。
2. 但**表达形式**本身值得采用：把当前「`sign=−1` ∧ `invert=true`」的一次双重取反
   写成显式的净式子，可读性最好。两者数学等价：

```text
当前实现（双重取反）：  delta = −(elbow − elbow0)   →   invert 再取反
                        joint3 = −0.6 + 0.00923 * (elbow − elbow0)
方案 C（显式净式）：     flexion = elbow_angle_deg（已定义）
                        joint3 = q3_neutral + (+1) * 0.00923 * (flexion − flexion_neutral)
```

→ **推荐**：方向语义采用 **C 的写法 + sign=+1**（下一轮重构时），
   **本轮不改**；并且**不要**给它套 `180 − x`。

### 10.2 顺带发现：现有「方向守卫测试」其实守不住真实语义（重要）

`test/test_mapping.py` 里有两条方向守卫测试，其判据写的是
**「人弯肘 → 机器人腕下降」**（与本审计 §8 的语义判据一致，方向是对的），但实现有问题：

```python
# test/test_mapping.py:305-315
#   q3 = -1.10 -> z = 0.5215
#   q3 = -0.85 -> z = 0.4478
#   q3 = -0.35 -> z = 0.2802
#   q3 = -0.10 -> z = 0.1970
#   z 随 q3 单调【上升】                                  ← 与上面 4 行数据相反
#   dz/dq3 ≈ (0.1970 - 0.5215) / (-0.10 - (-1.10)) = +0.3245 m/rad   ← 算出来是 −0.3245
DZ_DQ = {"joint3": +0.3245}
```

两处问题：

1. **常数符号与它自己的实测数据相反**：按其列出的 4 个点，
   `dz/dq3 = (0.1970−0.5215)/1.0 = −0.3245`，而文件里写 `+0.3245`，
   注释也写成「z 随 q3 单调上升」（应为下降）。
   本审计 §5 独立实测：`d(TCP_z)/dq3 = −0.333 m/rad`，**与 −0.3245 一致**。
2. **守卫测试绕过了 `sign` 层**：测试直接调 `r.map(r.human_max)`（delta=+65），
   而运行时「人弯肘」对应的是 `delta = −53`（因为 `sign=−1`）。
   两个取反叠加，恰好让断言以错误常数通过：

```text
测试：dz = DZ_DQ(+0.3245) × (q3(+65) − q3(0)) = 0.3245 × (−0.6) = −0.195 < 0  → PASS
若把常数改成正确的 −0.3245：dz = +0.195 > 0 → FAIL（但运行时语义其实是对的）
```

`pytest test/test_mapping.py -q` 现在 **38 passed** —— 也就是说：
**这套守卫在「常数符号错 + 运行时方向正确」的组合下通过，
但换一种错法（常数正确 + 运行时反向）同样会通过。它无法发现真实的方向错误。**

（本轮只报告，不修改测试与配置。）

---

## 11. 关于相关系数：不要用它的正负判断对错

同一段行为，用不同的中间变量算相关，符号可以完全相反：

| 相关 | 值 | 该不该为负 |
|---|---|---|
| `corr(raw_elbow, joint3)` | **+0.850**（live）/ +0.714~+0.866（仿真） | 应为正（屈曲量 vs 关节数值） |
| `corr(delta_elbow, joint3)` | **−0.980**（live） | **正常**：`delta` 被 `sign=−1` 取反过 |
| `corr(屈曲, \|TCP\|)` | −0.688（invert=true） | 应为负（越弯越收回） |
| `corr(delta_elbow, joint3)` 在 invert=false 时 | +0.839（对 raw 是 −0.465） | 反，说明语义不一致 |

历史报告里出现过的 `corr ≈ −0.69 ~ −0.94` 属于**第二行**那一类
（用的是 `delta`，已经带 `sign`），**不是错误证据**。
本审计的判据始终是**运动语义**：`人弯肘 → 机器人收臂；人伸肘 → 机器人伸臂`。

---

## 12. 未改动声明与复现方式

### 12.1 未改动声明（本轮）

```text
未修改：retargeting.yaml（invert/scale/neutral/human_min/max/robot_min/max 全部原值）
未修改：joint_limits.yaml、任何 Python 生产代码、测试代码
未修改：滤波、死区、安全限速、GUI、YOLO、MediaPipe、ROS2 Controller
未修改：官方 piper_ros（仅只读解析其 xacro 与在线 robot_description）
未新增生产日志系统：TCP 由【并行运行的只读 TF 记录器】采集，
                     事后按 joint3 序列互相关对齐到既有 CSV（corr≈0.93~0.97）
A/B 的临时配置写在 /tmp/ab_cfg_<mode>/，生产文件未写
```

（唯一的持续性副作用：仿真里机械臂被单关节测试移动过，现已回到 neutral。）

### 12.2 复现命令（远端 `zxmy@100.89.121.96`）

```bash
source /opt/ros/humble/setup.bash && source ~/Yolo_pose+piper/install/setup.bash

python3 /tmp/j3_chain_audit.py     # §1/§2/§3：链路逐级数值 + 生效配置 + live 复核
python3 /tmp/j3_probe.py           # §5：joint3 ±0.15/0.30 单关节实测（TCP）
python3 /tmp/j3_curve.py           # §5：joint3 全区间 TCP 曲线
AUDIT_WIN=831794.3,831884.3 AUDIT_OUT=/tmp/ab2_%s.csv python3 /tmp/ab_replay.py  # §7 A/B
python3 /tmp/record_clip.py        # （可选）真人按 A/B/C/D 阶段录像 -> /tmp/elbow_ab.mp4
python3 /tmp/run_ab.py             # （可选）用真人录像做在环 A/B 回放
```

产出数据：`/tmp/j3_probe.json`、`/tmp/j3_curve.json`、`/tmp/ab2_false.csv`、
`/tmp/ab2_true.csv`、`/tmp/ab_replay_*.csv`、`/tmp/live_run.csv`（105995 帧）。

---

## 13. Remaining Problem（问题分类，不混为一谈）

| 类别 | 是否存在 | 证据 | 影响 |
|---|---|---|---|
| **符号问题** | **否** | §5 单关节实测（q3↑=收臂）+ §7 A/B（invert=true 才满足「弯肘→收臂」） | 不需要改 `invert`/`sign` |
| **neutral 问题** | **是** | joint3 neutral −0.6 距上界 0.600 / 距下界 2.367（4:1 偏置）；joint2 neutral 0.6 距下界 0.600 / 距上界 2.540 | 一侧行程先天不足；「抬到一定程度就到顶」 |
| **scale / 窗口问题** | **是** | human 窗口 ±65°，而 live 实测 raw_elbow 走 0–180°、delta_elbow 走 [−130, +136] → **约 29% 帧处于夹紧**；A=0.6 rad 只对应 65° 肘部动作 | 关节行程只有 1.2 rad，动作被削平、平台化 |
| **joint limit 问题** | **是** | j3 的 `robot_max=0.00` 恰等于关节上限 0 → live 15.25% 骑在上界；j2 的 `robot_min=−1.00` < 关节下限 0 → 29.67% 被削到 ≤0.02（且会长时间钉死） | 饱和、到达极限后不再响应 |
| **人体/机械臂几何结构差异** | **是** | §5：q3↑（收臂）时 TCP **下降**（d z/dq3=−0.33）；而人「蜷臂」时手是**向上**的。实测 `corr(手高, 机械臂末端高) = −0.353`，每抬高 100 px 末端反而 −22 mm | 这是操作者**"感觉方向反了"的真正来源**；joint3 符号正确也消除不了它 |
| **测试守卫缺陷** | **是** | §10.2：`test_mapping.py` 的 `DZ_DQ=+0.3245` 与其自身实测数据（−0.3245）符号相反，且守卫绕过 `sign` 层 → 38 passed 但守不住真实方向 | 未来改配置若真把方向弄反，现有测试可能照过 |

> 结论：**符号没问题；"反直觉"来自几何结构（人蜷臂抬手 vs 机械臂屈肘收回/下降）**，
> 被 neutral 偏置、窗口过窄与限位饱和进一步放大。
> 这正好对应 V1.1 任务书里尚未完成的两件事：
> **§3 用 FK 扫描重设 J2/J3 neutral**、**§4 用 reach/elevation → X/Z → IK 的 2D 重映射**
> （后者从根上解耦「手的高度」与「肘的角度」，才是「手抬高 = 末端抬高」的正解）。

---

```text
J3 MAPPING AUDIT

RESULT:
    同向（不需要反向）。
    当前生产配置 sign = −1 ∧ invert = true 的净效果已经是
        joint3 = −0.6 + 0.00923 × (elbow_angle_deg − elbow_angle_neutral)
    即「人肘屈曲增大 → joint3 增大（收臂）」，与要求的
    「人弯肘→机器人收臂 / 人伸肘→机器人伸臂」一致。

EVIDENCE:
    1. 肘角定义（arm_geometry.py:207-223）：elbow_angle_deg = 180 − interior，
       0 = 伸直、180 = 完全折回；实调验证 8 个点精确复现。
    2. 单关节仿真实测（/tmp/j3_probe.py，TF 真值）：
       q3 −0.30 → |TCP| 0.2924；q3 −0.90 → 0.4733；q3 −2.10 → 0.6976
       => q3 增大 = 收臂 + 下降（d|TCP|/dq3 ≈ −0.28，d(TCP_z)/dq3 ≈ −0.33）。
    3. 真实 live 数据（48082 TRACKING 帧，生产配置）：
       corr(raw_elbow, joint3) = +0.850；按屈曲分档
       joint3 −0.984 → −0.275，TCP x 0.322 → 0.070（人弯肘 = 机器人收臂）。
    4. 同输入 A/B 仿真实测（907/1685 帧、两个窗口、仅改 j3.invert 一行）：
       invert=true  : corr(屈曲, joint3)=+0.714~+0.866，corr(屈曲,|TCP|)=−0.449~−0.688
       invert=false : corr(屈曲, joint3)=−0.465~−0.839，corr(屈曲,|TCP|)=+0.542~+0.702
       两个窗口同结论。
    5. 当前 joint3 数值增大的物理含义由在线 robot_description 的
       limit=[−2.967, 0] + TF 实测共同确定，未使用任何"按 axis 推断"的结论。

RECOMMENDED NEXT ACTION:
    1. 【不要】改 joint3 的 invert/sign —— 方向已正确；
       若要提高可读性，下一轮把「sign=−1 ∧ invert=true」重写为
       显式净式 q3 = q3_neutral + sign(+1) × scale × (flexion − flexion_neutral)
       （flexion 就用 elbow_angle_deg，切勿再取 180−x）。
    2. 【本轮之后】按 V1.1 §3 重设 J2/J3 neutral：joint3 neutral 目前距上界仅 0.600 rad
       而距下界 2.367 rad；joint2 亦然（0.600 / 2.540）。改 neutral 时同步让
       映射区间端点离开关节限位（现在 j3 的 robot_max=0.00 正好压在关节上限上）。
    3. 【本轮之后】处理 scale/窗口：human ±65° 对上实测 0–180° 行程导致约 29% 帧夹紧，
       应按实测行程重设 human_min/max 与振幅 A（并给肘角补 max_jump 门限，
       因为 raw_elbow 会退化到 0/180 端点）。
    4. 【根治项】按 V1.1 §4 启用 reach/elevation → X/Z → IK 的 2D 重映射，
       把"手的高度"从"肘的角度"里解耦出来 —— 
       实测 corr(手高, 末端高) = −0.353（每 100 px −22 mm）属于几何结构问题，
       只有该方案能消除操作者的反向手感。
    5. 【低风险清理】修正 test/test_mapping.py 的 DZ_DQ 符号（+0.3245 → −0.3245）
       并让方向守卫测试走真实人体量（含 sign 层），否则它无法守护真实方向。
```
