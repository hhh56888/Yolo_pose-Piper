# 项目结构与管理约定

> 本文件说明本工作区**哪些目录是我们自己的、哪些不要动**，以及新增文件应该放哪里。
> 最近整理：2026-09-29（清垃圾、收工具、归置证据、补索引）。

---

## 1. 顶层结构（本地 `~/Desktop/catkin_ws` 与远端 `~/Yolo_pose+piper` 一致）

```text
catkin_ws/                          ← 工作区根（远端同名工程为 ~/Yolo_pose+piper）
├── PROJECT_STRUCTURE.md            ← 本文件（远端在工程根，本地在工作区根）
├── fix_upstream_bugs.sh            ← 官方 piper_ros 的补丁脚本（两处布局一致）
├── .gitignore                      ← 忽略构建产物/缓存/临时文件
├── docs/                           ← 【全部文档与证据】
│   ├── README.md                   ← 文档索引（先看这个）
│   ├── evidence/                   ← 截图 / 实测 CSV / 日志（31 个文件）
│   ├── 阶段一~四_*.md              ← 分阶段测试报告
│   ├── 2D人体到Piper遥操作V1_优化与验证报告.md
│   ├── 2D人体到Piper重映射V1.1_优化报告.md
│   ├── J3_Elbow_Mapping_Audit.md
│   └── 2D_TaskSpace_Retargeting_V1_1_Report.md   ← 最新（V1.1 主报告）
├── models/                         ← YOLO 权重（yolo11n-pose.pt）
├── testdata/                       ← 测试视频（arm_fwd_up.mp4 / arm_test.mp4）
└── src/
    ├── piper_human_control/        ← 【我们的】关节控制 / 安全限速 / 流式下发
    ├── piper_human_perception/     ← 【我们的】YOLO Pose + MediaPipe 手部
    ├── piper_human_retargeting/    ← 【我们的】几何 → 重映射（legacy / task_space）
    ├── piper_ros/                  ← ⛔ 官方仓库，**不要改**
    ├── piper_sdk/                  ← ⛔ 官方 SDK
    └── project1|2|4, project4.zip  ← ⛔ 其它项目，与本项目无关
```

**判断标准**：只有 `piper_human_*` 三个包 + `docs/` + `models/` + `testdata/` 属于本项目。
`piper_ros`、`piper_sdk`、`project*`、`project4.zip` 一律不动（其中 `piper_ros` 仅允许通过
`fix_upstream_bugs.sh` 打补丁，且需在报告里记录）。

---

## 2. 单个 ROS 2 包的目录约定

以 `src/piper_human_retargeting` 为例（三个包结构相同）：

```text
piper_human_retargeting/
├── config/
│   └── retargeting.yaml            ← ★ 唯一配置源（映射/滤波/限位/task_space）
├── piper_human_retargeting/        ← Python 包本体
│   ├── data/                       ← ★ 运行时数据（见 §3）
│   │   ├── fk_chain.json           ← URDF 生成的 FK 链路（kinematics.py 用）
│   │   └── fk_table.json           ← 早期方向表（arm_direction，已停用但保留）
│   ├── *_*.py                      ← 模块：arm_geometry / mapping / retargeter /
│   │                                 task_space / kinematics / limits / tracker_state …
│   └── retarget_demo.py            ← ★ 主程序入口（--camera / --video / --image）
├── tools/                          ← ★ 可执行工具（不在 import 路径上）
│   ├── build_fk_chain.py           ← 从实际 URDF 生成 data/fk_chain.json
│   ├── scan_workspace.py           ← J2×J3 workspace 扫描 + neutral 评分搜索
│   ├── audit/                      ← 专项审计脚本（J3 审计、TF 探针…）
│   ├── ab/                         ← A/B 实测脚本（legacy vs task_space、真人在环）
│   └── legacy/                     ← 早期一次性排查脚本（保留备查）
├── scripts/
│   └── retarget_demo               ← ROS 2 可执行入口（ros2 run 用）
├── launch/ , rviz/ , resource/     ← ROS 2 包元数据 / rviz 配置
├── test/                           ← pytest 用例（315 条）
│   └── verify_joints.py            ← ⚠️ 手动逐关节验证脚本（非 pytest，见 §4）
├── package.xml , setup.py
```

**新增文件放哪**：

| 你要加的东西 | 放这里 |
|---|---|
| 新的映射/几何模块 | `piper_human_retargeting/*.py` |
| 新的配置项 | `config/retargeting.yaml` + `config.py` 的 dataclass 默认值（**两处都要**）|
| 新的测试 | `test/test_<主题>.py`（pytest 自动收集）|
| 一次性/离线分析脚本 | `tools/`（分析类）或 `tools/audit/`（审计类）|
| 实测证据（CSV/截图/日志） | `docs/evidence/` |
| 报告 | `docs/*.md`，并在 `docs/README.md` 补一行索引 |

---

## 3. 数据文件（`data/`）的来源与再生成

| 文件 | 用途 | 生成方式 | 溯源 |
|---|---|---|---|
| `fk_chain.json` | FK / Jacobian / 数值 IK 的链路 | `python3 tools/build_fk_chain.py <urdf> <out>` | 文件内记录 URDF md5（当前 `79c1b6e3…`）|
| `fk_table.json` | 早期"方向/伸展"表（`arm_direction`，已停用） | 仿真扫描（历史产物） | 保留以支持回退 |

**重新生成 `fk_chain.json`（必须来自实际加载的模型）**：

```bash
ros2 param get /robot_state_publisher robot_description > /tmp/rd.txt
python3 -c "t=open('/tmp/rd.txt').read(); open('/tmp/robot_description.urdf','w').write(t[t.find('<?xml'):])"
python3 tools/build_fk_chain.py /tmp/robot_description.urdf \
        piper_human_retargeting/data/fk_chain.json
```

⚠️ `data/` 必须留在 Python 包内：`setup.py` 的 `data_files` 与运行时的路径解析
（`Retargeter._resolve_table_path` 依次尝试 配置目录/data → 上级目录/data → 包目录/data）
都依赖它。移动它会同时破坏安装与运行。

---

## 4. 已知的"位置不理想但有意保留"的项

| 项 | 为什么保留 |
|---|---|
| `test/verify_joints.py` | 是**手动**验证脚本（`python3 test/verify_joints.py`），三份报告按此路径引用；pytest 不会收集它（不以 `test_` 开头）|
| `piper_human_retargeting/data/fk_table.json` | `arm_direction` 回退路径需要；删了会破坏"可回退"承诺 |
| `fix_upstream_bugs.sh` 放在工程根 | `docs/阶段一_*.md` 里按 `~/Yolo_pose+piper/fix_upstream_bugs.sh` 引用 |
| `tools/legacy/` 里的一次性脚本 | 排查过程留痕；不影响运行与测试 |

---

## 5. 日常维护命令

```bash
# 跑全部测试（本地 / 远端）
python3 -m pytest test/ -q -p no:anyio                 # 在 src/piper_human_retargeting 下

# 清缓存（构建产物、__pycache__、macOS 的 ._* 旁车文件）
find src/piper_human_retargeting -name __pycache__ -type d -prune -exec rm -rf {} +
find src -name "._*" -delete
rm -rf install/*/lib/python3*/site-packages/*/__pycache__   # 远端

# 远端重建（改完 Python 必须做，install 里是拷贝不是符号链接）
cd ~/Yolo_pose+piper && source /opt/ros/humble/setup.bash && \
  colcon build --packages-select piper_human_retargeting
```

**macOS 打包注意**：本机 `tar`/`scp` 会带上 `._*` 旁车文件（AppleDouble），
在 Linux 远端是纯垃圾，已清过两次。打包时加环境变量可避免：

```bash
COPYFILE_DISABLE=1 tar czf /tmp/x.tgz <目录>
```

**备份约定**：任何批量移动/删除前先

```bash
tar czf /tmp/backup_$(date +%Y%m%d_%H%M).tgz --exclude=__pycache__ \
    src/piper_human_retargeting src/piper_human_control src/piper_human_perception \
    docs models testdata
```

---

## 6. 远端工程额外说明

远端 `~/Yolo_pose+piper` 比本地多三个 **colcon 产物目录**，它们可随时重建、**不进备份**：

```text
build/    编译中间件（9 MB）
install/  安装空间（4 MB）—— ⚠️ Python 文件是**拷贝**，改完源码必须 colcon build
log/      colcon 日志（已清空；每次 build 会重新生成）
```

远端 `/tmp` 只用于**当次运行**的临时产物（CSV、日志、探针输出、临时配置）。
需要长期保留的脚本一律进 `tools/`，需要长期保留的实测数据一律进 `docs/evidence/`
—— 本机 `/tmp` 会被系统清理（本次整理时本地 `/tmp` 的审计脚本就已被清掉，
最后是从远端 `/tmp` 回收进仓库的）。
