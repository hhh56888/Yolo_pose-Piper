# 文档索引

> 目录约定见上一级 `PROJECT_STRUCTURE.md`。所有**证据文件**（截图 / 实测 CSV / 日志）
> 统一放在 `evidence/`，报告正文只引用、不内嵌。

## 按时间顺序（先读后面的）

| 报告 | 内容 | 关联证据 |
|---|---|---|
| `阶段一_Piper仿真环境测试报告.md` | Gazebo/URDF/controller 环境、关节限位与转轴实测 | — |
| `阶段二_独立Joint控制接口测试报告.md` | 单关节控制接口、限位/速度、`/arm_controller` 行为 | — |
| `阶段三_YOLO_Pose_2D手臂识别测试报告.md` | YOLO Pose 关键点、手部 21 点、腕点来源修订 | `evidence/阶段三扩展_*.jpg` |
| `阶段四_人体姿态到Piper关节映射测试报告.md` | 几何量 → 逐关节映射（legacy）的建立与实测 | `evidence/阶段四_*.jpg` |
| `2D人体到Piper遥操作V1_优化与验证报告.md` | V1 全面优化：滤波链、CSV 可信度、急停、腕部（J5/J6）| `evidence/实时_*.csv`、`evidence/实机_*.png` |
| `2D人体到Piper重映射V1.1_优化报告.md` | V1.1 目标空间重映射的第一版（reach/elevation 方案，后被取代）| — |
| `J3_Elbow_Mapping_Audit.md` | **J3/肘部方向专项审计**：同向 vs 反向、限位/neutral/几何问题分类 | `evidence/`（仿真 TF 实测在本报告内）|
| `2D_TaskSpace_Retargeting_V1_1_Report.md` | **最新主报告**：2D task-space（人的手去哪 → TCP 去哪）、FK/IK、neutral 搜索、A/B、实时识别测试 | `evidence/`；运行产物在远端 `/tmp/abv11_*.csv`、`/tmp/live_*.csv` |

## evidence/ 里有什么

| 类别 | 例子 |
|---|---|
| 实时摄像头识别与控制记录 | `实时_识别与控制记录.csv`、`实时_实测joint_states.csv` |
| 实机界面/姿态截图 | `实机_前伸姿态_机械臂向前下伸.png`、`实机_遥操作窗口与Gazebo同步_30Hz.png` |
| A/B 对比数据 | `AB_默认原始角.csv`、`AB_滤波后角度.csv` |
| V1.1 task-space 实测（专目录） | `V1.1_task_space/`：A/B 四份 CSV、实时测试抽样、workspace 扫描 JSON（含 README）|
| 异常路径留痕 | `急停_实测joint_states.csv`、`按键与急停与错误路径_日志.txt` |

新增证据时：文件名带**场景前缀**（`实时_` / `实机_` / `AB_` / `急停_`），
并在对应报告里用相对路径引用（`docs/evidence/xxx`）。
