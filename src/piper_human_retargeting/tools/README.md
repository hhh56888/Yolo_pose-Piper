# tools/ 工具索引

> 这些脚本**不在** Python import 路径上，都是离线/半离线工具。
> 路径里的 `~/Yolo_pose+piper` 表示在**远端工程目录**下运行（需要 ROS 环境与 Gazebo）。

## 顶层（建模/工作区分析，纯 Python，不需要 ROS）

| 脚本 | 作用 | 产物 |
|---|---|---|
| `build_fk_chain.py` | 从**实际加载的** URDF 生成 FK 链路 | `piper_human_retargeting/data/fk_chain.json`（内含 URDF md5 溯源）|
| `scan_workspace.py` | J2×J3 二维扫描：TCP x/z、到限位余量、σ_min；并用真实 IK 求"最大可达矩形" + neutral 评分搜索 | `--out` 指定的 JSON（含 Top5 neutral 建议）|

```bash
python3 tools/scan_workspace.py --n2 61 --n3 61 --out /tmp/workspace.json
```

## audit/ 专项审计（J3 方向审计本轮用）

| 脚本 | 作用 | 产物 |
|---|---|---|
| `j3_chain_audit.py` | 静态链路审计：肘角定义、elbow→joint3 每级数值、周期角检查、生效配置 | 终端表格；配套报告 `docs/J3_Elbow_Mapping_Audit.md` |
| `j3_probe.py` / `j3_curve.py` | **单关节仿真实测**：只改 joint3，读 `/joint_states` 与 TF 的 TCP | `/tmp/j3_probe.json`、`/tmp/j3_curve.json` |
| `tf_log.py` | 只读 TF+关节记录器（供 A/B 与实时测试并行采集实测 TCP）| CSV |
| `probe_j5.py` / `probe_j5b.py` | 只读探针：joint5 与末端指向/工具方向的偏回归（判定腕部方向）| `/tmp/probe_j5*.csv` |
| `parse_rd.py` | 解析在线 `robot_description`（确认实际加载的 joint3 定义）| `/tmp/robot_description.urdf` |

## ab/ A/B 与实机验证（需要 ROS + Gazebo）

| 脚本 | 作用 |
|---|---|
| `ab_v11.py` | **V1.1 A/B 主脚本**：同一段人体输入分别跑 legacy / task_space，逐帧记录 u/v、目标、IK 解、实测关节与 TF 的 TCP（`--input synth|real`、`--mode`、`--lambda-prev`）|
| `run_ab.py` + `record_clip.py` | 真人录像版 A/B：先录一段四阶段动作，再用同一段视频回放对比 |
| `ab_replay.py` | 用 live CSV 里的真实角度重建关键点后回放（第一版 A/B 工具）|
| `make_live_cfg.py` | 生成实时测试用的临时配置（**只改 `retarget_mode` 一行**，生产 yaml 不动）|

```bash
# 远端、已 source 环境
python3 tools/ab/ab_v11.py --mode legacy     --input synth
python3 tools/ab/ab_v11.py --mode task_space --input real
python3 tools/ab/make_live_cfg.py            # -> /tmp/live_ts_cfg/
```

## legacy/ 早期一次性排查脚本

`wrist_probe.py`、`thumb_diag.py`、`tracking_stats.py`、`verify_dir.py`、
`vid_dir.py`、`vid_probe.py`、`vid_height.py`、`setcam.py`
—— 记录当时怎么定位问题的，**不影响运行与测试**，留作备查。

## 约定

1. 新工具优先放本目录；一次性排查脚本放 `legacy/`，不要散落在 `/tmp`
   （本机 `/tmp` 会被系统清理，实测丢过一次脚本）。
2. 依赖 ROS 的脚本在文件头写清"需要 source 哪些环境 + 产物写到哪里"。
3. 只读优先：任何会**下发指令**的脚本（如 `j3_probe.py`）必须在注释里写明
   "会驱动仿真机械臂"。
