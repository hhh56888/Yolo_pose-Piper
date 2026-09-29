# V1.1 task-space 实测数据（2026-09-28 ~ 09-29）

本目录是从远端 `/tmp` 归档过来的**可复现证据**（`/tmp` 会被系统清理，故归档）。
原始逐帧文件在远端 `/tmp/abv11_*.csv`、`/tmp/live_*.csv`；这里是完整版 + 精简版。

## A/B 对比（同一段人体输入，只改 `retarget_mode` 一行）

| 文件 | 内容 |
|---|---|
| `abv11_legacy_synth.csv` / `abv11_task_space_synth.csv` | 四个基础动作（前/后/上/下）单轴输入，960 帧，30 Hz |
| `abv11_legacy_real.csv` / `abv11_task_space_real.csv` | 真实记录的人体角度序列回放（907 帧） |

列含义：`u/v` 人体手腕归一化位置、`du/dv` 相对标定位移、`tgt_x/tgt_z` TCP 目标、
`ik_*` IK 解与耗时、`j2_act/j3_act` **实测**关节角、`tcp_*` TF 实测 TCP。
分析脚本：`src/piper_human_retargeting/tools/ab/ab_v11.py`（采集）+ 报告 §8 表格。

## 实时识别测试（真摄像头 + 人在环）

| 文件 | 内容 | 对应报告章节 |
|---|---|---|
| `live_ts5_精简.csv` | sign_h 标定后（正面机位）3443 帧抽样 | §13.2 相机轴向标定 |
| `live_home_精简.csv` | 人走出画面 → 回归原位全过程 | §13.4 回归原位 |
| `live_wrist_精简.csv` | j5 invert 翻转后（腕俯仰方向） | §13.3 人手侧符号标定 |
| `workspace.json` | J2×J3 workspace 扫描 + Top5 neutral（3721 点）| §3/§4 |

> 精简版：每 10 帧抽 1 帧（保留全部关键列），便于随仓库携带；需要的完整数据可让
> 远端重新采集（`tools/ab/ab_v11.py`）。CSV 里的 `state` 列区分
> TRACKING / LOST_SHORT / LOST_LONG，`LOST_LONG` 段可直接看到回归原位的轨迹。
