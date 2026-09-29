# models/ 说明

| 文件 | 用途 |
|---|---|
| `yolo11n-pose.pt` | YOLO Pose 权重（人体 17 关键点），`retarget_demo` 默认使用 |

远端同一份位于 `~/Yolo_pose+piper/models/`。
如需换模型：放本目录，并用 `--pose-model <路径>` 指定，不要覆盖原文件（便于回退）。
