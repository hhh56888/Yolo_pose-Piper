# data/ 说明

本目录是**运行时数据**（不是配置），由 `setup.py` 的 `data_files` 一并安装到
`install/<pkg>/share/piper_human_retargeting/data/`。

| 文件 | 使用者 | 说明 |
|---|---|---|
| `fk_chain.json` | `kinematics.ChainModel`（FK / Jacobian / 数值 IK）| 由 `tools/build_fk_chain.py` 从**实际加载的 URDF** 生成；文件内 `source.urdf_md5` 是溯源信息（当前 `79c1b6e3cd92d4b571945c831c51c031`）|
| `fk_table.json` | `arm_direction.ArmDirectionMapper`（**已停用**，保留以支持回退）| 早期扫表：`{j2, j3, dir, reach}` |

重新生成见 `PROJECT_STRUCTURE.md` §3。**不要**把本目录移出 Python 包
（`setup.py` 与 `Retargeter._resolve_table_path` 都按包内相对路径解析）。
