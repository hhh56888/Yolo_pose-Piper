#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为实时测试生成 task_space 临时配置（只改 retarget_mode 一行）"""
import io
import os
import shutil
import subprocess
import sys

SRC = os.path.expanduser("~/Yolo_pose+piper/src/piper_human_retargeting")
CFG_SRC = os.path.join(SRC, "config")
DATA_SRC = os.path.join(SRC, "piper_human_retargeting", "data")
DST = "/tmp/live_ts_cfg"

if os.path.isdir(DST):
    shutil.rmtree(DST)
shutil.copytree(CFG_SRC, DST)
shutil.copytree(DATA_SRC, os.path.join(DST, "data"))

p = os.path.join(DST, "retargeting.yaml")
s = io.open(p, encoding="utf-8").read()
assert s.count('retarget_mode: "legacy"') == 1, "生产配置里的模式行不是预期的 legacy"
s = s.replace('retarget_mode: "legacy"', 'retarget_mode: "task_space"')
io.open(p, "w", encoding="utf-8").write(s)

d = subprocess.run(["diff", "-u", os.path.join(CFG_SRC, "retargeting.yaml"), p],
                   capture_output=True, text=True).stdout
print("临时配置: %s" % DST)
print("与生产配置的 diff:")
print(d.strip() or "(无差异)")
print("data/:", os.listdir(os.path.join(DST, "data")))
sys.exit(0)
