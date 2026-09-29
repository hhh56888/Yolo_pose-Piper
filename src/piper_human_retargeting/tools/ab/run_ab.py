#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
J3 审计 · A/B 回放运行器
========================
同一段人体视频，跑两遍：
    Test A: joint3 invert = false
    Test B: joint3 invert = true
**只改这一行**（脚本会 diff 出证据）。

配置来源：把生产 config 目录整份复制到 /tmp/ab_cfg_<mode>/，
只改 j3 的 invert，其余（scale/neutral/human_min/max/robot_min/max/
滤波/限速）完全不动，生产文件本身不写。

每遍同时启动只读 TF 记录器，事后可按 joint3 序列对齐。
"""
import os
import shutil
import subprocess
import sys
import time

SRC = os.path.expanduser("~/Yolo_pose+piper/src/piper_human_retargeting/config")
VIDEO = "/tmp/elbow_ab.mp4"
PY = sys.executable


def patch_config(mode):
    """复制生产 config 到独立目录，只改 j3.invert"""
    dst = "/tmp/ab_cfg_%s" % mode
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(SRC, dst)
    path = os.path.join(dst, "retargeting.yaml")
    lines = open(path, encoding="utf-8").read().split("\n")
    in_j3 = False
    changed = None
    for i, ln in enumerate(lines):
        if ln.startswith("  j3:"):
            in_j3 = True
            continue
        if in_j3 and ln.startswith("  ") and not ln.startswith("    ") \
                and ln.strip():
            in_j3 = False
        if in_j3 and ln.strip().startswith("invert:"):
            old = ln
            lines[i] = "    invert: %s" % ("true" if mode == "true" else "false")
            changed = (i + 1, old, lines[i])
            break
    if changed is None:
        raise RuntimeError("没有找到 j3 的 invert 行")
    open(path, "w", encoding="utf-8").write("\n".join(lines))
    # 证据：与生产文件的 diff
    d = subprocess.run(["diff", "-u",
                        os.path.join(SRC, "retargeting.yaml"), path],
                       capture_output=True, text=True).stdout
    print("[%s] 配置 diff（生产 vs 临时）:\n%s" % (mode, d.strip() or "(无)"))
    return path


def run_one(mode, retarget_cfg):
    tag = "true" if mode == "true" else "false"
    csv = "/tmp/ab_inv_%s.csv" % tag
    tf = "/tmp/ab_inv_%s_tf.csv" % tag
    log = "/tmp/ab_inv_%s.log" % tag
    for p in (csv, tf, log):
        if os.path.exists(p):
            os.remove(p)
    # 先起 TF 记录器（只读），覆盖整个回放时长
    tfp = subprocess.Popen([PY, "/tmp/tf_log.py", tf, "90"],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.STDOUT)
    time.sleep(2.0)
    cmd = ["ros2", "run", "piper_human_retargeting", "retarget_demo",
           "--video", VIDEO, "--side", "left",
           "--retarget-config", retarget_cfg,
           "--no-current-initial", "--auto-calibrate",
           "--csv", csv, "--no-display"]
    print("[%s] 运行: %s" % (tag, " ".join(cmd)))
    with open(log, "w", encoding="utf-8") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
    tfp.terminate()
    try:
        tfp.wait(timeout=10)
    except Exception:                                  # noqa: BLE001
        tfp.kill()
    print("[%s] demo 退出码=%s, CSV=%s, TF=%s" % (tag, rc, csv, tf))
    return rc


def main():
    if not os.path.isfile(VIDEO):
        print("[FAIL] 找不到 %s，请先运行 /tmp/record_clip.py 录制" % VIDEO)
        return 2
    cfgs = {}
    for mode in ("false", "true"):
        cfgs[mode] = patch_config(mode)
    for mode in ("false", "true"):
        run_one(mode, cfgs[mode])
    print("A/B 回放完成：/tmp/ab_inv_false.csv 与 /tmp/ab_inv_true.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
