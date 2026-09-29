#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/build_fk_chain.py
=======================
从**实际加载的** URDF 生成紧凑运动学链（供 FK/IK 使用）。

为什么不用现成 URDF 库：
    - 只需要 parent/child/origin/axis/limit 五个字段；
    - 需要把「哪一份 URDF」的来源（md5）写进产物，便于审计；
    - 运行时要零依赖（ROS 环境里只有 numpy）。

用法：
    python3 tools/build_fk_chain.py <urdf> <out.json> [--root base_link] [--tip gripper_base]

输入 URDF 的正确来源（本项目）：
    由 piper_gazebo.launch.py 用 xacro 解析
    piper_description/urdf/piper_description_gazebo.xacro 后得到的 robot_description。
    在线获取：ros2 param get /robot_state_publisher robot_description
"""
import argparse
import hashlib
import json
import sys
import xml.etree.ElementTree as ET


def parse_floats(s, n=3):
    if s is None:
        return [0.0] * n
    v = [float(x) for x in s.replace(",", " ").split()]
    while len(v) < n:
        v.append(0.0)
    return v[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("urdf")
    ap.add_argument("out")
    ap.add_argument("--root", default="base_link")
    ap.add_argument("--tip", default="gripper_base")
    a = ap.parse_args()

    raw = open(a.urdf, "rb").read()
    md5 = hashlib.md5(raw).hexdigest()
    root = ET.fromstring(raw.decode("utf-8"))

    joints = {}
    for j in root.findall("joint"):
        name = j.get("name")
        p = j.find("parent")
        c = j.find("child")
        o = j.find("origin")
        ax = j.find("axis")
        lim = j.find("limit")
        joints[name] = {
            "name": name,
            "type": j.get("type"),
            "parent": p.get("link") if p is not None else None,
            "child": c.get("link") if c is not None else None,
            "origin_xyz": parse_floats(o.get("xyz") if o is not None else None),
            "origin_rpy": parse_floats(o.get("rpy") if o is not None else None),
            "axis": parse_floats(ax.get("xyz") if ax is not None else None),
            "limit": (None if lim is None else {
                "lower": (float(lim.get("lower"))
                          if lim.get("lower") is not None else None),
                "upper": (float(lim.get("upper"))
                          if lim.get("upper") is not None else None),
                "effort": (float(lim.get("effort"))
                           if lim.get("effort") is not None else None),
                "velocity": (float(lim.get("velocity"))
                             if lim.get("velocity") is not None else None),
            }),
        }

    # 从 root 走到 tip，收集链路上的关节（按父子关系排序）
    link = a.root
    chain = []
    guard = 0
    while link != a.tip:
        guard += 1
        if guard > 50:
            raise SystemExit("链路搜索超过 50 步，root/tip 是否写错？")
        nxt = [j for j in joints.values() if j["parent"] == link]
        if not nxt:
            raise SystemExit("从 %s 找不到下一个关节（目标 tip=%s）" % (link, a.tip))
        if len(nxt) > 1:
            raise SystemExit("%s 有多个子关节：%s" % (link, [x["name"] for x in nxt]))
        j = nxt[0]
        chain.append(j)
        link = j["child"]

    out = {
        "source": {
            "urdf_md5": md5,
            "urdf_bytes": len(raw),
            "root_link": a.root,
            "tip_link": a.tip,
            "generator": "tools/build_fk_chain.py",
        },
        "chain": chain,
    }
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("写出 %s：%d 个关节，root=%s tip=%s，urdf md5=%s"
          % (a.out, len(chain), a.root, a.tip, md5))
    for j in chain:
        lim = j["limit"]
        print("  %-8s %-9s %s -> %s  xyz=%s rpy=%s axis=%s limit=%s"
              % (j["name"], j["type"], j["parent"], j["child"],
                 j["origin_xyz"], j["origin_rpy"], j["axis"],
                 None if lim is None else (lim["lower"], lim["upper"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
