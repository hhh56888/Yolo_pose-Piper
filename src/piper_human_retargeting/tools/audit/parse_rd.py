#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 /robot_state_publisher 的 robot_description 解析实际加载的关节定义"""
import xml.etree.ElementTree as ET

txt = open("/tmp/rd_raw.txt", encoding="utf-8").read()
i = txt.find("<?xml")
xml = txt[i:]
open("/tmp/robot_description.urdf", "w", encoding="utf-8").write(xml)
root = ET.fromstring(xml)
print("robot name =", root.get("name"))
print("链接数 %d, 关节数 %d" % (len(root.findall("link")), len(root.findall("joint"))))
print()
print("%-10s %-9s %-12s %-12s %-22s %-22s %10s %10s %8s %8s" % (
    "joint", "type", "parent", "child", "origin_xyz", "axis_xyz",
    "lower", "upper", "effort", "vel"))
for j in root.findall("joint"):
    nm = j.get("name")
    p = j.find("parent")
    c = j.find("child")
    o = j.find("origin")
    a = j.find("axis")
    l = j.find("limit")
    print("%-10s %-9s %-12s %-12s %-22s %-22s %10s %10s %8s %8s" % (
        nm, j.get("type"),
        p.get("link") if p is not None else "-",
        c.get("link") if c is not None else "-",
        (o.get("xyz") if o is not None else "-"),
        (a.get("xyz") if a is not None else "-"),
        (l.get("lower") if l is not None else "-"),
        (l.get("upper") if l is not None else "-"),
        (l.get("effort") if l is not None else "-"),
        (l.get("velocity") if l is not None else "-")))
print()
j3 = [j for j in root.findall("joint") if j.get("name") == "joint3"][0]
print("=== joint3 原始 XML ===")
print(ET.tostring(j3, encoding="unicode"))
print("=== 上游 link 惯性/几何（用于确认是哪个模型）===")
child = j3.find("child").get("link")
for lk in root.findall("link"):
    if lk.get("name") == child:
        print(ET.tostring(lk, encoding="unicode")[:1200])
