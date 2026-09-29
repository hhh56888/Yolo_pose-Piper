# -*- coding: utf-8 -*-
"""用 Gazebo transport 把 GUI 相机放回能看到机械臂的位置"""
import sys, time
try:
    from gazebo_msgs.msg import Pose
except Exception:
    Pose = None

try:
    from gz.msgs10.pose_pb2 import Pose as GzPose
    from gz.transport13 import Node
    HAVE_GZ = True
except Exception as e:
    HAVE_GZ = False
    print("gz python 绑定不可用:", e)

if HAVE_GZ:
    node = Node()
    p = GzPose()
    p.name = "user_camera"
    p.position.x, p.position.y, p.position.z = 1.1, -1.1, 0.75
    p.orientation.x, p.orientation.y = 0.0, 0.13
    p.orientation.z, p.orientation.w = 0.30, 0.945
    ok = node.request("/gazebo/default/user_camera/pose", p, GzPose, GzPose, 2000)
    print("request 返回:", ok)
