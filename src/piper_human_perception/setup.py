# ============================================================
# piper_human_perception 构建配置 (ament_python)
# ============================================================
import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'piper_human_perception'

# ------------------------------------------------------------
# 关于 ros2 run 的可执行文件位置
# ------------------------------------------------------------
# `ros2 run <pkg> <exe>` 只在 <prefix>/lib/<pkg>/ 下查找可执行文件，
# 而 ament_python 经 easy_install 生成的 entry_points 脚本在本环境下
# 会落到 <prefix>/bin/。详见 piper_human_control/setup.py 中的完整说明
# （那里记录了 4 种无效方案）。此处沿用同一套已验证的解法：
# 用 data_files 把 scripts/<命令名> 直接投放到 lib/<pkg>/。
# ------------------------------------------------------------
_LAUNCHERS = [
    'scripts/pose_demo',
    'scripts/camera_check',
]


setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        # ament 资源索引
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # 配置文件
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
        # ros2 run 入口
        (os.path.join('lib', package_name), _LAUNCHERS),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zxmy',
    maintainer_email='zxmy@example.com',
    description='人体手臂姿态感知层（阶段三）',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            # 阶段三演示：摄像头 -> YOLO Pose -> 肩肘腕可视化
            'pose_demo = piper_human_perception.pose_demo:main',
            # 摄像头自检
            'camera_check = piper_human_perception.camera_check:main',
        ],
    },
)
