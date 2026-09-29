# ============================================================
# piper_human_retargeting 构建配置 (ament_python)
# ============================================================
import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'piper_human_retargeting'

# ------------------------------------------------------------
# 关于 ros2 run 的可执行文件位置
# ------------------------------------------------------------
# `ros2 run <pkg> <exe>` 只在 <prefix>/lib/<pkg>/ 下查找可执行文件，
# 而 ament_python 经 easy_install 生成的 entry_points 脚本在本环境下
# 会落到 <prefix>/bin/。详见 piper_human_control/setup.py 中的完整说明
# （那里记录了 4 种无效方案）。此处沿用同一套已验证的解法：
# 用 data_files 把 scripts/<命令名> 直接投放到 lib/<pkg>/。
# ------------------------------------------------------------
_LAUNCHERS = ['scripts/retarget_demo']

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # 映射配置文件（唯一的参数来源）
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
        # launch 文件
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        # FK 表（方向驱动映射用）。放在包内 data/ 下，
        # 同时用 data_files 显式安装一份，避免依赖 setuptools 对
        # 非 .py 包数据的默认行为。
        (os.path.join('share', package_name, 'data'),
            glob('piper_human_retargeting/data/*.json')),
        # RViz 配置（TF 排查用）
        (os.path.join('share', package_name, 'rviz'),
            glob('rviz/*.rviz')),
        # ros2 run 入口
        (os.path.join('lib', package_name), _LAUNCHERS),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zxmy',
    maintainer_email='zxmy@example.com',
    description='人体手臂姿态 -> Piper 关节映射层（阶段四）',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'retarget_demo = piper_human_retargeting.retarget_demo:main',
        ],
    },
)
