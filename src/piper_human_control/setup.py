# ============================================================
# piper_human_control 构建配置 (ament_python)
# ============================================================
import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'piper_human_control'

# ------------------------------------------------------------
# 关于 ros2 run 的可执行文件位置
# ------------------------------------------------------------
# `ros2 run <pkg> <exe>` 与 `ros2 pkg executables <pkg>` 只在
#     <prefix>/lib/<pkg>/
# 下查找可执行文件。
#
# 而 ament_python 经 setuptools easy_install 生成的 entry_points 脚本，
# 在本环境（setuptools 59.6 + colcon --symlink-install）下会被装到
#     <prefix>/bin/
# 导致 `ros2 run` 报 "No executable found"。
#
# 已尝试但无效的方案（记录备查，避免重复踩坑）：
#   1. setup.py 里追加 --install-scripts 参数
#   2. 自定义 build_py.install_scripts    —— develop 模式不调用
#   3. 覆盖 develop / install 命令         —— 不改变 easy_install 行为
#   4. 覆盖 easy_install.finalize_options  —— script_dir 被再次改写
#
# 最终方案：用 data_files 把 scripts/<命令名> 直接投放到
#     lib/<pkg>/<命令名>
# 这几个文件内容相同，都按「自身文件名」分发到对应入口，
# 因此安装后文件名即命令名，`ros2 run piper_human_control joint_test` 可用。
# （data_files 不保留源文件名的别名，所以必须每个命令一个同名文件。）
# ------------------------------------------------------------
_LAUNCHERS = ['scripts/joint_test', 'scripts/joint_teleop']

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        # ament 资源索引
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # 配置文件（限位、速度、安全阈值等的唯一来源）
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
        # launch 文件
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        # ros2 run 入口（见上方说明）
        (os.path.join('lib', package_name), _LAUNCHERS),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zxmy',
    maintainer_email='zxmy@example.com',
    description='Piper 机械臂独立关节控制接口（阶段二）',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            # 阶段二验收测试：HOME -> J2 -> J3 -> J5 -> HOME
            'joint_test = piper_human_control.joint_test_node:main',
            # 交互式手动控制（键盘）
            'joint_teleop = piper_human_control.joint_teleop_node:main',
        ],
    },
)
