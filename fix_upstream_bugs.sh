#!/usr/bin/env bash
# ============================================================
# Piper 上肢动作重定向项目 · 环境准备与修复脚本
# ============================================================
# 用途：
#   在每次重新拉取/解压 piper_ros 源码、或在新机器上部署后执行一次，
#   修复已知的上游缺陷并编译工作空间。
#
# 用法：
#   bash ~/Yolo_pose+piper/fix_upstream_bugs.sh
# ============================================================
set -o pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WS" || exit 1

echo "############ 上游缺陷修复 ############"

# ------------------------------------------------------------
# 缺陷 1：joint8_ctrl.py 缺少可执行位
# ------------------------------------------------------------
# AgileX 在 humble 分支把 piper_gazebo/scripts/joint8_ctrl.py
# 提交为 100644（非可执行），但 piper_gazebo.launch.py 把它当
# 可执行文件启动：
#     Node(package='piper_gazebo', executable='joint8_ctrl.py')
# 配合 colcon --symlink-install（安装目录只是软链），
# ros2 launch 会报：
#     executable 'joint8_ctrl.py' not found on the libexec directory
#
# 修复：仅补可执行位，不改文件内容。
# 注意：该修复会被重新解压源码 / git checkout 覆盖，需重复执行。
# ------------------------------------------------------------
JT="src/piper_ros/src/piper_sim/piper_gazebo/scripts/joint8_ctrl.py"
if [ -f "$JT" ]; then
    if [ -x "$JT" ]; then
        echo "[OK]   joint8_ctrl.py 已有可执行位"
    else
        chmod +x "$JT"
        echo "[FIX]  已为 joint8_ctrl.py 补上可执行位"
    fi
    echo "       内容 md5: $(md5sum "$JT" | awk '{print $1}')  (应为 9d30891a5934d17a8ff4575bb155d8ad)"
else
    echo "[SKIP] 未找到 $JT（piper_ros 尚未就位？）"
fi

# ------------------------------------------------------------
# 缺陷 2（仅提示，未修改）：ros2_control command limit 失真
# ------------------------------------------------------------
# piper_description_gazebo.xacro 中 8 个关节的
# <command_interface name="position"> 被统一写成 min=-1 / max=1，
# 与真实限位不符（例如 joint2 实际范围 [0, 3.14]）。
# 实测该参数对 position 接口不生效（joint2 可到 3.08 rad），
# 但数字具有误导性。
# => 安全限位必须以官方 piper_sdk 为准，
#    已固化在 piper_human_control/config/joint_limits.yaml。
# 此处不修改上游文件，仅提示。
# ------------------------------------------------------------
echo "[INFO] 安全限位以 piper_human_control/config/joint_limits.yaml 为准，"
echo "       不要使用 xacro 中 command_interface 的 ±1。"

echo
echo "############ colcon 编译 ############"
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
colcon build --symlink-install
BUILD_RC=$?
echo "colcon exit=$BUILD_RC"

echo
echo "############ 完成 ############"
echo "工作空间: $WS"
echo
echo "启动仿真:"
echo "  export DISPLAY=:1"
echo "  source $WS/install/setup.bash"
echo "  ros2 launch piper_gazebo piper_gazebo.launch.py"
echo
echo "阶段二控制接口测试:"
echo "  ros2 run piper_human_control joint_test"

exit $BUILD_RC
