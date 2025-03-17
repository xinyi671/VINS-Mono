#!/bin/bash
if [ "$#" -ne 1 ]; then 
  echo "Usage: $0 LAUNCH_FILE" >&2
  exit 1
fi

# 安装依赖（使用sudo）
sudo apt-get update && sudo apt-get install -y \
    cmake \
    libatlas-base-dev \
    libeigen3-dev \
    libgoogle-glog-dev \
    libsuitesparse-dev \
    python-catkin-tools \
    ros-${ROS_DISTRO}-cv-bridge \
    ros-${ROS_DISTRO}-image-transport \
    ros-${ROS_DISTRO}-message-filters \
    ros-${ROS_DISTRO}-tf

# 安装Ceres
export CERES_VERSION="1.12.0"
git clone https://ceres-solver.googlesource.com/ceres-solver && \
cd ceres-solver && \
git checkout tags/${CERES_VERSION} && \
mkdir build && cd build && \
cmake .. && \
sudo make -j$(nproc) install && \
cd ../../ && rm -rf ceres-solver

# 设置工作空间
export CATKIN_WS=/root/catkin_ws
export TERM=xterm
export PYTHONIOENCODING=UTF-8

# # 创建工作空间并克隆VINS-Mono
# sudo mkdir -p $CATKIN_WS/src/VINS-Mono/
# cd $CATKIN_WS/src
# git clone https://github.com/HKUST-Aerial-Robotics/VINS-Mono.git

# 编译和运行
cd $CATKIN_WS
catkin config --env-cache --extend /opt/ros/$ROS_DISTRO --cmake-args -DCMAKE_BUILD_TYPE=Release
catkin build
source devel/setup.bash
roslaunch vins_estimator ${1}