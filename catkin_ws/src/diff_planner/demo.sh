#!/bin/bash

source /opt/ros/noetic/setup.bash
cd ~/Documents/MTP/code/diffusion/catkin_ws
source src/venv/bin/activate
source devel/setup.bash

roslaunch franka_gazebo panda.launch x:=-0.5 \
    world:=$(rospack find franka_gazebo)/world/stone.sdf \
    controller:=cartesian_impedance_example_controller \
    rviz:=true
