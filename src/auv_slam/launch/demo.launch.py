#!/usr/bin/env python3
"""
Standalone Motion Demo Launch File
Launches ONLY simulation + thruster control (NO detection/navigation)

Usage:
    Terminal 1: ros2 launch auv_slam motion_demo_standalone.launch.py
    Terminal 2: ros2 run auv_slam motion_demo.py
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
from launch.substitutions import Command, FindExecutable
import launch_ros.descriptions


def generate_launch_description():
    auv_slam_share = get_package_share_directory('auv_slam')
    
    # Paths
    urdf_file = os.path.join(auv_slam_share, 'urdf', 'orca4_description.urdf')
    bridge_config = os.path.join(auv_slam_share, 'config', 'ign_bridge.yaml')
    thruster_params = os.path.join(auv_slam_share, 'config', 'thruster_params.yaml')
    world_file = os.path.join(auv_slam_share, 'worlds', 'qualification_world.sdf')
    
    # Gazebo environment
    gz_models_path = os.path.join(auv_slam_share, "models")
    gz_resource_path = os.environ.get("GZ_SIM_RESOURCE_PATH", default="")
    gz_env = {
        'GZ_SIM_SYSTEM_PLUGIN_PATH':
           ':'.join([os.environ.get('GZ_SIM_SYSTEM_PLUGIN_PATH', default=''),
                     os.environ.get('LD_LIBRARY_PATH', default='')]),
        'IGN_GAZEBO_SYSTEM_PLUGIN_PATH':  
                      ':'.join([os.environ.get('IGN_GAZEBO_SYSTEM_PLUGIN_PATH', default=''),
                                os.environ.get('LD_LIBRARY_PATH', default='')]),
        'GZ_SIM_RESOURCE_PATH':
                      ':'.join([gz_resource_path, gz_models_path])
    }

    # 1. Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': launch_ros.descriptions.ParameterValue(
                Command(['xacro ', urdf_file]), value_type=str
            ),
            'use_sim_time': True
        }]
    )

    # 2. Joint State Publisher
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[{'use_sim_time': True}]
    )

    # 3. Gazebo Simulator
    gazebo_process = ExecuteProcess(
        cmd=['ruby', FindExecutable(name="ign"), 'gazebo', '-r', '-v', '3', world_file],
        output='screen',
        additional_env=gz_env,
        shell=False
    )

    # 4. Spawn Robot
    spawn_entity = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=[
            "-name", "orca4_ign",
            "-topic", "robot_description",
            "-z", "0.2",
            "-x", "-12.0",
            "-y", "0.0",
            "-Y", "0.0",
            "--ros-args", "--log-level", "warn"
        ],
        parameters=[{"use_sim_time": True}],
    )

    # 5. Bridge (ROS <-> Gazebo)
    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            '--ros-args',
            '-p', f'config_file:={bridge_config}'
        ],
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    # 6. Thruster Mapper (REQUIRED for motion)
    thruster_mapper = Node(
        package='auv_slam',
        executable='simple_thruster_mapper.py',
        name='thruster_mapper',
        output='screen',
        parameters=[thruster_params, {'use_sim_time': True}]
    )

    return LaunchDescription([
        robot_state_publisher,
        joint_state_publisher,
        gazebo_process,
        spawn_entity,
        bridge,
        thruster_mapper,
    ])


if __name__ == '__main__':
    generate_launch_description()