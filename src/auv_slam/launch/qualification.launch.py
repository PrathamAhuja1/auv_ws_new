#!/usr/bin/env python3
"""
Qualification Mission Launch File
Launches complete qualification system with gate detection and navigation
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, TimerAction, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    auv_slam_share = get_package_share_directory('auv_slam')
    
    # Config paths
    thruster_params = os.path.join(auv_slam_share, 'config', 'thruster_params.yaml')
    qual_params = os.path.join(auv_slam_share, 'config', 'qualification_params.yaml')
    
    # Launch arguments
    declare_enable_debug = DeclareLaunchArgument(
        'enable_debug_view',
        default_value='true',
        description='Launch rqt_image_view for gate debugging'
    )
    
    # 1. Simulation (Gazebo + RViz) with QUALIFICATION WORLD
    display_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(auv_slam_share, 'launch', 'display.launch.py')
        ),
        launch_arguments={
            'use_sim_time': 'True',
            'world_file': os.path.join(auv_slam_share, 'worlds', 'qualification_world.sdf')
        }.items()
    )
    
    # 2. Thruster Mapper (converts Twist → thruster commands)
    thruster_mapper = Node(
        package='auv_slam',
        executable='simple_thruster_mapper.py',
        name='thruster_mapper',
        output='screen',
        parameters=[thruster_params]
    )
    
    # 3. Qualification Gate Detector (delayed to let simulation stabilize)
    gate_detector = TimerAction(
        period=3.0,
        actions=[
            Node(
                package='auv_slam',
                executable='qualification_gate_detector.py',
                name='qualification_gate_detector',
                output='screen',
                parameters=[qual_params]
            )
        ]
    )
    
    # 4. Qualification Navigator (delayed to let detector initialize)
    navigator = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='auv_slam',
                executable='qualification_navigator.py',
                name='qualification_navigator',
                output='screen',
                parameters=[qual_params]
            )
        ]
    )
    
    # 5. Safety Monitor
    safety_monitor = Node(
        package='auv_slam',
        executable='safety_monitor_node.py',
        name='safety_monitor',
        output='screen',
        parameters=[{
            'max_depth': -1.55,  # Pool is 1.6m deep at walls
            'min_depth': 0.2,
            'max_roll': 0.785,
            'max_pitch': 0.785,
            'watchdog_timeout': 5.0,
            'max_mission_time': 900.0,
            'pool_bounds_x': [-12.5, 12.5],  # 25m pool
            'pool_bounds_y': [-8.0, 8.0],    # 16m pool
        }]
    )
    
    # 6. Debug visualization (delayed start)
    rqt_debug_view = TimerAction(
        period=7.0,
        actions=[
            ExecuteProcess(
                cmd=['rqt_image_view', '/qualification/debug_image'],
                output='screen',
                shell=False
            )
        ]
    )
    
    # 7. Optional: Raw camera view for comparison
    rqt_raw_camera = TimerAction(
        period=8.0,
        actions=[
            ExecuteProcess(
                cmd=['rqt_image_view', '/camera_forward/image_raw'],
                output='screen',
                shell=False
            )
        ]
    )
    
    return LaunchDescription([
        declare_enable_debug,
        display_launch,
        thruster_mapper,
        gate_detector,
        navigator,
        safety_monitor,
        rqt_debug_view
    ])


if __name__ == '__main__':
    generate_launch_description()