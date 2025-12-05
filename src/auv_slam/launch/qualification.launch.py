#!/usr/bin/env python3
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction, ExecuteProcess, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    pkg_path = get_package_share_directory('auv_slam')
    qual_config = os.path.join(pkg_path, 'config', 'qualification_run.yaml')
    
    # 1. Launch World (Gazebo)
    # Re-using logic from display.launch.py but overriding the world file
    display_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_path, 'launch', 'display.launch.py')
        ),
        launch_arguments={
            'use_sim_time': 'True',
            'world': os.path.join(pkg_path, 'worlds', 'qualification_world.sdf') # Pointing to new world
        }.items()
    )
    
    # If display.launch.py doesn't accept 'world' arg natively, we might need to 
    # adjust, but usually it's cleaner to just include the necessary nodes here 
    # if display.launch is rigid. However, assuming standard setup:
    
    # 2. Thruster Mapper
    thruster_mapper = Node(
        package='auv_slam',
        executable='simple_thruster_mapper.py',
        name='simple_thruster_mapper',
        output='screen',
        parameters=[qual_config]
    )
    
    # 3. Gate Detector (Redefine as qualification detector)
    gate_detector = Node(
        package='auv_slam',
        executable='gate_detector_node.py',
        name='qualification_gate_detector',
        output='screen',
        parameters=[qual_config],
        remappings=[('/gate/detected', '/gate/detected'),
                    ('/gate/frame_position', '/gate/frame_position')]
    )
    
    # 4. Qualification Navigator
    navigator = TimerAction(
        period=8.0,
        actions=[
            Node(
                package='auv_slam',
                executable='qualification_navigator.py',
                name='qualification_navigator',
                output='screen',
                parameters=[qual_config]
            )
        ]
    )

    return LaunchDescription([
        display_launch,
        thruster_mapper,
        gate_detector,
        navigator,
        # Debug view
        ExecuteProcess(cmd=['rqt_image_view', '/gate/debug_image'], output='screen')
    ])