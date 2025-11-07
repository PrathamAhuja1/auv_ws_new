#!/usr/bin/env python3
"""
Debug Mission Launch - Includes rqt_image_view for gate debugging
This launch file starts the full mission WITH visual debugging tools
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
    gate_params = os.path.join(auv_slam_share, 'config', 'gate_params.yaml')
    navig_params = os.path.join(auv_slam_share, 'config', 'navigation_params.yaml')
    
    # Launch arguments
    declare_enable_debug = DeclareLaunchArgument(
        'enable_debug_view',
        default_value='true',
        description='Launch rqt_image_view for gate debugging'
    )
    
    # 1. Simulation (Gazebo + RViz)
    display_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(auv_slam_share, 'launch', 'display.launch.py')
        ),
        launch_arguments={'use_sim_time': 'True'}.items()
    )
    
    # 2. Thruster Mapper
    thruster_mapper = Node(
        package='auv_slam',
        executable='simple_thruster_mapper.py',
        name='thruster_mapper',
        output='screen',
        parameters=[thruster_params]
    )
    
    # 3. ULTRA-FIXED Gate Detector
    gate_detector = Node(
        package='auv_slam',
        executable='gate_detector_node.py',
        name='gate_detector_node',
        output='screen',
        parameters=[gate_params]
    )
    
    # 4. Gate Navigator (delayed start)
    gate_navigator = TimerAction(
        period=3.0,
        actions=[
            Node(
                package='auv_slam',
                executable='gate_navigator_node.py',
                name='gate_navigator_node',
                output='screen',
                parameters=[gate_params]
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
            'max_depth': -3.5,
            'min_depth': 0.2,
            'max_roll': 0.785,
            'max_pitch': 0.785,
            'watchdog_timeout': 5.0,
            'max_mission_time': 900.0
        }]
    )
    
    # 6. rqt_image_view for gate debug visualization
    rqt_image_view = ExecuteProcess(
        cmd=['rqt_image_view', '/gate/debug_image'],
        output='screen',
        shell=False
    )
    
    # 7. Optional: Raw camera view
    rqt_raw_camera = ExecuteProcess(
        cmd=['rqt_image_view', '/camera_forward/image_raw'],
        output='screen',
        shell=False
    )
    diagnostic_node= Node(
        package='auv_slam',
        executable='diagnostic_node.py',
        name='diagnostic_node',
        output='screen',
        parameters=[thruster_params]
    )
             
    return LaunchDescription([
        declare_enable_debug,
        display_launch,
        thruster_mapper,
        gate_detector,
        gate_navigator,
        safety_monitor,
    #    diagnostic_node,
        # Debug visualization tools (delayed start to let everything initialize)
        TimerAction(
            period=5.0,
            actions=[rqt_image_view]
        ),
    ])