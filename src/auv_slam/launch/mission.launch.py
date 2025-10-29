#!/usr/bin/env python3
"""
Complete Autonomous Mission Launch File
Runs ALL components needed for autonomous operation
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, TimerAction

def generate_launch_description():
    auv_slam_share = get_package_share_directory('auv_slam')
    
    # Config paths
    thruster_params = os.path.join(auv_slam_share, 'config', 'thruster_params.yaml')
    gate_params = os.path.join(auv_slam_share, 'config', 'gate_params.yaml')
    flare_params = os.path.join(auv_slam_share, 'config', 'flare_params.yaml')
    navig_params= os.path.join(auv_slam_share, 'config', 'navigation_params.yaml')
    
    # 1. Simulation (Gazebo + RViz)
    display_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(auv_slam_share, 'launch', 'display.launch.py')
        ),
        launch_arguments={'use_sim_time': 'True'}.items()
    )

# Gazebo-ROS Bridge
#    bridge = Node(
#        package='ros_gz_bridge',
#        executable='parameter_bridge',
#        arguments=[
#            '--ros-args',
#            '-p', f'config_file:={os.path.join(auv_slam_share, "config", "gz_bridge.yaml")}'
#        ],
#        output='screen',
#        parameters=[{'use_sim_time': True}]
#    )

    # 2. Fixed Thruster Mapper
    thruster_mapper = Node(
        package='auv_slam',
        executable='simple_thruster_mapper.py',
        name='fixed_thruster_mapper',
        output='screen',
        parameters=[thruster_params]
    )
    
    # 3. Gate Detector
    gate_detector = Node(
        package='auv_slam',
        executable='gate_detector_node.py',
        name='gate_detector_node',
        output='screen',
        parameters=[gate_params]
    )
    gate_navigator = TimerAction(
        period=3.0,  # Wait 3 seconds for detection to initialize
        actions=[
            Node(
                package='auv_slam',
                executable='_gate_navigator_node.py',
                name='_gate_navigator_node',
                output='screen',
                parameters=[navig_params]
            )
        ]
    )
    
    # 4. Flare Detector
    flare_detector = Node(
        package='auv_slam',
        executable='flare_detection.py',
        name='flare_detector_node',
        output='screen',
        parameters=[flare_params]
    )
    
    # 5. FIXED Autonomous Mission Controller (BT.py replacement)
    gate_mission = Node(
        package='auv_slam',
        executable='BT.py',
        name='gate_detection_controller',
        output='screen',
        parameters=[navig_params]
    )
    
    # 6. Safety Monitor
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
    
    # 7. Diagnostic Node
    #diagnostic = Node(
    #   package='auv_slam',
    #    executable='diagnostic_node.py',
    #    name='diagnostic_node',
    #    output='screen'
    #)
    
    return LaunchDescription([
        display_launch,
    #    bridge,
        thruster_mapper,
        gate_detector,
        gate_mission,
#        gate_navigator,
        flare_detector,
    #    mission_controller,
        safety_monitor,
    #    diagnostic,
    ])