#!/usr/bin/env python3
"""
Qualification Mission Launch - Uses qualification_world.sdf with rqt_image_view
Launches Gazebo, RViz2, and all navigation nodes for qualification task
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction, ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    auv_slam_share = get_package_share_directory('auv_slam')
    
    # Config paths
    thruster_params = os.path.join(auv_slam_share, 'config', 'thruster_params.yaml')
    gate_params = os.path.join(auv_slam_share, 'config', 'qualification_gate_params.yaml')
    rviz_config = os.path.join(auv_slam_share, 'rviz', 'qualification_nav.rviz')
    
    # Launch arguments
    declare_enable_debug = DeclareLaunchArgument(
        'enable_debug_view',
        default_value='true',
        description='Launch rqt_image_view for gate debugging'
    )
    
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time'
    )
    
    # 1. Gazebo with qualification world
    gazebo_launch = ExecuteProcess(
        cmd=[
            'gz', 'sim',
            '-r',  # Run simulation immediately
            PathJoinSubstitution([
                FindPackageShare('auv_slam'),
                'worlds', 'qualification_world.sdf'
            ]),
            '--verbose'
        ],
        output='screen',
        shell=False
    )
    
    # 2. Bridge parameters from Gazebo to ROS
    bridge_params = os.path.join(auv_slam_share, 'config', 'gz_bridge.yaml')
    gazebo_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='gazebo_bridge',
        output='screen',
        parameters=[{
            'config_file': bridge_params,
            'use_sim_time': LaunchConfiguration('use_sim_time')
        }]
    )
    
    # 3. RViz2 for visualization
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config],
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time')
        }]
    )
    
    # 4. Thruster Mapper
    thruster_mapper = Node(
        package='auv_slam',
        executable='simple_thruster_mapper.py',
        name='thruster_mapper',
        output='screen',
        parameters=[thruster_params],
        remappings=[
            ('/cmd_vel', '/rp2040/cmd_vel')
        ]
    )
    
    # 5. Qualification Gate Detector
    gate_detector = Node(
        package='auv_slam',
        executable='gate_detector_node.py',
        name='gate_detector_node',
        output='screen',
        parameters=[gate_params],
        remappings=[
            ('/camera_forward/image_raw', '/front_camera/image_raw'),
            ('/camera_forward/camera_info', '/front_camera/camera_info')
        ]
    )
    
    # 6. Qualification Navigator
    gate_navigator = TimerAction(
        period=5.0,  # Delay to let simulation initialize
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
    
    # 7. Safety Monitor
    safety_monitor = Node(
        package='auv_slam',
        executable='safety_monitor_node.py',
        name='safety_monitor',
        output='screen',
        parameters=[{
            'max_depth': -0.1,      # Must stay submerged
            'min_depth': -2.5,      # Don't hit bottom
            'max_roll': 0.785,      # 45 degrees
            'max_pitch': 0.785,     # 45 degrees
            'watchdog_timeout': 5.0,
            'max_mission_time': 600.0  # 10 minutes max
        }]
    )
    
    # 8. rqt_image_view for gate detection visualization
    # Launches immediately to show detection feed
    rqt_image_view = Node(
        package='rqt_image_view',
        executable='rqt_image_view',
        name='rqt_gate_view',
        output='screen',
        arguments=['/gate/debug_image'],
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time')
        }]
    )
    
    rqt_raw_camera = Node(
        package='rqt_image_view',
        executable='rqt_image_view',
        name='rqt_raw_view',
        output='screen',
        arguments=['/front_camera/image_raw'],
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time')
        }]
    )
    
    return LaunchDescription([
        declare_enable_debug,
        declare_use_sim_time,
        gazebo_launch,
        gazebo_bridge,
        rviz_node,
        thruster_mapper,
        gate_detector,
        gate_navigator,
        safety_monitor,
        rqt_image_view,
    ])