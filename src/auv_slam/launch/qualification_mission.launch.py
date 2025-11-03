#!/usr/bin/env python3
"""
Qualification Mission Launch File
Launches complete qualification task with all required nodes
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    # Package directory
    auv_slam_share = get_package_share_directory('auv_slam')
    
    # Paths
    thruster_params = os.path.join(auv_slam_share, 'config', 'thruster_params.yaml')
    
    # Qualification world path - use installed location
    qual_world_path = os.path.join(auv_slam_share, 'worlds', 'qualification_world.sdf')
    
    # Bridge configuration
    bridge_config_path = os.path.join(auv_slam_share, 'config', 'ign_bridge.yaml')
    
    # Gazebo environment setup
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
    
    # 1. Gazebo Simulation with Qualification World
    gazebo_launch = ExecuteProcess(
        cmd=['gz', 'sim', '-r', '-v', '3', qual_world_path],
        output='screen',
        additional_env=gz_env,
        shell=False,
    )
    
    # 2. ROS-Gazebo Bridge
    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
                '--ros-args',
                '-p', f'config_file:={bridge_config_path}'
        ],
        output="screen",
    )
    
    # 3. Robot State Publisher
    urdf_path = os.path.join(auv_slam_share, 'urdf/orca4_description.urdf')
    
    # Read URDF file
    with open(urdf_path, 'r') as urdf_file:
        robot_description = urdf_file.read()
    
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': True
        }],
        output='screen'
    )
    
    # 4. Spawn Entity (AUV)
    spawn_entity = TimerAction(
        period=2.0,
        actions=[
            Node(
                package="ros_gz_sim",
                executable="create",
                output="screen",
                arguments=[
                    "-name", "orca4_ign",
                    "-topic", "robot_description",
                    "-x", "-14.3",  # Starting position at wall
                    "-y", "0.0",
                    "-z", "-0.5",
                    "--ros-args",
                    "--log-level", "warn",
                ],
                parameters=[{"use_sim_time": True}],
            )
        ]
    )
    
    # 5. Thruster Mapper
    thruster_mapper = Node(
        package='auv_slam',
        executable='simple_thruster_mapper.py',
        name='thruster_mapper',
        output='screen',
        parameters=[thruster_params]
    )
    
    # 6. Qualification Gate Detector (delayed start)
    gate_detector = TimerAction(
        period=3.0,
        actions=[
            Node(
                package='auv_slam',
                executable='qualification_gate_detector_node.py',
                name='qualification_gate_detector',
                output='screen',
                parameters=[{
                    'use_sim_time': True
                }]
            )
        ]
    )
    
    # 7. Qualification Navigator (delayed start)
    navigator = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='auv_slam',
                executable='qualification_navigator_node.py',
                name='qualification_navigator',
                output='screen',
                parameters=[{
                    'use_sim_time': True,
                    'target_depth': -1.0,
                    'search_speed': 0.4,
                    'approach_speed': 0.6,
                    'passing_speed': 0.8,
                    'alignment_threshold': 0.1,
                    'passing_trigger_distance': 1.5,
                    'gate_clearance': 2.0,
                    'u_turn_duration': 8.0,
                    'home_position_x': -14.3,
                    'home_position_y': 0.0,
                    'home_tolerance': 0.5
                }]
            )
        ]
    )
    
    # 8. Safety Monitor
    safety_monitor = TimerAction(
        period=4.0,
        actions=[
            Node(
                package='auv_slam',
                executable='safety_monitor_node.py',
                name='safety_monitor',
                output='screen',
                parameters=[{
                    'use_sim_time': True,
                    'max_depth': -3.0,
                    'min_depth': 0.2,
                    'max_roll': 0.785,
                    'max_pitch': 0.785,
                    'watchdog_timeout': 5.0,
                    'max_mission_time': 600.0  # 10 minutes
                }]
            )
        ]
    )
    
    return LaunchDescription([
        gazebo_launch,
        bridge,
        robot_state_publisher,
        spawn_entity,
        thruster_mapper,
        gate_detector,
        navigator,
        safety_monitor,
    ])