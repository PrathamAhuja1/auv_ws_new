#!/usr/bin/env python3
"""
Qualification Mission Launch File
Updated for 1.55m clearance, no CLEARING state
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import TimerAction, ExecuteProcess
from launch_ros.actions import Node
from launch.substitutions import Command, FindExecutable
import launch_ros.descriptions

def generate_launch_description():
    auv_slam_share = get_package_share_directory('auv_slam')
    
    # --- Paths ---
    urdf_file = os.path.join(auv_slam_share, 'urdf', 'orca4_description.urdf')
    rviz_config = os.path.join(auv_slam_share, 'rviz', 'urdf_config.rviz')
    bridge_config = os.path.join(auv_slam_share, 'config', 'ign_bridge.yaml')
    
    # Configs
    thruster_params = os.path.join(auv_slam_share, 'config', 'thruster_params.yaml')
    qual_params = os.path.join(auv_slam_share, 'config', 'qualification_params.yaml')
    
    # World: SAUVC Qualification Pool (25m x 16m)
    world_file = os.path.join(auv_slam_share, 'worlds', 'qualification_world.sdf')
    
    # Gazebo Environment Setup
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

    # --- Simulation Nodes ---

    # 1. Robot State Publisher (REQUIRED for spawning)
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

    # 4. Spawn Robot at STARTING LINE
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

    # --- Mission Nodes ---

    # 6. Thruster Mapper
    thruster_mapper = Node(
        package='auv_slam',
        executable='simple_thruster_mapper.py',
        name='thruster_mapper',
        output='screen',
        parameters=[thruster_params, {'use_sim_time': True}]
    )
    
    # 7. Qualification Gate Detector
    gate_detector = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='auv_slam',
                executable='qualification_detector_node.py',
                name='qualification_gate_detector',
                output='screen',
                parameters=[qual_params, {'use_sim_time': True}]
            )
        ]
    )
    
    # 8. Qualification Navigator
    navigator = TimerAction(
        period=8.0,
        actions=[
            Node(
                package='auv_slam',
                executable='qualification_navigator_node.py',
                name='qualification_navigator',
                output='screen',
                parameters=[qual_params, {'use_sim_time': True}]
            )
        ]
    )
    
    # 9. Safety Monitor - NO MORE TIMEOUT WARNINGS
    safety_monitor = Node(
        package='auv_slam',
        executable='safety_monitor_node.py',
        name='safety_monitor',
        output='screen',
        parameters=[{
            'max_depth': -1.55,
            'min_depth': 0.2,
            'max_roll': 0.785,
            'max_pitch': 0.785,
            'watchdog_timeout': 5.0,
            'max_mission_time': 36000.0,  # 10 hours - no timeout warnings
            'pool_bounds_x': [-12.5, 12.5],
            'pool_bounds_y': [-8.0, 8.0],
            'use_sim_time': True
        }]
    )
    
    # 10. RViz
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': True}]
    )
    
    # 11. Debug Image Viewer
    rqt_image_view = Node(
        package='rqt_image_view',
        executable='rqt_image_view',
        name='rqt_image_view',
        arguments=['/qualification/debug_image'],
        parameters=[{'use_sim_time': True}]
    )

    return LaunchDescription([
        robot_state_publisher,
        joint_state_publisher,
        gazebo_process,
        spawn_entity,
        bridge,
        thruster_mapper,
        gate_detector,
        navigator,
        safety_monitor,
        rqt_image_view,
        rviz_node
    ])

if __name__ == '__main__':
    generate_launch_description()