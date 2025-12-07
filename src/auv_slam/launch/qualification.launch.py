#!/usr/bin/env python3
"""
SAUVC Qualification Mission Launch File
Launches complete qualification task in qualification_world.sdf

Mission Requirements:
- Pass through orange-marked gate (forward)
- U-turn 180°
- Pass through gate again (reverse)
- Don't touch walls, floor, or gate
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import TimerAction, ExecuteProcess
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterValue


def generate_launch_description():
    auv_slam_share = get_package_share_directory('auv_slam')
    
    # --- Paths ---
    thruster_params = os.path.join(auv_slam_share, 'config', 'thruster_params.yaml')
    qual_params = os.path.join(auv_slam_share, 'config', 'qualification_params.yaml')
    bridge_config_path = os.path.join(auv_slam_share, 'config', 'ign_bridge.yaml')
    
    # Model & Config
    default_model_path = os.path.join(auv_slam_share, 'urdf', 'orca4_description.urdf')
    default_rviz_config_path = os.path.join(auv_slam_share, 'rviz', 'urdf_config.rviz')
    
    # Qualification World
    world_path = os.path.join(auv_slam_share, "worlds", "qualification_world.sdf")
    
    # --- Gazebo Environment Setup ---
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
    
    # --- Launch Arguments ---
    use_sim_time = LaunchConfiguration('use_sim_time', default='True')
    gz_verbosity = LaunchConfiguration('gz_verbosity', default='2')
    
    # 1. Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'robot_description': ParameterValue(Command(['xacro ', default_model_path]), value_type=str),
            'use_sim_time': use_sim_time
        }]
    )
    
    # 2. Joint State Publisher
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[{
            'robot_description': ParameterValue(Command(['xacro ', default_model_path]), value_type=str)
        }]
    )
    
    # 3. RViz
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', default_rviz_config_path],
    )
    
    # 4. Gazebo Simulation
    gz_sim = ExecuteProcess(
        cmd=['ign', 'gazebo', '-r', '-v', gz_verbosity, world_path],
        output='screen',
        additional_env=gz_env
    )
    
    # 5. Spawn Robot at Starting Zone (-10, 0, -0.3)
    spawn_entity = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=[
            "-name", "orca4_ign",
            "-topic", "robot_description",
            "-x", "-10.0",  # Starting zone X position
            "-y", "0.0",    # Centered in Y
            "-z", "-0.3",   # Just below surface
            "--ros-args"
        ],
        parameters=[{"use_sim_time": use_sim_time}],
    )
    
    # 6. ROS-Gazebo Bridge
    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=['--ros-args', '-p', f'config_file:={bridge_config_path}'],
        output="screen",
    )
    
    # --- QUALIFICATION MISSION NODES ---
    
    # 7. Thruster Mapper
    thruster_mapper = Node(
        package='auv_slam',
        executable='simple_thruster_mapper.py',
        name='thruster_mapper',
        output='screen',
        parameters=[thruster_params]
    )
    
    # 8. Qualification Gate Detector (ORANGE posts)
    gate_detector = Node(
        package='auv_slam',
        executable='qualification_detector_node.py',
        name='qualification_detector_node',
        output='screen',
        parameters=[qual_params]
    )
    
    # 9. Qualification Navigator (2-pass logic)
    gate_navigator = TimerAction(
        period=5.0,  # Wait for simulation to stabilize
        actions=[
            Node(
                package='auv_slam',
                executable='qualification_navigator_node.py',
                name='qualification_navigator_node',
                output='screen',
                parameters=[qual_params]
            )
        ]
    )
    
    # 10. Safety Monitor
    safety_monitor = Node(
        package='auv_slam',
        executable='safety_monitor_node.py',
        name='safety_monitor',
        output='screen',
        parameters=[{
            'max_depth': -1.5,      # Qualification pool is shallower
            'min_depth': 0.1,
            'max_roll': 0.785,
            'max_pitch': 0.785,
            'watchdog_timeout': 5.0,
            'max_mission_time': 300.0  # 5 minutes for qualification
        }]
    )
    
    # 11. Debug Image View (delayed start)
    rqt_image_view = ExecuteProcess(
        cmd=['rqt_image_view', '/gate/debug_image'],
        output='screen',
        shell=False
    )
    
    return LaunchDescription([
        # Simulation Stack
        robot_state_publisher,
        joint_state_publisher,
        rviz_node,
        gz_sim,
        spawn_entity,
        bridge,
        
        # Mission Stack
        thruster_mapper,
        gate_detector,
        gate_navigator,
        safety_monitor,
        
        # Debug visualization (delayed start)
        TimerAction(period=3.0, actions=[rqt_image_view]),
    ])