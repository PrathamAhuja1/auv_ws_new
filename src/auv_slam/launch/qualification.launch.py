#!/usr/bin/env python3
"""
Qualification Mission Launch File
Launches complete qualification system for SAUVC Qualification Task

Components:
- Gazebo simulation with qualification world
- RViz visualization
- Qualification gate detector (orange markers)
- Qualification navigator (state machine)
- Thruster mapper
- Safety monitor
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, TimerAction, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
import launch


def generate_launch_description():
    auv_slam_share = get_package_share_directory('auv_slam')
    
    # Config paths
    thruster_params = os.path.join(auv_slam_share, 'config', 'thruster_params.yaml')
    qualification_params = os.path.join(auv_slam_share, 'config', 'qualification_params.yaml')
    safety_params = os.path.join(auv_slam_share, 'config', 'safety_params.yaml')
    
    # World path
    qualification_world = os.path.join(auv_slam_share, 'worlds', 'qualification_world.sdf')
    
    # URDF path
    urdf_path = os.path.join(auv_slam_share, 'urdf', 'orca4_description.urdf')
    
    # RViz config
    rviz_config = os.path.join(auv_slam_share, 'rviz', 'urdf_config.rviz')
    
    # Bridge config
    bridge_config = os.path.join(auv_slam_share, 'config', 'ign_bridge.yaml')
    
    # Launch arguments
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='True',
        description='Use simulation time'
    )
    
    declare_log_level = DeclareLaunchArgument(
        'log_level',
        default_value='info',
        description='Logging level'
    )
    
    declare_enable_debug = DeclareLaunchArgument(
        'enable_debug_view',
        default_value='true',
        description='Launch rqt_image_view for gate debugging'
    )
    
    # ========================================================================
    # 1. Gazebo Simulation with Qualification World
    # ========================================================================
    
    # Gazebo environment variables
    pkg_share_dir = get_package_share_directory('auv_slam')
    gz_models_path = os.path.join(pkg_share_dir, "models")
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
    
    gazebo_launch = launch.actions.ExecuteProcess(
        cmd=['gz', 'sim', '-r', '-v', '3', qualification_world],
        output='screen',
        additional_env=gz_env,
        shell=False,
    )
    
    # ========================================================================
    # 2. Robot State Publisher
    # ========================================================================
    
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': launch.substitutions.Command(['xacro ', urdf_path]),
            'use_sim_time': LaunchConfiguration('use_sim_time')
        }]
    )
    
    # ========================================================================
    # 3. ROS-Gazebo Bridge
    # ========================================================================
    
    ros_gz_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            '--ros-args',
            '-p', f'config_file:={bridge_config}'
        ],
        output="screen",
    )
    
    # ========================================================================
    # 4. Spawn AUV in Qualification Arena
    # ========================================================================
    
    spawn_entity = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=[
            "-name", "orca4_ign",
            "-topic", "robot_description",
            "-z", "0.2",  # At water surface
            "-x", "-9.3",  # At starting zone
            "-y", "0.0",
            "--ros-args",
            "--log-level", LaunchConfiguration('log_level'),
        ],
        parameters=[{"use_sim_time": LaunchConfiguration('use_sim_time')}],
    )
    
    # ========================================================================
    # 5. Thruster Mapper
    # ========================================================================
    
    thruster_mapper = Node(
        package='auv_slam',
        executable='simple_thruster_mapper.py',
        name='thruster_mapper',
        output='screen',
        parameters=[thruster_params]
    )
    
    # ========================================================================
    # 6. Qualification Gate Detector (Orange Markers)
    # ========================================================================
    
    gate_detector = Node(
        package='auv_slam',
        executable='qualification_gate_detector.py',
        name='qualification_gate_detector',
        output='screen',
        parameters=[qualification_params]
    )
    
    # ========================================================================
    # 7. Qualification Navigator (State Machine)
    # ========================================================================
    
    qualification_navigator = TimerAction(
        period=5.0,  # Wait for systems to initialize
        actions=[
            Node(
                package='auv_slam',
                executable='qualification_navigator.py',
                name='qualification_navigator',
                output='screen',
                parameters=[qualification_params]
            )
        ]
    )
    
    # ========================================================================
    # 8. Safety Monitor
    # ========================================================================
    
    safety_monitor = Node(
        package='auv_slam',
        executable='safety_monitor_node.py',
        name='safety_monitor',
        output='screen',
        parameters=[qualification_params]
    )
    
    # ========================================================================
    # 9. RViz Visualization
    # ========================================================================
    
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config],
    )
    
    # ========================================================================
    # 10. Debug Visualization Tools
    # ========================================================================
    
    # rqt_image_view for gate debug
    rqt_gate_debug = TimerAction(
        period=7.0,
        actions=[
            ExecuteProcess(
                cmd=['rqt_image_view', '/qualification/debug_image'],
                output='screen',
                shell=False
            )
        ]
    )
    
    # rqt for raw camera
    rqt_raw_camera = TimerAction(
        period=7.0,
        actions=[
            ExecuteProcess(
                cmd=['rqt_image_view', '/camera_forward/image_raw'],
                output='screen',
                shell=False
            )
        ]
    )
    
    # ========================================================================
    # Launch Description
    # ========================================================================
    
    return LaunchDescription([
        # Arguments
        declare_use_sim_time,
        declare_log_level,
        declare_enable_debug,
        
        # Simulation
        gazebo_launch,
        
        # Robot infrastructure
        robot_state_publisher,
        ros_gz_bridge,
        spawn_entity,
        
        # Core systems
        thruster_mapper,
        gate_detector,
        qualification_navigator,
        safety_monitor,
        
        # Visualization
        rviz,
        rqt_gate_debug,
        
        # Optional: Raw camera view
        # rqt_raw_camera,
    ])


if __name__ == '__main__':
    generate_launch_description()