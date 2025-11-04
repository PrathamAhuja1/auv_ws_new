#!/usr/bin/env python3
"""
FIXED Qualification Launch File
- Corrected spawn position for qualification task
- Proper timing to ensure Gazebo is ready
- Robot spawns at starting zone
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription, 
    DeclareLaunchArgument, 
    TimerAction, 
    ExecuteProcess,
    RegisterEventHandler
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
import launch
import launch_ros


def generate_launch_description():
    auv_slam_share = get_package_share_directory('auv_slam')
    
    # Config paths
    thruster_params = os.path.join(auv_slam_share, 'config', 'thruster_params.yaml')
    qual_params = os.path.join(auv_slam_share, 'config', 'qualification_params.yaml')
    
    # World and model paths
    world_path = os.path.join(auv_slam_share, "worlds/qualification_world.sdf")
    bridge_config_path = os.path.join(auv_slam_share, 'config', 'ign_bridge.yaml')
    pkg_share_sub = launch_ros.substitutions.FindPackageShare(package='auv_slam').find('auv_slam')
    default_model_path = os.path.join(pkg_share_sub, 'urdf/orca4_description.urdf')
    
    # Launch arguments
    declare_enable_debug = DeclareLaunchArgument(
        'enable_debug_view',
        default_value='true',
        description='Launch rqt_image_view for gate debugging'
    )
    
    use_sim_time = LaunchConfiguration("use_sim_time")
    log_level = LaunchConfiguration("log_level")
    gz_verbosity = LaunchConfiguration("gz_verbosity")
    
    # ====================================================================
    # 1. GAZEBO SIMULATION (Qualification World)
    # ====================================================================
    
    # Gazebo environment setup
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
    
    # Launch Gazebo (ensure it's installed and accessible)
    gazebo_launch = ExecuteProcess(
        cmd=[
            'gz', 'sim', '-r', '-v', '4', world_path
        ],
        output='screen',
        additional_env=gz_env,
        shell=False,
        on_exit=lambda event, context: print("Gazebo exited!")
    )
    
    # ====================================================================
    # 2. ROBOT STATE PUBLISHER
    # ====================================================================
    
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'robot_description': launch_ros.descriptions.ParameterValue(
                launch.substitutions.Command(['xacro ', default_model_path]),
                value_type=str
            ),
            'use_sim_time': True
        }]
    )
    
    # ====================================================================
    # 3. SPAWN ROBOT AT STARTING ZONE (DELAYED)
    # ====================================================================
    
    # CRITICAL FIX: Spawn at starting zone for qualification task
    # Starting zone is at X=-14.3 (near the starting wall)
    spawn_entity = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=[
            "-name", "orca4_ign",
            "-topic", "robot_description",
            "-x", "-14.3",  # Starting zone X position
            "-y", "0.0",    # Centered
            "-z", "-0.5",   # Just below surface (shallow depth)
            "--ros-args",
            "--log-level", "info",
        ],
        parameters=[{"use_sim_time": True}],
    )
    
    # Delay spawn to ensure Gazebo is ready
    delayed_spawn = TimerAction(
        period=3.0,  # Wait 3 seconds for Gazebo to initialize
        actions=[spawn_entity]
    )
    
    # ====================================================================
    # 4. BRIDGE (Gazebo <-> ROS)
    # ====================================================================
    
    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            '--ros-args',
            '-p', f'config_file:={bridge_config_path}'
        ],
        output="screen",
        parameters=[{"use_sim_time": True}]
    )
    
    # ====================================================================
    # 5. THRUSTER MAPPER
    # ====================================================================
    
    thruster_mapper = Node(
        package='auv_slam',
        executable='simple_thruster_mapper.py',
        name='thruster_mapper',
        output='screen',
        parameters=[thruster_params, {"use_sim_time": True}]
    )
    
    # ====================================================================
    # 6. QUALIFICATION GATE DETECTOR
    # ====================================================================
    
    qual_gate_detector = Node(
        package='auv_slam',
        executable='qualification_gate_detector_node.py',
        name='qualification_gate_detector',
        output='screen',
        parameters=[qual_params, {"use_sim_time": True}]
    )
    
    # ====================================================================
    # 7. QUALIFICATION NAVIGATOR (Delayed start)
    # ====================================================================
    
    qual_gate_navigator = TimerAction(
        period=5.0,  # Start after robot is spawned and stable
        actions=[
            Node(
                package='auv_slam',
                executable='qualification_navigator_node.py',
                name='qualification_navigator',
                output='screen',
                parameters=[qual_params, {"use_sim_time": True}]
            )
        ]
    )
    
    # ====================================================================
    # 8. SAFETY MONITOR
    # ====================================================================
    
    safety_monitor = Node(
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
            'max_mission_time': 600.0,  # 10 minutes for qualification
            'pool_bounds_x': [-15.0, 15.0],
            'pool_bounds_y': [-7.5, 7.5]
        }]
    )
    
    # ====================================================================
    # 9. DEBUG VISUALIZATION (Optional)
    # ====================================================================
    
    rqt_image_view = TimerAction(
        period=6.0,
        actions=[
            ExecuteProcess(
                cmd=['rqt_image_view', '/qual_gate/debug_image'],
                output='screen',
                shell=False
            )
        ]
    )
    
    # ====================================================================
    # LAUNCH DESCRIPTION
    # ====================================================================
    
    return LaunchDescription([
        # Arguments
        DeclareLaunchArgument(
            name="use_sim_time",
            default_value="True",
            description="Use simulation time"
        ),
        DeclareLaunchArgument(
            name="log_level",
            default_value="info",
            description="Logging level"
        ),
        DeclareLaunchArgument(
            name="gz_verbosity",
            default_value="3",
            description="Gazebo verbosity level (0-4)"
        ),
        declare_enable_debug,
        
        # Core components (in order)
        gazebo_launch,              # 1. Start Gazebo with qualification world
        robot_state_publisher,      # 2. Publish robot description
        bridge,                     # 3. Bridge Gazebo and ROS
        delayed_spawn,              # 4. Spawn robot (delayed for stability)
        thruster_mapper,            # 5. Thruster control
        qual_gate_detector,         # 6. Gate detection
        qual_gate_navigator,        # 7. Navigation (delayed)
        safety_monitor,             # 8. Safety monitoring
        rqt_image_view,            # 9. Debug view (delayed)
    ])


if __name__ == '__main__':
    generate_launch_description()