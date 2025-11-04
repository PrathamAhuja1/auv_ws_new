#!/usr/bin/env python3
"""
FIXED Qualification Launch File
- Corrected URDF path
- Proper spawn timing
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
    
    # Try multiple URDF paths (it could be in different locations)
    possible_paths = [
        os.path.join(auv_slam_share, 'urdf/orca4_description.urdf'),
        os.path.join(auv_slam_share, 'src/description/orca4_description.urdf'),
        os.path.join(auv_slam_share, '../../../src/auv_slam/src/description/orca4_description.urdf'),
    ]
    
    urdf_path = None
    for path in possible_paths:
        if os.path.exists(path):
            urdf_path = path
            print(f"✓ Found URDF at: {urdf_path}")
            break
    
    if urdf_path is None:
        print(f"ERROR: URDF not found. Searched:")
        for p in possible_paths:
            print(f"  - {p}")
        print(f"Package share: {auv_slam_share}")
        print(f"Contents: {os.listdir(auv_slam_share)}")
        raise FileNotFoundError("URDF not found in any expected location")
    
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
    
    # Launch Gazebo
    gazebo_launch = ExecuteProcess(
        cmd=[
            'gz', 'sim', '-r', '-v', '4', world_path
        ],
        output='screen',
        additional_env=gz_env,
        shell=False
    )
    
    # ====================================================================
    # 2. ROBOT STATE PUBLISHER
    # ====================================================================
    
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
    
    # ====================================================================
    # 3. SPAWN ROBOT AT STARTING ZONE (DELAYED)
    # ====================================================================
    
    spawn_entity = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=[
            "-name", "orca4_ign",
            "-topic", "robot_description",
            "-x", "-14.3",  # Starting zone X position
            "-y", "0.0",    # Centered
            "-z", "-0.5",   # Just below surface
            "--ros-args",
            "--log-level", "info",
        ],
        parameters=[{"use_sim_time": True}],
    )
    
    # Delay spawn to ensure Gazebo is ready
    delayed_spawn = TimerAction(
        period=5.0,  # Wait 5 seconds for Gazebo to initialize
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
        period=8.0,  # Start after robot is spawned and stable
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
            'max_mission_time': 600.0,
            'pool_bounds_x': [-15.0, 15.0],
            'pool_bounds_y': [-7.5, 7.5]
        }]
    )
    
    # ====================================================================
    # 9. DEBUG VISUALIZATION (Optional - commented out if rqt not installed)
    # ====================================================================
    
    # Uncomment if you have rqt_image_view installed
    # rqt_image_view = TimerAction(
    #     period=10.0,
    #     actions=[
    #         ExecuteProcess(
    #             cmd=['ros2', 'run', 'rqt_image_view', 'rqt_image_view', '--topic', '/qual_gate/debug_image'],
    #             output='screen',
    #             shell=False
    #         )
    #     ]
    # )
    
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
        gazebo_launch,     
        robot_state_publisher,   
        bridge,             
        delayed_spawn,    
        thruster_mapper,  
        qual_gate_detector, 
        qual_gate_navigator, 
        safety_monitor,     
    #    rqt_image_view
    ])


if __name__ == '__main__':
    generate_launch_description()