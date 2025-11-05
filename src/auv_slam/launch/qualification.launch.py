#!/usr/bin/env python3
"""
Working Qualification Launch - Uses same pattern as mission.launch.py
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction, DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    pkg = get_package_share_directory('auv_slam')
    
    world_path = os.path.join(pkg, 'worlds/qualification_world.sdf')
    urdf_path = os.path.join(pkg, 'src/description/orca4_description.urdf')
    bridge_config = os.path.join(pkg, 'config/ign_bridge.yaml')
    thruster_params = os.path.join(pkg, 'config/thruster_params.yaml')
    qual_params = os.path.join(pkg, 'config/qualification_params.yaml')
    
    # Verify paths
    print(f"World: {world_path} - Exists: {os.path.exists(world_path)}")
    print(f"URDF: {urdf_path} - Exists: {os.path.exists(urdf_path)}")
    
    # Gazebo environment (SAME AS MISSION.LAUNCH.PY)
    models_path = os.path.join(pkg, "models")
    gz_resource = os.environ.get("GZ_SIM_RESOURCE_PATH", "")
    
    gz_env = {
        'GZ_SIM_SYSTEM_PLUGIN_PATH':
           ':'.join([os.environ.get('GZ_SIM_SYSTEM_PLUGIN_PATH', ''),
                     os.environ.get('LD_LIBRARY_PATH', '')]),
        'IGN_GAZEBO_SYSTEM_PLUGIN_PATH':  
           ':'.join([os.environ.get('IGN_GAZEBO_SYSTEM_PLUGIN_PATH', ''),
                     os.environ.get('LD_LIBRARY_PATH', '')]),
        'GZ_SIM_RESOURCE_PATH': f"{gz_resource}:{models_path}"
    }
    
    use_sim_time = LaunchConfiguration("use_sim_time")
    log_level = LaunchConfiguration("log_level")
    
    # Robot description (from file, not installed)
    with open(urdf_path, 'r') as f:
        robot_desc = f.read()
    
    # 1. Gazebo
    gazebo = ExecuteProcess(
        cmd=['gz', 'sim', '-r', '-v', '4', world_path],
        output='screen',
        additional_env=gz_env
    )
    
    # 2. Robot State Publisher
    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'robot_description': robot_desc,
            'use_sim_time': use_sim_time
        }],
        output='screen'
    )
    
    # 3. Bridge
    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=['--ros-args', '-p', f'config_file:={bridge_config}'],
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}]
    )
    
    # 4. Spawn - Ensure robot_description exists first
    spawn = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=[
            "-name", "orca4_ign",
            "-topic", "robot_description",
            "-x", "-14.3",
            "-y", "0.0", 
            "-z", "-0.5",
        ],
        parameters=[{"use_sim_time": True}],
    )
    
    # Wait for robot_state_publisher + Gazebo to be ready
    delayed_spawn = TimerAction(period=10.0, actions=[spawn])
    
    # 5. Thruster Mapper
    thruster = Node(
        package='auv_slam',
        executable='simple_thruster_mapper.py',
        name='thruster_mapper',
        output='screen',
        parameters=[thruster_params, {"use_sim_time": use_sim_time}]
    )
    
    # 6. Gate Detector
    detector = Node(
        package='auv_slam',
        executable='qualification_gate_detector_node.py',
        name='qualification_gate_detector',
        output='screen',
        parameters=[qual_params, {"use_sim_time": use_sim_time}]
    )
    
    # 7. Navigator
    navigator = TimerAction(
        period=10.0,
        actions=[Node(
            package='auv_slam',
            executable='qualification_navigator_node.py',
            name='qualification_navigator',
            output='screen',
            parameters=[qual_params, {"use_sim_time": use_sim_time}]
        )]
    )
    
    # 8. Safety
    safety = Node(
        package='auv_slam',
        executable='safety_monitor_node.py',
        name='safety_monitor',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
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
    
    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="True"),
        DeclareLaunchArgument("log_level", default_value="info"),
        gazebo,
        rsp,
        bridge,
        delayed_spawn,
        thruster,
        detector,
        navigator,
        safety
    ])