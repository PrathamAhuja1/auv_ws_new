#!/usr/bin/env python3
"""
FIXED Qualification Launch File
Key fix: Added proper thruster topic remapping in bridge
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction, DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    pkg = get_package_share_directory('auv_slam')
    
    # Paths
    world_path = os.path.join(pkg, 'worlds/qualification_world.sdf')
    urdf_path = os.path.join(pkg, 'src/description/orca4_description.urdf')
    thruster_params = os.path.join(pkg, 'config/thruster_params.yaml')
    qual_params = os.path.join(pkg, 'config/qualification_params.yaml')
    
    print(f"✓ World: {world_path}")
    print(f"✓ URDF: {urdf_path}")
    
    # Gazebo environment
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
    
    # Robot description
    with open(urdf_path, 'r') as f:
        robot_desc = f.read()
    
    # 1. Gazebo
    gazebo = ExecuteProcess(
        cmd=['gz', 'sim', '-r', '-v', '3', world_path],
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
    
    # 3. FIXED Bridge - Complete thruster remapping like mission.launch.py
    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            # Clock
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            # Odometry
            '/model/orca4_ign/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry',
            # Camera
            '/camera_forward/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
            '/camera_forward/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
            # Thrusters - CRITICAL: Proper remapping
            '/model/orca4_ign/joint/thruster1_joint/cmd_pos@std_msgs/msg/Float64]gz.msgs.Double',
            '/model/orca4_ign/joint/thruster2_joint/cmd_pos@std_msgs/msg/Float64]gz.msgs.Double',
            '/model/orca4_ign/joint/thruster3_joint/cmd_pos@std_msgs/msg/Float64]gz.msgs.Double',
            '/model/orca4_ign/joint/thruster4_joint/cmd_pos@std_msgs/msg/Float64]gz.msgs.Double',
            '/model/orca4_ign/joint/thruster5_joint/cmd_pos@std_msgs/msg/Float64]gz.msgs.Double',
            '/model/orca4_ign/joint/thruster6_joint/cmd_pos@std_msgs/msg/Float64]gz.msgs.Double',
            '--ros-args',
            # Remap odometry
            '-r', '/model/orca4_ign/odometry:=/ground_truth/odom',
            # CRITICAL: Remap ALL thruster commands
            '-r', '/model/orca4_ign/joint/thruster1_joint/cmd_pos:=/thruster1_cmd',
            '-r', '/model/orca4_ign/joint/thruster2_joint/cmd_pos:=/thruster2_cmd',
            '-r', '/model/orca4_ign/joint/thruster3_joint/cmd_pos:=/thruster3_cmd',
            '-r', '/model/orca4_ign/joint/thruster4_joint/cmd_pos:=/thruster4_cmd',
            '-r', '/model/orca4_ign/joint/thruster5_joint/cmd_pos:=/thruster5_cmd',
            '-r', '/model/orca4_ign/joint/thruster6_joint/cmd_pos:=/thruster6_cmd'
        ],
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}]
    )
    
    # 4. Thruster Mapper
    thruster = Node(
        package='auv_slam',
        executable='simple_thruster_mapper.py',
        name='thruster_mapper',
        output='screen',
        parameters=[thruster_params, {"use_sim_time": use_sim_time}]
    )
    
    # 5. Gate Detector
    detector = Node(
        package='auv_slam',
        executable='qualification_gate_detector_node.py',
        name='qualification_gate_detector',
        output='screen',
        parameters=[qual_params, {"use_sim_time": use_sim_time}]
    )
    
    # 6. Navigator (delayed 5s for faster start)
    navigator = TimerAction(
        period=5.0,
        actions=[Node(
            package='auv_slam',
            executable='qualification_navigator_node.py',
            name='qualification_navigator',
            output='screen',
            parameters=[qual_params, {"use_sim_time": use_sim_time}],
            # Force output to see if node is running
            emulate_tty=True
        )]
    )
    
    # 7. Safety Monitor
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
    
    # 8. OPTIONAL: Diagnostic node to verify thruster commands
    diagnostic = Node(
        package='auv_slam',
        executable='diagnostic_node.py',
        name='diagnostic_node',
        output='screen',
        parameters=[{"use_sim_time": use_sim_time}]
    )
    
    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="True"),
        gazebo,
        rsp,
        bridge,  # FIXED bridge with proper remapping
        thruster,
        detector,
        navigator,
        safety,
        # diagnostic  # Uncomment to see thruster diagnostics
    ])