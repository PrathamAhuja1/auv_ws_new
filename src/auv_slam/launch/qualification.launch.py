#!/usr/bin/env python3
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch.substitutions import Command
from launch_ros.actions import Node
import launch_ros.descriptions

def generate_launch_description():
    pkg_auv_slam = get_package_share_directory('auv_slam')
    
    # Paths
    qual_config = os.path.join(pkg_auv_slam, 'config', 'qualification_params.yaml')
    world_path = os.path.join(pkg_auv_slam, 'worlds', 'qualification_world.sdf')
    bridge_config = os.path.join(pkg_auv_slam, 'config', 'ign_bridge.yaml')
    urdf_path = os.path.join(pkg_auv_slam, 'urdf', 'orca4_description.urdf')

    # Environment
    gz_models_path = os.path.join(pkg_auv_slam, "models")
    gz_resource_path = os.environ.get("GZ_SIM_RESOURCE_PATH", default="")
    gz_env = {
        'GZ_SIM_RESOURCE_PATH': ':'.join([gz_resource_path, gz_models_path]),
        'IGN_GAZEBO_RESOURCE_PATH': ':'.join([gz_resource_path, gz_models_path])
    }

    # 1. Gazebo Sim
    gazebo_sim = ExecuteProcess(
        cmd=['ign', 'gazebo', '-r', '-v', '3', world_path],
        output='screen',
        additional_env=gz_env
    )

    # 2. Parameter Bridge (Clock, etc)
    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=['--ros-args', '-p', f'config_file:={bridge_config}'],
        output="screen",
    )
    

    # 4. Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': launch_ros.descriptions.ParameterValue(
                Command(['xacro ', urdf_path]), value_type=str
            ),
            'use_sim_time': True
        }]
    )

    # 5. Thruster Mapper
    thruster_mapper = Node(
        package='auv_slam',
        executable='simple_thruster_mapper.py',
        name='simple_thruster_mapper',
        output='screen',
        parameters=[qual_config]
    )


    gate_detector = Node(
    package='auv_slam',
    executable='qualification_detector.py',
    name='qualification_gate_detector',
    output='screen',
    parameters=[qual_config],
    remappings=[
        ('image_raw', '/camera_forward/image_raw'), 
        ('camera_info', '/camera_forward/camera_info'),
    ]
)

    # 7. Navigator (Delayed start)
    navigator = TimerAction(
        period=8.0,
        actions=[
            Node(
                package='auv_slam',
                executable='qualification_navigator.py',
                name='qualification_navigator',
                output='screen',
                parameters=[qual_config]
            )
        ]
    )

    # 8. Visualizer (RQT)
    # Starts automatically looking at the debug feed
    rqt_view = Node(
        package='rqt_image_view',
        executable='rqt_image_view',
        name='rqt_image_view',
        arguments=['/gate/debug_image']
    )

    return LaunchDescription([
        gazebo_sim,
        bridge,
        robot_state_publisher,
        thruster_mapper,
        gate_detector,
        navigator,
        TimerAction(period=5.0, actions=[rqt_view])
    ])