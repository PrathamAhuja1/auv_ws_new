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
    
    # --- PATHS ---
    # Ensure this points to your NEW world file
    world_path = os.path.join(pkg_auv_slam, 'worlds', 'qualification_world.sdf')
    qual_config = os.path.join(pkg_auv_slam, 'config', 'qualification_params.yaml')
    bridge_config = os.path.join(pkg_auv_slam, 'config', 'ign_bridge.yaml')
    rviz_config = os.path.join(pkg_auv_slam, 'rviz', 'urdf_config.rviz')
    urdf_path = os.path.join(pkg_auv_slam, 'urdf', 'orca4_description.urdf')

    # --- ENVIRONMENT ---
    # Setup paths so Gazebo can find the orange gate models
    gz_models_path = os.path.join(pkg_auv_slam, "models")
    gz_resource_path = os.environ.get("GZ_SIM_RESOURCE_PATH", default="")
    gz_env = {
        'GZ_SIM_RESOURCE_PATH': ':'.join([gz_resource_path, gz_models_path]),
        'IGN_GAZEBO_RESOURCE_PATH': ':'.join([gz_resource_path, gz_models_path])
    }

    # --- NODES ---

    # 1. State Publishers
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

    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[{'use_sim_time': True}]
    )

    # 2. Simulation (Ignition Gazebo)
    gazebo_sim = ExecuteProcess(
        cmd=['ign', 'gazebo', '-r', '-v', '3', world_path],
        output='screen',
        additional_env=gz_env
    )

    # 3. Bridges
    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=['--ros-args', '-p', f'config_file:={bridge_config}'],
        output="screen",
    )
    
    camera_bridge = Node(
        package='ros_gz_image',
        executable='image_bridge',
        name='bridge_gz_ros_camera_image',
        output='screen',
        parameters=[{'use_sim_time': True}],
        arguments=['/stereo_left/image', '/stereo_left/depth_image'],
    )

    # 4. RViz
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': True}]
    )

    # 5. Qualification Nodes
    thruster_mapper = Node(
        package='auv_slam',
        executable='simple_thruster_mapper.py',
        name='simple_thruster_mapper',
        output='screen',
        parameters=[qual_config]
    )

    gate_detector = Node(
        package='auv_slam',
        executable='gate_detector_node.py',
        name='qualification_gate_detector',
        output='screen',
        parameters=[qual_config],
        # Remap to ensure we are listening to the right camera topic
        remappings=[
            ('/image_raw', '/stereo_left/image'), 
            ('/gate/debug_image', '/gate/debug_image')
        ]
    )

    navigator = TimerAction(
        period=10.0,
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

    # 6. RQT Image View (Added as requested)
    # This will open a window showing what the robot sees
    rqt_view = Node(
        package='rqt_image_view',
        executable='rqt_image_view',
        name='rqt_image_view',
        arguments=['/gate/debug_image']
    )

    return LaunchDescription([
        gazebo_sim,
        robot_state_publisher,
        joint_state_publisher,
        bridge,
        camera_bridge,
        rviz,
        thruster_mapper,
        gate_detector,
        navigator,
        TimerAction(period=5.0, actions=[rqt_view])
    ])