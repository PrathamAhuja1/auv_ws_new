import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    auv_slam_share_dir = get_package_share_directory('auv_slam')

    # --- Paths ---
    display_launch_path = os.path.join(
        auv_slam_share_dir, 'launch', 'display.launch.py'
    )
    thruster_params_path = os.path.join(
        auv_slam_share_dir, 'config', 'thruster_params.yaml'
    )
    gate_params_path = os.path.join(
        auv_slam_share_dir, 'config', 'gate_params.yaml'
    )
    flare_params_path = os.path.join(
        auv_slam_share_dir, 'config', 'flare_params.yaml'
    )

    # --- Nodes ---

    # 1. Simulation Launch (Includes Gazebo, RViz, robot_state_publisher, etc.)
    simulation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(display_launch_path),
        launch_arguments={'use_sim_time': 'True'}.items()
    )

    # 2. Simple Thruster Mapper
    thruster_mapper_node = Node(
        package='auv_slam',
        executable='simple_thruster_mapper.py',
        name='simple_thruster_mapper',
        output='screen',
        parameters=[thruster_params_path]
    )

    # 3. Teleoperation Node (Publishes Twist commands now)
    teleop_node = Node(
        package='auv_slam',
        executable='bluerov_teleop_ign.py',
        name='bluerov_teleop_ign_node',
        output='screen',
        prefix='xterm -e'
    )


    # 4. Gate Detector
    gate_detector_node = Node(
        package='auv_slam',
        executable='gate_detector_node.py',
        name='gate_detector_node',
        output='screen',
        parameters=[gate_params_path]
    )

    # 5. Flare Detector
    flare_detector_node = Node(
        package='auv_slam',
        executable='flare_detection.py',
        name='flare_detector_node',
        output='screen',
        parameters=[flare_params_path]
    )

    return LaunchDescription([
        simulation_launch,
        thruster_mapper_node,
        teleop_node,
        gate_detector_node,
        flare_detector_node,
    ])