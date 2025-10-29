import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # Get the package share directory
    auv_slam_share_dir = get_package_share_directory('auv_slam')
    
    # Define paths to config files
    vocab_path = os.path.join(auv_slam_share_dir, 'config', 'ORBvoc.txt')
    settings_path = os.path.join(auv_slam_share_dir, 'config', 'config_stereo.yaml')
    
    return LaunchDescription([
        Node(
            package='auv_slam',
            executable='stereo_slam_node',
            name='stereo_slam_node',
            output='screen',
            parameters=[
                {'vocab_path': vocab_path},
                {'settings_path': settings_path},
                {'map_frame': 'map'}
            ]
        )
    ])