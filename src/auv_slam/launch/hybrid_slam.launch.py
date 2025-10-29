"""
Hybrid SLAM Launch File - ORB-SLAM3 + DSO Dynamic Switching
Compatible with your existing system
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition

def generate_launch_description():
    # Get package directory
    auv_slam_share_dir = get_package_share_directory('auv_slam')
    
    # Config paths
    vocab_path = os.path.join(auv_slam_share_dir, 'config', 'ORBvoc.txt')
    orb_settings_path = os.path.join(auv_slam_share_dir, 'config', 'config_stereo.yaml')
    dso_calib_path = os.path.join(auv_slam_share_dir, 'config', 'dso_camera_calibration.txt')
    
    # Launch arguments
    declare_enable_dso = DeclareLaunchArgument(
        'enable_dso',
        default_value='true',
        description='Enable DSO backend (requires DSO installation)'
    )
    
    declare_enable_optical_flow = DeclareLaunchArgument(
        'enable_optical_flow',
        default_value='true',
        description='Enable optical flow fallback'
    )
    
    declare_show_viewer = DeclareLaunchArgument(
        'show_slam_viewer',
        default_value='false',
        description='Show ORB-SLAM3 viewer window'
    )
    
    declare_feature_low = DeclareLaunchArgument(
        'feature_threshold_low',
        default_value='40',
        description='Feature count threshold to switch to DSO'
    )
    
    declare_feature_high = DeclareLaunchArgument(
        'feature_threshold_high',
        default_value='60',
        description='Feature count threshold to switch back to ORB-SLAM3'
    )
    
    # Hybrid SLAM Node
    hybrid_slam_node = Node(
        package='auv_slam',
        executable='hybrid_slam_node',
        name='hybrid_slam_node',
        output='screen',
        parameters=[{
            'vocab_path': vocab_path,
            'settings_path': orb_settings_path,
            'dso_calib_path': dso_calib_path,
            'map_frame': 'map',
            'show_slam_viewer': LaunchConfiguration('show_slam_viewer'),
            'enable_dso': LaunchConfiguration('enable_dso'),
            'enable_optical_flow_fallback': LaunchConfiguration('enable_optical_flow'),
            'feature_threshold_low': LaunchConfiguration('feature_threshold_low'),
            'feature_threshold_high': LaunchConfiguration('feature_threshold_high'),
            'switch_cooldown': 2.0,
        }],
    )
    
    return LaunchDescription([
        declare_enable_dso,
        declare_enable_optical_flow,
        declare_show_viewer,
        declare_feature_low,
        declare_feature_high,
        hybrid_slam_node,
    ])