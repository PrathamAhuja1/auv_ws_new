#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32 # Assuming pressure/depth comes as Float32 from serial bridge
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import numpy as np

class DepthProcessorNode(Node):
    def __init__(self):
        super().__init__('depth_processor_node')

        self.declare_parameter('depth_topic_in', '/auv/raw_depth')
        self.declare_parameter('pose_topic_out', '/depth/pose')
        self.declare_parameter('depth_frame_id', 'odom') # Frame ID for the pose message
        self.declare_parameter('depth_variance', 0.01) # TUNE: How much variance (uncertainty) in m^2

        depth_topic_in = self.get_parameter('depth_topic_in').value
        pose_topic_out = self.get_parameter('pose_topic_out').value
        self.frame_id = self.get_parameter('depth_frame_id').value
        self.depth_variance = self.get_parameter('depth_variance').value

        # --- Quality of Service ---
        qos_profile_sensor_data = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=5,
            durability=DurabilityPolicy.VOLATILE)
        qos_profile_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST, depth=5,
            durability=DurabilityPolicy.VOLATILE)


        self.depth_sub = self.create_subscription(
            Float32,
            depth_topic_in, # INPUT: Topic from serial_bridge_node
            self.depth_callback,
            qos_profile_sensor_data)

        self.pose_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            pose_topic_out, # OUTPUT: Topic for EKF
            qos_profile_reliable)

        self.get_logger().info(f"Depth Processor started. Input: '{depth_topic_in}', Output: '{pose_topic_out}'")

    def depth_callback(self, msg: Float32):
        depth_value = msg.data # Assuming positive value means depth below surface

        # Convert depth to Z position in the 'odom' frame.
        # CRITICAL: Define your odom frame convention!
        # Option 1: Z=0 is the water surface, positive Z is UP. Pose.z = -depth_value
        # Option 2: Z=0 is the pool bottom, positive Z is UP. Pose.z = pool_bottom_depth - depth_value (Needs pool depth)
        # Option 3: Z=0 is the start depth, positive Z is UP. Pose.z = start_depth - depth_value (Needs start depth)

        # Let's assume Option 1: Z=0 is surface, positive UP
        z_position = -depth_value

        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = self.frame_id # Should be 'odom' or 'map'

        pose_msg.pose.pose.position.x = 0.0 # We don't know X from depth
        pose_msg.pose.pose.position.y = 0.0 # We don't know Y from depth
        pose_msg.pose.pose.position.z = z_position

        # Orientation is unknown from depth sensor
        pose_msg.pose.pose.orientation.x = 0.0
        pose_msg.pose.pose.orientation.y = 0.0
        pose_msg.pose.pose.orientation.z = 0.0
        pose_msg.pose.pose.orientation.w = 1.0

        # Set covariance: High uncertainty for X, Y, Roll, Pitch, Yaw; Low for Z
        covariance = np.diag([
            9999.0, 9999.0, self.depth_variance, # X, Y, Z variance (m^2)
            9999.0, 9999.0, 9999.0             # Roll, Pitch, Yaw variance (rad^2)
        ]).flatten().tolist()
        pose_msg.pose.covariance = covariance

        self.pose_pub.publish(pose_msg)


def main(args=None):
    rclpy.init(args=args)
    node = DepthProcessorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()