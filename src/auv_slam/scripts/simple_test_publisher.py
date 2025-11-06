#!/usr/bin/env python3
"""
Simple Test Publisher - Just publishes constant forward motion
This tests if the basic pipeline works: cmd_vel → mapper → thrusters → Gazebo
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class SimpleTestPublisher(Node):
    def __init__(self):
        super().__init__('simple_test_publisher')
        
        self.cmd_pub = self.create_publisher(Twist, '/rp2040/cmd_vel', 10)
        self.timer = self.create_timer(0.1, self.publish_cmd)
        
        self.get_logger().info('='*70)
        self.get_logger().info('🚀 SIMPLE TEST PUBLISHER STARTED')
        self.get_logger().info('='*70)
        self.get_logger().info('Publishing constant FORWARD command...')
        self.get_logger().info('If robot moves forward, pipeline is working!')
        self.get_logger().info('='*70)
    
    def publish_cmd(self):
        cmd = Twist()
        cmd.linear.x = 0.5  # Forward at 0.5 m/s
        cmd.linear.y = 0.0
        cmd.linear.z = 0.0  # No depth change
        cmd.angular.z = 0.0
        
        self.cmd_pub.publish(cmd)
        
        # Log every 2 seconds
        if int(self.get_clock().now().nanoseconds / 1e9) % 2 == 0:
            self.get_logger().info(
                f'Publishing: vx={cmd.linear.x:.2f}',
                throttle_duration_sec=1.9
            )


def main(args=None):
    rclpy.init(args=args)
    node = SimpleTestPublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop robot
        cmd = Twist()
        node.cmd_pub.publish(cmd)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()