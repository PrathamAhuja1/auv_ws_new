#!/usr/bin/env python3
"""
Fixed Diagnostic Node - Uses Float64 only
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float64
import math

class DiagnosticNode(Node):
    def __init__(self):
        super().__init__('diagnostic_node')
        
        # State variables
        self.odom_received = False
        self.gate_detected = False
        self.cmd_vel_received = False
        self.thruster_cmds_received = [False] * 6
        self.last_thruster_values = [0.0] * 6
        
        self.last_odom = None
        self.last_cmd_vel = None
        
        # Subscriptions
        self.odom_sub = self.create_subscription(
            Odometry, '/ground_truth/odom', self.odom_callback, 10)
        
        self.cmd_vel_sub = self.create_subscription(
            Twist, '/rp2040/cmd_vel', self.cmd_vel_callback, 10)
        
        # Monitor thruster commands - FIXED: Use Float64
        for i in range(1, 7):
            self.create_subscription(
                Float64, f'/thruster{i}_cmd', 
                lambda msg, idx=i-1: self.thruster_callback(msg, idx), 10)
        
        # Status timer
        self.timer = self.create_timer(2.0, self.print_status)
        
        self.get_logger().info('Fixed Diagnostic Node started')
    
    def odom_callback(self, msg: Odometry):
        self.odom_received = True
        self.last_odom = msg
    
    def cmd_vel_callback(self, msg: Twist):
        self.cmd_vel_received = True
        self.last_cmd_vel = msg
    
    def thruster_callback(self, msg: Float64, idx: int):
        self.thruster_cmds_received[idx] = True
        self.last_thruster_values[idx] = msg.data
    
    def print_status(self):
        self.get_logger().info('='*60)
        self.get_logger().info('🔍 AUV DIAGNOSTIC STATUS')
        self.get_logger().info('='*60)
        
        # Odometry status
        if self.odom_received and self.last_odom:
            pos = self.last_odom.pose.pose.position
            self.get_logger().info(f'✓ Odom: X={pos.x:.2f}, Y={pos.y:.2f}, Z={pos.z:.2f}')
        else:
            self.get_logger().warn('✗ No odometry')
        
        # Command velocity
        if self.cmd_vel_received and self.last_cmd_vel:
            cv = self.last_cmd_vel
            self.get_logger().info(
                f'✓ Cmd: vx={cv.linear.x:.2f}, vy={cv.linear.y:.2f}, '
                f'vz={cv.linear.z:.2f}, yaw={cv.angular.z:.2f}'
            )
        else:
            self.get_logger().warn('✗ No cmd_vel')
        
        # Thruster commands - SHOW VALUES
        active = sum(self.thruster_cmds_received)
        if active > 0:
            self.get_logger().info(f'✓ Thrusters: {active}/6 active')
            self.get_logger().info(
                f'  T1={self.last_thruster_values[0]:.1f}, '
                f'T2={self.last_thruster_values[1]:.1f}, '
                f'T3={self.last_thruster_values[2]:.1f}, '
                f'T4={self.last_thruster_values[3]:.1f}, '
                f'T5={self.last_thruster_values[4]:.1f}, '
                f'T6={self.last_thruster_values[5]:.1f}'
            )
        else:
            self.get_logger().warn('✗ No thruster commands')
        
        self.get_logger().info('='*60)


def main(args=None):
    rclpy.init(args=args)
    node = DiagnosticNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()