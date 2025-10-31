#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Wrench, Twist
from std_msgs.msg import Float64
import numpy as np


class EnhancedThrusterMapper(Node):
    def __init__(self):
        super().__init__('enhanced_thruster_mapper')
        
        self.declare_parameter('max_thrust', 10.0)
        self.declare_parameter('dead_zone', 0.1)
        self.declare_parameter('use_twist_input', True)
        
        self.max_thrust = self.get_parameter('max_thrust').value
        self.dead_zone = self.get_parameter('dead_zone').value
        self.use_twist = self.get_parameter('use_twist_input').value
        
        # Thruster Allocation Matrix - FIXED SURGE DIRECTION
        self.TAM = np.array([
            [-1.0,  -1.0,   0.0,   0.0,   0.0,  -1.0],  # T1 (FL)
            [-1.0,   1.0,   0.0,   0.0,   0.0,   1.0],  # T2 (FR)
            [ 1.0,  -1.0,   0.0,   0.0,   0.0,   1.0],  # T3 (BL)
            [ 1.0,   1.0,   0.0,   0.0,   0.0,  -1.0],  # T4 (BR)
            [ 0.0,   0.0,   1.0,   0.0,   0.2,   0.0],  # T5 (D1)
            [ 0.0,   0.0,   1.0,   0.0,  -0.2,   0.0],  # T6 (D2)
        ], dtype=np.float64)
        
        self.TAM_pinv = np.linalg.pinv(self.TAM)
        
        if self.use_twist:
            self.twist_sub = self.create_subscription(
                Twist, '/rp2040/cmd_vel', self.twist_callback, 10)
        else:
            self.wrench_sub = self.create_subscription(
                Wrench, '/cmd_wrench', self.wrench_callback, 10)
        
        self.thruster_pubs = []
        for i in range(1, 7):
            pub = self.create_publisher(Float64, f'/thruster{i}_cmd', 10)
            self.thruster_pubs.append(pub)
        
        self.diag_timer = self.create_timer(1.0, self.publish_diagnostics)
        self.last_wrench = np.zeros(6)
        self.last_thrusts = np.zeros(6)
        
        self.get_logger().info('Enhanced Thruster Mapper initialized')
        self.get_logger().info(f'Max thrust: {self.max_thrust} N')
        self.get_logger().info(f'Input mode: {"Twist" if self.use_twist else "Wrench"}')
    
    def twist_callback(self, msg: Twist):
        wrench = np.array([
            msg.linear.x,
            msg.linear.y,
            -msg.linear.z,
            0.0,
            0.0,
            msg.angular.z,
        ])
        
        self.allocate_and_publish(wrench)
    
    def wrench_callback(self, msg: Wrench):
        wrench = np.array([
            msg.force.x,
            msg.force.y,
            msg.force.z,
            msg.torque.x,
            msg.torque.y,
            msg.torque.z,
        ])
        
        self.allocate_and_publish(wrench)
    
    def allocate_and_publish(self, wrench: np.ndarray):
        self.last_wrench = wrench
        
        raw_thrusts = self.TAM_pinv @ wrench
        
        saturated_thrusts = np.clip(raw_thrusts, -self.max_thrust, self.max_thrust)
        
        final_thrusts = np.where(
            np.abs(saturated_thrusts) < self.dead_zone,
            0.0,
            saturated_thrusts
        )
        
        self.last_thrusts = final_thrusts
        
        if not np.allclose(raw_thrusts, saturated_thrusts):
            self.get_logger().warn(
                'Thruster saturation occurred! '
                f'Max raw thrust: {np.max(np.abs(raw_thrusts)):.2f}'
            )
        
        for i, (pub, thrust) in enumerate(zip(self.thruster_pubs, final_thrusts)):
            msg = Float64()
            msg.data = float(thrust)
            pub.publish(msg)
    
    def publish_diagnostics(self):
        if np.any(self.last_thrusts != 0):
            self.get_logger().info(
                f'Wrench: [{", ".join([f"{w:6.2f}" for w in self.last_wrench])}] | '
                f'Thrusts: [{", ".join([f"{t:5.1f}" for t in self.last_thrusts])}]',
                throttle_duration_sec=2.0
            )


def main(args=None):
    rclpy.init(args=args)
    node = EnhancedThrusterMapper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()