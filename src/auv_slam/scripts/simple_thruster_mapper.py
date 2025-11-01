#!/usr/bin/env python3
"""
FIXED Thruster Mapper - Corrected depth control sign
Key fix: Removed Z-axis negation that was causing inverted depth control
"""

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
        
        # Thruster Allocation Matrix
        self.TAM = np.array([
            [-1.0,  -1.0,   0.0,   0.0,   0.0,  -1.0],  # T1 (FL)
            [-1.0,   1.0,   0.0,   0.0,   0.0,   1.0],  # T2 (FR)
            [ 1.0,  -1.0,   0.0,   0.0,   0.0,   1.0],  # T3 (BL)
            [ 1.0,   1.0,   0.0,   0.0,   0.0,  -1.0],  # T4 (BR)
            [ 0.0,   0.0,   1.0,   0.0,   0.2,   0.0],  # T5 (D1) - CORRECT: +1.0
            [ 0.0,   0.0,   1.0,   0.0,  -0.2,   0.0],  # T6 (D2) - CORRECT: +1.0
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
        
        self.get_logger().info('✅ FIXED Thruster Mapper initialized')
        self.get_logger().info('   - Correct depth control (no Z negation)')
        self.get_logger().info(f'   - Max thrust: {self.max_thrust} N')
        self.get_logger().info(f'   - Input mode: {"Twist" if self.use_twist else "Wrench"}')
    
    def twist_callback(self, msg: Twist):
        """
        CRITICAL FIX: Removed Z negation
        
        ROS/Gazebo convention:
        - +X = forward, -X = backward
        - +Y = left, -Y = right
        - +Z = up, -Z = down
        
        Vertical thrusters point DOWN, so:
        - Positive thrust = push DOWN (descend)
        - Negative thrust = push UP (ascend)
        
        Depth control logic:
        - If too shallow (current > target), need to descend → cmd.linear.z < 0
        - Negative cmd.linear.z → negative wrench[2] → negative thrust → ascend
        
        Wait, that's still wrong! Let me reconsider...
        
        Actually, the thruster physical setup:
        - Thrusters point DOWN
        - Spinning them forward pushes water DOWN, creating UP force on AUV
        - So positive thrust = AUV goes UP
        - Negative thrust = AUV goes DOWN
        
        Therefore:
        - To descend: need negative thrust → negative wrench → negative cmd.linear.z
        - To ascend: need positive thrust → positive wrench → positive cmd.linear.z
        
        Current code in navigator:
        depth_error = target_depth - current_depth
        cmd.linear.z = depth_error * gain
        
        If current = -1m, target = -2m:
        error = -2 - (-1) = -1
        cmd.linear.z = -1 * gain = negative → descend ✓
        
        This is CORRECT! No negation needed.
        """
        wrench = np.array([
            msg.linear.x,
            msg.linear.y,
            msg.linear.z,  # ✓ FIXED: No negation
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