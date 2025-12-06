#!/usr/bin/env python3
"""
FIXED Thruster Mapper (Transpose TAM)
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
        
        # Thruster Allocation Matrix (Rows = Thrusters, Cols = DOF)
        self.TAM = np.array([
            [-0.707, -0.707,  0.0,   0.0,   0.0,  -1.0],  # T1
            [-0.707,  0.707,  0.0,   0.0,   0.0,   1.0],  # T2
            [ 0.707, -0.707,  0.0,   0.0,   0.0,   1.0],  # T3
            [ 0.707,  0.707,  0.0,   0.0,   0.0,  -1.0],  # T4
            [ 0.0,    0.0,    1.0,   0.0,   0.0,   0.0],  # T5
            [ 0.0,    0.0,    1.0,   0.0,   0.0,   0.0],  # T6
        ], dtype=np.float64)
        
        # [CRITICAL FIX] Transpose before inversion
        self.TAM_pinv = np.linalg.pinv(self.TAM.T)
        
        if self.use_twist:
            self.twist_sub = self.create_subscription(Twist, '/rp2040/cmd_vel', self.twist_callback, 10)
        else:
            self.wrench_sub = self.create_subscription(Wrench, '/cmd_wrench', self.wrench_callback, 10)
        
        self.thruster_pubs = []
        for i in range(1, 7):
            pub = self.create_publisher(Float64, f'/thruster{i}_cmd', 10)
            self.thruster_pubs.append(pub)

    def twist_callback(self, msg: Twist):
        wrench = np.array([msg.linear.x, msg.linear.y, msg.linear.z, 0.0, 0.0, msg.angular.z])
        self.allocate_and_publish(wrench)
    
    def wrench_callback(self, msg: Wrench):
        wrench = np.array([msg.force.x, msg.force.y, msg.force.z, msg.torque.x, msg.torque.y, msg.torque.z])
        self.allocate_and_publish(wrench)
    
    def allocate_and_publish(self, wrench: np.ndarray):
        raw_thrusts = self.TAM_pinv @ wrench
        saturated_thrusts = np.clip(raw_thrusts, -self.max_thrust, self.max_thrust)
        final_thrusts = np.where(np.abs(saturated_thrusts) < self.dead_zone, 0.0, saturated_thrusts)
        
        for pub, thrust in zip(self.thruster_pubs, final_thrusts):
            msg = Float64()
            msg.data = float(thrust)
            pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = EnhancedThrusterMapper()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()