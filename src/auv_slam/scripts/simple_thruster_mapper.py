#!/usr/bin/env python3
"""
Simple Thruster Mapper for BlueROV2 Configuration
Converts Twist commands to individual thruster commands
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64
import numpy as np


class SimpleThrusterMapper(Node):
    def __init__(self):
        super().__init__('simple_thruster_mapper')
        
        # Parameters
        self.declare_parameter('max_thrust', 150.0)
        self.declare_parameter('thrust_scale', 15.0)
        self.declare_parameter('vertical_thrust_boost', 2.0)
        
        self.max_thrust = self.get_parameter('max_thrust').value
        self.thrust_scale = self.get_parameter('thrust_scale').value
        self.vertical_boost = self.get_parameter('vertical_thrust_boost').value
        
        # Subscriber
        self.cmd_vel_sub = self.create_subscription(
            Twist, '/rp2040/cmd_vel', self.cmd_vel_callback, 10)
        
        # Publishers for 6 thrusters
        self.thruster_pubs = []
        for i in range(1, 7):
            pub = self.create_publisher(Float64, f'/model/orca4_ign/joint/thruster{i}_joint/cmd_pos',10)
            self.thruster_pubs.append(pub)
        
        self.get_logger().info('Simple Thruster Mapper initialized')
        self.get_logger().info(f'Max thrust: {self.max_thrust}, Scale: {self.thrust_scale}')
    
    def cmd_vel_callback(self, msg: Twist):
        """
        Map twist commands to thruster forces
        BlueROV2 Configuration:
        - T1: Front-Left (FL)
        - T2: Front-Right (FR)  
        - T3: Back-Left (BL)
        - T4: Back-Right (BR)
        - T5: Vertical-Front (D1)
        - T6: Vertical-Back (D2)
        """
        
        # Extract commands
        surge = msg.linear.x * self.thrust_scale  # Forward/backward
        sway = msg.linear.y * self.thrust_scale   # Left/right
        heave = msg.linear.z * self.thrust_scale * self.vertical_boost  # Up/down
        yaw = msg.angular.z * self.thrust_scale   # Rotation
        
        # CRITICAL FIX: Correct thruster allocation
        # Horizontal thrusters (vectored configuration)
        # Positive surge = forward, Positive sway = left, Positive yaw = CCW
        
        # Fixed mapping for proper motion:
        t1 = surge - sway - yaw  # Front-Left
        t2 = surge + sway + yaw  # Front-Right
        t3 = -surge - sway + yaw  # Back-Left (reversed)
        t4 = -surge + sway - yaw  # Back-Right (reversed)
        
        # Vertical thrusters
        # IMPORTANT: In standard AUV convention, negative Z = DOWN
        # So negative heave command should produce downward thrust
        t5 = -heave  # Front vertical (inverted for correct direction)
        t6 = -heave  # Back vertical (inverted for correct direction)
        
        # Apply saturation
        thrusts = [t1, t2, t3, t4, t5, t6]
        thrusts = [np.clip(t, -self.max_thrust, self.max_thrust) for t in thrusts]
        
        # Publish to thrusters
        for i, (pub, thrust) in enumerate(zip(self.thruster_pubs, thrusts)):
            msg = Float64()
            msg.data = float(thrust)
            pub.publish(msg)
        
        # Debug logging
        if any(abs(t) > 0.1 for t in thrusts):
            self.get_logger().info(
                f'Cmd: [{surge:.1f}, {sway:.1f}, {heave:.1f}, {yaw:.1f}] -> '
                f'T: [{thrusts[0]:.1f}, {thrusts[1]:.1f}, {thrusts[2]:.1f}, '
                f'{thrusts[3]:.1f}, {thrusts[4]:.1f}, {thrusts[5]:.1f}]',
                throttle_duration_sec=0.5
            )


def main(args=None):
    rclpy.init(args=args)
    node = SimpleThrusterMapper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()