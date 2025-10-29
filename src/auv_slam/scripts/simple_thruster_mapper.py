#!/usr/bin/env python3
"""
CORRECTED Thruster Mapper for BlueROV2 Vectored Configuration
Uses proper vector mathematics for 45° angled thrusters
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64
import numpy as np
import math


class CorrectedThrusterMapper(Node):
    def __init__(self):
        super().__init__('corrected_thruster_mapper')
        
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
            pub = self.create_publisher(Float64, f'/thruster{i}_cmd', 10)
            self.thruster_pubs.append(pub)
        
        self.get_logger().info('✅ CORRECTED Thruster Mapper initialized')
        self.get_logger().info(f'Max thrust: {self.max_thrust}, Scale: {self.thrust_scale}')
        self.get_logger().info('Using vectored thruster geometry (45° angles)')
    
    def cmd_vel_callback(self, msg: Twist):
        """
        Vectored Thruster Configuration (45° angles):
        
        Top View:
             FRONT
               ↑
        T1 ↗     ↖ T2
           \  •  /
            \ | /
        T3 ←  •  → T4
            / | \
           /  •  \
        T3 ↙     ↘ T4
             BACK
        
        Each horizontal thruster at 45° provides:
        - Component along X-axis (surge)
        - Component along Y-axis (sway)
        
        T1: Front-Left  (-45° from X-axis)
        T2: Front-Right (+45° from X-axis)
        T3: Back-Left   (-135° from X-axis)
        T4: Back-Right  (+135° from X-axis)
        """
        
        # Extract commands (body frame)
        surge = msg.linear.x * self.thrust_scale   # +X = forward
        sway = msg.linear.y * self.thrust_scale    # +Y = left
        heave = msg.linear.z * self.thrust_scale * self.vertical_boost
        yaw = msg.angular.z * self.thrust_scale    # +Z = CCW
        
        # ==================== VECTORED THRUSTER MATH ====================
        # For 45° angled thrusters, we need to decompose forces
        # Each thruster contributes: thrust * cos(45°) to each axis
        
        # Simplification: cos(45°) = sin(45°) = 0.707
        # But we can work with the full values and normalize
        
        # FORWARD (+surge): All 4 thrusters spin forward
        # LEFT (+sway): T1,T3 forward, T2,T4 backward
        # CCW (+yaw): T1,T4 forward, T2,T3 backward
        
        # BlueROV2 Standard Allocation:
        # T1 (Front-Left):  +surge +sway +yaw
        # T2 (Front-Right): +surge -sway -yaw
        # T3 (Back-Left):   +surge +sway -yaw
        # T4 (Back-Right):  +surge -sway +yaw
        
        t1 = surge + sway + yaw   # Front-Left
        t2 = surge - sway - yaw   # Front-Right
        t3 = surge + sway - yaw   # Back-Left
        t4 = surge - sway + yaw   # Back-Right
        
        # Vertical thrusters (T5, T6)
        # Both point downward, so positive command = downward thrust
        # BUT: In our convention, negative Z command = submerge
        # So we need to INVERT the sign
        t5 = -heave  # Front vertical
        t6 = -heave  # Back vertical
        
        # Apply saturation
        thrusts = [t1, t2, t3, t4, t5, t6]
        thrusts = [np.clip(t, -self.max_thrust, self.max_thrust) for t in thrusts]
        
        # Publish to thrusters
        for i, (pub, thrust) in enumerate(zip(self.thruster_pubs, thrusts)):
            cmd_msg = Float64()
            cmd_msg.data = float(thrust)
            pub.publish(cmd_msg)
        
        # Debug logging
        if any(abs(t) > 0.1 for t in thrusts):
            self.get_logger().info(
                f'🎮 Input: surge={surge:.1f}, sway={sway:.1f}, heave={heave:.1f}, yaw={yaw:.1f}\n'
                f'   Output: [T1={thrusts[0]:.1f}, T2={thrusts[1]:.1f}, '
                f'T3={thrusts[2]:.1f}, T4={thrusts[3]:.1f}, '
                f'T5={thrusts[4]:.1f}, T6={thrusts[5]:.1f}]',
                throttle_duration_sec=1.0
            )


def main(args=None):
    rclpy.init(args=args)
    node = CorrectedThrusterMapper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop all thrusters on shutdown
        stop_cmd = Twist()
        node.cmd_vel_callback(stop_cmd)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()