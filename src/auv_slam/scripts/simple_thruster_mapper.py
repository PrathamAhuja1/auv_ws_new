#!/usr/bin/env python3
"""
FIXED Simple Thruster Mapper for BlueROV2 Configuration
Converts Twist commands to individual thruster commands
KEY FIX: Correct sign convention for vectored thrusters
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
        # CRITICAL: Must publish to /thrusterN_cmd (bridged topics), not direct Gazebo topics
        self.thruster_pubs = []
        for i in range(1, 7):
            pub = self.create_publisher(
                Float64, 
                f'/thruster{i}_cmd',
                10
            )
            self.thruster_pubs.append(pub)
        
        self.get_logger().info('✅ FIXED Thruster Mapper initialized')
        self.get_logger().info(f'Max thrust: {self.max_thrust}, Scale: {self.thrust_scale}')
    
    def cmd_vel_callback(self, msg: Twist):
        """
        Map twist commands to thruster forces
        BlueROV2 Configuration (Vectored):
        
        Front View:        Top View:
           T5  T6          T1 --- T2
            |  |            \   /
         T1-BODY-T2          \ /
            |  |            BODY
         T3-    -T4          / \
                            /   \
                          T3 --- T4
        
        KEY: Front thrusters (T1, T2) push FORWARD when positive
             Back thrusters (T3, T4) push FORWARD when NEGATIVE
             This is because they're facing opposite directions!
        """
        
        # Extract commands (in body frame)
        surge = msg.linear.x * self.thrust_scale   # +X = forward
        sway = msg.linear.y * self.thrust_scale    # +Y = left
        heave = msg.linear.z * self.thrust_scale * self.vertical_boost  # +Z = up
        yaw = msg.angular.z * self.thrust_scale    # +Z = CCW rotation
        
        # ==================== CRITICAL FIX ====================
        # Vectored thruster allocation (45° angled thrusters)
        # Each thruster contributes to both surge and sway
        
        # FORWARD MOTION: T1 & T2 positive, T3 & T4 NEGATIVE
        # LEFT MOTION: T1 & T3 negative, T2 & T4 positive  
        # CCW ROTATION: T1 & T3 negative, T2 & T4 positive
        
        t1 = surge - sway - yaw   # Front-Left: + for forward
        t2 = surge + sway + yaw   # Front-Right: + for forward
        t3 = -surge - sway + yaw  # Back-Left: - for forward (REVERSED)
        t4 = -surge + sway - yaw  # Back-Right: - for forward (REVERSED)
        
        # Vertical thrusters (pointing down)
        # IMPORTANT: In simulation, negative Z command = downward motion
        # So we INVERT the sign to match AUV convention
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
        
        # Debug logging (only when moving)
        if any(abs(t) > 0.1 for t in thrusts):
            self.get_logger().info(
                f'🎮 Cmd: surge={surge:.1f}, sway={sway:.1f}, heave={heave:.1f}, yaw={yaw:.1f}\n'
                f'   T: [T1={thrusts[0]:.1f}, T2={thrusts[1]:.1f}, T3={thrusts[2]:.1f}, '
                f'T4={thrusts[3]:.1f}, T5={thrusts[4]:.1f}, T6={thrusts[5]:.1f}]',
                throttle_duration_sec=1.0
            )


def main(args=None):
    rclpy.init(args=args)
    node = SimpleThrusterMapper()
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