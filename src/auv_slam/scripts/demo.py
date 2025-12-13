#!/usr/bin/env python3
"""
Simple 4-DOF Motion Demonstration
Demonstrates: HEAVE, SURGE, YAW, ROLL

Usage:
    Terminal 1: ros2 launch auv_slam motion_demo_standalone.launch.py
    Terminal 2: ros2 run auv_slam motion_demo.py
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import time
import math
from tf_transformations import euler_from_quaternion


class MotionDemo(Node):
    def __init__(self):
        super().__init__('motion_demo_node')
        
        # Publisher for velocity commands
        self.cmd_vel_pub = self.create_publisher(Twist, '/rp2040/cmd_vel', 10)
        
        # Subscriber for odometry feedback
        self.odom_sub = self.create_subscription(
            Odometry, '/ground_truth/odom', self.odom_callback, 10)
        
        # Current state
        self.current_position = None
        self.current_depth = 0.0
        self.current_roll = 0.0
        self.current_pitch = 0.0
        self.current_yaw = 0.0
        
        # Demo parameters
        self.motion_duration = 4.0  # seconds per motion
        self.rest_duration = 2.0    # seconds between motions
        
        self.get_logger().info('='*70)
        self.get_logger().info('🚀 AUV 4-DOF MOTION DEMONSTRATION')
        self.get_logger().info('='*70)
        self.get_logger().info('Demonstrating 4 degrees of freedom:')
        self.get_logger().info('  1. HEAVE  - Up/Down motion')
        self.get_logger().info('  2. SURGE  - Forward/Backward motion')
        self.get_logger().info('  3. YAW    - Turn Left/Right')
        self.get_logger().info('  4. ROLL   - Tilt Left/Right')
        self.get_logger().info('='*70)
        
        # Wait for odometry
        self.get_logger().info('⏳ Waiting for odometry data...')
        while self.current_position is None:
            rclpy.spin_once(self, timeout_sec=0.1)
        
        self.get_logger().info('✅ Ready! Starting in 3 seconds...')
        time.sleep(3)
    
    def odom_callback(self, msg: Odometry):
        """Update current state from odometry"""
        self.current_position = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        )
        self.current_depth = msg.pose.pose.position.z
        
        # Convert quaternion to euler angles
        q = msg.pose.pose.orientation
        self.current_roll, self.current_pitch, self.current_yaw = euler_from_quaternion(
            [q.x, q.y, q.z, q.w]
        )
    
    def publish_velocity(self, linear_x=0.0, linear_y=0.0, linear_z=0.0, 
                        angular_x=0.0, angular_y=0.0, angular_z=0.0):
        """Publish velocity command"""
        cmd = Twist()
        cmd.linear.x = linear_x
        cmd.linear.y = linear_y
        cmd.linear.z = linear_z
        cmd.angular.x = angular_x
        cmd.angular.y = angular_y
        cmd.angular.z = angular_z
        self.cmd_vel_pub.publish(cmd)
    
    def stop(self):
        """Stop all motion"""
        self.publish_velocity(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    
    def execute_motion(self, name: str, description: str, 
                      linear_x=0.0, linear_y=0.0, linear_z=0.0,
                      angular_x=0.0, angular_y=0.0, angular_z=0.0):
        """Execute a motion for specified duration with feedback"""
        
        self.get_logger().info('')
        self.get_logger().info('='*70)
        self.get_logger().info(f'🎯 {name}')
        self.get_logger().info(f'   {description}')
        self.get_logger().info('='*70)
        
        # Record starting values
        start_pos = self.current_position
        start_depth = self.current_depth
        start_yaw = self.current_yaw
        start_roll = self.current_roll
        
        # Execute motion
        start_time = time.time()
        last_log_time = start_time
        
        while (time.time() - start_time) < self.motion_duration:
            self.publish_velocity(linear_x, linear_y, linear_z, 
                                angular_x, angular_y, angular_z)
            
            # Log progress every 0.5 seconds
            current_time = time.time()
            if (current_time - last_log_time) >= 0.5:
                elapsed = current_time - start_time
                dx = self.current_position[0] - start_pos[0]
                dy = self.current_position[1] - start_pos[1]
                dz = self.current_depth - start_depth
                dyaw = math.degrees(self.current_yaw - start_yaw)
                droll = math.degrees(self.current_roll - start_roll)
                
                self.get_logger().info(
                    f'   {elapsed:.1f}s | '
                    f'ΔX={dx:+.2f}m, ΔY={dy:+.2f}m, ΔZ={dz:+.2f}m | '
                    f'Yaw={dyaw:+.1f}°, Roll={droll:+.1f}°'
                )
                last_log_time = current_time
            
            rclpy.spin_once(self, timeout_sec=0.01)
        
        # Stop motion
        self.stop()
        time.sleep(0.5)
        
        # Final report
        final_dx = self.current_position[0] - start_pos[0]
        final_dy = self.current_position[1] - start_pos[1]
        final_dz = self.current_depth - start_depth
        final_dyaw = math.degrees(self.current_yaw - start_yaw)
        final_droll = math.degrees(self.current_roll - start_roll)
        
        self.get_logger().info('✅ Motion complete!')
        self.get_logger().info(
            f'   Total: X={final_dx:+.2f}m, Y={final_dy:+.2f}m, '
            f'Z={final_dz:+.2f}m, Yaw={final_dyaw:+.1f}°, Roll={final_droll:+.1f}°'
        )
        self.get_logger().info(f'   Position: {self.current_position}')
        self.get_logger().info(f'   Depth: {self.current_depth:.2f}m')
        
        # Rest period
        self.get_logger().info(f'⏸️  Rest {self.rest_duration}s...')
        time.sleep(self.rest_duration)
    
    def run_demonstration(self):
        """Run 4-DOF demonstration"""
        
        try:
            # ============================================
            # 1. HEAVE - Upward motion
            # ============================================
            self.execute_motion(
                name="HEAVE UP",
                description="Vertical motion upward (negative Z)",
                linear_z=-0.4  # Negative Z = upward
            )
            
            # ============================================
            # 2. HEAVE - Downward motion
            # ============================================
            self.execute_motion(
                name="HEAVE DOWN",
                description="Vertical motion downward (positive Z)",
                linear_z=0.4  # Positive Z = downward
            )
            
            # ============================================
            # 3. SURGE - Forward motion
            # ============================================
            self.execute_motion(
                name="SURGE FORWARD",
                description="Linear motion forward (positive X)",
                linear_x=0.6
            )
            
            # ============================================
            # 4. SURGE - Backward motion
            # ============================================
            self.execute_motion(
                name="SURGE BACKWARD",
                description="Linear motion backward (negative X)",
                linear_x=-0.6
            )
            
            # ============================================
            # 5. YAW - Turn Right
            # ============================================
            self.execute_motion(
                name="YAW RIGHT",
                description="Rotation clockwise (negative angular Z)",
                angular_z=-0.5
            )
            
            # ============================================
            # 6. YAW - Turn Left
            # ============================================
            self.execute_motion(
                name="YAW LEFT",
                description="Rotation counter-clockwise (positive angular Z)",
                angular_z=0.5
            )
            
            # ============================================
            # 7. ROLL - Tilt Right (Limited Capability)
            # ============================================
            self.get_logger().info('')
            self.get_logger().info('='*70)
            self.get_logger().info('⚠️  ROLL DEMONSTRATION')
            self.get_logger().info('   Note: BlueROV configuration has limited roll control')
            self.get_logger().info('   Attempting roll using differential thruster commands')
            self.get_logger().info('='*70)
            
            self.execute_motion(
                name="ROLL RIGHT ATTEMPT",
                description="Differential thrust to induce roll",
                angular_x=0.3  # Roll command (limited effect)
            )
            
            # ============================================
            # 8. ROLL - Tilt Left (Limited Capability)
            # ============================================
            self.execute_motion(
                name="ROLL LEFT ATTEMPT",
                description="Differential thrust to induce roll",
                angular_x=-0.3  # Roll command (limited effect)
            )
            
            # Final stop
            self.stop()
            
            # ============================================
            # SUMMARY
            # ============================================
            self.get_logger().info('')
            self.get_logger().info('='*70)
            self.get_logger().info('🎉 DEMONSTRATION COMPLETE!')
            self.get_logger().info('='*70)
            self.get_logger().info('Motion Summary:')
            self.get_logger().info('  ✅ HEAVE   - Full up/down control')
            self.get_logger().info('  ✅ SURGE   - Full forward/backward control')
            self.get_logger().info('  ✅ YAW     - Full rotational control')
            self.get_logger().info('  ⚠️  ROLL    - Limited (symmetric thruster design)')
            self.get_logger().info('')
            self.get_logger().info(f'Final Position: {self.current_position}')
            self.get_logger().info(f'Final Depth: {self.current_depth:.2f}m')
            self.get_logger().info(f'Final Yaw: {math.degrees(self.current_yaw):.1f}°')
            self.get_logger().info(f'Final Roll: {math.degrees(self.current_roll):.1f}°')
            self.get_logger().info('='*70)
            
        except KeyboardInterrupt:
            self.get_logger().info('⚠️  Interrupted by user')
            self.stop()
        except Exception as e:
            self.get_logger().error(f'❌ Error: {e}')
            self.stop()


def main(args=None):
    rclpy.init(args=args)
    
    demo_node = MotionDemo()
    
    try:
        demo_node.run_demonstration()
    except KeyboardInterrupt:
        pass
    finally:
        demo_node.stop()
        demo_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()