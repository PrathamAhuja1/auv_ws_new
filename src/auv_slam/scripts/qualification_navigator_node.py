#!/usr/bin/env python3
"""
FINAL QUALIFICATION NAVIGATOR - Proper Alignment Strategy
Rules:
1. Submerge to -0.8m at start, maintain until within 3m of gate
2. ONLY yaw corrections until within 3m (NO lateral movement)
3. Within 3m: Align horizontally AND adjust depth to gate center
4. Pass → U-turn (submerged) → Pass → SURFACE
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import time
import math

class FinalQualificationNavigator(Node):
    def __init__(self):
        super().__init__('qualification_navigator')
        
        # States
        self.SUBMERGING = 0
        self.CRUISING = 1           # Until 3m: Just yaw, no lateral
        self.FINAL_ALIGNMENT = 2    # 3m-1m: Horizontal + vertical alignment
        self.PASSING = 3
        self.CLEARING = 4           # Ensure full clearance
        self.UTURN = 5
        self.REVERSE_CRUISING = 6
        self.REVERSE_ALIGNMENT = 7
        self.REVERSE_PASSING = 8
        self.REVERSE_CLEARING = 9
        self.SURFACING = 10
        self.COMPLETED = 11
        
        self.state = self.SUBMERGING
        
        # CRITICAL PARAMETERS
        self.cruise_depth = -0.8        # Maintain until close to gate
        self.gate_center_depth = -0.7   # Gate center is at -0.7m (1.4m height, bottom at -1.4m, top at 0m)
        self.gate_x_position = 0.0
        
        self.alignment_distance = 3.0   # Start aligning at 3m
        self.passing_distance = 1.0     # Start passing at 1m
        self.clearance_distance = 1.0   # Must travel 1m past gate
        
        # Speeds
        self.cruise_speed = 0.8         # Fast cruise until 3m
        self.alignment_speed = 0.4      # Slower for precision alignment
        self.passing_speed = 1.0        # Maximum through gate
        
        # State tracking
        self.gate_detected = False
        self.alignment_error = 0.0
        self.estimated_distance = 999.0
        self.current_depth = 0.0
        self.current_position = None
        self.current_yaw = 0.0
        
        self.passing_start_x = None
        self.uturn_start_yaw = 0.0
        self.state_start_time = time.time()
        self.mission_start_time = time.time()
        
        # Subscriptions
        self.create_subscription(Bool, '/qualification/gate_detected', self.gate_cb, 10)
        self.create_subscription(Float32, '/qualification/alignment_error', self.align_cb, 10)
        self.create_subscription(Float32, '/qualification/estimated_distance', self.dist_cb, 10)
        self.create_subscription(Odometry, '/ground_truth/odom', self.odom_cb, 10)
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/rp2040/cmd_vel', 10)
        self.state_pub = self.create_publisher(String, '/qualification/state', 10)
        self.reverse_mode_pub = self.create_publisher(Bool, '/mission/reverse_mode', 10)
        
        self.create_timer(0.05, self.control_loop)
        
        self.get_logger().info('='*70)
        self.get_logger().info('✅ FINAL QUALIFICATION NAVIGATOR')
        self.get_logger().info('='*70)
        self.get_logger().info(f'   Cruise depth: {self.cruise_depth}m (until 3m from gate)')
        self.get_logger().info(f'   Gate center depth: {self.gate_center_depth}m')
        self.get_logger().info(f'   Strategy: Cruise → Align@3m → Pass → Uturn → Align → Pass → Surface')
        self.get_logger().info('='*70)

    def gate_cb(self, msg):
        self.gate_detected = msg.data
    
    def align_cb(self, msg):
        self.alignment_error = msg.data
    
    def dist_cb(self, msg):
        self.estimated_distance = msg.data
    
    def odom_cb(self, msg):
        self.current_depth = msg.pose.pose.position.z
        self.current_position = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        )
        
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

    def control_loop(self):
        cmd = Twist()
        
        # Depth control based on state
        if self.state in [self.SUBMERGING, self.CRUISING, self.REVERSE_CRUISING, self.UTURN]:
            # Maintain cruise depth (-0.8m)
            cmd.linear.z = self.depth_control(self.cruise_depth)
            
        elif self.state in [self.FINAL_ALIGNMENT, self.REVERSE_ALIGNMENT]:
            # Align to gate center depth (-0.7m)
            cmd.linear.z = self.depth_control(self.gate_center_depth)
            
        elif self.state in [self.PASSING, self.CLEARING, self.REVERSE_PASSING, self.REVERSE_CLEARING]:
            # Maintain gate center depth during pass
            cmd.linear.z = self.depth_control(self.gate_center_depth)
            
        elif self.state == self.SURFACING:
            # Surface slowly
            cmd.linear.z = -0.5  # Upward
            
        else:
            cmd.linear.z = 0.0
        
        # State behaviors
        if self.state == self.SUBMERGING:
            cmd = self.submerge(cmd)
        elif self.state == self.CRUISING:
            cmd = self.cruise(cmd)
        elif self.state == self.FINAL_ALIGNMENT:
            cmd = self.final_align(cmd)
        elif self.state == self.PASSING:
            cmd = self.passing(cmd, "FORWARD")
        elif self.state == self.CLEARING:
            cmd = self.clearing(cmd, "FORWARD")
        elif self.state == self.UTURN:
            cmd = self.uturn(cmd)
        elif self.state == self.REVERSE_CRUISING:
            cmd = self.reverse_cruise(cmd)
        elif self.state == self.REVERSE_ALIGNMENT:
            cmd = self.reverse_align(cmd)
        elif self.state == self.REVERSE_PASSING:
            cmd = self.passing(cmd, "REVERSE")
        elif self.state == self.REVERSE_CLEARING:
            cmd = self.clearing(cmd, "REVERSE")
        elif self.state == self.SURFACING:
            cmd = self.surfacing(cmd)
        elif self.state == self.COMPLETED:
            cmd = self.completed(cmd)
        
        self.cmd_vel_pub.publish(cmd)
        self.state_pub.publish(String(data=self.get_state_name()))
        
        # Logging
        if self.current_position and int(time.time() * 2) % 4 == 0:
            self.get_logger().info(
                f'[{self.get_state_name()}] X={self.current_position[0]:.2f}, '
                f'Z={self.current_depth:.2f}, Dist={self.estimated_distance:.2f}m',
                throttle_duration_sec=1.9
            )

    def depth_control(self, target_depth):
        """
        SLOW, SMOOTH depth control to prevent overshooting
        Wide deadband for stability
        """
        depth_error = target_depth - self.current_depth
        DEADBAND = 0.25  # Tighter deadband for better precision
        
        if abs(depth_error) < DEADBAND:
            # Within acceptable range - stop corrections
            return 0.0
        elif abs(depth_error) < 0.5:
            # Small error - very gentle correction
            z_cmd = depth_error * 0.4
            return max(-0.25, min(z_cmd, 0.25))
        elif abs(depth_error) < 1.0:
            # Moderate error - slow correction
            z_cmd = depth_error * 0.6
            return max(-0.4, min(z_cmd, 0.4))
        else:
            # Large error - still controlled
            z_cmd = depth_error * 0.8
            return max(-0.6, min(z_cmd, 0.6))

    def submerge(self, cmd):
        """Wait for stable depth at -0.8m"""
        if abs(self.cruise_depth - self.current_depth) < 0.4:
            elapsed = time.time() - self.state_start_time
            if elapsed > 3.0:
                self.get_logger().info('✅ Submerged at -0.8m - Starting cruise')
                self.reverse_mode_pub.publish(Bool(data=False))
                self.transition_to(self.CRUISING)
        return cmd

    def cruise(self, cmd):
        """
        CRUISE PHASE: Fast forward, ONLY yaw corrections
        ABSOLUTELY NO lateral movement - pure forward motion
        """
        if not self.gate_detected:
            # Search pattern - still no lateral movement
            cmd.linear.x = 0.3
            cmd.linear.y = 0.0  # LOCKED
            cmd.angular.z = 0.4 if (time.time() % 8 < 4) else -0.4
            return cmd
        
        # Check if close enough for final alignment
        if self.estimated_distance <= self.alignment_distance:
            self.get_logger().info(
                f'🎯 Within 3m ({self.estimated_distance:.2f}m) - Starting alignment'
            )
            self.transition_to(self.FINAL_ALIGNMENT)
            return cmd
        
        # FAST CRUISE - Only yaw to keep gate centered
        # NO LATERAL MOVEMENT AT ALL
        cmd.linear.x = self.cruise_speed
        cmd.linear.y = 0.0  # ABSOLUTELY LOCKED - NO SIDEWAYS DRIFT
        cmd.angular.z = -self.alignment_error * 1.0  # Smooth yaw only
        
        return cmd

    def final_align(self, cmd):
        """
        FINAL ALIGNMENT: 3m-1m range
        - Horizontal alignment (yaw ONLY - no lateral drift)
        - Vertical alignment (slow depth adjustment)
        - Reduced forward speed during depth change
        """
        if not self.gate_detected:
            # Lost gate
            cmd.linear.x = 0.2
            cmd.linear.y = 0.0  # LOCKED
            cmd.angular.z = 0.3
            return cmd
        
        # Check if close enough to commit
        if self.estimated_distance <= self.passing_distance:
            if abs(self.alignment_error) < 0.15:
                self.get_logger().info(
                    f'✅ Aligned at {self.estimated_distance:.2f}m - COMMITTING TO PASS'
                )
                self.passing_start_x = self.current_position[0] if self.current_position else 0
                self.transition_to(self.PASSING)
                return cmd
            else:
                self.get_logger().warn(
                    f'⚠️ At 1m but misaligned ({self.alignment_error:+.2f}) - correcting'
                )
        
        # Check if depth is stable at target
        depth_error = abs(self.gate_center_depth - self.current_depth)
        
        if depth_error > 0.3:
            # Still adjusting depth - SLOW DOWN forward motion
            cmd.linear.x = 0.2  # Very slow while changing depth
            self.get_logger().info(
                f'⚠️ Adjusting depth: target={self.gate_center_depth:.2f}, '
                f'current={self.current_depth:.2f}, error={depth_error:.2f}',
                throttle_duration_sec=1.0
            )
        else:
            # Depth stable - can move faster
            cmd.linear.x = self.alignment_speed
        
        # NEVER use lateral movement
        cmd.linear.y = 0.0  # ABSOLUTELY LOCKED
        
        # Smooth yaw correction only
        cmd.angular.z = -self.alignment_error * 1.5
        
        return cmd

    def passing(self, cmd, direction):
        """Full speed straight through gate - NO lateral, stable depth"""
        if self.passing_start_x is None and self.current_position:
            self.passing_start_x = self.current_position[0]
        
        # FULL SPEED STRAIGHT - ZERO corrections
        cmd.linear.x = self.passing_speed
        cmd.linear.y = 0.0  # LOCKED - pure forward thrust
        cmd.angular.z = 0.0  # LOCKED - no turning during pass
        
        # Check if we've traveled enough to clear gate
        if self.current_position and self.passing_start_x is not None:
            distance_traveled = abs(self.current_position[0] - self.passing_start_x)
            
            if distance_traveled >= self.clearance_distance:
                self.get_logger().info(
                    f'✅ {direction} CLEARING - traveled {distance_traveled:.2f}m'
                )
                if direction == "FORWARD":
                    self.transition_to(self.CLEARING)
                else:
                    self.transition_to(self.REVERSE_CLEARING)
        
        return cmd

    def clearing(self, cmd, direction):
        """Continue forward to ensure complete clearance - NO lateral"""
        current_x = self.current_position[0] if self.current_position else 0
        
        if direction == "FORWARD":
            # Need X > gate_x + clearance
            if current_x > (self.gate_x_position + 1.5):
                elapsed = time.time() - self.mission_start_time
                self.get_logger().info('='*70)
                self.get_logger().info(f'✅ FORWARD PASS COMPLETE!')
                self.get_logger().info(f'   Position: X={current_x:.2f}m')
                self.get_logger().info(f'   Time: {elapsed:.1f}s')
                self.get_logger().info('   Starting U-turn (staying submerged)...')
                self.get_logger().info('='*70)
                
                self.uturn_start_yaw = self.current_yaw
                self.transition_to(self.UTURN)
        else:
            # Need X < gate_x - clearance
            if current_x < (self.gate_x_position - 1.5):
                elapsed = time.time() - self.mission_start_time
                self.get_logger().info('='*70)
                self.get_logger().info(f'✅ REVERSE PASS COMPLETE!')
                self.get_logger().info(f'   Position: X={current_x:.2f}m')
                self.get_logger().info(f'   Total time: {elapsed:.1f}s')
                self.get_logger().info('   🎉 QUALIFICATION COMPLETE - SURFACING!')
                self.get_logger().info('='*70)
                
                self.transition_to(self.SURFACING)
        
        # Continue forward - NO lateral drift
        cmd.linear.x = self.passing_speed
        cmd.linear.y = 0.0  # LOCKED
        cmd.angular.z = 0.0  # LOCKED
        
        return cmd

    def uturn(self, cmd):
        """180° turn while staying at cruise depth"""
        angle_turned = abs(self.normalize_angle(self.current_yaw - self.uturn_start_yaw))
        
        if angle_turned > (math.pi - 0.15):  # ~165°+
            self.get_logger().info('✅ U-turn complete - Starting reverse cruise')
            self.reverse_mode_pub.publish(Bool(data=True))
            self.transition_to(self.REVERSE_CRUISING)
            return cmd
        
        # Slow turn
        cmd.linear.x = 0.2
        cmd.angular.z = 0.7
        
        return cmd

    def reverse_cruise(self, cmd):
        """Cruise back toward gate (only yaw corrections, NO lateral)"""
        if not self.gate_detected:
            cmd.linear.x = 0.3
            cmd.linear.y = 0.0  # LOCKED
            cmd.angular.z = -0.4 if (time.time() % 8 < 4) else 0.4
            return cmd
        
        if self.estimated_distance <= self.alignment_distance:
            self.get_logger().info(
                f'🎯 Within 3m on reverse - Starting alignment'
            )
            self.transition_to(self.REVERSE_ALIGNMENT)
            return cmd
        
        cmd.linear.x = self.cruise_speed
        cmd.linear.y = 0.0  # ABSOLUTELY LOCKED
        cmd.angular.z = -self.alignment_error * 1.0
        
        return cmd

    def reverse_align(self, cmd):
        """Precision alignment for reverse pass - NO lateral movement"""
        if not self.gate_detected:
            cmd.linear.x = 0.2
            cmd.linear.y = 0.0  # LOCKED
            cmd.angular.z = -0.3
            return cmd
        
        if self.estimated_distance <= self.passing_distance:
            if abs(self.alignment_error) < 0.15:
                self.get_logger().info(
                    f'✅ Reverse aligned - COMMITTING TO PASS'
                )
                self.passing_start_x = self.current_position[0] if self.current_position else 0
                self.transition_to(self.REVERSE_PASSING)
                return cmd
        
        # Check depth stability
        depth_error = abs(self.gate_center_depth - self.current_depth)
        
        if depth_error > 0.3:
            # Adjusting depth - slow forward speed
            cmd.linear.x = 0.2
            self.get_logger().info(
                f'⚠️ Reverse depth adjust: error={depth_error:.2f}',
                throttle_duration_sec=1.0
            )
        else:
            cmd.linear.x = self.alignment_speed
        
        cmd.linear.y = 0.0  # ABSOLUTELY LOCKED
        cmd.angular.z = -self.alignment_error * 1.5
        
        return cmd

    def surfacing(self, cmd):
        """Surface slowly after mission complete"""
        if self.current_depth > -0.2:
            self.get_logger().info('✅ SURFACED - Mission complete!')
            self.transition_to(self.COMPLETED)
            return cmd
        
        cmd.linear.x = 0.0
        cmd.linear.y = 0.0
        cmd.linear.z = -0.5  # Upward
        cmd.angular.z = 0.0
        
        return cmd

    def completed(self, cmd):
        """Stop all motion"""
        cmd.linear.x = 0.0
        cmd.linear.y = 0.0
        cmd.linear.z = 0.0
        cmd.angular.z = 0.0
        return cmd

    def transition_to(self, new_state):
        self.state = new_state
        self.state_start_time = time.time()
        self.get_logger().info(f'🔄 → {self.get_state_name()}')
    
    def get_state_name(self):
        names = {
            self.SUBMERGING: 'SUBMERGING',
            self.CRUISING: 'CRUISING',
            self.FINAL_ALIGNMENT: 'FINAL_ALIGNMENT',
            self.PASSING: 'PASSING',
            self.CLEARING: 'CLEARING',
            self.UTURN: 'UTURN',
            self.REVERSE_CRUISING: 'REVERSE_CRUISING',
            self.REVERSE_ALIGNMENT: 'REVERSE_ALIGNMENT',
            self.REVERSE_PASSING: 'REVERSE_PASSING',
            self.REVERSE_CLEARING: 'REVERSE_CLEARING',
            self.SURFACING: 'SURFACING',
            self.COMPLETED: 'COMPLETED'
        }
        return names.get(self.state, 'UNKNOWN')
    
    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

def main(args=None):
    rclpy.init(args=args)
    node = FinalQualificationNavigator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()