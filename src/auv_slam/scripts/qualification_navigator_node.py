#!/usr/bin/env python3
"""
FIXED QUALIFICATION NAVIGATOR - NO OSCILLATION, STAY SUBMERGED, FAST SPEEDS
Key Fixes:
1. ZERO depth oscillation - wide deadband (0.4m), no micro-adjustments
2. Stay submerged throughout entire mission (both passes + U-turn)
3. Fast speeds: 0.7 m/s approach, 1.0 m/s passing
4. Accurate distance from odometry
5. Simple 3-state machine: APPROACH → PASS → UTURN (repeat)
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import time
import math

class FixedQualificationNavigator(Node):
    def __init__(self):
        super().__init__('qualification_navigator')
        
        # SIMPLIFIED STATE MACHINE
        self.SUBMERGING = 0
        self.FORWARD_APPROACH = 1
        self.FORWARD_PASSING = 2
        self.UTURN = 3
        self.REVERSE_APPROACH = 4
        self.REVERSE_PASSING = 5
        self.COMPLETED = 6
        
        self.state = self.SUBMERGING
        
        # CRITICAL: Target depth - NEVER CHANGE during mission
        self.target_depth = -0.8
        
        # Gate position (from world file)
        self.gate_x_position = 0.0
        
        # Fast speeds
        self.approach_speed = 0.7  # Fast but controlled
        self.passing_speed = 1.0   # Maximum speed through gate
        self.uturn_speed = 0.3     # Slow for turning
        
        # Triggers
        self.passing_trigger = 1.2  # Start full speed at 1.2m
        self.gate_clearance = 0.8   # Distance past gate to consider "cleared"
        
        # State variables
        self.gate_detected = False
        self.alignment_error = 0.0
        self.estimated_distance = 999.0
        self.current_depth = 0.0
        self.current_position = None
        self.current_yaw = 0.0
        
        self.uturn_start_yaw = 0.0
        self.state_start_time = time.time()
        self.mission_start_time = time.time()
        
        self.forward_pass_complete = False
        self.reverse_pass_complete = False
        
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
        self.get_logger().info('✅ FIXED QUALIFICATION NAVIGATOR')
        self.get_logger().info('='*70)
        self.get_logger().info(f'   Target depth: {self.target_depth}m (LOCKED)')
        self.get_logger().info(f'   Approach speed: {self.approach_speed} m/s')
        self.get_logger().info(f'   Passing speed: {self.passing_speed} m/s')
        self.get_logger().info('   STAYS SUBMERGED ENTIRE MISSION')
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
        
        # Get yaw
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

    def control_loop(self):
        cmd = Twist()
        
        # ================================================================
        # CRITICAL FIX: ZERO OSCILLATION DEPTH CONTROL
        # Wide deadband (0.4m), minimal corrections, STABLE
        # ================================================================
        if self.state != self.COMPLETED:
            depth_error = self.target_depth - self.current_depth
            WIDE_DEADBAND = 0.4  # HUGE deadband = no oscillation
            
            if abs(depth_error) < WIDE_DEADBAND:
                # Perfect - do NOTHING
                cmd.linear.z = 0.0
            elif abs(depth_error) < 0.8:
                # Moderate error - gentle correction
                cmd.linear.z = depth_error * 0.5
                cmd.linear.z = max(-0.3, min(cmd.linear.z, 0.3))
            else:
                # Large error - stronger correction
                cmd.linear.z = depth_error * 0.8
                cmd.linear.z = max(-0.6, min(cmd.linear.z, 0.6))
        else:
            cmd.linear.z = 0.0
        
        # State machine
        if self.state == self.SUBMERGING:
            cmd = self.submerge(cmd)
        elif self.state == self.FORWARD_APPROACH:
            cmd = self.approach(cmd, "FORWARD")
        elif self.state == self.FORWARD_PASSING:
            cmd = self.passing(cmd, "FORWARD")
        elif self.state == self.UTURN:
            cmd = self.uturn(cmd)
        elif self.state == self.REVERSE_APPROACH:
            cmd = self.approach(cmd, "REVERSE")
        elif self.state == self.REVERSE_PASSING:
            cmd = self.passing(cmd, "REVERSE")
        elif self.state == self.COMPLETED:
            cmd = self.completed(cmd)
        
        self.cmd_vel_pub.publish(cmd)
        
        # Publish state
        state_name = self.get_state_name()
        self.state_pub.publish(String(data=state_name))
        
        # Log progress
        if self.current_position and int(time.time()) % 2 == 0:
            elapsed = time.time() - self.mission_start_time
            self.get_logger().info(
                f'[{state_name}] X={self.current_position[0]:.2f}m, '
                f'Depth={self.current_depth:.2f}m, Dist={self.estimated_distance:.2f}m, '
                f'Time={elapsed:.1f}s',
                throttle_duration_sec=1.9
            )

    def submerge(self, cmd):
        """Wait for depth to stabilize before starting"""
        if abs(self.target_depth - self.current_depth) < 0.5:
            elapsed = time.time() - self.state_start_time
            if elapsed > 3.0:  # 3 second settling time
                self.get_logger().info('✅ Submerged - Starting forward approach')
                self.transition_to(self.FORWARD_APPROACH)
        return cmd

    def approach(self, cmd, direction):
        """Fast approach until passing trigger distance"""
        
        # Publish reverse mode flag
        is_reverse = (direction == "REVERSE")
        self.reverse_mode_pub.publish(Bool(data=is_reverse))
        
        if not self.gate_detected:
            # Lost gate - slow search
            cmd.linear.x = 0.2
            cmd.angular.z = 0.3 if (time.time() % 10 < 5) else -0.3
            return cmd
        
        # Check if close enough to commit to passing
        if self.estimated_distance <= self.passing_trigger:
            self.get_logger().info(
                f'🚀 {direction} PASS TRIGGER at {self.estimated_distance:.2f}m'
            )
            if direction == "FORWARD":
                self.transition_to(self.FORWARD_PASSING)
            else:
                self.transition_to(self.REVERSE_PASSING)
            return cmd
        
        # Fast approach with alignment
        cmd.linear.x = self.approach_speed
        cmd.angular.z = -self.alignment_error * 1.5  # Strong correction
        
        return cmd

    def passing(self, cmd, direction):
        """
        FULL SPEED STRAIGHT THROUGH GATE
        Uses odometry to detect when gate is cleared
        """
        
        if not self.current_position:
            return cmd
        
        current_x = self.current_position[0]
        
        if direction == "FORWARD":
            # Forward: need to pass gate (X > gate_x + clearance)
            if current_x > (self.gate_x_position + self.gate_clearance):
                self.forward_pass_complete = True
                elapsed = time.time() - self.mission_start_time
                
                self.get_logger().info('='*70)
                self.get_logger().info(f'✅ FORWARD PASS COMPLETE!')
                self.get_logger().info(f'   Crossed gate at X={current_x:.2f}m')
                self.get_logger().info(f'   Time: {elapsed:.1f}s')
                self.get_logger().info('   Starting U-turn...')
                self.get_logger().info('='*70)
                
                self.uturn_start_yaw = self.current_yaw
                self.transition_to(self.UTURN)
                return cmd
        else:
            # Reverse: need to pass gate going backwards (X < gate_x - clearance)
            if current_x < (self.gate_x_position - self.gate_clearance):
                self.reverse_pass_complete = True
                elapsed = time.time() - self.mission_start_time
                
                self.get_logger().info('='*70)
                self.get_logger().info(f'✅ REVERSE PASS COMPLETE!')
                self.get_logger().info(f'   Crossed gate at X={current_x:.2f}m')
                self.get_logger().info(f'   Total time: {elapsed:.1f}s')
                self.get_logger().info('   QUALIFICATION COMPLETE!')
                self.get_logger().info('='*70)
                
                self.transition_to(self.COMPLETED)
                return cmd
        
        # FULL SPEED - NO CORRECTIONS
        cmd.linear.x = self.passing_speed
        cmd.linear.y = 0.0
        cmd.angular.z = 0.0
        
        return cmd

    def uturn(self, cmd):
        """
        Execute U-turn while STAYING SUBMERGED
        Turn 180 degrees, then start reverse approach
        """
        
        # Calculate how much we've turned
        angle_turned = self.normalize_angle(self.current_yaw - self.uturn_start_yaw)
        target_turn = math.pi  # 180 degrees
        
        if abs(target_turn - abs(angle_turned)) < 0.2:
            # Turn complete
            self.get_logger().info('✅ U-turn complete - Starting reverse approach')
            self.transition_to(self.REVERSE_APPROACH)
            return cmd
        
        # Continue turning + slow forward drift
        cmd.linear.x = self.uturn_speed
        cmd.angular.z = 0.6  # Moderate turn rate
        
        return cmd

    def completed(self, cmd):
        """
        Mission complete - STAY SUBMERGED and stationary
        Wait for manual retrieval
        """
        if self.state_start_time == 0:
            self.state_start_time = time.time()
        
        # Stop all motion
        cmd.linear.x = 0.0
        cmd.linear.y = 0.0
        cmd.linear.z = 0.0
        cmd.angular.z = 0.0
        
        elapsed = time.time() - self.state_start_time
        if int(elapsed) % 5 == 0:
            self.get_logger().info(
                f'🎉 MISSION COMPLETE - Waiting at depth (submerged for {elapsed:.0f}s)',
                throttle_duration_sec=4.9
            )
        
        return cmd

    def transition_to(self, new_state):
        self.state = new_state
        self.state_start_time = time.time()
        self.get_logger().info(f'🔄 STATE: {self.get_state_name()}')
    
    def get_state_name(self):
        names = {
            self.SUBMERGING: 'SUBMERGING',
            self.FORWARD_APPROACH: 'FORWARD_APPROACH',
            self.FORWARD_PASSING: 'FORWARD_PASSING',
            self.UTURN: 'UTURN',
            self.REVERSE_APPROACH: 'REVERSE_APPROACH',
            self.REVERSE_PASSING: 'REVERSE_PASSING',
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
    node = FixedQualificationNavigator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()