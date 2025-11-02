#!/usr/bin/env python3
"""
IMPROVED Gate Navigator - Smooth Continuous Alignment
Fixed logging throttle issue
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
import time
import math


class ImprovedGateNavigator(Node):
    def __init__(self):
        super().__init__('gate_navigator_node')
        
        # Simplified state machine
        self.SEARCHING = 0
        self.APPROACHING = 1
        self.PASSING = 2
        self.COMPLETED = 3
        
        self.state = self.SEARCHING
        
        # Tuned parameters for smooth operation
        self.declare_parameter('target_depth', -1.7)
        self.declare_parameter('search_speed', 0.4)
        self.declare_parameter('approach_speed', 0.6)
        self.declare_parameter('passing_speed', 1.0)
        self.declare_parameter('passing_distance', 2.0)
        self.declare_parameter('passing_duration', 6.0)
        
        # Alignment parameters
        self.declare_parameter('max_yaw_correction', 0.8)
        self.declare_parameter('alignment_gain', 2.0)
        self.declare_parameter('min_confidence_for_approach', 0.3)
        
        # Depth control with proper deadband
        self.declare_parameter('depth_deadband', 0.25)
        self.declare_parameter('depth_gain_weak', 0.5)
        self.declare_parameter('depth_gain_strong', 1.5)
        
        self.target_depth = self.get_parameter('target_depth').value
        self.search_speed = self.get_parameter('search_speed').value
        self.approach_speed = self.get_parameter('approach_speed').value
        self.passing_speed = self.get_parameter('passing_speed').value
        self.passing_distance = self.get_parameter('passing_distance').value
        self.passing_duration = self.get_parameter('passing_duration').value
        self.max_yaw_correction = self.get_parameter('max_yaw_correction').value
        self.alignment_gain = self.get_parameter('alignment_gain').value
        self.min_confidence = self.get_parameter('min_confidence_for_approach').value
        self.depth_deadband = self.get_parameter('depth_deadband').value
        self.depth_gain_weak = self.get_parameter('depth_gain_weak').value
        self.depth_gain_strong = self.get_parameter('depth_gain_strong').value
        
        # State variables
        self.gate_detected = False
        self.flare_detected = False
        self.alignment_error = 0.0
        self.estimated_distance = 999.0
        self.current_depth = 0.0
        self.frame_position = 0.0
        self.confidence = 0.0
        self.partial_gate = False
        
        # Timing
        self.gate_lost_time = 0.0
        self.gate_lost_timeout = 2.0
        self.passing_start_time = 0.0
        self.state_start_time = time.time()
        self.last_gate_time = time.time()
        
        # Subscriptions
        self.gate_detected_sub = self.create_subscription(
            Bool, '/gate/detected', self.gate_detected_callback, 10)
        self.alignment_sub = self.create_subscription(
            Float32, '/gate/alignment_error', self.alignment_callback, 10)
        self.distance_sub = self.create_subscription(
            Float32, '/gate/estimated_distance', self.distance_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/ground_truth/odom', self.odom_callback, 10)
        self.frame_position_sub = self.create_subscription(
            Float32, '/gate/frame_position', self.frame_position_callback, 10)
        self.confidence_sub = self.create_subscription(
            Float32, '/gate/detection_confidence', self.confidence_callback, 10)
        self.partial_gate_sub = self.create_subscription(
            Bool, '/gate/partial_detection', self.partial_gate_callback, 10)
        self.flare_detected_sub = self.create_subscription(
            Bool, '/flare/detected', self.flare_detected_callback, 10)
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/rp2040/cmd_vel', 10)
        self.state_pub = self.create_publisher(String, '/gate/navigation_state', 10)
        
        # Control loop at 20Hz
        self.control_timer = self.create_timer(0.05, self.control_loop)
        
        self.get_logger().info('='*70)
        self.get_logger().info('✅ IMPROVED Gate Navigator - Continuous Alignment')
        self.get_logger().info('='*70)
        self.get_logger().info(f'   Approach speed: {self.approach_speed} m/s (constant)')
        self.get_logger().info(f'   Alignment gain: {self.alignment_gain}')
        self.get_logger().info(f'   Max yaw correction: {self.max_yaw_correction} rad/s')
        self.get_logger().info('='*70)
    
    # Callback functions
    def gate_detected_callback(self, msg: Bool):
        was_detected = self.gate_detected
        self.gate_detected = msg.data
        
        if self.gate_detected:
            self.last_gate_time = time.time()
            if not was_detected:
                self.get_logger().info('🎯 Gate detected!')
        else:
            if was_detected:
                self.gate_lost_time = time.time()
    
    def frame_position_callback(self, msg: Float32):
        self.frame_position = msg.data
    
    def confidence_callback(self, msg: Float32):
        self.confidence = msg.data
    
    def partial_gate_callback(self, msg: Bool):
        self.partial_gate = msg.data
    
    def alignment_callback(self, msg: Float32):
        self.alignment_error = msg.data
    
    def distance_callback(self, msg: Float32):
        self.estimated_distance = msg.data
    
    def odom_callback(self, msg: Odometry):
        self.current_depth = msg.pose.pose.position.z
    
    def flare_detected_callback(self, msg: Bool):
        self.flare_detected = msg.data
    
    def compute_depth_correction(self) -> float:
        """
        IMPROVED depth control with proper deadband
        Prevents oscillation while maintaining depth
        """
        depth_error = self.target_depth - self.current_depth
        
        # Deadband - no correction if within ±25cm
        if abs(depth_error) < self.depth_deadband:
            return 0.0
        
        # Apply correction outside deadband
        if abs(depth_error) < 0.5:
            # Small error - gentle correction
            correction = depth_error * self.depth_gain_weak
        else:
            # Large error - strong correction
            correction = depth_error * self.depth_gain_strong
        
        # Limit maximum correction
        return max(-0.8, min(correction, 0.8))
    
    def compute_yaw_correction(self) -> float:
        """
        CRITICAL: Compute yaw correction based on gate position
        Uses proportional control with saturation
        """
        if not self.gate_detected:
            return 0.0
        
        # Use frame_position for correction
        yaw_correction = -self.frame_position * self.alignment_gain
        
        # Apply confidence scaling
        if self.confidence < 0.8:
            yaw_correction *= self.confidence
        
        # Saturate to max yaw rate
        yaw_correction = max(-self.max_yaw_correction, 
                            min(yaw_correction, self.max_yaw_correction))
        
        return yaw_correction
    
    def control_loop(self):
        """Main control loop - runs at 20Hz"""
        cmd = Twist()
        
        # ALWAYS apply depth correction
        cmd.linear.z = self.compute_depth_correction()
        
        # State machine
        if self.state == self.SEARCHING:
            cmd = self.searching_behavior(cmd)
        elif self.state == self.APPROACHING:
            cmd = self.approaching_behavior(cmd)
        elif self.state == self.PASSING:
            cmd = self.passing_behavior(cmd)
        elif self.state == self.COMPLETED:
            cmd = self.completed_behavior(cmd)
        
        # Publish command
        self.cmd_vel_pub.publish(cmd)
        
        # Publish state
        state_msg = String()
        state_msg.data = self.get_state_name()
        self.state_pub.publish(state_msg)
    
    def searching_behavior(self, cmd: Twist) -> Twist:
        """Search for gate with gentle sweep pattern"""
        if self.gate_detected and self.confidence > self.min_confidence:
            self.get_logger().info('🎯 Gate acquired - switching to APPROACHING')
            self.transition_to(self.APPROACHING)
            return cmd
        
        # Sweep pattern
        elapsed = time.time() - self.state_start_time
        sweep_period = 10.0
        sweep_phase = (elapsed % sweep_period) / sweep_period
        
        # Gentle oscillating yaw while moving forward
        if sweep_phase < 0.5:
            cmd.angular.z = 0.2
        else:
            cmd.angular.z = -0.2
        
        cmd.linear.x = self.search_speed
        
        # FIXED: Use constant throttle duration
        self.get_logger().info(
            f'🔍 Searching... {elapsed:.0f}s',
            throttle_duration_sec=3.0
        )
        
        return cmd
    
    def approaching_behavior(self, cmd: Twist) -> Twist:
        """
        CRITICAL: Continuous forward motion with proportional yaw correction
        """
        # Check if gate lost
        if not self.gate_detected:
            lost_duration = time.time() - self.last_gate_time
            if lost_duration > self.gate_lost_timeout:
                self.get_logger().warn('❌ Gate lost - returning to search')
                self.transition_to(self.SEARCHING)
                return cmd
            else:
                # Recently lost - keep last command
                # FIXED: Use separate logging with constant throttle
                self.get_logger().warn(
                    '⚠️ Gate not visible - holding course',
                    throttle_duration_sec=1.0
                )
                cmd.linear.x = self.approach_speed * 0.5
                return cmd
        
        # Check if close enough to pass
        if self.estimated_distance < self.passing_distance and self.estimated_distance > 0.1:
            self.get_logger().info(
                f'🚀 Within passing distance ({self.estimated_distance:.2f}m)'
            )
            self.transition_to(self.PASSING)
            return cmd
        
        # CONTINUOUS APPROACH WITH DRIFT CORRECTION
        cmd.linear.x = self.approach_speed
        cmd.angular.z = self.compute_yaw_correction()
        
        # Reduce speed if alignment is very poor
        if abs(self.frame_position) > 0.6:
            cmd.linear.x *= 0.7
            # FIXED: Separate warning log with constant throttle
            self.get_logger().warn(
                f'⚠️ Poor alignment (pos={self.frame_position:+.2f}) - reducing speed',
                throttle_duration_sec=1.0
            )
        
        # FIXED: Single info log with constant throttle duration
        self.get_logger().info(
            f'➡️ APPROACH: dist={self.estimated_distance:.2f}m, '
            f'pos={self.frame_position:+.2f}, yaw={cmd.angular.z:+.2f}, '
            f'conf={self.confidence:.2f}',
            throttle_duration_sec=0.5
        )
        
        return cmd
    
    def passing_behavior(self, cmd: Twist) -> Twist:
        """Full speed straight through the gate"""
        if self.passing_start_time == 0.0:
            self.passing_start_time = time.time()
            self.get_logger().info('🚀 PASSING THROUGH GATE!')
        
        elapsed = time.time() - self.passing_start_time
        
        if elapsed > self.passing_duration:
            self.get_logger().info('✅ GATE PASSAGE COMPLETE!')
            self.transition_to(self.COMPLETED)
            return cmd
        
        # Full speed ahead, no corrections
        cmd.linear.x = self.passing_speed
        cmd.linear.y = 0.0
        cmd.angular.z = 0.0
        
        self.get_logger().info(
            f'🚀 PASSING... {elapsed:.1f}s / {self.passing_duration:.1f}s',
            throttle_duration_sec=0.5
        )
        
        return cmd
    
    def completed_behavior(self, cmd: Twist) -> Twist:
        """Mission complete - stop all motion"""
        cmd.linear.x = 0.0
        cmd.linear.y = 0.0
        cmd.linear.z = 0.0
        cmd.angular.z = 0.0
        
        if self.control_timer:
            self.control_timer.cancel()
            self.control_timer = None
            self.get_logger().info('='*70)
            self.get_logger().info('🎉 GATE NAVIGATION COMPLETE')
            self.get_logger().info('='*70)
        
        return cmd
    
    def transition_to(self, new_state: int):
        """State transition helper"""
        old_name = self.get_state_name()
        self.state = new_state
        self.state_start_time = time.time()
        new_name = self.get_state_name()
        self.get_logger().info(f'🔄 STATE: {old_name} → {new_name}')
    
    def get_state_name(self) -> str:
        """Get human-readable state name"""
        names = {
            self.SEARCHING: 'SEARCHING',
            self.APPROACHING: 'APPROACHING',
            self.PASSING: 'PASSING',
            self.COMPLETED: 'COMPLETED'
        }
        return names.get(self.state, 'UNKNOWN')


def main(args=None):
    rclpy.init(args=args)
    node = ImprovedGateNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop_cmd = Twist()
        node.cmd_vel_pub.publish(stop_cmd)
        node.get_logger().info('Gate Navigator shutting down')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()