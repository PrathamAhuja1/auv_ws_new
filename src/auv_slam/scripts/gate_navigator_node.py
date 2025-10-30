#!/usr/bin/env python3
"""
FIXED Gate Navigator Node - Starts immediately with working parameter loading
Key fixes:
1. Parameters declared with defaults in __init__
2. State starts as SEARCHING (not IDLE)
3. No waiting loop for parameter loading
4. Simplified control flow
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import time


class GateNavigatorNode(Node):
    def __init__(self):
        super().__init__('gate_navigator_node')
        
        # Mission states
        self.IDLE = 0
        self.SEARCHING = 1
        self.APPROACHING = 2
        self.AVOIDING_FLARE = 3
        self.PASSING = 4
        self.COMPLETED = 5
        
        # START SEARCHING IMMEDIATELY (not IDLE!)
        self.state = self.SEARCHING
        
        # Declare ALL parameters with defaults (so node works even without config file)
        self.declare_parameter('target_depth', -1.5)
        self.declare_parameter('depth_correction_gain', 1.5)
        self.declare_parameter('search_forward_speed', 0.5)
        self.declare_parameter('search_rotation_speed', 0.15)
        self.declare_parameter('approach_speed', 0.7)
        self.declare_parameter('passing_speed', 1.0)
        self.declare_parameter('passing_duration', 8.0)
        self.declare_parameter('yaw_correction_gain', 1.2)
        self.declare_parameter('approach_distance', 3.0)
        self.declare_parameter('passing_distance', 1.5)
        self.declare_parameter('flare_avoidance_gain', 0.8)
        self.declare_parameter('flare_avoidance_duration', 3.0)
        
        # Load parameters IMMEDIATELY in __init__
        self.target_depth = self.get_parameter('target_depth').value
        self.depth_gain = self.get_parameter('depth_correction_gain').value
        self.search_forward_speed = self.get_parameter('search_forward_speed').value
        self.search_rotation_speed = self.get_parameter('search_rotation_speed').value
        self.approach_speed = self.get_parameter('approach_speed').value
        self.passing_speed = self.get_parameter('passing_speed').value
        self.passing_duration = self.get_parameter('passing_duration').value
        self.yaw_correction_gain = self.get_parameter('yaw_correction_gain').value
        self.approach_distance = self.get_parameter('approach_distance').value
        self.passing_distance = self.get_parameter('passing_distance').value
        self.flare_avoidance_gain = self.get_parameter('flare_avoidance_gain').value
        self.flare_avoidance_duration = self.get_parameter('flare_avoidance_duration').value
        self.gate_lost_timeout = 3.0
        
        # State variables
        self.gate_detected = False
        self.flare_detected = False
        self.alignment_error = 0.0
        self.estimated_distance = 999.0
        self.current_depth = 0.0
        self.flare_avoidance_direction = 0.0
        
        self.gate_lost_time = None
        self.passing_start_time = None
        self.flare_avoidance_start_time = None
        self.state_start_time = time.time()
        
        # Performance tracking
        self.mission_start_time = time.time()
        self.gate_first_detected_time = None
        
        # Subscriptions
        self.gate_detected_sub = self.create_subscription(
            Bool, '/gate/detected', self.gate_detected_callback, 10)
        self.alignment_sub = self.create_subscription(
            Float32, '/gate/alignment_error', self.alignment_callback, 10)
        self.distance_sub = self.create_subscription(
            Float32, '/gate/estimated_distance', self.distance_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/ground_truth/odom', self.odom_callback, 10)
        self.flare_detected_sub = self.create_subscription(
            Bool, '/flare/detected', self.flare_detected_callback, 10)
        self.flare_avoidance_sub = self.create_subscription(
            Float32, '/flare/avoidance_direction', self.flare_avoidance_callback, 10)
        self.flare_warning_sub = self.create_subscription(
            String, '/flare/warning', self.flare_warning_callback, 10)
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/rp2040/cmd_vel', 10)
        self.state_pub = self.create_publisher(String, '/gate/navigation_state', 10)
        
        # Control loop (20 Hz for responsive control)
        self.control_timer = self.create_timer(0.05, self.control_loop)
        
        # LOG SUCCESS
        self.get_logger().info('='*70)
        self.get_logger().info('✅ Gate Navigator READY - Parameters loaded successfully!')
        self.get_logger().info(f'   Target depth: {self.target_depth}m')
        self.get_logger().info(f'   Approach speed: {self.approach_speed}m/s')
        self.get_logger().info(f'   Passing speed: {self.passing_speed}m/s')
        self.get_logger().info(f'   State: {self.get_state_name()}')
        self.get_logger().info('='*70)
    
    # ===== CALLBACKS =====
    
    def gate_detected_callback(self, msg: Bool):
        was_detected = self.gate_detected
        self.gate_detected = msg.data
        
        if not was_detected and self.gate_detected:
            if self.gate_first_detected_time is None:
                self.gate_first_detected_time = time.time()
                self.get_logger().info('🎯 GATE FIRST DETECTED!')
            self.gate_lost_time = None
        elif was_detected and not self.gate_detected:
            self.gate_lost_time = time.time()
            self.get_logger().warn('⚠️  Gate lost from view')
    
    def flare_detected_callback(self, msg: Bool):
        was_detected = self.flare_detected
        self.flare_detected = msg.data
        
        if not was_detected and self.flare_detected:
            self.get_logger().error('🚨 ORANGE FLARE DETECTED - INITIATING AVOIDANCE!')
            if self.state in [self.SEARCHING, self.APPROACHING]:
                self.flare_avoidance_start_time = time.time()
        elif was_detected and not self.flare_detected:
            self.get_logger().info('✓ Flare cleared')
            self.flare_avoidance_start_time = None
    
    def flare_avoidance_callback(self, msg: Float32):
        self.flare_avoidance_direction = msg.data
    
    def flare_warning_callback(self, msg: String):
        if "CRITICAL" in msg.data:
            self.get_logger().error(msg.data)
        else:
            self.get_logger().warn(msg.data)
    
    def alignment_callback(self, msg: Float32):
        self.alignment_error = msg.data
    
    def distance_callback(self, msg: Float32):
        self.estimated_distance = msg.data
    
    def odom_callback(self, msg: Odometry):
        self.current_depth = msg.pose.pose.position.z
    
    # ===== CONTROL =====
    
    def control_loop(self):
        """Main control loop - 20 Hz - NO WAITING!"""
        
        # Initialize command
        cmd = Twist()
        
        # Depth control (always active)
        depth_error = self.target_depth - self.current_depth
        cmd.linear.z = depth_error * self.depth_gain
        
        # Priority 1: Emergency flare avoidance
        if self.flare_detected and self.flare_avoidance_start_time:
            if self.state in [self.SEARCHING, self.APPROACHING]:
                self.state = self.AVOIDING_FLARE
        
        # State machine
        if self.state == self.SEARCHING:
            cmd = self.searching_behavior(cmd)
        elif self.state == self.APPROACHING:
            cmd = self.approaching_behavior(cmd)
        elif self.state == self.AVOIDING_FLARE:
            cmd = self.avoiding_flare_behavior(cmd)
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
        """OPTIMIZED SEARCH: Move forward while scanning"""
        
        if self.gate_detected:
            self.get_logger().info('✓ Gate detected during search - transitioning to approach')
            self.transition_to(self.APPROACHING)
            return cmd
        
        # Move forward at moderate speed
        cmd.linear.x = self.search_forward_speed
        
        # Gentle rotation to scan area
        cmd.angular.z = self.search_rotation_speed
        
        # Log progress
        elapsed = time.time() - self.state_start_time
        if int(elapsed) % 3 == 0 and elapsed > 0:
            self.get_logger().info(
                f'🔍 Searching... {elapsed:.0f}s elapsed',
                throttle_duration_sec=2.9
            )
        
        return cmd
    
    def approaching_behavior(self, cmd: Twist) -> Twist:
        """CONTINUOUS DRIFT CORRECTION: No stopping to align"""
        
        # Check if gate lost
        if not self.gate_detected:
            if self.gate_lost_time:
                lost_duration = time.time() - self.gate_lost_time
                if lost_duration > self.gate_lost_timeout:
                    self.get_logger().warn(
                        f'Gate lost for {lost_duration:.1f}s - returning to search'
                    )
                    self.transition_to(self.SEARCHING)
            return cmd  # Stop motion while waiting
        
        # Check if close enough to pass
        if self.estimated_distance < self.passing_distance and self.estimated_distance > 0:
            self.get_logger().info(f'✓ Within passing distance ({self.estimated_distance:.2f}m)')
            self.transition_to(self.PASSING)
            return cmd
        
        # CONTINUOUS DRIFT CORRECTION
        cmd.linear.x = self.approach_speed
        cmd.angular.z = -self.alignment_error * self.yaw_correction_gain
        
        # Log approach progress
        if self.estimated_distance < 999:
            self.get_logger().info(
                f'→ Approaching: {self.estimated_distance:.2f}m, '
                f'error: {self.alignment_error:+.3f}, '
                f'yaw: {cmd.angular.z:+.3f}',
                throttle_duration_sec=0.5
            )
        
        return cmd
    
    def avoiding_flare_behavior(self, cmd: Twist) -> Twist:
        """EMERGENCY FLARE AVOIDANCE: Strafe sideways"""
        
        # Check if flare cleared or timeout
        if not self.flare_detected:
            self.get_logger().info('✓ Flare avoidance complete - resuming approach')
            self.transition_to(self.APPROACHING)
            return cmd
        
        if self.flare_avoidance_start_time:
            elapsed = time.time() - self.flare_avoidance_start_time
            if elapsed > self.flare_avoidance_duration:
                self.get_logger().info('Flare avoidance timeout - resuming approach')
                self.transition_to(self.APPROACHING)
                return cmd
        
        # STRAFE SIDEWAYS to avoid flare
        cmd.linear.y = self.flare_avoidance_direction * self.flare_avoidance_gain
        cmd.linear.x = self.search_forward_speed * 0.5
        cmd.angular.z = 0.0
        
        self.get_logger().warn(
            f'⚠️  AVOIDING FLARE: Strafing {"RIGHT" if self.flare_avoidance_direction > 0 else "LEFT"}',
            throttle_duration_sec=0.5
        )
        
        return cmd
    
    def passing_behavior(self, cmd: Twist) -> Twist:
        """PASSING: Maximum speed straight through"""
        
        if self.passing_start_time is None:
            self.passing_start_time = time.time()
            self.get_logger().info('🚀 PASSING THROUGH GATE AT FULL SPEED!')
        
        elapsed = time.time() - self.passing_start_time
        
        if elapsed > self.passing_duration:
            self.get_logger().info('✅ GATE PASSAGE COMPLETE!')
            self.transition_to(self.COMPLETED)
            return cmd
        
        # Full speed ahead!
        cmd.linear.x = self.passing_speed
        cmd.linear.y = 0.0
        cmd.angular.z = 0.0
        
        self.get_logger().info(
            f'💨 PASSING... {elapsed:.1f}s / {self.passing_duration}s',
            throttle_duration_sec=0.5
        )
        
        return cmd
    
    def completed_behavior(self, cmd: Twist) -> Twist:
        """Mission complete - stop all motion"""
        
        # Calculate mission statistics
        if self.mission_start_time:
            total_time = time.time() - self.mission_start_time
            detection_time = (self.gate_first_detected_time - self.mission_start_time 
                             if self.gate_first_detected_time else 0)
            
            self.get_logger().info('='*70)
            self.get_logger().info('🎉 GATE NAVIGATION MISSION COMPLETE!')
            self.get_logger().info(f'   Total time: {total_time:.2f}s')
            self.get_logger().info(f'   Detection time: {detection_time:.2f}s')
            self.get_logger().info(f'   Navigation time: {total_time - detection_time:.2f}s')
            self.get_logger().info('='*70)
        
        # Stop all motion
        cmd.linear.x = 0.0
        cmd.linear.y = 0.0
        cmd.linear.z = 0.0
        cmd.angular.z = 0.0
        
        # Stop control loop
        if self.control_timer:
            self.control_timer.cancel()
            self.control_timer = None
        
        return cmd
    
    # ===== UTILITIES =====
    
    def transition_to(self, new_state: int):
        """Transition to new state with logging"""
        old_name = self.get_state_name()
        self.state = new_state
        self.state_start_time = time.time()
        new_name = self.get_state_name()
        
        self.get_logger().info(f'🔄 STATE TRANSITION: {old_name} → {new_name}')
    
    def get_state_name(self) -> str:
        """Get current state name"""
        names = {
            self.IDLE: 'IDLE',
            self.SEARCHING: 'SEARCHING',
            self.APPROACHING: 'APPROACHING',
            self.AVOIDING_FLARE: 'AVOIDING_FLARE',
            self.PASSING: 'PASSING',
            self.COMPLETED: 'COMPLETED'
        }
        return names.get(self.state, 'UNKNOWN')


def main(args=None):
    rclpy.init(args=args)
    node = GateNavigatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Ensure we stop the bot
        stop_cmd = Twist()
        node.cmd_vel_pub.publish(stop_cmd)
        node.get_logger().info('Gate Navigator shutting down')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()