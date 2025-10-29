#!/usr/bin/env python3
"""
Fixed Gate Mission Controller
Ensures proper forward motion and gate detection
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Float32
import time
import math


class SimpleGateMission(Node):
    """Fixed mission controller for gate passing"""
    
    def __init__(self):
        super().__init__('simple_gate_mission')
        
        # Mission states
        self.INIT = 0
        self.SUBMERGING = 1
        self.SEARCHING_GATE = 2
        self.APPROACHING_GATE = 3
        self.PASSING_THROUGH = 4
        self.COMPLETED = 5
        
        self.state = self.INIT
        
        # Parameters - OPTIMIZED FOR MOVEMENT
        self.declare_parameter('target_depth', -1.5)
        self.declare_parameter('depth_tolerance', 0.3)  # Looser tolerance
        self.declare_parameter('search_speed', 0.6)  # Good forward speed
        self.declare_parameter('approach_speed', 0.7)
        self.declare_parameter('passing_speed', 1.0)  # Full speed
        self.declare_parameter('alignment_threshold', 0.15)
        self.declare_parameter('approach_distance', 3.0)
        self.declare_parameter('passing_distance', 1.5)
        self.declare_parameter('passing_duration', 8.0)  # Longer to ensure complete passage
        self.declare_parameter('yaw_gain', 0.8)
        self.declare_parameter('depth_gain', 1.2)  # Higher gain for faster depth control
        
        self.target_depth = self.get_parameter('target_depth').value
        self.depth_tolerance = self.get_parameter('depth_tolerance').value
        self.search_speed = self.get_parameter('search_speed').value
        self.approach_speed = self.get_parameter('approach_speed').value
        self.passing_speed = self.get_parameter('passing_speed').value
        self.alignment_threshold = self.get_parameter('alignment_threshold').value
        self.approach_distance = self.get_parameter('approach_distance').value
        self.passing_distance = self.get_parameter('passing_distance').value
        self.passing_duration = self.get_parameter('passing_duration').value
        self.yaw_gain = self.get_parameter('yaw_gain').value
        self.depth_gain = self.get_parameter('depth_gain').value
        
        # State variables
        self.current_depth = 0.0
        self.gate_detected = False
        self.gate_alignment_error = 0.0
        self.gate_distance = 999.0
        self.state_start_time = time.time()
        self.passing_start_time = None
        self.last_cmd_time = time.time()
        
        # Subscriptions
        self.odom_sub = self.create_subscription(
            Odometry, '/ground_truth/odom', self.odom_callback, 10)
        
        self.gate_detected_sub = self.create_subscription(
            Bool, '/gate/detected', self.gate_detected_callback, 10)
        
        self.gate_alignment_sub = self.create_subscription(
            Float32, '/gate/alignment_error', self.gate_alignment_callback, 10)
        
        self.gate_distance_sub = self.create_subscription(
            Float32, '/gate/estimated_distance', self.gate_distance_callback, 10)
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/rp2040/cmd_vel', 10)
        
        # Control loop (20 Hz)
        self.timer = self.create_timer(0.05, self.control_loop)
        
        self.get_logger().info('='*70)
        self.get_logger().info('🚀 FIXED Gate Mission Controller Started!')
        self.get_logger().info('='*70)
        self.get_logger().info(f'📍 Target Depth: {self.target_depth}m')
        self.get_logger().info(f'🎯 Search Speed: {self.search_speed} m/s')
        self.get_logger().info(f'⚡ Passing Speed: {self.passing_speed} m/s')
        self.get_logger().info('='*70)
    
    # ===== CALLBACKS =====
    
    def odom_callback(self, msg: Odometry):
        """Update current depth from odometry"""
        self.current_depth = msg.pose.pose.position.z
        
        # Log position every 2 seconds to verify movement
        if time.time() - self.last_cmd_time > 2.0:
            pos = msg.pose.pose.position
            self.get_logger().info(
                f'📍 Position: X={pos.x:.2f}, Y={pos.y:.2f}, Z={pos.z:.2f}',
                throttle_duration_sec=2.0
            )
            self.last_cmd_time = time.time()
    
    def gate_detected_callback(self, msg: Bool):
        """Update gate detection status"""
        was_detected = self.gate_detected
        self.gate_detected = msg.data
        
        if not was_detected and self.gate_detected:
            self.get_logger().info('✅ GATE DETECTED!')
    
    def gate_alignment_callback(self, msg: Float32):
        """Update gate alignment error"""
        self.gate_alignment_error = msg.data
    
    def gate_distance_callback(self, msg: Float32):
        """Update estimated distance to gate"""
        self.gate_distance = msg.data
    
    # ===== CONTROL LOOP =====
    
    def control_loop(self):
        """Main control loop"""
        
        if self.state == self.INIT:
            self.handle_init()
        elif self.state == self.SUBMERGING:
            self.handle_submerging()
        elif self.state == self.SEARCHING_GATE:
            self.handle_searching()
        elif self.state == self.APPROACHING_GATE:
            self.handle_approaching()
        elif self.state == self.PASSING_THROUGH:
            self.handle_passing()
        elif self.state == self.COMPLETED:
            self.handle_completed()
    
    # ===== STATE HANDLERS =====
    
    def handle_init(self):
        """Initialize mission"""
        self.get_logger().info('📍 State: INIT → Starting mission...')
        time.sleep(1.0)  # Give systems time to initialize
        self.transition_to(self.SUBMERGING)
    
    def handle_submerging(self):
        """Submerge to target depth while moving forward"""
        depth_error = self.target_depth - self.current_depth
        elapsed = time.time() - self.state_start_time

        # Create command
        cmd = Twist()
        
        # CRITICAL: Always move forward while submerging
        cmd.linear.x = 0.4  # Forward motion
        cmd.linear.y = 0.0
        cmd.angular.z = 0.0
        
        # Depth control
        cmd.linear.z = depth_error * self.depth_gain
        
        # Publish command
        self.cmd_vel_pub.publish(cmd)

        # Log progress
        if int(elapsed) % 2 == 0:
            self.get_logger().info(
                f'⬇️  Submerging & Moving: Depth={self.current_depth:.2f}m, '
                f'Target={self.target_depth:.2f}m, Forward Speed=0.4',
                throttle_duration_sec=2.0
            )

        # Check if depth reached
        if abs(depth_error) < self.depth_tolerance:
            self.get_logger().info(f'✅ Target depth reached: {self.current_depth:.2f}m')
            self.get_logger().info('➡️  Continuing forward to find gate...')
            self.transition_to(self.SEARCHING_GATE)
            return

        # Timeout - continue anyway
        if elapsed > 15.0:
            self.get_logger().warn(f'⚠️  Depth timeout! Moving to search...')
            self.transition_to(self.SEARCHING_GATE)
    
    def handle_searching(self):
        """Search for gate while maintaining depth and moving forward"""
        elapsed = time.time() - self.state_start_time

        # Check if gate detected
        if self.gate_detected:
            self.get_logger().info('✅ Gate found! Moving to approach...')
            self.transition_to(self.APPROACHING_GATE)
            return

        # Create command
        cmd = Twist()
        
        # CRITICAL: Keep moving forward
        cmd.linear.x = self.search_speed  # Continuous forward motion
        cmd.linear.y = 0.0
        
        # Add slight scanning after 10 seconds
        if elapsed > 10.0:
            cmd.angular.z = 0.1 * math.sin(elapsed * 0.5)  # Gentle oscillation
        else:
            cmd.angular.z = 0.0  # Straight ahead first
        
        # Maintain depth
        depth_error = self.target_depth - self.current_depth
        cmd.linear.z = depth_error * self.depth_gain
        
        # Publish command
        self.cmd_vel_pub.publish(cmd)

        # Log search status
        if int(elapsed) % 3 == 0:
            self.get_logger().info(
                f'🔍 Searching: Time={elapsed:.1f}s, '
                f'Speed={self.search_speed:.2f}, '
                f'Depth={self.current_depth:.2f}m',
                throttle_duration_sec=3.0
            )

        # Timeout - just keep going forward
        if elapsed > 60.0:
            self.get_logger().warn('⏱️  Search timeout - proceeding forward anyway!')
            # Don't stop, just keep moving forward
            cmd.linear.x = self.passing_speed
            cmd.linear.z = depth_error * self.depth_gain
            self.cmd_vel_pub.publish(cmd)
    
    def handle_approaching(self):
        """Approach gate while maintaining alignment"""
        elapsed = time.time() - self.state_start_time

        # Check if gate lost
        if not self.gate_detected:
            if elapsed > 2.0:  # Give it 2 seconds before giving up
                self.get_logger().warn('⚠️  Gate lost! Returning to search...')
                self.transition_to(self.SEARCHING_GATE)
            return

        # Check if close enough to pass
        if (self.gate_distance < self.passing_distance and self.gate_distance > 0.1) or elapsed > 10.0:
            self.get_logger().info(
                f'✅ Ready to pass! Distance: {self.gate_distance:.2f}m'
            )
            self.transition_to(self.PASSING_THROUGH)
            return

        # Create command
        cmd = Twist()
        
        # Keep moving forward
        cmd.linear.x = self.approach_speed
        cmd.linear.y = 0.0
        
        # Alignment correction
        cmd.angular.z = -self.gate_alignment_error * self.yaw_gain
        
        # Maintain depth
        depth_error = self.target_depth - self.current_depth
        cmd.linear.z = depth_error * self.depth_gain
        
        # Publish command
        self.cmd_vel_pub.publish(cmd)

        # Log status
        if int(elapsed * 2) % 2 == 0:
            self.get_logger().info(
                f'➡️  Approaching: Distance={self.gate_distance:.2f}m, '
                f'Alignment={self.gate_alignment_error:.3f}, '
                f'Speed={self.approach_speed:.2f}',
                throttle_duration_sec=0.5
            )
    
    def handle_passing(self):
        """Pass through gate at maximum speed"""

        # Initialize passing timer
        if self.passing_start_time is None:
            self.passing_start_time = time.time()
            self.get_logger().info('🚀 PASSING THROUGH GATE - FULL SPEED!')

        elapsed = time.time() - self.passing_start_time

        # Check if passed through
        if elapsed > self.passing_duration:
            self.get_logger().info(
                f'✅ GATE PASSED! Duration: {elapsed:.1f}s'
            )
            self.transition_to(self.COMPLETED)
            return

        # Create command
        cmd = Twist()
        
        # MAXIMUM FORWARD SPEED
        cmd.linear.x = self.passing_speed
        cmd.linear.y = 0.0
        cmd.angular.z = 0.0
        
        # Maintain depth
        depth_error = self.target_depth - self.current_depth
        cmd.linear.z = depth_error * self.depth_gain
        
        # Publish command
        self.cmd_vel_pub.publish(cmd)

        # Log status
        if int(elapsed) % 1 == 0:
            self.get_logger().info(
                f'🚀 Passing: {elapsed:.1f}s / {self.passing_duration:.1f}s, '
                f'Speed={self.passing_speed:.2f}',
                throttle_duration_sec=1.0
            )
    
    def handle_completed(self):
        """Mission complete - stop all motion"""
        
        # Log completion once
        elapsed = time.time() - self.state_start_time
        if elapsed < 0.5:
            self.get_logger().info('='*70)
            self.get_logger().info('✅ GATE MISSION COMPLETED!')
            self.get_logger().info('='*70)
        
        # Stop all motion
        cmd = Twist()
        self.cmd_vel_pub.publish(cmd)
    
    # ===== UTILITIES =====
    
    def transition_to(self, new_state: int):
        """Transition to new state"""
        state_names = {
            self.INIT: 'INIT',
            self.SUBMERGING: 'SUBMERGING',
            self.SEARCHING_GATE: 'SEARCHING_GATE',
            self.APPROACHING_GATE: 'APPROACHING_GATE',
            self.PASSING_THROUGH: 'PASSING_THROUGH',
            self.COMPLETED: 'COMPLETED'
        }
        
        old_name = state_names.get(self.state, 'UNKNOWN')
        new_name = state_names.get(new_state, 'UNKNOWN')
        
        self.get_logger().info(f'🔄 STATE CHANGE: {old_name} → {new_name}')
        
        self.state = new_state
        self.state_start_time = time.time()
        
        # Reset passing timer when leaving passing state
        if self.state != self.PASSING_THROUGH:
            self.passing_start_time = None


def main(args=None):
    rclpy.init(args=args)
    node = SimpleGateMission()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down mission controller...')
    finally:
        # Stop all motion on shutdown
        cmd = Twist()
        node.cmd_vel_pub.publish(cmd)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()