#!/usr/bin/env python3
"""
Simple Gate Mission Controller
Focuses only on: Detection → Navigation → Passing
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Float32
import time
import math


class SimpleGateMission(Node):
    """Simplified mission controller for gate passing only"""
    
    def __init__(self):
        super().__init__('simple_gate_mission')
        
        # Mission states
        self.INIT = 0
        self.SUBMERGING = 1
        self.SEARCHING_GATE = 2
        self.ALIGNING_TO_GATE = 3
        self.APPROACHING_GATE = 4
        self.PASSING_THROUGH = 5
        self.COMPLETED = 6
        
        self.state = self.INIT
        
        # Parameters - OPTIMIZED FOR SPEED AND ACCURACY
        self.declare_parameter('target_depth', -1.5)
        self.declare_parameter('depth_tolerance', 0.2)  # Slightly looser to reduce oscillation
        self.declare_parameter('search_speed', 0.5)  # Increased for faster search
        self.declare_parameter('approach_speed', 0.6)  # Increased for faster approach
        self.declare_parameter('passing_speed', 0.8)  # Increased for faster passing
        self.declare_parameter('alignment_threshold', 0.10)  # Tighter for better accuracy
        self.declare_parameter('approach_distance', 3.0)  # Start approach earlier
        self.declare_parameter('passing_distance', 1.2)  # Start passing closer
        self.declare_parameter('passing_duration', 6.0)  # Longer to ensure complete passage
        self.declare_parameter('yaw_gain', 1.0)  # Higher for faster alignment
        self.declare_parameter('depth_gain', 0.8)  # Lower to prevent oscillation
        
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
        self.get_logger().info('🚀 Simple Gate Mission Controller Started!')
        self.get_logger().info('='*70)
        self.get_logger().info(f'📍 Target Depth: {self.target_depth}m')
        self.get_logger().info(f'🎯 Approach Distance: {self.approach_distance}m')
        self.get_logger().info(f'⚡ Passing Distance: {self.passing_distance}m')
        self.get_logger().info('='*70)
    
    # ===== CALLBACKS =====
    
    def odom_callback(self, msg: Odometry):
        """Update current depth from odometry"""
        self.current_depth = msg.pose.pose.position.z
    
    def gate_detected_callback(self, msg: Bool):
        """Update gate detection status"""
        self.gate_detected = msg.data
    
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
        elif self.state == self.ALIGNING_TO_GATE:
            self.handle_aligning()
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
        time.sleep(0.5)
        self.transition_to(self.SUBMERGING)
    
    def handle_submerging(self):
        """Submerge to target depth while moving forward"""
        depth_error = self.target_depth - self.current_depth
        elapsed = time.time() - self.state_start_time

        # Log progress every 1 second
        if int(elapsed) % 1 == 0:
            self.get_logger().info(
                f'⬇️  Submerging: Current={self.current_depth:.2f}m, '
                f'Target={self.target_depth:.2f}m, Error={depth_error:.2f}m'
            )

        # Check if depth reached
        if abs(depth_error) < self.depth_tolerance:
            self.get_logger().info(f'✅ Target depth reached: {self.current_depth:.2f}m')
            self.transition_to(self.SEARCHING_GATE)
            return

        # Timeout after 20 seconds (reduced from 30)
        if elapsed > 20.0:
            self.get_logger().warn(f'⚠️  Submerging timeout! Continuing to search...')
            self.transition_to(self.SEARCHING_GATE)
            return

        # Apply depth control with forward motion (multitasking!)
        cmd = Twist()
        cmd.linear.z = depth_error * self.depth_gain
        cmd.linear.x = 0.3  # Move forward while submerging
        cmd.linear.y = 0.0
        cmd.angular.z = 0.0

        self.cmd_vel_pub.publish(cmd)
    
    def handle_searching(self):
        """Search for gate while maintaining depth"""
        elapsed = time.time() - self.state_start_time

        # Check if gate detected
        if self.gate_detected:
            self.get_logger().info('✅ Gate detected! Transitioning to alignment...')
            self.transition_to(self.ALIGNING_TO_GATE)
            return

        # Log search status every 2 seconds
        if int(elapsed) % 2 == 0:
            self.get_logger().info(
                f'🔍 Searching for gate... Time: {elapsed:.1f}s, '
                f'Depth: {self.current_depth:.2f}m'
            )

        # Timeout after 45 seconds (reduced from 60)
        if elapsed > 45.0:
            self.get_logger().error('❌ Gate search timeout!')
            self.transition_to(self.COMPLETED)
            return

        # Search pattern: move forward faster without rotation initially
        # Rotation only after 10 seconds if not found
        cmd = Twist()
        cmd.linear.x = self.search_speed
        cmd.linear.y = 0.0

        # Add scanning rotation only after 10 seconds
        if elapsed > 10.0:
            cmd.angular.z = 0.15  # Slightly faster rotation
        else:
            cmd.angular.z = 0.0  # Go straight first

        # Maintain depth
        depth_error = self.target_depth - self.current_depth
        cmd.linear.z = depth_error * self.depth_gain

        self.cmd_vel_pub.publish(cmd)
    
    def handle_aligning(self):
        """Align with gate center"""
        elapsed = time.time() - self.state_start_time

        # Check if gate lost
        if not self.gate_detected:
            self.get_logger().warn('⚠️  Gate lost! Returning to search...')
            self.transition_to(self.SEARCHING_GATE)
            return

        # Check if aligned
        if abs(self.gate_alignment_error) < self.alignment_threshold:
            self.get_logger().info(
                f'✅ Aligned with gate! Error: {self.gate_alignment_error:.3f}'
            )
            self.transition_to(self.APPROACHING_GATE)
            return

        # Reduced timeout
        if elapsed > 10.0:
            self.get_logger().warn('⚠️  Alignment timeout! Proceeding to approach...')
            self.transition_to(self.APPROACHING_GATE)
            return

        # Log alignment status every 0.5 seconds
        if int(elapsed * 2) % 2 == 0:
            self.get_logger().info(
                f'🎯 Aligning: Error={self.gate_alignment_error:.3f}, '
                f'Distance={self.gate_distance:.2f}m',
                throttle_duration_sec=0.5
            )

        # Apply alignment correction with moderate forward motion
        cmd = Twist()
        cmd.linear.x = 0.25  # Increased forward motion during alignment
        cmd.linear.y = 0.0
        cmd.angular.z = -self.gate_alignment_error * self.yaw_gain

        # Maintain depth
        depth_error = self.target_depth - self.current_depth
        cmd.linear.z = depth_error * self.depth_gain

        self.cmd_vel_pub.publish(cmd)
    
    def handle_approaching(self):
        """Approach gate while maintaining alignment"""
        elapsed = time.time() - self.state_start_time

        # Check if gate lost
        if not self.gate_detected:
            self.get_logger().warn('⚠️  Gate lost during approach! Returning to search...')
            self.transition_to(self.SEARCHING_GATE)
            return

        # Check if close enough to pass
        if self.gate_distance < self.passing_distance and self.gate_distance > 0.1:
            self.get_logger().info(
                f'✅ Close enough to gate! Distance: {self.gate_distance:.2f}m'
            )
            self.transition_to(self.PASSING_THROUGH)
            return

        # Also transition if we've been approaching for too long (likely passed it)
        if elapsed > 15.0:
            self.get_logger().warn('⚠️  Approach timeout - likely at gate, initiating pass!')
            self.transition_to(self.PASSING_THROUGH)
            return

        # Log approach status every 0.5 seconds
        if int(elapsed * 2) % 2 == 0:
            self.get_logger().info(
                f'➡️  Approaching: Distance={self.gate_distance:.2f}m, '
                f'Alignment={self.gate_alignment_error:.3f}',
                throttle_duration_sec=0.5
            )

        # Approach with alignment correction
        cmd = Twist()
        cmd.linear.x = self.approach_speed
        cmd.linear.y = 0.0

        # Moderate alignment correction during approach
        if abs(self.gate_alignment_error) > self.alignment_threshold:
            cmd.angular.z = -self.gate_alignment_error * self.yaw_gain * 0.6
        else:
            cmd.angular.z = 0.0

        # Maintain depth
        depth_error = self.target_depth - self.current_depth
        cmd.linear.z = depth_error * self.depth_gain

        self.cmd_vel_pub.publish(cmd)
    
    def handle_passing(self):
        """Pass through gate at maximum speed"""

        # Initialize passing timer
        if self.passing_start_time is None:
            self.passing_start_time = time.time()
            self.get_logger().info('🚀 PASSING THROUGH GATE - FULL SPEED AHEAD!')

        elapsed = time.time() - self.passing_start_time

        # Check if passed through
        if elapsed > self.passing_duration:
            self.get_logger().info(
                f'✅ GATE PASSED SUCCESSFULLY! Duration: {elapsed:.1f}s'
            )
            self.transition_to(self.COMPLETED)
            return

        # Log passing status every second
        if int(elapsed) % 1 == 0:
            self.get_logger().info(
                f'🚀 Passing... {elapsed:.1f}s / {self.passing_duration:.1f}s',
                throttle_duration_sec=1.0
            )

        # Maximum speed ahead! No corrections, just go!
        cmd = Twist()
        cmd.linear.x = self.passing_speed
        cmd.linear.y = 0.0
        cmd.angular.z = 0.0

        # Maintain depth
        depth_error = self.target_depth - self.current_depth
        cmd.linear.z = depth_error * self.depth_gain

        self.cmd_vel_pub.publish(cmd)
    
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
            self.ALIGNING_TO_GATE: 'ALIGNING_TO_GATE',
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