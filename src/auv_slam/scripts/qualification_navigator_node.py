#!/usr/bin/env python3
"""
FIXED Qualification Navigator - Active Rotation Stabilization
Key fix: Continuously corrects unwanted rotation caused by thruster mapper bug
Works the same way as gate_navigator_node.py
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
import time
import math


class QualificationNavigator(Node):
    def __init__(self):
        super().__init__('qualification_navigator')
        
        # State machine
        self.WAITING_TO_START = 0
        self.SUBMERGING = 1
        self.FORWARD_SEARCH = 2
        self.FORWARD_APPROACH = 3
        self.FORWARD_PASSING = 4
        self.U_TURN = 5
        self.REVERSE_SEARCH = 6
        self.REVERSE_APPROACH = 7
        self.REVERSE_PASSING = 8
        self.COMPLETED = 9
        
        self.state = self.WAITING_TO_START
        
        # Parameters - Depth control
        self.declare_parameter('target_depth', -0.8)
        self.declare_parameter('depth_tolerance', 0.15)
        self.declare_parameter('depth_correction_gain', 2.0)
        
        # CRITICAL FIX: Rotation stabilization parameters
        self.declare_parameter('rotation_stabilization_gain', 2.0)  # NEW
        
        # Search parameters
        self.declare_parameter('search_forward_speed', 0.4)
        self.declare_parameter('search_rotation_speed', 0.2)
        
        # Approach parameters
        self.declare_parameter('approach_speed', 0.5)
        self.declare_parameter('approach_yaw_gain', 1.5)
        self.declare_parameter('approach_threshold_distance', 2.5)
        
        # Passing parameters
        self.declare_parameter('passing_speed', 0.8)
        self.declare_parameter('passing_trigger_distance', 1.2)
        self.declare_parameter('passing_clearance', 1.5)
        
        # U-turn parameters
        self.declare_parameter('uturn_rotation_speed', 0.5)
        self.declare_parameter('uturn_target_angle', 3.14159)
        self.declare_parameter('uturn_angle_tolerance', 0.15)
        
        # Gate position tracking
        self.declare_parameter('gate_x_position', -2.5)
        
        # Get parameters
        self.target_depth = self.get_parameter('target_depth').value
        self.depth_tolerance = self.get_parameter('depth_tolerance').value
        self.depth_gain = self.get_parameter('depth_correction_gain').value
        
        # CRITICAL FIX: Get rotation stabilization gain
        self.rotation_stab_gain = self.get_parameter('rotation_stabilization_gain').value
        
        self.search_forward_speed = self.get_parameter('search_forward_speed').value
        self.search_rotation_speed = self.get_parameter('search_rotation_speed').value
        
        self.approach_speed = self.get_parameter('approach_speed').value
        self.approach_yaw_gain = self.get_parameter('approach_yaw_gain').value
        self.approach_threshold = self.get_parameter('approach_threshold_distance').value
        
        self.passing_speed = self.get_parameter('passing_speed').value
        self.passing_trigger = self.get_parameter('passing_trigger_distance').value
        self.passing_clearance = self.get_parameter('passing_clearance').value
        
        self.uturn_rotation_speed = self.get_parameter('uturn_rotation_speed').value
        self.uturn_target_angle = self.get_parameter('uturn_target_angle').value
        self.uturn_angle_tolerance = self.get_parameter('uturn_angle_tolerance').value
        
        self.gate_x_position = self.get_parameter('gate_x_position').value
        
        # State variables
        self.gate_detected = False
        self.partial_gate = False
        self.alignment_error = 0.0
        self.estimated_distance = 999.0
        self.current_depth = 0.0
        self.confidence = 0.0
        
        # Position and orientation
        self.current_position = None
        self.current_yaw = 0.0
        
        # CRITICAL FIX: Track target yaw for stabilization
        self.target_yaw = 0.0  # Will be set to initial yaw
        self.initial_yaw_set = False
        
        self.uturn_start_yaw = 0.0
        
        # Passing tracking
        self.forward_pass_start_x = None
        self.reverse_pass_start_x = None
        
        # Timing
        self.state_start_time = time.time()
        self.mission_start_time = None
        self.forward_pass_time = None
        self.qualification_points = 0
        
        # Gate lost tracking
        self.gate_lost_time = 0.0
        self.gate_lost_timeout = 3.0
        
        # Subscriptions
        self.gate_detected_sub = self.create_subscription(
            Bool, '/qualification/gate_detected', self.gate_detected_callback, 10)
        self.alignment_sub = self.create_subscription(
            Float32, '/qualification/alignment_error', self.alignment_callback, 10)
        self.distance_sub = self.create_subscription(
            Float32, '/qualification/estimated_distance', self.distance_callback, 10)
        self.confidence_sub = self.create_subscription(
            Float32, '/qualification/confidence', self.confidence_callback, 10)
        self.partial_gate_sub = self.create_subscription(
            Bool, '/qualification/partial_detection', self.partial_gate_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/ground_truth/odom', self.odom_callback, 10)
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/rp2040/cmd_vel', 10)
        self.state_pub = self.create_publisher(String, '/qualification/state', 10)
        self.points_pub = self.create_publisher(Float32, '/qualification/points', 10)
        
        # Control timer
        self.control_timer = self.create_timer(0.05, self.control_loop)
        
        self.get_logger().info('='*70)
        self.get_logger().info('✅ QUALIFICATION NAVIGATOR INITIALIZED')
        self.get_logger().info('='*70)
        self.get_logger().info('   Task: Forward pass → U-turn → Reverse pass')
        self.get_logger().info('   Points: 1st pass = 1 point, 2nd pass = 2 points total')
        self.get_logger().info('   FIX: Active rotation stabilization enabled')
        self.get_logger().info('   Waiting to start...')
        self.get_logger().info('='*70)
    
    def gate_detected_callback(self, msg: Bool):
        was_detected = self.gate_detected
        self.gate_detected = msg.data
        
        if not was_detected and self.gate_detected:
            self.gate_lost_time = 0.0
        elif was_detected and not self.gate_detected:
            self.gate_lost_time = time.time()
    
    def alignment_callback(self, msg: Float32):
        self.alignment_error = msg.data
    
    def distance_callback(self, msg: Float32):
        self.estimated_distance = msg.data
    
    def confidence_callback(self, msg: Float32):
        self.confidence = msg.data
    
    def partial_gate_callback(self, msg: Bool):
        self.partial_gate = msg.data
    
    def odom_callback(self, msg: Odometry):
        self.current_depth = msg.pose.pose.position.z
        self.current_position = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        )
        
        # Extract yaw from quaternion
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)
        
        # CRITICAL FIX: Set initial target yaw (facing forward toward gate)
        if not self.initial_yaw_set:
            self.target_yaw = self.current_yaw
            self.initial_yaw_set = True
            self.get_logger().info(f'🎯 Initial yaw locked: {math.degrees(self.target_yaw):.1f}°')
    
    def control_loop(self):
        cmd = Twist()
        
        # Depth control (active in all states except WAITING)
        if self.state != self.WAITING_TO_START:
            depth_error = self.target_depth - self.current_depth
            
            if abs(depth_error) < self.depth_tolerance:
                cmd.linear.z = 0.0
            else:
                cmd.linear.z = depth_error * self.depth_gain
                cmd.linear.z = max(-1.0, min(cmd.linear.z, 1.0))
        
        # State machine
        if self.state == self.WAITING_TO_START:
            cmd = self.waiting_to_start(cmd)
        elif self.state == self.SUBMERGING:
            cmd = self.submerging_behavior(cmd)
        elif self.state == self.FORWARD_SEARCH:
            cmd = self.forward_search_behavior(cmd)
        elif self.state == self.FORWARD_APPROACH:
            cmd = self.forward_approach_behavior(cmd)
        elif self.state == self.FORWARD_PASSING:
            cmd = self.forward_passing_behavior(cmd)
        elif self.state == self.U_TURN:
            cmd = self.uturn_behavior(cmd)
        elif self.state == self.REVERSE_SEARCH:
            cmd = self.reverse_search_behavior(cmd)
        elif self.state == self.REVERSE_APPROACH:
            cmd = self.reverse_approach_behavior(cmd)
        elif self.state == self.REVERSE_PASSING:
            cmd = self.reverse_passing_behavior(cmd)
        elif self.state == self.COMPLETED:
            cmd = self.completed_behavior(cmd)
        
        self.cmd_vel_pub.publish(cmd)
        
        # Publish state and points
        state_msg = String()
        state_msg.data = self.get_state_name()
        self.state_pub.publish(state_msg)
        
        points_msg = Float32()
        points_msg.data = float(self.qualification_points)
        self.points_pub.publish(points_msg)
    
    def waiting_to_start(self, cmd: Twist) -> Twist:
        """Wait at starting zone, ready to begin"""
        cmd.linear.x = 0.0
        cmd.linear.y = 0.0
        cmd.linear.z = 0.0
        cmd.angular.z = 0.0
        
        # Auto-start after 3 seconds
        elapsed = time.time() - self.state_start_time
        if elapsed > 3.0:
            self.get_logger().info('🚀 STARTING QUALIFICATION RUN!')
            self.mission_start_time = time.time()
            self.transition_to(self.SUBMERGING)
        
        return cmd
    
    def submerging_behavior(self, cmd: Twist) -> Twist:
        """
        CRITICAL FIX: Active rotation stabilization during descent
        This prevents unwanted spinning caused by thruster mapper bug
        """
        # Check if reached target depth
        depth_error = abs(self.target_depth - self.current_depth)
        
        if depth_error < self.depth_tolerance:
            self.get_logger().info(
                f'✓ Reached operational depth: {self.current_depth:.2f}m'
            )
            self.transition_to(self.FORWARD_SEARCH)
            return cmd
        
        # CRITICAL FIX: Apply rotation stabilization (like gate navigator does)
        yaw_error = self.normalize_angle(self.target_yaw - self.current_yaw)
        cmd.angular.z = yaw_error * self.rotation_stab_gain
        
        # No forward movement during submerge
        cmd.linear.x = 0.0
        
        elapsed = time.time() - self.state_start_time
        if int(elapsed) % 2 == 0:
            self.get_logger().info(
                f'⬇️ Submerging... depth={self.current_depth:.2f}m target={self.target_depth:.2f}m '
                f'yaw_err={math.degrees(yaw_error):.1f}°',
                throttle_duration_sec=1.9
            )
        
        # Timeout check
        if elapsed > 15.0:
            self.get_logger().warn('⚠️ Submerge timeout - proceeding anyway')
            self.transition_to(self.FORWARD_SEARCH)
        
        return cmd
    
    def forward_search_behavior(self, cmd: Twist) -> Twist:
        """Search for gate while moving forward"""
        if self.gate_detected:
            self.get_logger().info('🎯 Gate detected - Starting forward approach')
            self.transition_to(self.FORWARD_APPROACH)
            return cmd
        
        # Search pattern: move forward with sweep
        elapsed = time.time() - self.state_start_time
        sweep_period = 8.0
        sweep_phase = (elapsed % sweep_period) / sweep_period
        
        cmd.linear.x = self.search_forward_speed
        
        if sweep_phase < 0.5:
            cmd.angular.z = self.search_rotation_speed
        else:
            cmd.angular.z = -self.search_rotation_speed
        
        if int(elapsed) % 3 == 0:
            direction = "LEFT" if sweep_phase < 0.5 else "RIGHT"
            self.get_logger().info(
                f'🔍 Forward search ({direction})... {elapsed:.0f}s',
                throttle_duration_sec=2.9
            )
        
        return cmd
    
    def forward_approach_behavior(self, cmd: Twist) -> Twist:
        """Approach gate from forward direction"""
        if not self.gate_detected:
            if self.gate_lost_time > 0.0:
                lost_duration = time.time() - self.gate_lost_time
                if lost_duration > self.gate_lost_timeout:
                    self.get_logger().warn('❌ Gate lost - returning to search')
                    self.transition_to(self.FORWARD_SEARCH)
                else:
                    # CRITICAL FIX: Maintain heading even when gate lost
                    cmd.linear.x = 0.2
                    yaw_error = self.normalize_angle(self.target_yaw - self.current_yaw)
                    cmd.angular.z = yaw_error * self.rotation_stab_gain
            return cmd
        
        # Check if close enough to commit to passing
        if self.estimated_distance < self.passing_trigger:
            self.get_logger().info(
                f'🚀 Committing to forward pass at {self.estimated_distance:.2f}m'
            )
            self.forward_pass_start_x = self.current_position[0]
            self.transition_to(self.FORWARD_PASSING)
            return cmd
        
        # Approach with alignment
        cmd.linear.x = self.approach_speed
        cmd.angular.z = -self.alignment_error * self.approach_yaw_gain
        
        self.get_logger().info(
            f'➡️ Forward approach: dist={self.estimated_distance:.1f}m, '
            f'align={self.alignment_error:+.3f}',
            throttle_duration_sec=0.5
        )
        
        return cmd
    
    def forward_passing_behavior(self, cmd: Twist) -> Twist:
        """Pass through gate in forward direction"""
        if self.forward_pass_start_x is None:
            self.forward_pass_start_x = self.current_position[0]
        
        # Check if cleared gate
        if self.current_position:
            current_x = self.current_position[0]
            distance_traveled = abs(current_x - self.forward_pass_start_x)
            
            if current_x > (self.gate_x_position + self.passing_clearance):
                self.qualification_points = 1
                self.forward_pass_time = time.time() - self.mission_start_time
                
                self.get_logger().info('='*70)
                self.get_logger().info('✅ FORWARD PASS COMPLETE - 1 POINT EARNED!')
                self.get_logger().info(f'   Time: {self.forward_pass_time:.2f}s')
                self.get_logger().info('   Starting U-turn...')
                self.get_logger().info('='*70)
                
                self.uturn_start_yaw = self.current_yaw
                self.transition_to(self.U_TURN)
                return cmd
            
            self.get_logger().info(
                f'🚀 FORWARD PASSING: X={current_x:.2f}m, traveled={distance_traveled:.2f}m',
                throttle_duration_sec=0.4
            )
        
        # Full speed through gate
        cmd.linear.x = self.passing_speed
        
        # CRITICAL FIX: Maintain heading during pass
        yaw_error = self.normalize_angle(self.target_yaw - self.current_yaw)
        cmd.angular.z = yaw_error * self.rotation_stab_gain * 0.5
        
        return cmd
    
    def uturn_behavior(self, cmd: Twist) -> Twist:
        """Perform 180-degree turn"""
        # Calculate angle turned
        angle_turned = abs(self.normalize_angle(self.current_yaw - self.uturn_start_yaw))
        angle_remaining = self.uturn_target_angle - angle_turned
        
        if angle_remaining < self.uturn_angle_tolerance:
            # CRITICAL FIX: Update target yaw after U-turn
            self.target_yaw = self.normalize_angle(self.target_yaw + math.pi)
            
            self.get_logger().info(
                f'✓ U-turn complete! Turned {math.degrees(angle_turned):.1f}°, '
                f'new target yaw: {math.degrees(self.target_yaw):.1f}°'
            )
            self.transition_to(self.REVERSE_SEARCH)
            return cmd
        
        # Rotate in place
        cmd.linear.x = 0.1
        cmd.angular.z = self.uturn_rotation_speed
        
        elapsed = time.time() - self.state_start_time
        if int(elapsed * 2) % 2 == 0:
            self.get_logger().info(
                f'🔄 U-turn: {math.degrees(angle_turned):.0f}° / 180°',
                throttle_duration_sec=0.4
            )
        
        # Timeout check
        if elapsed > 20.0:
            self.get_logger().warn('⚠️ U-turn timeout - proceeding to reverse search')
            self.target_yaw = self.normalize_angle(self.target_yaw + math.pi)
            self.transition_to(self.REVERSE_SEARCH)
        
        return cmd
    
    def reverse_search_behavior(self, cmd: Twist) -> Twist:
        """Search for gate from reverse direction"""
        if self.gate_detected:
            self.get_logger().info('🎯 Gate detected - Starting reverse approach')
            self.transition_to(self.REVERSE_APPROACH)
            return cmd
        
        # Search pattern
        elapsed = time.time() - self.state_start_time
        sweep_period = 8.0
        sweep_phase = (elapsed % sweep_period) / sweep_period
        
        cmd.linear.x = self.search_forward_speed
        
        if sweep_phase < 0.5:
            cmd.angular.z = self.search_rotation_speed
        else:
            cmd.angular.z = -self.search_rotation_speed
        
        if int(elapsed) % 3 == 0:
            direction = "LEFT" if sweep_phase < 0.5 else "RIGHT"
            self.get_logger().info(
                f'🔍 Reverse search ({direction})... {elapsed:.0f}s',
                throttle_duration_sec=2.9
            )
        
        return cmd
    
    def reverse_approach_behavior(self, cmd: Twist) -> Twist:
        """Approach gate from reverse direction"""
        if not self.gate_detected:
            if self.gate_lost_time > 0.0:
                lost_duration = time.time() - self.gate_lost_time
                if lost_duration > self.gate_lost_timeout:
                    self.get_logger().warn('❌ Gate lost - returning to search')
                    self.transition_to(self.REVERSE_SEARCH)
                else:
                    # CRITICAL FIX: Maintain heading
                    cmd.linear.x = 0.2
                    yaw_error = self.normalize_angle(self.target_yaw - self.current_yaw)
                    cmd.angular.z = yaw_error * self.rotation_stab_gain
            return cmd
        
        # Check if close enough to commit
        if self.estimated_distance < self.passing_trigger:
            self.get_logger().info(
                f'🚀 Committing to reverse pass at {self.estimated_distance:.2f}m'
            )
            self.reverse_pass_start_x = self.current_position[0]
            self.transition_to(self.REVERSE_PASSING)
            return cmd
        
        # Approach with alignment
        cmd.linear.x = self.approach_speed
        cmd.angular.z = -self.alignment_error * self.approach_yaw_gain
        
        self.get_logger().info(
            f'⬅️ Reverse approach: dist={self.estimated_distance:.1f}m, '
            f'align={self.alignment_error:+.3f}',
            throttle_duration_sec=0.5
        )
        
        return cmd
    
    def reverse_passing_behavior(self, cmd: Twist) -> Twist:
        """Pass through gate in reverse direction"""
        if self.reverse_pass_start_x is None:
            self.reverse_pass_start_x = self.current_position[0]
        
        # Check if cleared gate (now going negative X)
        if self.current_position:
            current_x = self.current_position[0]
            distance_traveled = abs(current_x - self.reverse_pass_start_x)
            
            if current_x < (self.gate_x_position - self.passing_clearance):
                self.qualification_points = 2
                total_time = time.time() - self.mission_start_time
                
                self.get_logger().info('='*70)
                self.get_logger().info('🎉 REVERSE PASS COMPLETE - 2 POINTS EARNED!')
                self.get_logger().info('='*70)
                self.get_logger().info('✅ QUALIFICATION RUN COMPLETE!')
                self.get_logger().info(f'   Total Points: {self.qualification_points}')
                self.get_logger().info(f'   Total Time: {total_time:.2f}s')
                self.get_logger().info(f'   Forward pass: {self.forward_pass_time:.2f}s')
                self.get_logger().info(f'   Reverse pass: {total_time - self.forward_pass_time:.2f}s')
                self.get_logger().info('='*70)
                
                self.transition_to(self.COMPLETED)
                return cmd
            
            self.get_logger().info(
                f'🚀 REVERSE PASSING: X={current_x:.2f}m, traveled={distance_traveled:.2f}m',
                throttle_duration_sec=0.4
            )
        
        # Full speed through gate
        cmd.linear.x = self.passing_speed
        
        # CRITICAL FIX: Maintain heading
        yaw_error = self.normalize_angle(self.target_yaw - self.current_yaw)
        cmd.angular.z = yaw_error * self.rotation_stab_gain * 0.5
        
        return cmd
    
    def completed_behavior(self, cmd: Twist) -> Twist:
        """Mission complete - stop all movement"""
        cmd.linear.x = 0.0
        cmd.linear.y = 0.0
        cmd.linear.z = 0.0
        cmd.angular.z = 0.0
        
        return cmd
    
    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def transition_to(self, new_state: int):
        """Transition to new state"""
        old_name = self.get_state_name()
        self.state = new_state
        self.state_start_time = time.time()
        new_name = self.get_state_name()
        
        self.get_logger().info(f'🔄 STATE TRANSITION: {old_name} → {new_name}')
    
    def get_state_name(self) -> str:
        """Get human-readable state name"""
        names = {
            self.WAITING_TO_START: 'WAITING_TO_START',
            self.SUBMERGING: 'SUBMERGING',
            self.FORWARD_SEARCH: 'FORWARD_SEARCH',
            self.FORWARD_APPROACH: 'FORWARD_APPROACH',
            self.FORWARD_PASSING: 'FORWARD_PASSING',
            self.U_TURN: 'U_TURN',
            self.REVERSE_SEARCH: 'REVERSE_SEARCH',
            self.REVERSE_APPROACH: 'REVERSE_APPROACH',
            self.REVERSE_PASSING: 'REVERSE_PASSING',
            self.COMPLETED: 'COMPLETED'
        }
        return names.get(self.state, 'UNKNOWN')


def main(args=None):
    rclpy.init(args=args)
    node = QualificationNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop_cmd = Twist()
        node.cmd_vel_pub.publish(stop_cmd)
        node.get_logger().info('Qualification Navigator shutting down')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()