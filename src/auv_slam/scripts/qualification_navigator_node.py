#!/usr/bin/env python3
"""
SAUVC QUALIFICATION NAVIGATOR - FIXED DEPTH CONTROL
Complete qualification task with stable depth control:
1. Maintain constant depth during search/approach
2. Only adjust height when within 3m for alignment
3. U-turn and repeat for second pass
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
        super().__init__('qualification_navigator_node')
        
        # State machine
        self.IDLE = 0
        self.SEARCHING = 1
        self.APPROACHING = 2
        self.ALIGNING = 3
        self.FINAL_APPROACH = 4
        self.PASSING = 5
        self.UTURN = 6
        self.COMPLETED = 7
        
        self.state = self.SEARCHING
        self.passes_completed = 0
        
        # Parameters
        self.declare_parameter('target_depth', -0.8)
        self.declare_parameter('depth_correction_gain', 2.0)
        
        # Search parameters
        self.declare_parameter('search_forward_speed', 0.4)
        self.declare_parameter('search_rotation_speed', 0.2)
        
        # Approach parameters
        self.declare_parameter('approach_start_distance', 10.0)
        self.declare_parameter('approach_stop_distance', 3.0)
        self.declare_parameter('approach_speed', 0.6)
        self.declare_parameter('approach_yaw_gain', 1.0)
        
        # Alignment parameters
        self.declare_parameter('alignment_distance', 3.0)
        self.declare_parameter('alignment_threshold', 0.10)
        self.declare_parameter('alignment_max_time', 20.0)
        self.declare_parameter('alignment_yaw_gain', 3.0)
        
        # Final approach parameters
        self.declare_parameter('final_approach_start', 3.0)
        self.declare_parameter('final_approach_speed', 0.5)
        self.declare_parameter('final_approach_threshold', 0.15)
        
        # Passing parameters
        self.declare_parameter('passing_trigger_distance', 1.2)
        self.declare_parameter('passing_speed', 1.0)
        self.declare_parameter('passing_duration', 7.0)
        self.declare_parameter('gate_width', 1.5)
        
        # U-turn parameters
        self.declare_parameter('uturn_rotation_speed', 0.6)
        self.declare_parameter('uturn_forward_speed', 0.2)
        
        # Gate position
        self.declare_parameter('gate_x_position', 0.0)
        self.declare_parameter('gate_clearance_distance', 0.5)
        
        self.target_depth = self.get_parameter('target_depth').value
        self.depth_gain = self.get_parameter('depth_correction_gain').value
        self.search_forward_speed = self.get_parameter('search_forward_speed').value
        self.search_rotation_speed = self.get_parameter('search_rotation_speed').value
        
        self.approach_start_distance = self.get_parameter('approach_start_distance').value
        self.approach_stop_distance = self.get_parameter('approach_stop_distance').value
        self.approach_speed = self.get_parameter('approach_speed').value
        self.approach_yaw_gain = self.get_parameter('approach_yaw_gain').value
        
        self.alignment_distance = self.get_parameter('alignment_distance').value
        self.alignment_threshold = self.get_parameter('alignment_threshold').value
        self.alignment_max_time = self.get_parameter('alignment_max_time').value
        self.alignment_yaw_gain = self.get_parameter('alignment_yaw_gain').value
        
        self.final_approach_start = self.get_parameter('final_approach_start').value
        self.final_approach_speed = self.get_parameter('final_approach_speed').value
        self.final_approach_threshold = self.get_parameter('final_approach_threshold').value
        
        self.passing_trigger_distance = self.get_parameter('passing_trigger_distance').value
        self.passing_speed = self.get_parameter('passing_speed').value
        self.passing_duration = self.get_parameter('passing_duration').value
        self.gate_width = self.get_parameter('gate_width').value
        
        self.uturn_rotation_speed = self.get_parameter('uturn_rotation_speed').value
        self.uturn_forward_speed = self.get_parameter('uturn_forward_speed').value
        
        self.gate_x_position = self.get_parameter('gate_x_position').value
        self.gate_clearance_distance = self.get_parameter('gate_clearance_distance').value
        
        # State variables
        self.gate_detected = False
        self.partial_gate = False
        self.alignment_error = 0.0
        self.estimated_distance = 999.0
        self.current_depth = 0.0
        self.frame_position = 0.0
        self.confidence = 0.0
        self.gate_center_y = 0.0  # For visual height alignment
        
        # Position tracking
        self.current_position = None
        self.current_yaw = 0.0
        self.passing_start_position = None
        self.uturn_start_yaw = 0.0
        
        # Timing
        self.gate_lost_time = 0.0
        self.gate_lost_timeout = 3.0
        self.alignment_start_time = 0.0
        self.state_start_time = time.time()
        
        self.mission_start_time = time.time()
        self.pass1_complete_time = None
        self.pass2_complete_time = None
        
        # Camera parameters for visual depth alignment
        self.img_height = 960  # From camera specs
        self.img_center_y = self.img_height / 2
        
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
        self.partial_gate_sub = self.create_subscription(
            Bool, '/gate/partial_detection', self.partial_gate_callback, 10)
        self.confidence_sub = self.create_subscription(
            Float32, '/gate/detection_confidence', self.confidence_callback, 10)
        self.gate_center_sub = self.create_subscription(
            Point, '/gate/center_point', self.gate_center_callback, 10)
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/rp2040/cmd_vel', 10)
        self.state_pub = self.create_publisher(String, '/gate/navigation_state', 10)
        self.reverse_mode_pub = self.create_publisher(Bool, '/mission/reverse_mode', 10)
        
        self.control_timer = self.create_timer(0.05, self.control_loop)
        
        self.get_logger().info('='*70)
        self.get_logger().info('🏆 QUALIFICATION NAVIGATOR - STABLE DEPTH CONTROL')
        self.get_logger().info('='*70)
        self.get_logger().info('   Depth strategy:')
        self.get_logger().info('   - Constant depth during search/approach')
        self.get_logger().info('   - Height adjustment ONLY within 3m for alignment')
        self.get_logger().info('='*70)
    
    def gate_detected_callback(self, msg: Bool):
        was_detected = self.gate_detected
        self.gate_detected = msg.data
        
        if not was_detected and self.gate_detected:
            self.gate_lost_time = 0.0
        elif was_detected and not self.gate_detected:
            self.gate_lost_time = time.time()
    
    def frame_position_callback(self, msg: Float32):
        self.frame_position = msg.data
    
    def partial_gate_callback(self, msg: Bool):
        self.partial_gate = msg.data
    
    def confidence_callback(self, msg: Float32):
        self.confidence = msg.data
    
    def alignment_callback(self, msg: Float32):
        self.alignment_error = msg.data
    
    def distance_callback(self, msg: Float32):
        self.estimated_distance = msg.data
    
    def gate_center_callback(self, msg: Point):
        self.gate_center_y = msg.y
    
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
    
    def control_loop(self):
        cmd = Twist()
        
        # ============================================================
        # DEPTH CONTROL - CRITICAL FIX
        # ============================================================
        
        # Decide target depth based on state and distance
        target_depth = self.target_depth
        
        # ONLY adjust height when close enough AND in alignment states
        close_enough = self.estimated_distance < 3.0
        in_alignment_state = self.state in [self.ALIGNING, self.FINAL_APPROACH]
        
        if close_enough and in_alignment_state and self.gate_detected:
            # VISUAL SERVO: Adjust depth based on gate center pixel position
            pixel_error = self.gate_center_y - self.img_center_y
            
            # Scale: 100 pixels off = 0.1m adjustment
            # Negative pixel_error (gate too high) → need to go UP (less negative Z)
            # Positive pixel_error (gate too low) → need to go DOWN (more negative Z)
            depth_adjustment = (pixel_error / 1000.0)
            
            # Apply adjustment smoothly with limits
            target_depth = self.target_depth + depth_adjustment
            target_depth = max(-1.2, min(-0.4, target_depth))
            
            self.get_logger().info(
                f'📐 Height adjust: pixel_err={pixel_error:.0f}, '
                f'target_depth={target_depth:.2f}m',
                throttle_duration_sec=1.0
            )
        else:
            # Maintain constant depth - NO adjustment
            target_depth = self.target_depth
        
        # Calculate depth error
        depth_error = target_depth - self.current_depth
        
        # PROPER DEADBAND to prevent oscillation
        depth_deadband = 0.15  # 15cm deadband
        
        if abs(depth_error) < depth_deadband:
            # Within deadband - stop depth control
            cmd.linear.z = 0.0
        elif abs(depth_error) < 0.5:
            # Small error - gentle correction
            cmd.linear.z = depth_error * 0.8
            cmd.linear.z = max(-0.4, min(cmd.linear.z, 0.4))
        else:
            # Large error - stronger correction
            cmd.linear.z = depth_error * 1.2
            cmd.linear.z = max(-1.0, min(cmd.linear.z, 1.0))
        
        # ============================================================
        # STATE MACHINE
        # ============================================================
        
        if self.state == self.SEARCHING:
            cmd = self.searching_behavior(cmd)
        elif self.state == self.APPROACHING:
            cmd = self.approaching_behavior(cmd)
        elif self.state == self.ALIGNING:
            cmd = self.aligning_behavior(cmd)
        elif self.state == self.FINAL_APPROACH:
            cmd = self.final_approach_behavior(cmd)
        elif self.state == self.PASSING:
            cmd = self.passing_behavior(cmd)
        elif self.state == self.UTURN:
            cmd = self.uturn_behavior(cmd)
        elif self.state == self.COMPLETED:
            cmd = self.completed_behavior(cmd)
        
        self.cmd_vel_pub.publish(cmd)
        
        # Publish state and reverse mode
        state_msg = String()
        state_msg.data = self.get_state_name()
        self.state_pub.publish(state_msg)
        
        reverse_mode = Bool()
        reverse_mode.data = (self.passes_completed == 1)
        self.reverse_mode_pub.publish(reverse_mode)
    
    def searching_behavior(self, cmd: Twist) -> Twist:
        """Search for gate - CONSTANT DEPTH"""
        if self.gate_detected and self.estimated_distance < 999:
            pass_num = self.passes_completed + 1
            self.get_logger().info(
                f'🎯 Gate found (PASS {pass_num}) at {self.estimated_distance:.2f}m'
            )
            self.transition_to(self.APPROACHING)
            return cmd
        
        elapsed = time.time() - self.state_start_time
        sweep_period = 10.0
        sweep_phase = (elapsed % sweep_period) / sweep_period
        
        if sweep_phase < 0.5:
            cmd.angular.z = self.search_rotation_speed
        else:
            cmd.angular.z = -self.search_rotation_speed
        
        cmd.linear.x = self.search_forward_speed
        
        if int(elapsed) % 3 == 0:
            direction = "LEFT" if sweep_phase < 0.5 else "RIGHT"
            pass_num = self.passes_completed + 1
            self.get_logger().info(
                f'🔍 Searching (PASS {pass_num}, {direction})... {elapsed:.0f}s',
                throttle_duration_sec=2.9
            )
        
        return cmd
    
    def approaching_behavior(self, cmd: Twist) -> Twist:
        """Approach gate - CONSTANT DEPTH until 3m"""
        if not self.gate_detected:
            if self.gate_lost_time > 0.0:
                lost_duration = time.time() - self.gate_lost_time
                if lost_duration > self.gate_lost_timeout:
                    self.get_logger().warn('❌ Gate lost - returning to search')
                    self.transition_to(self.SEARCHING)
                else:
                    cmd.linear.x = 0.2
                    cmd.angular.z = 0.0
            return cmd
        
        # Stop at 3m to align
        if self.estimated_distance <= self.approach_stop_distance:
            self.get_logger().info(
                f'🛑 Reached {self.approach_stop_distance}m - ALIGNING'
            )
            self.transition_to(self.ALIGNING)
            return cmd
        
        # Casual approach with light correction
        cmd.linear.x = self.approach_speed
        cmd.angular.z = -self.frame_position * self.approach_yaw_gain
        
        self.get_logger().info(
            f'🚶 APPROACHING: dist={self.estimated_distance:.2f}m, '
            f'depth={self.current_depth:.2f}m (STABLE)',
            throttle_duration_sec=0.5
        )
        
        return cmd
    
    def aligning_behavior(self, cmd: Twist) -> Twist:
        """Align at 3m - NOW adjusting height"""
        if not self.gate_detected:
            if self.gate_lost_time > 0.0:
                lost_duration = time.time() - self.gate_lost_time
                if lost_duration > self.gate_lost_timeout:
                    self.get_logger().warn('❌ Gate lost - returning to search')
                    self.alignment_start_time = 0.0
                    self.transition_to(self.SEARCHING)
                else:
                    cmd.linear.x = 0.0
                    cmd.angular.z = 0.0
            return cmd
        
        # Initialize alignment timer
        if self.alignment_start_time == 0.0:
            self.alignment_start_time = time.time()
            self.get_logger().info(f'🎯 ALIGNING (with height adjustment)')
        
        alignment_elapsed = time.time() - self.alignment_start_time
        
        # Timeout check
        if alignment_elapsed > self.alignment_max_time:
            self.get_logger().warn('⏰ Alignment timeout - proceeding')
            self.alignment_start_time = 0.0
            self.transition_to(self.FINAL_APPROACH)
            return cmd
        
        # Check alignment quality
        is_well_aligned = abs(self.frame_position) < self.alignment_threshold
        has_confidence = self.confidence > 0.7 and not self.partial_gate
        
        if is_well_aligned and has_confidence:
            self.get_logger().info(
                f'✅ ALIGNED! (took {alignment_elapsed:.1f}s) → Final approach'
            )
            self.alignment_start_time = 0.0
            self.transition_to(self.FINAL_APPROACH)
            return cmd
        
        # Alignment strategy
        alignment_quality = abs(self.frame_position)
        
        if alignment_quality > 0.2:
            cmd.linear.x = 0.0
            cmd.angular.z = -self.frame_position * self.alignment_yaw_gain
            status = "MAJOR"
        elif alignment_quality > 0.1:
            cmd.linear.x = 0.1
            cmd.angular.z = -self.frame_position * self.alignment_yaw_gain * 0.8
            status = "MODERATE"
        else:
            cmd.linear.x = 0.15
            cmd.angular.z = -self.frame_position * self.alignment_yaw_gain * 0.5
            status = "FINE"
        
        self.get_logger().info(
            f'🔄 ALIGNING ({status}): pos={self.frame_position:+.3f}, '
            f't={alignment_elapsed:.1f}s',
            throttle_duration_sec=0.3
        )
        
        return cmd
    
    def final_approach_behavior(self, cmd: Twist) -> Twist:
        """Final approach - continuing height adjustment"""
        if not self.gate_detected:
            if self.gate_lost_time > 0.0:
                lost_duration = time.time() - self.gate_lost_time
                if lost_duration > self.gate_lost_timeout:
                    self.get_logger().warn('❌ Gate lost - returning to search')
                    self.transition_to(self.SEARCHING)
                else:
                    cmd.linear.x = 0.1
                    cmd.angular.z = 0.0
            return cmd
        
        # Check if close enough to commit
        if self.estimated_distance <= self.passing_trigger_distance:
            if abs(self.frame_position) < 0.2:
                self.get_logger().info(
                    f'🚀 COMMITTING at {self.estimated_distance:.2f}m'
                )
                self.passing_start_position = self.current_position
                self.transition_to(self.PASSING)
                return cmd
            else:
                # Emergency realignment
                self.get_logger().error(
                    f'🚨 Misaligned at trigger ({self.frame_position:+.3f})!'
                )
                cmd.linear.x = 0.0
                cmd.angular.z = -self.frame_position * self.alignment_yaw_gain * 2.0
                return cmd
        
        # Check drift
        if abs(self.frame_position) > self.final_approach_threshold:
            cmd.linear.x = self.final_approach_speed * 0.6
            cmd.angular.z = -self.frame_position * self.alignment_yaw_gain * 0.6
        else:
            cmd.linear.x = self.final_approach_speed
            cmd.angular.z = -self.frame_position * self.alignment_yaw_gain * 0.3
        
        self.get_logger().info(
            f'➡️ FINAL: dist={self.estimated_distance:.2f}m, '
            f'pos={self.frame_position:+.3f}',
            throttle_duration_sec=0.5
        )
        
        return cmd
    
    def passing_behavior(self, cmd: Twist) -> Twist:
        """Pass through gate - LOCK DEPTH"""
        elapsed = time.time() - self.state_start_time
        
        if elapsed > self.passing_duration:
            self.passes_completed += 1
            
            if self.passes_completed == 1:
                self.pass1_complete_time = time.time()
                self.get_logger().info('='*70)
                self.get_logger().info('✅ PASS 1 COMPLETE! (1 qualification point)')
                self.get_logger().info('   Starting U-turn...')
                self.get_logger().info('='*70)
                self.uturn_start_yaw = self.current_yaw
                self.transition_to(self.UTURN)
            else:
                self.pass2_complete_time = time.time()
                self.get_logger().info('='*70)
                self.get_logger().info('🏆 PASS 2 COMPLETE! (2 qualification points)')
                self.get_logger().info('   QUALIFICATION SUCCESSFUL!')
                self.get_logger().info('='*70)
                self.transition_to(self.COMPLETED)
            
            return cmd
        
        # Full speed straight - depth locked
        cmd.linear.x = self.passing_speed
        cmd.linear.y = 0.0
        cmd.angular.z = 0.0
        
        pass_num = self.passes_completed + 1
        self.get_logger().info(
            f'🚀 PASSING (PASS {pass_num}): {elapsed:.1f}s / {self.passing_duration:.1f}s',
            throttle_duration_sec=0.5
        )
        
        return cmd
    
    def uturn_behavior(self, cmd: Twist) -> Twist:
        """U-turn - CONSTANT DEPTH"""
        yaw_diff = self.current_yaw - self.uturn_start_yaw
        
        while yaw_diff > math.pi:
            yaw_diff -= 2 * math.pi
        while yaw_diff < -math.pi:
            yaw_diff += 2 * math.pi
        
        abs_yaw_diff = abs(yaw_diff)
        
        if abs_yaw_diff > (math.pi * 0.85):
            self.get_logger().info('✅ U-turn complete - searching for gate')
            self.transition_to(self.SEARCHING)
            return cmd
        
        cmd.linear.x = self.uturn_forward_speed
        cmd.angular.z = self.uturn_rotation_speed
        
        progress_pct = (abs_yaw_diff / math.pi) * 100
        self.get_logger().info(
            f'🔄 U-TURNING: {math.degrees(abs_yaw_diff):.0f}° / 180° ({progress_pct:.0f}%)',
            throttle_duration_sec=0.5
        )
        
        return cmd
    
    def completed_behavior(self, cmd: Twist) -> Twist:
        """Mission complete"""
        if self.mission_start_time:
            total_time = time.time() - self.mission_start_time
            
            self.get_logger().info('='*70)
            self.get_logger().info('🏆🏆🏆 QUALIFICATION COMPLETE! 🏆🏆🏆')
            self.get_logger().info(f'   Total time: {total_time:.2f}s')
            if self.pass1_complete_time:
                self.get_logger().info(f'   Pass 1: {(self.pass1_complete_time - self.mission_start_time):.2f}s')
            if self.pass2_complete_time:
                self.get_logger().info(f'   Pass 2: {(self.pass2_complete_time - self.mission_start_time):.2f}s')
            self.get_logger().info('   Qualification points: 2/2')
            self.get_logger().info('='*70)
            
            self.mission_start_time = None
        
        cmd.linear.x = 0.0
        cmd.linear.y = 0.0
        cmd.linear.z = 0.0
        cmd.angular.z = 0.0
        
        return cmd
    
    def transition_to(self, new_state: int):
        """Transition to new state"""
        old_name = self.get_state_name()
        self.state = new_state
        self.state_start_time = time.time()
        new_name = self.get_state_name()
        
        self.get_logger().info(f'🔄 STATE: {old_name} → {new_name}')
    
    def get_state_name(self) -> str:
        """Get human-readable state name"""
        names = {
            self.IDLE: 'IDLE',
            self.SEARCHING: 'SEARCHING',
            self.APPROACHING: 'APPROACHING',
            self.ALIGNING: 'ALIGNING',
            self.FINAL_APPROACH: 'FINAL_APPROACH',
            self.PASSING: 'PASSING',
            self.UTURN: 'UTURN',
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