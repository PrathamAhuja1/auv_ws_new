#!/usr/bin/env python3
"""
COMPLETE Qualification Navigator - SAUVC Compliant
- 0.10m clearance margin (reduced from 0.55m)
- Proper U-turn with forward motion (no spinning in place)
- Active depth control during U-turn (prevents surfacing)
- Post-U-turn alignment before reverse pass
- Gradual surfacing with forward motion
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import time
import math


class QualificationNavigator(Node):
    def __init__(self):
        super().__init__('qualification_navigator')
        
        # State machine
        self.SUBMERGING = 0
        self.SEARCHING = 1
        self.APPROACHING = 2
        self.ALIGNING = 3
        self.FINAL_APPROACH = 4
        self.PASSING = 5
        self.CLEARING = 6
        self.UTURN = 7
        self.POST_UTURN_ALIGN = 8  # NEW: Align with gate after U-turn
        self.REVERSE_SEARCHING = 9
        self.REVERSE_APPROACHING = 10
        self.REVERSE_ALIGNING = 11
        self.REVERSE_FINAL_APPROACH = 12
        self.REVERSE_PASSING = 13
        self.REVERSE_CLEARING = 14
        self.SURFACING = 15
        self.COMPLETED = 16
        
        self.state = self.SUBMERGING
        
        # Gate position in world (from world file)
        self.gate_x_position = 0.0
        self.mission_depth = -0.8
        
        # CRITICAL: AUV dimensions
        self.auv_length = 0.46  # meters
        
        # CRITICAL FIX: Clearance = AUV length + 0.10m extra
        # This ensures AUV's back passes gate + travels 0.10m more
        self.clearance_margin = 0.10  # Extra distance after back passes (REDUCED)
        
        # Forward pass: AUV must reach X > (gate_x + auv_length + 0.10)
        self.forward_clearance_x = self.gate_x_position + self.auv_length + self.clearance_margin
        
        # Reverse pass: AUV must reach X < (gate_x - auv_length - 0.10)
        self.reverse_clearance_x = self.gate_x_position - self.auv_length - self.clearance_margin
        
        # Parameters
        self.declare_parameter('search_forward_speed', 0.4)
        self.declare_parameter('approach_speed', 0.6)
        self.declare_parameter('approach_stop_distance', 3.0)
        self.declare_parameter('alignment_distance', 3.0)
        self.declare_parameter('alignment_threshold', 0.06)
        self.declare_parameter('alignment_max_time', 20.0)
        self.declare_parameter('final_approach_speed', 0.5)
        self.declare_parameter('passing_trigger_distance', 1.0)
        self.declare_parameter('passing_speed', 1.0)
        
        # U-turn parameters
        self.declare_parameter('uturn_forward_speed', 0.4)
        self.declare_parameter('uturn_angular_speed', 0.5)
        self.declare_parameter('uturn_depth', -0.8)  # Stay at mission depth
        
        self.search_forward_speed = self.get_parameter('search_forward_speed').value
        self.approach_speed = self.get_parameter('approach_speed').value
        self.approach_stop_distance = self.get_parameter('approach_stop_distance').value
        self.alignment_distance = self.get_parameter('alignment_distance').value
        self.alignment_threshold = self.get_parameter('alignment_threshold').value
        self.alignment_max_time = self.get_parameter('alignment_max_time').value
        self.final_approach_speed = self.get_parameter('final_approach_speed').value
        self.passing_trigger_distance = self.get_parameter('passing_trigger_distance').value
        self.passing_speed = self.get_parameter('passing_speed').value
        self.gate_width = 1.5
        
        self.uturn_forward_speed = self.get_parameter('uturn_forward_speed').value
        self.uturn_angular_speed = self.get_parameter('uturn_angular_speed').value
        self.uturn_depth = self.get_parameter('uturn_depth').value
        
        # State variables
        self.gate_detected = False
        self.alignment_error = 0.0
        self.estimated_distance = 999.0
        self.frame_position = 0.0
        self.confidence = 0.0
        self.current_depth = 0.0
        self.current_position = None
        self.current_yaw = 0.0
        
        self.passing_start_position = None
        self.alignment_start_time = 0.0
        self.state_start_time = time.time()
        self.uturn_start_yaw = 0.0
        self.uturn_start_time = 0.0
        self.uturn_start_x = 0.0  # NEW: Track X position at start of U-turn
        self.reverse_mode = False
        
        self.first_pass_complete = False
        self.second_pass_complete = False
        
        # Timing
        self.gate_lost_time = 0.0
        self.gate_lost_timeout = 3.0
        self.mission_start_time = time.time()
        
        # Subscriptions
        self.create_subscription(Bool, '/qualification/gate_detected', self.gate_cb, 10)
        self.create_subscription(Float32, '/qualification/alignment_error', self.align_cb, 10)
        self.create_subscription(Float32, '/qualification/estimated_distance', self.dist_cb, 10)
        self.create_subscription(Float32, '/qualification/frame_position', self.frame_pos_cb, 10)
        self.create_subscription(Float32, '/qualification/confidence', self.conf_cb, 10)
        self.create_subscription(Odometry, '/ground_truth/odom', self.odom_cb, 10)
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/rp2040/cmd_vel', 10)
        self.state_pub = self.create_publisher(String, '/qualification/state', 10)
        self.reverse_mode_pub = self.create_publisher(Bool, '/mission/reverse_mode', 10)
        
        self.create_timer(0.05, self.control_loop)
        
        self.get_logger().info('='*70)
        self.get_logger().info('✅ QUALIFICATION NAVIGATOR - SAUVC COMPLIANT')
        self.get_logger().info('='*70)
        self.get_logger().info(f'   AUV length: {self.auv_length}m')
        self.get_logger().info(f'   Gate at X={self.gate_x_position}m')
        self.get_logger().info(f'   Clearance margin: {self.clearance_margin}m (REDUCED for efficiency)')
        self.get_logger().info(f'   Forward clearance: X > {self.forward_clearance_x:.2f}m')
        self.get_logger().info(f'   Reverse clearance: X < {self.reverse_clearance_x:.2f}m')
        self.get_logger().info('   NO TIMEOUT - Will run until completion')
        self.get_logger().info('   IMPROVED: Proper U-turn with forward motion')
        self.get_logger().info('   IMPROVED: Post-U-turn alignment before reverse pass')
        self.get_logger().info('   IMPROVED: Gradual surfacing with forward motion')
        self.get_logger().info('='*70)
    
    def gate_cb(self, msg: Bool):
        self.gate_detected = msg.data
    
    def align_cb(self, msg: Float32):
        self.alignment_error = msg.data
    
    def dist_cb(self, msg: Float32):
        self.estimated_distance = msg.data
    
    def frame_pos_cb(self, msg: Float32):
        self.frame_position = msg.data
    
    def conf_cb(self, msg: Float32):
        self.confidence = msg.data
    
    def odom_cb(self, msg: Odometry):
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
        
        # Gentle depth control during passage, normal otherwise
        if self.state == self.PASSING or self.state == self.REVERSE_PASSING:
            cmd.linear.z = self.gentle_depth_control(self.mission_depth)
        else:
            cmd.linear.z = self.depth_control(self.mission_depth)
        
        # State machine
        if self.state == self.SUBMERGING:
            cmd = self.submerge(cmd)
        elif self.state == self.SEARCHING:
            cmd = self.searching(cmd)
        elif self.state == self.APPROACHING:
            cmd = self.approaching(cmd)
        elif self.state == self.ALIGNING:
            cmd = self.aligning(cmd)
        elif self.state == self.FINAL_APPROACH:
            cmd = self.final_approach(cmd)
        elif self.state == self.PASSING:
            cmd = self.passing(cmd)
        elif self.state == self.CLEARING:
            cmd = self.clearing(cmd)
        elif self.state == self.UTURN:
            cmd = self.uturn(cmd)
        elif self.state == self.POST_UTURN_ALIGN:
            cmd = self.post_uturn_align(cmd)
        elif self.state == self.REVERSE_SEARCHING:
            cmd = self.searching(cmd)
        elif self.state == self.REVERSE_APPROACHING:
            cmd = self.approaching(cmd)
        elif self.state == self.REVERSE_ALIGNING:
            cmd = self.aligning(cmd)
        elif self.state == self.REVERSE_FINAL_APPROACH:
            cmd = self.final_approach(cmd)
        elif self.state == self.REVERSE_PASSING:
            cmd = self.passing(cmd)
        elif self.state == self.REVERSE_CLEARING:
            cmd = self.reverse_clearing(cmd)
        elif self.state == self.SURFACING:
            cmd = self.surfacing(cmd)
        elif self.state == self.COMPLETED:
            cmd = self.completed(cmd)
        
        self.cmd_vel_pub.publish(cmd)
        self.state_pub.publish(String(data=self.get_state_name()))
    
    def depth_control(self, target_depth: float) -> float:
        """Normal depth control"""
        depth_error = target_depth - self.current_depth
        deadband = 0.15
        
        if abs(depth_error) < deadband:
            return 0.0
        
        if abs(depth_error) < 0.4:
            z_cmd = depth_error * 0.4
        elif abs(depth_error) < 0.8:
            z_cmd = depth_error * 0.6
        else:
            z_cmd = depth_error * 0.8
        
        return max(-0.6, min(z_cmd, 0.6))
    
    def gentle_depth_control(self, target_depth: float) -> float:
        """Gentle depth control during passage"""
        depth_error = target_depth - self.current_depth
        deadband = 0.25
        
        if abs(depth_error) < deadband:
            return 0.0
        
        z_cmd = depth_error * 0.2
        return max(-0.3, min(z_cmd, 0.3))
    
    def submerge(self, cmd: Twist) -> Twist:
        if abs(self.mission_depth - self.current_depth) < 0.3:
            if time.time() - self.state_start_time > 3.0:
                self.get_logger().info('✅ Submerged - starting search')
                self.reverse_mode_pub.publish(Bool(data=self.reverse_mode))
                self.transition_to(self.SEARCHING)
        return cmd
    
    def searching(self, cmd: Twist) -> Twist:
        if self.gate_detected and self.estimated_distance < 999:
            self.get_logger().info(
                f'🎯 Gate found at {self.estimated_distance:.2f}m'
            )
            if self.reverse_mode:
                self.transition_to(self.REVERSE_APPROACHING)
            else:
                self.transition_to(self.APPROACHING)
            return cmd
        
        cmd.linear.x = self.search_forward_speed
        cmd.angular.z = 0.3 if (time.time() % 8 < 4) else -0.3
        return cmd
    
    def approaching(self, cmd: Twist) -> Twist:
        if not self.gate_detected:
            cmd.linear.x = 0.2
            cmd.angular.z = 0.3
            return cmd
        
        if self.estimated_distance <= self.approach_stop_distance:
            self.get_logger().info(f'🛑 Reached 3m - ALIGNING (pos={self.frame_position:+.3f})')
            if self.reverse_mode:
                self.transition_to(self.REVERSE_ALIGNING)
            else:
                self.transition_to(self.ALIGNING)
            return cmd
        
        cmd.linear.x = self.approach_speed
        cmd.angular.z = -self.frame_position * 1.0
        
        return cmd
    
    def aligning(self, cmd: Twist) -> Twist:
        if not self.gate_detected:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.3
            return cmd
        
        if self.alignment_start_time == 0.0:
            self.alignment_start_time = time.time()
            self.get_logger().info(f'🎯 ALIGNING at 3m (pos={self.frame_position:+.3f})')
        
        elapsed = time.time() - self.alignment_start_time
        
        if elapsed > self.alignment_max_time:
            self.get_logger().warn('⏰ Alignment timeout - proceeding')
            self.alignment_start_time = 0.0
            if self.reverse_mode:
                self.transition_to(self.REVERSE_FINAL_APPROACH)
            else:
                self.transition_to(self.FINAL_APPROACH)
            return cmd
        
        is_well_aligned = abs(self.frame_position) < self.alignment_threshold
        has_confidence = self.confidence > 0.8
        
        if is_well_aligned and has_confidence:
            self.get_logger().info(f'✅ ALIGNED (pos={self.frame_position:+.3f}, {elapsed:.1f}s)')
            self.alignment_start_time = 0.0
            if self.reverse_mode:
                self.transition_to(self.REVERSE_FINAL_APPROACH)
            else:
                self.transition_to(self.FINAL_APPROACH)
            return cmd
        
        quality = abs(self.frame_position)
        
        if quality > 0.2:
            cmd.linear.x = 0.0
            cmd.angular.z = -self.frame_position * 4.0
        elif quality > 0.1:
            cmd.linear.x = 0.1
            cmd.angular.z = -self.frame_position * 3.0
        else:
            cmd.linear.x = 0.15
            cmd.angular.z = -self.frame_position * 2.0
        
        return cmd
    
    def final_approach(self, cmd: Twist) -> Twist:
        if self.estimated_distance <= self.passing_trigger_distance:
            if abs(self.frame_position) < 0.15:
                self.get_logger().info(
                    f'🚀 COMMITTING TO PASSAGE at {self.estimated_distance:.2f}m '
                    f'(alignment={self.frame_position:+.3f})'
                )
                self.passing_start_position = self.current_position
                if self.reverse_mode:
                    self.transition_to(self.REVERSE_PASSING)
                else:
                    self.transition_to(self.PASSING)
                return cmd
            else:
                cmd.linear.x = 0.0
                cmd.angular.z = -self.frame_position * 4.0
                return cmd
        
        if abs(self.frame_position) > 0.10:
            cmd.linear.x = self.final_approach_speed * 0.6
            cmd.angular.z = -self.frame_position * 3.0
        else:
            cmd.linear.x = self.final_approach_speed
            cmd.angular.z = -self.frame_position * 1.5
        
        return cmd
    
    def passing(self, cmd: Twist) -> Twist:
        """
        PASSING STATE: Full speed through gate
        Transitions to CLEARING when AUV's BACK passes the gate plane
        """
        if self.passing_start_position is None:
            self.passing_start_position = self.current_position
            direction = "REVERSE" if self.reverse_mode else "FORWARD"
            self.get_logger().info(f'🚀 {direction} PASSAGE STARTED')
        
        if self.current_position:
            current_x = self.current_position[0]
            
            # Determine if AUV's BACK has passed the gate
            if not self.reverse_mode:
                # Forward: back is at (current_x - auv_length)
                auv_back_x = current_x - self.auv_length
                back_passed = auv_back_x > self.gate_x_position
            else:
                # Reverse: back is at (current_x + auv_length)
                auv_back_x = current_x + self.auv_length
                back_passed = auv_back_x < self.gate_x_position
            
            # Transition to CLEARING as soon as back passes
            if back_passed:
                self.get_logger().info('='*70)
                self.get_logger().info(f'✅ AUV BACK PASSED GATE!')
                self.get_logger().info(f'   Current position: X={current_x:.2f}m')
                self.get_logger().info(f'   AUV back at: X={auv_back_x:.2f}m')
                self.get_logger().info(f'   Gate at: X={self.gate_x_position:.2f}m')
                self.get_logger().info(f'   → Entering CLEARING (0.10m more)')
                self.get_logger().info('='*70)
                
                if not self.reverse_mode:
                    self.first_pass_complete = True
                    self.transition_to(self.CLEARING)
                else:
                    self.second_pass_complete = True
                    self.transition_to(self.REVERSE_CLEARING)
                return cmd
            
            # Show progress
            distance_to_gate = abs(current_x - self.gate_x_position)
            direction = "FORWARD" if not self.reverse_mode else "REVERSE"
            self.get_logger().info(
                f'🚀 PASSING ({direction}): X={current_x:.2f}m, '
                f'gate at {self.gate_x_position:.2f}m, '
                f'{distance_to_gate:.2f}m to gate',
                throttle_duration_sec=0.5
            )
        
        # Full speed straight ahead
        cmd.linear.x = self.passing_speed
        cmd.angular.z = 0.0
        
        return cmd
    
    def clearing(self, cmd: Twist) -> Twist:
        """
        CLEARING STATE: Travel 0.10m more after back passes gate
        """
        if self.current_position:
            current_x = self.current_position[0]
            
            # Check if we've traveled clearance_margin past gate
            if current_x >= self.forward_clearance_x:
                distance_past = current_x - self.gate_x_position
                self.get_logger().info('='*70)
                self.get_logger().info(f'✅ CLEARANCE COMPLETE!')
                self.get_logger().info(f'   Current X: {current_x:.2f}m')
                self.get_logger().info(f'   Distance past gate: {distance_past:.2f}m')
                self.get_logger().info(f'   → Starting U-TURN')
                self.get_logger().info('='*70)
                
                # Reset U-turn timer for proper initialization
                self.uturn_start_time = 0.0
                self.transition_to(self.UTURN)
                return cmd
            
            # Show progress
            distance_needed = self.forward_clearance_x - current_x
            self.get_logger().info(
                f'🏃 CLEARING: X={current_x:.2f}m, '
                f'need {distance_needed:.2f}m more',
                throttle_duration_sec=0.4
            )
        
        cmd.linear.x = 0.8
        return cmd
    
    def uturn(self, cmd: Twist) -> Twist:
        """
        PROPER U-TURN: Wide turning maneuver while maintaining depth
        - Moves forward while turning (not spinning in place)
        - Maintains constant depth to avoid surfacing
        - Completes 180-degree turn to face gate
        - Transitions to alignment after turn
        """
        
        # Initialize U-turn
        if self.uturn_start_time == 0.0:
            self.uturn_start_yaw = self.current_yaw
            self.uturn_start_time = time.time()
            self.uturn_start_x = self.current_position[0] if self.current_position else 0.0
            self.get_logger().info('='*70)
            self.get_logger().info('🔄 STARTING PROPER U-TURN')
            self.get_logger().info(f'   Starting yaw: {math.degrees(self.uturn_start_yaw):.1f}°')
            self.get_logger().info(f'   Starting X: {self.uturn_start_x:.2f}m')
            self.get_logger().info('='*70)
        
        # Calculate how much we've turned
        angle_turned = abs(self.normalize_angle(self.current_yaw - self.uturn_start_yaw))
        elapsed = time.time() - self.uturn_start_time
        
        # Check if U-turn is complete (180 degrees ± 10 degrees)
        if angle_turned > (math.pi - 0.17):  # ~170 degrees
            turn_distance = abs(self.current_position[0] - self.uturn_start_x) if self.current_position else 0
            
            self.get_logger().info('='*70)
            self.get_logger().info(
                f'✅ U-TURN COMPLETE (turned {math.degrees(angle_turned):.0f}°, {elapsed:.1f}s)'
            )
            self.get_logger().info(f'   Travel distance: {turn_distance:.2f}m')
            self.get_logger().info(f'   Final yaw: {math.degrees(self.current_yaw):.1f}°')
            self.get_logger().info('   → Aligning with gate for reverse pass')
            self.get_logger().info('='*70)
            
            self.reverse_mode = True
            self.reverse_mode_pub.publish(Bool(data=True))
            self.uturn_start_time = 0.0  # Reset for next time
            self.transition_to(self.POST_UTURN_ALIGN)
            return cmd
        
        # PROPER U-TURN MANEUVER
        # Move forward while turning - creates a proper arc
        cmd.linear.x = self.uturn_forward_speed  # Forward motion
        cmd.angular.z = self.uturn_angular_speed  # Turning
        
        # CRITICAL: Maintain depth during U-turn
        depth_error = self.uturn_depth - self.current_depth
        if abs(depth_error) > 0.15:
            cmd.linear.z = depth_error * 1.0  # Strong depth correction
            cmd.linear.z = max(-0.5, min(cmd.linear.z, 0.5))
        else:
            cmd.linear.z = depth_error * 0.3  # Gentle adjustment
        
        # Log progress
        progress_pct = (angle_turned / math.pi) * 100
        self.get_logger().info(
            f'🔄 U-TURN: {math.degrees(angle_turned):.0f}° / 180° ({progress_pct:.0f}%) | '
            f'depth={self.current_depth:.2f}m | X={self.current_position[0]:.2f}m',
            throttle_duration_sec=0.5
        )
        
        return cmd
    
    def post_uturn_align(self, cmd: Twist) -> Twist:
        """
        POST U-TURN ALIGNMENT: Align with gate after completing U-turn
        - Ensures AUV is properly oriented toward gate
        - Searches for gate if not immediately visible
        - Only proceeds when gate is detected and aligned
        """
        
        # Check if we can see the gate
        if self.gate_detected:
            # Gate visible - check alignment
            if abs(self.frame_position) < 0.15:
                # Well aligned - start reverse approach
                self.get_logger().info('='*70)
                self.get_logger().info('✅ POST-UTURN ALIGNMENT COMPLETE')
                self.get_logger().info(f'   Alignment: {self.frame_position:+.3f}')
                self.get_logger().info('   → Starting reverse approach')
                self.get_logger().info('='*70)
                self.transition_to(self.REVERSE_APPROACHING)
                return cmd
            else:
                # Need to adjust alignment
                cmd.linear.x = 0.2  # Slow forward
                cmd.angular.z = -self.frame_position * 2.0  # Correct alignment
                
                self.get_logger().info(
                    f'🎯 POST-UTURN ALIGN: pos={self.frame_position:+.3f}, adjusting...',
                    throttle_duration_sec=0.5
                )
        else:
            # Gate not visible - search for it
            cmd.linear.x = 0.3  # Move forward slowly
            cmd.angular.z = 0.2  # Gentle rotation to find gate
            
            self.get_logger().info(
                '🔍 POST-UTURN ALIGN: Searching for gate...',
                throttle_duration_sec=0.5
            )
        
        return cmd
    
    def reverse_clearing(self, cmd: Twist) -> Twist:
        """
        REVERSE CLEARING: Travel 0.10m more after back passes gate (reverse direction)
        """
        if self.current_position:
            current_x = self.current_position[0]
            
            if current_x <= self.reverse_clearance_x:
                distance_past = self.gate_x_position - current_x
                self.get_logger().info('='*70)
                self.get_logger().info(f'✅ REVERSE CLEARANCE COMPLETE!')
                self.get_logger().info(f'   Current X: {current_x:.2f}m')
                self.get_logger().info(f'   Distance past gate: {distance_past:.2f}m')
                self.get_logger().info(f'   → Surfacing')
                self.get_logger().info('='*70)
                
                self.transition_to(self.SURFACING)
                return cmd
            
            distance_needed = current_x - self.reverse_clearance_x
            self.get_logger().info(
                f'🏃 REVERSE CLEARING: X={current_x:.2f}m, '
                f'need {distance_needed:.2f}m more',
                throttle_duration_sec=0.4
            )
        
        cmd.linear.x = 0.8
        return cmd
    
    def surfacing(self, cmd: Twist) -> Twist:
        """
        IMPROVED SURFACING: Gradual, controlled ascent with forward motion
        - Starts with strong upward thrust deep underwater
        - Reduces thrust progressively as approaching surface
        - Moves forward to clear the area
        - Smooth transition to surface
        """
        depth_to_surface = abs(self.current_depth)
        
        # Check if reached surface
        if self.current_depth > -0.15:
            total_time = time.time() - self.mission_start_time
            
            self.get_logger().info('='*70)
            self.get_logger().info('🏆 QUALIFICATION MISSION COMPLETE!')
            self.get_logger().info('='*70)
            self.get_logger().info(f'   Pass 1: {"✅ COMPLETE" if self.first_pass_complete else "❌ FAILED"}')
            self.get_logger().info(f'   Pass 2: {"✅ COMPLETE" if self.second_pass_complete else "❌ FAILED"}')
            self.get_logger().info(f'   Total time: {total_time:.1f}s')
            
            if self.first_pass_complete and self.second_pass_complete:
                self.get_logger().info('   🏆 QUALIFICATION SCORE: 2 POINTS')
                self.get_logger().info('   ✅ QUALIFIED FOR FINALS!')
            elif self.first_pass_complete:
                self.get_logger().info('   ⚠️ QUALIFICATION SCORE: 1 POINT')
            else:
                self.get_logger().info('   ❌ QUALIFICATION SCORE: 0 POINTS')
            
            self.get_logger().info('='*70)
            self.transition_to(self.COMPLETED)
            return cmd
        
        # IMPROVED: Progressive thrust reduction based on depth
        if depth_to_surface > 0.6:
            # Deep underwater - strong upward thrust
            cmd.linear.z = -0.6
            cmd.linear.x = 0.3  # Move forward while ascending
            status = "DEEP"
        elif depth_to_surface > 0.4:
            # Mid-depth - moderate thrust
            cmd.linear.z = -0.4
            cmd.linear.x = 0.2
            status = "MID"
        elif depth_to_surface > 0.2:
            # Approaching surface - gentle thrust
            cmd.linear.z = -0.25
            cmd.linear.x = 0.15
            status = "SHALLOW"
        else:
            # Very close to surface - minimal thrust
            cmd.linear.z = -0.15
            cmd.linear.x = 0.1
            status = "SURFACE"
        
        # Log surfacing progress
        self.get_logger().info(
            f'⬆️ SURFACING ({status}): depth={self.current_depth:.2f}m, '
            f'thrust={cmd.linear.z:.2f}, forward={cmd.linear.x:.2f}',
            throttle_duration_sec=0.5
        )
        
        return cmd
    
    def completed(self, cmd: Twist) -> Twist:
        """Mission complete - stop all movement"""
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
        
        self.get_logger().info(f'🔄 STATE TRANSITION: {old_name} → {new_name}')
    
    def get_state_name(self) -> str:
        """Get human-readable state name"""
        names = {
            self.SUBMERGING: 'SUBMERGING',
            self.SEARCHING: 'SEARCHING',
            self.APPROACHING: 'APPROACHING',
            self.ALIGNING: 'ALIGNING',
            self.FINAL_APPROACH: 'FINAL_APPROACH',
            self.PASSING: 'PASSING',
            self.CLEARING: 'CLEARING',
            self.UTURN: 'UTURN',
            self.POST_UTURN_ALIGN: 'POST_UTURN_ALIGN',
            self.REVERSE_SEARCHING: 'REVERSE_SEARCHING',
            self.REVERSE_APPROACHING: 'REVERSE_APPROACHING',
            self.REVERSE_ALIGNING: 'REVERSE_ALIGNING',
            self.REVERSE_FINAL_APPROACH: 'REVERSE_FINAL_APPROACH',
            self.REVERSE_PASSING: 'REVERSE_PASSING',
            self.REVERSE_CLEARING: 'REVERSE_CLEARING',
            self.SURFACING: 'SURFACING',
            self.COMPLETED: 'COMPLETED',
        }
        return names.get(self.state, 'UNKNOWN')
    
    @staticmethod
    def normalize_angle(angle: float) -> float:
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle


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
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()