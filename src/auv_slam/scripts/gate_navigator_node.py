#!/usr/bin/env python3
"""
PROPER Gate Navigator - Align at 3m Before Final Approach
Flow:
1. SEARCHING → Find gate
2. APPROACHING → Move toward gate (relaxed, ~5m to 3m)
3. ALIGNING → At 3m, STOP and center align properly
4. FINAL_APPROACH → Aligned approach from 3m to 1.2m
5. PASSING → Full speed through gate
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
import time
import math


class ProperGateNavigator(Node):
    def __init__(self):
        super().__init__('gate_navigator_node')
        
        # State machine - CORRECTED FLOW
        self.IDLE = 0
        self.SEARCHING = 1
        self.APPROACHING = 2       # Casual approach from far (5m → 3m)
        self.ALIGNING = 3          # Stop at 3m and align properly
        self.FINAL_APPROACH = 4    # Aligned slow approach (3m → 1.2m)
        self.PASSING = 5           # Full speed passage
        self.COMPLETED = 6
        
        self.state = self.SEARCHING

        self.declare_parameter('gate_x_position', -8.0)
        self.declare_parameter('gate_clearance_distance', 0.5)

        self.gate_x_position = self.get_parameter('gate_x_position').value
        self.gate_clearance_distance = self.get_parameter('gate_clearance_distance').value
        
        # Parameters
        self.declare_parameter('target_depth', -1.7)
        self.declare_parameter('depth_correction_gain', 2.0)
        
        # Search parameters
        self.declare_parameter('search_forward_speed', 0.5)
        self.declare_parameter('search_rotation_speed', 0.15)
        
        # Approaching parameters (far → 3m)
        self.declare_parameter('approach_start_distance', 8.0)  # Start approaching when detected
        self.declare_parameter('approach_stop_distance', 3.0)   # Stop to align at 3m
        self.declare_parameter('approach_speed', 0.6)           # Moderate speed
        self.declare_parameter('approach_yaw_gain', 1.0)        # Light yaw correction
        
        # Alignment parameters (at 3m)
        self.declare_parameter('alignment_distance', 3.0)       # Align at 3m
        self.declare_parameter('alignment_threshold', 0.08)     # Must be within ±8%
        self.declare_parameter('alignment_max_time', 15.0)
        self.declare_parameter('alignment_yaw_gain', 3.5)       # Strong yaw for alignment
        
        # Final approach parameters (3m → 1.2m)
        self.declare_parameter('final_approach_start', 3.0)     # Start after alignment at 3m
        self.declare_parameter('final_approach_speed', 0.5)     # Slow and controlled
        self.declare_parameter('final_approach_threshold', 0.12) # Maintain alignment
        
        # Passing parameters
        self.declare_parameter('passing_trigger_distance', 1.2)
        self.declare_parameter('passing_speed', 1.2)
        self.declare_parameter('gate_width', 1.5)
        
        # Flare avoidance
        self.declare_parameter('flare_critical_distance', 2.5)
        self.declare_parameter('flare_lateral_speed', 0.4)
        self.declare_parameter('flare_forward_speed', 0.3)
        
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
        self.gate_width = self.get_parameter('gate_width').value
        
        self.flare_critical_distance = self.get_parameter('flare_critical_distance').value
        self.flare_lateral_speed = self.get_parameter('flare_lateral_speed').value
        self.flare_forward_speed = self.get_parameter('flare_forward_speed').value
        
        # State variables
        self.gate_detected = False
        self.flare_detected = False
        self.partial_gate = False
        self.alignment_error = 0.0
        self.estimated_distance = 999.0
        self.current_depth = 0.0
        self.frame_position = 0.0
        self.confidence = 0.0
        
        # Position tracking
        self.current_position = None
        self.passing_start_position = None
        self.flare_world_position = (-14.0, 2.0, -2.25)
        self.flare_distance = 999.0
        self.flare_avoidance_direction = 0.0
        
        # Timing
        self.gate_lost_time = 0.0
        self.gate_lost_timeout = 3.0
        self.alignment_start_time = 0.0
        self.state_start_time = time.time()
        
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
        self.frame_position_sub = self.create_subscription(
            Float32, '/gate/frame_position', self.frame_position_callback, 10)
        self.partial_gate_sub = self.create_subscription(
            Bool, '/gate/partial_detection', self.partial_gate_callback, 10)
        self.confidence_sub = self.create_subscription(
            Float32, '/gate/detection_confidence', self.confidence_callback, 10)
        self.flare_detected_sub = self.create_subscription(
            Bool, '/flare/detected', self.flare_detected_callback, 10)
        self.flare_avoidance_sub = self.create_subscription(
            Float32, '/flare/avoidance_direction', self.flare_avoidance_callback, 10)
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/rp2040/cmd_vel', 10)
        self.state_pub = self.create_publisher(String, '/gate/navigation_state', 10)
        
        self.control_timer = self.create_timer(0.05, self.control_loop)
        
        self.get_logger().info('='*70)
        self.get_logger().info('✅ PROPER Gate Navigator - Align at 3m')
        self.get_logger().info('='*70)
        self.get_logger().info('   Flow: SEARCH → APPROACH → ALIGN@3m → FINAL_APPROACH → PASS')
        self.get_logger().info(f'   Approach: {self.approach_start_distance}m → {self.approach_stop_distance}m')
        self.get_logger().info(f'   Align at: {self.alignment_distance}m (±{self.alignment_threshold*100:.0f}%)')
        self.get_logger().info(f'   Final approach: {self.final_approach_start}m → {self.passing_trigger_distance}m')
        self.get_logger().info('='*70)
    
    def gate_detected_callback(self, msg: Bool):
        was_detected = self.gate_detected
        self.gate_detected = msg.data
        
        if not was_detected and self.gate_detected:
            if self.gate_first_detected_time is None:
                self.gate_first_detected_time = time.time()
                self.get_logger().info('🎯 GATE FIRST DETECTED')
            self.gate_lost_time = 0.0
        elif was_detected and not self.gate_detected:
            self.gate_lost_time = time.time()
            self.get_logger().warn('⚠️ Gate lost from view')
    
    def frame_position_callback(self, msg: Float32):
        self.frame_position = msg.data
    
    def partial_gate_callback(self, msg: Bool):
        self.partial_gate = msg.data
    
    def confidence_callback(self, msg: Float32):
        self.confidence = msg.data
    
    def flare_detected_callback(self, msg: Bool):
        self.flare_detected = msg.data
    
    def flare_avoidance_callback(self, msg: Float32):
        self.flare_avoidance_direction = msg.data
    
    def alignment_callback(self, msg: Float32):
        self.alignment_error = msg.data
    
    def distance_callback(self, msg: Float32):
        self.estimated_distance = msg.data
    
    def odom_callback(self, msg: Odometry):
        self.current_depth = msg.pose.pose.position.z
        self.current_position = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        )
        
        # Calculate distance to flare
        if self.current_position:
            dx = self.flare_world_position[0] - self.current_position[0]
            dy = self.flare_world_position[1] - self.current_position[1]
            self.flare_distance = math.sqrt(dx*dx + dy*dy)
    
    def control_loop(self):
        cmd = Twist()
        
        # Depth control with proper deadband
        depth_error = self.target_depth - self.current_depth
        depth_deadband = 0.3
        
        if abs(depth_error) < depth_deadband:
            cmd.linear.z = 0.0
        elif abs(depth_error) < 0.6:
            cmd.linear.z = depth_error * 0.8
            cmd.linear.z = max(-0.4, min(cmd.linear.z, 0.4))
        else:
            cmd.linear.z = depth_error * 1.2
            cmd.linear.z = max(-1.0, min(cmd.linear.z, 1.0))
        
        # State machine
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
        elif self.state == self.COMPLETED:
            cmd = self.completed_behavior(cmd)
        
        self.cmd_vel_pub.publish(cmd)
        
        # Publish state
        state_msg = String()
        state_msg.data = self.get_state_name()
        self.state_pub.publish(state_msg)
    
    def searching_behavior(self, cmd: Twist) -> Twist:
        """Search for gate with sweep pattern"""
        if self.gate_detected and self.estimated_distance < 999:
            self.get_logger().info(
                f'🎯 Gate found at {self.estimated_distance:.2f}m - Starting approach'
            )
            self.transition_to(self.APPROACHING)
            return cmd
        
        elapsed = time.time() - self.state_start_time
        sweep_period = 8.0
        sweep_phase = (elapsed % sweep_period) / sweep_period
        
        if sweep_phase < 0.5:
            cmd.angular.z = self.search_rotation_speed
        else:
            cmd.angular.z = -self.search_rotation_speed
        
        cmd.linear.x = self.search_forward_speed
        
        if int(elapsed) % 3 == 0:
            direction = "LEFT" if sweep_phase < 0.5 else "RIGHT"
            self.get_logger().info(
                f'🔍 Searching ({direction})... {elapsed:.0f}s',
                throttle_duration_sec=2.9
            )
        
        return cmd
    
    def approaching_behavior(self, cmd: Twist) -> Twist:
        """
        CASUAL APPROACH: Move toward gate from far (5m → 3m)
        - Moderate speed
        - Light yaw correction to keep gate in view
        - NOT trying to perfectly align yet
        - Stop at 3m to do proper alignment
        """
        
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
        
        # CRITICAL: Stop at 3m to align properly
        if self.estimated_distance <= self.approach_stop_distance:
            self.get_logger().info(
                f'🛑 Reached 3m - STOPPING to align properly '
                f'(distance={self.estimated_distance:.2f}m)'
            )
            self.transition_to(self.ALIGNING)
            return cmd
        
        # Casual approach with light correction
        cmd.linear.x = self.approach_speed
        
        # Light yaw correction - just keep gate roughly centered
        cmd.angular.z = -self.frame_position * self.approach_yaw_gain
        
        self.get_logger().info(
            f'🚶 APPROACHING: dist={self.estimated_distance:.2f}m, '
            f'pos={self.frame_position:+.3f}, speed={cmd.linear.x:.2f}',
            throttle_duration_sec=0.5
        )
        
        return cmd
    
    def aligning_behavior(self, cmd: Twist) -> Twist:
        """
        PROPER ALIGNMENT AT 3M:
        - STOP forward movement
        - Pure rotation to center the gate
        - Only proceed when perfectly aligned
        - Strict alignment threshold
        """
        
        if not self.gate_detected:
            if self.gate_lost_time > 0.0:
                lost_duration = time.time() - self.gate_lost_time
                if lost_duration > self.gate_lost_timeout:
                    self.get_logger().warn('❌ Gate lost during alignment - returning to search')
                    self.alignment_start_time = 0.0
                    self.transition_to(self.SEARCHING)
                else:
                    cmd.linear.x = 0.0
                    cmd.angular.z = 0.0
            return cmd
        
        # Initialize alignment timer
        if self.alignment_start_time == 0.0:
            self.alignment_start_time = time.time()
            self.get_logger().info(
                f'🎯 STARTING ALIGNMENT at {self.estimated_distance:.2f}m '
                f'(current pos={self.frame_position:+.3f})'
            )
        
        alignment_elapsed = time.time() - self.alignment_start_time
        
        # Timeout check
        if alignment_elapsed > self.alignment_max_time:
            self.get_logger().warn('⏰ Alignment timeout - proceeding anyway')
            self.alignment_start_time = 0.0
            self.transition_to(self.FINAL_APPROACH)
            return cmd
        
        # Check alignment quality
        is_well_aligned = abs(self.frame_position) < self.alignment_threshold
        has_confidence = self.confidence > 0.8 and not self.partial_gate
        
        # DECISION: Is alignment good enough?
        if is_well_aligned and has_confidence:
            self.get_logger().info(
                f'✅ ALIGNMENT COMPLETE! '
                f'(pos={self.frame_position:+.3f}, took {alignment_elapsed:.1f}s) '
                f'→ Starting final approach'
            )
            self.alignment_start_time = 0.0
            self.transition_to(self.FINAL_APPROACH)
            return cmd
        
        # ALIGNMENT STRATEGY: Pure rotation around vertical axis
        # Stay roughly in place, just rotate to center
        
        alignment_quality = abs(self.frame_position)
        
        if alignment_quality > 0.2:
            # Far off center - strong rotation, NO forward
            cmd.linear.x = 0.0
            cmd.angular.z = -self.frame_position * self.alignment_yaw_gain
            status = "MAJOR"
        elif alignment_quality > 0.1:
            # Moderately off - moderate rotation, tiny forward
            cmd.linear.x = 0.1
            cmd.angular.z = -self.frame_position * self.alignment_yaw_gain * 0.8
            status = "MODERATE"
        else:
            # Nearly aligned - fine tuning
            cmd.linear.x = 0.15
            cmd.angular.z = -self.frame_position * self.alignment_yaw_gain * 0.5
            status = "FINE"
        
        self.get_logger().info(
            f'🔄 ALIGNING ({status}): pos={self.frame_position:+.3f}, '
            f'yaw={cmd.angular.z:+.2f}, t={alignment_elapsed:.1f}s',
            throttle_duration_sec=0.3
        )
        
        return cmd
    
    def final_approach_behavior(self, cmd: Twist) -> Twist:
        """
        FINAL APPROACH (3m → 1.2m):
        - Slow controlled movement
        - Maintain alignment while approaching
        - Commit to passage at 1.2m
        """
        
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
        
        # CRITICAL: Check if close enough to commit
        if self.estimated_distance <= self.passing_trigger_distance:
            # Final check before committing
            if abs(self.frame_position) < 0.2:
                self.get_logger().info(
                    f'🚀 COMMITTING TO PASSAGE at {self.estimated_distance:.2f}m '
                    f'(alignment={self.frame_position:+.3f})'
                )
                self.passing_start_position = self.current_position
                self.transition_to(self.PASSING)
                return cmd
            else:
                # Too misaligned at trigger point - emergency realign
                self.get_logger().error(
                    f'🚨 At trigger point but misaligned ({self.frame_position:+.3f})! '
                    f'Emergency realignment...'
                )
                cmd.linear.x = 0.0
                cmd.angular.z = -self.frame_position * self.alignment_yaw_gain * 2.0
                return cmd
        
        # Check if drifting during approach
        if abs(self.frame_position) > self.final_approach_threshold:
            # Drifting - slow down and correct
            self.get_logger().warn(
                f'⚠️ Drifting (pos={self.frame_position:+.3f}) - correcting',
                throttle_duration_sec=0.5
            )
            cmd.linear.x = self.final_approach_speed * 0.6
            cmd.angular.z = -self.frame_position * self.alignment_yaw_gain * 0.6
        else:
            # Good alignment - proceed smoothly
            cmd.linear.x = self.final_approach_speed
            cmd.angular.z = -self.frame_position * self.alignment_yaw_gain * 0.3
        
        self.get_logger().info(
            f'➡️ FINAL APPROACH: dist={self.estimated_distance:.2f}m, '
            f'pos={self.frame_position:+.3f}, speed={cmd.linear.x:.2f}',
            throttle_duration_sec=0.5
        )
        
        return cmd
    
    def passing_behavior(self, cmd: Twist) -> Twist:
        """
        FULL PASSAGE: Maximum speed straight through
        Completes when AUV safely clears the gate with proper buffer
        """
        
        if self.passing_start_position is None:
            self.passing_start_position = self.current_position
            # Gate is at X=-8 (from world file)
            self.gate_x_position = -8.0
            self.get_logger().info('🚀 PASSAGE STARTED - FULL SPEED AHEAD!')
        
        # Check if we've SAFELY cleared the gate based on X-position
        if self.current_position:
            current_x = self.current_position[0]
            

            gate_clearance_distance = 0.5
            gate_cleared_threshold = self.gate_x_position + gate_clearance_distance
            
            if current_x > gate_cleared_threshold:
                # Calculate actual distance traveled for logging
                if self.passing_start_position:
                    dx = self.current_position[0] - self.passing_start_position[0]
                    dy = self.current_position[1] - self.passing_start_position[1]
                    distance_traveled = math.sqrt(dx*dx + dy*dy)
                    
                    self.get_logger().info('='*70)
                    self.get_logger().info(f'✅ GATE SAFELY CLEARED!')
                    self.get_logger().info(f'   Current position: X={current_x:.2f}m')
                    self.get_logger().info(f'   Gate position: X={self.gate_x_position:.2f}m')
                    self.get_logger().info(f'   Clearance: {current_x - self.gate_x_position:.2f}m past gate')
                    self.get_logger().info(f'   Distance traveled: {distance_traveled:.2f}m')
                    self.get_logger().info('='*70)
                
                self.transition_to(self.COMPLETED)
                return cmd
            
            # Show detailed progress through gate
            distance_past_gate = current_x - self.gate_x_position
            
            if distance_past_gate < 0:
                status = "APPROACHING"
                progress_pct = 0
            else:
                status = "CLEARING"
                progress_pct = (distance_past_gate / gate_clearance_distance) * 100
            
            self.get_logger().info(
                f'🚀 PASSING ({status}): X={current_x:.2f}m, '
                f'{abs(distance_past_gate):.2f}m {"past" if distance_past_gate > 0 else "before"} gate '
                f'({progress_pct:.0f}% cleared)',
                throttle_duration_sec=0.4
            )
        
        # FULL SPEED - no corrections during passage
        cmd.linear.x = self.passing_speed
        cmd.linear.y = 0.0
        cmd.angular.z = 0.0
        
        return cmd
    
    def completed_behavior(self, cmd: Twist) -> Twist:
        """Mission complete - stop all movement"""
        if self.mission_start_time:
            total_time = time.time() - self.mission_start_time
            detection_time = (self.gate_first_detected_time - self.mission_start_time 
                             if self.gate_first_detected_time else 0)
            
            self.get_logger().info('='*70)
            self.get_logger().info('🎉 MISSION COMPLETE - GATE PASSED!')
            self.get_logger().info(f'   Total time: {total_time:.2f}s')
            self.get_logger().info(f'   Detection: {detection_time:.2f}s')
            self.get_logger().info(f'   Navigation: {total_time - detection_time:.2f}s')
            self.get_logger().info('='*70)
            
            # Only log once
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
        
        self.get_logger().info(f'🔄 STATE TRANSITION: {old_name} → {new_name}')
    
    def get_state_name(self) -> str:
        """Get human-readable state name"""
        names = {
            self.IDLE: 'IDLE',
            self.SEARCHING: 'SEARCHING',
            self.APPROACHING: 'APPROACHING',
            self.ALIGNING: 'ALIGNING',
            self.FINAL_APPROACH: 'FINAL_APPROACH',
            self.PASSING: 'PASSING',
            self.COMPLETED: 'COMPLETED'
        }
        return names.get(self.state, 'UNKNOWN')


def main(args=None):
    rclpy.init(args=args)
    node = ProperGateNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop_cmd = Twist()
        node.cmd_vel_pub.publish(stop_cmd)
        node.get_logger().info('Proper Gate Navigator shutting down')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()