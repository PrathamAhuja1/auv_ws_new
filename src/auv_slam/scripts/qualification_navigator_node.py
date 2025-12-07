#!/usr/bin/env python3
"""
QUALIFICATION Gate Navigator - Forward pass, U-turn, Reverse pass
Flow:
1. SUBMERGING → Start at surface, dive to target depth
2. SEARCHING → Find gate
3. APPROACHING → Move toward gate (5m to 3m)
4. ALIGNING → At 3m, STOP and center align
5. FINAL_APPROACH → Aligned approach (3m to 1.5m)
6. PASSING → Full speed through gate
7. U_TURNING → After clearing, perform 180° turn
8. REVERSE_APPROACH → Approach gate from other side
9. REVERSE_PASSING → Pass through gate backward
10. COMPLETED → Mission complete
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
        super().__init__('gate_navigator_node')
        
        # State machine for qualification task
        self.IDLE = 0
        self.SUBMERGING = 1          # Initial submersion from surface
        self.SEARCHING = 2           # Search for gate
        self.APPROACHING = 3         # Casual approach (5m → 3m)
        self.ALIGNING = 4            # Stop at 3m and align
        self.FINAL_APPROACH = 5      # Aligned approach (3m → 1.5m)
        self.PASSING = 6             # Full speed through gate
        self.U_TURNING = 7           # 180° turn after first pass
        self.REVERSE_APPROACH = 8    # Approach for second pass (backward)
        self.REVERSE_PASSING = 9     # Pass through gate backward
        self.COMPLETED = 10
        
        self.state = self.SUBMERGING
        
        # Mission tracking
        self.pass_number = 1  # 1 = forward pass, 2 = reverse pass
        self.gate_cleared = False
        
        # Position tracking
        self.gate_x_position = 10.0  # Gate at ~10m from start
        self.gate_clearance_buffer = 0.5  # Must clear gate by this distance
        self.auv_length = 0.8  # Approximate AUV length for clearance calculation
        
        # Parameters
        self.declare_parameter('target_depth', -1.5)  # Qualification depth
        self.declare_parameter('depth_correction_gain', 2.0)
        
        # Submersion parameters
        self.declare_parameter('submersion_speed', 0.3)
        self.declare_parameter('submersion_depth', -1.5)
        self.declare_parameter('submuration_duration', 3.0)
        
        # Search parameters
        self.declare_parameter('search_forward_speed', 0.4)
        self.declare_parameter('search_rotation_speed', 0.15)
        
        # Approach parameters
        self.declare_parameter('approach_start_distance', 8.0)
        self.declare_parameter('approach_stop_distance', 3.5)
        self.declare_parameter('approach_speed', 0.5)
        self.declare_parameter('approach_yaw_gain', 1.0)
        
        # Alignment parameters
        self.declare_parameter('alignment_distance', 3.5)
        self.declare_parameter('alignment_threshold', 0.08)
        self.declare_parameter('alignment_max_time', 12.0)
        self.declare_parameter('alignment_yaw_gain', 3.5)
        
        # Final approach
        self.declare_parameter('final_approach_trigger', 3.5)
        self.declare_parameter('final_approach_speed', 0.4)
        self.declare_parameter('final_approach_threshold', 0.12)
        
        # Passing parameters
        self.declare_parameter('passing_trigger_distance', 1.8)
        self.declare_parameter('passing_speed', 0.8)
        self.declare_parameter('gate_width', 1.5)
        
        # U-turn parameters
        self.declare_parameter('uturn_rotation_speed', 0.4)
        self.declare_parameter('uturn_duration', 8.0)
        self.declare_parameter('uturn_depth_gain', 1.5)
        
        # Reverse pass parameters
        self.declare_parameter('reverse_approach_speed', -0.4)  # Negative = backward
        self.declare_parameter('reverse_passing_speed', -0.8)
        
        # Load parameters
        self.target_depth = self.get_parameter('target_depth').value
        self.depth_gain = self.get_parameter('depth_correction_gain').value
        self.submersion_speed = self.get_parameter('submersion_speed').value
        self.submersion_depth = self.get_parameter('submersion_depth').value
        self.submersion_duration = self.get_parameter('submuration_duration').value
        
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
        
        self.final_approach_trigger = self.get_parameter('final_approach_trigger').value
        self.final_approach_speed = self.get_parameter('final_approach_speed').value
        self.final_approach_threshold = self.get_parameter('final_approach_threshold').value
        
        self.passing_trigger_distance = self.get_parameter('passing_trigger_distance').value
        self.passing_speed = self.get_parameter('passing_speed').value
        self.gate_width = self.get_parameter('gate_width').value
        
        self.uturn_rotation_speed = self.get_parameter('uturn_rotation_speed').value
        self.uturn_duration = self.get_parameter('uturn_duration').value
        self.uturn_depth_gain = self.get_parameter('uturn_depth_gain').value
        
        self.reverse_approach_speed = self.get_parameter('reverse_approach_speed').value
        self.reverse_passing_speed = self.get_parameter('reverse_passing_speed').value
        
        # State variables
        self.gate_detected = False
        self.partial_gate = False
        self.alignment_error = 0.0
        self.estimated_distance = 999.0
        self.current_depth = 0.0
        self.frame_position = 0.0
        self.confidence = 0.0
        
        # Position tracking
        self.current_position = None
        self.passing_start_position = None
        self.uturn_start_position = None
        
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
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/rp2040/cmd_vel', 10)
        self.state_pub = self.create_publisher(String, '/gate/navigation_state', 10)
        self.pass_number_pub = self.create_publisher(String, '/gate/pass_number', 10)
        
        self.control_timer = self.create_timer(0.05, self.control_loop)
        
        self.get_logger().info('='*70)
        self.get_logger().info('✅ QUALIFICATION Navigator - Forward + U-Turn + Reverse')
        self.get_logger().info('='*70)
        self.get_logger().info('   Flow: SUBMERGE → SEARCH → APPROACH → ALIGN → PASS → U-TURN → REVERSE')
        self.get_logger().info(f'   Gate at: {self.gate_x_position}m | Target depth: {self.target_depth}m')
        self.get_logger().info(f'   Pass 1: Forward | Pass 2: Reverse (U-turn then back through)')
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
    
    def control_loop(self):
        cmd = Twist()
        
        # Depth control with deadband
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
        if self.state == self.SUBMERGING:
            cmd = self.submerging_behavior(cmd)
        elif self.state == self.SEARCHING:
            cmd = self.searching_behavior(cmd)
        elif self.state == self.APPROACHING:
            cmd = self.approaching_behavior(cmd)
        elif self.state == self.ALIGNING:
            cmd = self.aligning_behavior(cmd)
        elif self.state == self.FINAL_APPROACH:
            cmd = self.final_approach_behavior(cmd)
        elif self.state == self.PASSING:
            cmd = self.passing_behavior(cmd)
        elif self.state == self.U_TURNING:
            cmd = self.uturn_behavior(cmd)
        elif self.state == self.REVERSE_APPROACH:
            cmd = self.reverse_approach_behavior(cmd)
        elif self.state == self.REVERSE_PASSING:
            cmd = self.reverse_passing_behavior(cmd)
        elif self.state == self.COMPLETED:
            cmd = self.completed_behavior(cmd)
        
        self.cmd_vel_pub.publish(cmd)
        
        # Publish state and pass number
        state_msg = String()
        state_msg.data = self.get_state_name()
        self.state_pub.publish(state_msg)
        
        pass_msg = String()
        pass_msg.data = f"PASS_{self.pass_number}"
        self.pass_number_pub.publish(pass_msg)
    
    def submerging_behavior(self, cmd: Twist) -> Twist:
        """Initial submersion from surface starting zone"""
        elapsed = time.time() - self.state_start_time
        
        if elapsed < self.submersion_duration:
            # Submerge while maintaining position
            cmd.linear.z = -self.submersion_speed
            cmd.linear.x = 0.05  # Tiny forward movement to leave zone
            self.get_logger().info(
                f'🌊 SUBMERGING: {elapsed:.1f}s / {self.submersion_duration:.1f}s',
                throttle_duration_sec=1.0
            )
        else:
            self.get_logger().info('✅ Submersion complete - starting search')
            self.transition_to(self.SEARCHING)
        
        return cmd
    
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
        
        direction = "LEFT" if sweep_phase < 0.5 else "RIGHT"
        self.get_logger().info(
            f'🔍 Searching ({direction})... {elapsed:.0f}s | Depth: {self.current_depth:.2f}m',
            throttle_duration_sec=3.0
        )
        
        return cmd
    
    def approaching_behavior(self, cmd: Twist) -> Twist:
        """CASUAL APPROACH: 5m → 3.5m with light correction"""
        if not self.gate_detected:
            if self.gate_lost_time > 0.0:
                lost_duration = time.time() - self.gate_lost_time
                if lost_duration > self.gate_lost_timeout:
                    self.get_logger().warn('❌ Gate lost - returning to search')
                    self.transition_to(self.SEARCHING)
                else:
                    cmd.linear.x = 0.2
            return cmd
        
        # Stop at alignment distance
        if self.estimated_distance <= self.approach_stop_distance:
            self.get_logger().info(
                f'🛑 Reached {self.approach_stop_distance:.1f}m - STOPPING to align'
            )
            self.transition_to(self.ALIGNING)
            return cmd
        
        # Casual approach
        cmd.linear.x = self.approach_speed
        cmd.angular.z = -self.frame_position * self.approach_yaw_gain
        
        self.get_logger().info(
            f'🚶 APPROACHING: dist={self.estimated_distance:.2f}m, '
            f'pos={self.frame_position:+.3f}, speed={cmd.linear.x:.2f}',
            throttle_duration_sec=0.5
        )
        
        return cmd
    
    def aligning_behavior(self, cmd: Twist) -> Twist:
        """PROPER ALIGNMENT AT 3.5m: Pure rotation"""
        if not self.gate_detected:
            if self.gate_lost_time > 0.0:
                lost_duration = time.time() - self.gate_lost_time
                if lost_duration > self.gate_lost_timeout:
                    self.get_logger().warn('❌ Gate lost during alignment - searching')
                    self.alignment_start_time = 0.0
                    self.transition_to(self.SEARCHING)
                else:
                    cmd.linear.x = 0.0
            return cmd
        
        if self.alignment_start_time == 0.0:
            self.alignment_start_time = time.time()
            self.get_logger().info(
                f'🎯 STARTING ALIGNMENT at {self.estimated_distance:.2f}m'
            )
        
        alignment_elapsed = time.time() - self.alignment_start_time
        
        if alignment_elapsed > self.alignment_max_time:
            self.get_logger().warn('⏰ Alignment timeout - proceeding')
            self.alignment_start_time = 0.0
            self.transition_to(self.FINAL_APPROACH)
            return cmd
        
        # Check alignment quality
        is_well_aligned = abs(self.frame_position) < self.alignment_threshold
        has_confidence = self.confidence > 0.7
        
        if is_well_aligned and has_confidence:
            self.get_logger().info(
                f'✅ ALIGNMENT COMPLETE! (took {alignment_elapsed:.1f}s)'
            )
            self.alignment_start_time = 0.0
            self.transition_to(self.FINAL_APPROACH)
            return cmd
        
        # Pure rotation alignment
        cmd.linear.x = 0.0
        cmd.angular.z = -self.frame_position * self.alignment_yaw_gain
        
        status = "MAJOR" if abs(self.frame_position) > 0.2 else "FINE"
        self.get_logger().info(
            f'🔄 ALIGNING ({status}): pos={self.frame_position:+.3f}, yaw={cmd.angular.z:+.2f}',
            throttle_duration_sec=0.3
        )
        
        return cmd
    
    def final_approach_behavior(self, cmd: Twist) -> Twist:
        """FINAL APPROACH: 3.5m → 1.8m with tight alignment"""
        if not self.gate_detected:
            if self.gate_lost_time > 0.0:
                lost_duration = time.time() - self.gate_lost_time
                if lost_duration > self.gate_lost_timeout:
                    self.get_logger().warn('❌ Gate lost - searching')
                    self.transition_to(self.SEARCHING)
                else:
                    cmd.linear.x = 0.1
            return cmd
        
        # Check if ready to commit to passing
        if self.estimated_distance <= self.passing_trigger_distance:
            if abs(self.frame_position) < 0.25:
                self.get_logger().info(
                    f'🚀 COMMITTING TO PASS #{self.pass_number} at {self.estimated_distance:.2f}m'
                )
                self.passing_start_position = self.current_position
                self.transition_to(self.PASSING)
                return cmd
            else:
                self.get_logger().error(
                    f'🚨 Misaligned at trigger! pos={self.frame_position:+.3f}'
                )
                cmd.linear.x = 0.0
                cmd.angular.z = -self.frame_position * self.alignment_yaw_gain * 2.0
                return cmd
        
        # Maintain alignment during approach
        if abs(self.frame_position) > self.final_approach_threshold:
            cmd.linear.x = self.final_approach_speed * 0.6
            cmd.angular.z = -self.frame_position * self.alignment_yaw_gain * 0.7
        else:
            cmd.linear.x = self.final_approach_speed
            cmd.angular.z = -self.frame_position * self.alignment_yaw_gain * 0.3
        
        self.get_logger().info(
            f'➡️ FINAL APPROACH: dist={self.estimated_distance:.2f}m, pos={self.frame_position:+.3f}',
            throttle_duration_sec=0.5
        )
        
        return cmd
    
    def passing_behavior(self, cmd: Twist) -> Twist:
        """FULL SPEED PASSAGE - Monitor clearance for transition"""
        if self.passing_start_position is None:
            self.passing_start_position = self.current_position
            self.get_logger().info(f'🚀 PASS #{self.pass_number} STARTED - FULL SPEED')
        
        # Check if AUV has completely cleared the gate
        # "Back-most part crosses line" = AUV front is (gate + AUV length + buffer) past gate
        if self.current_position:
            current_x = self.current_position[0]
            
            # Clearance threshold: gate position + AUV length + buffer
            clearance_threshold = (self.gate_x_position + 
                                 self.auv_length + self.gate_clearance_buffer)
            
            # For reverse pass, we're moving backward
            if self.pass_number == 2:
                clearance_threshold = (self.gate_x_position - 
                                     self.auv_length - self.gate_clearance_buffer)
            
            # Check clearance based on pass direction
            is_cleared = False
            if self.pass_number == 1 and current_x > clearance_threshold:
                is_cleared = True
            elif self.pass_number == 2 and current_x < clearance_threshold:
                is_cleared = True
            
            if is_cleared:
                distance_traveled = abs(current_x - self.passing_start_position[0])
                self.get_logger().info('='*70)
                self.get_logger().info(f'✅ PASS #{self.pass_number} CLEARED!')
                self.get_logger().info(f'   Position: X={current_x:.2f}m')
                self.get_logger().info(f'   Distance: {distance_traveled:.2f}m')
                self.get_logger().info('='*70)
                
                if self.pass_number == 1:
                    self.get_logger().info('🔄 INITIATING U-TURN FOR PASS #2')
                    self.pass_number = 2
                    self.transition_to(self.U_TURNING)
                else:
                    self.get_logger().info('🎉 QUALIFICATION COMPLETE!')
                    self.transition_to(self.COMPLETED)
                return cmd
            
            # Show progress
            if self.pass_number == 1:
                progress = (current_x - self.passing_start_position[0]) / clearance_threshold * 100
                status = "CLEARING GATE"
            else:
                progress = (self.passing_start_position[0] - current_x) / abs(clearance_threshold) * 100
                status = "CLEARING GATE REVERSE"
            
            self.get_logger().info(
                f'🚀 PASS #{self.pass_number} ({status}): {progress:.0f}% cleared',
                throttle_duration_sec=0.4
            )
        
        # Full speed - no corrections
        cmd.linear.x = self.passing_speed if self.pass_number == 1 else self.reverse_passing_speed
        cmd.linear.y = 0.0
        cmd.angular.z = 0.0
        
        return cmd
    
    def uturn_behavior(self, cmd: Twist) -> Twist:
        """U-TURN: Rotate 180° while maintaining depth"""
        if self.uturn_start_position is None:
            self.uturn_start_position = self.current_position
            self.get_logger().info('🔄 STARTING U-TURN (180° rotation)')
        
        elapsed = time.time() - self.state_start_time
        
        # Maintain depth with slight forward movement to clear gate area
        cmd.linear.z = (self.target_depth - self.current_depth) * self.uturn_depth_gain
        
        # Rotate continuously for U-turn
        cmd.angular.z = self.uturn_rotation_speed
        
        # Small forward movement to avoid hitting gate
        cmd.linear.x = 0.2
        
        # Check if turn is complete (duration-based)
        if elapsed > self.uturn_duration:
            self.get_logger().info('✅ U-TURN COMPLETE - Starting reverse approach')
            self.uturn_start_position = None
            self.passing_start_position = None
            self.transition_to(self.REVERSE_APPROACH)
        else:
            self.get_logger().info(
                f'🔄 U-TURNING: {elapsed:.1f}s / {self.uturn_duration:.1f}s',
                throttle_duration_sec=1.0
            )
        
        return cmd
    
    def reverse_approach_behavior(self, cmd: Twist) -> Twist:
        """REVERSE APPROACH: Move backward toward gate for second pass"""
        if not self.gate_detected:
            if self.gate_lost_time > 0.0:
                lost_duration = time.time() - self.gate_lost_time
                if lost_duration > self.gate_lost_timeout:
                    self.get_logger().warn('❌ Gate lost during reverse approach')
                    # Try to reacquire by rotating
                    cmd.linear.x = 0.0
                    cmd.angular.z = 0.3
                else:
                    cmd.linear.x = self.reverse_approach_speed * 0.5
            return cmd
        
        # Stop at appropriate distance (moving backward, so distance increases)
        if self.estimated_distance <= self.approach_stop_distance:
            self.get_logger().info(
                f'🛑 Reverse approach at {self.estimated_distance:.2f}m - aligning'
            )
            self.transition_to(self.ALIGNING)
            return cmd
        
        # Approach gate backward
        cmd.linear.x = self.reverse_approach_speed
        cmd.angular.z = -self.frame_position * self.approach_yaw_gain
        
        self.get_logger().info(
            f'⬅️ REVERSE APPROACH: dist={self.estimated_distance:.2f}m, '
            f'pos={self.frame_position:+.3f}',
            throttle_duration_sec=0.5
        )
        
        return cmd
    
    def reverse_passing_behavior(self, cmd: Twist) -> Twist:
        """REVERSE PASSING: Move backward through gate"""
        # Same logic as forward passing but with negative speed
        return self.passing_behavior(cmd)
    
    def completed_behavior(self, cmd: Twist) -> Twist:
        """Mission complete - stop all movement"""
        if self.mission_start_time:
            total_time = time.time() - self.mission_start_time
            detection_time = (self.gate_first_detected_time - self.mission_start_time 
                             if self.gate_first_detected_time else 0)
            
            self.get_logger().info('='*70)
            self.get_logger().info('🎉 QUALIFICATION COMPLETE!')
            self.get_logger().info(f'   Total time: {total_time:.2f}s')
            self.get_logger().info(f'   Points earned: 2 (both passes)')
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
        if new_state == self.SEARCHING and self.pass_number == 2:
            self.get_logger().info('🔁 STARTING REVERSE PASS SEQUENCE')
    
    def get_state_name(self) -> str:
        """Get human-readable state name"""
        names = {
            self.IDLE: 'IDLE',
            self.SUBMERGING: 'SUBMERGING',
            self.SEARCHING: 'SEARCHING',
            self.APPROACHING: 'APPROACHING',
            self.ALIGNING: 'ALIGNING',
            self.FINAL_APPROACH: 'FINAL_APPROACH',
            self.PASSING: 'PASSING',
            self.U_TURNING: 'U_TURNING',
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