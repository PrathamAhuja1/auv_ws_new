#!/usr/bin/env python3
"""
SMART Gate Navigator - Smooth Flare Avoidance with Gate Tracking
Key improvements:
1. Only avoids flare when very close (< 2m)
2. Maintains gate visibility during avoidance
3. Minimal lateral movement - just enough to clear flare
4. Returns to center alignment after clearing flare
5. Proper depth control throughout
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
import time
import math


class SmartGateNavigator(Node):
    def __init__(self):
        super().__init__('gate_navigator_node')
        
        # State machine
        self.IDLE = 0
        self.SEARCHING = 1
        self.CENTERING = 2
        self.APPROACHING = 3
        self.AVOIDING_FLARE = 4
        self.PASSING = 5
        self.COMPLETED = 6
        
        self.state = self.SEARCHING
        
        # Parameters - OPTIMIZED
        self.declare_parameter('target_depth', -1.7)
        self.declare_parameter('depth_correction_gain', 2.0)
        self.declare_parameter('search_forward_speed', 0.5)
        self.declare_parameter('search_rotation_speed', 0.15)
        self.declare_parameter('approach_speed', 0.7)
        self.declare_parameter('passing_speed', 1.0)
        self.declare_parameter('passing_duration', 8.0)
        self.declare_parameter('yaw_correction_gain', 2.5)
        self.declare_parameter('approach_distance', 3.0)
        self.declare_parameter('passing_distance', 1.5)
        self.declare_parameter('centering_threshold', 0.12)
        self.declare_parameter('centering_max_time', 10.0)
        
        # CRITICAL: Flare avoidance parameters
        self.declare_parameter('flare_critical_distance', 2.0)  # Only avoid when < 2m
        self.declare_parameter('flare_lateral_speed', 0.4)  # Gentle lateral movement
        self.declare_parameter('flare_forward_speed', 0.3)  # Keep moving forward
        self.declare_parameter('flare_max_duration', 10.0)  # Max 10s avoidance
        
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
        self.centering_threshold = self.get_parameter('centering_threshold').value
        self.centering_max_time = self.get_parameter('centering_max_time').value
        
        self.flare_critical_distance = self.get_parameter('flare_critical_distance').value
        self.flare_lateral_speed = self.get_parameter('flare_lateral_speed').value
        self.flare_forward_speed = self.get_parameter('flare_forward_speed').value
        self.flare_max_duration = self.get_parameter('flare_max_duration').value
        
        self.gate_lost_timeout = 3.0
        
        # State variables
        self.gate_detected = False
        self.flare_detected = False
        self.partial_gate = False
        self.alignment_error = 0.0
        self.estimated_distance = 999.0
        self.current_depth = 0.0
        self.frame_position = 0.0
        self.confidence = 0.0
        
        # Flare tracking
        self.flare_position = None  # (x, y) in image frame
        self.flare_distance = 999.0
        self.flare_avoidance_direction = 0.0
        
        # Current position (for flare distance calculation)
        self.current_position = None
        self.flare_world_position = (-14.0, 2.0, -2.25)  # From world file
        
        # Timing variables
        self.gate_lost_time = 0.0
        self.passing_start_time = 0.0
        self.flare_avoidance_start_time = 0.0
        self.centering_start_time = 0.0
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
        self.flare_warning_sub = self.create_subscription(
            String, '/flare/warning', self.flare_warning_callback, 10)
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/rp2040/cmd_vel', 10)
        self.state_pub = self.create_publisher(String, '/gate/navigation_state', 10)
        
        self.control_timer = self.create_timer(0.05, self.control_loop)
        
        self.get_logger().info('='*70)
        self.get_logger().info('✅ SMART Gate Navigator with Smooth Flare Avoidance')
        self.get_logger().info('='*70)
        self.get_logger().info(f'   Target depth: {self.target_depth}m')
        self.get_logger().info(f'   Flare critical distance: {self.flare_critical_distance}m')
        self.get_logger().info(f'   Flare lateral speed: {self.flare_lateral_speed} m/s (gentle)')
        self.get_logger().info(f'   Centering threshold: ±{self.centering_threshold*100:.0f}%')
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
    
    def flare_warning_callback(self, msg: String):
        # Extract flare X position from warning
        try:
            if "X=" in msg.data:
                x_str = msg.data.split("X=")[1]
                flare_x = float(x_str)
                # Store flare position for distance calculation
                # (This is image X coordinate, not world position)
        except:
            pass
    
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
    
    def should_avoid_flare(self) -> bool:
        """
        CRITICAL LOGIC: Only avoid flare when:
        1. Flare is detected
        2. Distance < critical distance (2m)
        3. Gate is still visible (to maintain tracking)
        """
        if not self.flare_detected:
            return False
        
        if self.flare_distance >= self.flare_critical_distance:
            return False
        
        # IMPORTANT: Only avoid if we can see the gate
        # This prevents losing gate during avoidance
        if not self.gate_detected and self.state != self.AVOIDING_FLARE:
            return False
        
        return True
    
    def control_loop(self):
        cmd = Twist()
        
        # FIXED DEPTH CONTROL - Proper deadband to prevent oscillation
        depth_error = self.target_depth - self.current_depth
        depth_deadband = 0.3  # No correction within ±30cm
        
        if abs(depth_error) < depth_deadband:
            # Within deadband - NO correction
            cmd.linear.z = 0.0
        elif abs(depth_error) < 0.6:
            # Small error - gentle correction
            cmd.linear.z = depth_error * 0.8
            cmd.linear.z = max(-0.4, min(cmd.linear.z, 0.4))
        else:
            # Large error - stronger correction
            cmd.linear.z = depth_error * 1.2
            cmd.linear.z = max(-1.0, min(cmd.linear.z, 1.0))
            self.get_logger().warn(
                f'⚠️ Depth error: {depth_error:.2f}m',
                throttle_duration_sec=2.0
            )
        
        # Check flare avoidance
        if self.should_avoid_flare() and self.state in [self.SEARCHING, self.CENTERING, self.APPROACHING]:
            self.get_logger().warn(
                f'🚧 FLARE CLOSE ({self.flare_distance:.2f}m) - Gentle avoidance',
                throttle_duration_sec=1.0
            )
            if self.flare_avoidance_start_time == 0.0:
                self.flare_avoidance_start_time = time.time()
            self.state = self.AVOIDING_FLARE
        
        # State machine
        if self.state == self.SEARCHING:
            cmd = self.searching_behavior(cmd)
        elif self.state == self.CENTERING:
            cmd = self.centering_behavior(cmd)
        elif self.state == self.APPROACHING:
            cmd = self.approaching_behavior(cmd)
        elif self.state == self.AVOIDING_FLARE:
            cmd = self.avoiding_flare_behavior(cmd)
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
        """Search for gate with oscillating pattern"""
        if self.gate_detected:
            self.get_logger().info('🎯 Gate found - transitioning to CENTERING')
            self.transition_to(self.CENTERING)
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
                f'🔍 Searching ({direction})... {elapsed:.0f}s | Depth: {self.current_depth:.2f}m',
                throttle_duration_sec=2.9
            )
        
        return cmd
    
    def centering_behavior(self, cmd: Twist) -> Twist:
        """PROPER ALIGNMENT: Get gate centered before approaching"""
        
        if not self.gate_detected:
            if self.gate_lost_time > 0.0:
                lost_duration = time.time() - self.gate_lost_time
                if lost_duration > self.gate_lost_timeout:
                    self.get_logger().warn('❌ Gate lost during centering - returning to search')
                    self.transition_to(self.SEARCHING)
                else:
                    cmd.linear.x = 0.0
                    cmd.angular.z = -self.frame_position * self.yaw_correction_gain
                    self.get_logger().warn(
                        f'🔍 Gate lost - holding position',
                        throttle_duration_sec=0.5
                    )
            return cmd
        
        # CRITICAL: Proper alignment for safe passage
        # Center the gate PROPERLY before approaching
        
        # Calculate how well centered we are
        alignment_quality = abs(self.frame_position)
        
        # Good alignment: within ±15% of center
        is_well_aligned = alignment_quality < 0.15
        
        # Has confidence (full gate visible)
        has_confidence = self.confidence > 0.8 and not self.partial_gate
        
        # Check centering timeout
        if self.centering_start_time == 0.0:
            self.centering_start_time = time.time()
        
        centering_elapsed = time.time() - self.centering_start_time
        
        if centering_elapsed > 8.0:
            self.get_logger().warn('⏰ Centering timeout - proceeding anyway')
            self.centering_start_time = 0.0
            self.transition_to(self.APPROACHING)
            return cmd
        
        # If well aligned and confident, proceed
        if is_well_aligned and has_confidence:
            self.get_logger().info(
                f'✅ GATE PROPERLY ALIGNED (pos={self.frame_position:+.2f}) - APPROACHING'
            )
            self.centering_start_time = 0.0
            self.transition_to(self.APPROACHING)
            return cmd
        
        # ALIGNMENT STRATEGY:
        # 1. Stop forward motion to align properly
        # 2. Apply pure yaw rotation to center gate
        # 3. Once aligned, move forward
        
        yaw_correction = -self.frame_position * 3.0  # Strong yaw gain for alignment
        
        if alignment_quality > 0.3:
            # Far from center - STOP and rotate
            cmd.linear.x = 0.0
            cmd.angular.z = yaw_correction
            self.get_logger().info(
                f'🔄 ALIGNING (far): pos={self.frame_position:+.2f}, yaw={cmd.angular.z:+.2f}',
                throttle_duration_sec=0.3
            )
        elif alignment_quality > 0.15:
            # Moderately off - Slow forward + rotation
            cmd.linear.x = 0.2
            cmd.angular.z = yaw_correction * 0.8
            self.get_logger().info(
                f'🔄 ALIGNING (moderate): pos={self.frame_position:+.2f}, yaw={cmd.angular.z:+.2f}',
                throttle_duration_sec=0.3
            )
        else:
            # Nearly centered - gentle correction
            cmd.linear.x = 0.3
            cmd.angular.z = yaw_correction * 0.5
            self.get_logger().info(
                f'🎯 FINE TUNING: pos={self.frame_position:+.2f}, yaw={cmd.angular.z:+.2f}',
                throttle_duration_sec=0.3
            )
        
        return cmd
    
    def approaching_behavior(self, cmd: Twist) -> Twist:
        """Approach gate while keeping it in frame"""
        
        if not self.gate_detected:
            if self.gate_lost_time > 0.0:
                lost_duration = time.time() - self.gate_lost_time
                if lost_duration > self.gate_lost_timeout:
                    self.get_logger().warn('❌ Gate lost - returning to search')
                    self.transition_to(self.SEARCHING)
                else:
                    cmd.linear.x = 0.1
                    cmd.angular.z = -self.frame_position * self.yaw_correction_gain
            return cmd
        
        # Check if close enough to start passing
        if self.estimated_distance < self.passing_distance and self.estimated_distance > 0:
            self.get_logger().info(f'🚀 Within passing distance ({self.estimated_distance:.2f}m)')
            self.transition_to(self.PASSING)
            return cmd
        
        # Add hysteresis to prevent oscillation
        if not hasattr(self, 'frame_violation_count'):
            self.frame_violation_count = 0
        
        if abs(self.frame_position) > 0.7:
            self.frame_violation_count += 1
        else:
            self.frame_violation_count = 0
        
        # Only trigger re-center after 10 consecutive violations (0.5 seconds)
        if self.frame_violation_count > 10:
            self.get_logger().warn(f'⚠️ Gate sustained edge violation - re-center')
            self.frame_violation_count = 0
            self.transition_to(self.CENTERING)
            return cmd
        
        # APPROACH: Just move forward with gentle yaw
        cmd.linear.x = self.approach_speed
        cmd.angular.z = -self.frame_position * self.yaw_correction_gain * 0.3  # Even gentler
        
        if self.partial_gate or self.confidence < 0.8:
            cmd.linear.x *= 0.8
        
        if self.estimated_distance < 999:
            self.get_logger().info(
                f'➡️ APPROACH: dist={self.estimated_distance:.2f}m, pos={self.frame_position:+.2f}',
                throttle_duration_sec=0.5
            )
        
        return cmd
    
    def avoiding_flare_behavior(self, cmd: Twist) -> Twist:
        """
        CRITICAL: Smart flare avoidance
        1. Minimal lateral movement (just enough to clear)
        2. Continue moving forward (don't stop)
        3. Maintain gate tracking via yaw correction
        4. Exit as soon as flare is clear
        """
        
        if self.flare_avoidance_start_time == 0.0:
            self.flare_avoidance_start_time = time.time()
        
        elapsed = time.time() - self.flare_avoidance_start_time
        
        # Check exit conditions
        flare_cleared = not self.flare_detected and elapsed > 1.0
        flare_far = self.flare_distance > self.flare_critical_distance + 0.5
        timeout = elapsed > self.flare_max_duration
        
        if flare_cleared or flare_far:
            self.get_logger().info('✅ Flare cleared - resuming navigation')
            self.flare_avoidance_start_time = 0.0
            
            if self.gate_detected:
                if abs(self.frame_position) > 0.2:
                    self.transition_to(self.CENTERING)
                else:
                    self.transition_to(self.APPROACHING)
            else:
                self.transition_to(self.SEARCHING)
            return cmd
        
        if timeout:
            self.get_logger().warn('⏰ Flare avoidance timeout - forcing exit')
            self.flare_avoidance_start_time = 0.0
            self.transition_to(self.SEARCHING)
            return cmd
        
        # SMART AVOIDANCE STRATEGY:
        # 1. Small lateral movement to side
        # 2. Keep moving forward (don't stop!)
        # 3. Correct yaw to maintain gate tracking
        
        # Gentle lateral movement
        lateral_speed = self.flare_avoidance_direction * self.flare_lateral_speed
        
        # CRITICAL: Reduce lateral if gate is off-center
        # This prevents over-correction that loses gate
        if self.gate_detected:
            if abs(self.frame_position) > 0.4:
                # Gate is getting off-center, reduce lateral movement
                lateral_speed *= 0.3
                self.get_logger().warn(
                    f'⚠️ Reducing lateral - gate drifting (pos={self.frame_position:+.2f})',
                    throttle_duration_sec=0.5
                )
            
            # Maintain alignment with gate
            cmd.angular.z = -self.frame_position * self.yaw_correction_gain * 0.5
        else:
            # Lost gate - gentle search while avoiding
            cmd.angular.z = 0.2 * self.flare_avoidance_direction
            self.get_logger().error(
                '❌ GATE LOST DURING AVOIDANCE - gentle recovery',
                throttle_duration_sec=0.5
            )
        
        cmd.linear.y = lateral_speed
        cmd.linear.x = self.flare_forward_speed  # Keep moving forward
        
        self.get_logger().warn(
            f'🚧 AVOIDING: lat={cmd.linear.y:+.2f}, fwd={cmd.linear.x:.2f}, '
            f'yaw={cmd.angular.z:+.2f}, dist={self.flare_distance:.2f}m, '
            f'gate_pos={self.frame_position:+.2f}, t={elapsed:.1f}s',
            throttle_duration_sec=0.3
        )
        
        return cmd
    
    def passing_behavior(self, cmd: Twist) -> Twist:
        """Pass through gate at full speed"""
        if self.passing_start_time == 0.0:
            self.passing_start_time = time.time()
            self.get_logger().info('🚀 PASSING THROUGH GATE - FULL SPEED!')
        
        elapsed = time.time() - self.passing_start_time
        
        if elapsed > self.passing_duration:
            self.get_logger().info('✅ GATE PASSAGE COMPLETE')
            self.transition_to(self.COMPLETED)
            return cmd
        
        cmd.linear.x = self.passing_speed
        cmd.linear.y = 0.0
        cmd.angular.z = 0.0
        
        self.get_logger().info(
            f'🚀 PASSING... {elapsed:.1f}s / {self.passing_duration}s',
            throttle_duration_sec=0.5
        )
        
        return cmd
    
    def completed_behavior(self, cmd: Twist) -> Twist:
        """Mission complete"""
        if self.mission_start_time:
            total_time = time.time() - self.mission_start_time
            detection_time = (self.gate_first_detected_time - self.mission_start_time 
                             if self.gate_first_detected_time else 0)
            
            self.get_logger().info('='*70)
            self.get_logger().info('🎉 GATE NAVIGATION MISSION COMPLETE')
            self.get_logger().info(f'   Total time: {total_time:.2f}s')
            self.get_logger().info(f'   Detection: {detection_time:.2f}s')
            self.get_logger().info(f'   Navigation: {total_time - detection_time:.2f}s')
            self.get_logger().info('='*70)
        
        cmd.linear.x = 0.0
        cmd.linear.y = 0.0
        cmd.linear.z = 0.0
        cmd.angular.z = 0.0
        
        if self.control_timer:
            self.control_timer.cancel()
            self.control_timer = None
        
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
            self.CENTERING: 'CENTERING',
            self.APPROACHING: 'APPROACHING',
            self.AVOIDING_FLARE: 'AVOIDING_FLARE',
            self.PASSING: 'PASSING',
            self.COMPLETED: 'COMPLETED'
        }
        return names.get(self.state, 'UNKNOWN')


def main(args=None):
    rclpy.init(args=args)
    node = SmartGateNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop_cmd = Twist()
        node.cmd_vel_pub.publish(stop_cmd)
        node.get_logger().info('Smart Gate Navigator shutting down')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()