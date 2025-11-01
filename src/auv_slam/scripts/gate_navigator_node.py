#!/usr/bin/env python3
"""
IMPROVED Gate Navigator - Better depth control and centering
Key improvements:
1. Optimized target depth for gate detection
2. More aggressive centering with no timeout if gate visible
3. Better state transitions
4. Proper depth maintenance
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import time


class ImprovedGateNavigator(Node):
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
        
        # Parameters - OPTIMIZED for gate passage
        self.declare_parameter('target_depth', -1.7)  # Slightly below gate center for better view
        self.declare_parameter('depth_correction_gain', 2.0)  # Increased for faster response
        self.declare_parameter('search_forward_speed', 0.5)
        self.declare_parameter('search_rotation_speed', 0.15)
        self.declare_parameter('approach_speed', 0.7)
        self.declare_parameter('passing_speed', 1.0)
        self.declare_parameter('passing_duration', 8.0)
        self.declare_parameter('yaw_correction_gain', 2.5)  # More aggressive centering
        self.declare_parameter('approach_distance', 3.0)
        self.declare_parameter('passing_distance', 1.5)
        self.declare_parameter('flare_avoidance_gain', 0.8)
        self.declare_parameter('flare_avoidance_duration', 3.0)
        self.declare_parameter('centering_threshold', 0.12)  # Tighter tolerance
        self.declare_parameter('centering_max_time', 10.0)  # Longer timeout
        
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
        self.centering_threshold = self.get_parameter('centering_threshold').value
        self.centering_max_time = self.get_parameter('centering_max_time').value
        
        self.gate_lost_timeout = 3.0
        
        # State variables
        self.gate_detected = False
        self.flare_detected = False
        self.partial_gate = False
        self.alignment_error = 0.0
        self.estimated_distance = 999.0
        self.current_depth = 0.0
        self.flare_avoidance_direction = 0.0
        self.frame_position = 0.0
        self.confidence = 0.0
        
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
        
        self.get_logger().info('✅ IMPROVED Gate Navigator READY')
        self.get_logger().info(f'   Target depth: {self.target_depth}m (optimized for gate center)')
        self.get_logger().info(f'   Centering threshold: ±{self.centering_threshold*100:.0f}%')
        self.get_logger().info(f'   Depth gain: {self.depth_gain} (faster response)')
        self.get_logger().info(f'State: {self.get_state_name()}')
    
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
        was_detected = self.flare_detected
        self.flare_detected = msg.data
        
        if not was_detected and self.flare_detected:
            self.get_logger().error('🚨 ORANGE FLARE DETECTED - INITIATING SMART AVOIDANCE')
            if self.state in [self.SEARCHING, self.CENTERING, self.APPROACHING]:
                self.flare_avoidance_start_time = time.time()
                self.transition_to(self.AVOIDING_FLARE)
        elif was_detected and not self.flare_detected:
            self.get_logger().info('✅ Flare cleared')
    
    def flare_avoidance_callback(self, msg: Float32):
        self.flare_avoidance_direction = msg.data
    
    def flare_warning_callback(self, msg: String):
        self.get_logger().warn(msg.data)
    
    def alignment_callback(self, msg: Float32):
        self.alignment_error = msg.data
    
    def distance_callback(self, msg: Float32):
        self.estimated_distance = msg.data
    
    def odom_callback(self, msg: Odometry):
        self.current_depth = msg.pose.pose.position.z
    
    def control_loop(self):
        cmd = Twist()
        
        # CRITICAL: Depth control (always active) - NO INVERSION
        # If we're too high (current=-1m, target=-2m), error=-1, cmd.z=-2 (descend)
        depth_error = self.target_depth - self.current_depth
        cmd.linear.z = depth_error * self.depth_gain
        
        # Clamp depth command for safety
        cmd.linear.z = max(-1.5, min(cmd.linear.z, 1.5))
        
        # Trigger flare avoidance if flare appears
        if self.flare_detected and self.state in [self.SEARCHING, self.CENTERING, self.APPROACHING]:
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
        
        # Oscillating search
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
        """
        IMPROVED: More aggressive centering, longer timeout
        Only proceed when gate is well-centered OR timeout reached
        """
        
        # Check if gate is lost
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
                        f'🔍 Gate lost - holding (last pos: {self.frame_position:+.2f})',
                        throttle_duration_sec=0.5
                    )
            return cmd
        
        # Initialize centering timer
        if self.centering_start_time == 0.0:
            self.centering_start_time = time.time()
        
        centering_elapsed = time.time() - self.centering_start_time
        
        # IMPROVED: Only timeout if we've been trying for a long time
        if centering_elapsed > self.centering_max_time:
            self.get_logger().warn(f'⏰ Centering max time reached ({centering_elapsed:.1f}s) - proceeding')
            self.centering_start_time = 0.0
            self.transition_to(self.APPROACHING)
            return cmd
        
        # Calculate yaw correction from frame position
        yaw_correction = -self.frame_position * self.yaw_correction_gain
        
        # Check if centered
        is_centered = abs(self.frame_position) < self.centering_threshold
        is_confident = self.confidence > 0.7
        
        if is_centered and is_confident:
            self.get_logger().info('✅ Gate CENTERED - transitioning to APPROACH')
            self.centering_start_time = 0.0
            self.transition_to(self.APPROACHING)
            return cmd
        
        # CENTERING MOTION: NO FORWARD, ONLY YAW
        cmd.linear.x = 0.0
        cmd.linear.y = 0.0
        cmd.angular.z = yaw_correction
        
        # Extra aggressive if near edge
        if abs(self.frame_position) > 0.6:
            cmd.angular.z *= 1.8
            self.get_logger().warn(
                f'🚨 GATE NEAR EDGE (pos={self.frame_position:+.2f}) - AGGRESSIVE YAW',
                throttle_duration_sec=0.3
            )
        
        # Log progress
        self.get_logger().info(
            f'🎯 CENTERING: pos={self.frame_position:+.2f}, '
            f'yaw={cmd.angular.z:+.2f}, conf={self.confidence:.2f}, '
            f't={centering_elapsed:.1f}s, depth={self.current_depth:.2f}m',
            throttle_duration_sec=0.3
        )
        
        return cmd
    
    def approaching_behavior(self, cmd: Twist) -> Twist:
        """Approach gate while maintaining center alignment"""
        
        if not self.gate_detected:
            if self.gate_lost_time > 0.0:
                lost_duration = time.time() - self.gate_lost_time
                if lost_duration > self.gate_lost_timeout:
                    self.get_logger().warn(f'❌ Gate lost for {lost_duration:.1f}s - returning to search')
                    self.transition_to(self.SEARCHING)
                else:
                    cmd.linear.x = 0.1
                    cmd.angular.z = -self.frame_position * self.yaw_correction_gain
                    self.get_logger().warn(
                        f'🔍 Gate lost - slowing (last pos: {self.frame_position:+.2f})',
                        throttle_duration_sec=0.5
                    )
            return cmd
        
        # Check if ready to pass
        if self.estimated_distance < self.passing_distance and self.estimated_distance > 0:
            self.get_logger().info(f'🚀 Within passing distance ({self.estimated_distance:.2f}m)')
            self.transition_to(self.PASSING)
            return cmd
        
        # If gate drifts too far, go back to centering
        if abs(self.frame_position) > 0.35:
            self.get_logger().warn(
                f'⚠️ Gate drifted (pos={self.frame_position:+.2f}) - RE-CENTERING'
            )
            self.transition_to(self.CENTERING)
            return cmd
        
        # Approach with gentle alignment correction
        cmd.linear.x = self.approach_speed
        cmd.angular.z = -self.frame_position * self.yaw_correction_gain * 0.6
        
        # Reduce speed for partial gate
        if self.partial_gate or self.confidence < 0.8:
            cmd.linear.x *= 0.7
        
        if self.estimated_distance < 999:
            self.get_logger().info(
                f'➡️ APPROACH: dist={self.estimated_distance:.2f}m, '
                f'pos={self.frame_position:+.2f}, yaw={cmd.angular.z:+.2f}, '
                f'depth={self.current_depth:.2f}m',
                throttle_duration_sec=0.4
            )
        
        return cmd
    
    def avoiding_flare_behavior(self, cmd: Twist) -> Twist:
        """Frame-aware flare avoidance"""
        
        if self.flare_avoidance_start_time == 0.0:
            self.flare_avoidance_start_time = time.time()
        
        elapsed = time.time() - self.flare_avoidance_start_time
        
        if not self.flare_detected and elapsed > 1.0:
            self.get_logger().info('✅ Flare cleared for 1s - resuming navigation')
            self.flare_avoidance_start_time = 0.0
            self.transition_to(self.CENTERING if self.gate_detected else self.SEARCHING)
            return cmd
        
        if elapsed > 10.0:
            self.get_logger().warn('⏰ Flare avoidance timeout - returning to search')
            self.flare_avoidance_start_time = 0.0
            self.transition_to(self.SEARCHING)
            return cmd
        
        # Smart avoidance considering gate position
        base_lateral_speed = self.flare_avoidance_direction * 0.8
        
        if self.gate_detected:
            if self.frame_position < -0.3 and base_lateral_speed < 0:
                base_lateral_speed *= 0.3
            elif self.frame_position > 0.3 and base_lateral_speed > 0:
                base_lateral_speed *= 0.3
            
            cmd.angular.z = -self.frame_position * self.yaw_correction_gain * 0.5
        else:
            self.get_logger().error('❌ GATE LOST DURING FLARE AVOIDANCE')
            cmd.angular.z = 0.3
            base_lateral_speed *= 0.5
        
        cmd.linear.y = base_lateral_speed
        cmd.linear.x = 0.3
        
        self.get_logger().warn(
            f'🚧 AVOIDING: lat={cmd.linear.y:.2f}, yaw={cmd.angular.z:.2f}, '
            f'pos={self.frame_position:+.2f}, t={elapsed:.1f}s',
            throttle_duration_sec=0.5
        )
        
        return cmd
    
    def passing_behavior(self, cmd: Twist) -> Twist:
        """Pass through gate at full speed"""
        if self.passing_start_time == 0.0:
            self.passing_start_time = time.time()
            self.get_logger().info('🚀 PASSING THROUGH GATE - FULL SPEED AHEAD!')
        
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
            
            self.get_logger().info('🎉 GATE NAVIGATION MISSION COMPLETE')
            self.get_logger().info(f'   Total time: {total_time:.2f}s')
            self.get_logger().info(f'   Detection: {detection_time:.2f}s')
            self.get_logger().info(f'   Navigation: {total_time - detection_time:.2f}s')
        
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