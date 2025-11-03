#!/usr/bin/env python3
"""
Qualification Navigator - Complete Qualification Task
Sequence:
1. SEARCHING - Find gate
2. APPROACHING - Move toward gate
3. ALIGNING - Align with gate
4. PASSING_FORWARD - Pass through gate (1st time)
5. U_TURNING - Perform 180° turn
6. RETURNING - Navigate back to gate
7. ALIGNING_RETURN - Align for return pass
8. PASSING_RETURN - Pass through gate (2nd time)
9. HOMING - Return to starting position
10. SURFACING - Surface at starting position
11. COMPLETED - Task complete, stop
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
        self.SEARCHING = 0
        self.APPROACHING = 1
        self.ALIGNING = 2
        self.PASSING_FORWARD = 3
        self.U_TURNING = 4
        self.RETURNING = 5
        self.ALIGNING_RETURN = 6
        self.PASSING_RETURN = 7
        self.HOMING = 8
        self.SURFACING = 9
        self.COMPLETED = 10
        
        self.state = self.SEARCHING
        
        # Parameters
        self.declare_parameter('target_depth', -1.0)
        self.declare_parameter('search_speed', 0.4)
        self.declare_parameter('approach_speed', 0.6)
        self.declare_parameter('passing_speed', 0.8)
        self.declare_parameter('alignment_threshold', 0.1)
        self.declare_parameter('passing_trigger_distance', 1.5)
        self.declare_parameter('gate_clearance', 2.0)
        self.declare_parameter('u_turn_duration', 8.0)
        self.declare_parameter('home_position_x', -14.3)
        self.declare_parameter('home_position_y', 0.0)
        self.declare_parameter('home_tolerance', 0.5)
        
        self.target_depth = self.get_parameter('target_depth').value
        self.search_speed = self.get_parameter('search_speed').value
        self.approach_speed = self.get_parameter('approach_speed').value
        self.passing_speed = self.get_parameter('passing_speed').value
        self.alignment_threshold = self.get_parameter('alignment_threshold').value
        self.passing_trigger = self.get_parameter('passing_trigger_distance').value
        self.gate_clearance = self.get_parameter('gate_clearance').value
        self.u_turn_duration = self.get_parameter('u_turn_duration').value
        self.home_x = self.get_parameter('home_position_x').value
        self.home_y = self.get_parameter('home_position_y').value
        self.home_tolerance = self.get_parameter('home_tolerance').value
        
        # State variables
        self.gate_detected = False
        self.alignment_error = 0.0
        self.estimated_distance = 999.0
        self.current_depth = 0.0
        self.current_position = None
        self.current_yaw = 0.0
        
        # Position tracking
        self.gate_position_x = -4.0  # Known from world file
        self.passing_start_time = None
        self.u_turn_start_time = None
        self.state_start_time = time.time()
        
        # Mission tracking
        self.mission_start_time = time.time()
        self.forward_pass_time = None
        self.return_pass_time = None
        
        # Subscriptions
        self.gate_detected_sub = self.create_subscription(
            Bool, '/qual_gate/detected', self.gate_detected_callback, 10)
        self.alignment_sub = self.create_subscription(
            Float32, '/qual_gate/alignment_error', self.alignment_callback, 10)
        self.distance_sub = self.create_subscription(
            Float32, '/qual_gate/estimated_distance', self.distance_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/ground_truth/odom', self.odom_callback, 10)
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/rp2040/cmd_vel', 10)
        self.state_pub = self.create_publisher(String, '/qual/navigation_state', 10)
        
        # Control timer
        self.control_timer = self.create_timer(0.05, self.control_loop)
        
        self.get_logger().info('='*70)
        self.get_logger().info('✅ Qualification Navigator Started')
        self.get_logger().info('='*70)
        self.get_logger().info('   Sequence: SEARCH → APPROACH → ALIGN → PASS_FWD →')
        self.get_logger().info('             U_TURN → RETURN → ALIGN → PASS_RTN →')
        self.get_logger().info('             HOME → SURFACE → COMPLETE')
        self.get_logger().info('='*70)
    
    def gate_detected_callback(self, msg: Bool):
        self.gate_detected = msg.data
    
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
        
        # Extract yaw from quaternion
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)
    
    def control_loop(self):
        cmd = Twist()
        
        # Depth control
        depth_error = self.target_depth - self.current_depth
        if abs(depth_error) < 0.2:
            cmd.linear.z = 0.0
        else:
            cmd.linear.z = depth_error * 1.0
            cmd.linear.z = max(-0.8, min(cmd.linear.z, 0.8))
        
        # State machine
        if self.state == self.SEARCHING:
            cmd = self.searching_behavior(cmd)
        elif self.state == self.APPROACHING:
            cmd = self.approaching_behavior(cmd)
        elif self.state == self.ALIGNING:
            cmd = self.aligning_behavior(cmd)
        elif self.state == self.PASSING_FORWARD:
            cmd = self.passing_forward_behavior(cmd)
        elif self.state == self.U_TURNING:
            cmd = self.u_turning_behavior(cmd)
        elif self.state == self.RETURNING:
            cmd = self.returning_behavior(cmd)
        elif self.state == self.ALIGNING_RETURN:
            cmd = self.aligning_return_behavior(cmd)
        elif self.state == self.PASSING_RETURN:
            cmd = self.passing_return_behavior(cmd)
        elif self.state == self.HOMING:
            cmd = self.homing_behavior(cmd)
        elif self.state == self.SURFACING:
            cmd = self.surfacing_behavior(cmd)
        elif self.state == self.COMPLETED:
            cmd = self.completed_behavior(cmd)
        
        self.cmd_vel_pub.publish(cmd)
        
        # Publish state
        state_msg = String()
        state_msg.data = self.get_state_name()
        self.state_pub.publish(state_msg)
    
    def searching_behavior(self, cmd: Twist) -> Twist:
        """Search for gate with rotation"""
        if self.gate_detected:
            self.get_logger().info('🎯 Gate found - approaching')
            self.transition_to(self.APPROACHING)
            return cmd
        
        # Rotate to search
        cmd.linear.x = 0.2
        cmd.angular.z = 0.3
        
        elapsed = time.time() - self.state_start_time
        if int(elapsed) % 2 == 0:
            self.get_logger().info(f'🔍 Searching... {elapsed:.0f}s', throttle_duration_sec=1.9)
        
        return cmd
    
    def approaching_behavior(self, cmd: Twist) -> Twist:
        """Approach gate"""
        if not self.gate_detected:
            cmd.linear.x = 0.2
            return cmd
        
        if self.estimated_distance <= 2.5:
            self.get_logger().info('📍 Close to gate - aligning')
            self.transition_to(self.ALIGNING)
            return cmd
        
        cmd.linear.x = self.approach_speed
        cmd.angular.z = -self.alignment_error * 1.5
        
        self.get_logger().info(
            f'🚶 APPROACHING: dist={self.estimated_distance:.1f}m, align={self.alignment_error:+.2f}',
            throttle_duration_sec=0.5
        )
        
        return cmd
    
    def aligning_behavior(self, cmd: Twist) -> Twist:
        """Align with gate before passing"""
        if not self.gate_detected:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            return cmd
        
        if abs(self.alignment_error) < self.alignment_threshold:
            if self.estimated_distance <= self.passing_trigger:
                self.get_logger().info('🚀 ALIGNED - Passing forward')
                self.passing_start_time = time.time()
                self.transition_to(self.PASSING_FORWARD)
                return cmd
        
        cmd.linear.x = 0.3
        cmd.angular.z = -self.alignment_error * 2.5
        
        self.get_logger().info(
            f'🎯 ALIGNING: error={self.alignment_error:+.3f}',
            throttle_duration_sec=0.3
        )
        
        return cmd
    
    def passing_forward_behavior(self, cmd: Twist) -> Twist:
        """Pass through gate (forward)"""
        if self.current_position:
            if self.current_position[0] > self.gate_position_x + self.gate_clearance:
                self.forward_pass_time = time.time()
                elapsed = self.forward_pass_time - self.mission_start_time
                
                self.get_logger().info('='*70)
                self.get_logger().info('✅ FORWARD PASS COMPLETE!')
                self.get_logger().info(f'   Time: {elapsed:.2f}s')
                self.get_logger().info('   Starting U-turn...')
                self.get_logger().info('='*70)
                
                self.u_turn_start_time = time.time()
                self.transition_to(self.U_TURNING)
                return cmd
        
        cmd.linear.x = self.passing_speed
        cmd.angular.z = 0.0
        
        self.get_logger().info('🚀 PASSING FORWARD...', throttle_duration_sec=0.5)
        
        return cmd
    
    def u_turning_behavior(self, cmd: Twist) -> Twist:
        """Perform 180° U-turn"""
        elapsed = time.time() - self.u_turn_start_time
        
        if elapsed >= self.u_turn_duration:
            self.get_logger().info('🔄 U-turn complete - returning to gate')
            self.transition_to(self.RETURNING)
            return cmd
        
        # Execute turn
        cmd.linear.x = 0.2
        cmd.angular.z = -0.8  # Turn right
        
        progress = (elapsed / self.u_turn_duration) * 100
        self.get_logger().info(
            f'🔄 U-TURNING: {progress:.0f}% ({elapsed:.1f}s/{self.u_turn_duration:.1f}s)',
            throttle_duration_sec=0.5
        )
        
        return cmd
    
    def returning_behavior(self, cmd: Twist) -> Twist:
        """Return to gate"""
        if self.gate_detected:
            if self.estimated_distance <= 2.5:
                self.get_logger().info('📍 Close to gate - aligning for return')
                self.transition_to(self.ALIGNING_RETURN)
                return cmd
        
        cmd.linear.x = self.approach_speed
        
        if self.gate_detected:
            cmd.angular.z = -self.alignment_error * 1.5
            self.get_logger().info(
                f'↩️ RETURNING: dist={self.estimated_distance:.1f}m',
                throttle_duration_sec=0.5
            )
        else:
            cmd.angular.z = 0.2  # Gentle search
            self.get_logger().info('↩️ RETURNING (searching gate)...', throttle_duration_sec=1.0)
        
        return cmd
    
    def aligning_return_behavior(self, cmd: Twist) -> Twist:
        """Align for return pass"""
        if not self.gate_detected:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            return cmd
        
        if abs(self.alignment_error) < self.alignment_threshold:
            if self.estimated_distance <= self.passing_trigger:
                self.get_logger().info('🚀 ALIGNED - Passing return')
                self.passing_start_time = time.time()
                self.transition_to(self.PASSING_RETURN)
                return cmd
        
        cmd.linear.x = 0.3
        cmd.angular.z = -self.alignment_error * 2.5
        
        self.get_logger().info(
            f'🎯 ALIGNING RETURN: error={self.alignment_error:+.3f}',
            throttle_duration_sec=0.3
        )
        
        return cmd
    
    def passing_return_behavior(self, cmd: Twist) -> Twist:
        """Pass through gate (return)"""
        if self.current_position:
            if self.current_position[0] < self.gate_position_x - self.gate_clearance:
                self.return_pass_time = time.time()
                elapsed = self.return_pass_time - self.mission_start_time
                
                self.get_logger().info('='*70)
                self.get_logger().info('✅ RETURN PASS COMPLETE!')
                self.get_logger().info(f'   Total time: {elapsed:.2f}s')
                self.get_logger().info('   Qualification points: 2 ✓')
                self.get_logger().info('   Returning home...')
                self.get_logger().info('='*70)
                
                self.transition_to(self.HOMING)
                return cmd
        
        cmd.linear.x = self.passing_speed
        cmd.angular.z = 0.0
        
        self.get_logger().info('🚀 PASSING RETURN...', throttle_duration_sec=0.5)
        
        return cmd
    
    def homing_behavior(self, cmd: Twist) -> Twist:
        """Return to starting position"""
        if not self.current_position:
            cmd.linear.x = 0.0
            return cmd
        
        # Calculate distance to home
        dx = self.home_x - self.current_position[0]
        dy = self.home_y - self.current_position[1]
        distance_to_home = math.sqrt(dx*dx + dy*dy)
        
        if distance_to_home < self.home_tolerance:
            self.get_logger().info('🏠 HOME REACHED - Surfacing')
            self.transition_to(self.SURFACING)
            return cmd
        
        # Calculate heading to home
        target_yaw = math.atan2(dy, dx)
        yaw_error = target_yaw - self.current_yaw
        
        # Normalize yaw error to [-pi, pi]
        while yaw_error > math.pi:
            yaw_error -= 2 * math.pi
        while yaw_error < -math.pi:
            yaw_error += 2 * math.pi
        
        cmd.linear.x = 0.6
        cmd.angular.z = yaw_error * 1.0
        
        self.get_logger().info(
            f'🏠 HOMING: dist={distance_to_home:.1f}m, heading={math.degrees(target_yaw):.0f}°',
            throttle_duration_sec=0.5
        )
        
        return cmd
    
    def surfacing_behavior(self, cmd: Twist) -> Twist:
        """Surface at starting position"""
        if self.current_depth >= -0.2:
            self.get_logger().info('✅ SURFACED - Task complete')
            self.transition_to(self.COMPLETED)
            return cmd
        
        cmd.linear.x = 0.0
        cmd.linear.y = 0.0
        cmd.linear.z = 1.0  # Ascend
        cmd.angular.z = 0.0
        
        self.get_logger().info(
            f'⬆️ SURFACING: depth={self.current_depth:.2f}m',
            throttle_duration_sec=0.5
        )
        
        return cmd
    
    def completed_behavior(self, cmd: Twist) -> Twist:
        """Task complete - stop"""
        if self.mission_start_time:
            total_time = time.time() - self.mission_start_time
            
            self.get_logger().info('='*70)
            self.get_logger().info('🎉 QUALIFICATION TASK COMPLETE!')
            self.get_logger().info('='*70)
            self.get_logger().info(f'   Total mission time: {total_time:.2f}s')
            if self.forward_pass_time:
                fwd_time = self.forward_pass_time - self.mission_start_time
                self.get_logger().info(f'   Forward pass: {fwd_time:.2f}s')
            if self.return_pass_time:
                rtn_time = self.return_pass_time - self.forward_pass_time
                self.get_logger().info(f'   Return pass: {rtn_time:.2f}s')
            self.get_logger().info('   Qualification points: 2/2 ✓')
            self.get_logger().info('   Status: QUALIFIED FOR FINAL ROUND')
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
            self.SEARCHING: 'SEARCHING',
            self.APPROACHING: 'APPROACHING',
            self.ALIGNING: 'ALIGNING',
            self.PASSING_FORWARD: 'PASSING_FORWARD',
            self.U_TURNING: 'U_TURNING',
            self.RETURNING: 'RETURNING',
            self.ALIGNING_RETURN: 'ALIGNING_RETURN',
            self.PASSING_RETURN: 'PASSING_RETURN',
            self.HOMING: 'HOMING',
            self.SURFACING: 'SURFACING',
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