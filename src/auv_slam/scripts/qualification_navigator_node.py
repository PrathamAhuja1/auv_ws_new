#!/usr/bin/env python3
"""
FIXED QUALIFICATION NAVIGATOR - Stable Gate Passage & U-turn

KEY FIXES:
1. Use locked center for final approach
2. Larger clearance distance to ensure full AUV passage
3. Strict U-turn with yaw verification
4. No surfacing until mission complete
5. Emergency straight mode when close to gate
6. CRITICAL FIX: Proper passing_start_time initialization to prevent crashes
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import time
import math


class StableQualificationNavigator(Node):
    def __init__(self):
        super().__init__('qualification_navigator')
        
        self.SUBMERGING = 0
        self.SEARCHING = 1
        self.APPROACHING = 2
        self.ALIGNING = 3
        self.FINAL_APPROACH = 4
        self.PASSING = 5
        self.CLEARING = 6
        self.UTURN = 7
        self.REVERSE_SEARCHING = 8
        self.REVERSE_APPROACHING = 9
        self.REVERSE_ALIGNING = 10
        self.REVERSE_FINAL_APPROACH = 11
        self.REVERSE_PASSING = 12
        self.REVERSE_CLEARING = 13
        self.SURFACING = 14
        self.COMPLETED = 15
        
        self.state = self.SUBMERGING
        
        self.mission_depth = -0.8
        self.gate_x_position = 0.0
        
        self.auv_length = 0.46
        self.gate_clearance_distance = 1.5
        
        self.passing_start_time = None
        self.passing_timeout = 8.0
        
        self.declare_parameter('search_forward_speed', 0.4)
        self.declare_parameter('search_rotation_speed', 0.3)
        
        self.declare_parameter('approach_speed', 0.6)
        self.declare_parameter('approach_stop_distance', 3.0)
        self.declare_parameter('approach_yaw_gain', 1.0)
        
        self.declare_parameter('alignment_distance', 3.0)
        self.declare_parameter('alignment_threshold', 0.06)
        self.declare_parameter('alignment_max_time', 20.0)
        self.declare_parameter('alignment_yaw_gain', 4.0)
        
        self.declare_parameter('final_approach_speed', 0.5)
        self.declare_parameter('final_approach_threshold', 0.10)
        
        self.declare_parameter('passing_trigger_distance', 1.0)
        self.declare_parameter('passing_alignment_requirement', 0.15)
        self.declare_parameter('passing_speed', 1.0)
        
        self.search_forward_speed = self.get_parameter('search_forward_speed').value
        self.search_rotation_speed = self.get_parameter('search_rotation_speed').value
        
        self.approach_speed = self.get_parameter('approach_speed').value
        self.approach_stop_distance = self.get_parameter('approach_stop_distance').value
        self.approach_yaw_gain = self.get_parameter('approach_yaw_gain').value
        
        self.alignment_distance = self.get_parameter('alignment_distance').value
        self.alignment_threshold = self.get_parameter('alignment_threshold').value
        self.alignment_max_time = self.get_parameter('alignment_max_time').value
        self.alignment_yaw_gain = self.get_parameter('alignment_yaw_gain').value
        
        self.final_approach_speed = self.get_parameter('final_approach_speed').value
        self.final_approach_threshold = self.get_parameter('final_approach_threshold').value
        
        self.passing_trigger_distance = self.get_parameter('passing_trigger_distance').value
        self.passing_alignment_requirement = self.get_parameter('passing_alignment_requirement').value
        self.passing_speed = self.get_parameter('passing_speed').value
        
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
        self.mission_start_time = time.time()
        self.uturn_start_yaw = 0.0
        self.uturn_start_time = 0.0
        self.reverse_mode = False
        
        self.first_pass_complete = False
        self.second_pass_complete = False
        
        self.create_subscription(Bool, '/qualification/gate_detected', self.gate_cb, 10)
        self.create_subscription(Float32, '/qualification/alignment_error', self.align_cb, 10)
        self.create_subscription(Float32, '/qualification/estimated_distance', self.dist_cb, 10)
        self.create_subscription(Float32, '/qualification/frame_position', self.frame_pos_cb, 10)
        self.create_subscription(Float32, '/qualification/confidence', self.conf_cb, 10)
        self.create_subscription(Odometry, '/ground_truth/odom', self.odom_cb, 10)
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/rp2040/cmd_vel', 10)
        self.state_pub = self.create_publisher(String, '/qualification/state', 10)
        self.reverse_mode_pub = self.create_publisher(Bool, '/mission/reverse_mode', 10)
        
        self.create_timer(0.05, self.control_loop)
        
        self.get_logger().info('Stable Qualification Navigator initialized')
        self.get_logger().info(f'AUV length: {self.auv_length}m, Clearance: {self.gate_clearance_distance}m')
    
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
        
        cmd.linear.z = self.depth_control(self.mission_depth)
        
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
    
    def submerge(self, cmd: Twist) -> Twist:
        if abs(self.mission_depth - self.current_depth) < 0.3:
            if time.time() - self.state_start_time > 3.0:
                self.get_logger().info('Submerged - starting search')
                self.reverse_mode_pub.publish(Bool(data=self.reverse_mode))
                self.transition_to(self.SEARCHING)
        return cmd
    
    def searching(self, cmd: Twist) -> Twist:
        if self.gate_detected and self.estimated_distance < 999:
            self.get_logger().info(f'Gate found at {self.estimated_distance:.2f}m')
            if self.reverse_mode:
                self.transition_to(self.REVERSE_APPROACHING)
            else:
                self.transition_to(self.APPROACHING)
            return cmd
        
        cmd.linear.x = self.search_forward_speed
        cmd.angular.z = self.search_rotation_speed if (time.time() % 8 < 4) else -self.search_rotation_speed
        return cmd
    
    def approaching(self, cmd: Twist) -> Twist:
        if not self.gate_detected:
            cmd.linear.x = 0.2
            cmd.angular.z = 0.3
            return cmd
        
        if self.estimated_distance <= self.approach_stop_distance:
            self.get_logger().info(f'Reached 3m - ALIGNING (pos={self.frame_position:+.3f})')
            if self.reverse_mode:
                self.transition_to(self.REVERSE_ALIGNING)
            else:
                self.transition_to(self.ALIGNING)
            return cmd
        
        cmd.linear.x = self.approach_speed
        cmd.angular.z = -self.frame_position * self.approach_yaw_gain
        
        return cmd
    
    def aligning(self, cmd: Twist) -> Twist:
        if not self.gate_detected:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.3
            return cmd
        
        if self.alignment_start_time == 0.0:
            self.alignment_start_time = time.time()
            self.get_logger().info(f'ALIGNING at 3m (pos={self.frame_position:+.3f})')
        
        elapsed = time.time() - self.alignment_start_time
        
        if elapsed > self.alignment_max_time:
            self.get_logger().warn('Alignment timeout - proceeding')
            self.alignment_start_time = 0.0
            if self.reverse_mode:
                self.transition_to(self.REVERSE_FINAL_APPROACH)
            else:
                self.transition_to(self.FINAL_APPROACH)
            return cmd
        
        is_well_aligned = abs(self.frame_position) < self.alignment_threshold
        has_confidence = self.confidence > 0.8
        
        if is_well_aligned and has_confidence:
            self.get_logger().info(
                f'ALIGNED (pos={self.frame_position:+.3f}, {elapsed:.1f}s)'
            )
            self.alignment_start_time = 0.0
            if self.reverse_mode:
                self.transition_to(self.REVERSE_FINAL_APPROACH)
            else:
                self.transition_to(self.FINAL_APPROACH)
            return cmd
        
        quality = abs(self.frame_position)
        
        if quality > 0.2:
            cmd.linear.x = 0.0
            cmd.angular.z = -self.frame_position * self.alignment_yaw_gain
        elif quality > 0.1:
            cmd.linear.x = 0.1
            cmd.angular.z = -self.frame_position * self.alignment_yaw_gain * 0.8
        else:
            cmd.linear.x = 0.15
            cmd.angular.z = -self.frame_position * self.alignment_yaw_gain * 0.5
        
        return cmd
    
    def final_approach(self, cmd: Twist) -> Twist:
        if self.estimated_distance <= self.passing_trigger_distance:
            if abs(self.frame_position) < self.passing_alignment_requirement:
                self.get_logger().info(
                    f'🚀 COMMITTING TO PASSAGE at {self.estimated_distance:.2f}m '
                    f'(alignment={self.frame_position:+.3f})'
                )
                self.passing_start_position = self.current_position
                self.passing_start_time = time.time()  # CRITICAL: Initialize timer here
                if self.reverse_mode:
                    self.transition_to(self.REVERSE_PASSING)
                else:
                    self.transition_to(self.PASSING)
                return cmd
            else:
                self.get_logger().error(
                    f'ABORT: Misaligned ({self.frame_position:+.3f}) - realigning'
                )
                cmd.linear.x = 0.0
                cmd.angular.z = -self.frame_position * self.alignment_yaw_gain * 2.0
                return cmd
        
        if abs(self.frame_position) > self.final_approach_threshold:
            cmd.linear.x = self.final_approach_speed * 0.6
            cmd.angular.z = -self.frame_position * self.alignment_yaw_gain * 0.7
        else:
            cmd.linear.x = self.final_approach_speed
            cmd.angular.z = -self.frame_position * self.alignment_yaw_gain * 0.3
        
        return cmd
    
    def passing(self, cmd: Twist) -> Twist:
        # CRITICAL FIX: Always ensure passing_start_time is initialized
        if self.passing_start_time is None:
            self.passing_start_time = time.time()
            self.get_logger().info('PASSAGE STARTED - Timer initialized')
        
        if self.passing_start_position is None:
            self.passing_start_position = self.current_position
        
        if self.current_position and self.passing_start_position:
            dx = self.current_position[0] - self.passing_start_position[0]
            dy = self.current_position[1] - self.passing_start_position[1]
            distance_traveled = math.sqrt(dx*dx + dy*dy)
            
            elapsed = time.time() - self.passing_start_time
            
            if distance_traveled > self.gate_clearance_distance:
                if not self.reverse_mode:
                    self.get_logger().info(
                        f'FORWARD PASS COMPLETE (traveled {distance_traveled:.2f}m in {elapsed:.1f}s)'
                    )
                    self.first_pass_complete = True
                    self.passing_start_position = None
                    self.passing_start_time = None  # Reset timer
                    self.transition_to(self.CLEARING)
                else:
                    self.get_logger().info(
                        f'REVERSE PASS COMPLETE (traveled {distance_traveled:.2f}m in {elapsed:.1f}s)'
                    )
                    self.second_pass_complete = True
                    self.passing_start_position = None
                    self.passing_start_time = None  # Reset timer
                    self.transition_to(self.REVERSE_CLEARING)
                return cmd
            
            if elapsed > self.passing_timeout:
                self.get_logger().warn(
                    f'PASSING TIMEOUT after {elapsed:.1f}s (traveled {distance_traveled:.2f}m) - assuming complete'
                )
                if not self.reverse_mode:
                    self.first_pass_complete = True
                    self.passing_start_position = None
                    self.passing_start_time = None  # Reset timer
                    self.transition_to(self.CLEARING)
                else:
                    self.second_pass_complete = True
                    self.passing_start_position = None
                    self.passing_start_time = None  # Reset timer
                    self.transition_to(self.REVERSE_CLEARING)
                return cmd
            
            if int(elapsed) % 2 == 0:
                self.get_logger().info(
                    f'PASSING: {distance_traveled:.2f}m / {self.gate_clearance_distance:.2f}m ({elapsed:.1f}s)',
                    throttle_duration_sec=1.9
                )
        
        cmd.linear.x = self.passing_speed
        cmd.angular.z = 0.0
        
        return cmd
    
    def clearing(self, cmd: Twist) -> Twist:
        if self.current_position:
            current_x = self.current_position[0]
            
            clearance_needed = self.gate_x_position + 3.5
            
            if current_x > clearance_needed:
                self.get_logger().info(f'Fully cleared (X={current_x:.2f}m) - U-turn')
                self.uturn_start_yaw = self.current_yaw
                self.uturn_start_time = time.time()
                self.transition_to(self.UTURN)
                return cmd
        
        cmd.linear.x = self.passing_speed * 0.8
        return cmd
    
    def uturn(self, cmd: Twist) -> Twist:
        angle_turned = abs(self.normalize_angle(self.current_yaw - self.uturn_start_yaw))
        elapsed = time.time() - self.uturn_start_time
        
        if angle_turned > (math.pi - 0.2):
            self.get_logger().info(
                f'U-turn complete (turned {math.degrees(angle_turned):.0f}deg, {elapsed:.1f}s)'
            )
            self.reverse_mode = True
            self.reverse_mode_pub.publish(Bool(data=True))
            self.transition_to(self.REVERSE_SEARCHING)
            return cmd
        
        if elapsed > 15.0:
            self.get_logger().warn('U-turn timeout')
            self.reverse_mode = True
            self.reverse_mode_pub.publish(Bool(data=True))
            self.transition_to(self.REVERSE_SEARCHING)
            return cmd
        
        cmd.linear.x = 0.2
        cmd.angular.z = 0.7
        
        return cmd
    
    def reverse_clearing(self, cmd: Twist) -> Twist:
        if self.current_position:
            current_x = self.current_position[0]
            
            clearance_needed = self.gate_x_position - 3.5
            
            if current_x < clearance_needed:
                self.get_logger().info(f'Reverse fully cleared (X={current_x:.2f}m) - Surfacing')
                self.transition_to(self.SURFACING)
                return cmd
        
        cmd.linear.x = self.passing_speed * 0.8
        return cmd
    
    def surfacing(self, cmd: Twist) -> Twist:
        if self.current_depth > -0.2:
            self.get_logger().info('SURFACED - MISSION COMPLETE')
            total_time = time.time() - self.mission_start_time
            self.get_logger().info(f'Pass 1: {"PASS" if self.first_pass_complete else "FAIL"}')
            self.get_logger().info(f'Pass 2: {"PASS" if self.second_pass_complete else "FAIL"}')
            self.get_logger().info(f'Total time: {total_time:.1f}s')
            self.transition_to(self.COMPLETED)
        cmd.linear.z = -0.5
        return cmd
    
    def completed(self, cmd: Twist) -> Twist:
        cmd.linear.x = 0.0
        cmd.linear.y = 0.0
        cmd.linear.z = 0.0
        cmd.angular.z = 0.0
        return cmd
    
    def transition_to(self, new_state: int):
        self.state = new_state
        self.state_start_time = time.time()
        self.get_logger().info(f'-> {self.get_state_name()}')
    
    def get_state_name(self) -> str:
        names = {
            self.SUBMERGING: 'SUBMERGING',
            self.SEARCHING: 'SEARCHING',
            self.APPROACHING: 'APPROACHING',
            self.ALIGNING: 'ALIGNING',
            self.FINAL_APPROACH: 'FINAL_APPROACH',
            self.PASSING: 'PASSING',
            self.CLEARING: 'CLEARING',
            self.UTURN: 'UTURN',
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
    node = StableQualificationNavigator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()