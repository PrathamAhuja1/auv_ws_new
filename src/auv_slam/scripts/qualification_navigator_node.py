#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import time
import math


class FixedQualificationNavigator(Node):
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
        
        self.gate_x_position = 0.0
        self.mission_depth = -0.8
        
        self.auv_length = 0.46
        
        self.forward_clearance_x = self.gate_x_position + 1.0
        self.reverse_clearance_x = self.gate_x_position - 1.0
        
        self.passing_timeout = 15.0
        
        self.declare_parameter('search_forward_speed', 0.4)
        self.declare_parameter('approach_speed', 0.6)
        self.declare_parameter('approach_stop_distance', 3.0)
        self.declare_parameter('approach_yaw_gain', 1.0)
        self.declare_parameter('alignment_distance', 3.0)
        self.declare_parameter('alignment_threshold', 0.06)
        self.declare_parameter('alignment_max_time', 20.0)
        self.declare_parameter('final_approach_speed', 0.5)
        self.declare_parameter('final_approach_threshold', 0.12)
        self.declare_parameter('passing_trigger_distance', 1.0)
        self.declare_parameter('passing_speed', 1.0)
        
        self.search_forward_speed = self.get_parameter('search_forward_speed').value
        self.approach_speed = self.get_parameter('approach_speed').value
        self.approach_stop_distance = self.get_parameter('approach_stop_distance').value
        self.approach_yaw_gain = self.get_parameter('approach_yaw_gain').value
        self.alignment_distance = self.get_parameter('alignment_distance').value
        self.alignment_threshold = self.get_parameter('alignment_threshold').value
        self.alignment_max_time = self.get_parameter('alignment_max_time').value
        self.final_approach_speed = self.get_parameter('final_approach_speed').value
        self.final_approach_threshold = self.get_parameter('final_approach_threshold').value
        self.passing_trigger_distance = self.get_parameter('passing_trigger_distance').value
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
        self.passing_start_time = None
        self.alignment_start_time = 0.0
        self.gate_lost_time = 0.0
        self.gate_lost_timeout = 3.0
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
        
        self.get_logger().info('Qualification Navigator initialized')
        self.get_logger().info('AUV length: {:.2f}m'.format(self.auv_length))
        self.get_logger().info('Gate at X={:.2f}m'.format(self.gate_x_position))
        self.get_logger().info('Forward clearance: X > {:.2f}m'.format(self.forward_clearance_x))
        self.get_logger().info('Reverse clearance: X < {:.2f}m'.format(self.reverse_clearance_x))
        self.get_logger().info('Passing timeout: {:.1f}s'.format(self.passing_timeout))
    
    def gate_cb(self, msg: Bool):
        was_detected = self.gate_detected
        self.gate_detected = msg.data
        
        if not was_detected and self.gate_detected:
            self.gate_lost_time = 0.0
        elif was_detected and not self.gate_detected:
            self.gate_lost_time = time.time()
    
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
        
        if self.state == self.PASSING or self.state == self.REVERSE_PASSING:
            cmd.linear.z = self.gentle_depth_control(self.mission_depth)
        else:
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
    
    def gentle_depth_control(self, target_depth: float) -> float:
        depth_error = target_depth - self.current_depth
        deadband = 0.25
        
        if abs(depth_error) < deadband:
            return 0.0
        
        z_cmd = depth_error * 0.2
        return max(-0.3, min(z_cmd, 0.3))
    
    def submerge(self, cmd: Twist) -> Twist:
        if abs(self.mission_depth - self.current_depth) < 0.3:
            if time.time() - self.state_start_time > 3.0:
                self.get_logger().info('Submerged - starting search')
                self.reverse_mode_pub.publish(Bool(data=self.reverse_mode))
                self.transition_to(self.SEARCHING)
        return cmd
    
    def searching(self, cmd: Twist) -> Twist:
        if self.gate_detected and self.estimated_distance < 999:
            self.get_logger().info('Gate found at {:.2f}m'.format(self.estimated_distance))
            if self.reverse_mode:
                self.transition_to(self.REVERSE_APPROACHING)
            else:
                self.transition_to(self.APPROACHING)
            return cmd
        
        cmd.linear.x = self.search_forward_speed
        cmd.angular.z = 0.3 if (time.time() % 8 < 4) else -0.3
        
        elapsed = time.time() - self.state_start_time
        sweep_period = 8.0
        sweep_phase = (elapsed % sweep_period) / sweep_period
        
        if int(elapsed) % 3 == 0:
            direction = "LEFT" if sweep_phase < 0.5 else "RIGHT"
            self.get_logger().info(
                'Searching ({})... {:.0f}s'.format(direction, elapsed),
                throttle_duration_sec=2.9
            )
        
        return cmd
    
    def approaching(self, cmd: Twist) -> Twist:
        if not self.gate_detected:
            cmd.linear.x = 0.2
            cmd.angular.z = 0.3
            return cmd
        
        if self.estimated_distance <= self.approach_stop_distance:
            self.get_logger().info('Reached 3m - ALIGNING (pos={:+.3f})'.format(self.frame_position))
            if self.reverse_mode:
                self.transition_to(self.REVERSE_ALIGNING)
            else:
                self.transition_to(self.ALIGNING)
            return cmd
        
        cmd.linear.x = self.approach_speed
        
        cmd.angular.z = -self.frame_position * self.approach_yaw_gain
        
        self.get_logger().info(
            'APPROACHING: dist={:.2f}m, pos={:+.3f}, speed={:.2f}'.format(
                self.estimated_distance, self.frame_position, cmd.linear.x
            ),
            throttle_duration_sec=0.5
        )
        
        return cmd
    
    def aligning(self, cmd: Twist) -> Twist:
        if not self.gate_detected:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.3
            return cmd
        
        if self.alignment_start_time == 0.0:
            self.alignment_start_time = time.time()
            self.get_logger().info('ALIGNING at 3m (pos={:+.3f})'.format(self.frame_position))
        
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
            self.get_logger().info('ALIGNED (pos={:+.3f}, {:.1f}s)'.format(
                self.frame_position, elapsed
            ))
            self.alignment_start_time = 0.0
            if self.reverse_mode:
                self.transition_to(self.REVERSE_FINAL_APPROACH)
            else:
                self.transition_to(self.FINAL_APPROACH)
            return cmd
        
        alignment_quality = abs(self.frame_position)
        
        if alignment_quality > 0.2:
            cmd.linear.x = 0.0
            cmd.angular.z = -self.frame_position * 4.0
            status = "MAJOR"
        elif alignment_quality > 0.1:
            cmd.linear.x = 0.1
            cmd.angular.z = -self.frame_position * 3.0
            status = "MODERATE"
        else:
            cmd.linear.x = 0.15
            cmd.angular.z = -self.frame_position * 2.0
            status = "FINE"
        
        self.get_logger().info(
            'ALIGNING ({}): pos={:+.3f}, yaw={:+.2f}, t={:.1f}s'.format(
                status, self.frame_position, cmd.angular.z, elapsed
            ),
            throttle_duration_sec=0.3
        )
        
        return cmd
    
    def final_approach(self, cmd: Twist) -> Twist:
        if not self.gate_detected:
            if self.gate_lost_time > 0.0:
                lost_duration = time.time() - self.gate_lost_time
                if lost_duration > self.gate_lost_timeout:
                    self.get_logger().warn('Gate lost - returning to search')
                    self.transition_to(self.SEARCHING)
                else:
                    cmd.linear.x = 0.1
                    cmd.angular.z = 0.0
            return cmd
        
        if self.estimated_distance <= self.passing_trigger_distance:
            if abs(self.frame_position) < 0.15:
                self.get_logger().info(
                    'COMMITTING TO PASSAGE at {:.2f}m (alignment={:+.3f})'.format(
                        self.estimated_distance, self.frame_position
                    )
                )
                self.passing_start_position = self.current_position
                self.passing_start_time = time.time()
                if self.reverse_mode:
                    self.transition_to(self.REVERSE_PASSING)
                else:
                    self.transition_to(self.PASSING)
                return cmd
            else:
                cmd.linear.x = 0.0
                cmd.angular.z = -self.frame_position * 4.0
                return cmd
        
        if abs(self.frame_position) > self.final_approach_threshold:
            self.get_logger().warn(
                'Drifting (pos={:+.3f}) - correcting'.format(self.frame_position),
                throttle_duration_sec=0.5
            )
            cmd.linear.x = self.final_approach_speed * 0.6
            cmd.angular.z = -self.frame_position * 3.0
        else:
            cmd.linear.x = self.final_approach_speed
            cmd.angular.z = -self.frame_position * 1.5
        
        self.get_logger().info(
            'FINAL APPROACH: dist={:.2f}m, pos={:+.3f}, speed={:.2f}'.format(
                self.estimated_distance, self.frame_position, cmd.linear.x
            ),
            throttle_duration_sec=0.5
        )
        
        return cmd
    
    def passing(self, cmd: Twist) -> Twist:
        if self.passing_start_time is None:
            self.passing_start_time = time.time()
            self.get_logger().info('PASSAGE STARTED')
        
        if self.passing_start_position is None:
            self.passing_start_position = self.current_position
        
        if self.current_position:
            current_x = self.current_position[0]
            elapsed = time.time() - self.passing_start_time
            
            if not self.reverse_mode:
                clearance_needed = self.forward_clearance_x
                distance_to_clearance = clearance_needed - current_x
                cleared = current_x > clearance_needed
                direction = "FORWARD"
            else:
                clearance_needed = self.reverse_clearance_x
                distance_to_clearance = current_x - clearance_needed
                cleared = current_x < clearance_needed
                direction = "REVERSE"
            
            if cleared:
                if self.passing_start_position:
                    dx = self.current_position[0] - self.passing_start_position[0]
                    dy = self.current_position[1] - self.passing_start_position[1]
                    distance_traveled = math.sqrt(dx*dx + dy*dy)
                    
                    self.get_logger().info('{} PASS COMPLETE'.format(direction))
                    self.get_logger().info('Current X: {:.2f}m'.format(current_x))
                    self.get_logger().info('Clearance needed: {:.2f}m'.format(clearance_needed))
                    self.get_logger().info('Beyond gate: {:.2f}m'.format(abs(distance_to_clearance)))
                    self.get_logger().info('Distance traveled: {:.2f}m'.format(distance_traveled))
                    self.get_logger().info('Time: {:.1f}s'.format(elapsed))
                
                if not self.reverse_mode:
                    self.first_pass_complete = True
                    self.passing_start_position = None
                    self.passing_start_time = None
                    self.transition_to(self.CLEARING)
                else:
                    self.second_pass_complete = True
                    self.passing_start_position = None
                    self.passing_start_time = None
                    self.transition_to(self.REVERSE_CLEARING)
                return cmd
            
            if elapsed > self.passing_timeout:
                self.get_logger().error('PASSING TIMEOUT after {:.1f}s'.format(elapsed))
                self.get_logger().error('Current X: {:.2f}m'.format(current_x))
                self.get_logger().error('Needed: {:.2f}m'.format(clearance_needed))
                self.get_logger().error('Still need: {:.2f}m more'.format(abs(distance_to_clearance)))
                self.get_logger().error('AUV did NOT fully pass through gate')
                
                if not self.reverse_mode:
                    self.passing_start_position = None
                    self.passing_start_time = None
                    self.transition_to(self.CLEARING)
                else:
                    self.passing_start_position = None
                    self.passing_start_time = None
                    self.transition_to(self.REVERSE_CLEARING)
                return cmd
            
            if int(elapsed * 10) % 5 == 0:
                self.get_logger().info(
                    'PASSING ({}): X={:.2f}m, need {:.2f}m, {:.2f}m to go, {:.1f}s'.format(
                        direction, current_x, clearance_needed, abs(distance_to_clearance), elapsed
                    ),
                    throttle_duration_sec=0.4
                )
        
        cmd.linear.x = self.passing_speed
        cmd.angular.z = 0.0
        
        return cmd
    
    def clearing(self, cmd: Twist) -> Twist:
        if self.current_position:
            current_x = self.current_position[0]
            
            clearance_needed = self.forward_clearance_x + 2.0
            
            if current_x > clearance_needed:
                self.get_logger().info('Fully cleared (X={:.2f}m) - U-turn'.format(current_x))
                self.uturn_start_yaw = self.current_yaw
                self.uturn_start_time = time.time()
                self.transition_to(self.UTURN)
                return cmd
        
        cmd.linear.x = 0.8
        return cmd
    
    def uturn(self, cmd: Twist) -> Twist:
        angle_turned = abs(self.normalize_angle(self.current_yaw - self.uturn_start_yaw))
        elapsed = time.time() - self.uturn_start_time
        
        if angle_turned > (math.pi - 0.2):
            self.get_logger().info(
                'U-turn complete (turned {:.0f}deg, {:.1f}s)'.format(
                    math.degrees(angle_turned), elapsed
                )
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
            
            clearance_needed = self.reverse_clearance_x - 2.0
            
            if current_x < clearance_needed:
                self.get_logger().info('Reverse fully cleared (X={:.2f}m) - Surfacing'.format(current_x))
                self.transition_to(self.SURFACING)
                return cmd
        
        cmd.linear.x = 0.8
        return cmd
    
    def surfacing(self, cmd: Twist) -> Twist:
        if self.current_depth > -0.2:
            self.get_logger().info('SURFACED - MISSION COMPLETE')
            total_time = time.time() - self.mission_start_time
            if self.first_pass_complete:
                self.get_logger().info('Pass 1: COMPLETE')
            else:
                self.get_logger().info('Pass 1: FAILED')
            if self.second_pass_complete:
                self.get_logger().info('Pass 2: COMPLETE')
            else:
                self.get_logger().info('Pass 2: FAILED')
            self.get_logger().info('Total time: {:.1f}s'.format(total_time))
            
            if self.first_pass_complete and self.second_pass_complete:
                self.get_logger().info('QUALIFICATION SCORE: 2 POINTS')
            elif self.first_pass_complete:
                self.get_logger().info('QUALIFICATION SCORE: 1 POINT')
            else:
                self.get_logger().info('QUALIFICATION SCORE: 0 POINTS')
            
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
        self.get_logger().info('STATE: {}'.format(self.get_state_name()))
    
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
    node = FixedQualificationNavigator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()