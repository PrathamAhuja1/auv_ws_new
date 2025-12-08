#!/usr/bin/env python3
"""
FIXED QUALIFICATION NAVIGATOR - Proper Gate Passage

KEY FIXES:
1. Larger clearance distance (2.0m) to ensure back of AUV clears gate
2. Better alignment maintenance during final approach
3. Stricter alignment requirements before committing to pass
4. Proper U-turn with yaw verification
5. Emergency abort if severely misaligned during passage
"""

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
        
        # State machine
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
        
        # Mission parameters
        self.mission_depth = -0.8
        self.gate_x_position = 0.0
        
        # CRITICAL: Larger clearance to ensure AUV back clears gate
        # AUV length ~0.46m, so 2.0m clearance ensures full passage
        self.gate_clearance_distance = 2.0
        
        # Navigation parameters
        self.declare_parameter('search_forward_speed', 0.4)
        self.declare_parameter('search_rotation_speed', 0.3)
        
        self.declare_parameter('approach_speed', 0.6)
        self.declare_parameter('approach_stop_distance', 3.0)
        self.declare_parameter('approach_yaw_gain', 1.0)
        
        # CRITICAL: Stricter alignment requirements
        self.declare_parameter('alignment_distance', 3.0)
        self.declare_parameter('alignment_threshold', 0.06)  # Tighter: ±6%
        self.declare_parameter('alignment_max_time', 20.0)
        self.declare_parameter('alignment_yaw_gain', 4.0)  # Stronger yaw
        
        self.declare_parameter('final_approach_speed', 0.5)
        self.declare_parameter('final_approach_threshold', 0.10)
        
        # CRITICAL: Only commit when very close AND well aligned
        self.declare_parameter('passing_trigger_distance', 1.0)
        self.declare_parameter('passing_alignment_requirement', 0.15)  # Must be within ±15%
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
        self.mission_start_time = time.time()
        self.uturn_start_yaw = 0.0
        self.uturn_start_time = 0.0
        self.reverse_mode = False
        
        self.first_pass_complete = False
        self.second_pass_complete = False
        
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
        self.get_logger().info('✅ FIXED QUALIFICATION NAVIGATOR')
        self.get_logger().info('   - Proper gate center detection')
        self.get_logger().info('   - Larger clearance: 2.0m (ensures back clears)')
        self.get_logger().info('   - Stricter alignment before commitment')
        self.get_logger().info('   - Emergency abort on severe misalignment')
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
        
        # Depth control
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
            cmd = self.clearing(cmd)
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
                self.get_logger().info('✅ Submerged - starting search')
                self.reverse_mode_pub.publish(Bool(data=self.reverse_mode))
                self.transition_to(self.SEARCHING)
        return cmd
    
    def searching(self, cmd: Twist) -> Twist:
        if self.gate_detected and self.estimated_distance < 999:
            self.get_logger().info(f'🎯 Gate found at {self.estimated_distance:.2f}m')
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
            self.get_logger().info(f'🛑 Reached 3m - ALIGNING (pos={self.frame_position:+.3f})')
            if self.reverse_mode:
                self.transition_to(self.REVERSE_ALIGNING)
            else:
                self.transition_to(self.ALIGNING)
            return cmd
        
        cmd.linear.x = self.approach_speed
        cmd.angular.z = -self.frame_position * self.approach_yaw_gain
        
        return cmd
    
    def aligning(self, cmd: Twist) -> Twist:
        """CRITICAL: Proper rotation-based alignment"""
        if not self.gate_detected:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.3
            return cmd
        
        if self.alignment_start_time == 0.0:
            self.alignment_start_time = time.time()
            self.get_logger().info(f'🎯 ALIGNING at 3m (pos={self.frame_position:+.3f})')
        
        elapsed = time.time() - self.alignment_start_time
        
        if elapsed > self.alignment_max_time:
            self.get_logger().warn('⏰ Alignment timeout - proceeding with caution')
            self.alignment_start_time = 0.0
            if self.reverse_mode:
                self.transition_to(self.REVERSE_FINAL_APPROACH)
            else:
                self.transition_to(self.FINAL_APPROACH)
            return cmd
        
        # Check alignment quality
        is_well_aligned = abs(self.frame_position) < self.alignment_threshold
        has_confidence = self.confidence > 0.8
        
        if is_well_aligned and has_confidence:
            self.get_logger().info(
                f'✅ ALIGNED! (pos={self.frame_position:+.3f}, {elapsed:.1f}s) '
                f'→ Final approach'
            )
            self.alignment_start_time = 0.0
            if self.reverse_mode:
                self.transition_to(self.REVERSE_FINAL_APPROACH)
            else:
                self.transition_to(self.FINAL_APPROACH)
            return cmd
        
        # Pure rotation with minimal forward
        quality = abs(self.frame_position)
        
        if quality > 0.2:
            cmd.linear.x = 0.0
            cmd.linear.y = 0.0
            cmd.angular.z = -self.frame_position * self.alignment_yaw_gain
        elif quality > 0.1:
            cmd.linear.x = 0.1
            cmd.linear.y = 0.0
            cmd.angular.z = -self.frame_position * self.alignment_yaw_gain * 0.8
        else:
            cmd.linear.x = 0.15
            cmd.linear.y = 0.0
            cmd.angular.z = -self.frame_position * self.alignment_yaw_gain * 0.5
        
        return cmd
    
    def final_approach(self, cmd: Twist) -> Twist:
        """CRITICAL: Maintain alignment while approaching"""
        if not self.gate_detected:
            cmd.linear.x = 0.1
            return cmd
        
        # CRITICAL: Check alignment before committing
        if self.estimated_distance <= self.passing_trigger_distance:
            if abs(self.frame_position) < self.passing_alignment_requirement:
                self.get_logger().info(
                    f'🚀 COMMITTING at {self.estimated_distance:.2f}m '
                    f'(align={self.frame_position:+.3f})'
                )
                self.passing_start_position = self.current_position
                if self.reverse_mode:
                    self.transition_to(self.REVERSE_PASSING)
                else:
                    self.transition_to(self.PASSING)
                return cmd
            else:
                # TOO MISALIGNED - Emergency realignment
                self.get_logger().error(
                    f'🚨 ABORT COMMIT: Misaligned ({self.frame_position:+.3f}) '
                    f'at trigger point! Emergency realign...'
                )
                cmd.linear.x = 0.0
                cmd.angular.z = -self.frame_position * self.alignment_yaw_gain * 2.0
                return cmd
        
        # Check drift during approach
        if abs(self.frame_position) > self.final_approach_threshold:
            self.get_logger().warn(
                f'⚠️ Drifting (pos={self.frame_position:+.3f}) - correcting',
                throttle_duration_sec=0.5
            )
            cmd.linear.x = self.final_approach_speed * 0.6
            cmd.angular.z = -self.frame_position * self.alignment_yaw_gain * 0.7
        else:
            cmd.linear.x = self.final_approach_speed
            cmd.angular.z = -self.frame_position * self.alignment_yaw_gain * 0.3
        
        return cmd
    
    def passing(self, cmd: Twist) -> Twist:
        """FULL SPEED PASSAGE - Monitor for severe misalignment"""
        if self.passing_start_position is None:
            self.passing_start_position = self.current_position
            self.get_logger().info('🚀 PASSAGE STARTED - FULL SPEED!')
        
        # CRITICAL: Emergency abort on severe misalignment during passage
        if self.gate_detected and abs(self.frame_position) > 0.35:
            self.get_logger().error(
                f'🚨 SEVERE MISALIGNMENT during passage! '
                f'pos={self.frame_position:+.3f} - ABORTING'
            )
            # Emergency stop and realign
            cmd.linear.x = 0.0
            cmd.angular.z = -self.frame_position * 2.0
            return cmd
        
        # Check if cleared using X-position
        if self.current_position:
            current_x = self.current_position[0]
            
            if not self.reverse_mode:
                if current_x > (self.gate_x_position + self.gate_clearance_distance):
                    self.get_logger().info(
                        f'✅ FORWARD PASS COMPLETE (X={current_x:.2f}m, '
                        f'cleared by {current_x - self.gate_x_position:.2f}m)'
                    )
                    self.first_pass_complete = True
                    self.transition_to(self.CLEARING)
            else:
                if current_x < (self.gate_x_position - self.gate_clearance_distance):
                    self.get_logger().info(
                        f'✅ REVERSE PASS COMPLETE (X={current_x:.2f}m, '
                        f'cleared by {abs(current_x - self.gate_x_position):.2f}m)'
                    )
                    self.second_pass_complete = True
                    self.transition_to(self.REVERSE_CLEARING)
        
        # FULL SPEED - minimal corrections
        cmd.linear.x = self.passing_speed
        cmd.linear.y = 0.0
        cmd.angular.z = 0.0
        
        return cmd
    
    def clearing(self, cmd: Twist) -> Twist:
        """Ensure fully past gate before U-turn"""
        if self.current_position:
            current_x = self.current_position[0]
            
            if current_x > (self.gate_x_position + 2.5):
                self.get_logger().info('✅ FULLY CLEARED - Starting U-turn')
                self.uturn_start_yaw = self.current_yaw
                self.uturn_start_time = time.time()
                self.transition_to(self.UTURN)
                return cmd
        
        cmd.linear.x = self.passing_speed * 0.8
        return cmd
    
    def uturn(self, cmd: Twist) -> Twist:
        """180° turn with timeout"""
        angle_turned = abs(self.normalize_angle(self.current_yaw - self.uturn_start_yaw))
        elapsed = time.time() - self.uturn_start_time
        
        # Success condition
        if angle_turned > (math.pi - 0.2):
            self.get_logger().info(
                f'✅ U-turn complete (turned {math.degrees(angle_turned):.0f}°, '
                f'took {elapsed:.1f}s)'
            )
            self.reverse_mode = True
            self.reverse_mode_pub.publish(Bool(data=True))
            self.transition_to(self.REVERSE_SEARCHING)
            return cmd
        
        # Timeout check
        if elapsed > 15.0:
            self.get_logger().warn('⏰ U-turn timeout - proceeding anyway')
            self.reverse_mode = True
            self.reverse_mode_pub.publish(Bool(data=True))
            self.transition_to(self.REVERSE_SEARCHING)
            return cmd
        
        # U-turn execution
        cmd.linear.x = 0.2
        cmd.angular.z = 0.7
        
        if int(elapsed) % 2 == 0:
            self.get_logger().info(
                f'🔄 U-turning: {math.degrees(angle_turned):.0f}° / 180°',
                throttle_duration_sec=1.9
            )
        
        return cmd
    
    def surfacing(self, cmd: Twist) -> Twist:
        if self.current_depth > -0.2:
            self.get_logger().info('✅ SURFACED - MISSION COMPLETE!')
            self.get_logger().info('='*70)
            self.get_logger().info('🎉 QUALIFICATION COMPLETE!')
            self.get_logger().info(f'   Pass 1: {"✅" if self.first_pass_complete else "❌"}')
            self.get_logger().info(f'   Pass 2: {"✅" if self.second_pass_complete else "❌"}')
            total_time = time.time() - self.mission_start_time
            self.get_logger().info(f'   Total time: {total_time:.1f}s')
            self.get_logger().info('='*70)
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
        self.get_logger().info(f'🔄 → {self.get_state_name()}')
    
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