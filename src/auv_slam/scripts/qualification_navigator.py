#!/usr/bin/env python3
"""
Qualification Navigator - State machine for SAUVC Qualification Task
Per rulebook:
1. Start from starting zone, touching wall
2. Submerge autonomously before leaving zone
3. Navigate to gate (~10m away)
4. Pass completely through gate (1 point)
5. Perform U-turn
6. Pass completely through gate again (2 points total)
7. Complete without surfacing or touching gate/walls/bottom
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
        self.IDLE = 0
        self.STARTING = 1          # At starting zone, touching wall
        self.SUBMERGING = 2        # Autonomous submersion
        self.SEARCHING = 3         # Search for gate
        self.APPROACHING_1 = 4     # First approach to gate
        self.ALIGNING_1 = 5        # Align for first pass
        self.PASSING_1 = 6         # First passage through gate
        self.UTURN = 7             # Perform U-turn
        self.APPROACHING_2 = 8     # Second approach to gate
        self.ALIGNING_2 = 9        # Align for second pass
        self.PASSING_2 = 10        # Second passage through gate
        self.COMPLETED = 11
        
        self.state = self.STARTING
        
        # Parameters
        self.declare_parameter('target_depth', -1.3)
        self.declare_parameter('submerge_depth', -0.8)
        self.declare_parameter('approach_speed', 0.5)
        self.declare_parameter('passing_speed', 0.7)
        self.declare_parameter('alignment_threshold', 0.10)
        self.declare_parameter('alignment_yaw_gain', 3.0)
        self.declare_parameter('uturn_radius', 1.5)
        self.declare_parameter('gate_x_position', 0.0)
        self.declare_parameter('starting_wall_x', -10.0)
        
        self.target_depth = self.get_parameter('target_depth').value
        self.submerge_depth = self.get_parameter('submerge_depth').value
        self.approach_speed = self.get_parameter('approach_speed').value
        self.passing_speed = self.get_parameter('passing_speed').value
        self.alignment_threshold = self.get_parameter('alignment_threshold').value
        self.alignment_yaw_gain = self.get_parameter('alignment_yaw_gain').value
        self.uturn_radius = self.get_parameter('uturn_radius').value
        self.gate_x_position = self.get_parameter('gate_x_position').value
        self.starting_wall_x = self.get_parameter('starting_wall_x').value
        
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
        self.current_yaw = 0.0
        self.first_pass_start_x = None
        self.first_pass_complete = False
        self.second_pass_complete = False
        self.uturn_start_position = None
        self.uturn_start_yaw = None
        
        # Timing
        self.state_start_time = time.time()
        self.mission_start_time = None
        self.first_pass_time = None
        self.completion_time = None
        self.submerge_start_time = None
        
        # Qualification points
        self.qualification_points = 0
        
        # Subscriptions
        self.gate_detected_sub = self.create_subscription(
            Bool, '/qualification/gate_detected', self.gate_detected_callback, 10)
        self.alignment_sub = self.create_subscription(
            Float32, '/qualification/alignment_error', self.alignment_callback, 10)
        self.distance_sub = self.create_subscription(
            Float32, '/qualification/estimated_distance', self.distance_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/ground_truth/odom', self.odom_callback, 10)
        self.frame_position_sub = self.create_subscription(
            Float32, '/qualification/frame_position', self.frame_position_callback, 10)
        self.partial_gate_sub = self.create_subscription(
            Bool, '/qualification/partial_detection', self.partial_gate_callback, 10)
        self.confidence_sub = self.create_subscription(
            Float32, '/qualification/confidence', self.confidence_callback, 10)
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/rp2040/cmd_vel', 10)
        self.state_pub = self.create_publisher(String, '/qualification/state', 10)
        self.points_pub = self.create_publisher(Float32, '/qualification/points', 10)
        self.status_pub = self.create_publisher(String, '/qualification/detailed_status', 10)
        
        self.control_timer = self.create_timer(0.05, self.control_loop)
        
        self.get_logger().info('='*70)
        self.get_logger().info('✅ Qualification Navigator Started')
        self.get_logger().info('='*70)
        self.get_logger().info('   Flow: START → SUBMERGE → SEARCH → APPROACH1')
        self.get_logger().info('         → PASS1 → UTURN → APPROACH2 → PASS2')
        self.get_logger().info(f'   Gate at X={self.gate_x_position}m')
        self.get_logger().info(f'   Starting wall at X={self.starting_wall_x}m')
        self.get_logger().info('='*70)
    
    def gate_detected_callback(self, msg: Bool):
        self.gate_detected = msg.data
    
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
        
        # Extract yaw from quaternion
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)
    
    def control_loop(self):
        cmd = Twist()
        
        # Depth control (except during starting phase)
        if self.state not in [self.STARTING]:
            target_depth = self.submerge_depth if self.state == self.SUBMERGING else self.target_depth
            depth_error = target_depth - self.current_depth
            depth_deadband = 0.2
            
            if abs(depth_error) < depth_deadband:
                cmd.linear.z = 0.0
            else:
                cmd.linear.z = depth_error * 1.2
                cmd.linear.z = max(-1.0, min(cmd.linear.z, 1.0))
        
        # State machine
        if self.state == self.STARTING:
            cmd = self.starting_behavior(cmd)
        elif self.state == self.SUBMERGING:
            cmd = self.submerging_behavior(cmd)
        elif self.state == self.SEARCHING:
            cmd = self.searching_behavior(cmd)
        elif self.state == self.APPROACHING_1:
            cmd = self.approaching_behavior(cmd, pass_number=1)
        elif self.state == self.ALIGNING_1:
            cmd = self.aligning_behavior(cmd, pass_number=1)
        elif self.state == self.PASSING_1:
            cmd = self.passing_behavior(cmd, pass_number=1)
        elif self.state == self.UTURN:
            cmd = self.uturn_behavior(cmd)
        elif self.state == self.APPROACHING_2:
            cmd = self.approaching_behavior(cmd, pass_number=2)
        elif self.state == self.ALIGNING_2:
            cmd = self.aligning_behavior(cmd, pass_number=2)
        elif self.state == self.PASSING_2:
            cmd = self.passing_behavior(cmd, pass_number=2)
        elif self.state == self.COMPLETED:
            cmd = self.completed_behavior(cmd)
        
        self.cmd_vel_pub.publish(cmd)
        
        # Publish state and points
        state_msg = String()
        state_msg.data = self.get_state_name()
        self.state_pub.publish(state_msg)
        
        points_msg = Float32()
        points_msg.data = float(self.qualification_points)
        self.points_pub.publish(points_msg)
        
        # Publish detailed status
        self.publish_detailed_status()
    
    def starting_behavior(self, cmd: Twist) -> Twist:
        """Initial state - at starting zone, touching wall"""
        elapsed = time.time() - self.state_start_time
        
        # Wait 2 seconds before starting mission
        if elapsed < 2.0:
            self.get_logger().info(
                f'📍 STARTING POSITION - Waiting to begin... {elapsed:.1f}s',
                throttle_duration_sec=0.9
            )
            return cmd
        
        # Start mission timing
        if self.mission_start_time is None:
            self.mission_start_time = time.time()
            self.get_logger().info('🚀 MISSION START - Beginning qualification run!')
        
        # Transition to submerging
        self.get_logger().info('⬇️ Starting autonomous submersion')
        self.transition_to(self.SUBMERGING)
        
        return cmd
    
    def submerging_behavior(self, cmd: Twist) -> Twist:
        """Submerge autonomously before leaving starting zone"""
        if self.submerge_start_time is None:
            self.submerge_start_time = time.time()
        
        elapsed = time.time() - self.submerge_start_time
        
        # Check if submerged enough
        if self.current_depth < self.submerge_depth:
            self.get_logger().info(
                f'⬇️ SUBMERGING: depth={self.current_depth:.2f}m / {self.submerge_depth:.2f}m',
                throttle_duration_sec=0.5
            )
            # Depth control is handled above
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            return cmd
        
        # Successfully submerged - start searching
        self.get_logger().info(f'✅ Submerged to {self.current_depth:.2f}m - Starting search')
        self.transition_to(self.SEARCHING)
        
        return cmd
    
    def searching_behavior(self, cmd: Twist) -> Twist:
        """Search for qualification gate"""
        if self.gate_detected and self.estimated_distance < 999:
            self.get_logger().info(
                f'🎯 Gate found at {self.estimated_distance:.2f}m - Starting first approach'
            )
            self.transition_to(self.APPROACHING_1)
            return cmd
        
        elapsed = time.time() - self.state_start_time
        
        # Sweep search pattern
        sweep_period = 6.0
        sweep_phase = (elapsed % sweep_period) / sweep_period
        
        if sweep_phase < 0.5:
            cmd.angular.z = 0.2
        else:
            cmd.angular.z = -0.2
        
        cmd.linear.x = 0.4
        
        if int(elapsed) % 2 == 0:
            direction = "LEFT" if sweep_phase < 0.5 else "RIGHT"
            self.get_logger().info(
                f'🔍 SEARCHING ({direction})... {elapsed:.0f}s',
                throttle_duration_sec=1.9
            )
        
        return cmd
    
    def approaching_behavior(self, cmd: Twist, pass_number: int) -> Twist:
        """Approach gate for pass 1 or 2"""
        if not self.gate_detected:
            self.get_logger().warn('❌ Gate lost during approach')
            cmd.linear.x = 0.2
            cmd.angular.z = 0.0
            return cmd
        
        # Check if close enough to align
        if self.estimated_distance <= 3.0:
            self.get_logger().info(
                f'🛑 Reached 3m - Aligning for pass {pass_number}'
            )
            if pass_number == 1:
                self.transition_to(self.ALIGNING_1)
            else:
                self.transition_to(self.ALIGNING_2)
            return cmd
        
        # Approach with light correction
        cmd.linear.x = self.approach_speed
        cmd.angular.z = -self.frame_position * 1.0
        
        self.get_logger().info(
            f'🚶 APPROACH {pass_number}: dist={self.estimated_distance:.2f}m, '
            f'pos={self.frame_position:+.3f}',
            throttle_duration_sec=0.5
        )
        
        return cmd
    
    def aligning_behavior(self, cmd: Twist, pass_number: int) -> Twist:
        """Align for gate passage"""
        if not self.gate_detected:
            self.get_logger().warn('❌ Gate lost during alignment')
            return cmd
        
        # Check alignment
        is_aligned = abs(self.frame_position) < self.alignment_threshold
        has_confidence = self.confidence > 0.7 and not self.partial_gate
        
        if is_aligned and has_confidence:
            self.get_logger().info(
                f'✅ ALIGNED for pass {pass_number}! Starting passage...'
            )
            if pass_number == 1:
                self.first_pass_start_x = self.current_position[0]
                self.transition_to(self.PASSING_1)
            else:
                self.transition_to(self.PASSING_2)
            return cmd
        
        # Alignment corrections
        if abs(self.frame_position) > 0.15:
            cmd.linear.x = 0.1
            cmd.angular.z = -self.frame_position * self.alignment_yaw_gain
        else:
            cmd.linear.x = 0.2
            cmd.angular.z = -self.frame_position * self.alignment_yaw_gain * 0.5
        
        self.get_logger().info(
            f'🔄 ALIGNING {pass_number}: pos={self.frame_position:+.3f}',
            throttle_duration_sec=0.3
        )
        
        return cmd
    
    def passing_behavior(self, cmd: Twist, pass_number: int) -> Twist:
        """Pass through gate"""
        if self.current_position is None:
            return cmd
        
        current_x = self.current_position[0]
        
        # Determine passage completion based on X position
        if pass_number == 1:
            # First pass: moving in positive X direction
            gate_clearance = 0.5
            if current_x > self.gate_x_position + gate_clearance:
                self.first_pass_complete = True
                self.first_pass_time = time.time()
                self.qualification_points = 1
                
                self.get_logger().info('='*70)
                self.get_logger().info('🎉 FIRST PASS COMPLETE!')
                self.get_logger().info(f'   ✅ 1 QUALIFICATION POINT EARNED')
                self.get_logger().info(f'   Position: X={current_x:.2f}m')
                self.get_logger().info('='*70)
                
                self.transition_to(self.UTURN)
                return cmd
            
            distance_past_gate = current_x - self.gate_x_position
            self.get_logger().info(
                f'🚀 PASSING 1: X={current_x:.2f}m, '
                f'{abs(distance_past_gate):.2f}m {"past" if distance_past_gate > 0 else "before"} gate',
                throttle_duration_sec=0.4
            )
        
        else:  # pass_number == 2
            # Second pass: moving in negative X direction
            gate_clearance = 0.5
            if current_x < self.gate_x_position - gate_clearance:
                self.second_pass_complete = True
                self.completion_time = time.time()
                self.qualification_points = 2
                
                total_time = self.completion_time - self.mission_start_time
                
                self.get_logger().info('='*70)
                self.get_logger().info('🎉🎉 SECOND PASS COMPLETE!')
                self.get_logger().info(f'   ✅✅ 2 QUALIFICATION POINTS EARNED')
                self.get_logger().info(f'   Total time: {total_time:.2f}s')
                self.get_logger().info('='*70)
                
                self.transition_to(self.COMPLETED)
                return cmd
            
            distance_past_gate = self.gate_x_position - current_x
            self.get_logger().info(
                f'🚀 PASSING 2: X={current_x:.2f}m, '
                f'{abs(distance_past_gate):.2f}m {"past" if distance_past_gate > 0 else "before"} gate',
                throttle_duration_sec=0.4
            )
        
        # Full speed straight passage
        cmd.linear.x = self.passing_speed
        cmd.linear.y = 0.0
        cmd.angular.z = 0.0
        
        return cmd
    
    def uturn_behavior(self, cmd: Twist) -> Twist:
        """Perform U-turn after first pass"""
        if self.uturn_start_position is None:
            self.uturn_start_position = self.current_position
            self.uturn_start_yaw = self.current_yaw
            self.get_logger().info('🔄 Starting U-turn...')
        
        # Calculate yaw change
        yaw_change = self.normalize_angle(self.current_yaw - self.uturn_start_yaw)
        
        # U-turn complete when rotated ~180 degrees
        if abs(abs(yaw_change) - math.pi) < 0.3:
            self.get_logger().info(
                f'✅ U-turn complete! Rotated {math.degrees(yaw_change):.1f}°'
            )
            self.get_logger().info('🔍 Searching for gate (reverse approach)...')
            self.transition_to(self.SEARCHING)
            return cmd
        
        # Execute U-turn: rotate + slight forward
        cmd.linear.x = 0.3
        cmd.angular.z = 0.8  # Moderate rotation speed
        
        self.get_logger().info(
            f'🔄 U-TURN: rotated {math.degrees(yaw_change):.1f}° / 180°',
            throttle_duration_sec=0.5
        )
        
        return cmd
    
    def completed_behavior(self, cmd: Twist) -> Twist:
        """Mission complete"""
        if self.completion_time:
            total_time = self.completion_time - self.mission_start_time
            
            self.get_logger().info('='*70)
            self.get_logger().info('🏆 QUALIFICATION COMPLETE!')
            self.get_logger().info(f'   Points: {self.qualification_points} / 2')
            self.get_logger().info(f'   Total time: {total_time:.2f}s')
            self.get_logger().info('='*70)
            
            self.completion_time = None  # Only log once
        
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
            self.STARTING: 'STARTING',
            self.SUBMERGING: 'SUBMERGING',
            self.SEARCHING: 'SEARCHING',
            self.APPROACHING_1: 'APPROACHING_1',
            self.ALIGNING_1: 'ALIGNING_1',
            self.PASSING_1: 'PASSING_1',
            self.UTURN: 'UTURN',
            self.APPROACHING_2: 'APPROACHING_2',
            self.ALIGNING_2: 'ALIGNING_2',
            self.PASSING_2: 'PASSING_2',
            self.COMPLETED: 'COMPLETED'
        }
        return names.get(self.state, 'UNKNOWN')
    
    def publish_detailed_status(self):
        """Publish detailed mission status"""
        if int(time.time()) % 2 == 0:  # Every 2 seconds
            status_lines = [
                f"State: {self.get_state_name()}",
                f"Points: {self.qualification_points}/2",
                f"Depth: {self.current_depth:.2f}m",
            ]
            
            if self.current_position:
                status_lines.append(f"Position: X={self.current_position[0]:.2f}m")
            
            if self.gate_detected:
                status_lines.append(f"Gate: {self.estimated_distance:.2f}m")
            
            if self.mission_start_time:
                elapsed = time.time() - self.mission_start_time
                status_lines.append(f"Time: {elapsed:.1f}s")
            
            status_msg = String()
            status_msg.data = " | ".join(status_lines)
            self.status_pub.publish(status_msg)
    
    @staticmethod
    def normalize_angle(angle):
        """Normalize angle to [-pi, pi]"""
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
        node.get_logger().info('Qualification Navigator shutting down')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()