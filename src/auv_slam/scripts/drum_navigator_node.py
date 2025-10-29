#!/usr/bin/env python3
"""
Drum Navigator Node - Navigates to and hovers over drums for ball dropping
Task 2: Target Acquisition
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
import time
import math


class DrumNavigatorNode(Node):
    def __init__(self):
        super().__init__('drum_navigator_node')
        
        # States
        self.IDLE = 0
        self.SEARCHING = 1
        self.CENTERING_BLUE = 2
        self.HOVERING_BLUE = 3
        self.CENTERING_RED = 4
        self.HOVERING_RED = 5
        self.COMPLETED = 6
        
        self.state = self.IDLE
        
        # Parameters
        self.declare_parameter('target_depth', -2.7)  # Just above drums
        self.declare_parameter('hover_height', 0.5)    # Height above drum
        self.declare_parameter('centering_speed', 0.15)
        self.declare_parameter('centering_threshold', 30)  # pixels
        self.declare_parameter('hover_duration', 3.0)      # Time to hover for drop
        
        self.target_depth = self.get_parameter('target_depth').value
        self.hover_height = self.get_parameter('hover_height').value
        self.centering_speed = self.get_parameter('centering_speed').value
        self.center_threshold = self.get_parameter('centering_threshold').value
        self.hover_duration = self.get_parameter('hover_duration').value
        
        # State variables
        self.current_depth = 0.0
        self.current_position = None
        self.blue_drum_detected = False
        self.red_drum_detected = False
        self.blue_drum_center = None
        self.red_drum_center = None
        self.blue_drum_distance = 999.0
        self.closest_drum_color = None
        self.image_center_x = 400  # Will be updated from camera info
        self.image_center_y = 300
        
        self.state_start_time = time.time()
        self.ball_drop_commanded = False
        
        # Subscriptions
        self.odom_sub = self.create_subscription(
            Odometry, '/ground_truth/odom', self.odom_callback, 10)
        
        self.blue_detected_sub = self.create_subscription(
            Bool, '/drum/blue_detected', self.blue_detected_callback, 10)
        
        self.blue_pose_sub = self.create_subscription(
            PoseStamped, '/drum/blue_pose', self.blue_pose_callback, 10)
        
        self.blue_distance_sub = self.create_subscription(
            Float32, '/drum/blue_distance', self.blue_distance_callback, 10)
        
        self.red_detected_sub = self.create_subscription(
            Bool, '/drum/red_detected', self.red_detected_callback, 10)
        
        self.red_pose_sub = self.create_subscription(
            PoseStamped, '/drum/red_pose', self.red_pose_callback, 10)
        
        self.closest_sub = self.create_subscription(
            String, '/drum/closest_color', self.closest_callback, 10)
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel_input', 10)
        self.state_pub = self.create_publisher(String, '/drum_nav/state', 10)
        self.drop_cmd_pub = self.create_publisher(Bool, '/ball_dropper/drop', 10)
        
        # Control loop (20 Hz)
        self.timer = self.create_timer(0.05, self.control_loop)
        
        self.get_logger().info('Drum Navigator Node initialized')
        self.get_logger().info(f'Target depth: {self.target_depth}m')
    
    # ===== CALLBACKS =====
    
    def odom_callback(self, msg: Odometry):
        self.current_depth = msg.pose.pose.position.z
        self.current_position = msg.pose.pose.position
    
    def blue_detected_callback(self, msg: Bool):
        self.blue_drum_detected = msg.data
    
    def blue_pose_callback(self, msg: PoseStamped):
        self.blue_drum_center = (int(msg.pose.position.x), int(msg.pose.position.y))
    
    def blue_distance_callback(self, msg: Float32):
        self.blue_drum_distance = msg.data
    
    def red_detected_callback(self, msg: Bool):
        self.red_drum_detected = msg.data
    
    def red_pose_callback(self, msg: PoseStamped):
        self.red_drum_center = (int(msg.pose.position.x), int(msg.pose.position.y))
    
    def closest_callback(self, msg: String):
        self.closest_drum_color = msg.data
    
    # ===== CONTROL =====
    
    def start_navigation(self):
        """Called by main mission controller to start drum navigation"""
        self.state = self.SEARCHING
        self.state_start_time = time.time()
        self.get_logger().info('Starting drum navigation')
    
    def control_loop(self):
        """Main control loop"""
        
        if self.state == self.IDLE:
            return
        
        cmd = Twist()
        
        # Depth control (always active)
        depth_error = self.target_depth - self.current_depth
        cmd.linear.z = depth_error * 1.0
        
        # State machine
        if self.state == self.SEARCHING:
            cmd = self.handle_searching(cmd)
        elif self.state == self.CENTERING_BLUE:
            cmd = self.handle_centering_blue(cmd)
        elif self.state == self.HOVERING_BLUE:
            cmd = self.handle_hovering_blue(cmd)
        elif self.state == self.CENTERING_RED:
            cmd = self.handle_centering_red(cmd)
        elif self.state == self.HOVERING_RED:
            cmd = self.handle_hovering_red(cmd)
        elif self.state == self.COMPLETED:
            cmd = self.handle_completed(cmd)
        
        self.cmd_vel_pub.publish(cmd)
        
        # Publish state
        state_msg = String()
        state_msg.data = self.get_state_name()
        self.state_pub.publish(state_msg)
    
    # ===== STATE HANDLERS =====
    
    def handle_searching(self, cmd: Twist) -> Twist:
        """Search for drums"""
        elapsed = time.time() - self.state_start_time
        
        # Check if any drum detected
        if self.blue_drum_detected:
            self.get_logger().info('✅ Blue drum detected! Targeting for 30 points')
            self.transition_to(self.CENTERING_BLUE)
            return cmd
        
        if self.red_drum_detected and elapsed > 5.0:  # Prefer blue, but accept red
            self.get_logger().info('Red drum detected - will target if no blue found')
        
        # Timeout - if no blue drum after 20 seconds, go for red
        if elapsed > 20.0 and self.red_drum_detected:
            self.get_logger().warn('Blue drum not found - targeting red drum')
            self.transition_to(self.CENTERING_RED)
            return cmd
        
        # Search pattern: move forward slowly
        cmd.linear.x = 0.2
        cmd.linear.y = 0.0
        cmd.angular.z = 0.0
        
        if int(elapsed) % 3 == 0:
            self.get_logger().info(
                f'🔍 Searching for drums... Blue: {self.blue_drum_detected}, '
                f'Red: {self.red_drum_detected}'
            )
        
        return cmd
    
    def handle_centering_blue(self, cmd: Twist) -> Twist:
        """Center over blue drum"""
        
        if not self.blue_drum_detected or self.blue_drum_center is None:
            self.get_logger().warn('Lost blue drum - returning to search')
            self.transition_to(self.SEARCHING)
            return cmd
        
        # Calculate centering error
        error_x = self.blue_drum_center[0] - self.image_center_x
        error_y = self.blue_drum_center[1] - self.image_center_y
        
        # Check if centered
        if abs(error_x) < self.center_threshold and abs(error_y) < self.center_threshold:
            self.get_logger().info('✅ Centered over blue drum!')
            self.transition_to(self.HOVERING_BLUE)
            return cmd
        
        # Move to center drum
        # Camera frame: X=right, Y=down
        # AUV frame: X=forward, Y=left
        cmd.linear.y = -error_x * 0.0005  # Move left/right
        cmd.linear.x = -error_y * 0.0005  # Move forward/back
        cmd.angular.z = 0.0
        
        self.get_logger().info(
            f'Centering: error_x={error_x:.0f}px, error_y={error_y:.0f}px, '
            f'dist={self.blue_drum_distance:.2f}m',
            throttle_duration_sec=0.5
        )
        
        return cmd
    
    def handle_hovering_blue(self, cmd: Twist) -> Twist:
        """Hover over blue drum and drop ball"""
        elapsed = time.time() - self.state_start_time
        
        # Hover for specified duration
        if elapsed > self.hover_duration:
            if not self.ball_drop_commanded:
                self.get_logger().info('💧 DROPPING BALL IN BLUE DRUM!')
                drop_msg = Bool()
                drop_msg.data = True
                self.drop_cmd_pub.publish(drop_msg)
                self.ball_drop_commanded = True
            
            # Wait a bit more to ensure drop
            if elapsed > self.hover_duration + 2.0:
                self.get_logger().info('✅ Ball dropped! Task complete')
                self.transition_to(self.COMPLETED)
                return cmd
        
        # Maintain hover position
        if self.blue_drum_center:
            error_x = self.blue_drum_center[0] - self.image_center_x
            error_y = self.blue_drum_center[1] - self.image_center_y
            
            cmd.linear.y = -error_x * 0.0003
            cmd.linear.x = -error_y * 0.0003
            cmd.angular.z = 0.0
        
        self.get_logger().info(
            f'🎯 Hovering over blue drum... {elapsed:.1f}s',
            throttle_duration_sec=0.5
        )
        
        return cmd
    
    def handle_centering_red(self, cmd: Twist) -> Twist:
        """Center over red drum (fallback)"""
        
        if not self.red_drum_detected or self.red_drum_center is None:
            self.get_logger().warn('Lost red drum - returning to search')
            self.transition_to(self.SEARCHING)
            return cmd
        
        error_x = self.red_drum_center[0] - self.image_center_x
        error_y = self.red_drum_center[1] - self.image_center_y
        
        if abs(error_x) < self.center_threshold and abs(error_y) < self.center_threshold:
            self.get_logger().info('✅ Centered over red drum!')
            self.transition_to(self.HOVERING_RED)
            return cmd
        
        cmd.linear.y = -error_x * 0.0005
        cmd.linear.x = -error_y * 0.0005
        cmd.angular.z = 0.0
        
        self.get_logger().info(
            f'Centering (RED): error_x={error_x:.0f}px, error_y={error_y:.0f}px',
            throttle_duration_sec=0.5
        )
        
        return cmd
    
    def handle_hovering_red(self, cmd: Twist) -> Twist:
        """Hover over red drum"""
        elapsed = time.time() - self.state_start_time
        
        if elapsed > self.hover_duration:
            if not self.ball_drop_commanded:
                self.get_logger().info('💧 DROPPING BALL IN RED DRUM (10 points)')
                drop_msg = Bool()
                drop_msg.data = True
                self.drop_cmd_pub.publish(drop_msg)
                self.ball_drop_commanded = True
            
            if elapsed > self.hover_duration + 2.0:
                self.get_logger().info('✅ Ball dropped in red drum')
                self.transition_to(self.COMPLETED)
                return cmd
        
        if self.red_drum_center:
            error_x = self.red_drum_center[0] - self.image_center_x
            error_y = self.red_drum_center[1] - self.image_center_y
            
            cmd.linear.y = -error_x * 0.0003
            cmd.linear.x = -error_y * 0.0003
            cmd.angular.z = 0.0
        
        self.get_logger().info(
            f'🎯 Hovering over red drum... {elapsed:.1f}s',
            throttle_duration_sec=0.5
        )
        
        return cmd
    
    def handle_completed(self, cmd: Twist) -> Twist:
        """Task completed - stop motion"""
        cmd.linear.x = 0.0
        cmd.linear.y = 0.0
        cmd.linear.z = 0.0
        cmd.angular.z = 0.0
        return cmd
    
    # ===== UTILITIES =====
    
    def transition_to(self, new_state: int):
        """Transition to new state"""
        old_name = self.get_state_name()
        self.state = new_state
        self.state_start_time = time.time()
        new_name = self.get_state_name()
        
        self.get_logger().info(f'🔄 DRUM NAV: {old_name} → {new_name}')
    
    def get_state_name(self) -> str:
        """Get current state name"""
        names = {
            self.IDLE: 'IDLE',
            self.SEARCHING: 'SEARCHING',
            self.CENTERING_BLUE: 'CENTERING_BLUE',
            self.HOVERING_BLUE: 'HOVERING_BLUE',
            self.CENTERING_RED: 'CENTERING_RED',
            self.HOVERING_RED: 'HOVERING_RED',
            self.COMPLETED: 'COMPLETED'
        }
        return names.get(self.state, 'UNKNOWN')


def main(args=None):
    rclpy.init(args=args)
    node = DrumNavigatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()