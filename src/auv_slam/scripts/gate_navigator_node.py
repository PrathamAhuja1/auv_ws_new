#!/usr/bin/env python3
"""
Gate Navigator Node - Controls AUV to navigate through gate while avoiding orange flare
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import time

class GateNavigatorNode(Node):
    def __init__(self):
        super().__init__('gate_navigator_node', automatically_declare_parameters_from_overrides=True)
        
        # States
        self.SEARCHING = 0
        self.ALIGNING = 1
        self.APPROACHING = 2
        self.AVOIDING_FLARE = 3
        self.PASSING = 4
        self.COMPLETED = 5
        
        self.state = self.SEARCHING
        
        # --- NEW: Parameter loading flag ---
        self.params_loaded = False
        
        # --- Parameters will be initialized in control_loop ---
        self.target_depth = 0.0
        self.search_speed = 0.0
        self.approach_speed = 0.0
        self.passing_speed = 0.0
        self.passing_duration = 0.0
        self.align_threshold = 0.0
        self.safe_distance = 0.0
        self.yaw_gain = 0.0
        self.depth_gain = 0.0
        self.flare_gain = 0.0
        self.flare_avoidance_duration = 0.0
        
        # State variables
        self.gate_detected = False
        self.flare_detected = False
        self.alignment_error = 0.0
        self.estimated_distance = 0.0
        self.current_depth = 0.0
        self.flare_avoidance_direction = 0.0
        self.gate_lost_time = None
        self.passing_start_time = None 
        self.flare_avoidance_timer = None
        
        # Subscriptions
        self.gate_detected_sub = self.create_subscription(
            Bool, '/gate/detected', self.gate_detected_callback, 10)
        self.alignment_sub = self.create_subscription(
            Float32, '/gate/alignment_error', self.alignment_callback, 10)
        self.distance_sub = self.create_subscription(
            Float32, '/gate/estimated_distance', self.distance_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odometry/filtered', self.odom_callback, 10)
        self.flare_detected_sub = self.create_subscription(
            Bool, '/flare/detected', self.flare_detected_callback, 10)
        self.flare_avoidance_sub = self.create_subscription(
            Float32, '/flare/avoidance_direction', self.flare_avoidance_callback, 10)
        self.flare_warning_sub = self.create_subscription(
            String, '/flare/warning', self.flare_warning_callback, 10)
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/rp2040/cmd_vel', 10)
        self.state_pub = self.create_publisher(Float32, '/gate/navigation_state', 10)
        
        # Control loop (20 Hz)
        self.timer = self.create_timer(0.05, self.control_loop)
        
        self.get_logger().info('Gate Navigator with Flare Avoidance initialized')
    
    def gate_detected_callback(self, msg: Bool):
        was_detected = self.gate_detected
        self.gate_detected = msg.data
        if not was_detected and self.gate_detected:
            self.get_logger().info('Gate detected!')
            self.gate_lost_time = None
        elif was_detected and not self.gate_detected:
            self.gate_lost_time = time.time()
    
    def flare_detected_callback(self, msg: Bool):
        was_detected = self.flare_detected
        self.flare_detected = msg.data
        if not was_detected and self.flare_detected:
            self.get_logger().warn('Orange flare detected - initiating avoidance!')
            self.flare_avoidance_timer = time.time()
        elif was_detected and not self.flare_detected:
             self.get_logger().info('Flare no longer detected.')
             self.flare_avoidance_timer = None 
    
    def flare_avoidance_callback(self, msg: Float32):
        self.flare_avoidance_direction = msg.data
    
    def flare_warning_callback(self, msg: String):
        if "CRITICAL" in msg.data:
            self.get_logger().error(msg.data)
        else:
            self.get_logger().warn(msg.data)
    
    def alignment_callback(self, msg: Float32):
        self.alignment_error = msg.data
    
    def distance_callback(self, msg: Float32):
        self.estimated_distance = msg.data
    
    def odom_callback(self, msg: Odometry):
        self.current_depth = msg.pose.pose.position.z
    
    def control_loop(self):
        # --- NEW: Robust parameter loading ---
        if not self.params_loaded:
            try:
                self.target_depth = self.get_parameter('target_depth').value
                self.search_speed = self.get_parameter('search_forward_speed').value
                self.approach_speed = self.get_parameter('approach_speed').value
                self.passing_speed = self.get_parameter('passing_speed').value
                self.passing_duration = self.get_parameter('passing_duration').value
                self.align_threshold = self.get_parameter('alignment_threshold').value
                self.safe_distance = self.get_parameter('safe_distance_threshold').value
                self.yaw_gain = self.get_parameter('yaw_correction_gain').value
                self.depth_gain = self.get_parameter('depth_correction_gain').value
                self.flare_gain = self.get_parameter('flare_avoidance_gain').value
                self.flare_avoidance_duration = self.get_parameter('flare_avoidance_duration').value
                
                self.params_loaded = True
                self.get_logger().info('Gate navigator parameters loaded successfully.')
            except rclpy.exceptions.ParameterNotDeclaredException as e:
                self.get_logger().warn(f'Waiting for gate navigator parameters...: {e}')
                return # Try again on the next loop

        cmd = Twist()
        
        # Depth control (always active)
        depth_error = self.target_depth - self.current_depth
        cmd.linear.z = depth_error * self.depth_gain
        
        # Priority 1: Flare Avoidance
        if self.flare_detected and self.state in [self.ALIGNING, self.APPROACHING]:
            if self.flare_avoidance_timer:
                self.state = self.AVOIDING_FLARE
        
        # State machine
        if self.state == self.SEARCHING:
            cmd = self.searching_behavior(cmd)
        elif self.state == self.ALIGNING:
            cmd = self.aligning_behavior(cmd)
        elif self.state == self.APPROACHING:
            cmd = self.approaching_behavior(cmd)
        elif self.state == self.AVOIDING_FLARE:
            cmd = self.avoiding_flare_behavior(cmd)
        elif self.state == self.PASSING:
            cmd = self.passing_behavior(cmd)
        elif self.state == self.COMPLETED:
            cmd = self.completed_behavior(cmd)
        
        self.cmd_vel_pub.publish(cmd)
        
        state_msg = Float32()
        state_msg.data = float(self.state)
        self.state_pub.publish(state_msg)
    
    def searching_behavior(self, cmd: Twist) -> Twist:
        if self.gate_detected:
            self.get_logger().info('STATE: SEARCHING -> ALIGNING')
            self.state = self.ALIGNING
            return cmd
        
        cmd.linear.x = self.search_speed
        cmd.angular.z = 0.0 
        return cmd
    
    def aligning_behavior(self, cmd: Twist) -> Twist:
        if not self.gate_detected:
            if self.gate_lost_time and (time.time() - self.gate_lost_time > 2.0):
                self.get_logger().warn('STATE: ALIGNING -> SEARCHING (Gate lost)')
                self.state = self.SEARCHING
            return cmd
        
        if abs(self.alignment_error) < self.align_threshold:
            self.get_logger().info('STATE: ALIGNING -> APPROACHING')
            self.state = self.APPROACHING
            return cmd
        
        cmd.linear.x = 0.1 # Creep forward
        cmd.angular.z = -self.alignment_error * self.yaw_gain
        return cmd
    
    def approaching_behavior(self, cmd: Twist) -> Twist:
        if not self.gate_detected:
            if self.gate_lost_time and (time.time() - self.gate_lost_time > 1.5):
                self.get_logger().warn('STATE: APPROACHING -> SEARCHING (Gate lost)')
                self.state = self.SEARCHING
            return cmd
        
        if self.estimated_distance < self.safe_distance and self.estimated_distance > 0:
            self.get_logger().info('STATE: APPROACHING -> PASSING')
            self.state = self.PASSING
            self.passing_start_time = time.time()
            return cmd
        
        cmd.linear.x = self.approach_speed
        
        if abs(self.alignment_error) > self.align_threshold:
            cmd.angular.z = -self.alignment_error * self.yaw_gain * 0.5 # Gentle correction
        else:
            cmd.angular.z = 0.0
        
        return cmd
    
    def avoiding_flare_behavior(self, cmd: Twist) -> Twist:
        self.get_logger().info('STATE: AVOIDING FLARE')

        if not self.flare_detected or (time.time() - self.flare_avoidance_timer > self.flare_avoidance_duration):
            self.get_logger().info('STATE: AVOIDING_FLARE -> ALIGNING (Avoidance complete)')
            self.state = self.ALIGNING
            self.flare_avoidance_timer = None
            return cmd
        
        cmd.linear.y = self.flare_avoidance_direction * self.flare_gain
        
        if self.gate_detected:
            cmd.angular.z = -self.alignment_error * self.yaw_gain * 0.3
        
        cmd.linear.x = self.search_speed 
        
        return cmd

    def passing_behavior(self, cmd: Twist) -> Twist:
        self.get_logger().info('STATE: PASSING')

        if time.time() - self.passing_start_time > self.passing_duration:
            self.get_logger().info('STATE: PASSING -> COMPLETED')
            self.state = self.COMPLETED
            return self.completed_behavior(cmd) 
        
        cmd.linear.x = self.passing_speed
        cmd.angular.z = 0.0 
        cmd.linear.y = 0.0
        
        return cmd

    def completed_behavior(self, cmd: Twist) -> Twist:
        self.get_logger().info('STATE: COMPLETED. Gate navigation finished.')
        cmd.linear.x = 0.0
        cmd.linear.y = 0.0
        cmd.linear.z = 0.0 
        cmd.angular.z = 0.0
        
        if self.timer:
            self.timer.cancel()
            self.timer = None
            
        return cmd

def main(args=None):
    rclpy.init(args=args)
    node = GateNavigatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()