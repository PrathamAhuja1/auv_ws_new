#!/usr/bin/env python3
"""
QUALIFICATION NAVIGATOR (FIXED DEPTH)
- Target Depth: -0.8m (Pool floor is -1.4m!)
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
        super().__init__('gate_navigator_node')
        
        # --- STATE MACHINE ---
        self.S_SEARCH_FWD = 0
        self.S_APPROACH_FWD = 1
        self.S_ALIGN_FWD = 2
        self.S_FINAL_APPROACH_FWD = 3
        self.S_PASS_FWD = 4
        self.S_U_TURN = 5
        self.S_SEARCH_REV = 6
        self.S_APPROACH_REV = 7
        self.S_ALIGN_REV = 8
        self.S_FINAL_APPROACH_REV = 9
        self.S_PASS_REV = 10
        self.S_COMPLETED = 11
        
        self.state = self.S_SEARCH_FWD
        
        # [CRITICAL FIX] Target depth must be shallower than pool floor (-1.4m)
        self.declare_parameter('target_depth', -0.8) 
        self.target_depth = self.get_parameter('target_depth').value
        
        self.approach_stop_dist = 3.0
        self.passing_trigger_dist = 1.0
        
        self.gate_detected = False
        self.frame_pos = 0.0
        self.dist = 999.0
        self.current_yaw = 0.0
        self.current_depth = 0.0
        self.start_u_turn_yaw = 0.0
        self.state_start_time = time.time()
        
        self.cmd_pub = self.create_publisher(Twist, '/rp2040/cmd_vel', 10)
        self.state_pub = self.create_publisher(String, '/mission/state', 10)
        self.reverse_mode_pub = self.create_publisher(Bool, '/mission/reverse_mode', 10)
        
        self.create_subscription(Bool, '/gate/detected', self.cb_detected, 10)
        self.create_subscription(Float32, '/gate/frame_position', self.cb_frame, 10)
        self.create_subscription(Float32, '/gate/estimated_distance', self.cb_dist, 10)
        self.create_subscription(Odometry, '/ground_truth/odom', self.cb_odom, 10)
        
        self.create_timer(0.05, self.control_loop)
        self.get_logger().info('🚀 QUALIFICATION NAVIGATOR (Depth Fixed to -0.8m)')

    def cb_detected(self, msg): self.gate_detected = msg.data
    def cb_frame(self, msg): self.frame_pos = msg.data
    def cb_dist(self, msg): self.dist = msg.data
    
    def cb_odom(self, msg):
        self.current_depth = msg.pose.pose.position.z
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

    def control_loop(self):
        cmd = Twist()
        
        # Depth Control
        depth_err = self.target_depth - self.current_depth
        # Simple P-Control for depth
        if abs(depth_err) > 0.05:
            cmd.linear.z = depth_err * 1.5 
            cmd.linear.z = max(-0.8, min(0.8, cmd.linear.z))
        
        # State Machine
        if self.state == self.S_SEARCH_FWD:
            cmd = self.behavior_search(cmd, self.S_APPROACH_FWD)
            self.reverse_mode_pub.publish(Bool(data=False))
            
        elif self.state == self.S_APPROACH_FWD:
            cmd = self.behavior_approach(cmd, self.S_ALIGN_FWD)
            self.reverse_mode_pub.publish(Bool(data=False))
            
        elif self.state == self.S_ALIGN_FWD:
            cmd = self.behavior_align(cmd, self.S_FINAL_APPROACH_FWD)
            self.reverse_mode_pub.publish(Bool(data=False))
            
        elif self.state == self.S_FINAL_APPROACH_FWD:
            cmd = self.behavior_final_approach(cmd, self.S_PASS_FWD)
            self.reverse_mode_pub.publish(Bool(data=False))
            
        elif self.state == self.S_PASS_FWD:
            cmd = self.behavior_pass(cmd, 6.0, self.S_U_TURN)
            self.reverse_mode_pub.publish(Bool(data=False))
            
        elif self.state == self.S_U_TURN:
            cmd = self.behavior_u_turn(cmd)
            self.reverse_mode_pub.publish(Bool(data=False))
            
        elif self.state == self.S_SEARCH_REV:
            cmd = self.behavior_search(cmd, self.S_APPROACH_REV)
            self.reverse_mode_pub.publish(Bool(data=True))
            
        elif self.state == self.S_APPROACH_REV:
            cmd = self.behavior_approach(cmd, self.S_ALIGN_REV)
            self.reverse_mode_pub.publish(Bool(data=True))
            
        elif self.state == self.S_ALIGN_REV:
            cmd = self.behavior_align(cmd, self.S_FINAL_APPROACH_REV)
            self.reverse_mode_pub.publish(Bool(data=True))
            
        elif self.state == self.S_FINAL_APPROACH_REV:
            cmd = self.behavior_final_approach(cmd, self.S_PASS_REV)
            self.reverse_mode_pub.publish(Bool(data=True))

        elif self.state == self.S_PASS_REV:
            cmd = self.behavior_pass(cmd, 6.0, self.S_COMPLETED)
            self.reverse_mode_pub.publish(Bool(data=True))
            
        elif self.state == self.S_COMPLETED:
            self.get_logger().info("🏆 MISSION COMPLETED", throttle_duration_sec=2)
            cmd = Twist()
            
        self.cmd_pub.publish(cmd)
        self.state_pub.publish(String(data=str(self.state)))

    # --- BEHAVIORS ---
    def behavior_search(self, cmd, next_state):
        # [FIX] Allow transition even if dist is 999 (partial detection)
        if self.gate_detected and (self.dist < 15.0 or self.dist > 900.0):
            self.set_state(next_state)
            return cmd
        
        t = time.time() - self.state_start_time
        cmd.angular.z = 0.3 if (t % 14 < 7) else -0.3
        cmd.linear.x = 0.3
        return cmd

    def behavior_approach(self, cmd, next_state):
        # Stop at 3m to align
        if self.dist <= self.approach_stop_dist and self.dist > 0.5:
            self.get_logger().info('🛑 3m Reached - Aligning')
            self.set_state(next_state)
            return cmd
        
        if not self.gate_detected: return cmd
        
        cmd.linear.x = 0.5
        cmd.angular.z = -self.frame_pos * 0.8
        return cmd

    def behavior_align(self, cmd, next_state):
        if abs(self.frame_pos) < 0.1:
            self.get_logger().info('✅ Aligned - Final Approach')
            self.set_state(next_state)
            return cmd
            
        cmd.linear.x = 0.0
        cmd.angular.z = -self.frame_pos * 1.5
        return cmd

    def behavior_final_approach(self, cmd, next_state):
        if self.dist < self.passing_trigger_dist:
            self.set_state(next_state)
            return cmd
        cmd.linear.x = 0.4
        cmd.angular.z = -self.frame_pos * 1.0
        return cmd

    def behavior_pass(self, cmd, duration, next_state):
        if time.time() - self.state_start_time > duration:
            self.set_state(next_state)
            return cmd
        cmd.linear.x = 1.0
        return cmd

    def behavior_u_turn(self, cmd):
        if self.start_u_turn_yaw == 0.0:
            self.start_u_turn_yaw = self.current_yaw
            
        diff = abs(self.current_yaw - self.start_u_turn_yaw)
        if diff > math.pi: diff = 2*math.pi - diff
        
        if diff > (math.pi * 0.9):
            self.start_u_turn_yaw = 0.0
            self.set_state(self.S_SEARCH_REV)
            return cmd
            
        cmd.linear.x = 0.3
        cmd.angular.z = 0.8
        return cmd

    def set_state(self, s):
        self.state = s
        self.state_start_time = time.time()
        self.start_u_turn_yaw = 0.0

def main(args=None):
    rclpy.init(args=args)
    node = QualificationNavigator()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()