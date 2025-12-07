#!/usr/bin/env python3
"""
FIXED QUALIFICATION NAVIGATOR
1. DEPTH: EXACT COPY of gate_navigator_node logic (0.3m deadband)
2. LOGIC: Stop at 3m -> Align -> Final Approach
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
        self.WAITING_TO_START = 0
        self.SUBMERGING = 1
        
        self.FORWARD_SEARCH = 2
        self.FORWARD_APPROACH = 3      # > 3m
        self.FORWARD_ALIGNING = 4      # Stop at 3m
        self.FORWARD_FINAL_APPROACH = 5 # 3m -> 1.2m
        self.FORWARD_PASSING = 6
        
        self.U_TURN = 7
        
        self.REVERSE_SEARCH = 8
        self.REVERSE_APPROACH = 9
        self.REVERSE_ALIGNING = 10
        self.REVERSE_FINAL_APPROACH = 11
        self.REVERSE_PASSING = 12
        
        self.COMPLETED = 13
        
        self.state = self.WAITING_TO_START
        
        # Params
        self.declare_parameter('target_depth', -0.8)
        self.declare_parameter('passing_trigger_distance', 1.2)
        
        self.target_depth = self.get_parameter('target_depth').value
        self.passing_trigger = self.get_parameter('passing_trigger_distance').value
        
        # Variables
        self.gate_detected = False
        self.alignment_error = 0.0
        self.estimated_distance = 999.0
        self.current_depth = 0.0
        self.current_yaw = 0.0
        self.target_yaw = 0.0
        self.initial_yaw_set = False
        
        self.gate_lost_time = 0.0
        self.state_start_time = time.time()
        self.uturn_start_yaw = 0.0
        
        # Subs
        self.create_subscription(Bool, '/qualification/gate_detected', self.gate_cb, 10)
        self.create_subscription(Float32, '/qualification/alignment_error', self.align_cb, 10)
        self.create_subscription(Float32, '/qualification/estimated_distance', self.dist_cb, 10)
        self.create_subscription(Odometry, '/ground_truth/odom', self.odom_cb, 10)
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/rp2040/cmd_vel', 10)
        self.state_pub = self.create_publisher(String, '/qualification/state', 10)
        
        self.create_timer(0.05, self.control_loop)
        self.get_logger().info('✅ FIXED NAVIGATOR: Reference Depth Logic')

    # --- CALLBACKS ---
    def gate_cb(self, msg):
        if self.gate_detected and not msg.data: self.gate_lost_time = time.time()
        elif not self.gate_detected and msg.data: self.gate_lost_time = 0.0
        self.gate_detected = msg.data

    def align_cb(self, msg): self.alignment_error = msg.data
    def dist_cb(self, msg): self.estimated_distance = msg.data
    
    def odom_cb(self, msg):
        self.current_depth = msg.pose.pose.position.z
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)
        
        if not self.initial_yaw_set:
            self.target_yaw = self.current_yaw
            self.initial_yaw_set = True

    # --- CONTROL LOOP ---
    def control_loop(self):
        cmd = Twist()
        
        # ==========================================================
        # 1. DEPTH CONTROL - COPIED EXACTLY FROM gate_navigator_node.py
        # This fixes the oscillation/jumping
        # ==========================================================
        if self.state != self.WAITING_TO_START:
            depth_error = self.target_depth - self.current_depth
            depth_deadband = 0.3  # Wide deadband from reference
            
            if abs(depth_error) < depth_deadband:
                cmd.linear.z = 0.0
            elif abs(depth_error) < 0.6:
                cmd.linear.z = depth_error * 0.8
                cmd.linear.z = max(-0.4, min(cmd.linear.z, 0.4))
            else:
                cmd.linear.z = depth_error * 1.2
                cmd.linear.z = max(-1.0, min(cmd.linear.z, 1.0))

        # 2. STATE MACHINE
        if self.state == self.WAITING_TO_START:     cmd = self.wait(cmd)
        elif self.state == self.SUBMERGING:         cmd = self.submerge(cmd)
        
        # Forward
        elif self.state == self.FORWARD_SEARCH:         cmd = self.search(cmd, "FWD")
        elif self.state == self.FORWARD_APPROACH:       cmd = self.approach_far(cmd, "FWD")
        elif self.state == self.FORWARD_ALIGNING:       cmd = self.align_stop(cmd, "FWD")
        elif self.state == self.FORWARD_FINAL_APPROACH: cmd = self.approach_near(cmd, "FWD")
        elif self.state == self.FORWARD_PASSING:        cmd = self.passing(cmd, "FWD")
        
        elif self.state == self.U_TURN:                 cmd = self.uturn(cmd)
        
        # Reverse
        elif self.state == self.REVERSE_SEARCH:         cmd = self.search(cmd, "REV")
        elif self.state == self.REVERSE_APPROACH:       cmd = self.approach_far(cmd, "REV")
        elif self.state == self.REVERSE_ALIGNING:       cmd = self.align_stop(cmd, "REV")
        elif self.state == self.REVERSE_FINAL_APPROACH: cmd = self.approach_near(cmd, "REV")
        elif self.state == self.REVERSE_PASSING:        cmd = self.passing(cmd, "REV")
        
        self.cmd_vel_pub.publish(cmd)
        self.state_pub.publish(String(data=str(self.state)))

    # --- BEHAVIORS ---
    def wait(self, cmd):
        if time.time() - self.state_start_time > 3.0: self.transition_to(self.SUBMERGING)
        return cmd

    def submerge(self, cmd):
        # Using 0.3 here to match the control loop deadband roughly
        if abs(self.target_depth - self.current_depth) < 0.35:
            self.transition_to(self.FORWARD_SEARCH)
        return cmd

    def search(self, cmd, mode):
        t = time.time()
        cmd.angular.z = 0.3 if (t % 10 < 5) else -0.3
        cmd.linear.x = 0.1 # Very slow forward drift to help
        
        if self.gate_detected and self.estimated_distance < 999:
            self.get_logger().info(f"🎯 Gate Found at {self.estimated_distance:.1f}m")
            if mode == "FWD": self.transition_to(self.FORWARD_APPROACH)
            else:             self.transition_to(self.REVERSE_APPROACH)
        return cmd

    def approach_far(self, cmd, mode):
        """Fast approach until 3m"""
        if self.check_lost(mode): return cmd
        
        # STOP AT 3 METERS
        if self.estimated_distance <= 3.0:
            self.get_logger().info(f"🛑 Reached {self.estimated_distance:.2f}m -> STOPPING to Align")
            if mode == "FWD": self.transition_to(self.FORWARD_ALIGNING)
            else:             self.transition_to(self.REVERSE_ALIGNING)
            return cmd
            
        cmd.linear.x = 0.5
        cmd.angular.z = -self.alignment_error * 1.2
        return cmd

    def align_stop(self, cmd, mode):
        """Zero forward speed, rotate only"""
        if self.check_lost(mode): return cmd
        
        cmd.linear.x = 0.0
        cmd.angular.z = -self.alignment_error * 1.5
        
        if abs(self.alignment_error) < 0.08:
            self.get_logger().info("✅ Aligned -> Final Approach")
            if mode == "FWD": self.transition_to(self.FORWARD_FINAL_APPROACH)
            else:             self.transition_to(self.REVERSE_FINAL_APPROACH)
            
        if time.time() - self.state_start_time > 8.0:
            self.get_logger().warn("Align Timeout")
            if mode == "FWD": self.transition_to(self.FORWARD_FINAL_APPROACH)
            else:             self.transition_to(self.REVERSE_FINAL_APPROACH)
        return cmd

    def approach_near(self, cmd, mode):
        if self.check_lost(mode): return cmd
        
        if self.estimated_distance <= self.passing_trigger:
            self.get_logger().info("🚀 Committing to Pass")
            if mode == "FWD": self.transition_to(self.FORWARD_PASSING)
            else:             self.transition_to(self.REVERSE_PASSING)
            return cmd
            
        cmd.linear.x = 0.3
        cmd.angular.z = -self.alignment_error * 1.5
        return cmd

    def passing(self, cmd, mode):
        cmd.linear.x = 0.8
        if time.time() - self.state_start_time > 6.0:
            if mode == "FWD": 
                self.uturn_start_yaw = self.current_yaw
                self.transition_to(self.U_TURN)
            else:
                self.transition_to(self.COMPLETED)
        return cmd

    def uturn(self, cmd):
        turned = abs(self.normalize_angle(self.current_yaw - self.uturn_start_yaw))
        if abs(math.pi - turned) < 0.2:
            self.target_yaw = self.normalize_angle(self.target_yaw + math.pi)
            self.transition_to(self.REVERSE_SEARCH)
            return cmd
        cmd.linear.x = 0.1
        cmd.angular.z = 0.5
        return cmd

    def check_lost(self, mode):
        if not self.gate_detected and (time.time() - self.gate_lost_time > 4.0) and self.gate_lost_time > 0:
            self.get_logger().warn("Lost Gate -> Search")
            if mode == "FWD": self.transition_to(self.FORWARD_SEARCH)
            else:             self.transition_to(self.REVERSE_SEARCH)
            return True
        return False

    def transition_to(self, state):
        self.state = state
        self.state_start_time = time.time()
    
    def normalize_angle(self, angle):
        while angle > math.pi: angle -= 2 * math.pi
        while angle < -math.pi: angle += 2 * math.pi
        return angle

def main(args=None):
    rclpy.init(args=args)
    node = QualificationNavigator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()