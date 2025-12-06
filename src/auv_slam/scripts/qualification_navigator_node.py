#!/usr/bin/env python3
"""
QUALIFICATION Navigator
Sequences: Forward Search -> Approach -> Pass -> U-Turn -> Reverse Search -> Approach -> Pass
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
        self.S_PASS_FWD = 3
        self.S_POST_PASS_FWD = 4  # Move clear of gate
        self.S_U_TURN = 5         # Perform 180
        self.S_SEARCH_REV = 6
        self.S_APPROACH_REV = 7
        self.S_ALIGN_REV = 8
        self.S_PASS_REV = 9
        self.S_COMPLETED = 10
        
        self.state = self.S_SEARCH_FWD
        
        # Parameters
        self.declare_parameter('target_depth', -1.7)
        self.target_depth = self.get_parameter('target_depth').value
        
        # Config
        self.search_speed = 0.5
        self.search_yaw = 0.15
        self.approach_stop_dist = 3.0
        self.pass_speed = 1.0
        
        # U-Turn Params
        self.u_turn_yaw_speed = 0.4
        self.u_turn_duration = 4.0 # Fallback if no odom
        self.target_yaw_u_turn = math.pi # 180 degrees
        
        # Inputs
        self.gate_detected = False
        self.frame_pos = 0.0
        self.dist = 999.0
        self.current_yaw = 0.0
        self.start_u_turn_yaw = 0.0
        self.current_pos = None
        self.pass_start_pos = None
        
        # Timers
        self.state_start_time = time.time()
        
        # Pubs/Subs
        self.cmd_pub = self.create_publisher(Twist, '/rp2040/cmd_vel', 10)
        self.reverse_mode_pub = self.create_publisher(Bool, '/mission/reverse_mode', 10)
        self.state_pub = self.create_publisher(String, '/mission/state', 10)
        
        self.create_subscription(Bool, '/gate/detected', self.cb_detected, 10)
        self.create_subscription(Float32, '/gate/frame_position', self.cb_frame, 10)
        self.create_subscription(Float32, '/gate/estimated_distance', self.cb_dist, 10)
        self.create_subscription(Odometry, '/ground_truth/odom', self.cb_odom, 10)
        
        self.create_timer(0.05, self.control_loop)
        self.get_logger().info('🚀 QUALIFICATION NAVIGATOR STARTED')

    # --- CALLBACKS ---
    def cb_detected(self, msg): self.gate_detected = msg.data
    def cb_frame(self, msg): self.frame_pos = msg.data
    def cb_dist(self, msg): self.dist = msg.data
    def cb_odom(self, msg):
        self.current_pos = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        # Convert quaternion to yaw
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

    # --- CONTROL LOOP ---
    def control_loop(self):
        cmd = Twist()
        
        # Depth Hold
        cmd.linear.z = max(-0.5, min(0.5, (self.target_depth - (-1.7)) * 1.5)) 
        
        # State Machine Logic
        if self.state == self.S_SEARCH_FWD:
            cmd = self.behavior_search(cmd)
        elif self.state == self.S_APPROACH_FWD:
            cmd = self.behavior_approach(cmd, next_state=self.S_ALIGN_FWD)
        elif self.state == self.S_ALIGN_FWD:
            cmd = self.behavior_align(cmd, next_state=self.S_PASS_FWD)
        elif self.state == self.S_PASS_FWD:
            cmd = self.behavior_pass(cmd, duration=8.0, next_state=self.S_POST_PASS_FWD)
            
        elif self.state == self.S_POST_PASS_FWD:
            # Move forward blindly for 3 seconds to ensure we are well clear of gate for U-turn
            if time.time() - self.state_start_time > 3.0:
                self.set_state(self.S_U_TURN)
            cmd.linear.x = 0.5
            
        elif self.state == self.S_U_TURN:
            cmd = self.behavior_u_turn(cmd)
            
        elif self.state == self.S_SEARCH_REV:
            self.reverse_mode_pub.publish(Bool(data=True)) # Tell detector to swap colors
            cmd = self.behavior_search(cmd)
            
        elif self.state == self.S_APPROACH_REV:
            self.reverse_mode_pub.publish(Bool(data=True))
            cmd = self.behavior_approach(cmd, next_state=self.S_ALIGN_REV)
            
        elif self.state == self.S_ALIGN_REV:
            self.reverse_mode_pub.publish(Bool(data=True))
            cmd = self.behavior_align(cmd, next_state=self.S_PASS_REV)
            
        elif self.state == self.S_PASS_REV:
            self.reverse_mode_pub.publish(Bool(data=True))
            cmd = self.behavior_pass(cmd, duration=8.0, next_state=self.S_COMPLETED)
            
        elif self.state == self.S_COMPLETED:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.get_logger().info("🏆 QUALIFICATION TASK COMPLETED", throttle_duration_sec=2.0)
            
        self.cmd_pub.publish(cmd)
        self.state_pub.publish(String(data=str(self.state)))

    # --- BEHAVIORS ---
    
    def behavior_search(self, cmd):
        if self.gate_detected and self.dist < 15.0:
            if self.state == self.S_SEARCH_FWD: self.set_state(self.S_APPROACH_FWD)
            else: self.set_state(self.S_APPROACH_REV)
            return cmd
        
        # Sweep
        t = time.time()
        cmd.angular.z = self.search_yaw if (t % 10 < 5) else -self.search_yaw
        cmd.linear.x = 0.2
        return cmd

    def behavior_approach(self, cmd, next_state):
        if not self.gate_detected: return cmd # Wait/Coast
        
        if self.dist < self.approach_stop_dist:
            self.set_state(next_state)
            return cmd
            
        cmd.linear.x = 0.5
        cmd.angular.z = -self.frame_pos * 1.0 # P-control to keep gate centered
        return cmd

    def behavior_align(self, cmd, next_state):
        if time.time() - self.state_start_time > 10.0: # Timeout
            self.set_state(next_state)
        
        if abs(self.frame_pos) < 0.1:
            self.get_logger().info("Aligned! Starting Pass.")
            self.set_state(next_state)
            
        cmd.linear.x = 0.05 # Creep forward
        cmd.angular.z = -self.frame_pos * 1.5 # Strong correction
        return cmd

    def behavior_pass(self, cmd, duration, next_state):
        # Use Distance Based Passing (Robust)
        if self.pass_start_pos is None:
            self.pass_start_pos = self.current_pos
            
        dist_traveled = 0.0
        if self.current_pos and self.pass_start_pos:
            dx = self.current_pos[0] - self.pass_start_pos[0]
            dy = self.current_pos[1] - self.pass_start_pos[1]
            dist_traveled = math.sqrt(dx*dx + dy*dy)
            
        # Complete if moved 3 meters past start of pass
        if dist_traveled > 3.0 or (time.time() - self.state_start_time > duration):
            self.set_state(next_state)
            self.pass_start_pos = None
            
        cmd.linear.x = self.pass_speed
        cmd.angular.z = 0.0 # Straight
        return cmd

    def behavior_u_turn(self, cmd):
        if self.start_u_turn_yaw == 0.0:
            self.start_u_turn_yaw = self.current_yaw
            self.get_logger().info(f"Starting U-Turn from yaw={self.current_yaw:.2f}")
            
        # Check angle difference
        diff = abs(self.current_yaw - self.start_u_turn_yaw)
        if diff > math.pi: diff = 2*math.pi - diff # Wrap handling
        
        if diff > (math.pi * 0.9): # 90% of 180 degrees done
            self.get_logger().info("U-Turn Complete")
            self.set_state(self.S_SEARCH_REV)
            self.start_u_turn_yaw = 0.0
            return cmd
            
        cmd.linear.x = 0.2 # Small forward movement to create turning radius
        cmd.angular.z = self.u_turn_yaw_speed
        return cmd

    def set_state(self, new_state):
        self.state = new_state
        self.state_start_time = time.time()
        self.get_logger().info(f"State -> {new_state}")

def main(args=None):
    rclpy.init(args=args)
    node = QualificationNavigator()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()