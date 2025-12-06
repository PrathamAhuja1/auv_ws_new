#!/usr/bin/env python3
"""
QUALIFICATION NAVIGATOR (FINAL)
- Logic: Search -> Approach -> Align(XYZ) -> Pass -> U-Turn -> Pass Again
- Fixes: Depth Oscillation (Deadband), Floor Collisions
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
        super().__init__('qualification_navigator_node')
        
        # --- CONFIGURATION ---
        self.declare_parameter('target_depth', -0.8)
        self.safe_depth = self.get_parameter('target_depth').value
        
        # Camera Parameters (for Visual Depth Alignment)
        self.img_height = 600
        self.img_center_y = self.img_height // 2
        
        # PID Depth Gains (Tuned for Stability)
        self.kp_depth = 1.0
        self.kd_depth = 0.5
        self.depth_deadband = 0.05  # Ignore errors smaller than 5cm
        
        # Mission Parameters
        self.align_dist = 3.0       # Start aligning height at 3m
        self.pass_dist = 1.2        # Commit to pass at 1.2m
        
        # --- STATE MACHINE ---
        self.S_SEARCH = 0
        self.S_APPROACH = 1
        self.S_ALIGN = 2
        self.S_FINAL = 3
        self.S_PASS = 4
        self.S_UTURN = 5
        self.S_COMPLETE = 6
        
        self.state = self.S_SEARCH
        self.passes_completed = 0
        self.target_depth_dynamic = self.safe_depth
        
        # Variables
        self.gate_detected = False
        self.gate_pos_x = 0.0       # -1 to 1 (Frame Position)
        self.gate_center_y = 0.0    # Pixel Y
        self.dist = 999.0
        self.current_depth = 0.0
        self.current_yaw = 0.0
        self.start_yaw = 0.0
        
        self.last_time = time.time()
        self.state_start_time = time.time()
        
        # --- ROS INTERFACE ---
        self.cmd_pub = self.create_publisher(Twist, '/rp2040/cmd_vel', 10)
        self.state_pub = self.create_publisher(String, '/mission/state', 10)
        
        self.create_subscription(Bool, '/gate/detected', self.cb_detected, 10)
        self.create_subscription(Float32, '/gate/frame_position', self.cb_frame_x, 10)
        self.create_subscription(Point, '/gate/center_point', self.cb_center_point, 10)
        self.create_subscription(Float32, '/gate/estimated_distance', self.cb_dist, 10)
        self.create_subscription(Odometry, '/ground_truth/odom', self.cb_odom, 10)
        
        self.create_timer(0.05, self.control_loop) # 20Hz
        
        self.get_logger().info('🚀 NAVIGATOR READY: 2-Pass Logic + Stable Depth')

    # --- CALLBACKS ---
    def cb_detected(self, msg): self.gate_detected = msg.data
    def cb_frame_x(self, msg): self.gate_pos_x = msg.data
    def cb_dist(self, msg): self.dist = msg.data
    def cb_center_point(self, msg): 
        # msg.y is the pixel height (0 is top, 600 is bottom)
        self.gate_center_y = msg.y
        
    def cb_odom(self, msg):
        self.current_depth = msg.pose.pose.position.z
        # Extract Yaw
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

    # --- MAIN LOOP ---
    def control_loop(self):
        cmd = Twist()
        
        # 1. DEPTH CONTROL (Always Active)
        # ---------------------------------------------------
        # Decide Target Depth
        target = self.safe_depth
        
        # "Adjust height" logic: Only when close and aligning
        if self.state in [self.S_ALIGN, self.S_FINAL] and self.gate_detected:
            # Visual Servo: Map pixel error to depth adjustment
            # Center is 300. If y=400 (lower), we need to go DOWN (more negative Z)
            pixel_error = (self.gate_center_y - self.img_center_y)
            
            # Simple P-control for target adjustment
            # Scale: 100 pixels off = 0.1m adjustment
            adjustment = -(pixel_error / 1000.0) 
            
            # Update dynamic target smoothly
            self.target_depth_dynamic += adjustment * 0.1
            
            # Safety Clamp (Don't go too deep or surface)
            self.target_depth_dynamic = max(-1.2, min(-0.4, self.target_depth_dynamic))
            target = self.target_depth_dynamic
        else:
            # Reset to safe depth when searching/moving fast
            self.target_depth_dynamic = self.safe_depth
        
        # Calculate Thrust with Deadband
        error = target - self.current_depth
        if abs(error) < self.depth_deadband:
            cmd.linear.z = 0.0 # Stop jitter
        else:
            cmd.linear.z = error * self.kp_depth
            cmd.linear.z = max(-0.5, min(0.5, cmd.linear.z)) # Limit speed
            
        # 2. STATE MACHINE
        # ---------------------------------------------------
        if self.state == self.S_SEARCH:
            cmd = self.behavior_search(cmd)
            
        elif self.state == self.S_APPROACH:
            cmd = self.behavior_approach(cmd)
            
        elif self.state == self.S_ALIGN:
            cmd = self.behavior_align(cmd)
            
        elif self.state == self.S_FINAL:
            cmd = self.behavior_final(cmd)
            
        elif self.state == self.S_PASS:
            cmd = self.behavior_pass(cmd)
            
        elif self.state == self.S_UTURN:
            cmd = self.behavior_uturn(cmd)
            
        elif self.state == self.S_COMPLETE:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.get_logger().info("🏆 MISSION ACCOMPLISHED", throttle_duration_sec=2)

        self.cmd_pub.publish(cmd)
        
        status = f"State:{self.state} | Pass:{self.passes_completed} | Z_Tgt:{target:.2f}"
        self.state_pub.publish(String(data=status))

    # --- BEHAVIORS ---
    
    def behavior_search(self, cmd):
        """Find the gate. Maintain Safe Depth."""
        if self.gate_detected and (self.dist < 15.0):
            self.get_logger().info("🎯 Gate Detected! Approach.")
            self.transition(self.S_APPROACH)
            return cmd
            
        # Search Pattern
        t = time.time() - self.state_start_time
        cmd.angular.z = 0.3 if (t % 10 < 5) else -0.3
        cmd.linear.x = 0.0 # Don't move forward blindly
        return cmd

    def behavior_approach(self, cmd):
        """Move to 3m mark. Hold Safe Depth."""
        if self.dist <= self.align_dist:
            self.get_logger().info(f"🛑 3m Reached ({self.dist:.1f}m). Starting Alignment.")
            self.transition(self.S_ALIGN)
            return cmd
            
        if not self.gate_detected: 
            # If lost briefly, stop and wait
            return cmd 
            
        cmd.linear.x = 0.5
        cmd.angular.z = -self.gate_pos_x * 0.8 # Simple Yaw correction
        return cmd

    def behavior_align(self, cmd):
        """Stop at 3m. Align Yaw AND Depth."""
        # 1. Check Alignment Quality
        yaw_aligned = abs(self.gate_pos_x) < 0.1
        depth_aligned = abs(self.gate_center_y - self.img_center_y) < 50 # 50px tolerance
        
        if yaw_aligned and depth_aligned:
            self.get_logger().info("✅ Aligned (XYZ). Starting Final Approach.")
            self.transition(self.S_FINAL)
            return cmd
            
        # 2. Align Actions
        cmd.linear.x = 0.0 # STOP forward motion
        cmd.angular.z = -self.gate_pos_x * 1.0 # Align Yaw
        
        # Depth is handled in main loop (Visual Servo)
        return cmd

    def behavior_final(self, cmd):
        """Slow approach 3m -> 1.2m while keeping alignment."""
        if self.dist < self.pass_dist:
            self.get_logger().info("🚀 Committing to PASS!")
            self.transition(self.S_PASS)
            return cmd
            
        cmd.linear.x = 0.4
        cmd.angular.z = -self.gate_pos_x * 1.0
        return cmd

    def behavior_pass(self, cmd):
        """Blind forward burst."""
        elapsed = time.time() - self.state_start_time
        duration = 7.0 # Seconds to clear gate
        
        if elapsed > duration:
            self.passes_completed += 1
            if self.passes_completed < 2:
                self.get_logger().info("🔄 Pass 1 Complete. Starting U-Turn.")
                self.start_yaw = self.current_yaw
                self.transition(self.S_UTURN)
            else:
                self.transition(self.S_COMPLETE)
            return cmd
            
        cmd.linear.x = 1.0 # Full speed
        cmd.angular.z = 0.0 # Lock yaw
        return cmd

    def behavior_uturn(self, cmd):
        """Turn 180 degrees."""
        # Calculate difference from start yaw
        diff = abs(self.current_yaw - self.start_yaw)
        if diff > math.pi: diff = 2*math.pi - diff
        
        if diff > (math.pi * 0.90): # 90% of turn done
            self.get_logger().info("Search for Pass 2.")
            self.transition(self.S_SEARCH)
            return cmd
            
        cmd.linear.x = 0.2 # Slight forward to help turn dynamics
        cmd.angular.z = 0.8 # Hard Left
        return cmd

    def transition(self, new_state):
        self.state = new_state
        self.state_start_time = time.time()

def main(args=None):
    rclpy.init(args=args)
    node = QualificationNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()