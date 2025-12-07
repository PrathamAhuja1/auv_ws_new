#!/usr/bin/env python3
"""
FIXED QUALIFICATION Navigator
Critical fixes:
1. Added missing start_u_turn_yaw initialization
2. Improved depth control
3. Better state transitions
4. Enhanced logging for debugging
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
        self.S_POST_PASS_FWD = 4
        self.S_U_TURN = 5
        self.S_SEARCH_REV = 6
        self.S_APPROACH_REV = 7
        self.S_ALIGN_REV = 8
        self.S_PASS_REV = 9
        self.S_COMPLETED = 10
        
        self.state = self.S_SEARCH_FWD
        
        # Parameters
        self.declare_parameter('target_depth', -1.2)  # Shallower for better gate visibility
        self.target_depth = self.get_parameter('target_depth').value
        
        self.search_speed = 0.5
        self.search_yaw = 0.25
        self.approach_stop_dist = 2.5
        self.pass_speed = 1.0
        
        # Inputs
        self.gate_detected = False
        self.frame_pos = 0.0
        self.dist = 999.0
        self.current_yaw = 0.0
        self.current_depth = 0.0
        self.current_pos = None
        self.pass_start_pos = None
        
        # CRITICAL FIX: Initialize U-turn yaw tracking
        self.start_u_turn_yaw = 0.0
        
        self.state_start_time = time.time()
        self.last_detection_time = 0.0
        
        # Gate clearance tracking
        self.gate_x_position = 0.0  # Will be updated from world
        
        # Pubs/Subs
        self.cmd_pub = self.create_publisher(Twist, '/rp2040/cmd_vel', 10)
        self.state_pub = self.create_publisher(String, '/mission/state', 10)
        self.reverse_mode_pub = self.create_publisher(Bool, '/mission/reverse_mode', 10)
        
        self.create_subscription(Bool, '/gate/detected', self.cb_detected, 10)
        self.create_subscription(Float32, '/gate/frame_position', self.cb_frame, 10)
        self.create_subscription(Float32, '/gate/estimated_distance', self.cb_dist, 10)
        self.create_subscription(Odometry, '/ground_truth/odom', self.cb_odom, 10)
        
        self.create_timer(0.05, self.control_loop)
        
        self.get_logger().info('='*70)
        self.get_logger().info('🚀 FIXED QUALIFICATION NAVIGATOR')
        self.get_logger().info('='*70)
        self.get_logger().info('   Task: Forward Pass → U-Turn → Reverse Pass')
        self.get_logger().info(f'   Target depth: {self.target_depth}m')
        self.get_logger().info('='*70)

    # --- CALLBACKS ---
    def cb_detected(self, msg): 
        self.gate_detected = msg.data
        if self.gate_detected:
            self.last_detection_time = time.time()

    def cb_frame(self, msg): 
        self.frame_pos = msg.data
    
    def cb_dist(self, msg): 
        self.dist = msg.data
    
    def cb_odom(self, msg):
        self.current_pos = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        self.current_depth = msg.pose.pose.position.z
        
        # Extract yaw from quaternion
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

    # --- CONTROL LOOP ---
    def control_loop(self):
        cmd = Twist()
        
        # Depth Control (fixed sign)
        depth_error = self.target_depth - self.current_depth
        depth_deadband = 0.2
        
        if abs(depth_error) < depth_deadband:
            cmd.linear.z = 0.0
        else:
            cmd.linear.z = depth_error * 1.2
            cmd.linear.z = max(-0.8, min(0.8, cmd.linear.z))
        
        # State Machine
        if self.state == self.S_SEARCH_FWD:
            cmd = self.behavior_search(cmd, self.S_APPROACH_FWD)
            self.reverse_mode_pub.publish(Bool(data=False))
            
        elif self.state == self.S_APPROACH_FWD:
            cmd = self.behavior_approach(cmd, self.S_ALIGN_FWD)
            self.reverse_mode_pub.publish(Bool(data=False))
            
        elif self.state == self.S_ALIGN_FWD:
            cmd = self.behavior_align(cmd, self.S_PASS_FWD)
            self.reverse_mode_pub.publish(Bool(data=False))
            
        elif self.state == self.S_PASS_FWD:
            cmd = self.behavior_pass(cmd, 6.0, self.S_POST_PASS_FWD)
            self.reverse_mode_pub.publish(Bool(data=False))
            
        elif self.state == self.S_POST_PASS_FWD:
            # Move forward blindly to clear gate area
            if time.time() - self.state_start_time > 4.0:
                self.set_state(self.S_U_TURN)
            cmd.linear.x = 0.6
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
            cmd = self.behavior_align(cmd, self.S_PASS_REV)
            self.reverse_mode_pub.publish(Bool(data=True))
            
        elif self.state == self.S_PASS_REV:
            cmd = self.behavior_pass(cmd, 6.0, self.S_COMPLETED)
            self.reverse_mode_pub.publish(Bool(data=True))
            
        elif self.state == self.S_COMPLETED:
            cmd.linear.x = 0.0
            cmd.linear.y = 0.0
            cmd.angular.z = 0.0
            self.get_logger().info("🏆 QUALIFICATION COMPLETED!", throttle_duration_sec=2.0)
            
        self.cmd_pub.publish(cmd)
        self.state_pub.publish(String(data=f"State_{self.state}"))
        
        # Log current state periodically
        if int(time.time()) % 2 == 0:
            self.get_logger().info(
                f"State: {self.get_state_name()} | "
                f"Gate: {'YES' if self.gate_detected else 'NO'} | "
                f"Dist: {self.dist:.1f}m | "
                f"Depth: {self.current_depth:.2f}m | "
                f"Pos: {self.frame_pos:+.2f}",
                throttle_duration_sec=1.9
            )

    # --- BEHAVIORS ---
    def behavior_search(self, cmd, next_state):
        """Search for gate with sweep pattern"""
        if self.gate_detected and self.dist < 15.0:
            self.get_logger().info(f'🎯 Gate found at {self.dist:.1f}m - Approaching')
            self.set_state(next_state)
            return cmd
        
        t = time.time() - self.state_start_time
        # Sweep pattern: turn left for 7s, right for 7s
        cmd.angular.z = self.search_yaw if (t % 14 < 7) else -self.search_yaw
        cmd.linear.x = self.search_speed
        
        return cmd

    def behavior_approach(self, cmd, next_state):
        """Approach gate until close enough to align"""
        if not self.gate_detected:
            lost_time = time.time() - self.last_detection_time
            if lost_time > 2.0:
                self.get_logger().warn('Gate lost - searching')
                return cmd
            # Coast briefly if just lost
        
        if self.dist < self.approach_stop_dist and self.dist > 0.1:
            self.get_logger().info(f'🛑 Close enough ({self.dist:.1f}m) - Aligning')
            self.set_state(next_state)
            return cmd
            
        cmd.linear.x = 0.6
        cmd.angular.z = -self.frame_pos * 1.2
        
        return cmd

    def behavior_align(self, cmd, next_state):
        """Align with gate center"""
        elapsed = time.time() - self.state_start_time
        
        if elapsed > 10.0:  # Timeout
            self.get_logger().warn('Alignment timeout - proceeding')
            self.set_state(next_state)
            return cmd
        
        if abs(self.frame_pos) < 0.12:
            self.get_logger().info('✓ Aligned - Starting pass')
            self.set_state(next_state)
            return cmd
            
        cmd.linear.x = 0.15
        cmd.angular.z = -self.frame_pos * 2.0
        
        return cmd

    def behavior_pass(self, cmd, duration, next_state):
        """Pass through gate at full speed"""
        if self.pass_start_pos is None: 
            self.pass_start_pos = self.current_pos
            self.get_logger().info('🚀 PASSING THROUGH GATE')
        
        dist_traveled = 0.0
        if self.current_pos and self.pass_start_pos:
            dx = self.current_pos[0] - self.pass_start_pos[0]
            dy = self.current_pos[1] - self.pass_start_pos[1]
            dist_traveled = math.sqrt(dx*dx + dy*dy)
            
            # Log progress
            self.get_logger().info(
                f'Passing: {dist_traveled:.1f}m traveled',
                throttle_duration_sec=0.5
            )
            
        if dist_traveled > 3.5 or (time.time() - self.state_start_time > duration):
            self.get_logger().info(f'✅ Pass complete ({dist_traveled:.1f}m)')
            self.set_state(next_state)
            self.pass_start_pos = None
            return cmd
            
        cmd.linear.x = self.pass_speed
        cmd.angular.z = 0.0  # Straight through
        
        return cmd

    def behavior_u_turn(self, cmd):
        """Execute 180-degree U-turn"""
        # CRITICAL FIX: Initialize on first entry
        if self.start_u_turn_yaw == 0.0:
            self.start_u_turn_yaw = self.current_yaw
            self.get_logger().info(f'Starting U-turn from yaw={math.degrees(self.current_yaw):.1f}°')
        
        # Calculate how much we've turned
        diff = abs(self.current_yaw - self.start_u_turn_yaw)
        if diff > math.pi:
            diff = 2*math.pi - diff
        
        # Check if we've turned ~180 degrees
        if diff > (math.pi * 0.90):  # 162 degrees (allow some tolerance)
            self.get_logger().info(f'✓ U-turn complete (turned {math.degrees(diff):.1f}°)')
            self.set_state(self.S_SEARCH_REV)
            self.start_u_turn_yaw = 0.0  # Reset for next time
            return cmd
        
        # Continue turning
        cmd.linear.x = 0.3  # Move forward while turning
        cmd.angular.z = 0.7  # Turn rate
        
        # Log progress
        self.get_logger().info(
            f'U-turning: {math.degrees(diff):.1f}° / 180°',
            throttle_duration_sec=0.5
        )
        
        return cmd

    def set_state(self, new_state):
        """Transition to new state"""
        old_name = self.get_state_name()
        self.state = new_state
        self.state_start_time = time.time()
        new_name = self.get_state_name()
        
        self.get_logger().info(f'🔄 STATE: {old_name} → {new_name}')
    
    def get_state_name(self):
        """Get human-readable state name"""
        names = {
            self.S_SEARCH_FWD: 'SEARCH_FWD',
            self.S_APPROACH_FWD: 'APPROACH_FWD',
            self.S_ALIGN_FWD: 'ALIGN_FWD',
            self.S_PASS_FWD: 'PASS_FWD',
            self.S_POST_PASS_FWD: 'POST_PASS_FWD',
            self.S_U_TURN: 'U_TURN',
            self.S_SEARCH_REV: 'SEARCH_REV',
            self.S_APPROACH_REV: 'APPROACH_REV',
            self.S_ALIGN_REV: 'ALIGN_REV',
            self.S_PASS_REV: 'PASS_REV',
            self.S_COMPLETED: 'COMPLETED'
        }
        return names.get(self.state, 'UNKNOWN')


def main(args=None):
    rclpy.init(args=args)
    node = QualificationNavigator()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop robot
        stop_cmd = Twist()
        node.cmd_pub.publish(stop_cmd)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()