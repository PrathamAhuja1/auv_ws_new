#!/usr/bin/env python3
"""
SAUVC Qualification Navigator - Stop & Turn Logic
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
        
        # State Machine
        self.IDLE = 0
        self.SUBMERGING = 1   
        self.SEARCHING = 2
        self.ALIGNING = 3
        self.PASSING = 4
        self.SURFACING = 5    
        self.FINISHED = 6
        
        self.state = self.IDLE
        self.state_names = ["IDLE", "SUBMERGING", "SEARCHING", "ALIGNING", "PASSING", "SURFACING", "FINISHED"]

        # Parameters (Must match yaml)
        self.declare_parameter('target_depth', -1.3)
        self.declare_parameter('search_speed', 0.2)
        self.declare_parameter('search_yaw_speed', 0.15)
        self.declare_parameter('approach_speed', 0.4)
        self.declare_parameter('passing_speed', 1.0)
        self.declare_parameter('alignment_yaw_gain', 1.5)
        
        self.target_depth = self.get_parameter('target_depth').value
        self.search_speed = self.get_parameter('search_speed').value
        self.search_yaw_speed = self.get_parameter('search_yaw_speed').value
        self.approach_speed = self.get_parameter('approach_speed').value
        self.passing_speed = self.get_parameter('passing_speed').value
        self.yaw_gain = self.get_parameter('alignment_yaw_gain').value
        
        # Variables
        self.gate_detected = False
        self.frame_position = 0.0 
        self.distance = 999.0
        self.current_depth = 0.0
        self.start_time = time.time()
        self.state_start_time = time.time()
        
        # Stability counters
        self.lost_gate_count = 0
        
        # Subscriptions
        self.create_subscription(Bool, '/gate/detected', self.gate_cb, 10)
        self.create_subscription(Float32, '/gate/frame_position', self.pos_cb, 10)
        self.create_subscription(Float32, '/gate/estimated_distance', self.dist_cb, 10)
        self.create_subscription(Odometry, '/ground_truth/odom', self.odom_cb, 10)
        
        # Publishers
        self.vel_pub = self.create_publisher(Twist, '/rp2040/cmd_vel', 10)
        self.state_pub = self.create_publisher(String, '/qualification/state', 10)
        
        self.timer = self.create_timer(0.05, self.control_loop)
        self.get_logger().info('✅ SAUVC Qualification Navigator Started (Stop & Turn Mode)')

    def gate_cb(self, msg): self.gate_detected = msg.data
    def pos_cb(self, msg): self.frame_position = msg.data
    def dist_cb(self, msg): self.distance = msg.data
    def odom_cb(self, msg): self.current_depth = msg.pose.pose.position.z

    def control_loop(self):
        cmd = Twist()
        
        # 1. Universal Depth Control (Always active unless surfacing)
        if self.state != self.SURFACING and self.state != self.FINISHED:
            depth_err = self.target_depth - self.current_depth
            # Simple P-control for depth
            cmd.linear.z = max(-0.8, min(depth_err * 1.5, 0.8))
        
        # 2. State Machine
        if self.state == self.IDLE:
            if time.time() - self.start_time > 2.0:
                self.get_logger().info("🏁 Starting qualification task - Beginning submersion")
                self.transition(self.SUBMERGING)
                
        elif self.state == self.SUBMERGING:
            # Wait until we reach depth
            if abs(self.current_depth - self.target_depth) < 0.15:
                self.get_logger().info("✅ Depth reached. Searching for gate...")
                self.transition(self.SEARCHING)

        elif self.state == self.SEARCHING:
            # Move forward slowly and sweep yaw
            cmd.linear.x = self.search_speed
            
            # Sweep pattern: 6s Left, 6s Right
            sweep_time = time.time() - self.state_start_time
            if (sweep_time % 12.0) < 6.0:
                cmd.angular.z = self.search_yaw_speed
            else:
                cmd.angular.z = -self.search_yaw_speed

            if self.gate_detected:
                self.get_logger().info("👀 Gate detected! Switching to Alignment.")
                self.transition(self.ALIGNING)

        elif self.state == self.ALIGNING:
            # --- ROBUST ALIGNMENT LOGIC ---
            
            # Check for lost signal
            if not self.gate_detected:
                self.lost_gate_count += 1
                if self.lost_gate_count > 20: # 1 second at 20Hz
                    self.get_logger().warn("❌ Lost gate track! Resuming Search.")
                    self.transition(self.SEARCHING)
                
                # Stop turning if lost to avoid spinning wildly
                cmd.angular.z = 0.0
                return
            else:
                self.lost_gate_count = 0 

            # CRITICAL: Stop-and-Turn Logic
            # Frame position is -1.0 (Left) to +1.0 (Right)
            # If gate is not centered (abs > 0.15), STOP moving forward and JUST TURN
            if abs(self.frame_position) > 0.15:
                cmd.linear.x = 0.0
                cmd.angular.z = -self.frame_position * self.yaw_gain
            else:
                # Gate is centered, move forward while correcting
                cmd.linear.x = self.approach_speed
                cmd.angular.z = -self.frame_position * self.yaw_gain

            # Trigger passing
            # If we are close enough, assume alignment is good and GO
            if self.distance < 2.0 and self.distance > 0.1: 
                self.get_logger().info(f"Gate close ({self.distance:.1f}m). Charging!")
                self.transition(self.PASSING)

        elif self.state == self.PASSING:
            # Blind charge forward
            cmd.linear.x = self.passing_speed
            cmd.angular.z = 0.0
            
            if time.time() - self.state_start_time > 8.0:
                self.get_logger().info("Pass complete. Surfacing.")
                self.transition(self.SURFACING)

        elif self.state == self.SURFACING:
            cmd.linear.x = 0.0
            cmd.linear.z = 1.0  # Go up
            if self.current_depth > -0.2:
                self.transition(self.FINISHED)
                
        elif self.state == self.FINISHED:
            cmd = Twist() # Stop

        self.vel_pub.publish(cmd)
        self.state_pub.publish(String(data=self.state_names[self.state]))

    def transition(self, new_state):
        self.state = new_state
        self.state_start_time = time.time()
        self.get_logger().info(f'Transitioned to State: {self.state_names[new_state]}')

def main():
    rclpy.init()
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