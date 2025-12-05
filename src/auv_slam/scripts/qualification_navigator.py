#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import time

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

        # Parameters
        self.declare_parameter('target_depth', -1.3)
        self.declare_parameter('search_speed', 0.3)
        self.declare_parameter('approach_speed', 0.4)
        self.declare_parameter('passing_speed', 0.8) 
        self.declare_parameter('alignment_yaw_gain', 2.0)
        
        self.target_depth = self.get_parameter('target_depth').value
        self.search_speed = self.get_parameter('search_speed').value
        self.approach_speed = self.get_parameter('approach_speed').value
        self.passing_speed = self.get_parameter('passing_speed').value
        self.yaw_gain = self.get_parameter('alignment_yaw_gain').value
        
        # Variables
        self.gate_detected = False
        self.frame_position = 0.0 
        self.distance = 999.0
        self.current_depth = 0.0
        self.start_time = time.time()
        self.pass_start_time = 0.0
        
        # Subscriptions
        self.create_subscription(Bool, '/gate/detected', self.gate_cb, 10)
        self.create_subscription(Float32, '/gate/frame_position', self.pos_cb, 10)
        self.create_subscription(Float32, '/gate/estimated_distance', self.dist_cb, 10)
        self.create_subscription(Odometry, '/ground_truth/odom', self.odom_cb, 10)
        
        # Publishers
        self.vel_pub = self.create_publisher(Twist, '/rp2040/cmd_vel', 10)
        self.state_pub = self.create_publisher(String, '/qualification/state', 10)
        
        self.timer = self.create_timer(0.05, self.control_loop)
        self.get_logger().info('✅ SAUVC Qualification Node Started')

    def gate_cb(self, msg): self.gate_detected = msg.data
    def pos_cb(self, msg): self.frame_position = msg.data
    def dist_cb(self, msg): self.distance = msg.data
    def odom_cb(self, msg): self.current_depth = msg.pose.pose.position.z

    def control_loop(self):
        cmd = Twist()
        
        # Depth Control (Always Active unless Surfacing)
        if self.state != self.SURFACING and self.state != self.FINISHED:
            err = self.target_depth - self.current_depth
            cmd.linear.z = max(-1.0, min(err * 1.5, 1.0))
        
        # State Logic
        if self.state == self.IDLE:
            if time.time() - self.start_time > 5.0:
                self.transition(self.SUBMERGING)
                
        elif self.state == self.SUBMERGING:
            if abs(self.current_depth - self.target_depth) < 0.2:
                self.transition(self.SEARCHING)

        elif self.state == self.SEARCHING:
            cmd.linear.x = self.search_speed
            if self.gate_detected:
                self.transition(self.ALIGNING)

        elif self.state == self.ALIGNING:
            cmd.angular.z = -self.frame_position * self.yaw_gain
            cmd.linear.x = self.approach_speed
            
            if not self.gate_detected:
                cmd.angular.z = 0.0 # Don't spin if lost

            # Trigger passing when close (1.5m)
            if self.distance < 1.5 and self.distance > 0.1: 
                if abs(self.frame_position) < 0.3:
                    self.transition(self.PASSING)
                else:
                    cmd.linear.x = 0.1 # Slow down to align

        elif self.state == self.PASSING:
            cmd.linear.x = self.passing_speed
            cmd.angular.z = 0.0
            
            # Pass for 10 seconds to ensure full clearance
            if time.time() - self.pass_start_time > 10.0:
                self.transition(self.SURFACING)

        elif self.state == self.SURFACING:
            cmd.linear.x = 0.0
            cmd.linear.z = 1.0 # Surface
            if self.current_depth > -0.2:
                self.transition(self.FINISHED)
                
        elif self.state == self.FINISHED:
            cmd = Twist()

        self.vel_pub.publish(cmd)
        self.state_pub.publish(String(data=str(self.state)))

    def transition(self, new_state):
        self.state = new_state
        if new_state == self.PASSING:
            self.pass_start_time = time.time()
        self.get_logger().info(f'Transitioned to State: {new_state}')

def main():
    rclpy.init()
    node = QualificationNavigator()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()