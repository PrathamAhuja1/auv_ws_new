#!/usr/bin/env python3
"""
FIXED Qualification Navigator - Proper depth control and state machine
"""
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
        self.SUBMERGING = 0
        self.SEARCHING = 1
        self.ALIGNING = 2
        self.PASSING = 3
        self.SURFACING = 4
        self.FINISHED = 5
        
        self.state = self.SUBMERGING
        self.state_names = ["SUBMERGING", "SEARCHING", "ALIGNING", "PASSING", "SURFACING", "FINISHED"]

        # Parameters
        self.declare_parameter('target_depth', -1.0)
        self.declare_parameter('search_speed', 0.3)
        self.declare_parameter('approach_speed', 0.5)
        self.declare_parameter('passing_speed', 1.0)
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
        self.current_position = None
        self.start_time = time.time()
        self.state_start_time = time.time()
        self.passing_start_x = None
        
        # Subscriptions
        self.create_subscription(Bool, '/gate/detected', self.gate_cb, 10)
        self.create_subscription(Float32, '/gate/frame_position', self.pos_cb, 10)
        self.create_subscription(Float32, '/gate/estimated_distance', self.dist_cb, 10)
        self.create_subscription(Odometry, '/ground_truth/odom', self.odom_cb, 10)
        
        # Publishers
        self.vel_pub = self.create_publisher(Twist, '/rp2040/cmd_vel', 10)
        self.state_pub = self.create_publisher(String, '/qualification/state', 10)
        
        self.timer = self.create_timer(0.05, self.control_loop)
        self.get_logger().info('✅ FIXED Qualification Navigator Started')
        self.get_logger().info(f'   Target depth: {self.target_depth}m')

    def gate_cb(self, msg): 
        self.gate_detected = msg.data
    
    def pos_cb(self, msg): 
        self.frame_position = msg.data
    
    def dist_cb(self, msg): 
        self.distance = msg.data
    
    def odom_cb(self, msg): 
        self.current_depth = msg.pose.pose.position.z
        self.current_position = msg.pose.pose.position

    def control_loop(self):
        cmd = Twist()
        
        # === DEPTH CONTROL (Active in all states except SURFACING/FINISHED) ===
        if self.state not in [self.SURFACING, self.FINISHED]:
            depth_err = self.target_depth - self.current_depth
            
            # Dead band to prevent oscillation
            if abs(depth_err) < 0.15:
                cmd.linear.z = 0.0
            else:
                # Proportional control
                cmd.linear.z = depth_err * 1.2
                # Clamp
                cmd.linear.z = max(-0.8, min(cmd.linear.z, 0.8))
            
            # Log depth periodically
            if int(time.time()) % 3 == 0:
                self.get_logger().info(
                    f'Depth: current={self.current_depth:.2f}m, '
                    f'target={self.target_depth:.2f}m, '
                    f'cmd_z={cmd.linear.z:.2f}',
                    throttle_duration_sec=2.9
                )
        
        # === STATE MACHINE ===
        if self.state == self.SUBMERGING:
            # Wait until we reach target depth
            if abs(self.current_depth - self.target_depth) < 0.2:
                self.get_logger().info("✅ Depth reached. Starting search...")
                self.transition(self.SEARCHING)
            else:
                # Just maintain position while submerging
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0

        elif self.state == self.SEARCHING:
            # Sweep search pattern
            elapsed = time.time() - self.state_start_time
            sweep_period = 8.0
            
            cmd.linear.x = self.search_speed
            
            # Sweep: 4s left, 4s right
            if (elapsed % sweep_period) < 4.0:
                cmd.angular.z = 0.2
            else:
                cmd.angular.z = -0.2

            if self.gate_detected:
                self.get_logger().info("👀 Gate detected! Aligning...")
                self.transition(self.ALIGNING)

        elif self.state == self.ALIGNING:
            if not self.gate_detected:
                # Lost gate - go back to search
                self.get_logger().warn("Lost gate, resuming search")
                self.transition(self.SEARCHING)
                return

            # STOP AND TURN logic
            alignment_quality = abs(self.frame_position)
            
            if alignment_quality > 0.15:
                # Not aligned - STOP and TURN ONLY
                cmd.linear.x = 0.0
                cmd.angular.z = -self.frame_position * self.yaw_gain
                
                self.get_logger().info(
                    f'🔄 Aligning: pos={self.frame_position:+.3f}, '
                    f'turning={cmd.angular.z:+.2f}',
                    throttle_duration_sec=0.5
                )
            else:
                # Well aligned - move forward
                cmd.linear.x = self.approach_speed
                cmd.angular.z = -self.frame_position * self.yaw_gain * 0.5
                
                self.get_logger().info(
                    f'➡️ Approaching: dist={self.distance:.1f}m, '
                    f'pos={self.frame_position:+.3f}',
                    throttle_duration_sec=0.5
                )

            # Check if close enough to pass
            if self.distance < 1.5 and self.distance > 0.1:
                self.get_logger().info(f"🚀 Passing at {self.distance:.1f}m")
                self.passing_start_x = self.current_position.x if self.current_position else None
                self.transition(self.PASSING)

        elif self.state == self.PASSING:
            # Full speed through gate
            cmd.linear.x = self.passing_speed
            cmd.angular.z = 0.0
            
            # Check if we've cleared the gate (2m past gate at X=0)
            if self.current_position and self.current_position.x > 2.0:
                self.get_logger().info("✅ Gate cleared! Surfacing...")
                self.transition(self.SURFACING)
            
            # Timeout safety
            if time.time() - self.state_start_time > 10.0:
                self.get_logger().info("Pass timeout, surfacing")
                self.transition(self.SURFACING)

        elif self.state == self.SURFACING:
            # Stop horizontal, go up
            cmd.linear.x = 0.0
            cmd.linear.z = 1.0  # Max up
            cmd.angular.z = 0.0
            
            if self.current_depth > -0.2:
                self.transition(self.FINISHED)
                
        elif self.state == self.FINISHED:
            cmd = Twist()  # All stop

        # Publish
        self.vel_pub.publish(cmd)
        self.state_pub.publish(String(data=self.state_names[self.state]))

    def transition(self, new_state):
        old_name = self.state_names[self.state]
        self.state = new_state
        self.state_start_time = time.time()
        new_name = self.state_names[new_state]
        self.get_logger().info(f'🔄 {old_name} → {new_name}')

def main():
    rclpy.init()
    node = QualificationNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop on exit
        cmd = Twist()
        node.vel_pub.publish(cmd)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()