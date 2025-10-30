#!/usr/bin/env python3
"""
FIXED Gate Mission Controller with Smart Search Strategy
Solves the searching loop issue with intelligent scanning
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Float32
import time
import math


class SmartGateMission(Node):
    """Fixed mission controller with intelligent search"""
    
    def __init__(self):
        super().__init__('smart_gate_mission')
        
        # Mission states
        self.INIT = 0
        self.SUBMERGING = 1
        self.SEARCHING_GATE = 2
        self.APPROACHING_GATE = 3
        self.PASSING_THROUGH = 4
        self.COMPLETED = 5
        
        self.state = self.INIT
        
        # Parameters
        self.declare_parameter('target_depth', -1.5)
        self.declare_parameter('depth_tolerance', 0.3)
        self.declare_parameter('search_forward_speed', 0.5)  # Move forward while searching
        self.declare_parameter('search_rotation_speed', 0.15)  # Slow rotation
        self.declare_parameter('approach_speed', 0.7)
        self.declare_parameter('passing_speed', 1.0)
        self.declare_parameter('passing_duration', 8.0)
        self.declare_parameter('alignment_threshold', 0.15)
        self.declare_parameter('approach_distance', 3.0)
        self.declare_parameter('passing_distance', 1.5)
        self.declare_parameter('yaw_gain', 1.2)  # Higher gain for responsive turning
        self.declare_parameter('depth_gain', 1.5)
        
        self.target_depth = self.get_parameter('target_depth').value
        self.depth_tolerance = self.get_parameter('depth_tolerance').value
        self.search_forward_speed = self.get_parameter('search_forward_speed').value
        self.search_rotation_speed = self.get_parameter('search_rotation_speed').value
        self.approach_speed = self.get_parameter('approach_speed').value
        self.passing_speed = self.get_parameter('passing_speed').value
        self.passing_duration = self.get_parameter('passing_duration').value
        self.alignment_threshold = self.get_parameter('alignment_threshold').value
        self.approach_distance = self.get_parameter('approach_distance').value
        self.passing_distance = self.get_parameter('passing_distance').value
        self.yaw_gain = self.get_parameter('yaw_gain').value
        self.depth_gain = self.get_parameter('depth_gain').value
        
        # State variables
        self.current_depth = 0.0
        self.current_position = None
        self.gate_detected = False
        self.gate_alignment_error = 0.0
        self.gate_distance = 999.0
        self.state_start_time = time.time()
        self.passing_start_time = None
        
        # Search strategy variables
        self.search_start_position = None
        self.search_phase = 0  # 0: forward, 1: left sweep, 2: right sweep, 3: advance
        self.search_phase_start = None
        self.total_rotation = 0.0
        self.last_yaw = None
        
        # Subscriptions
        self.odom_sub = self.create_subscription(
            Odometry, '/ground_truth/odom', self.odom_callback, 10)
        
        self.gate_detected_sub = self.create_subscription(
            Bool, '/gate/detected', self.gate_detected_callback, 10)
        
        self.gate_alignment_sub = self.create_subscription(
            Float32, '/gate/alignment_error', self.gate_alignment_callback, 10)
        
        self.gate_distance_sub = self.create_subscription(
            Float32, '/gate/estimated_distance', self.gate_distance_callback, 10)
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/rp2040/cmd_vel', 10)
        
        # Control loop (20 Hz)
        self.timer = self.create_timer(0.05, self.control_loop)
        
        self.get_logger().info('='*70)
        self.get_logger().info('🚀 SMART Gate Mission Controller Started!')
        self.get_logger().info('='*70)
        self.get_logger().info(f'📍 Target Depth: {self.target_depth}m')
        self.get_logger().info(f'🔍 Search Speed: {self.search_forward_speed} m/s + rotation')
        self.get_logger().info(f'⚡ Passing Speed: {self.passing_speed} m/s')
        self.get_logger().info('='*70)
    
    def odom_callback(self, msg: Odometry):
        """Update current pose"""
        self.current_depth = msg.pose.pose.position.z
        self.current_position = msg.pose.pose.position
        
        # Extract yaw for search tracking
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.last_yaw = math.atan2(siny_cosp, cosy_cosp)
    
    def gate_detected_callback(self, msg: Bool):
        was_detected = self.gate_detected
        self.gate_detected = msg.data
        
        if not was_detected and self.gate_detected:
            self.get_logger().info('✅ GATE DETECTED! Switching to approach mode')
    
    def gate_alignment_callback(self, msg: Float32):
        self.gate_alignment_error = msg.data
    
    def gate_distance_callback(self, msg: Float32):
        self.gate_distance = msg.data
    
    def control_loop(self):
        """Main control loop"""
        
        if self.state == self.INIT:
            self.handle_init()
        elif self.state == self.SUBMERGING:
            self.handle_submerging()
        elif self.state == self.SEARCHING_GATE:
            self.handle_searching()
        elif self.state == self.APPROACHING_GATE:
            self.handle_approaching()
        elif self.state == self.PASSING_THROUGH:
            self.handle_passing()
        elif self.state == self.COMPLETED:
            self.handle_completed()
    
    def handle_init(self):
        """Initialize mission"""
        self.get_logger().info('📍 State: INIT → Starting mission...')
        time.sleep(1.0)
        self.transition_to(self.SUBMERGING)
    
    def handle_submerging(self):
        """Submerge to target depth while moving forward"""
        depth_error = self.target_depth - self.current_depth
        elapsed = time.time() - self.state_start_time

        cmd = Twist()
        cmd.linear.x = 0.3  # Move forward while submerging
        cmd.linear.z = depth_error * self.depth_gain
        
        self.cmd_vel_pub.publish(cmd)

        if int(elapsed) % 2 == 0:
            self.get_logger().info(
                f'⬇️  Submerging: Depth={self.current_depth:.2f}m, '
                f'Target={self.target_depth:.2f}m',
                throttle_duration_sec=2.0
            )

        if abs(depth_error) < self.depth_tolerance:
            self.get_logger().info(f'✅ Target depth reached: {self.current_depth:.2f}m')
            self.transition_to(self.SEARCHING_GATE)
            return

        if elapsed > 15.0:
            self.get_logger().warn(f'⚠️  Depth timeout! Moving to search...')
            self.transition_to(self.SEARCHING_GATE)
    
    def handle_searching(self):
        """SMART SEARCH: Move forward + scan left/right"""
        elapsed = time.time() - self.state_start_time

        # Initialize search position
        if self.search_start_position is None:
            self.search_start_position = self.current_position
            self.search_phase_start = time.time()

        # Check if gate found
        if self.gate_detected:
            self.get_logger().info('✅ Gate found during search!')
            self.transition_to(self.APPROACHING_GATE)
            return

        cmd = Twist()
        
        # ALWAYS maintain depth
        depth_error = self.target_depth - self.current_depth
        cmd.linear.z = depth_error * self.depth_gain
        
        # SMART SEARCH PATTERN: Forward + Sweeping
        phase_duration = time.time() - self.search_phase_start
        
        if self.search_phase == 0:  # Initial forward push
            cmd.linear.x = self.search_forward_speed
            cmd.angular.z = 0.0
            
            if phase_duration > 3.0:  # Search straight for 3 seconds
                self.search_phase = 1
                self.search_phase_start = time.time()
                self.total_rotation = 0.0
                self.get_logger().info('🔍 Search: Starting LEFT sweep')
        
        elif self.search_phase == 1:  # Sweep LEFT while moving forward
            cmd.linear.x = self.search_forward_speed * 0.7  # Slower while turning
            cmd.angular.z = self.search_rotation_speed  # Turn left
            
            # Track rotation
            if phase_duration > 0.05:  # Update every cycle
                self.total_rotation += self.search_rotation_speed * 0.05
            
            if self.total_rotation > math.radians(60):  # 60 degree sweep
                self.search_phase = 2
                self.search_phase_start = time.time()
                self.total_rotation = 0.0
                self.get_logger().info('🔍 Search: Starting RIGHT sweep')
        
        elif self.search_phase == 2:  # Sweep RIGHT while moving forward
            cmd.linear.x = self.search_forward_speed * 0.7
            cmd.angular.z = -self.search_rotation_speed  # Turn right
            
            if phase_duration > 0.05:
                self.total_rotation += self.search_rotation_speed * 0.05
            
            if self.total_rotation > math.radians(120):  # 120 degree sweep (back to center + 60)
                self.search_phase = 3
                self.search_phase_start = time.time()
                self.get_logger().info('🔍 Search: Advancing forward')
        
        elif self.search_phase == 3:  # Advance forward, then repeat
            cmd.linear.x = self.search_forward_speed
            cmd.angular.z = -self.search_rotation_speed * 0.5  # Center back
            
            if phase_duration > 2.0:  # Advance for 2 seconds
                self.search_phase = 0  # Reset to forward phase
                self.search_phase_start = time.time()
                self.get_logger().info('🔍 Search: Repeating pattern')
        
        # Publish command
        self.cmd_vel_pub.publish(cmd)

        # Log search status
        if int(elapsed) % 2 == 0:
            distance_moved = 0.0
            if self.search_start_position:
                dx = self.current_position.x - self.search_start_position.x
                dy = self.current_position.y - self.search_start_position.y
                distance_moved = math.sqrt(dx**2 + dy**2)
            
            self.get_logger().info(
                f'🔍 Searching: Phase={self.search_phase}, '
                f'Time={elapsed:.1f}s, Moved={distance_moved:.1f}m',
                throttle_duration_sec=2.0
            )
    
    def handle_approaching(self):
        """Approach gate with continuous alignment correction"""
        elapsed = time.time() - self.state_start_time

        # Check if gate lost
        if not self.gate_detected:
            if elapsed > 2.0:
                self.get_logger().warn('⚠️  Gate lost! Returning to search...')
                self.transition_to(self.SEARCHING_GATE)
            return

        # Check if close enough to pass
        if self.gate_distance < self.passing_distance and self.gate_distance > 0.1:
            self.get_logger().info(f'✅ Ready to pass! Distance: {self.gate_distance:.2f}m')
            self.transition_to(self.PASSING_THROUGH)
            return

        cmd = Twist()
        
        # CONTINUOUS DRIFT APPROACH
        cmd.linear.x = self.approach_speed
        cmd.angular.z = -self.gate_alignment_error * self.yaw_gain
        
        # Maintain depth
        depth_error = self.target_depth - self.current_depth
        cmd.linear.z = depth_error * self.depth_gain
        
        self.cmd_vel_pub.publish(cmd)

        if int(elapsed * 2) % 2 == 0:
            self.get_logger().info(
                f'➡️  Approaching: Dist={self.gate_distance:.2f}m, '
                f'Align={self.gate_alignment_error:.3f}, '
                f'Speed={self.approach_speed:.2f}',
                throttle_duration_sec=0.5
            )
    
    def handle_passing(self):
        """Pass through gate at maximum speed"""

        if self.passing_start_time is None:
            self.passing_start_time = time.time()
            self.get_logger().info('🚀 PASSING THROUGH GATE - FULL SPEED!')

        elapsed = time.time() - self.passing_start_time

        if elapsed > self.passing_duration:
            self.get_logger().info(f'✅ GATE PASSED! Duration: {elapsed:.1f}s')
            self.transition_to(self.COMPLETED)
            return

        cmd = Twist()
        
        # MAXIMUM FORWARD SPEED
        cmd.linear.x = self.passing_speed
        cmd.linear.y = 0.0
        cmd.angular.z = 0.0
        
        # Maintain depth
        depth_error = self.target_depth - self.current_depth
        cmd.linear.z = depth_error * self.depth_gain
        
        self.cmd_vel_pub.publish(cmd)

        if int(elapsed) % 1 == 0:
            self.get_logger().info(
                f'🚀 Passing: {elapsed:.1f}s / {self.passing_duration:.1f}s',
                throttle_duration_sec=1.0
            )
    
    def handle_completed(self):
        """Mission complete"""
        
        elapsed = time.time() - self.state_start_time
        if elapsed < 0.5:
            self.get_logger().info('='*70)
            self.get_logger().info('✅ GATE MISSION COMPLETED!')
            self.get_logger().info('='*70)
        
        cmd = Twist()
        self.cmd_vel_pub.publish(cmd)
    
    def transition_to(self, new_state: int):
        """Transition to new state"""
        state_names = {
            self.INIT: 'INIT',
            self.SUBMERGING: 'SUBMERGING',
            self.SEARCHING_GATE: 'SEARCHING_GATE',
            self.APPROACHING_GATE: 'APPROACHING_GATE',
            self.PASSING_THROUGH: 'PASSING_THROUGH',
            self.COMPLETED: 'COMPLETED'
        }
        
        old_name = state_names.get(self.state, 'UNKNOWN')
        new_name = state_names.get(new_state, 'UNKNOWN')
        
        self.get_logger().info(f'🔄 STATE CHANGE: {old_name} → {new_name}')
        
        self.state = new_state
        self.state_start_time = time.time()
        
        # Reset search variables when leaving search
        if self.state != self.SEARCHING_GATE:
            self.search_start_position = None
            self.search_phase = 0
            self.search_phase_start = None
            self.total_rotation = 0.0
        
        # Reset passing timer
        if self.state != self.PASSING_THROUGH:
            self.passing_start_time = None


def main(args=None):
    rclpy.init(args=args)
    node = SmartGateMission()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down mission controller...')
    finally:
        cmd = Twist()
        node.cmd_vel_pub.publish(cmd)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()