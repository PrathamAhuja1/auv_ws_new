#!/usr/bin/env python3
"""
SAUVC Qualification Task Navigator
Handles the complete qualification sequence:
1. SUBMERGING: Start at surface, submerge to target depth in starting zone
2. SEARCHING: Look for orange qualification gate
3. ALIGNING: Center on gate
4. APPROACHING: Move toward gate while maintaining alignment
5. PASSING: Pass through gate
6. SURFACING: Return to surface after passing
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
        self.SUBMERGING = 1      # Submerge in starting zone
        self.SEARCHING = 2       # Search for gate
        self.ALIGNING = 3        # Align with gate
        self.APPROACHING = 4     # Approach gate
        self.PASSING = 5         # Pass through gate
        self.SURFACING = 6       # Return to surface
        self.FINISHED = 7
        
        self.state = self.IDLE
        
        # Parameters
        self.declare_parameter('target_depth', -1.0)
        self.declare_parameter('starting_zone_x', -10.0)
        self.declare_parameter('starting_zone_size', 1.4)
        self.declare_parameter('gate_x_position', 0.0)
        
        self.declare_parameter('search_speed', 0.3)
        self.declare_parameter('search_yaw_speed', 0.2)
        self.declare_parameter('approach_speed', 0.4)
        self.declare_parameter('passing_speed', 0.8)
        self.declare_parameter('alignment_threshold', 0.15)
        self.declare_parameter('alignment_yaw_gain', 2.5)
        self.declare_parameter('passing_distance', 1.5)
        
        # Load parameters
        self.target_depth = self.get_parameter('target_depth').value
        self.starting_zone_x = self.get_parameter('starting_zone_x').value
        self.starting_zone_size = self.get_parameter('starting_zone_size').value
        self.gate_x_position = self.get_parameter('gate_x_position').value
        
        self.search_speed = self.get_parameter('search_speed').value
        self.search_yaw_speed = self.get_parameter('search_yaw_speed').value
        self.approach_speed = self.get_parameter('approach_speed').value
        self.passing_speed = self.get_parameter('passing_speed').value
        self.alignment_threshold = self.get_parameter('alignment_threshold').value
        self.alignment_yaw_gain = self.get_parameter('alignment_yaw_gain').value
        self.passing_distance = self.get_parameter('passing_distance').value
        
        # State variables
        self.gate_detected = False
        self.frame_position = 0.0
        self.estimated_distance = 999.0
        self.current_depth = 0.0
        self.current_position = None
        self.start_position = None
        
        # Timing
        self.mission_start_time = time.time()
        self.state_start_time = time.time()
        
        # Subscriptions
        self.create_subscription(Bool, '/gate/detected', self.gate_cb, 10)
        self.create_subscription(Float32, '/gate/frame_position', self.pos_cb, 10)
        self.create_subscription(Float32, '/gate/estimated_distance', self.dist_cb, 10)
        self.create_subscription(Odometry, '/ground_truth/odom', self.odom_cb, 10)
        
        # Publishers
        self.vel_pub = self.create_publisher(Twist, '/rp2040/cmd_vel', 10)
        self.state_pub = self.create_publisher(String, '/qualification/state', 10)
        
        # Control loop
        self.timer = self.create_timer(0.05, self.control_loop)
        
        self.get_logger().info('='*70)
        self.get_logger().info('🏊 SAUVC QUALIFICATION NAVIGATOR STARTED')
        self.get_logger().info('='*70)
        self.get_logger().info(f'   Starting zone: X={self.starting_zone_x:.1f}m')
        self.get_logger().info(f'   Gate position: X={self.gate_x_position:.1f}m')
        self.get_logger().info(f'   Target depth: {self.target_depth:.1f}m')
        self.get_logger().info('='*70)
    
    def gate_cb(self, msg): 
        self.gate_detected = msg.data
    
    def pos_cb(self, msg): 
        self.frame_position = msg.data
    
    def dist_cb(self, msg): 
        self.estimated_distance = msg.data
    
    def odom_cb(self, msg):
        self.current_depth = msg.pose.pose.position.z
        self.current_position = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        )
        
        if self.start_position is None:
            self.start_position = self.current_position
    
    def control_loop(self):
        """Main control loop - state machine"""
        cmd = Twist()
        
        # Depth control (active in most states)
        if self.state not in [self.SURFACING, self.FINISHED]:
            depth_error = self.target_depth - self.current_depth
            depth_deadband = 0.15
            
            if abs(depth_error) < depth_deadband:
                cmd.linear.z = 0.0
            else:
                cmd.linear.z = depth_error * 2.0
                cmd.linear.z = max(-1.0, min(cmd.linear.z, 1.0))
        
        # State machine logic
        if self.state == self.IDLE:
            cmd = self.idle_behavior(cmd)
        elif self.state == self.SUBMERGING:
            cmd = self.submerging_behavior(cmd)
        elif self.state == self.SEARCHING:
            cmd = self.searching_behavior(cmd)
        elif self.state == self.ALIGNING:
            cmd = self.aligning_behavior(cmd)
        elif self.state == self.APPROACHING:
            cmd = self.approaching_behavior(cmd)
        elif self.state == self.PASSING:
            cmd = self.passing_behavior(cmd)
        elif self.state == self.SURFACING:
            cmd = self.surfacing_behavior(cmd)
        elif self.state == self.FINISHED:
            cmd = self.finished_behavior(cmd)
        
        # Publish command
        self.vel_pub.publish(cmd)
        
        # Publish state
        state_msg = String()
        state_msg.data = self.get_state_name()
        self.state_pub.publish(state_msg)
    
    def idle_behavior(self, cmd: Twist) -> Twist:
        """Wait for initialization"""
        elapsed = time.time() - self.mission_start_time
        
        if elapsed > 3.0 and self.current_position is not None:
            self.get_logger().info('🏁 Starting qualification task - Beginning submersion')
            self.transition_to(self.SUBMERGING)
        
        return cmd
    
    def submerging_behavior(self, cmd: Twist) -> Twist:
        """
        CRITICAL: Must submerge in starting zone before leaving
        Rulebook: AUV must autonomously submerge before leaving starting zone
        """
        
        if not self.current_position:
            return cmd
        
        elapsed = time.time() - self.state_start_time
        
        # Check if we're in starting zone
        dx = abs(self.current_position[0] - self.starting_zone_x)
        dy = abs(self.current_position[1] - 0.0)
        in_starting_zone = (dx < self.starting_zone_size/2 and 
                           dy < self.starting_zone_size/2)
        
        # Stay in place while submerging
        cmd.linear.x = 0.0
        cmd.linear.y = 0.0
        cmd.angular.z = 0.0
        
        # Check if we've reached target depth
        depth_error = abs(self.current_depth - self.target_depth)
        
        if depth_error < 0.2:
            # Successfully submerged
            self.get_logger().info(
                f'✅ Submersion complete at depth {self.current_depth:.2f}m '
                f'(target: {self.target_depth:.2f}m)'
            )
            self.transition_to(self.SEARCHING)
        elif elapsed > 15.0:
            # Timeout - proceed anyway
            self.get_logger().warn('⏰ Submersion timeout - proceeding to search')
            self.transition_to(self.SEARCHING)
        elif not in_starting_zone:
            # Drifted out of zone
            self.get_logger().warn('⚠️ Drifted out of starting zone - compensating')
            cmd.linear.x = -0.1 if self.current_position[0] > self.starting_zone_x else 0.1
        else:
            # Still submerging
            if int(elapsed) % 2 == 0:
                self.get_logger().info(
                    f'⬇️  Submerging: {self.current_depth:.2f}m / {self.target_depth:.2f}m '
                    f'({elapsed:.1f}s)',
                    throttle_duration_sec=1.9
                )
        
        return cmd
    
    def searching_behavior(self, cmd: Twist) -> Twist:
        """Search for orange qualification gate"""
        
        if self.gate_detected:
            self.get_logger().info(f'🎯 Gate detected! Distance: {self.estimated_distance:.2f}m')
            self.transition_to(self.ALIGNING)
            return cmd
        
        elapsed = time.time() - self.state_start_time
        
        # Search pattern: Forward motion with sweeping rotation
        cmd.linear.x = self.search_speed
        
        # Sweep left and right (8 second cycle)
        sweep_cycle = 8.0
        sweep_phase = (elapsed % sweep_cycle) / sweep_cycle
        
        if sweep_phase < 0.5:
            cmd.angular.z = self.search_yaw_speed
            direction = "LEFT"
        else:
            cmd.angular.z = -self.search_yaw_speed
            direction = "RIGHT"
        
        if int(elapsed) % 3 == 0:
            self.get_logger().info(
                f'🔍 Searching ({direction})... {elapsed:.0f}s',
                throttle_duration_sec=2.9
            )
        
        return cmd
    
    def aligning_behavior(self, cmd: Twist) -> Twist:
        """Align with gate before approaching"""
        
        if not self.gate_detected:
            self.get_logger().warn('❌ Lost gate during alignment')
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            
            # Return to search after 2 seconds
            if time.time() - self.state_start_time > 2.0:
                self.transition_to(self.SEARCHING)
            return cmd
        
        # Check alignment quality
        is_aligned = abs(self.frame_position) < self.alignment_threshold
        
        if is_aligned and self.estimated_distance < 999:
            self.get_logger().info(
                f'✅ Alignment complete (pos={self.frame_position:+.3f}) - '
                f'Starting approach from {self.estimated_distance:.2f}m'
            )
            self.transition_to(self.APPROACHING)
            return cmd
        
        # Alignment strategy: rotation + slow forward
        cmd.angular.z = -self.frame_position * self.alignment_yaw_gain
        cmd.linear.x = 0.2  # Slow forward creep while aligning
        
        elapsed = time.time() - self.state_start_time
        if int(elapsed * 2) % 2 == 0:
            self.get_logger().info(
                f'🔄 Aligning: pos={self.frame_position:+.3f}, '
                f'dist={self.estimated_distance:.2f}m',
                throttle_duration_sec=0.4
            )
        
        # Timeout
        if elapsed > 20.0:
            self.get_logger().warn('⏰ Alignment timeout - proceeding anyway')
            self.transition_to(self.APPROACHING)
        
        return cmd
    
    def approaching_behavior(self, cmd: Twist) -> Twist:
        """Approach gate while maintaining alignment"""
        
        if not self.gate_detected:
            self.get_logger().warn('❌ Lost gate during approach')
            cmd.linear.x = 0.1
            cmd.angular.z = 0.0
            
            if time.time() - self.state_start_time > 3.0:
                self.transition_to(self.SEARCHING)
            return cmd
        
        # Check if close enough to commit to passing
        if self.estimated_distance < self.passing_distance:
            self.get_logger().info(
                f'🚀 Reached passing distance ({self.estimated_distance:.2f}m) - '
                f'Committing to passage'
            )
            self.transition_to(self.PASSING)
            return cmd
        
        # Approach with active alignment correction
        cmd.linear.x = self.approach_speed
        cmd.angular.z = -self.frame_position * self.alignment_yaw_gain * 0.5
        
        if int(time.time() * 2) % 2 == 0:
            self.get_logger().info(
                f'➡️  Approaching: dist={self.estimated_distance:.2f}m, '
                f'pos={self.frame_position:+.3f}',
                throttle_duration_sec=0.4
            )
        
        return cmd
    
    def passing_behavior(self, cmd: Twist) -> Twist:
        """Pass through gate at full speed"""
        
        if not self.current_position:
            return cmd
        
        # Full speed straight ahead
        cmd.linear.x = self.passing_speed
        cmd.linear.y = 0.0
        cmd.angular.z = 0.0
        
        # Check if we've cleared the gate (passed X = gate_x_position + buffer)
        clearance_distance = 1.5
        if self.current_position[0] > (self.gate_x_position + clearance_distance):
            elapsed = time.time() - self.mission_start_time
            self.get_logger().info('='*70)
            self.get_logger().info('✅ GATE CLEARED!')
            self.get_logger().info(f'   Total mission time: {elapsed:.2f}s')
            self.get_logger().info('   Beginning surface procedure')
            self.get_logger().info('='*70)
            self.transition_to(self.SURFACING)
            return cmd
        
        # Show progress
        distance_to_gate = self.gate_x_position - self.current_position[0]
        if int(time.time() * 2) % 2 == 0:
            self.get_logger().info(
                f'🚀 PASSING: {abs(distance_to_gate):.2f}m to gate center',
                throttle_duration_sec=0.4
            )
        
        return cmd
    
    def surfacing_behavior(self, cmd: Twist) -> Twist:
        """Return to surface after completing task"""
        
        # Stop forward motion
        cmd.linear.x = 0.0
        cmd.linear.y = 0.0
        cmd.angular.z = 0.0
        
        # Ascend
        cmd.linear.z = 0.8
        
        # Check if at surface
        if self.current_depth > -0.2:
            self.get_logger().info('🏁 Surfaced - Mission complete!')
            self.transition_to(self.FINISHED)
        else:
            if int(time.time()) % 2 == 0:
                self.get_logger().info(
                    f'⬆️  Surfacing: {self.current_depth:.2f}m',
                    throttle_duration_sec=1.9
                )
        
        return cmd
    
    def finished_behavior(self, cmd: Twist) -> Twist:
        """Mission complete - hold position at surface"""
        elapsed = time.time() - self.mission_start_time
        
        if int(time.time()) % 10 == 0:
            self.get_logger().info('='*70)
            self.get_logger().info('🎉 QUALIFICATION TASK COMPLETE')
            self.get_logger().info(f'   Total time: {elapsed:.2f}s')
            self.get_logger().info('='*70)
        
        # Zero velocity
        cmd.linear.x = 0.0
        cmd.linear.y = 0.0
        cmd.linear.z = 0.0
        cmd.angular.z = 0.0
        
        return cmd
    
    def transition_to(self, new_state: int):
        """Transition to new state"""
        old_name = self.get_state_name()
        self.state = new_state
        self.state_start_time = time.time()
        new_name = self.get_state_name()
        
        self.get_logger().info(f'🔄 STATE: {old_name} → {new_name}')
    
    def get_state_name(self) -> str:
        """Get human-readable state name"""
        names = {
            self.IDLE: 'IDLE',
            self.SUBMERGING: 'SUBMERGING',
            self.SEARCHING: 'SEARCHING',
            self.ALIGNING: 'ALIGNING',
            self.APPROACHING: 'APPROACHING',
            self.PASSING: 'PASSING',
            self.SURFACING: 'SURFACING',
            self.FINISHED: 'FINISHED'
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
        node.vel_pub.publish(stop_cmd)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()