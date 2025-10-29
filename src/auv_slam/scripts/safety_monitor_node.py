#!/usr/bin/env python3
"""
Enhanced Safety Monitor Node with comprehensive fault detection
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Float32, String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
import time
import math
from tf_transformations import euler_from_quaternion
from enum import IntEnum

class SafetyState(IntEnum):
    NORMAL = 0
    WARNING = 1
    CRITICAL = 2
    EMERGENCY = 3

class EnhancedSafetyMonitor(Node):
    def __init__(self):
        super().__init__('enhanced_safety_monitor')
        
        # Parameters
        self.declare_parameter('max_depth', -3.5)
        self.declare_parameter('min_depth', 0.2)
        self.declare_parameter('max_roll', 0.785)
        self.declare_parameter('max_pitch', 0.785)
        self.declare_parameter('watchdog_timeout', 5.0)
        self.declare_parameter('max_mission_time', 900.0)
        self.declare_parameter('gate_collision_depth', -0.5)
        self.declare_parameter('pool_bounds_x', [-25.0, 25.0])
        self.declare_parameter('pool_bounds_y', [-12.5, 12.5])
        self.declare_parameter('emergency_surface_thrust', 1.0)
        
        self.max_depth = self.get_parameter('max_depth').value
        self.min_depth = self.get_parameter('min_depth').value
        self.max_roll = self.get_parameter('max_roll').value
        self.max_pitch = self.get_parameter('max_pitch').value
        self.watchdog_timeout = self.get_parameter('watchdog_timeout').value
        self.max_mission_time = self.get_parameter('max_mission_time').value
        self.gate_collision_depth = self.get_parameter('gate_collision_depth').value
        self.pool_bounds_x = self.get_parameter('pool_bounds_x').value
        self.pool_bounds_y = self.get_parameter('pool_bounds_y').value
        self.emergency_thrust = self.get_parameter('emergency_surface_thrust').value
        
        # State tracking
        self.safety_state = SafetyState.NORMAL
        self.mission_start_time = time.time()
        self.last_odom_time = None
        self.last_imu_time = None
        self.last_cmd_vel_time = None
        
        self.current_odom = None
        self.current_imu = None
        self.emergency_triggered = False
        
        # Fault counters
        self.consecutive_depth_violations = 0
        self.consecutive_tilt_violations = 0
        self.wall_touches = 0
        self.floor_touches = 0
        
        # Subscriptions
        self.odom_sub = self.create_subscription(
            Odometry, '/odometry/filtered', self.odom_callback, 10)
        self.gt_odom_sub = self.create_subscription(
            Odometry, '/ground_truth/odom', self.gt_odom_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10)
        self.cmd_vel_sub = self.create_subscription(
            Twist, '/rp2040/cmd_vel', self.cmd_vel_callback, 10)
        
        # Publishers
        self.emergency_pub = self.create_publisher(Bool, '/safety/emergency_stop', 10)
        self.emergency_cmd_pub = self.create_publisher(Twist, '/safety/emergency_cmd', 10)
        self.state_pub = self.create_publisher(String, '/safety/state', 10)
        self.status_pub = self.create_publisher(Float32, '/safety/status_code', 10)
        self.diagnostics_pub = self.create_publisher(String, '/safety/diagnostics', 10)
        
        # Monitoring timer (10 Hz)
        self.timer = self.create_timer(0.1, self.monitor_loop)
        
        self.get_logger().info('Enhanced Safety Monitor initialized')
    
    def odom_callback(self, msg: Odometry):
        self.last_odom_time = time.time()
        self.current_odom = msg
    
    def gt_odom_callback(self, msg: Odometry):
        # Use ground truth for additional verification
        pass
    
    def imu_callback(self, msg: Imu):
        self.last_imu_time = time.time()
        self.current_imu = msg
    
    def cmd_vel_callback(self, msg: Twist):
        self.last_cmd_vel_time = time.time()
    
    def monitor_loop(self):
        """Main safety monitoring loop"""
        if self.emergency_triggered:
            self.execute_emergency_surface()
            return
        
        # Check all safety conditions
        checks = [
            self.check_mission_timeout(),
            self.check_watchdog_timeout(),
            self.check_depth_limits(),
            self.check_orientation_limits(),
            self.check_pool_boundaries(),
            self.check_sensor_health()
        ]
        
        # Determine safety state
        if any(check == SafetyState.EMERGENCY for check in checks):
            self.trigger_emergency("Multiple critical faults detected")
        elif any(check == SafetyState.CRITICAL for check in checks):
            self.safety_state = SafetyState.CRITICAL
        elif any(check == SafetyState.WARNING for check in checks):
            self.safety_state = SafetyState.WARNING
        else:
            self.safety_state = SafetyState.NORMAL
        
        # Publish status
        self.publish_status()
    
    def check_mission_timeout(self) -> SafetyState:
        """Check if mission has exceeded time limit"""
        elapsed = time.time() - self.mission_start_time
        
        if elapsed > self.max_mission_time:
            self.trigger_emergency(f"Mission timeout ({elapsed:.1f}s)")
            return SafetyState.EMERGENCY
        elif elapsed > self.max_mission_time * 0.9:
            self.get_logger().warn(f"Mission time warning: {elapsed:.1f}s / {self.max_mission_time}s")
            return SafetyState.WARNING
        
        return SafetyState.NORMAL
    
    def check_watchdog_timeout(self) -> SafetyState:
        """Check for sensor/control timeouts"""
        now = time.time()
        
        if self.last_odom_time:
            odom_age = now - self.last_odom_time
            if odom_age > self.watchdog_timeout:
                self.trigger_emergency(f"Odometry timeout ({odom_age:.1f}s)")
                return SafetyState.EMERGENCY
        
        if self.last_imu_time:
            imu_age = now - self.last_imu_time
            if imu_age > self.watchdog_timeout:
                self.get_logger().error(f"IMU timeout ({imu_age:.1f}s)")
                return SafetyState.CRITICAL
        
        return SafetyState.NORMAL
    
    def check_depth_limits(self) -> SafetyState:
        """Check depth boundaries"""
        if not self.current_odom:
            return SafetyState.WARNING
        
        depth = self.current_odom.pose.pose.position.z
        
        # Check violation
        if depth < self.max_depth or depth > self.min_depth:
            self.consecutive_depth_violations += 1
            
            if self.consecutive_depth_violations > 5:
                self.trigger_emergency(f"Depth violation: {depth:.2f}m")
                return SafetyState.EMERGENCY
            
            self.get_logger().warn(f"Depth limit exceeded: {depth:.2f}m")
            return SafetyState.CRITICAL
        else:
            self.consecutive_depth_violations = 0
        
        # Warning zone
        if depth < self.max_depth + 0.5 or depth > self.min_depth - 0.3:
            return SafetyState.WARNING
        
        return SafetyState.NORMAL
    
    def check_orientation_limits(self) -> SafetyState:
        """Check for excessive roll/pitch"""
        if not self.current_imu:
            return SafetyState.WARNING
        
        q = self.current_imu.orientation
        roll, pitch, _ = euler_from_quaternion([q.x, q.y, q.z, q.w])
        
        if abs(roll) > self.max_roll or abs(pitch) > self.max_pitch:
            self.consecutive_tilt_violations += 1
            
            if self.consecutive_tilt_violations > 10:
                self.trigger_emergency(f"Excessive tilt: R={math.degrees(roll):.1f}°, P={math.degrees(pitch):.1f}°")
                return SafetyState.EMERGENCY
            
            self.get_logger().warn(f"Orientation warning: R={math.degrees(roll):.1f}°, P={math.degrees(pitch):.1f}°")
            return SafetyState.CRITICAL
        else:
            self.consecutive_tilt_violations = 0
        
        return SafetyState.NORMAL
    
    def check_pool_boundaries(self) -> SafetyState:
        """Check if AUV is within pool bounds"""
        if not self.current_odom:
            return SafetyState.WARNING
        
        pos = self.current_odom.pose.pose.position
        
        if not (self.pool_bounds_x[0] < pos.x < self.pool_bounds_x[1]):
            self.get_logger().error(f"X boundary violation: {pos.x:.2f}m")
            return SafetyState.CRITICAL
        
        if not (self.pool_bounds_y[0] < pos.y < self.pool_bounds_y[1]):
            self.get_logger().error(f"Y boundary violation: {pos.y:.2f}m")
            return SafetyState.CRITICAL
        
        return SafetyState.NORMAL
    
    def check_sensor_health(self) -> SafetyState:
        """Verify sensor data quality"""
        if not self.current_odom or not self.current_imu:
            return SafetyState.WARNING
        
        # Check for NaN values
        pos = self.current_odom.pose.pose.position
        if math.isnan(pos.x) or math.isnan(pos.y) or math.isnan(pos.z):
            self.get_logger().error("NaN detected in odometry")
            return SafetyState.CRITICAL
        
        # Check IMU data
        accel = self.current_imu.linear_acceleration
        if math.isnan(accel.x) or math.isnan(accel.y) or math.isnan(accel.z):
            self.get_logger().error("NaN detected in IMU")
            return SafetyState.CRITICAL
        
        return SafetyState.NORMAL
    
    def trigger_emergency(self, reason: str):
        """Activate emergency protocols"""
        if self.emergency_triggered:
            return
        
        self.emergency_triggered = True
        self.safety_state = SafetyState.EMERGENCY
        
        self.get_logger().error(f"🚨 EMERGENCY STOP: {reason}")
        
        # Publish emergency flag
        emergency_msg = Bool()
        emergency_msg.data = True
        self.emergency_pub.publish(emergency_msg)
        
        # Start emergency surfacing
        self.execute_emergency_surface()
    
    def execute_emergency_surface(self):
        """Command maximum upward thrust"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.linear.y = 0.0
        cmd.linear.z = -self.emergency_thrust  # Maximum up
        cmd.angular.z = 0.0
        
        self.emergency_cmd_pub.publish(cmd)
    
    def publish_status(self):
        """Publish current safety status"""
        # State string
        state_msg = String()
        state_msg.data = self.safety_state.name
        self.state_pub.publish(state_msg)
        
        # Status code
        status_msg = Float32()
        status_msg.data = float(self.safety_state.value)
        self.status_pub.publish(status_msg)
        
        # Diagnostics
        if self.safety_state != SafetyState.NORMAL:
            diag_msg = String()
            diag_msg.data = self.generate_diagnostics()
            self.diagnostics_pub.publish(diag_msg)
    
    def generate_diagnostics(self) -> str:
        """Generate diagnostic report"""
        diag = []
        
        if not self.current_odom:
            diag.append("No odometry data")
        else:
            pos = self.current_odom.pose.pose.position
            diag.append(f"Position: ({pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f})")
        
        if not self.current_imu:
            diag.append("No IMU data")
        else:
            q = self.current_imu.orientation
            roll, pitch, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
            diag.append(f"Orientation: R={math.degrees(roll):.1f}° P={math.degrees(pitch):.1f}° Y={math.degrees(yaw):.1f}°")
        
        elapsed = time.time() - self.mission_start_time
        diag.append(f"Mission time: {elapsed:.1f}s / {self.max_mission_time}s")
        
        diag.append(f"Depth violations: {self.consecutive_depth_violations}")
        diag.append(f"Tilt violations: {self.consecutive_tilt_violations}")
        
        return " | ".join(diag)


def main(args=None):
    rclpy.init(args=args)
    node = EnhancedSafetyMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()