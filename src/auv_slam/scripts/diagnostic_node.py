#!/usr/bin/env python3
"""
Movement Diagnostic & Test Script
Helps identify why the AUV is not moving
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
import time


class MovementDiagnostic(Node):
    def __init__(self):
        super().__init__('movement_diagnostic')
        
        # State tracking
        self.odom_received = False
        self.cmd_vel_sent = False
        self.thruster_cmds_received = [False] * 6
        self.last_thruster_values = [0.0] * 6
        
        self.initial_position = None
        self.current_position = None
        self.movement_detected = False
        
        # Subscriptions
        self.odom_sub = self.create_subscription(
            Odometry, '/ground_truth/odom', self.odom_callback, 10)
        
        # Monitor thruster commands
        for i in range(1, 7):
            self.create_subscription(
                Float64, f'/thruster{i}_cmd',
                lambda msg, idx=i-1: self.thruster_callback(msg, idx), 10)
        
        # Publisher for test commands
        self.cmd_vel_pub = self.create_publisher(Twist, '/rp2040/cmd_vel', 10)
        
        # Test timer
        self.test_phase = 0
        self.test_timer = self.create_timer(5.0, self.run_test)
        
        # Status timer
        self.status_timer = self.create_timer(2.0, self.print_status)
        
        self.get_logger().info('='*70)
        self.get_logger().info('🔍 MOVEMENT DIAGNOSTIC STARTED')
        self.get_logger().info('='*70)
    
    def odom_callback(self, msg: Odometry):
        self.odom_received = True
        pos = msg.pose.pose.position
        
        if self.initial_position is None:
            self.initial_position = (pos.x, pos.y, pos.z)
            self.get_logger().info(
                f'📍 Initial position: X={pos.x:.2f}, Y={pos.y:.2f}, Z={pos.z:.2f}'
            )
        
        self.current_position = (pos.x, pos.y, pos.z)
        
        # Check for movement
        if self.initial_position and self.cmd_vel_sent:
            dx = abs(pos.x - self.initial_position[0])
            dy = abs(pos.y - self.initial_position[1])
            dz = abs(pos.z - self.initial_position[2])
            total_movement = (dx**2 + dy**2 + dz**2)**0.5
            
            if total_movement > 0.05 and not self.movement_detected:
                self.movement_detected = True
                self.get_logger().info(
                    f'✅ MOVEMENT DETECTED! Distance: {total_movement:.3f}m'
                )
    
    def thruster_callback(self, msg: Float64, idx: int):
        self.thruster_cmds_received[idx] = True
        self.last_thruster_values[idx] = msg.data
    
    def run_test(self):
        """Run systematic movement tests"""
        cmd = Twist()
        
        if self.test_phase == 0:
            self.get_logger().info('='*70)
            self.get_logger().info('🧪 TEST 1: Vertical thrust DOWN (negative Z)')
            self.get_logger().info('='*70)
            cmd.linear.z = -1.0  # DOWN in AUV convention
            self.cmd_vel_sent = True
            
        elif self.test_phase == 1:
            self.get_logger().info('='*70)
            self.get_logger().info('🧪 TEST 2: Vertical thrust UP (positive Z)')
            self.get_logger().info('='*70)
            cmd.linear.z = 1.0  # UP
            
        elif self.test_phase == 2:
            self.get_logger().info('='*70)
            self.get_logger().info('🧪 TEST 3: Forward thrust (positive X)')
            self.get_logger().info('='*70)
            cmd.linear.x = 0.5  # FORWARD
            cmd.linear.z = 0.0
            
        elif self.test_phase == 3:
            self.get_logger().info('='*70)
            self.get_logger().info('🧪 TEST 4: Stop all motion')
            self.get_logger().info('='*70)
            cmd = Twist()  # All zeros
            
        elif self.test_phase == 4:
            self.get_logger().info('='*70)
            self.get_logger().info('✅ DIAGNOSTIC COMPLETE')
            self.get_logger().info('='*70)
            self.print_final_report()
            self.test_timer.cancel()
            return
        
        self.cmd_vel_pub.publish(cmd)
        self.test_phase += 1
    
    def print_status(self):
        """Print current diagnostic status"""
        self.get_logger().info('-'*70)
        
        # Odometry check
        if self.odom_received and self.current_position:
            x, y, z = self.current_position
            self.get_logger().info(
                f'✓ Odometry: X={x:.3f}, Y={y:.3f}, Z={z:.3f}'
            )
        else:
            self.get_logger().error('✗ NO ODOMETRY DATA!')
        
        # Thruster command check
        active_thrusters = sum(self.thruster_cmds_received)
        if active_thrusters > 0:
            self.get_logger().info(f'✓ Thrusters active: {active_thrusters}/6')
            
            # Show non-zero thrusters
            non_zero = [f'T{i+1}={v:.1f}' for i, v in enumerate(self.last_thruster_values) if abs(v) > 0.1]
            if non_zero:
                self.get_logger().info(f'  Active: {", ".join(non_zero)}')
        else:
            self.get_logger().warn('✗ No thruster commands received!')
        
        # Movement check
        if self.movement_detected:
            self.get_logger().info('✓ Movement confirmed!')
        elif self.cmd_vel_sent:
            self.get_logger().warn('⚠️  No movement detected yet...')
    
    def print_final_report(self):
        """Print comprehensive diagnostic report"""
        self.get_logger().info('')
        self.get_logger().info('='*70)
        self.get_logger().info('📊 FINAL DIAGNOSTIC REPORT')
        self.get_logger().info('='*70)
        
        # 1. Odometry
        if self.odom_received:
            self.get_logger().info('✅ Odometry: WORKING')
        else:
            self.get_logger().error('❌ Odometry: NOT WORKING - Check bridge configuration!')
        
        # 2. Thruster commands
        active = sum(self.thruster_cmds_received)
        if active == 6:
            self.get_logger().info('✅ All 6 thrusters: RECEIVING COMMANDS')
        elif active > 0:
            self.get_logger().warn(f'⚠️  Only {active}/6 thrusters active')
        else:
            self.get_logger().error('❌ Thrusters: NO COMMANDS - Check thruster mapper!')
        
        # 3. Movement
        if self.movement_detected:
            self.get_logger().info('✅ Movement: CONFIRMED')
            if self.initial_position and self.current_position:
                dx = self.current_position[0] - self.initial_position[0]
                dy = self.current_position[1] - self.initial_position[1]
                dz = self.current_position[2] - self.initial_position[2]
                self.get_logger().info(
                    f'   Displacement: ΔX={dx:.3f}, ΔY={dy:.3f}, ΔZ={dz:.3f}'
                )
        else:
            self.get_logger().error('❌ Movement: NOT DETECTED')
            self.get_logger().error('')
            self.get_logger().error('POSSIBLE CAUSES:')
            self.get_logger().error('1. Sign convention error in thruster mapper')
            self.get_logger().error('2. Thrust magnitude too low')
            self.get_logger().error('3. URDF thruster plugin misconfiguration')
            self.get_logger().error('4. Bridge topic mismatch')
        
        self.get_logger().info('='*70)


def main(args=None):
    rclpy.init(args=args)
    node = MovementDiagnostic()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()