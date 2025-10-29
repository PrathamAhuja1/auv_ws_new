#!/usr/bin/env python3
"""
Thruster Diagnostic Tool
Tests each thruster individually to determine correct mapping
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from nav_msgs.msg import Odometry
import time


class ThrusterDiagnostic(Node):
    def __init__(self):
        super().__init__('thruster_diagnostic')
        
        # Publishers for each thruster
        self.thruster_pubs = []
        for i in range(1, 7):
            pub = self.create_publisher(Float64, f'/thruster{i}_cmd', 10)
            self.thruster_pubs.append(pub)
        
        # Odometry subscriber
        self.odom_sub = self.create_subscription(
            Odometry, '/ground_truth/odom', self.odom_callback, 10)
        
        self.start_position = None
        self.current_position = None
        self.test_phase = 0
        self.phase_start_time = None
        
        # Wait for first odom
        self.get_logger().info('Waiting for odometry...')
        time.sleep(2.0)
        
        # Start testing
        self.test_timer = self.create_timer(0.5, self.run_test)
        
        self.get_logger().info('='*70)
        self.get_logger().info('🔧 THRUSTER DIAGNOSTIC STARTED')
        self.get_logger().info('='*70)
        self.get_logger().info('This will test each thruster individually')
        self.get_logger().info('Watch the Gazebo window to see which way the robot moves')
        self.get_logger().info('='*70)
    
    def odom_callback(self, msg: Odometry):
        pos = msg.pose.pose.position
        self.current_position = (pos.x, pos.y, pos.z)
        
        if self.start_position is None:
            self.start_position = self.current_position
    
    def stop_all_thrusters(self):
        """Stop all thrusters"""
        for pub in self.thruster_pubs:
            msg = Float64()
            msg.data = 0.0
            pub.publish(msg)
    
    def run_test(self):
        """Run systematic thruster tests"""
        
        if self.current_position is None:
            return
        
        # Test sequence: 3 seconds per thruster
        test_duration = 3.0
        thrust_value = 50.0
        
        if self.phase_start_time is None:
            self.phase_start_time = time.time()
        
        elapsed = time.time() - self.phase_start_time
        
        # Phase 0-5: Test individual thrusters
        if self.test_phase < 6:
            if elapsed < test_duration:
                # Apply thrust to current thruster
                msg = Float64()
                msg.data = thrust_value
                self.thruster_pubs[self.test_phase].publish(msg)
                
                # Calculate displacement
                dx = self.current_position[0] - self.start_position[0]
                dy = self.current_position[1] - self.start_position[1]
                dz = self.current_position[2] - self.start_position[2]
                
                if int(elapsed * 2) % 2 == 0:  # Log every 0.5s
                    self.get_logger().info(
                        f'T{self.test_phase + 1} @ {thrust_value:.0f}: '
                        f'ΔX={dx:+.3f}, ΔY={dy:+.3f}, ΔZ={dz:+.3f}'
                    )
            else:
                # Phase complete - analyze results
                dx = self.current_position[0] - self.start_position[0]
                dy = self.current_position[1] - self.start_position[1]
                dz = self.current_position[2] - self.start_position[2]
                
                self.get_logger().info('='*70)
                self.get_logger().info(f'✅ THRUSTER {self.test_phase + 1} RESULTS:')
                self.get_logger().info(f'   Total displacement: ΔX={dx:+.3f}, ΔY={dy:+.3f}, ΔZ={dz:+.3f}')
                
                # Interpret results
                if abs(dz) > 0.1:
                    self.get_logger().info(f'   → VERTICAL thruster ({"UP" if dz > 0 else "DOWN"})')
                elif abs(dx) > abs(dy):
                    self.get_logger().info(f'   → SURGE thruster ({"FORWARD" if dx > 0 else "BACKWARD"})')
                elif abs(dy) > abs(dx):
                    self.get_logger().info(f'   → SWAY thruster ({"LEFT" if dy > 0 else "RIGHT"})')
                else:
                    self.get_logger().warn(f'   → MINIMAL MOVEMENT - Check thruster!')
                
                self.get_logger().info('='*70)
                
                # Stop all thrusters
                self.stop_all_thrusters()
                time.sleep(1.0)
                
                # Move to next phase
                self.test_phase += 1
                self.phase_start_time = None
                self.start_position = self.current_position
        
        else:
            # All tests complete
            self.get_logger().info('='*70)
            self.get_logger().info('🎉 DIAGNOSTIC COMPLETE!')
            self.get_logger().info('='*70)
            self.get_logger().info('Now analyze the results above to determine correct mapping')
            self.get_logger().info('='*70)
            self.test_timer.cancel()
            self.stop_all_thrusters()


def main(args=None):
    rclpy.init(args=args)
    node = ThrusterDiagnostic()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_all_thrusters()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()