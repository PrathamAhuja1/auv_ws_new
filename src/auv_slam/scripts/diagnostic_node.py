#!/usr/bin/env python3
"""
DEEP DIAGNOSTIC - Find EXACTLY why thrusters aren't working
Run this AFTER launching qualification.launch.py

Usage:
    ros2 run auv_slam deep_diagnostic.py
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import time


class DeepDiagnostic(Node):
    def __init__(self):
        super().__init__('deep_diagnostic')
        
        # Tracking
        self.cmd_vel_received = False
        self.odom_received = False
        self.thruster_cmds_received = [False] * 6
        self.last_cmd_vel = None
        self.last_odom = None
        self.last_thruster_values = [0.0] * 6
        
        self.odom_history = []
        
        # Subscribe to everything
        self.cmd_vel_sub = self.create_subscription(
            Twist, '/rp2040/cmd_vel', self.cmd_vel_callback, 10)
        
        self.odom_sub = self.create_subscription(
            Odometry, '/ground_truth/odom', self.odom_callback, 10)
        
        # Monitor ALL possible thruster topics
        self.thruster_subs = []
        for i in range(1, 7):
            # Subscribe to our expected topic
            self.create_subscription(
                Float64, f'/thruster{i}_cmd', 
                lambda msg, idx=i-1: self.thruster_callback(msg, idx, 'thruster_cmd'), 10)
            
            # Subscribe to Gazebo topics (in case remapping failed)
            self.create_subscription(
                Float64, f'/model/orca4_ign/joint/thruster{i}_joint/cmd_pos',
                lambda msg, idx=i-1: self.thruster_callback(msg, idx, 'gazebo_direct'), 10)
        
        # Publishers for manual testing
        self.thruster_pubs = []
        for i in range(1, 7):
            pub = self.create_publisher(Float64, f'/thruster{i}_cmd', 10)
            self.thruster_pubs.append(pub)
        
        self.get_logger().info('='*70)
        self.get_logger().info('🔍 DEEP DIAGNOSTIC STARTED')
        self.get_logger().info('='*70)
        self.get_logger().info('Monitoring all topics for 10 seconds...')
        
        # Phase 1: Passive monitoring
        self.phase = 1
        self.phase_start = time.time()
        self.timer = self.create_timer(0.5, self.diagnostic_loop)
        
    def cmd_vel_callback(self, msg: Twist):
        self.cmd_vel_received = True
        self.last_cmd_vel = msg
        
    def odom_callback(self, msg: Odometry):
        self.odom_received = True
        self.last_odom = msg
        self.odom_history.append((
            time.time(),
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ))
        
    def thruster_callback(self, msg: Float64, idx: int, source: str):
        self.thruster_cmds_received[idx] = True
        self.last_thruster_values[idx] = msg.data
        
        if abs(msg.data) > 0.1:
            self.get_logger().info(
                f'  💨 T{idx+1} = {msg.data:.2f} (from {source})',
                throttle_duration_sec=0.5
            )
    
    def diagnostic_loop(self):
        elapsed = time.time() - self.phase_start
        
        if self.phase == 1:  # Passive monitoring (10s)
            if elapsed < 10.0:
                if int(elapsed) % 2 == 0:
                    self.get_logger().info(f'⏱️  Monitoring... {elapsed:.0f}s / 10s')
                return
            else:
                self.print_phase1_results()
                self.phase = 2
                self.phase_start = time.time()
                self.get_logger().info('')
                self.get_logger().info('='*70)
                self.get_logger().info('📋 PHASE 2: MANUAL THRUSTER TEST')
                self.get_logger().info('='*70)
                self.get_logger().info('Sending direct thruster commands...')
                
        elif self.phase == 2:  # Manual thruster test (5s)
            if elapsed < 5.0:
                # Send test commands
                for i, pub in enumerate(self.thruster_pubs):
                    msg = Float64()
                    msg.data = 30.0 if i < 4 else 0.0  # Only horizontal thrusters
                    pub.publish(msg)
                return
            else:
                self.print_phase2_results()
                self.phase = 3
                self.phase_start = time.time()
                
        elif self.phase == 3:  # Stop and analyze
            # Stop all thrusters
            for pub in self.thruster_pubs:
                msg = Float64()
                msg.data = 0.0
                pub.publish(msg)
            
            if elapsed < 2.0:
                return
            else:
                self.print_final_diagnosis()
                self.timer.cancel()
    
    def print_phase1_results(self):
        self.get_logger().info('='*70)
        self.get_logger().info('📊 PHASE 1 RESULTS: Passive Monitoring')
        self.get_logger().info('='*70)
        
        # 1. Odometry check
        if self.odom_received:
            self.get_logger().info('✅ Odometry: WORKING')
            if len(self.odom_history) > 1:
                dx = self.odom_history[-1][1] - self.odom_history[0][1]
                dy = self.odom_history[-1][2] - self.odom_history[0][2]
                dz = self.odom_history[-1][3] - self.odom_history[0][3]
                movement = (dx**2 + dy**2 + dz**2)**0.5
                self.get_logger().info(f'   Movement: {movement:.3f}m in 10s')
                if movement < 0.1:
                    self.get_logger().warn('   ⚠️  Robot is NOT MOVING!')
        else:
            self.get_logger().error('❌ Odometry: NO DATA')
        
        # 2. Cmd_vel check
        if self.cmd_vel_received:
            self.get_logger().info('✅ Cmd_vel: WORKING')
            if self.last_cmd_vel:
                self.get_logger().info(
                    f'   Last: vx={self.last_cmd_vel.linear.x:.2f}, '
                    f'vy={self.last_cmd_vel.linear.y:.2f}, '
                    f'vz={self.last_cmd_vel.linear.z:.2f}, '
                    f'yaw={self.last_cmd_vel.angular.z:.2f}'
                )
        else:
            self.get_logger().error('❌ Cmd_vel: NO DATA')
            self.get_logger().error('   → Navigator is not publishing commands!')
        
        # 3. Thruster commands check
        active_thrusters = sum(self.thruster_cmds_received)
        if active_thrusters > 0:
            self.get_logger().info(f'✅ Thruster Commands: {active_thrusters}/6 active')
            for i, (received, value) in enumerate(zip(self.thruster_cmds_received, self.last_thruster_values)):
                status = '✓' if received else '✗'
                self.get_logger().info(f'   {status} T{i+1}: {value:.2f}')
        else:
            self.get_logger().error('❌ Thruster Commands: NONE RECEIVED')
            self.get_logger().error('   → Bridge or thruster mapper not working!')
    
    def print_phase2_results(self):
        self.get_logger().info('='*70)
        self.get_logger().info('📊 PHASE 2 RESULTS: Manual Thruster Test')
        self.get_logger().info('='*70)
        
        if len(self.odom_history) > 10:
            # Compare position before and after manual test
            before = self.odom_history[-10]
            after = self.odom_history[-1]
            
            dx = after[1] - before[1]
            dy = after[2] - before[2]
            dz = after[3] - before[3]
            movement = (dx**2 + dy**2 + dz**2)**0.5
            
            self.get_logger().info(f'Movement during manual test: {movement:.3f}m')
            
            if movement > 0.5:
                self.get_logger().info('✅ Manual thruster commands WORK!')
                self.get_logger().info('   → Bridge and Gazebo physics are OK')
                self.get_logger().info('   → Problem is in thruster_mapper or navigator')
            else:
                self.get_logger().error('❌ Manual thruster commands FAILED!')
                self.get_logger().error('   → Bridge or Gazebo configuration issue')
    
    def print_final_diagnosis(self):
        self.get_logger().info('='*70)
        self.get_logger().info('🔬 FINAL DIAGNOSIS')
        self.get_logger().info('='*70)
        
        # Determine the root cause
        if not self.odom_received:
            self.get_logger().error('❌ ROOT CAUSE: No odometry data')
            self.get_logger().error('   FIX: Check bridge configuration for odometry topic')
            
        elif not self.cmd_vel_received:
            self.get_logger().error('❌ ROOT CAUSE: Navigator not publishing cmd_vel')
            self.get_logger().error('   FIX: Check qualification_navigator_node.py is running')
            self.get_logger().error('        Run: ros2 node list')
            
        elif sum(self.thruster_cmds_received) == 0:
            self.get_logger().error('❌ ROOT CAUSE: No thruster commands received')
            self.get_logger().error('   FIX: Thruster mapper not working')
            self.get_logger().error('        Check simple_thruster_mapper.py is running')
            
        else:
            # Movement check
            if len(self.odom_history) > 20:
                total_movement = 0
                for i in range(1, len(self.odom_history)):
                    dx = self.odom_history[i][1] - self.odom_history[i-1][1]
                    dy = self.odom_history[i][2] - self.odom_history[i-1][2]
                    dz = self.odom_history[i][3] - self.odom_history[i-1][3]
                    total_movement += (dx**2 + dy**2 + dz**2)**0.5
                
                if total_movement < 0.5:
                    self.get_logger().error('❌ ROOT CAUSE: All systems working but robot not moving')
                    self.get_logger().error('   FIX: Possible causes:')
                    self.get_logger().error('        1. Thruster commands too weak')
                    self.get_logger().error('        2. Robot stuck on ground/wall')
                    self.get_logger().error('        3. Buoyancy misconfigured')
                    self.get_logger().error('        4. Thruster allocation matrix wrong')
                else:
                    self.get_logger().info('✅ DIAGNOSIS: Everything working!')
                    self.get_logger().info(f'   Total movement: {total_movement:.2f}m')
        
        self.get_logger().info('='*70)
        self.get_logger().info('🔍 RECOMMENDED CHECKS:')
        self.get_logger().info('='*70)
        self.get_logger().info('1. Run: ros2 topic list | grep thruster')
        self.get_logger().info('2. Run: ros2 topic echo /thruster1_cmd')
        self.get_logger().info('3. Run: ros2 topic echo /rp2040/cmd_vel')
        self.get_logger().info('4. Run: ros2 node list')
        self.get_logger().info('5. Check Gazebo GUI - is robot visible and at correct position?')
        self.get_logger().info('='*70)


def main(args=None):
    rclpy.init(args=args)
    node = DeepDiagnostic()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop all thrusters
        for pub in node.thruster_pubs:
            msg = Float64()
            msg.data = 0.0
            pub.publish(msg)
        
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()