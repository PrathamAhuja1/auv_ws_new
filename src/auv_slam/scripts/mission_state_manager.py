#!/usr/bin/env python3
"""
Enhanced Mission State Manager with persistence and recovery
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Float32
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import json
import os
from pathlib import Path
from enum import IntEnum
import time

class MissionTask(IntEnum):
    SUBMERGE = 0
    NAVIGATE_GATE = 1
    FLARE_LOCALIZATION = 2
    SURFACE = 3
    COMPLETED = 4

class MissionStateManagerEnhanced(Node):
    def __init__(self):
        super().__init__('mission_state_manager_enhanced')
        
        # State storage
        self.state_dir = Path.home() / '.ros' / 'auv_mission_state'
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.state_dir / 'mission_state.json'
        self.backup_file = self.state_dir / 'mission_state_backup.json'
        
        # Parameters
        self.declare_parameter('auto_recover', True)
        self.declare_parameter('save_interval', 2.0)
        self.declare_parameter('max_retries', 3)
        
        self.auto_recover = self.get_parameter('auto_recover').value
        self.save_interval = self.get_parameter('save_interval').value
        self.max_retries = self.get_parameter('max_retries').value
        
        # Current state
        self.state = {
            'current_task': MissionTask.SUBMERGE.name,
            'tasks_completed': [],
            'navigation_complete': False,
            'gate_passed': False,
            'flare_sequence': [],
            'flares_bumped': [],
            'flares_bumped_in_order': False,
            'attempt_count': 0,
            'retry_count': 0,
            'last_known_position': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'mission_start_time': time.time(),
            'total_mission_time': 0.0,
            'emergency_surfaced': False,
            'score': 0,
            'penalties': 0
        }
        
        # Load previous state if recovering
        if self.auto_recover:
            self.load_state()
        
        # Subscriptions
        self.task_update_sub = self.create_subscription(
            String, '/mission/task_update', self.task_update_callback, 10)
        self.gate_passed_sub = self.create_subscription(
            Bool, '/gate/passed', self.gate_passed_callback, 10)
        self.flare_bumped_sub = self.create_subscription(
            String, '/flare/bumped', self.flare_bumped_callback, 10)
        self.flare_sequence_sub = self.create_subscription(
            String, '/auv/flare_sequence', self.flare_sequence_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odometry/filtered', self.odom_callback, 10)
        self.emergency_sub = self.create_subscription(
            Bool, '/safety/emergency_stop', self.emergency_callback, 10)
        
        # Publishers
        self.state_pub = self.create_publisher(String, '/mission/current_state', 10)
        self.task_pub = self.create_publisher(String, '/mission/current_task', 10)
        self.score_pub = self.create_publisher(Float32, '/mission/score', 10)
        self.recovery_pub = self.create_publisher(Bool, '/mission/recovery_mode', 10)
        
        # Timers
        self.save_timer = self.create_timer(self.save_interval, self.save_state)
        self.publish_timer = self.create_timer(0.5, self.publish_state)
        
        self.get_logger().info(f'Mission State Manager initialized. Current task: {self.state["current_task"]}')
        if self.auto_recover and self.state['attempt_count'] > 0:
            self.get_logger().warn(f'⚠️  Recovered from previous mission (attempt {self.state["attempt_count"]})')
    
    def load_state(self):
        """Load state from file"""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    loaded_state = json.load(f)
                    self.state.update(loaded_state)
                    self.get_logger().info(f'Loaded mission state: {self.state}')
            elif self.backup_file.exists():
                self.get_logger().warn('Primary state file missing, loading backup...')
                with open(self.backup_file, 'r') as f:
                    loaded_state = json.load(f)
                    self.state.update(loaded_state)
        except Exception as e:
            self.get_logger().error(f'Failed to load state: {e}')
    
    def save_state(self):
        """Persist current state to disk"""
        try:
            # Update mission time
            self.state['total_mission_time'] = time.time() - self.state['mission_start_time']
            
            # Create backup
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    backup_data = f.read()
                with open(self.backup_file, 'w') as f:
                    f.write(backup_data)
            
            # Write current state
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2)
                
        except Exception as e:
            self.get_logger().error(f'Failed to save state: {e}')
    
    def publish_state(self):
        """Publish current state to topics"""
        # Full state JSON
        state_msg = String()
        state_msg.data = json.dumps(self.state)
        self.state_pub.publish(state_msg)
        
        # Current task
        task_msg = String()
        task_msg.data = self.state['current_task']
        self.task_pub.publish(task_msg)
        
        # Score
        score_msg = Float32()
        score_msg.data = float(self.state['score'])
        self.score_pub.publish(score_msg)
        
        # Recovery mode
        recovery_msg = Bool()
        recovery_msg.data = self.state['attempt_count'] > 0
        self.recovery_pub.publish(recovery_msg)
    
    def task_update_callback(self, msg: String):
        """Handle task completion updates"""
        update = json.loads(msg.data)
        
        if 'task' in update:
            task_name = update['task']
            
            # Mark task as complete
            if task_name not in self.state['tasks_completed']:
                self.state['tasks_completed'].append(task_name)
                self.get_logger().info(f'✓ Task completed: {task_name}')
            
            # Update score
            if 'points' in update:
                self.state['score'] += update['points']
                self.get_logger().info(f'Score: {self.state["score"]} (+{update["points"]})')
            
            # Update current task
            if 'next_task' in update:
                self.state['current_task'] = update['next_task']
                self.get_logger().info(f'→ Next task: {update["next_task"]}')
        
        # Handle penalties
        if 'penalty' in update:
            self.state['penalties'] += update['penalty']
            self.state['score'] -= update['penalty']
            self.get_logger().warn(f'Penalty: -{update["penalty"]} points')
        
        self.save_state()
    
    def gate_passed_callback(self, msg: Bool):
        """Track gate passage"""
        if msg.data and not self.state['gate_passed']:
            self.state['gate_passed'] = True
            self.state['navigation_complete'] = True
            self.state['score'] += 15  # Navigation task points
            self.get_logger().info('✓ Gate passed! Navigation complete.')
            
            # Update task
            update = {
                'task': 'NAVIGATION',
                'points': 15,
                'next_task': MissionTask.FLARE_LOCALIZATION.name
            }
            task_msg = String()
            task_msg.data = json.dumps(update)
            self.task_update_callback(task_msg)
    
    def flare_sequence_callback(self, msg: String):
        """Receive flare sequence from surface"""
        sequence = msg.data.strip().split('-')
        self.state['flare_sequence'] = sequence
        self.get_logger().info(f'Received flare sequence: {"-".join(sequence)}')
        self.save_state()
    
    def flare_bumped_callback(self, msg: String):
        """Track flare bumping"""
        flare_color = msg.data.strip()
        
        if flare_color not in self.state['flares_bumped']:
            self.state['flares_bumped'].append(flare_color)
            self.state['score'] += 20  # 20 points per flare
            self.get_logger().info(f'✓ Flare bumped: {flare_color} (+20 points)')
            
            # Check if bumped in correct order
            if len(self.state['flare_sequence']) > 0:
                expected_index = len(self.state['flares_bumped']) - 1
                if expected_index < len(self.state['flare_sequence']):
                    expected = self.state['flare_sequence'][expected_index]
                    if flare_color == expected:
                        self.get_logger().info(f'  ✓ Correct order!')
                    else:
                        self.get_logger().warn(f'  ✗ Wrong order (expected {expected})')
            
            # Check if all 3 bumped in order
            if len(self.state['flares_bumped']) == 3 and \
               self.state['flares_bumped'] == self.state['flare_sequence']:
                self.state['flares_bumped_in_order'] = True
                self.state['score'] += 60  # Bonus for correct order
                self.get_logger().info('🎯 All flares bumped in correct order! (+60 bonus)')
            
            self.save_state()
    
    def odom_callback(self, msg: Odometry):
        """Track position for recovery"""
        pos = msg.pose.pose.position
        self.state['last_known_position'] = {
            'x': pos.x,
            'y': pos.y,
            'z': pos.z
        }
    
    def emergency_callback(self, msg: Bool):
        """Handle emergency surfacing"""
        if msg.data:
            self.state['emergency_surfaced'] = True
            self.state['current_task'] = MissionTask.SURFACE.name
            self.get_logger().error('Emergency surfacing triggered!')
            self.save_state()
    
    def reset_mission(self):
        """Reset mission state for new attempt"""
        self.state = {
            'current_task': MissionTask.SUBMERGE.name,
            'tasks_completed': [],
            'navigation_complete': False,
            'gate_passed': False,
            'flare_sequence': [],
            'flares_bumped': [],
            'flares_bumped_in_order': False,
            'attempt_count': 0,
            'retry_count': 0,
            'last_known_position': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'mission_start_time': time.time(),
            'total_mission_time': 0.0,
            'emergency_surfaced': False,
            'score': 0,
            'penalties': 0
        }
        self.save_state()
        self.get_logger().info('Mission state reset.')


def main(args=None):
    rclpy.init(args=args)
    node = MissionStateManagerEnhanced()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_state()  # Save on shutdown
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()