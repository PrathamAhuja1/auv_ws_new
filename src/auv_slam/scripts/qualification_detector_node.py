#!/usr/bin/env python3
"""
FINAL FIXED Qualification Gate Detector
Key Fixes:
1. NO emergency straight mode
2. Center lock clears only after clearance
3. Proper re-locking for reverse pass
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Bool, Float32, String
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from collections import deque
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy


class QualificationDetector(Node):
    def __init__(self):
        super().__init__('qualification_gate_detector')
        self.bridge = CvBridge()
        self.camera_matrix = None
        
        # Orange HSV range for gate posts
        self.orange_lower = np.array([0, 20, 40])
        self.orange_upper = np.array([35, 255, 255])
        
        self.min_area = 3
        self.gate_detection_history = deque(maxlen=2)
        self.reverse_mode = False
        
        self.gate_x_position = 0.0
        self.current_position = None
        
        self.expected_gate_width = 1.5
        self.gate_width_tolerance = 0.3
        
        # Center locking
        self.gate_center_locked = False
        self.locked_center_x = None
        self.locked_center_y = None
        self.lock_confidence_threshold = 0.8
        self.lock_distance_threshold = 2.5
        self.center_lock_enabled = True
        
        self.center_history = deque(maxlen=5)
        self.center_smoothing_alpha = 0.3
        
        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1, 
            durability=DurabilityPolicy.VOLATILE
        )
        
        self.image_sub = self.create_subscription(
            Image, '/camera_forward/image_raw', self.image_callback, qos_sensor)
        
        self.cam_info_sub = self.create_subscription(
            CameraInfo, '/camera_forward/camera_info', self.cam_info_callback, 10)
        
        self.reverse_mode_sub = self.create_subscription(
            Bool, '/mission/reverse_mode', self.reverse_mode_callback, 10)
        
        self.clear_lock_sub = self.create_subscription(
            Bool, '/mission/clear_center_lock', self.clear_lock_callback, 10)
        
        self.odom_sub = self.create_subscription(
            Odometry, '/ground_truth/odom', self.odom_callback, 10)
        
        # Publishers
        self.gate_detected_pub = self.create_publisher(Bool, '/qualification/gate_detected', 10)
        self.alignment_pub = self.create_publisher(Float32, '/qualification/alignment_error', 10)
        self.distance_pub = self.create_publisher(Float32, '/qualification/estimated_distance', 10)
        self.confidence_pub = self.create_publisher(Float32, '/qualification/confidence', 10) 
        self.partial_gate_pub = self.create_publisher(Bool, '/qualification/partial_detection', 10)
        self.gate_center_pub = self.create_publisher(Point, '/qualification/center_point', 10)
        self.debug_pub = self.create_publisher(Image, '/qualification/debug_image', 10)
        self.status_pub = self.create_publisher(String, '/qualification/status', 10)
        self.frame_position_pub = self.create_publisher(Float32, '/qualification/frame_position', 10)
        
        self.get_logger().info('='*70)
        self.get_logger().info('✅ FINAL FIXED Qualification Detector')
        self.get_logger().info('   ✓ NO emergency straight mode')
        self.get_logger().info('   ✓ Lock clears after clearance only')
        self.get_logger().info('   ✓ Proper re-locking for reverse')
        self.get_logger().info('='*70)
    
    def reverse_mode_callback(self, msg: Bool):
        """Handle reverse mode activation"""
        was_reverse = self.reverse_mode
        self.reverse_mode = msg.data
        
        if not was_reverse and msg.data:
            # Entering reverse mode - clear lock and re-enable
            self.gate_center_locked = False
            self.locked_center_x = None
            self.locked_center_y = None
            self.center_lock_enabled = True
            self.center_history.clear()
            
            self.get_logger().info('='*70)
            self.get_logger().info('🔄 REVERSE MODE ACTIVATED')
            self.get_logger().info('   ✓ Center lock cleared')
            self.get_logger().info('   ✓ Locking enabled for reverse pass')
            self.get_logger().info('='*70)
    
    def clear_lock_callback(self, msg: Bool):
        """Handle center lock clear signal (after clearance)"""
        if msg.data:
            self.gate_center_locked = False
            self.locked_center_x = None
            self.locked_center_y = None
            self.center_lock_enabled = False
            self.center_history.clear()
            
            self.get_logger().info('='*70)
            self.get_logger().info('🚫 CENTER LOCK CLEARED (After Clearance)')
            self.get_logger().info('   ⚠️ Locking DISABLED for U-turn')
            self.get_logger().info('='*70)
    
    def odom_callback(self, msg: Odometry):
        self.current_position = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        )
    
    def cam_info_callback(self, msg: CameraInfo):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape((3, 3))
            self.image_width = msg.width
            self.image_height = msg.height
            self.fx = self.camera_matrix[0, 0]

    def image_callback(self, msg: Image):
        if self.camera_matrix is None or self.current_position is None:
            return
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        except CvBridgeError:
            return
        
        debug_img = cv_image.copy()
        h, w = cv_image.shape[:2]
        
        # Detect orange posts
        orange_mask = cv2.inRange(hsv_image, self.orange_lower, self.orange_upper)
        kernel = np.ones((5, 5), np.uint8)
        orange_mask_clean = cv2.dilate(orange_mask, kernel, iterations=3)
        
        orange_contours, _ = cv2.findContours(
            orange_mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        posts = []
        for cnt in orange_contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue
            
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            x, y, w_box, h_box = cv2.boundingRect(cnt)
            
            posts.append({
                'center': (cx, cy),
                'area': area,
                'bbox': (x, y, w_box, h_box),
                'x_pos': cx
            })
        
        # Estimate distance from odometry
        current_x = self.current_position[0]
        if not self.reverse_mode:
            estimated_distance = abs(self.gate_x_position - current_x)
        else:
            estimated_distance = abs(current_x - self.gate_x_position)
        estimated_distance = max(0.5, min(estimated_distance, 15.0))
        
        gate_detected = False
        partial_gate = False
        alignment_error = 0.0
        gate_center_x = w // 2
        gate_center_y = h // 2
        frame_position = 0.0
        confidence = 0.0
        
        # NO EMERGENCY STRAIGHT MODE - removed completely
        
        if self.gate_center_locked:
            # Use locked center
            gate_detected = True
            confidence = 1.0
            
            if len(self.center_history) > 0:
                smoothed_x = int(np.mean([c[0] for c in self.center_history]))
                smoothed_y = int(np.mean([c[1] for c in self.center_history]))
                gate_center_x = smoothed_x
                gate_center_y = smoothed_y
            else:
                gate_center_x = self.locked_center_x
                gate_center_y = self.locked_center_y
            
            frame_position = (gate_center_x - w/2) / (w/2)
            alignment_error = frame_position
            
            cv2.circle(debug_img, (gate_center_x, gate_center_y), 30, (0, 255, 255), -1)
            cv2.line(debug_img, (gate_center_x, 0), (gate_center_x, h), (0, 255, 255), 5)
            cv2.putText(debug_img, "CENTER LOCKED", 
                       (gate_center_x - 120, gate_center_y - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
        
        elif len(posts) >= 2:
            # Two posts detected
            posts_sorted = sorted(posts, key=lambda p: p['x_pos'])
            left_post = posts_sorted[0]
            right_post = posts_sorted[1]
            
            detected_center_x = (left_post['center'][0] + right_post['center'][0]) // 2
            detected_center_y = (left_post['center'][1] + right_post['center'][1]) // 2
            
            gate_detected = True
            partial_gate = False
            confidence = 1.0
            
            # Smooth center
            if len(self.center_history) > 0:
                prev_center = self.center_history[-1]
                gate_center_x = int(self.center_smoothing_alpha * detected_center_x + 
                                   (1 - self.center_smoothing_alpha) * prev_center[0])
                gate_center_y = int(self.center_smoothing_alpha * detected_center_y + 
                                   (1 - self.center_smoothing_alpha) * prev_center[1])
            else:
                gate_center_x = detected_center_x
                gate_center_y = detected_center_y
            
            self.center_history.append((gate_center_x, gate_center_y))
            
            # Lock if enabled and close enough
            if self.center_lock_enabled and not self.gate_center_locked:
                if estimated_distance < 3.5 and confidence >= 0.9:
                    self.gate_center_locked = True
                    self.locked_center_x = gate_center_x
                    self.locked_center_y = gate_center_y
                    self.get_logger().info(
                        f'🔒 CENTER LOCKED at {estimated_distance:.2f}m '
                        f'(reverse={self.reverse_mode})'
                    )
            
            if self.gate_center_locked:
                self.locked_center_x = gate_center_x
                self.locked_center_y = gate_center_y
            
            frame_position = (gate_center_x - w/2) / (w/2)
            alignment_error = frame_position
            
            # Draw detections
            cv2.circle(debug_img, (gate_center_x, gate_center_y), 25, (0, 255, 255), -1)
            cv2.line(debug_img, left_post['center'], right_post['center'], (0, 255, 0), 3)
            
            for post in [left_post, right_post]:
                x, y, w_b, h_b = post['bbox']
                cv2.rectangle(debug_img, (x, y), (x+w_b, y+h_b), (255, 0, 255), 3)
            
            cv2.putText(debug_img, "FULL GATE", 
                       (gate_center_x - 80, gate_center_y - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
        
        elif len(posts) == 1:
            # Single post
            gate_detected = True
            partial_gate = True
            confidence = 0.5
            
            post = posts[0]
            gate_center_x = post['center'][0]
            gate_center_y = post['center'][1]
            
            frame_position = (gate_center_x - w/2) / (w/2)
            alignment_error = frame_position
            
            x, y, w_b, h_b = post['bbox']
            cv2.rectangle(debug_img, (x, y), (x+w_b, y+h_b), (0, 165, 255), 3)
            cv2.circle(debug_img, (gate_center_x, gate_center_y), 20, (0, 165, 255), 3)
            cv2.putText(debug_img, "PARTIAL", 
                       (gate_center_x - 60, gate_center_y - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        else:
            # No posts detected
            gate_detected = False
        
        # Draw center line
        cv2.line(debug_img, (w//2, 0), (w//2, h), (255, 255, 0), 2)
        if gate_detected:
            cv2.line(debug_img, (gate_center_x, 0), (gate_center_x, h), (0, 255, 0), 3)
        
        # Status overlay
        status_y = 40
        cv2.putText(debug_img, f"Dist: {estimated_distance:.2f}m", 
                   (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        status_y += 35
        
        cv2.putText(debug_img, f"Posts: {len(posts)}", 
                   (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        status_y += 35
        
        lock_status = "LOCKED" if self.gate_center_locked else "UNLOCKED"
        lock_color = (0, 255, 0) if self.gate_center_locked else (100, 100, 100)
        cv2.putText(debug_img, f"Lock: {lock_status}", 
                   (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, lock_color, 2)
        status_y += 35
        
        if not self.center_lock_enabled:
            cv2.putText(debug_img, "LOCK DISABLED (U-TURN)", 
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            status_y += 35
        
        if self.reverse_mode:
            cv2.putText(debug_img, "MODE: REVERSE", 
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
            status_y += 35
        
        if gate_detected:
            status_color = (0, 255, 0) if not partial_gate else (0, 165, 255)
            cv2.putText(debug_img, f"Conf: {confidence:.2f} | Align: {alignment_error:+.3f}", 
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Publish data
        self.gate_detection_history.append(gate_detected)
        confirmed = sum(self.gate_detection_history) >= 1
        
        self.gate_detected_pub.publish(Bool(data=confirmed))
        self.partial_gate_pub.publish(Bool(data=partial_gate))
        self.confidence_pub.publish(Float32(data=confidence))
        
        if confirmed:
            self.alignment_pub.publish(Float32(data=float(alignment_error)))
            self.distance_pub.publish(Float32(data=float(estimated_distance)))
            self.frame_position_pub.publish(Float32(data=float(frame_position)))
            
            center_msg = Point()
            center_msg.x = float(gate_center_x)
            center_msg.y = float(gate_center_y)
            center_msg.z = float(estimated_distance)
            self.gate_center_pub.publish(center_msg)
        
        # Debug image
        try:
            self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug_img, "bgr8"))
        except:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = QualificationDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()