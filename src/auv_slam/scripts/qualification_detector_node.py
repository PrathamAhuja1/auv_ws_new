#!/usr/bin/env python3
"""
ROBUST QUALIFICATION DETECTOR - Fixed Gate Center Detection

KEY FIXES:
1. Finds TWO leftmost posts (both orange posts of gate)
2. Calculates center between the TWO POSTS, not just one
3. Uses geometric constraints to reject false positives
4. Validates gate structure (posts should be ~1.5m apart)
5. Provides warning when only seeing one post (partial view)
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
import math
from collections import deque
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

class RobustQualificationDetector(Node):
    def __init__(self):
        super().__init__('qualification_detector_node')
        self.bridge = CvBridge()
        self.camera_matrix = None
        
        # ULTRA-PERMISSIVE HSV for orange gate posts
        self.orange_lower = np.array([0, 20, 40])
        self.orange_upper = np.array([35, 255, 255])
        
        self.min_area = 3
        self.gate_detection_history = deque(maxlen=2)
        self.reverse_mode = False
        
        # Gate position (X=0 in qualification world)
        self.gate_x_position = 0.0
        self.current_position = None
        
        # CRITICAL: Expected gate width for validation
        self.expected_gate_width = 1.5  # meters
        self.gate_width_tolerance = 0.3  # ±30cm tolerance
        
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
        
        self.get_logger().info('✅ ROBUST DETECTOR: Fixed gate center calculation')
    
    def reverse_mode_callback(self, msg: Bool):
        self.reverse_mode = msg.data
        if msg.data:
            self.get_logger().info('🔄 REVERSE MODE ACTIVATED')
    
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
        
        # Create orange mask
        orange_mask = cv2.inRange(hsv_image, self.orange_lower, self.orange_upper)
        
        # Aggressive dilation for distant detection
        kernel = np.ones((5, 5), np.uint8)
        orange_mask_clean = cv2.dilate(orange_mask, kernel, iterations=3)
        
        # Find contours
        orange_contours, _ = cv2.findContours(orange_mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Extract all potential gate posts
        posts = []
        for cnt in orange_contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue
            
            x, y, w_box, h_box = cv2.boundingRect(cnt)
            
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            posts.append({
                'center': (cx, cy),
                'area': area,
                'bbox': (x, y, w_box, h_box),
                'x_pos': cx  # For sorting left-to-right
            })
        
        # Calculate accurate distance using odometry
        current_x = self.current_position[0]
        
        if not self.reverse_mode:
            estimated_distance = abs(self.gate_x_position - current_x)
        else:
            estimated_distance = abs(current_x - self.gate_x_position)
        
        estimated_distance = max(0.5, min(estimated_distance, 15.0))
        
        # ================================================================
        # CRITICAL FIX: Proper gate center detection
        # ================================================================
        
        gate_detected = False
        partial_gate = False
        alignment_error = 0.0
        gate_center_x = w // 2
        gate_center_y = h // 2
        frame_position = 0.0
        confidence = 0.0
        
        if len(posts) >= 2:
            # Sort posts by X-position (left to right)
            posts_sorted = sorted(posts, key=lambda p: p['x_pos'])
            
            # Take the TWO LEFTMOST posts (should be the gate posts)
            left_post = posts_sorted[0]
            right_post = posts_sorted[1]
            
            # CRITICAL: Calculate center between the TWO POSTS
            gate_center_x = (left_post['center'][0] + right_post['center'][0]) // 2
            gate_center_y = (left_post['center'][1] + right_post['center'][1]) // 2
            
            # Calculate pixel distance between posts
            pixel_separation = abs(right_post['center'][0] - left_post['center'][0])
            
            # Validate this is actually the gate using expected width
            # At distance D, gate width W should span (W * fx / D) pixels
            if estimated_distance > 0.5 and estimated_distance < 15.0:
                expected_pixel_width = (self.expected_gate_width * self.fx) / estimated_distance
                width_ratio = pixel_separation / expected_pixel_width
                
                # Check if separation matches expected gate width (±50% tolerance)
                if 0.5 < width_ratio < 1.5:
                    # VALID GATE DETECTION
                    gate_detected = True
                    partial_gate = False
                    confidence = 1.0
                    
                    # Draw gate center and connection line
                    cv2.circle(debug_img, (gate_center_x, gate_center_y), 25, (0, 255, 255), -1)
                    cv2.line(debug_img, left_post['center'], right_post['center'], (0, 255, 0), 3)
                    
                    # Draw posts with labels
                    cv2.rectangle(debug_img, 
                                (left_post['bbox'][0], left_post['bbox'][1]),
                                (left_post['bbox'][0] + left_post['bbox'][2], 
                                 left_post['bbox'][1] + left_post['bbox'][3]),
                                (255, 0, 255), 3)
                    cv2.putText(debug_img, "LEFT POST", 
                              (left_post['center'][0] - 50, left_post['center'][1] - 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                    
                    cv2.rectangle(debug_img,
                                (right_post['bbox'][0], right_post['bbox'][1]),
                                (right_post['bbox'][0] + right_post['bbox'][2],
                                 right_post['bbox'][1] + right_post['bbox'][3]),
                                (255, 0, 255), 3)
                    cv2.putText(debug_img, "RIGHT POST",
                              (right_post['center'][0] - 50, right_post['center'][1] - 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                    
                    cv2.putText(debug_img, f"FULL GATE - CENTER LOCKED",
                              (gate_center_x - 150, gate_center_y - 50),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
                    
                    cv2.putText(debug_img, f"Sep: {pixel_separation}px (exp: {expected_pixel_width:.0f}px)",
                              (10, h - 60),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                else:
                    # Posts too close/far - likely false positive
                    self.get_logger().warn(
                        f'Posts rejected: sep={pixel_separation}px, expected={expected_pixel_width:.0f}px, '
                        f'ratio={width_ratio:.2f}',
                        throttle_duration_sec=1.0
                    )
                    
                    # Draw rejected posts
                    for post in posts_sorted[:2]:
                        cv2.rectangle(debug_img,
                                    (post['bbox'][0], post['bbox'][1]),
                                    (post['bbox'][0] + post['bbox'][2],
                                     post['bbox'][1] + post['bbox'][3]),
                                    (0, 0, 255), 2)
                        cv2.putText(debug_img, "REJECTED",
                                  (post['center'][0] - 40, post['center'][1]),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
        elif len(posts) == 1:
            # PARTIAL GATE - Only one post visible
            gate_detected = True
            partial_gate = True
            confidence = 0.5
            
            post = posts[0]
            gate_center_x = post['center'][0]
            gate_center_y = post['center'][1]
            
            # Draw single post
            cv2.rectangle(debug_img,
                        (post['bbox'][0], post['bbox'][1]),
                        (post['bbox'][0] + post['bbox'][2],
                         post['bbox'][1] + post['bbox'][3]),
                        (0, 165, 255), 3)
            cv2.circle(debug_img, (gate_center_x, gate_center_y), 20, (0, 165, 255), 3)
            cv2.putText(debug_img, "PARTIAL GATE - 1 POST",
                       (gate_center_x - 100, gate_center_y - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            cv2.putText(debug_img, "CENTERING ON VISIBLE POST",
                       (gate_center_x - 150, gate_center_y + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Calculate alignment if gate detected
        if gate_detected:
            frame_position = (gate_center_x - w/2) / (w/2)
            alignment_error = frame_position
            
            # Draw center line and gate line
            cv2.line(debug_img, (w//2, 0), (w//2, h), (255, 255, 0), 2)
            cv2.line(debug_img, (gate_center_x, 0), (gate_center_x, h), (0, 255, 0), 3)
            
            # Draw alignment arrow
            arrow_start = (w//2, h//2)
            arrow_end = (gate_center_x, h//2)
            cv2.arrowedLine(debug_img, arrow_start, arrow_end, (255, 0, 255), 3, tipLength=0.3)
        
        # Status overlay
        cv2.putText(debug_img, f"ODOM DIST: {estimated_distance:.2f}m",
                   (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(debug_img, f"X: {current_x:.2f}m",
                   (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(debug_img, f"Posts: {len(posts)}",
                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if gate_detected:
            status_color = (0, 255, 0) if not partial_gate else (0, 165, 255)
            status_text = "FULL GATE" if not partial_gate else "PARTIAL"
            cv2.putText(debug_img, f"{status_text} | Conf: {confidence:.2f}",
                       (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            cv2.putText(debug_img, f"Align: {alignment_error:+.3f}",
                       (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        else:
            cv2.putText(debug_img, "SEARCHING...",
                       (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)
        
        # Publish
        self.gate_detection_history.append(gate_detected)
        confirmed = sum(self.gate_detection_history) >= 1
        
        self.publish_gate_data(confirmed, partial_gate, alignment_error, 
                              estimated_distance, gate_center_x, gate_center_y, 
                              frame_position, confidence, msg.header)
        
        try:
            self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug_img, "bgr8"))
        except:
            pass

    def publish_gate_data(self, confirmed, partial, align, dist, cx, cy, frame_pos, conf, header):
        self.gate_detected_pub.publish(Bool(data=confirmed))
        self.partial_gate_pub.publish(Bool(data=partial))
        self.confidence_pub.publish(Float32(data=conf))
        
        if confirmed:
            self.alignment_pub.publish(Float32(data=float(align)))
            self.distance_pub.publish(Float32(data=float(dist)))
            self.frame_position_pub.publish(Float32(data=float(frame_pos)))
            
            center_msg = Point()
            center_msg.x = float(cx)
            center_msg.y = float(cy)
            center_msg.z = float(dist)
            self.gate_center_pub.publish(center_msg)

def main(args=None):
    rclpy.init(args=args)
    node = RobustQualificationDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()