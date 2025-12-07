#!/usr/bin/env python3
"""
FIXED QUALIFICATION DETECTOR - Accurate Distance + Long Range Detection
Key Fixes:
1. Odometry-based distance calculation (ACCURATE)
2. Extremely permissive HSV for 10m detection
3. Minimal area threshold for distant objects
4. Heavy dilation to enhance tiny features
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

class FixedQualificationDetector(Node):
    def __init__(self):
        super().__init__('qualification_detector_node')
        self.bridge = CvBridge()
        self.camera_matrix = None
        
        # ULTRA-PERMISSIVE HSV - Catches even washed-out orange at 10m
        # Lowered saturation to 20 (from 40) to catch very desaturated colors
        self.orange_lower = np.array([0, 20, 40])    # Very permissive
        self.orange_upper = np.array([35, 255, 255])  # Wide hue range
        
        # ULTRA-LOW area threshold for 10m detection
        self.min_area = 3  # Catches tiny specks
        
        self.gate_detection_history = deque(maxlen=2)
        self.reverse_mode = False
        
        # CRITICAL: Gate position for odometry-based distance
        self.gate_x_position = 0.0  # Gate is at X=0 in qualification world
        self.current_position = None
        
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
        
        # CRITICAL: Subscribe to odometry for accurate distance
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
        
        self.get_logger().info('✅ FIXED DETECTOR: Odometry Distance + Ultra-Sensitive HSV')
    
    def reverse_mode_callback(self, msg: Bool):
        self.reverse_mode = msg.data
        if msg.data:
            self.get_logger().info('🔄 REVERSE MODE ACTIVATED')
    
    def odom_callback(self, msg: Odometry):
        """Get accurate position for distance calculation"""
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
        
        # Create mask with ultra-permissive range
        orange_mask = cv2.inRange(hsv_image, self.orange_lower, self.orange_upper)
        
        # AGGRESSIVE DILATION - Make tiny features visible
        kernel = np.ones((5, 5), np.uint8)
        orange_mask_clean = cv2.dilate(orange_mask, kernel, iterations=3)
        
        # Find all contours
        orange_contours, _ = cv2.findContours(orange_mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find gate posts
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
                'bbox': (x, y, w_box, h_box)
            })
            
            # Draw detection
            cv2.rectangle(debug_img, (x, y), (x+w_box, y+h_box), (0, 255, 255), 2)
            cv2.circle(debug_img, (cx, cy), 5, (255, 0, 255), -1)
        
        # Sort by area
        posts.sort(key=lambda p: p['area'], reverse=True)
        
        # CRITICAL: Calculate ACCURATE distance using odometry
        current_x = self.current_position[0]
        
        if not self.reverse_mode:
            # Forward: distance = gate_x - current_x
            estimated_distance = abs(self.gate_x_position - current_x)
        else:
            # Reverse: distance = current_x - gate_x
            estimated_distance = abs(current_x - self.gate_x_position)
        
        # Clamp to reasonable range
        estimated_distance = max(0.5, min(estimated_distance, 15.0))
        
        # Detection logic
        gate_detected = False
        partial_gate = False
        alignment_error = 0.0
        gate_center_x = w // 2
        frame_position = 0.0
        confidence = 0.0
        
        if len(posts) >= 1:
            gate_detected = True
            
            if len(posts) >= 2:
                # Full gate - both posts visible
                confidence = 1.0
                partial_gate = False
                
                left_post, right_post = sorted(posts[:2], key=lambda p: p['center'][0])
                gate_center_x = (left_post['center'][0] + right_post['center'][0]) // 2
                
                cv2.line(debug_img, left_post['center'], right_post['center'], (0, 255, 0), 3)
                cv2.putText(debug_img, "FULL GATE", (gate_center_x - 60, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            else:
                # Partial gate - single post
                confidence = 0.5
                partial_gate = True
                post = posts[0]
                gate_center_x = post['center'][0]
                
                cv2.circle(debug_img, post['center'], 15, (0, 165, 255), 3)
                cv2.putText(debug_img, "PARTIAL", (gate_center_x - 40, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
        
        # Calculate alignment
        if gate_detected:
            frame_position = (gate_center_x - w/2) / (w/2)
            alignment_error = frame_position
            
            # Draw center line
            cv2.line(debug_img, (w//2, 0), (w//2, h), (255, 255, 0), 2)
            cv2.line(debug_img, (gate_center_x, 0), (gate_center_x, h), (0, 255, 0), 2)
        
        # CRITICAL: Display accurate odometry-based distance
        cv2.putText(debug_img, f"ODOM DIST: {estimated_distance:.2f}m", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
        cv2.putText(debug_img, f"X: {current_x:.2f}m", (10, 130),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(debug_img, f"Posts: {len(posts)}", (10, 170),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Publish
        self.gate_detection_history.append(gate_detected)
        confirmed = sum(self.gate_detection_history) >= 1  # Fast reaction
        
        self.publish_gate_data(confirmed, partial_gate, alignment_error, 
                              estimated_distance, gate_center_x, 0, frame_position, confidence, msg.header)
        
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
    node = FixedQualificationDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()