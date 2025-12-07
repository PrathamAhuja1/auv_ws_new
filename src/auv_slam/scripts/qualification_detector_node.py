#!/usr/bin/env python3
"""
FIXED QUALIFICATION DETECTOR - LONG RANGE SENSITIVITY
Fixes:
1. HSV Range: Lowered Saturation/Value to detect "washed out" orange at 10m
2. Area Threshold: Reduced to 5 pixels to catch tiny poles at start
3. Dilation: Aggressive thickening of distant objects
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Bool, Float32, String
from geometry_msgs.msg import Point
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from collections import deque
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

class QualificationGateDetector(Node):
    def __init__(self):
        super().__init__('qualification_detector_node')
        self.bridge = CvBridge()
        self.camera_matrix = None
        
        # FIXED: WIDER HSV RANGE for distant underwater orange
        # Distant orange looks grey/desaturated. Lowered S/V from 120 to 40/50.
        self.orange_lower = np.array([0, 40, 50])
        self.orange_upper = np.array([30, 255, 255])
        
        # FIXED: EXTREMELY LOW THRESHOLD for starting location (10m away)
        self.min_area_strict = 5    # Detects tiny specks
        self.min_area_relaxed = 2
        self.aspect_threshold = 1.0 # Relaxed for blobs
        self.gate_width = 1.5
        
        self.gate_detection_history = deque(maxlen=3)
        self.reverse_mode = False
        
        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1, durability=DurabilityPolicy.VOLATILE
        )
        
        self.image_sub = self.create_subscription(
            Image, '/camera_forward/image_raw', self.image_callback, qos_sensor)
        
        self.cam_info_sub = self.create_subscription(
            CameraInfo, '/camera_forward/camera_info', self.cam_info_callback, 10)
        
        self.reverse_mode_sub = self.create_subscription(
            Bool, '/mission/reverse_mode', self.reverse_mode_callback, 10)
        
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
        
        self.get_logger().info('✅ FIXED DETECTOR: High Sensitivity for 10m Detection')
    
    def reverse_mode_callback(self, msg: Bool):
        self.reverse_mode = msg.data
    
    def cam_info_callback(self, msg: CameraInfo):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape((3, 3))
            self.image_width = msg.width
            self.image_height = msg.height
            self.fx = self.camera_matrix[0, 0]

    def image_callback(self, msg: Image):
        if self.camera_matrix is None: return
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        except CvBridgeError: return
        
        debug_img = cv_image.copy()
        h, w = cv_image.shape[:2]
        
        # Create mask
        orange_mask = cv2.inRange(hsv_image, self.orange_lower, self.orange_upper)
        
        # FIXED: ONLY DILATE. NO EROSION.
        # Erosion removes pixels. We need to KEEP every pixel from 10m away.
        kernel = np.ones((3, 3), np.uint8)
        orange_mask_clean = cv2.dilate(orange_mask, kernel, iterations=2) # Inflate twice
        
        orange_contours, _ = cv2.findContours(orange_mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        posts = self.find_gate_posts(orange_contours, debug_img, w, h)
        
        gate_detected = False
        partial_gate = False
        alignment_error = 0.0
        estimated_distance = 999.0
        gate_center_x = w // 2
        frame_position = 0.0
        confidence = 0.0
        
        if len(posts) >= 1:
            # Even 1 post is enough to start approach
            gate_detected = True
            
            if len(posts) >= 2:
                # Full Gate
                confidence = 1.0
                left_post, right_post = sorted(posts[:2], key=lambda p: p['center'][0])
                gate_center_x = (left_post['center'][0] + right_post['center'][0]) // 2
                
                stripe_dist = abs(right_post['center'][0] - left_post['center'][0])
                if stripe_dist > 2: # Very lenient
                    estimated_distance = (self.gate_width * self.fx) / stripe_dist
                    estimated_distance = min(estimated_distance, 50.0)
                
                cv2.line(debug_img, left_post['center'], right_post['center'], (0, 255, 0), 2)
            else:
                # Partial Gate (1 post)
                partial_gate = True
                confidence = 0.5
                post = posts[0]
                
                # Assume center based on mode
                offset = w * 0.2
                if not self.reverse_mode:
                    # Forward: If post on left, gate is to right? 
                    # Actually at 10m, just aim for the post. It's fine.
                    gate_center_x = post['center'][0] 
                else:
                    gate_center_x = post['center'][0]
                
                estimated_distance = 8.0 # Guess
                cv2.circle(debug_img, post['center'], 10, (0, 165, 255), 2)

        # Output Logic
        if gate_detected:
            frame_position = (gate_center_x - w/2) / (w/2)
            alignment_error = frame_position
            cv2.putText(debug_img, f"DIST: {estimated_distance:.1f}", (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Publish
        self.gate_detection_history.append(gate_detected)
        # Fast reaction: 1 frame is enough if we are searching
        confirmed = sum(self.gate_detection_history) >= 1 
        
        self.publish_gate_data(confirmed, partial_gate, alignment_error, 
                              estimated_distance, gate_center_x, 0, frame_position, confidence, msg.header)
        
        try:
            self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug_img, "bgr8"))
        except: pass

    def find_gate_posts(self, contours, debug_img, img_w, img_h):
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area_strict: continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Draw raw detection
            cv2.rectangle(debug_img, (x,y), (x+w,y+h), (0, 0, 255), 1)
            
            M = cv2.moments(cnt)
            if M["m00"] == 0: continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            candidates.append({'center': (cx, cy), 'area': area})

        # Sort by largest area
        candidates.sort(key=lambda p: p['area'], reverse=True)
        return candidates[:2]

    def publish_gate_data(self, confirmed, partial, align, dist, cx, cy, frame_pos, conf, header):
        self.gate_detected_pub.publish(Bool(data=confirmed))
        self.partial_gate_pub.publish(Bool(data=partial))
        self.confidence_pub.publish(Float32(data=conf))
        if confirmed:
            self.alignment_pub.publish(Float32(data=float(align)))
            self.distance_pub.publish(Float32(data=float(dist)))
            self.frame_position_pub.publish(Float32(data=float(frame_pos)))

def main(args=None):
    rclpy.init(args=args)
    node = QualificationGateDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()