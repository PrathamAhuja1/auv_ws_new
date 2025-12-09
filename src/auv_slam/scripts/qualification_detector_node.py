#!/usr/bin/env python3
"""
FIXED QUALIFICATION DETECTOR - Stable Center Locking

KEY FIXES:
1. Center locking mechanism - once gate center found, maintain it
2. Distance-based detection modes (far, medium, close)
3. Emergency straight mode when gate lost at close range
4. No geometric validation when very close to gate
5. Exponential moving average for center smoothing
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

class StableCenterDetector(Node):
    def __init__(self):
        super().__init__('qualification_detector_node')
        self.bridge = CvBridge()
        self.camera_matrix = None
        
        self.orange_lower = np.array([0, 20, 40])
        self.orange_upper = np.array([35, 255, 255])
        
        self.min_area = 3
        self.gate_detection_history = deque(maxlen=2)
        self.reverse_mode = False
        
        self.gate_x_position = 0.0
        self.current_position = None
        
        self.expected_gate_width = 1.5
        self.gate_width_tolerance = 0.3
        
        self.gate_center_locked = False
        self.locked_center_x = None
        self.locked_center_y = None
        self.lock_confidence_threshold = 0.8
        self.lock_distance_threshold = 2.5
        
        self.center_history = deque(maxlen=5)
        self.center_smoothing_alpha = 0.3
        
        self.emergency_straight_mode = False
        self.emergency_trigger_distance = 1.5
        
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
        
        self.gate_detected_pub = self.create_publisher(Bool, '/qualification/gate_detected', 10)
        self.alignment_pub = self.create_publisher(Float32, '/qualification/alignment_error', 10)
        self.distance_pub = self.create_publisher(Float32, '/qualification/estimated_distance', 10)
        self.confidence_pub = self.create_publisher(Float32, '/qualification/confidence', 10) 
        self.partial_gate_pub = self.create_publisher(Bool, '/qualification/partial_detection', 10)
        self.gate_center_pub = self.create_publisher(Point, '/qualification/center_point', 10)
        self.debug_pub = self.create_publisher(Image, '/qualification/debug_image', 10)
        self.status_pub = self.create_publisher(String, '/qualification/status', 10)
        self.frame_position_pub = self.create_publisher(Float32, '/qualification/frame_position', 10)
        
        self.get_logger().info('Stable Center Detector initialized')
        self.get_logger().info('Center locking enabled for close-range stability')
    
    def reverse_mode_callback(self, msg: Bool):
        self.reverse_mode = msg.data
        if msg.data:
            self.gate_center_locked = False
            self.locked_center_x = None
            self.locked_center_y = None
            self.emergency_straight_mode = False
            self.get_logger().info('REVERSE MODE - Reset center lock')
    
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
        
        orange_mask = cv2.inRange(hsv_image, self.orange_lower, self.orange_upper)
        
        kernel = np.ones((5, 5), np.uint8)
        orange_mask_clean = cv2.dilate(orange_mask, kernel, iterations=3)
        
        orange_contours, _ = cv2.findContours(orange_mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
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
                'x_pos': cx
            })
        
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
        
        if estimated_distance < self.emergency_trigger_distance and len(posts) == 0 and not self.gate_center_locked:
            if not self.emergency_straight_mode:
                self.emergency_straight_mode = True
                self.get_logger().warn('EMERGENCY: Gate lost at close range - proceeding straight')
        
        if self.emergency_straight_mode and not self.gate_center_locked:
            gate_detected = True
            partial_gate = False
            confidence = 0.5
            gate_center_x = w // 2
            gate_center_y = h // 2
            alignment_error = 0.0
            frame_position = 0.0
            
            cv2.putText(debug_img, "EMERGENCY STRAIGHT MODE", 
                       (w//2 - 200, h//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            
        elif self.gate_center_locked:
            gate_detected = True
            partial_gate = False
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
            cv2.putText(debug_img, f"Dist: {estimated_distance:.2f}m", 
                       (gate_center_x - 80, gate_center_y + 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            if len(posts) == 0 and estimated_distance < 1.5:
                cv2.putText(debug_img, "POSTS OUT OF FRAME - USING LOCK", 
                           (gate_center_x - 200, gate_center_y + 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
        elif len(posts) >= 2:
            posts_sorted = sorted(posts, key=lambda p: p['x_pos'])
            
            left_post = posts_sorted[0]
            right_post = posts_sorted[1]
            
            detected_center_x = (left_post['center'][0] + right_post['center'][0]) // 2
            detected_center_y = (left_post['center'][1] + right_post['center'][1]) // 2
            
            pixel_separation = abs(right_post['center'][0] - left_post['center'][0])
            
            skip_validation = estimated_distance < 2.0
            
            if skip_validation:
                valid_gate = True
                width_ratio = 1.0
            else:
                if estimated_distance > 0.5 and estimated_distance < 15.0:
                    expected_pixel_width = (self.expected_gate_width * self.fx) / estimated_distance
                    width_ratio = pixel_separation / expected_pixel_width
                    valid_gate = 0.5 < width_ratio < 1.5
                else:
                    valid_gate = True
                    width_ratio = 1.0
            
            if valid_gate:
                gate_detected = True
                partial_gate = False
                confidence = 1.0
                
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
                
                if not self.gate_center_locked and estimated_distance < 3.5 and confidence >= 0.9:
                    self.gate_center_locked = True
                    self.locked_center_x = gate_center_x
                    self.locked_center_y = gate_center_y
                    self.get_logger().info(f'CENTER LOCKED at {estimated_distance:.2f}m')
                
                if self.gate_center_locked:
                    self.locked_center_x = gate_center_x
                    self.locked_center_y = gate_center_y
                
                frame_position = (gate_center_x - w/2) / (w/2)
                alignment_error = frame_position
                
                cv2.circle(debug_img, (gate_center_x, gate_center_y), 25, (0, 255, 255), -1)
                cv2.line(debug_img, left_post['center'], right_post['center'], (0, 255, 0), 3)
                
                cv2.rectangle(debug_img, 
                            (left_post['bbox'][0], left_post['bbox'][1]),
                            (left_post['bbox'][0] + left_post['bbox'][2], 
                             left_post['bbox'][1] + left_post['bbox'][3]),
                            (255, 0, 255), 3)
                cv2.putText(debug_img, "LEFT", 
                          (left_post['center'][0] - 30, left_post['center'][1] - 20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                
                cv2.rectangle(debug_img,
                            (right_post['bbox'][0], right_post['bbox'][1]),
                            (right_post['bbox'][0] + right_post['bbox'][2],
                             right_post['bbox'][1] + right_post['bbox'][3]),
                            (255, 0, 255), 3)
                cv2.putText(debug_img, "RIGHT",
                          (right_post['center'][0] - 30, right_post['center'][1] - 20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                
                cv2.putText(debug_img, f"FULL GATE - CENTERED",
                          (gate_center_x - 150, gate_center_y - 50),
                          cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
                
                if not skip_validation:
                    cv2.putText(debug_img, f"Sep: {pixel_separation}px (ratio: {width_ratio:.2f})",
                              (10, h - 60),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
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
            gate_detected = True
            partial_gate = True
            confidence = 0.5
            
            post = posts[0]
            gate_center_x = post['center'][0]
            gate_center_y = post['center'][1]
            
            frame_position = (gate_center_x - w/2) / (w/2)
            alignment_error = frame_position
            
            cv2.rectangle(debug_img,
                        (post['bbox'][0], post['bbox'][1]),
                        (post['bbox'][0] + post['bbox'][2],
                         post['bbox'][1] + post['bbox'][3]),
                        (0, 165, 255), 3)
            cv2.circle(debug_img, (gate_center_x, gate_center_y), 20, (0, 165, 255), 3)
            cv2.putText(debug_img, "PARTIAL - 1 POST",
                       (gate_center_x - 100, gate_center_y - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        
        if gate_detected:
            cv2.line(debug_img, (w//2, 0), (w//2, h), (255, 255, 0), 2)
            cv2.line(debug_img, (gate_center_x, 0), (gate_center_x, h), (0, 255, 0), 3)
            
            arrow_start = (w//2, h//2)
            arrow_end = (gate_center_x, h//2)
            cv2.arrowedLine(debug_img, arrow_start, arrow_end, (255, 0, 255), 3, tipLength=0.3)
        
        cv2.putText(debug_img, f"ODOM: {estimated_distance:.2f}m",
                   (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(debug_img, f"X: {current_x:.2f}m",
                   (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(debug_img, f"Posts: {len(posts)}",
                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if self.gate_center_locked:
            cv2.putText(debug_img, "LOCK: ON",
                       (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        if gate_detected:
            status_color = (0, 255, 0) if not partial_gate else (0, 165, 255)
            status_text = "FULL" if not partial_gate else "PARTIAL"
            cv2.putText(debug_img, f"{status_text} | Conf: {confidence:.2f}",
                       (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            cv2.putText(debug_img, f"Align: {alignment_error:+.3f}",
                       (10, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        else:
            cv2.putText(debug_img, "SEARCHING",
                       (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)
        
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
    node = StableCenterDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()