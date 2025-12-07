#!/usr/bin/env python3
"""
Qualification Gate Detector - Detects orange-marked gate posts
Designed for SAUVC qualification task
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
        super().__init__('qualification_gate_detector')
        self.bridge = CvBridge()
        self.camera_matrix = None
        self.image_width = None
        self.image_height = None
        
        # HSV RANGES for ORANGE gate posts
        self.orange_lower = np.array([10, 100, 100])
        self.orange_upper = np.array([25, 255, 255])
        
        # Detection parameters
        self.declare_parameter('min_area_strict', 200)
        self.declare_parameter('min_area_relaxed', 80)
        self.declare_parameter('aspect_threshold', 3.0)  # Vertical posts
        self.declare_parameter('gate_width_meters', 1.5)
        self.declare_parameter('detection_history_size', 5)
        self.declare_parameter('min_detections_for_confirm', 2)
        
        self.min_area_strict = self.get_parameter('min_area_strict').value
        self.min_area_relaxed = self.get_parameter('min_area_relaxed').value
        self.aspect_threshold = self.get_parameter('aspect_threshold').value
        self.gate_width = self.get_parameter('gate_width_meters').value
        self.detection_history_size = self.get_parameter('detection_history_size').value
        self.min_confirmations = self.get_parameter('min_detections_for_confirm').value
        
        self.gate_detection_history = deque(maxlen=self.detection_history_size)
        self.frame_count = 0
        
        # QoS profiles
        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE
        )
        
        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE
        )
        
        # Subscriptions
        self.image_sub = self.create_subscription(
            Image,
            '/camera_forward/image_raw',
            self.image_callback,
            qos_sensor
        )
        
        self.cam_info_sub = self.create_subscription(
            CameraInfo,
            '/camera_forward/camera_info',
            self.cam_info_callback,
            qos_reliable
        )
        
        # Publishers
        self.gate_detected_pub = self.create_publisher(Bool, '/qualification/gate_detected', 10)
        self.alignment_pub = self.create_publisher(Float32, '/qualification/alignment_error', 10)
        self.distance_pub = self.create_publisher(Float32, '/qualification/estimated_distance', 10)
        self.gate_center_pub = self.create_publisher(Point, '/qualification/gate_center', 10)
        self.debug_pub = self.create_publisher(Image, '/qualification/debug_image', 10)
        self.status_pub = self.create_publisher(String, '/qualification/gate_status', 10)
        self.confidence_pub = self.create_publisher(Float32, '/qualification/confidence', 10)
        self.partial_gate_pub = self.create_publisher(Bool, '/qualification/partial_detection', 10)
        
        self.get_logger().info('✅ Qualification Gate Detector initialized')
        self.get_logger().info('   - Detecting ORANGE gate posts')
        self.get_logger().info('   - Gate width: 1.5m')
    
    def cam_info_callback(self, msg: CameraInfo):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape((3, 3))
            self.image_width = msg.width
            self.image_height = msg.height
            self.fx = self.camera_matrix[0, 0]
            self.fy = self.camera_matrix[1, 1]
            self.cx = self.camera_matrix[0, 2]
            self.cy = self.camera_matrix[1, 2]
            self.get_logger().info(f'Camera initialized: {self.image_width}x{self.image_height}')
    
    def image_callback(self, msg: Image):
        if self.camera_matrix is None:
            return
        
        self.frame_count += 1
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        except CvBridgeError as e:
            self.get_logger().error(f'CV Bridge error: {e}')
            return
        
        debug_img = cv_image.copy()
        h, w = cv_image.shape[:2]
        
        # Create orange mask for gate posts
        orange_mask = cv2.inRange(hsv_image, self.orange_lower, self.orange_upper)
        
        # Morphological operations
        kernel = np.ones((5, 5), np.uint8)
        orange_mask_clean = cv2.morphologyEx(orange_mask, cv2.MORPH_CLOSE, kernel)
        orange_mask_clean = cv2.morphologyEx(orange_mask_clean, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        orange_contours, _ = cv2.findContours(orange_mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Detect gate posts
        left_post = self.find_best_post(orange_contours, debug_img, (255, 128, 0), 
                                        "LEFT", self.min_area_strict, strict=True)
        right_post = self.find_best_post(orange_contours, debug_img, (255, 165, 0), 
                                         "RIGHT", self.min_area_strict, strict=True)
        
        # Try relaxed detection if strict fails
        if not left_post:
            left_post = self.find_best_post(orange_contours, debug_img, (200, 100, 0), 
                                            "LEFT*", self.min_area_relaxed, strict=False)
        if not right_post:
            right_post = self.find_best_post(orange_contours, debug_img, (200, 130, 0), 
                                             "RIGHT*", self.min_area_relaxed, strict=False)
        
        # Gate detection logic
        gate_detected = False
        partial_gate = False
        alignment_error = 0.0
        estimated_distance = 999.0
        gate_center_x = w // 2
        gate_center_y = h // 2
        confidence = 0.0
        
        if left_post and right_post:
            # FULL GATE DETECTED
            gate_detected = True
            partial_gate = False
            confidence = 1.0
            
            gate_center_x = (left_post['center'][0] + right_post['center'][0]) // 2
            gate_center_y = (left_post['center'][1] + right_post['center'][1]) // 2
            
            # Calculate alignment error
            image_center_x = w / 2
            pixel_error = gate_center_x - image_center_x
            alignment_error = pixel_error / image_center_x
            
            # Estimate distance using known gate width
            post_distance = abs(left_post['center'][0] - right_post['center'][0])
            if post_distance > 30:
                estimated_distance = (self.gate_width * self.fx) / post_distance
                estimated_distance = max(0.5, min(estimated_distance, 50.0))
            
            # Visualization
            cv2.circle(debug_img, (gate_center_x, gate_center_y), 30, (0, 255, 0), -1)
            cv2.line(debug_img, (gate_center_x, 0), (gate_center_x, h), (0, 255, 0), 4)
            cv2.line(debug_img, left_post['center'], right_post['center'], (0, 255, 255), 5)
            
            cv2.putText(debug_img, f"FULL GATE {estimated_distance:.1f}m", 
                       (gate_center_x - 150, gate_center_y - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 4)
        
        elif left_post or right_post:
            # PARTIAL GATE DETECTED
            gate_detected = True
            partial_gate = True
            confidence = 0.5
            
            post = left_post if left_post else right_post
            post_name = "LEFT POST" if left_post else "RIGHT POST"
            
            gate_center_x = post['center'][0]
            gate_center_y = post['center'][1]
            
            # Calculate alignment based on single post
            image_center_x = w / 2
            if left_post:
                # Left post visible - should be at ~30% from left when centered
                desired_x = w * 0.30
                pixel_error = post['center'][0] - desired_x
            else:
                # Right post visible - should be at ~70% from left when centered
                desired_x = w * 0.70
                pixel_error = post['center'][0] - desired_x
            
            alignment_error = pixel_error / image_center_x
            estimated_distance = 999.0
            
            # Visualization
            cv2.circle(debug_img, (gate_center_x, gate_center_y), 30, (255, 165, 0), 5)
            cv2.putText(debug_img, f"PARTIAL: {post_name}", 
                       (gate_center_x - 150, gate_center_y - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 165, 0), 3)
        
        # Draw center line
        cv2.line(debug_img, (w//2, 0), (w//2, h), (0, 255, 255), 2)
        
        # Status overlay
        status_lines = [
            f"Frame {self.frame_count}",
            f"Confidence: {confidence:.2f}",
        ]
        
        if gate_detected:
            if partial_gate:
                status_lines.append("PARTIAL GATE")
            else:
                status_lines.append(f"FULL GATE {estimated_distance:.1f}m")
            status_lines.append(f"Align: {alignment_error:+.3f}")
        else:
            status_lines.append("NO GATE DETECTED")
        
        # Draw status box
        box_height = len(status_lines) * 35 + 20
        cv2.rectangle(debug_img, (5, 5), (450, box_height), (0, 0, 0), -1)
        box_color = (0, 255, 0) if (gate_detected and not partial_gate) else \
                    (255, 165, 0) if partial_gate else (100, 100, 100)
        cv2.rectangle(debug_img, (5, 5), (450, box_height), box_color, 3)
        
        for i, line in enumerate(status_lines):
            cv2.putText(debug_img, line, (15, 35 + i*35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Temporal filtering
        self.gate_detection_history.append(gate_detected)
        confirmed_gate = sum(self.gate_detection_history) >= self.min_confirmations
        
        # Publish all data
        self.publish_gate_data(confirmed_gate, partial_gate, alignment_error, 
                              estimated_distance, gate_center_x, gate_center_y, 
                              confidence, msg.header)
        
        # Publish debug image
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(debug_img, "bgr8")
            debug_msg.header = msg.header
            self.debug_pub.publish(debug_msg)
        except CvBridgeError as e:
            self.get_logger().error(f'Debug image error: {e}')
    
    def find_best_post(self, contours, debug_img, color, label, min_area, strict=True):
        """Find best gate post (vertical orange stripe)"""
        if not contours:
            return None
        
        best_post = None
        best_score = 0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            if w == 0:
                continue
            
            aspect_ratio = float(h) / w
            
            # Check if at image edge
            image_width = debug_img.shape[1] if debug_img is not None else 1280
            at_edge = (x < 50 or (x + w) > image_width - 50)
            
            score = area
            
            if strict:
                # Strict mode: must be tall and vertical
                if aspect_ratio > self.aspect_threshold or at_edge:
                    score *= 2.0
                else:
                    continue
            else:
                # Relaxed mode
                if aspect_ratio > self.aspect_threshold or at_edge:
                    score *= 1.5
            
            if score > best_score:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    best_post = {
                        'center': (cx, cy),
                        'bbox': (x, y, w, h),
                        'area': area,
                        'aspect': aspect_ratio,
                        'score': score,
                        'at_edge': at_edge
                    }
                    best_score = score
        
        if best_post and debug_img is not None:
            cx, cy = best_post['center']
            x, y, w, h = best_post['bbox']
            
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), color, 3)
            cv2.circle(debug_img, (cx, cy), 15, color, -1)
            cv2.circle(debug_img, (cx, cy), 17, (255, 255, 255), 2)
            
            label_text = f"{label} A:{int(best_post['area'])}"
            if best_post['at_edge']:
                label_text += " @EDGE"
            cv2.putText(debug_img, label_text, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return best_post
    
    def publish_gate_data(self, confirmed_gate, partial_gate, alignment_error, 
                         estimated_distance, center_x, center_y, confidence, header):
        """Publish all gate detection data"""
        self.gate_detected_pub.publish(Bool(data=confirmed_gate))
        self.partial_gate_pub.publish(Bool(data=partial_gate))
        self.confidence_pub.publish(Float32(data=confidence))
        
        if confirmed_gate:
            self.alignment_pub.publish(Float32(data=float(alignment_error)))
            self.distance_pub.publish(Float32(data=float(estimated_distance)))
            
            center_msg = Point()
            center_msg.x = float(center_x)
            center_msg.y = float(center_y)
            center_msg.z = float(estimated_distance)
            self.gate_center_pub.publish(center_msg)
            
            status = f"{'PARTIAL' if partial_gate else 'FULL'} | Conf:{confidence:.2f} | Dist:{estimated_distance:.1f}m"
            self.status_pub.publish(String(data=status))


def main(args=None):
    rclpy.init(args=args)
    node = QualificationGateDetector()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()