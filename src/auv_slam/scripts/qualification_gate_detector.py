#!/usr/bin/env python3
"""
Qualification Gate Detector - Detects ORANGE markers on qualification gate
Per SAUVC rulebook: Gate has orange markings on both port and starboard sides
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
        
        # HSV RANGES for ORANGE qualification gate markers
        # Per rulebook: Orange markings on both sides
        self.orange_lower = np.array([10, 120, 120])
        self.orange_upper = np.array([25, 255, 255])
        
        # Detection parameters
        self.min_area_strict = 400
        self.min_area_relaxed = 100
        self.aspect_threshold = 1.5  # Orange posts are vertical
        self.gate_width = 1.5  # 150cm per rulebook
        
        # Temporal filtering
        self.gate_detection_history = deque(maxlen=5)
        self.min_confirmations = 2
        
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
        self.status_pub = self.create_publisher(String, '/qualification/status', 10)
        self.frame_position_pub = self.create_publisher(Float32, '/qualification/frame_position', 10)
        self.confidence_pub = self.create_publisher(Float32, '/qualification/confidence', 10)
        self.partial_gate_pub = self.create_publisher(Bool, '/qualification/partial_detection', 10)
        
        self.get_logger().info('='*70)
        self.get_logger().info('✅ Qualification Gate Detector initialized')
        self.get_logger().info('   - Detecting ORANGE markers on qualification gate')
        self.get_logger().info('   - Gate dimensions: 150cm wide x 100cm deep')
        self.get_logger().info('='*70)
    
    def cam_info_callback(self, msg: CameraInfo):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape((3, 3))
            self.image_width = msg.width
            self.image_height = msg.height
            self.fx = self.camera_matrix[0, 0]
            self.fy = self.camera_matrix[1, 1]
            self.cx = self.camera_matrix[0, 2]
            self.cy = self.camera_matrix[1, 2]
            self.get_logger().info(f'Camera: {self.image_width}x{self.image_height}')
    
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
        
        # Create ORANGE mask
        orange_mask = cv2.inRange(hsv_image, self.orange_lower, self.orange_upper)
        
        # Morphological operations to clean up mask
        kernel = np.ones((5, 5), np.uint8)
        orange_mask_clean = cv2.morphologyEx(orange_mask, cv2.MORPH_CLOSE, kernel)
        orange_mask_clean = cv2.morphologyEx(orange_mask_clean, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        orange_contours, _ = cv2.findContours(orange_mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Detect orange posts
        left_post = self.find_best_post(orange_contours, debug_img, (0, 140, 255), 
                                        "LEFT", self.min_area_strict, strict=True)
        right_post = self.find_best_post(orange_contours, debug_img, (0, 140, 255), 
                                          "RIGHT", self.min_area_strict, strict=True)
        
        # Relaxed detection if strict didn't work
        if not left_post:
            left_post = self.find_best_post(orange_contours, debug_img, (0, 100, 200), 
                                            "LEFT*", self.min_area_relaxed, strict=False)
        if not right_post:
            right_post = self.find_best_post(orange_contours, debug_img, (0, 100, 200), 
                                             "RIGHT*", self.min_area_relaxed, strict=False)
        
        # Gate detection logic
        gate_detected = False
        partial_gate = False
        alignment_error = 0.0
        estimated_distance = 999.0
        gate_center_x = w // 2
        gate_center_y = h // 2
        frame_position = 0.0
        confidence = 0.0
        
        if left_post and right_post:
            # FULL GATE DETECTED
            gate_detected = True
            partial_gate = False
            confidence = 1.0
            
            gate_center_x = (left_post['center'][0] + right_post['center'][0]) // 2
            gate_center_y = (left_post['center'][1] + right_post['center'][1]) // 2
            
            image_center_x = w / 2
            pixel_error = gate_center_x - image_center_x
            alignment_error = pixel_error / image_center_x
            
            post_distance = abs(right_post['center'][0] - left_post['center'][0])
            if post_distance > 20:
                estimated_distance = (self.gate_width * self.fx) / post_distance
                estimated_distance = max(0.5, min(estimated_distance, 50.0))
            
            frame_position = (gate_center_x - w/2) / (w/2)
            
            # Visualization
            cv2.circle(debug_img, (gate_center_x, gate_center_y), 25, (0, 255, 255), -1)
            cv2.line(debug_img, (gate_center_x, 0), (gate_center_x, h), (0, 255, 255), 4)
            cv2.line(debug_img, left_post['center'], right_post['center'], (0, 255, 0), 5)
            
            cv2.putText(debug_img, f"QUAL GATE {estimated_distance:.1f}m", 
                       (gate_center_x - 150, gate_center_y - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 4)
        
        elif left_post or right_post:
            # PARTIAL GATE DETECTED
            gate_detected = True
            partial_gate = True
            confidence = 0.5
            
            post = left_post if left_post else right_post
            post_name = "LEFT (PORT)" if left_post else "RIGHT (STARBOARD)"
            
            gate_center_x = post['center'][0]
            gate_center_y = post['center'][1]
            
            image_center_x = w / 2
            
            if left_post:
                # Left post visible - gate center should be to the right
                desired_x = w * 0.30
                pixel_error = post['center'][0] - desired_x
            else:
                # Right post visible - gate center should be to the left
                desired_x = w * 0.70
                pixel_error = post['center'][0] - desired_x
            
            alignment_error = pixel_error / image_center_x
            estimated_distance = 999.0
            frame_position = (gate_center_x - w/2) / (w/2)
            
            # Visualization
            cv2.circle(debug_img, (gate_center_x, gate_center_y), 30, (0, 165, 255), 5)
            cv2.putText(debug_img, f"PARTIAL: {post_name}", 
                       (gate_center_x - 150, gate_center_y - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)
        
        # Edge warning
        if gate_detected and abs(frame_position) > 0.6:
            edge_warning = "NEAR EDGE!" if abs(frame_position) > 0.8 else "edge warning"
            color = (0, 0, 255) if abs(frame_position) > 0.8 else (0, 165, 255)
            cv2.putText(debug_img, edge_warning, (w//2 - 150, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        # Draw center line
        cv2.line(debug_img, (w//2, 0), (w//2, h), (0, 255, 255), 2)
        
        # Status overlay
        status_lines = [
            f"QUALIFICATION Frame {self.frame_count}",
            f"Confidence: {confidence:.2f}",
        ]
        
        if gate_detected:
            if partial_gate:
                status_lines.append(f"PARTIAL @ {frame_position:+.2f}")
            else:
                status_lines.append(f"FULL GATE {estimated_distance:.1f}m")
            status_lines.append(f"Align: {alignment_error:+.2f}")
        else:
            status_lines.append("SEARCHING FOR ORANGE MARKERS")
        
        # Draw status box
        box_height = len(status_lines) * 35 + 20
        cv2.rectangle(debug_img, (5, 5), (500, box_height), (0, 0, 0), -1)
        box_color = (0, 255, 0) if (gate_detected and not partial_gate) else \
                    (255, 165, 0) if partial_gate else (100, 100, 100)
        cv2.rectangle(debug_img, (5, 5), (500, box_height), box_color, 3)
        
        for i, line in enumerate(status_lines):
            cv2.putText(debug_img, line, (15, 35 + i*35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Temporal filtering
        self.gate_detection_history.append(gate_detected)
        confirmed_gate = sum(self.gate_detection_history) >= self.min_confirmations
        
        # Publish all data
        self.publish_gate_data(confirmed_gate, partial_gate, alignment_error, 
                              estimated_distance, gate_center_x, gate_center_y, 
                              frame_position, confidence, msg.header)
        
        # Publish debug image
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(debug_img, "bgr8")
            debug_msg.header = msg.header
            self.debug_pub.publish(debug_msg)
        except CvBridgeError as e:
            self.get_logger().error(f'Debug image error: {e}')
    
    def find_best_post(self, contours, debug_img, color, label, min_area, strict=True):
        """Find best orange post with improved edge handling"""
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
                # Vertical posts (aspect > 1.5)
                if aspect_ratio > self.aspect_threshold or at_edge:
                    score *= 2.0
                else:
                    continue
            else:
                if aspect_ratio > self.aspect_threshold or at_edge:
                    score *= 1.5
            
            # Bonus for posts near expected positions
            center_x = x + w/2
            # Left post should be around 35% from left
            # Right post should be around 65% from left
            if "LEFT" in label:
                target_x = image_width * 0.35
            else:
                target_x = image_width * 0.65
            
            distance_from_target = abs(center_x - target_x) / image_width
            if distance_from_target < 0.2:
                score *= 1.3
            
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
            
            label_text = f"{label} {int(best_post['area'])}"
            if best_post['at_edge']:
                label_text += " @EDGE"
            cv2.putText(debug_img, label_text, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return best_post
    
    def publish_gate_data(self, confirmed_gate, partial_gate, alignment_error, 
                         estimated_distance, center_x, center_y, frame_position, 
                         confidence, header):
        """Publish all gate detection data"""
        self.gate_detected_pub.publish(Bool(data=confirmed_gate))
        self.partial_gate_pub.publish(Bool(data=partial_gate))
        self.confidence_pub.publish(Float32(data=confidence))
        
        if confirmed_gate:
            self.alignment_pub.publish(Float32(data=float(alignment_error)))
            self.distance_pub.publish(Float32(data=float(estimated_distance)))
            self.frame_position_pub.publish(Float32(data=float(frame_position)))
            
            center_msg = Point()
            center_msg.x = float(center_x)
            center_msg.y = float(center_y)
            center_msg.z = float(estimated_distance)
            self.gate_center_pub.publish(center_msg)
            
            status = f"{'PARTIAL' if partial_gate else 'FULL'} | Pos:{frame_position:+.2f} | Conf:{confidence:.2f}"
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