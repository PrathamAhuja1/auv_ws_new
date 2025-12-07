#!/usr/bin/env python3
"""
SAUVC Qualification Gate Detector
Detects orange-marked gate posts for qualification task
Handles both forward and reverse passes
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
        self.image_width = None
        self.image_height = None
        
        # HSV RANGES - ORANGE for qualification gate posts
        self.orange_lower = np.array([5, 120, 120])
        self.orange_upper = np.array([25, 255, 255])
        
        # Detection parameters
        self.min_area_strict = 200
        self.min_area_relaxed = 80
        self.aspect_threshold = 1.5  # Gate posts are tall and thin
        self.gate_width = 1.5
        
        self.gate_detection_history = deque(maxlen=3)
        self.min_confirmations = 1
        
        self.frame_count = 0
        self.reverse_mode = False
        
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
        
        self.reverse_mode_sub = self.create_subscription(
            Bool,
            '/mission/reverse_mode',
            self.reverse_mode_callback,
            10
        )
        
        # Publishers
        self.gate_detected_pub = self.create_publisher(Bool, '/gate/detected', 10)
        self.alignment_pub = self.create_publisher(Float32, '/gate/alignment_error', 10)
        self.distance_pub = self.create_publisher(Float32, '/gate/estimated_distance', 10)
        self.gate_center_pub = self.create_publisher(Point, '/gate/center_point', 10)
        self.debug_pub = self.create_publisher(Image, '/gate/debug_image', 10)
        self.status_pub = self.create_publisher(String, '/gate/status', 10)
        
        self.frame_position_pub = self.create_publisher(Float32, '/gate/frame_position', 10)
        self.confidence_pub = self.create_publisher(Float32, '/gate/detection_confidence', 10)
        self.partial_gate_pub = self.create_publisher(Bool, '/gate/partial_detection', 10)
        
        self.get_logger().info('✅ QUALIFICATION Gate Detector initialized')
        self.get_logger().info('   - Detecting ORANGE posts')
        self.get_logger().info('   - Supports forward and reverse passes')
    
    def reverse_mode_callback(self, msg: Bool):
        if self.reverse_mode != msg.data:
            self.reverse_mode = msg.data
            mode = "REVERSE" if self.reverse_mode else "FORWARD"
            self.get_logger().warn(f'🔄 MODE: {mode}')
    
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
        
        # Create orange mask
        orange_mask = cv2.inRange(hsv_image, self.orange_lower, self.orange_upper)
        
        # Morphological operations
        kernel = np.ones((3, 3), np.uint8)
        orange_mask_clean = cv2.morphologyEx(orange_mask, cv2.MORPH_CLOSE, kernel)
        orange_mask_clean = cv2.morphologyEx(orange_mask_clean, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        orange_contours, _ = cv2.findContours(orange_mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find gate posts
        posts = self.find_gate_posts(orange_contours, debug_img, w, h)
        
        # Gate detection logic
        gate_detected = False
        partial_gate = False
        alignment_error = 0.0
        estimated_distance = 999.0
        gate_center_x = w // 2
        gate_center_y = h // 2
        frame_position = 0.0
        confidence = 0.0
        
        if len(posts) == 2:
            # FULL GATE - Both posts visible
            gate_detected = True
            partial_gate = False
            confidence = 1.0
            
            left_post, right_post = sorted(posts, key=lambda p: p['center'][0])
            
            gate_center_x = (left_post['center'][0] + right_post['center'][0]) // 2
            gate_center_y = (left_post['center'][1] + right_post['center'][1]) // 2
            
            image_center_x = w / 2
            pixel_error = gate_center_x - image_center_x
            alignment_error = pixel_error / image_center_x
            
            stripe_distance = abs(right_post['center'][0] - left_post['center'][0])
            if stripe_distance > 20:
                estimated_distance = (self.gate_width * self.fx) / stripe_distance
                estimated_distance = max(0.5, min(estimated_distance, 50.0))
            
            frame_position = (gate_center_x - w/2) / (w/2)
            
            # Visualization
            cv2.circle(debug_img, (gate_center_x, gate_center_y), 25, (255, 0, 255), -1)
            cv2.line(debug_img, (gate_center_x, 0), (gate_center_x, h), (255, 0, 255), 4)
            cv2.line(debug_img, left_post['center'], right_post['center'], (0, 255, 255), 5)
            
            cv2.putText(debug_img, f"FULL GATE {estimated_distance:.1f}m", 
                       (gate_center_x - 100, gate_center_y - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 4)
        
        elif len(posts) == 1:
            # PARTIAL GATE - Only one post visible
            gate_detected = True
            partial_gate = True
            confidence = 0.5
            
            post = posts[0]
            post_x = post['center'][0]
            gate_center_y = post['center'][1]
            
            # Determine which side and infer gate center
            if not self.reverse_mode:
                # FORWARD MODE
                if post_x < w * 0.5:
                    # Left post visible
                    gate_center_x = int(post_x + w * 0.25)
                    side = "LEFT"
                    desired_x = w * 0.30
                else:
                    # Right post visible
                    gate_center_x = int(post_x - w * 0.25)
                    side = "RIGHT"
                    desired_x = w * 0.70
            else:
                # REVERSE MODE
                if post_x < w * 0.5:
                    gate_center_x = int(post_x + w * 0.25)
                    side = "LEFT (REV)"
                    desired_x = w * 0.30
                else:
                    gate_center_x = int(post_x - w * 0.25)
                    side = "RIGHT (REV)"
                    desired_x = w * 0.70
            
            pixel_error = post_x - desired_x
            alignment_error = pixel_error / (w / 2)
            frame_position = (gate_center_x - w/2) / (w/2)
            
            # Visualization
            cv2.circle(debug_img, (gate_center_x, gate_center_y), 30, (255, 165, 0), 5)
            cv2.putText(debug_img, f"PARTIAL: {side}", 
                       (gate_center_x - 150, gate_center_y - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 165, 0), 3)
        
        # Draw frame edges warning
        if gate_detected:
            if abs(frame_position) > 0.6:
                edge_warning = "NEAR EDGE!" if abs(frame_position) > 0.8 else "edge warning"
                color = (0, 0, 255) if abs(frame_position) > 0.8 else (0, 165, 255)
                cv2.putText(debug_img, edge_warning, (w//2 - 150, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        # Draw center line
        cv2.line(debug_img, (w//2, 0), (w//2, h), (0, 255, 255), 2)
        
        # Status overlay
        mode_text = "REVERSE" if self.reverse_mode else "FORWARD"
        status_lines = [
            f"Frame {self.frame_count}",
            f"Mode: {mode_text}",
            f"Posts: {len(posts)}",
            f"Confidence: {confidence:.2f}",
        ]
        
        if gate_detected:
            if partial_gate:
                status_lines.append(f"PARTIAL @ {frame_position:+.2f}")
            else:
                status_lines.append(f"FULL GATE {estimated_distance:.1f}m")
            status_lines.append(f"Align: {alignment_error:+.2f}")
        else:
            status_lines.append("SEARCHING")
        
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
                              frame_position, confidence, msg.header)
        
        # Publish debug image
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(debug_img, "bgr8")
            debug_msg.header = msg.header
            self.debug_pub.publish(debug_msg)
        except CvBridgeError as e:
            self.get_logger().error(f'Debug image error: {e}')
    
    def find_gate_posts(self, contours, debug_img, img_w, img_h):
        """Find the two most likely gate posts (orange, tall, thin)"""
        candidates = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area_relaxed:
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            if w == 0 or h == 0:
                continue
            
            aspect_ratio = float(h) / w
            
            # Gate posts should be tall and thin (aspect > 1.5)
            if aspect_ratio < self.aspect_threshold:
                continue
            
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Score based on area and aspect ratio
            score = area * aspect_ratio
            
            candidates.append({
                'center': (cx, cy),
                'bbox': (x, y, w, h),
                'area': area,
                'aspect': aspect_ratio,
                'score': score,
                'contour': cnt
            })
        
        # Sort by score and take top 2
        candidates.sort(key=lambda p: p['score'], reverse=True)
        posts = candidates[:2]
        
        # Draw detected posts
        for post in posts:
            cx, cy = post['center']
            x, y, w, h = post['bbox']
            
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 165, 255), 3)
            cv2.circle(debug_img, (cx, cy), 12, (0, 165, 255), -1)
            cv2.circle(debug_img, (cx, cy), 15, (255, 255, 255), 2)
            
            cv2.putText(debug_img, f"POST {int(post['area'])}", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        
        return posts
    
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