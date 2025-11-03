#!/usr/bin/env python3
"""
Qualification Gate Detector
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
        
        # HSV RANGE for ORANGE qualification gate
        self.orange_lower = np.array([5, 120, 100])
        self.orange_upper = np.array([25, 255, 255])
        
        # Detection parameters
        self.min_area = 200
        self.gate_width_meters = 1.5  # 150cm per rulebook
        
        # History for temporal filtering
        self.detection_history = deque(maxlen=5)
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
        self.gate_detected_pub = self.create_publisher(Bool, '/qual_gate/detected', 10)
        self.alignment_pub = self.create_publisher(Float32, '/qual_gate/alignment_error', 10)
        self.distance_pub = self.create_publisher(Float32, '/qual_gate/estimated_distance', 10)
        self.gate_center_pub = self.create_publisher(Point, '/qual_gate/center_point', 10)
        self.debug_pub = self.create_publisher(Image, '/qual_gate/debug_image', 10)
        self.status_pub = self.create_publisher(String, '/qual_gate/status', 10)
        self.confidence_pub = self.create_publisher(Float32, '/qual_gate/confidence', 10)
        
        self.get_logger().info('✅ Qualification Gate Detector initialized')
        self.get_logger().info('   - Detecting ORANGE gate (150cm wide)')
    
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
        kernel = np.ones((5, 5), np.uint8)
        orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_CLOSE, kernel)
        orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find gate posts (two largest orange regions)
        gate_detected = False
        alignment_error = 0.0
        estimated_distance = 999.0
        gate_center_x = w // 2
        gate_center_y = h // 2
        confidence = 0.0
        
        if len(contours) >= 2:
            # Sort by area
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            # Get two largest
            post1 = sorted_contours[0]
            post2 = sorted_contours[1]
            
            area1 = cv2.contourArea(post1)
            area2 = cv2.contourArea(post2)
            
            if area1 > self.min_area and area2 > self.min_area:
                # Calculate centers
                M1 = cv2.moments(post1)
                M2 = cv2.moments(post2)
                
                if M1["m00"] > 0 and M2["m00"] > 0:
                    cx1 = int(M1["m10"] / M1["m00"])
                    cy1 = int(M1["m01"] / M1["m00"])
                    cx2 = int(M2["m10"] / M2["m00"])
                    cy2 = int(M2["m01"] / M2["m00"])
                    
                    gate_detected = True
                    confidence = 1.0
                    
                    # Calculate gate center
                    gate_center_x = (cx1 + cx2) // 2
                    gate_center_y = (cy1 + cy2) // 2
                    
                    # Alignment error
                    image_center_x = w / 2
                    pixel_error = gate_center_x - image_center_x
                    alignment_error = pixel_error / image_center_x
                    
                    # Distance estimation
                    pixel_distance = abs(cx1 - cx2)
                    if pixel_distance > 20:
                        estimated_distance = (self.gate_width_meters * self.fx) / pixel_distance
                        estimated_distance = max(0.5, min(estimated_distance, 50.0))
                    
                    # Draw on debug image
                    cv2.circle(debug_img, (cx1, cy1), 15, (0, 165, 255), -1)
                    cv2.circle(debug_img, (cx2, cy2), 15, (0, 165, 255), -1)
                    cv2.circle(debug_img, (gate_center_x, gate_center_y), 25, (255, 0, 255), -1)
                    cv2.line(debug_img, (cx1, cy1), (cx2, cy2), (0, 255, 255), 5)
                    cv2.line(debug_img, (gate_center_x, 0), (gate_center_x, h), (255, 0, 255), 3)
                    
                    # Draw bounding boxes
                    x1, y1, w1, h1 = cv2.boundingRect(post1)
                    x2, y2, w2, h2 = cv2.boundingRect(post2)
                    cv2.rectangle(debug_img, (x1, y1), (x1+w1, y1+h1), (0, 165, 255), 3)
                    cv2.rectangle(debug_img, (x2, y2), (x2+w2, y2+h2), (0, 165, 255), 3)
                    
                    cv2.putText(debug_img, f"QUAL GATE {estimated_distance:.1f}m", 
                               (gate_center_x - 120, gate_center_y - 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        
        # Draw center line
        cv2.line(debug_img, (w//2, 0), (w//2, h), (0, 255, 255), 2)
        
        # Status overlay
        status_lines = [
            f"Frame {self.frame_count}",
            f"Confidence: {confidence:.2f}",
        ]
        
        if gate_detected:
            status_lines.append(f"GATE DETECTED {estimated_distance:.1f}m")
            status_lines.append(f"Align: {alignment_error:+.2f}")
        else:
            status_lines.append("SEARCHING")
        
        # Draw status box
        box_height = len(status_lines) * 35 + 20
        cv2.rectangle(debug_img, (5, 5), (450, box_height), (0, 0, 0), -1)
        box_color = (0, 255, 0) if gate_detected else (100, 100, 100)
        cv2.rectangle(debug_img, (5, 5), (450, box_height), box_color, 3)
        
        for i, line in enumerate(status_lines):
            cv2.putText(debug_img, line, (15, 35 + i*35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Temporal filtering
        self.detection_history.append(gate_detected)
        confirmed_gate = sum(self.detection_history) >= self.min_confirmations
        
        # Publish data
        self.gate_detected_pub.publish(Bool(data=confirmed_gate))
        self.confidence_pub.publish(Float32(data=confidence))
        
        if confirmed_gate:
            self.alignment_pub.publish(Float32(data=float(alignment_error)))
            self.distance_pub.publish(Float32(data=float(estimated_distance)))
            
            center_msg = Point()
            center_msg.x = float(gate_center_x)
            center_msg.y = float(gate_center_y)
            center_msg.z = float(estimated_distance)
            self.gate_center_pub.publish(center_msg)
            
            status = f"DETECTED | Dist:{estimated_distance:.1f}m | Align:{alignment_error:+.2f}"
            self.status_pub.publish(String(data=status))
        
        # Publish debug image
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(debug_img, "bgr8")
            debug_msg.header = msg.header
            self.debug_pub.publish(debug_msg)
        except CvBridgeError as e:
            self.get_logger().error(f'Debug image error: {e}')


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