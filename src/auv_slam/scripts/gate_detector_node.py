#!/usr/bin/env python3
"""
FIXED SAUVC Gate Detector - Gazebo Optimized
Key fixes for detection issues:
1. ULTRA-WIDE HSV ranges (Gazebo lighting is different from real world)
2. Adaptive area thresholds based on camera distance
3. Better contour filtering (accepts partial stripes)
4. Enhanced debug output showing what's being detected
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


class GateDetectorNode(Node):
    def __init__(self):
        super().__init__('gate_detector_node')
        self.bridge = CvBridge()
        self.camera_matrix = None
        self.image_width = None
        self.image_height = None
        
        # CRITICAL FIX: Ultra-wide HSV ranges for Gazebo
        # Red stripes - captures bright red in simulation
        self.red_lower1 = np.array([0, 20, 50])      # Very permissive low-H red
        self.red_upper1 = np.array([25, 255, 255])   # Extended upper bound
        self.red_lower2 = np.array([145, 20, 50])    # Very permissive high-H red  
        self.red_upper2 = np.array([180, 255, 255])
        
        # Green stripes - captures all green tones
        self.green_lower = np.array([25, 15, 40])    # Very permissive
        self.green_upper = np.array([110, 255, 255]) # Extended range
        
        # Orange flare - permissive to catch obstacle
        self.orange_lower = np.array([3, 20, 40])
        self.orange_upper = np.array([40, 255, 255])
        
        # Detection parameters - RELAXED for distance
        self.min_area = 15  # Even smaller for 12m distance
        self.aspect_threshold = 1.0  # Vertical stripes: height > width
        self.gate_width = 1.5
        
        # Temporal filtering - faster response
        self.gate_detection_history = deque(maxlen=3)
        self.min_confirmations = 1  # Accept immediately for faster response
        
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
        self.gate_detected_pub = self.create_publisher(Bool, '/gate/detected', 10)
        self.alignment_pub = self.create_publisher(Float32, '/gate/alignment_error', 10)
        self.distance_pub = self.create_publisher(Float32, '/gate/estimated_distance', 10)
        self.gate_center_pub = self.create_publisher(Point, '/gate/center_point', 10)
        self.debug_pub = self.create_publisher(Image, '/gate/debug_image', 10)
        self.status_pub = self.create_publisher(String, '/gate/status', 10)
        
        self.frame_count = 0
        
        self.get_logger().info('='*70)
        self.get_logger().info('🔍 FIXED Gate Detector initialized')
        self.get_logger().info('   Ultra-wide HSV ranges for Gazebo simulation')
        self.get_logger().info('   Min area: {} pixels'.format(self.min_area))
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
            self.get_logger().info(f'📷 Camera: {self.image_width}x{self.image_height}, fx={self.fx:.1f}')
    
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
        
        # ENHANCED DETECTION: Look for ANY colored regions
        red_mask1 = cv2.inRange(hsv_image, self.red_lower1, self.red_upper1)
        red_mask2 = cv2.inRange(hsv_image, self.red_lower2, self.red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        green_mask = cv2.inRange(hsv_image, self.green_lower, self.green_upper)
        
        # Show masks in debug
        red_pixels = cv2.countNonZero(red_mask)
        green_pixels = cv2.countNonZero(green_mask)
        
        # Morphological operations - GENTLE to preserve small features
        kernel = np.ones((3, 3), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find best red and green regions
        red_stripe = self.find_best_stripe(red_contours, debug_img, (0, 0, 255), "RED")
        green_stripe = self.find_best_stripe(green_contours, debug_img, (0, 255, 0), "GREEN")
        
        # Gate detection logic
        gate_detected = False
        alignment_error = 0.0
        estimated_distance = 999.0
        gate_center_x = self.image_width // 2
        gate_center_y = self.image_height // 2
        
        status_text = f"Frame {self.frame_count}"
        
        if red_stripe and green_stripe:
            # GATE DETECTED!
            gate_detected = True
            
            # Calculate gate center
            gate_center_x = (red_stripe['center'][0] + green_stripe['center'][0]) // 2
            gate_center_y = (red_stripe['center'][1] + green_stripe['center'][1]) // 2
            
            # Alignment error (-1 = left, +1 = right)
            image_center_x = self.image_width / 2
            pixel_error = gate_center_x - image_center_x
            alignment_error = pixel_error / image_center_x
            
            # Distance estimation
            stripe_distance = abs(red_stripe['center'][0] - green_stripe['center'][0])
            if stripe_distance > 10:
                estimated_distance = (self.gate_width * self.fx) / stripe_distance
            else:
                estimated_distance = 999.0
            
            # Draw gate visualization
            cv2.circle(debug_img, (gate_center_x, gate_center_y), 15, (255, 0, 255), -1)
            cv2.line(debug_img, (gate_center_x, 0), (gate_center_x, self.image_height), 
                    (255, 0, 255), 3)
            cv2.line(debug_img, red_stripe['center'], green_stripe['center'], (255, 255, 0), 3)
            
            status_text = f"GATE FOUND! Dist={estimated_distance:.1f}m Align={alignment_error:+.2f}"
            
        elif red_stripe or green_stripe:
            status_text = f"PARTIAL: Red={red_stripe is not None} Green={green_stripe is not None}"
        else:
            status_text = f"SEARCHING: R_px={red_pixels} G_px={green_pixels}"
        
        # Draw image center line
        cv2.line(debug_img, (self.image_width//2, 0), 
                (self.image_width//2, self.image_height), (0, 255, 255), 2)
        
        # Status overlay
        cv2.putText(debug_img, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(debug_img, f"Red pixels: {red_pixels}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(debug_img, f"Green pixels: {green_pixels}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Temporal filtering
        self.gate_detection_history.append(gate_detected)
        confirmed_gate = sum(self.gate_detection_history) >= self.min_confirmations
        
        # Publish results
        gate_msg = Bool()
        gate_msg.data = confirmed_gate
        self.gate_detected_pub.publish(gate_msg)
        
        if confirmed_gate:
            align_msg = Float32()
            align_msg.data = float(alignment_error)
            self.alignment_pub.publish(align_msg)
            
            dist_msg = Float32()
            dist_msg.data = float(estimated_distance)
            self.distance_pub.publish(dist_msg)
            
            center_msg = Point()
            center_msg.x = float(gate_center_x)
            center_msg.y = float(gate_center_y)
            center_msg.z = float(estimated_distance)
            self.gate_center_pub.publish(center_msg)
        
        # Publish status
        status_msg = String()
        status_msg.data = status_text
        self.status_pub.publish(status_msg)
        
        # Publish debug image
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(debug_img, "bgr8")
            debug_msg.header = msg.header
            self.debug_pub.publish(debug_msg)
        except CvBridgeError as e:
            pass
        
        # Log periodically
        if self.frame_count % 30 == 0:
            self.get_logger().info(status_text)
    
    def find_best_stripe(self, contours, debug_img, color, label):
        """Find the largest valid stripe (relaxed criteria)"""
        if not contours:
            return None
        
        best_stripe = None
        best_area = 0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            # CRITICAL FIX: Accept smaller contours for distance
            if area < self.min_area:
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            if w == 0:
                continue
            
            aspect_ratio = float(h) / w
            
            # RELAXED: Accept if somewhat vertical OR just large
            if (aspect_ratio > self.aspect_threshold) or (area > 100):
                if area > best_area:
                    M = cv2.moments(cnt)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        best_stripe = {
                            'center': (cx, cy),
                            'bbox': (x, y, w, h),
                            'area': area,
                            'aspect': aspect_ratio
                        }
                        best_area = area
        
        # Draw visualization
        if best_stripe and debug_img is not None:
            cv2.circle(debug_img, best_stripe['center'], 8, color, -1)
            x, y, w, h = best_stripe['bbox']
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(debug_img, f"{label} {int(best_stripe['area'])}", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return best_stripe


def main(args=None):
    rclpy.init(args=args)
    node = GateDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()