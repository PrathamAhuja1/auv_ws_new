#!/usr/bin/env python3
"""
ULTRA-FIXED Gate Detector - Comprehensive Debug + Gazebo Optimized
Key improvements:
1. MAXIMUM HSV tolerance for Gazebo's lighting
2. Multi-stage fallback detection (relaxed criteria if nothing found)
3. Rich debug visualization showing EXACTLY what's being detected
4. Pixel-level analysis logging
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
        
        # ULTRA-PERMISSIVE HSV ranges for Gazebo (catches ANYTHING remotely red/green)
        # RED - catches orange, pink, red, maroon
        self.red_lower1 = np.array([0, 15, 30])      # Ultra-low saturation
        self.red_upper1 = np.array([30, 255, 255])   # Extended to orange
        self.red_lower2 = np.array([140, 15, 30])    # High H red
        self.red_upper2 = np.array([180, 255, 255])
        
        # GREEN - catches yellow-green, pure green, cyan-green
        self.green_lower = np.array([20, 10, 25])    # Minimal saturation
        self.green_upper = np.array([120, 255, 255]) # Extended range
        
        # ORANGE flare - very permissive
        self.orange_lower = np.array([0, 10, 30])
        self.orange_upper = np.array([45, 255, 255])
        
        # Detection parameters - ULTRA-RELAXED
        self.min_area_strict = 15        # For confident detection
        self.min_area_relaxed = 5        # For desperate search
        self.aspect_threshold = 0.5      # Very permissive (h/w > 0.5)
        self.gate_width = 1.5
        
        # Temporal filtering
        self.gate_detection_history = deque(maxlen=3)
        self.min_confirmations = 1
        
        # Debug counters
        self.frame_count = 0
        self.detection_stats = {
            'red_pixels': 0,
            'green_pixels': 0,
            'red_contours': 0,
            'green_contours': 0,
            'valid_red': 0,
            'valid_green': 0
        }
        
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
        
        self.get_logger().info('='*70)
        self.get_logger().info('🔍 ULTRA-FIXED Gate Detector initialized')
        self.get_logger().info('   Maximum HSV tolerance for Gazebo')
        self.get_logger().info('   Multi-stage fallback detection enabled')
        self.get_logger().info('   Publishing to /gate/debug_image for rqt_image_view')
        self.get_logger().info('='*70)
        self.get_logger().info('To view: rqt_image_view /gate/debug_image')
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
            self.get_logger().info(f'📷 Camera initialized: {self.image_width}x{self.image_height}')
    
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
        
        # Create comprehensive debug image
        debug_img = cv_image.copy()
        h, w = cv_image.shape[:2]
        
        # Create side-by-side comparison for masks
        mask_viz = np.zeros((h, w*3, 3), dtype=np.uint8)
        mask_viz[:, 0:w] = cv_image  # Original
        
        # STAGE 1: Try strict detection
        red_mask1 = cv2.inRange(hsv_image, self.red_lower1, self.red_upper1)
        red_mask2 = cv2.inRange(hsv_image, self.red_lower2, self.red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        green_mask = cv2.inRange(hsv_image, self.green_lower, self.green_upper)
        
        # Count pixels
        red_pixels = cv2.countNonZero(red_mask)
        green_pixels = cv2.countNonZero(green_mask)
        self.detection_stats['red_pixels'] = red_pixels
        self.detection_stats['green_pixels'] = green_pixels
        
        # Morphological operations
        kernel_small = np.ones((3, 3), np.uint8)
        kernel_large = np.ones((5, 5), np.uint8)
        
        red_mask_clean = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel_small)
        red_mask_clean = cv2.morphologyEx(red_mask_clean, cv2.MORPH_OPEN, kernel_small)
        green_mask_clean = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel_small)
        green_mask_clean = cv2.morphologyEx(green_mask_clean, cv2.MORPH_OPEN, kernel_small)
        
        # Add masks to visualization
        mask_viz[:, w:2*w, 2] = red_mask_clean  # Red channel
        mask_viz[:, 2*w:3*w, 1] = green_mask_clean  # Green channel
        
        # Find contours
        red_contours, _ = cv2.findContours(red_mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        green_contours, _ = cv2.findContours(green_mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        self.detection_stats['red_contours'] = len(red_contours)
        self.detection_stats['green_contours'] = len(green_contours)
        
        # Try strict criteria first
        red_stripe = self.find_best_stripe(red_contours, debug_img, (0, 0, 255), 
                                           "RED", self.min_area_strict, strict=True)
        green_stripe = self.find_best_stripe(green_contours, debug_img, (0, 255, 0), 
                                             "GREEN", self.min_area_strict, strict=True)
        
        # STAGE 2: Fallback to relaxed criteria
        if not red_stripe:
            red_stripe = self.find_best_stripe(red_contours, debug_img, (100, 0, 200), 
                                               "RED*", self.min_area_relaxed, strict=False)
        if not green_stripe:
            green_stripe = self.find_best_stripe(green_contours, debug_img, (0, 200, 100), 
                                                 "GRN*", self.min_area_relaxed, strict=False)
        
        self.detection_stats['valid_red'] = 1 if red_stripe else 0
        self.detection_stats['valid_green'] = 1 if green_stripe else 0
        
        # Gate detection logic
        gate_detected = False
        alignment_error = 0.0
        estimated_distance = 999.0
        gate_center_x = self.image_width // 2
        gate_center_y = self.image_height // 2
        
        if red_stripe and green_stripe:
            gate_detected = True
            
            # Calculate gate center
            gate_center_x = (red_stripe['center'][0] + green_stripe['center'][0]) // 2
            gate_center_y = (red_stripe['center'][1] + green_stripe['center'][1]) // 2
            
            # Alignment error
            image_center_x = self.image_width / 2
            pixel_error = gate_center_x - image_center_x
            alignment_error = pixel_error / image_center_x
            
            # Distance estimation
            stripe_distance = abs(red_stripe['center'][0] - green_stripe['center'][0])
            if stripe_distance > 10:
                estimated_distance = (self.gate_width * self.fx) / stripe_distance
                estimated_distance = max(0.5, min(estimated_distance, 50.0))  # Clamp
            
            # Draw gate visualization
            cv2.circle(debug_img, (gate_center_x, gate_center_y), 20, (255, 0, 255), -1)
            cv2.line(debug_img, (gate_center_x, 0), (gate_center_x, self.image_height), 
                    (255, 0, 255), 4)
            cv2.line(debug_img, red_stripe['center'], green_stripe['center'], (0, 255, 255), 4)
            
            # Draw distance arc
            cv2.putText(debug_img, f"{estimated_distance:.1f}m", 
                       (gate_center_x - 40, gate_center_y - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
        
        # Draw image center reference
        cv2.line(debug_img, (self.image_width//2, 0), 
                (self.image_width//2, self.image_height), (0, 255, 255), 3)
        
        # Comprehensive status overlay
        status_lines = [
            f"Frame {self.frame_count}",
            f"Red: {red_pixels}px ({len(red_contours)} contours)",
            f"Green: {green_pixels}px ({len(green_contours)} contours)",
            f"Valid: R={self.detection_stats['valid_red']} G={self.detection_stats['valid_green']}",
        ]
        
        if gate_detected:
            status_lines.append(f"GATE FOUND!")
            status_lines.append(f"Dist={estimated_distance:.1f}m Align={alignment_error:+.2f}")
        elif red_stripe or green_stripe:
            status_lines.append(f"PARTIAL DETECTION")
        else:
            status_lines.append(f"SEARCHING...")
        
        # Draw status box
        box_height = len(status_lines) * 35 + 20
        cv2.rectangle(debug_img, (5, 5), (450, box_height), (0, 0, 0), -1)
        cv2.rectangle(debug_img, (5, 5), (450, box_height), (0, 255, 0), 2)
        
        for i, line in enumerate(status_lines):
            color = (0, 255, 0) if gate_detected else (255, 255, 0) if (red_stripe or green_stripe) else (255, 255, 255)
            cv2.putText(debug_img, line, (15, 35 + i*35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Add HSV range info at bottom
        info_text = f"HSV: R[{self.red_lower1[0]}-{self.red_upper1[0]}] G[{self.green_lower[0]}-{self.green_upper[0]}]"
        cv2.putText(debug_img, info_text, (10, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
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
        
        # Publish status string
        status_msg = String()
        status_msg.data = " | ".join(status_lines)
        self.status_pub.publish(status_msg)
        
        # Publish debug image
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(debug_img, "bgr8")
            debug_msg.header = msg.header
            self.debug_pub.publish(debug_msg)
        except CvBridgeError as e:
            self.get_logger().error(f'Debug image publish error: {e}')
        
        # Periodic logging
        if self.frame_count % 30 == 0:
            self.get_logger().info(
                f"📊 Stats: R={red_pixels}px G={green_pixels}px | "
                f"Contours: R={len(red_contours)} G={len(green_contours)} | "
                f"Gate: {'✓' if confirmed_gate else '✗'}"
            )
    
    def find_best_stripe(self, contours, debug_img, color, label, min_area, strict=True):
        """Find best stripe with configurable criteria"""
        if not contours:
            return None
        
        best_stripe = None
        best_score = 0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            if area < min_area:
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            if w == 0:
                continue
            
            aspect_ratio = float(h) / w
            
            # Scoring system
            score = area
            
            if strict:
                # Strict mode: prefer vertical stripes
                if aspect_ratio > self.aspect_threshold:
                    score *= 2.0
                else:
                    continue  # Skip non-vertical in strict mode
            else:
                # Relaxed mode: accept anything with reasonable area
                if aspect_ratio > self.aspect_threshold:
                    score *= 1.5
            
            if score > best_score:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    best_stripe = {
                        'center': (cx, cy),
                        'bbox': (x, y, w, h),
                        'area': area,
                        'aspect': aspect_ratio,
                        'score': score
                    }
                    best_score = score
        
        # Draw visualization
        if best_stripe and debug_img is not None:
            cx, cy = best_stripe['center']
            x, y, w, h = best_stripe['bbox']
            
            # Draw bounding box
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), color, 3)
            
            # Draw center point
            cv2.circle(debug_img, (cx, cy), 12, color, -1)
            cv2.circle(debug_img, (cx, cy), 14, (255, 255, 255), 2)
            
            # Draw label with stats
            label_text = f"{label} A={int(best_stripe['area'])} AR={best_stripe['aspect']:.1f}"
            cv2.putText(debug_img, label_text, (x, y-15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return best_stripe


def main(args=None):
    rclpy.init(args=args)
    node = GateDetectorNode()
    
    node.get_logger().info('='*70)
    node.get_logger().info('🚀 Gate detector ready!')
    node.get_logger().info('   View debug feed: rqt_image_view /gate/debug_image')
    node.get_logger().info('='*70)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()