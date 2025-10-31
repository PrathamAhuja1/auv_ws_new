#!/usr/bin/env python3
"""
IMPROVED Gate Detector - Handles Partial Views & Provides Better Tracking
Key improvements:
1. Detects partial gates (single stripe visible)
2. Publishes frame position for better navigation
3. Provides confidence metrics
4. Better handling of edge cases
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


class ImprovedGateDetector(Node):
    def __init__(self):
        super().__init__('gate_detector_node')
        self.bridge = CvBridge()
        self.camera_matrix = None
        self.image_width = None
        self.image_height = None
        
        # HSV RANGES - Pure red and green for Gazebo
        self.red_lower1 = np.array([0, 100, 100])
        self.red_upper1 = np.array([10, 255, 255])
        self.red_lower2 = np.array([170, 100, 100])
        self.red_upper2 = np.array([180, 255, 255])
        
        self.green_lower = np.array([40, 100, 100])
        self.green_upper = np.array([80, 255, 255])
        
        self.orange_lower = np.array([10, 120, 120])
        self.orange_upper = np.array([25, 255, 255])
        
        # Detection parameters
        self.min_area_strict = 300
        self.min_area_relaxed = 80
        self.aspect_threshold = 0.8
        self.gate_width = 1.5
        
        self.gate_detection_history = deque(maxlen=5)
        self.min_confirmations = 2
        
        self.frame_count = 0
        
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
        
        # Publishers
        self.gate_detected_pub = self.create_publisher(Bool, '/gate/detected', 10)
        self.alignment_pub = self.create_publisher(Float32, '/gate/alignment_error', 10)
        self.distance_pub = self.create_publisher(Float32, '/gate/estimated_distance', 10)
        self.gate_center_pub = self.create_publisher(Point, '/gate/center_point', 10)
        self.debug_pub = self.create_publisher(Image, '/gate/debug_image', 10)
        self.status_pub = self.create_publisher(String, '/gate/status', 10)
        
        # NEW PUBLISHERS for better navigation
        self.frame_position_pub = self.create_publisher(Float32, '/gate/frame_position', 10)
        self.confidence_pub = self.create_publisher(Float32, '/gate/detection_confidence', 10)
        self.partial_gate_pub = self.create_publisher(Bool, '/gate/partial_detection', 10)
        
        self.flare_detected_pub = self.create_publisher(Bool, '/flare/detected', 10)
        self.flare_direction_pub = self.create_publisher(Float32, '/flare/avoidance_direction', 10)
        self.flare_warning_pub = self.create_publisher(String, '/flare/warning', 10)
        
        self.get_logger().info('✅ IMPROVED Gate Detector initialized')
        self.get_logger().info('   - Handles partial gate views')
        self.get_logger().info('   - Provides frame position feedback')
        self.get_logger().info('   - Enhanced tracking for continuous visibility')
    
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
        
        # Create masks
        red_mask1 = cv2.inRange(hsv_image, self.red_lower1, self.red_upper1)
        red_mask2 = cv2.inRange(hsv_image, self.red_lower2, self.red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        green_mask = cv2.inRange(hsv_image, self.green_lower, self.green_upper)
        orange_mask = cv2.inRange(hsv_image, self.orange_lower, self.orange_upper)
        
        # Morphological operations
        kernel = np.ones((3, 3), np.uint8)
        red_mask_clean = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        red_mask_clean = cv2.morphologyEx(red_mask_clean, cv2.MORPH_OPEN, kernel)
        green_mask_clean = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
        green_mask_clean = cv2.morphologyEx(green_mask_clean, cv2.MORPH_OPEN, kernel)
        orange_mask_clean = cv2.morphologyEx(orange_mask, cv2.MORPH_CLOSE, kernel)
        orange_mask_clean = cv2.morphologyEx(orange_mask_clean, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        red_contours, _ = cv2.findContours(red_mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        green_contours, _ = cv2.findContours(green_mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        orange_contours, _ = cv2.findContours(orange_mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Detect orange flare
        flare_detected = self.detect_flare(orange_contours, debug_img, w)
        
        # Detect gate stripes
        red_stripe = self.find_best_stripe(red_contours, debug_img, (0, 0, 255), 
                                           "RED", self.min_area_strict, strict=True)
        green_stripe = self.find_best_stripe(green_contours, debug_img, (0, 255, 0), 
                                             "GREEN", self.min_area_strict, strict=True)
        
        if not red_stripe:
            red_stripe = self.find_best_stripe(red_contours, debug_img, (100, 0, 200), 
                                               "RED*", self.min_area_relaxed, strict=False)
        if not green_stripe:
            green_stripe = self.find_best_stripe(green_contours, debug_img, (0, 200, 100), 
                                                 "GRN*", self.min_area_relaxed, strict=False)
        
        # IMPROVED GATE DETECTION - Handles partial views
        gate_detected = False
        partial_gate = False
        alignment_error = 0.0
        estimated_distance = 999.0
        gate_center_x = w // 2
        gate_center_y = h // 2
        frame_position = 0.0  # -1 (left edge) to +1 (right edge)
        confidence = 0.0
        
        if red_stripe and green_stripe:
            # FULL GATE DETECTED - Both stripes visible
            gate_detected = True
            partial_gate = False
            confidence = 1.0
            
            gate_center_x = (red_stripe['center'][0] + green_stripe['center'][0]) // 2
            gate_center_y = (red_stripe['center'][1] + green_stripe['center'][1]) // 2
            
            image_center_x = w / 2
            pixel_error = gate_center_x - image_center_x
            alignment_error = pixel_error / image_center_x
            
            stripe_distance = abs(red_stripe['center'][0] - green_stripe['center'][0])
            if stripe_distance > 20:
                estimated_distance = (self.gate_width * self.fx) / stripe_distance
                estimated_distance = max(0.5, min(estimated_distance, 50.0))
            
            # Frame position (0 = center, -1 = left, +1 = right)
            frame_position = (gate_center_x - w/2) / (w/2)
            
            # Visualization
            cv2.circle(debug_img, (gate_center_x, gate_center_y), 25, (255, 0, 255), -1)
            cv2.line(debug_img, (gate_center_x, 0), (gate_center_x, h), (255, 0, 255), 4)
            cv2.line(debug_img, red_stripe['center'], green_stripe['center'], (0, 255, 255), 5)
            
            cv2.putText(debug_img, f"FULL GATE {estimated_distance:.1f}m", 
                       (gate_center_x - 100, gate_center_y - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 4)
        
        elif red_stripe or green_stripe:
            # PARTIAL GATE DETECTED - Only one stripe visible
            gate_detected = True
            partial_gate = True
            confidence = 0.5
            
            stripe = red_stripe if red_stripe else green_stripe
            stripe_name = "RED (PORT)" if red_stripe else "GREEN (STARBOARD)"
            
            gate_center_x = stripe['center'][0]
            gate_center_y = stripe['center'][1]
            
            # CRITICAL: Calculate alignment to keep gate in frame
            # Strategy: Use stripe position to infer where we should point
            image_center_x = w / 2
            
            if red_stripe:
                # Red (port/left) stripe visible
                # When centered, red should be at ~35% from left (gate center at 50%)
                # If red is too far right, turn LEFT to bring full gate into view
                desired_x = w * 0.30
                pixel_error = stripe['center'][0] - desired_x
                
                # If red is at right edge, turn LEFT aggressively
                if stripe['center'][0] > w * 0.7:
                    pixel_error = stripe['center'][0] - w * 0.5
                    confidence = 0.3
            else:  # green_stripe
                # Green (starboard/right) stripe visible
                # When centered, green should be at ~65% from left
                # If green is too far left, turn RIGHT to bring full gate into view
                desired_x = w * 0.70
                pixel_error = stripe['center'][0] - desired_x
                
                # If green is at left edge, turn RIGHT aggressively
                if stripe['center'][0] < w * 0.3:
                    pixel_error = stripe['center'][0] - w * 0.5
                    confidence = 0.3
            
            alignment_error = pixel_error / image_center_x
            estimated_distance = 999.0  # Unreliable with one stripe
            frame_position = (gate_center_x - w/2) / (w/2)
            
            # Visualization
            cv2.circle(debug_img, (gate_center_x, gate_center_y), 30, (255, 165, 0), 5)
            cv2.putText(debug_img, f"PARTIAL: {stripe_name}", 
                       (gate_center_x - 150, gate_center_y - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 165, 0), 3)
            cv2.putText(debug_img, "CENTERING...", 
                       (gate_center_x - 100, gate_center_y + 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        
        # Draw frame edges warning
        if gate_detected:
            if abs(frame_position) > 0.6:  # Near edge
                edge_warning = "NEAR EDGE!" if abs(frame_position) > 0.8 else "edge warning"
                color = (0, 0, 255) if abs(frame_position) > 0.8 else (0, 165, 255)
                cv2.putText(debug_img, edge_warning, (w//2 - 150, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        # Draw center line
        cv2.line(debug_img, (w//2, 0), (w//2, h), (0, 255, 255), 2)
        
        # Status overlay
        status_lines = [
            f"Frame {self.frame_count}",
            f"Confidence: {confidence:.2f}",
        ]
        
        if flare_detected:
            status_lines.append("⚠️ FLARE!")
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
    
    def detect_flare(self, orange_contours, debug_img, w):
        """Detect orange flare obstacle"""
        flare_detected = False
        
        if orange_contours:
            largest_orange = max(orange_contours, key=cv2.contourArea)
            orange_area = cv2.contourArea(largest_orange)
            
            if orange_area > 500:
                M = cv2.moments(largest_orange)
                if M["m00"] > 0:
                    flare_center_x = int(M["m10"] / M["m00"])
                    flare_center_y = int(M["m01"] / M["m00"])
                    
                    flare_detected = True
                    
                    cv2.circle(debug_img, (flare_center_x, flare_center_y), 30, (0, 140, 255), 5)
                    cv2.putText(debug_img, "FLARE", (flare_center_x - 50, flare_center_y - 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 140, 255), 3)
                    
                    avoidance_direction = -1.0 if flare_center_x < w/2 else 1.0
                    
                    self.flare_detected_pub.publish(Bool(data=True))
                    self.flare_direction_pub.publish(Float32(data=avoidance_direction))
                    self.flare_warning_pub.publish(String(data=f"FLARE at X={flare_center_x}"))
        
        if not flare_detected:
            self.flare_detected_pub.publish(Bool(data=False))
        
        return flare_detected
    
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
    
    def find_best_stripe(self, contours, debug_img, color, label, min_area, strict=True):
        """Find best gate stripe with improved edge handling"""
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
            
            # Check if at image edge (partial view)
            image_width = debug_img.shape[1] if debug_img is not None else 1280
            at_edge = (x < 50 or (x + w) > image_width - 50)
            
            score = area
            
            if strict:
                if aspect_ratio > self.aspect_threshold or at_edge:
                    score *= 2.0
                else:
                    continue
            else:
                if aspect_ratio > self.aspect_threshold or at_edge:
                    score *= 1.5
            
            # Bonus for stripes near center
            center_x = x + w/2
            center_distance = abs(center_x - image_width/2) / (image_width/2)
            if center_distance < 0.3:
                score *= 1.2
            
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
                        'score': score,
                        'at_edge': at_edge
                    }
                    best_score = score
        
        if best_stripe and debug_img is not None:
            cx, cy = best_stripe['center']
            x, y, w, h = best_stripe['bbox']
            
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), color, 3)
            cv2.circle(debug_img, (cx, cy), 15, color, -1)
            cv2.circle(debug_img, (cx, cy), 17, (255, 255, 255), 2)
            
            label_text = f"{label} {int(best_stripe['area'])}"
            if best_stripe['at_edge']:
                label_text += " @EDGE"
            cv2.putText(debug_img, label_text, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return best_stripe


def main(args=None):
    rclpy.init(args=args)
    node = ImprovedGateDetector()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()