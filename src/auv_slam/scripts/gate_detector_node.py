#!/usr/bin/env python3
"""
SAUVC Gate Detector Node - Optimized for Competition
Detects 150cm x 150cm gate with red (port) and green (starboard) stripes
Provides alignment error and distance estimation for navigation
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
        
        # Parameters from config
        self.declare_parameter('min_contour_area', 50)
        self.declare_parameter('aspect_ratio_threshold', 0.8)
        self.declare_parameter('gate_width_meters', 1.5)
        self.declare_parameter('flare_min_area', 50)
        self.declare_parameter('flare_aspect_min', 1.5)
        self.declare_parameter('flare_danger_threshold', 0.3)
        self.declare_parameter('publish_debug', True)
        self.declare_parameter('detection_history_size', 5)
        self.declare_parameter('min_detections_for_confirm', 2)
        
        self.min_area = self.get_parameter('min_contour_area').value
        self.aspect_threshold = self.get_parameter('aspect_ratio_threshold').value
        self.gate_width = self.get_parameter('gate_width_meters').value
        self.flare_min_area = self.get_parameter('flare_min_area').value
        self.flare_aspect_min = self.get_parameter('flare_aspect_min').value
        self.flare_danger = self.get_parameter('flare_danger_threshold').value
        self.publish_debug = self.get_parameter('publish_debug').value
        self.history_size = self.get_parameter('detection_history_size').value
        self.min_confirmations = self.get_parameter('min_detections_for_confirm').value
        
        # HSV ranges from config (optimized for underwater)
        # Red stripes (port side)
        self.red_lower1 = np.array([0, 60, 60])
        self.red_upper1 = np.array([15, 255, 255])
        self.red_lower2 = np.array([155, 60, 60])
        self.red_upper2 = np.array([180, 255, 255])
        
        # Green stripes (starboard side)
        self.green_lower = np.array([35, 30, 30])
        self.green_upper = np.array([95, 255, 255])
        
        # Orange flare (MUST AVOID)
        self.orange_lower = np.array([5, 60, 60])
        self.orange_upper = np.array([30, 255, 255])
        
        # Detection history for temporal filtering
        self.gate_detection_history = deque(maxlen=self.history_size)
        self.flare_detection_history = deque(maxlen=self.history_size)
        
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
        
        self.flare_detected_pub = self.create_publisher(Bool, '/flare/detected', 10)
        self.flare_avoidance_pub = self.create_publisher(Float32, '/flare/avoidance_direction', 10)
        self.flare_warning_pub = self.create_publisher(String, '/flare/warning', 10)
        
        if self.publish_debug:
            self.debug_pub = self.create_publisher(Image, '/gate/debug_image', 10)
        
        self.get_logger().info('Gate Detector Node initialized')
        self.get_logger().info(f'Minimum area: {self.min_area}, Aspect threshold: {self.aspect_threshold}')
    
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
            self.get_logger().info(f'Focal length: fx={self.fx:.1f}, fy={self.fy:.1f}')
            # Don't destroy subscription - might need it if camera restarts
    
    def image_callback(self, msg: Image):
        if self.camera_matrix is None:
            return
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        except CvBridgeError as e:
            self.get_logger().error(f'CV Bridge error: {e}')
            return
        
        debug_img = cv_image.copy() if self.publish_debug else None
        
        # Detect gate components
        red_stripe = self.detect_stripe(hsv_image, self.red_lower1, self.red_upper1, 
                                       self.red_lower2, self.red_upper2, debug_img, 
                                       (0, 0, 255), "RED (PORT)")
        
        green_stripe = self.detect_stripe(hsv_image, self.green_lower, self.green_upper,
                                         None, None, debug_img, (0, 255, 0), "GREEN (STBD)")
        
        # Detect orange flare
        flare_info = self.detect_flare(hsv_image, debug_img)
        
        # Process gate detection
        gate_detected = False
        alignment_error = 0.0
        estimated_distance = 0.0
        gate_center_x = 0
        gate_center_y = 0
        
        if red_stripe is not None and green_stripe is not None:
            # Both stripes detected - calculate gate properties
            gate_detected = True
            
            # Gate center is midpoint between stripes
            gate_center_x = (red_stripe['center'][0] + green_stripe['center'][0]) // 2
            gate_center_y = (red_stripe['center'][1] + green_stripe['center'][1]) // 2
            
            # Calculate alignment error (normalized)
            # Negative = gate is left, Positive = gate is right
            image_center_x = self.image_width / 2
            pixel_error = gate_center_x - image_center_x
            alignment_error = pixel_error / image_center_x  # Normalized [-1, 1]
            
            # Estimate distance using gate width
            stripe_distance_pixels = abs(red_stripe['center'][0] - green_stripe['center'][0])
            if stripe_distance_pixels > 0:
                # Distance = (real_width * focal_length) / pixel_width
                estimated_distance = (self.gate_width * self.fx) / stripe_distance_pixels
            else:
                estimated_distance = 999.0
            
            if debug_img is not None:
                # Draw gate center
                cv2.circle(debug_img, (gate_center_x, gate_center_y), 10, (255, 0, 255), -1)
                cv2.line(debug_img, (gate_center_x, 0), (gate_center_x, self.image_height), 
                        (255, 0, 255), 2)
                
                # Draw image center line
                cv2.line(debug_img, (int(image_center_x), 0), 
                        (int(image_center_x), self.image_height), (0, 255, 255), 2)
                
                # Draw gate dimensions
                cv2.line(debug_img, red_stripe['center'], green_stripe['center'], (255, 255, 0), 3)
                
                # Display info
                info_text = [
                    f"GATE DETECTED",
                    f"Distance: {estimated_distance:.2f}m",
                    f"Alignment: {alignment_error:+.3f}",
                    f"Center: ({gate_center_x}, {gate_center_y})"
                ]
                y_offset = 30
                for text in info_text:
                    cv2.putText(debug_img, text, (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    y_offset += 30
        
        elif red_stripe is not None or green_stripe is not None:
            # Only one stripe detected
            gate_detected = False
            if debug_img is not None:
                cv2.putText(debug_img, "PARTIAL GATE DETECTION", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Temporal filtering for gate
        self.gate_detection_history.append(gate_detected)
        confirmed_gate = sum(self.gate_detection_history) >= self.min_confirmations
        
        # Temporal filtering for flare
        self.flare_detection_history.append(flare_info is not None)
        confirmed_flare = sum(self.flare_detection_history) >= self.min_confirmations
        
        # Publish gate detection
        gate_msg = Bool()
        gate_msg.data = confirmed_gate
        self.gate_detected_pub.publish(gate_msg)
        
        if confirmed_gate:
            # Publish alignment error
            align_msg = Float32()
            align_msg.data = float(alignment_error)
            self.alignment_pub.publish(align_msg)
            
            # Publish distance
            dist_msg = Float32()
            dist_msg.data = float(estimated_distance)
            self.distance_pub.publish(dist_msg)
            
            # Publish gate center
            center_msg = Point()
            center_msg.x = float(gate_center_x)
            center_msg.y = float(gate_center_y)
            center_msg.z = float(estimated_distance)
            self.gate_center_pub.publish(center_msg)
        
        # Publish flare detection
        flare_msg = Bool()
        flare_msg.data = confirmed_flare
        self.flare_detected_pub.publish(flare_msg)
        
        if confirmed_flare and flare_info:
            # Calculate avoidance direction
            flare_x = flare_info['center'][0]
            image_center = self.image_width / 2
            
            # If flare is left of center, avoid by going right (+1)
            # If flare is right of center, avoid by going left (-1)
            avoidance_direction = 1.0 if flare_x < image_center else -1.0
            
            avoidance_msg = Float32()
            avoidance_msg.data = avoidance_direction
            self.flare_avoidance_pub.publish(avoidance_msg)
            
            # Calculate danger level
            flare_distance_from_center = abs(flare_x - image_center) / image_center
            
            if flare_distance_from_center < self.flare_danger:
                warning = String()
                warning.data = "CRITICAL: Orange flare dead ahead! Emergency avoidance!"
                self.flare_warning_pub.publish(warning)
                
                if debug_img is not None:
                    cv2.putText(debug_img, "!!! FLARE DANGER !!!", (10, self.image_height - 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
        # Publish debug image
        if self.publish_debug and debug_img is not None:
            try:
                debug_msg = self.bridge.cv2_to_imgmsg(debug_img, "bgr8")
                debug_msg.header = msg.header
                self.debug_pub.publish(debug_msg)
            except CvBridgeError as e:
                self.get_logger().error(f'Debug image error: {e}')
    
    def detect_stripe(self, hsv_image, lower1, upper1, lower2, upper2, debug_img, color, label):
        """Detect a single stripe (red or green) of the gate"""
        
        # Create mask
        mask1 = cv2.inRange(hsv_image, lower1, upper1)
        if lower2 is not None and upper2 is not None:
            mask2 = cv2.inRange(hsv_image, lower2, upper2)
            mask = mask1 | mask2
        else:
            mask = mask1
        
        # Morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find largest valid contour
        best_contour = None
        best_area = 0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue
            
            # Check if it's vertical (stripe-like)
            x, y, w, h = cv2.boundingRect(cnt)
            if w == 0:
                continue
            aspect_ratio = float(h) / w
            
            # Stripe should be taller than wide
            if aspect_ratio > self.aspect_threshold and area > best_area:
                best_contour = cnt
                best_area = area
        
        if best_contour is None:
            return None
        
        # Get stripe properties
        M = cv2.moments(best_contour)
        if M["m00"] == 0:
            return None
        
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        x, y, w, h = cv2.boundingRect(best_contour)
        
        stripe_info = {
            'center': (cx, cy),
            'bbox': (x, y, w, h),
            'area': best_area
        }
        
        # Draw on debug image
        if debug_img is not None:
            cv2.drawContours(debug_img, [best_contour], -1, color, 3)
            cv2.circle(debug_img, (cx, cy), 8, color, -1)
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(debug_img, label, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return stripe_info
    
    def detect_flare(self, hsv_image, debug_img):
        """Detect orange flare (obstacle)"""
        
        mask = cv2.inRange(hsv_image, self.orange_lower, self.orange_upper)
        
        # Morphological operations
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find largest flare
        best_contour = None
        best_area = 0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.flare_min_area:
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            if w == 0:
                continue
            aspect_ratio = float(h) / w
            
            # Flare is tall and thin
            if aspect_ratio > self.flare_aspect_min and area > best_area:
                best_contour = cnt
                best_area = area
        
        if best_contour is None:
            return None
        
        M = cv2.moments(best_contour)
        if M["m00"] == 0:
            return None
        
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        x, y, w, h = cv2.boundingRect(best_contour)
        
        flare_info = {
            'center': (cx, cy),
            'bbox': (x, y, w, h),
            'area': best_area
        }
        
        if debug_img is not None:
            cv2.drawContours(debug_img, [best_contour], -1, (0, 165, 255), 4)
            cv2.circle(debug_img, (cx, cy), 10, (0, 0, 255), -1)
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 0, 255), 3)
            cv2.putText(debug_img, "ORANGE FLARE", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return flare_info


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