#!/usr/bin/env python3
"""
FIXED Gate Detector Node - Typo corrected
Line 109: get_Logger() -> get_logger()
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, Float32, String
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

class FixedGateDetectorNode(Node):
    def __init__(self):
        super().__init__('fixed_gate_detector_node')
        self.bridge = CvBridge()
        self.camera_matrix = None
        self.image_width = None
        self.image_height = None
        
        # FIXED PARAMETERS
        self.declare_parameter('min_contour_area', 50)
        self.declare_parameter('aspect_ratio_threshold', 0.8)
        self.declare_parameter('gate_width_meters', 1.5)
        self.declare_parameter('flare_min_area', 50)
        self.declare_parameter('flare_aspect_min', 1.5)
        self.declare_parameter('flare_danger_threshold', 0.3)
        self.declare_parameter('publish_debug', True)
        
        self.min_area = self.get_parameter('min_contour_area').value
        self.aspect_threshold = self.get_parameter('aspect_ratio_threshold').value
        self.gate_width = self.get_parameter('gate_width_meters').value
        self.flare_min_area = self.get_parameter('flare_min_area').value
        self.flare_aspect_min = self.get_parameter('flare_aspect_min').value
        self.flare_danger_threshold = self.get_parameter('flare_danger_threshold').value
        self.publish_debug = self.get_parameter('publish_debug').value
        
        # ENHANCED HSV ranges
        self.red_lower1 = np.array([0, 60, 60])
        self.red_upper1 = np.array([15, 255, 255])
        self.red_lower2 = np.array([155, 60, 60])
        self.red_upper2 = np.array([180, 255, 255])
        
        self.green_lower = np.array([35, 30, 30])
        self.green_upper = np.array([95, 255, 255])
        
        self.orange_lower = np.array([5, 60, 60])
        self.orange_upper = np.array([30, 255, 255])
        
        # Detection history
        self.detection_history = []
        self.history_size = 5
        self.min_confirmations = 2
        
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
            Image, '/camera_forward/image_raw',
            self.image_callback, qos_sensor
        )
        self.cam_info_sub = self.create_subscription(
            CameraInfo, '/camera_forward/camera_info',
            self.cam_info_callback, qos_reliable
        )
        
        # Publishers
        self.gate_detected_pub = self.create_publisher(Bool, '/gate/detected', 10)
        self.gate_center_pub = self.create_publisher(PoseStamped, '/gate/center_pose', 10)
        self.gate_alignment_pub = self.create_publisher(Float32, '/gate/alignment_error', 10)
        self.gate_distance_pub = self.create_publisher(Float32, '/gate/estimated_distance', 10)
        
        self.flare_detected_pub = self.create_publisher(Bool, '/flare/detected', 10)
        self.flare_position_pub = self.create_publisher(PoseStamped, '/flare/position', 10)
        self.flare_warning_pub = self.create_publisher(String, '/flare/warning', 10)
        self.flare_avoidance_pub = self.create_publisher(Float32, '/flare/avoidance_direction', 10)
        
        self.debug_pub = self.create_publisher(Image, '/gate/debug_image', 10)
        
        self.get_logger().info('='*70)
        self.get_logger().info('✅ FIXED Gate Detector with Enhanced Debugging')
        self.get_logger().info('='*70)
        self.get_logger().info('📺 Subscribe to /gate/debug_image in rqt_image_view')
        self.get_logger().info(f'🔍 Min area: {self.min_area} (reduced for long-range)')
        self.get_logger().info(f'📐 Aspect ratio: {self.aspect_threshold} (more lenient)')  # FIXED: lowercase 'l'
        self.get_logger().info('='*70)
    
    def cam_info_callback(self, msg: CameraInfo):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape((3, 3))
            self.image_width = msg.width
            self.image_height = msg.height
            self.get_logger().info(f'📷 Camera: {self.image_width}x{self.image_height}')
            self.destroy_subscription(self.cam_info_sub)
    
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
        
        # Detect components
        flare_info = self.detect_orange_flare(hsv_image, debug_img)
        red_bars = self.detect_color_bars(hsv_image, self.red_lower1, self.red_upper1,
                                          self.red_lower2, self.red_upper2, debug_img, 
                                          (0, 0, 255), "RED")
        green_bars = self.detect_color_bars(hsv_image, self.green_lower, self.green_upper,
                                            None, None, debug_img, (0, 255, 0), "GREEN")
        
        # Gate detection with temporal filtering
        gate_detected_now = len(red_bars) > 0 and len(green_bars) > 0
        self.detection_history.append(gate_detected_now)
        if len(self.detection_history) > self.history_size:
            self.detection_history.pop(0)
        
        confirmations = sum(self.detection_history)
        gate_detected = confirmations >= self.min_confirmations
        
        # Publish detection status
        detected_msg = Bool()
        detected_msg.data = gate_detected
        self.gate_detected_pub.publish(detected_msg)
        
        # Draw detection status on debug image
        if debug_img is not None:
            status_color = (0, 255, 0) if gate_detected else (0, 0, 255)
            cv2.putText(debug_img, 
                       f"GATE: {'DETECTED' if gate_detected else 'NOT DETECTED'}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)
            cv2.putText(debug_img, 
                       f"Confirmations: {confirmations}/{self.history_size}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Process flare
        if flare_info['detected']:
            self.process_flare_avoidance(flare_info, msg.header, debug_img)
        
        # Process gate
        if gate_detected and len(red_bars) > 0 and len(green_bars) > 0:
            self.process_gate_detection(red_bars[0], green_bars[0], flare_info, 
                                       msg.header, debug_img)
        else:
            if debug_img is not None:
                if len(red_bars) == 0:
                    cv2.putText(debug_img, "Missing: RED stripes", 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                if len(green_bars) == 0:
                    cv2.putText(debug_img, "Missing: GREEN stripes", 
                               (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Publish debug image
        if self.publish_debug and debug_img is not None:
            try:
                debug_msg = self.bridge.cv2_to_imgmsg(debug_img, "bgr8")
                debug_msg.header = msg.header
                self.debug_pub.publish(debug_msg)
            except CvBridgeError as e:
                self.get_logger().error(f'Debug image error: {e}')
    
    def detect_color_bars(self, hsv_image, lower1, upper1, lower2, upper2, 
                          debug_img, color, label):
        """Detect colored bars"""
        
        mask1 = cv2.inRange(hsv_image, lower1, upper1)
        if lower2 is not None and upper2 is not None:
            mask2 = cv2.inRange(hsv_image, lower2, upper2)
            mask = mask1 | mask2
        else:
            mask = mask1
        
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected_bars = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(h) / w if w > 0 else 0
            
            if aspect_ratio > self.aspect_threshold:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    detected_bars.append({
                        'center': (cx, cy),
                        'bbox': (x, y, w, h),
                        'area': area,
                        'aspect_ratio': aspect_ratio
                    })
                    
                    if debug_img is not None:
                        cv2.rectangle(debug_img, (x, y), (x + w, y + h), color, 3)
                        cv2.circle(debug_img, (cx, cy), 7, color, -1)
                        cv2.putText(debug_img, f"{label} {area:.0f}px", (x, y - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        detected_bars.sort(key=lambda x: x['area'], reverse=True)
        return detected_bars
    
    def detect_orange_flare(self, hsv_image, debug_img):
        """Detect orange flare"""
        mask = cv2.inRange(hsv_image, self.orange_lower, self.orange_upper)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        flare_info = {'detected': False, 'position': None, 'bbox': None, 
                     'area': 0, 'normalized_x': 0.0}
        largest_flare = None
        max_area = 0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.flare_min_area:
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(h) / w if w > 0 else 0
            
            if aspect_ratio > self.flare_aspect_min and area > max_area:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    max_area = area
                    largest_flare = {'center': (cx, cy), 'bbox': (x, y, w, h), 'area': area}
        
        if largest_flare:
            flare_info['detected'] = True
            flare_info['position'] = largest_flare['center']
            flare_info['bbox'] = largest_flare['bbox']
            flare_info['area'] = largest_flare['area']
            cx = largest_flare['center'][0]
            flare_info['normalized_x'] = (cx - self.image_width / 2) / (self.image_width / 2)
            
            if debug_img is not None:
                x, y, w, h = largest_flare['bbox']
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 165, 255), 4)
                cv2.circle(debug_img, largest_flare['center'], 10, (0, 165, 255), -1)
                cv2.putText(debug_img, "DANGER: ORANGE FLARE!", (x, y - 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        
        return flare_info
    
    def process_gate_detection(self, red_bar, green_bar, flare_info, header, debug_img):
        """Process gate detection"""
        red_cx, red_cy = red_bar['center']
        green_cx, green_cy = green_bar['center']
        
        gate_center_x = (red_cx + green_cx) // 2
        gate_center_y = (red_cy + green_cy) // 2
        
        image_center_x = self.image_width // 2
        alignment_error_px = gate_center_x - image_center_x
        alignment_error_normalized = alignment_error_px / (self.image_width / 2.0)
        
        gate_width_px = abs(green_cx - red_cx)
        fx = self.camera_matrix[0, 0]
        estimated_distance = (self.gate_width * fx) / gate_width_px if gate_width_px > 0 else 0.0
        
        # Draw on debug image
        if debug_img is not None:
            gate_left = min(red_bar['bbox'][0], green_bar['bbox'][0])
            gate_right = max(red_bar['bbox'][0] + red_bar['bbox'][2], 
                            green_bar['bbox'][0] + green_bar['bbox'][2])
            gate_top = min(red_bar['bbox'][1], green_bar['bbox'][1])
            gate_bottom = max(red_bar['bbox'][1] + red_bar['bbox'][3], 
                             green_bar['bbox'][1] + green_bar['bbox'][3])
            
            cv2.rectangle(debug_img, (gate_left, gate_top), (gate_right, gate_bottom), 
                         (255, 255, 0), 3)
            cv2.circle(debug_img, (gate_center_x, gate_center_y), 15, (0, 255, 255), -1)
            cv2.line(debug_img, (red_cx, red_cy), (green_cx, green_cy), (255, 255, 0), 3)
            cv2.line(debug_img, (image_center_x, 0), (image_center_x, self.image_height), 
                    (255, 0, 255), 2)
            
            info_y = gate_bottom + 30
            cv2.putText(debug_img, f"Distance: {estimated_distance:.2f}m", 
                       (gate_left, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(debug_img, f"Alignment: {alignment_error_normalized:+.2f}", 
                       (gate_left, info_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                       (0, 255, 0) if abs(alignment_error_normalized) < 0.1 else (0, 0, 255), 2)
        
        # Publish data
        pose_msg = PoseStamped()
        pose_msg.header = header
        pose_msg.header.frame_id = 'camera_forward'
        pose_msg.pose.position.x = float(gate_center_x)
        pose_msg.pose.position.y = float(gate_center_y)
        pose_msg.pose.position.z = estimated_distance
        pose_msg.pose.orientation.w = 1.0
        self.gate_center_pub.publish(pose_msg)
        
        alignment_msg = Float32()
        alignment_msg.data = alignment_error_normalized
        self.gate_alignment_pub.publish(alignment_msg)
        
        distance_msg = Float32()
        distance_msg.data = estimated_distance
        self.gate_distance_pub.publish(distance_msg)
    
    def process_flare_avoidance(self, flare_info, header, debug_img):
        """Process flare detection"""
        flare_msg = Bool()
        flare_msg.data = True
        self.flare_detected_pub.publish(flare_msg)
        
        pos_msg = PoseStamped()
        pos_msg.header = header
        pos_msg.header.frame_id = 'camera_forward'
        pos_msg.pose.position.x = float(flare_info['position'][0])
        pos_msg.pose.position.y = float(flare_info['position'][1])
        pos_msg.pose.orientation.w = 1.0
        self.flare_position_pub.publish(pos_msg)
        
        avoidance_direction = -flare_info['normalized_x']
        avoidance_msg = Float32()
        avoidance_msg.data = avoidance_direction
        self.flare_avoidance_pub.publish(avoidance_msg)


def main(args=None):
    rclpy.init(args=args)
    node = FixedGateDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
