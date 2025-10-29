#!/usr/bin/env python3
"""
Gate Detector Node - Detects the navigation gate and orange flare
Publishes gate center position, alignment, and flare warning
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import Bool, Float32, String
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

class GateDetectorNode(Node):
    def __init__(self):
        super().__init__('gate_detector_node', automatically_declare_parameters_from_overrides=True)
        self.bridge = CvBridge()
        self.camera_matrix = None
        self.image_width = None
        self.image_height = None
        
        # --- NEW: Parameter loading flag ---
        self.params_loaded = False
        
        # --- Parameters will be initialized in image_callback ---
        self.min_area = 0
        self.aspect_threshold = 0.0
        self.gate_width = 0.0
        self.flare_min_area = 0
        self.flare_aspect_min = 0.0
        self.flare_danger_threshold = 0.0
        self.publish_debug = False
        
        # HSV color ranges
        self.red_lower1 = np.array([0, 100, 100])
        self.red_upper1 = np.array([10, 255, 255])
        self.red_lower2 = np.array([160, 100, 100])
        self.red_upper2 = np.array([180, 255, 255])
        
        self.green_lower = np.array([40, 50, 50])
        self.green_upper = np.array([90, 255, 255])
        
        self.orange_lower = np.array([10, 100, 100])
        self.orange_upper = np.array([25, 255, 255])
        
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
        self.gate_center_pub = self.create_publisher(PoseStamped, '/gate/center_pose', 10)
        self.gate_alignment_pub = self.create_publisher(Float32, '/gate/alignment_error', 10)
        self.gate_distance_pub = self.create_publisher(Float32, '/gate/estimated_distance', 10)
        
        self.flare_detected_pub = self.create_publisher(Bool, '/flare/detected', 10)
        self.flare_position_pub = self.create_publisher(PoseStamped, '/flare/position', 10)
        self.flare_warning_pub = self.create_publisher(String, '/flare/warning', 10)
        self.flare_avoidance_pub = self.create_publisher(Float32, '/flare/avoidance_direction', 10)
        
        self.debug_pub = None # Will be created after params are loaded
        
        self.get_logger().info('Gate Detector Node initialized with flare avoidance')
    
    def cam_info_callback(self, msg: CameraInfo):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape((3, 3))
            self.image_width = msg.width
            self.image_height = msg.height
            self.get_logger().info(f'Camera info received: {self.image_width}x{self.image_height}')
            self.destroy_subscription(self.cam_info_sub)
    
    def image_callback(self, msg: Image):
        if self.camera_matrix is None:
            self.get_logger().warn('Waiting for camera info...')
            return

        # --- NEW: Robust parameter loading ---
        if not self.params_loaded:
            try:
                self.min_area = self.get_parameter('min_contour_area').value
                self.aspect_threshold = self.get_parameter('aspect_ratio_threshold').value
                self.gate_width = self.get_parameter('gate_width_meters').value
                self.flare_min_area = self.get_parameter('flare_min_area').value
                self.flare_aspect_min = self.get_parameter('flare_aspect_min').value
                self.flare_danger_threshold = self.get_parameter('flare_danger_threshold').value
                self.publish_debug = self.get_parameter('publish_debug').value
                
                if self.publish_debug:
                    self.debug_pub = self.create_publisher(Image, '/gate/debug_image', 10)
                
                self.params_loaded = True
                self.get_logger().info('Gate detector parameters loaded successfully.')
            except rclpy.exceptions.ParameterNotDeclaredException as e:
                self.get_logger().warn(f'Waiting for gate detector parameters...: {e}')
                return
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        except CvBridgeError as e:
            self.get_logger().error(f'CV Bridge error: {e}')
            return
        
        debug_img = cv_image.copy() if self.publish_debug else None
        
        # PRIORITY 1: Detect orange flare
        flare_info = self.detect_orange_flare(hsv_image, debug_img, self.flare_min_area, self.flare_aspect_min)
        
        # PRIORITY 2: Detect gate
        red_bars = self.detect_color_bars(
            hsv_image, self.red_lower1, self.red_upper1,
            self.red_lower2, self.red_upper2, debug_img, (0, 0, 255), "RED",
            self.min_area, self.aspect_threshold
        )
        
        green_bars = self.detect_color_bars(
            hsv_image, self.green_lower, self.green_upper,
            None, None, debug_img, (0, 255, 0), "GREEN",
            self.min_area, self.aspect_threshold
        )
        
        gate_detected = len(red_bars) > 0 and len(green_bars) > 0
        
        detected_msg = Bool()
        detected_msg.data = gate_detected
        self.gate_detected_pub.publish(detected_msg)
        
        if flare_info['detected']:
            self.process_flare_avoidance(flare_info, msg.header, debug_img, self.flare_danger_threshold)
        
        if gate_detected:
            self.process_gate_detection(red_bars[0], green_bars[0], 
                                       flare_info, msg.header, debug_img, self.gate_width)
        
        if self.publish_debug and debug_img is not None and self.debug_pub is not None:
            try:
                debug_msg = self.bridge.cv2_to_imgmsg(debug_img, "bgr8")
                debug_msg.header = msg.header
                self.debug_pub.publish(debug_msg)
            except CvBridgeError as e:
                self.get_logger().error(f'Debug image error: {e}')
    
    def detect_orange_flare(self, hsv_image, debug_img, flare_min_area, flare_aspect_min):
        mask = cv2.inRange(hsv_image, self.orange_lower, self.orange_upper)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        flare_info = {'detected': False, 'position': None, 'bbox': None, 'area': 0, 'normalized_x': 0.0}
        largest_flare = None
        max_area = 0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < flare_min_area:
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(h) / w if w > 0 else 0
            
            if aspect_ratio > flare_aspect_min and area > max_area:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    max_area = area
                    largest_flare = {'center': (cx, cy), 'bbox': (x, y, w, h), 'area': area, 'aspect_ratio': aspect_ratio}
        
        if largest_flare:
            flare_info['detected'] = True
            flare_info['position'] = largest_flare['center']
            flare_info['bbox'] = largest_flare['bbox']
            flare_info['area'] = largest_flare['area']
            cx = largest_flare['center'][0]
            flare_info['normalized_x'] = (cx - self.image_width / 2) / (self.image_width / 2)
            
            if debug_img is not None:
                x, y, w, h = largest_flare['bbox']
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 165, 255), 3)
                cv2.circle(debug_img, largest_flare['center'], 8, (0, 165, 255), -1)
                cv2.putText(debug_img, "ORANGE FLARE - AVOID!", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                danger_left = int(x - w * 0.5)
                danger_right = int(x + w * 1.5)
                cv2.rectangle(debug_img, (danger_left, 0), (danger_right, self.image_height), (0, 0, 255), 2)
        
        return flare_info
    
    def process_flare_avoidance(self, flare_info, header, debug_img, flare_danger_threshold):
        flare_msg = Bool(); flare_msg.data = True
        self.flare_detected_pub.publish(flare_msg)
        
        pos_msg = PoseStamped(); pos_msg.header = header; pos_msg.header.frame_id = 'camera_forward'
        pos_msg.pose.position.x = float(flare_info['position'][0]); pos_msg.pose.position.y = float(flare_info['position'][1])
        pos_msg.pose.orientation.w = 1.0
        self.flare_position_pub.publish(pos_msg)
        
        avoidance_direction = -flare_info['normalized_x']
        avoidance_msg = Float32(); avoidance_msg.data = avoidance_direction
        self.flare_avoidance_pub.publish(avoidance_msg)
        
        warning = String()
        if abs(flare_info['normalized_x']) < flare_danger_threshold:
            warning.data = "CRITICAL: Flare directly ahead! Taking evasive action!"
            self.get_logger().warn(warning.data)
        else:
            side = "left" if flare_info['normalized_x'] < 0 else "right"
            warning.data = f"Flare detected on {side} side - adjusting path"
        self.flare_warning_pub.publish(warning)
    
    def detect_color_bars(self, hsv_image, lower1, upper1, lower2, upper2, 
                          debug_img, color, label, min_area, aspect_threshold):
        mask1 = cv2.inRange(hsv_image, lower1, upper1)
        if lower2 is not None and upper2 is not None:
            mask2 = cv2.inRange(hsv_image, lower2, upper2)
            mask = mask1 | mask2
        else:
            mask = mask1
        
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected_bars = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(h) / w if w > 0 else 0
            
            if aspect_ratio > aspect_threshold:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    # --- THIS IS THE FIXED TYPO ---
                    cx = int(M["m10"] / M["m00"]) 
                    cy = int(M["m01"] / M["m00"])
                    
                    detected_bars.append({'center': (cx, cy), 'bbox': (x, y, w, h), 'area': area})
                    
                    if debug_img is not None:
                        cv2.rectangle(debug_img, (x, y), (x + w, y + h), color, 2)
                        cv2.circle(debug_img, (cx, cy), 5, color, -1)
                        cv2.putText(debug_img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        detected_bars.sort(key=lambda x: x['area'], reverse=True)
        return detected_bars
    
    def process_gate_detection(self, red_bar, green_bar, flare_info, header, debug_img, gate_width):
        red_cx, red_cy = red_bar['center']
        green_cx, green_cy = green_bar['center']
        
        gate_center_x = (red_cx + green_cx) // 2
        gate_center_y = (red_cy + green_cy) // 2
        
        image_center_x = self.image_width // 2
        alignment_error_px = gate_center_x - image_center_x
        
        if flare_info['detected']:
            flare_x = flare_info['position'][0]
            distance_to_flare = abs(gate_center_x - flare_x)
            
            if distance_to_flare < self.image_width * 0.3:
                bias_amount = int(self.image_width * 0.1 * np.sign(gate_center_x - flare_x))
                alignment_error_px += bias_amount
                
                if debug_img is not None:
                    cv2.putText(debug_img, "Adjusting path to avoid flare", (10, self.image_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        alignment_error_normalized = alignment_error_px / (self.image_width / 2.0)
        
        gate_width_px = abs(green_cx - red_cx)
        fx = self.camera_matrix[0, 0]
        estimated_distance = (gate_width * fx) / gate_width_px if gate_width_px > 0 else 0.0
        
        pose_msg = PoseStamped(); pose_msg.header = header; pose_msg.header.frame_id = 'camera_forward'
        pose_msg.pose.position.x = float(gate_center_x); pose_msg.pose.position.y = float(gate_center_y)
        pose_msg.pose.position.z = estimated_distance
        pose_msg.pose.orientation.w = 1.0
        self.gate_center_pub.publish(pose_msg)
        
        alignment_msg = Float32(); alignment_msg.data = alignment_error_normalized
        self.gate_alignment_pub.publish(alignment_msg)
        
        distance_msg = Float32(); distance_msg.data = estimated_distance
        self.gate_distance_pub.publish(distance_msg)
        
        if debug_img is not None:
            cv2.circle(debug_img, (gate_center_x, gate_center_y), 10, (0, 255, 255), -1)
            cv2.line(debug_img, (red_cx, red_cy), (green_cx, green_cy), (255, 255, 0), 2)
            cv2.line(debug_img, (image_center_x, 0), (image_center_x, self.image_height), (255, 0, 255), 1)
            
            info_text = [
                f"Distance: {estimated_distance:.2f}m",
                f"Alignment: {alignment_error_normalized:.2f}",
                f"Gate Width: {gate_width_px}px"
            ]
            
            if flare_info['detected']: info_text.append("FLARE DETECTED - ADJUSTING")
            for i, text in enumerate(info_text):
                cv2.putText(debug_img, text, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

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