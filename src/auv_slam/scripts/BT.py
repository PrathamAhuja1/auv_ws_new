#!/usr/bin/env python3
"""
Enhanced Gate Detector with Live OpenCV Visualization
Shows detection process in real-time with detailed feedback
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

class VisualGateDetectorNode(Node):
    def __init__(self):
        super().__init__('visual_gate_detector_node')
        self.bridge = CvBridge()
        self.camera_matrix = None
        self.image_width = None
        self.image_height = None
        
        # Parameters
        self.declare_parameter('min_contour_area', 50)
        self.declare_parameter('aspect_ratio_threshold', 0.8)
        self.declare_parameter('gate_width_meters', 1.5)
        self.declare_parameter('show_opencv_window', True)
        
        self.min_area = self.get_parameter('min_contour_area').value
        self.aspect_threshold = self.get_parameter('aspect_ratio_threshold').value
        self.gate_width = self.get_parameter('gate_width_meters').value
        self.show_window = self.get_parameter('show_opencv_window').value
        
        # Enhanced HSV ranges (underwater optimized)
        self.red_lower1 = np.array([0, 50, 50])
        self.red_upper1 = np.array([15, 255, 255])
        self.red_lower2 = np.array([165, 50, 50])
        self.red_upper2 = np.array([180, 255, 255])
        
        self.green_lower = np.array([40, 40, 40])
        self.green_upper = np.array([90, 255, 255])
        
        self.orange_lower = np.array([5, 50, 50])
        self.orange_upper = np.array([25, 255, 255])
        
        # Detection history for stability
        self.detection_history = []
        self.history_size = 5
        self.min_confirmations = 2
        
        # Frame counter for diagnostics
        self.frame_count = 0
        self.detected_frames = 0
        
        # QoS profiles
        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE
        )
        
        # Subscriptions
        self.image_sub = self.create_subscription(
            Image, '/camera_forward/image_raw', 
            self.image_callback, qos_sensor
        )
        self.cam_info_sub = self.create_subscription(
            CameraInfo, '/camera_forward/camera_info',
            self.cam_info_callback, 10
        )
        
        # Publishers
        self.gate_detected_pub = self.create_publisher(Bool, '/gate/detected', 10)
        self.gate_center_pub = self.create_publisher(PoseStamped, '/gate/center_pose', 10)
        self.gate_alignment_pub = self.create_publisher(Float32, '/gate/alignment_error', 10)
        self.gate_distance_pub = self.create_publisher(Float32, '/gate/estimated_distance', 10)
        self.debug_pub = self.create_publisher(Image, '/gate/debug_image', 10)
        
        # OpenCV windows
        if self.show_window:
            cv2.namedWindow('Gate Detection', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Gate Detection', 1280, 960)
            cv2.namedWindow('Red Mask', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Red Mask', 640, 480)
            cv2.namedWindow('Green Mask', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Green Mask', 640, 480)
        
        self.get_logger().info('='*70)
        self.get_logger().info('📹 Visual Gate Detector with OpenCV Started')
        self.get_logger().info('='*70)
        self.get_logger().info('🖥️  OpenCV windows will show live detection')
        self.get_logger().info('='*70)
    
    def cam_info_callback(self, msg: CameraInfo):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape((3, 3))
            self.image_width = msg.width
            self.image_height = msg.height
            self.get_logger().info(f'📷 Camera: {self.image_width}x{self.image_height}')
            self.get_logger().info(f'📐 Focal length: fx={self.camera_matrix[0,0]:.1f}')
    
    def image_callback(self, msg: Image):
        if self.camera_matrix is None:
            return
        
        self.frame_count += 1
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f'CV Bridge error: {e}')
            return
        
        # Create visualization canvas
        vis_image = cv_image.copy()
        h, w = cv_image.shape[:2]
        
        # Convert to HSV
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        
        # STEP 1: Detect red and green bars
        red_mask = self.create_color_mask(hsv_image, 
            self.red_lower1, self.red_upper1, self.red_lower2, self.red_upper2)
        green_mask = self.create_color_mask(hsv_image,
            self.green_lower, self.green_upper, None, None)
        
        red_bars = self.detect_bars(red_mask, vis_image, (0, 0, 255), "RED")
        green_bars = self.detect_bars(green_mask, vis_image, (0, 255, 0), "GREEN")
        
        # STEP 2: Gate detection with temporal filtering
        gate_detected_now = len(red_bars) > 0 and len(green_bars) > 0
        self.detection_history.append(gate_detected_now)
        if len(self.detection_history) > self.history_size:
            self.detection_history.pop(0)
        
        confirmations = sum(self.detection_history)
        gate_detected = confirmations >= self.min_confirmations
        
        if gate_detected:
            self.detected_frames += 1
        
        # STEP 3: Draw detection info on visualization
        self.draw_status_overlay(vis_image, gate_detected, confirmations, 
                                len(red_bars), len(green_bars))
        
        # STEP 4: Process gate if detected
        if gate_detected and len(red_bars) > 0 and len(green_bars) > 0:
            gate_info = self.process_gate_detection(red_bars[0], green_bars[0], vis_image)
            self.publish_gate_data(gate_info, msg.header)
            
            # Draw gate bounding box
            self.draw_gate_box(vis_image, gate_info)
        
        # Publish detection status
        detected_msg = Bool()
        detected_msg.data = gate_detected
        self.gate_detected_pub.publish(detected_msg)
        
        # Show OpenCV windows
        if self.show_window:
            cv2.imshow('Gate Detection', vis_image)
            cv2.imshow('Red Mask', red_mask)
            cv2.imshow('Green Mask', green_mask)
            cv2.waitKey(1)
        
        # Publish debug image
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(vis_image, "bgr8")
            debug_msg.header = msg.header
            self.debug_pub.publish(debug_msg)
        except CvBridgeError as e:
            self.get_logger().error(f'Debug image error: {e}')
    
    def create_color_mask(self, hsv, lower1, upper1, lower2, upper2):
        """Create mask for color detection"""
        mask1 = cv2.inRange(hsv, lower1, upper1)
        if lower2 is not None and upper2 is not None:
            mask2 = cv2.inRange(hsv, lower2, upper2)
            mask = mask1 | mask2
        else:
            mask = mask1
        
        # Morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def detect_bars(self, mask, vis_image, color, label):
        """Detect vertical bars from mask"""
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
                    
                    # Draw detection on visualization
                    cv2.rectangle(vis_image, (x, y), (x+w, y+h), color, 3)
                    cv2.circle(vis_image, (cx, cy), 8, color, -1)
                    cv2.putText(vis_image, f"{label}", (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.putText(vis_image, f"{area:.0f}px", (x, y+h+20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        detected_bars.sort(key=lambda x: x['area'], reverse=True)
        return detected_bars
    
    def process_gate_detection(self, red_bar, green_bar, vis_image):
        """Process gate detection and calculate metrics"""
        red_cx, red_cy = red_bar['center']
        green_cx, green_cy = green_bar['center']
        
        gate_center_x = (red_cx + green_cx) // 2
        gate_center_y = (red_cy + green_cy) // 2
        
        # Alignment error (normalized)
        image_center_x = self.image_width // 2
        alignment_error_px = gate_center_x - image_center_x
        alignment_error = alignment_error_px / (self.image_width / 2.0)
        
        # Distance estimation
        gate_width_px = abs(green_cx - red_cx)
        fx = self.camera_matrix[0, 0]
        distance = (self.gate_width * fx) / gate_width_px if gate_width_px > 0 else 0.0
        
        # Gate bounding box
        gate_left = min(red_bar['bbox'][0], green_bar['bbox'][0])
        gate_right = max(red_bar['bbox'][0] + red_bar['bbox'][2],
                        green_bar['bbox'][0] + green_bar['bbox'][2])
        gate_top = min(red_bar['bbox'][1], green_bar['bbox'][1])
        gate_bottom = max(red_bar['bbox'][1] + red_bar['bbox'][3],
                         green_bar['bbox'][1] + green_bar['bbox'][3])
        
        return {
            'center': (gate_center_x, gate_center_y),
            'alignment_error': alignment_error,
            'distance': distance,
            'width_px': gate_width_px,
            'bbox': (gate_left, gate_top, gate_right, gate_bottom)
        }
    
    def draw_gate_box(self, vis_image, gate_info):
        """Draw comprehensive gate visualization"""
        gate_left, gate_top, gate_right, gate_bottom = gate_info['bbox']
        cx, cy = gate_info['center']
        
        # Main gate bounding box (thick yellow)
        cv2.rectangle(vis_image, (gate_left, gate_top), (gate_right, gate_bottom),
                     (0, 255, 255), 4)
        
        # Gate center marker (large cyan circle)
        cv2.circle(vis_image, (cx, cy), 20, (255, 255, 0), -1)
        cv2.circle(vis_image, (cx, cy), 25, (0, 255, 255), 3)
        
        # Alignment line
        image_center_x = self.image_width // 2
        cv2.line(vis_image, (image_center_x, 0), (image_center_x, self.image_height),
                (255, 0, 255), 2)
        cv2.line(vis_image, (cx, cy), (image_center_x, cy), (0, 255, 255), 3)
        
        # Information overlay
        info_y = gate_bottom + 40
        cv2.putText(vis_image, f"GATE DETECTED", (gate_left, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        cv2.putText(vis_image, f"Distance: {gate_info['distance']:.2f}m",
                   (gate_left, info_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(vis_image, f"Alignment: {gate_info['alignment_error']:+.3f}",
                   (gate_left, info_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                   (0, 255, 0) if abs(gate_info['alignment_error']) < 0.1 else (0, 0, 255), 2)
        cv2.putText(vis_image, f"Width: {gate_info['width_px']}px",
                   (gate_left, info_y + 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    def draw_status_overlay(self, vis_image, detected, confirmations, red_count, green_count):
        """Draw comprehensive status overlay"""
        h, w = vis_image.shape[:2]
        
        # Semi-transparent status panel
        overlay = vis_image.copy()
        cv2.rectangle(overlay, (10, 10), (500, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, vis_image, 0.4, 0, vis_image)
        
        # Status text
        status_color = (0, 255, 0) if detected else (0, 0, 255)
        status_text = "GATE DETECTED ✓" if detected else "SEARCHING..."
        
        cv2.putText(vis_image, status_text, (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)
        
        cv2.putText(vis_image, f"Frame: {self.frame_count}", (20, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis_image, f"Detected frames: {self.detected_frames}", (20, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis_image, f"Confirmations: {confirmations}/{self.history_size}", (20, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(vis_image, f"Red bars: {red_count} | Green bars: {green_count}", (20, 180),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Center crosshair
        cv2.line(vis_image, (w//2 - 30, h//2), (w//2 + 30, h//2), (255, 255, 0), 2)
        cv2.line(vis_image, (w//2, h//2 - 30), (w//2, h//2 + 30), (255, 255, 0), 2)
    
    def publish_gate_data(self, gate_info, header):
        """Publish all gate detection data"""
        # Pose
        pose_msg = PoseStamped()
        pose_msg.header = header
        pose_msg.header.frame_id = 'camera_forward'
        pose_msg.pose.position.x = float(gate_info['center'][0])
        pose_msg.pose.position.y = float(gate_info['center'][1])
        pose_msg.pose.position.z = gate_info['distance']
        pose_msg.pose.orientation.w = 1.0
        self.gate_center_pub.publish(pose_msg)
        
        # Alignment
        alignment_msg = Float32()
        alignment_msg.data = gate_info['alignment_error']
        self.gate_alignment_pub.publish(alignment_msg)
        
        # Distance
        distance_msg = Float32()
        distance_msg.data = gate_info['distance']
        self.gate_distance_pub.publish(distance_msg)
    
    def __del__(self):
        if self.show_window:
            cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = VisualGateDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()