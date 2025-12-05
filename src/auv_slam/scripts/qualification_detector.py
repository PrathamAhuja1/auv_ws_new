#!/usr/bin/env python3
"""
FIXED Qualification Detector - Properly loads config parameters
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from collections import deque

class QualificationDetector(Node):
    def __init__(self):
        super().__init__('qualification_gate_detector')
        self.bridge = CvBridge()
        
        # CRITICAL FIX: Actually declare and load parameters from config
        self.declare_parameter('orange_lower', [0, 50, 50])
        self.declare_parameter('orange_upper', [30, 255, 255])
        self.declare_parameter('min_area_strict', 300)
        self.declare_parameter('gate_width', 1.5)
        self.declare_parameter('publish_debug', True)
        
        # Load parameters
        orange_lower_list = self.get_parameter('orange_lower').value
        orange_upper_list = self.get_parameter('orange_upper').value
        self.min_area = self.get_parameter('min_area_strict').value
        self.gate_width = self.get_parameter('gate_width').value
        
        self.lower_orange = np.array(orange_lower_list)
        self.upper_orange = np.array(orange_upper_list)
        
        # Detection stability
        self.history = deque(maxlen=5)
        self.min_confirmations = 2
        
        # Camera info
        self.fx = 300.0  # Will be updated from camera_info
        
        self.get_logger().info(f'✅ Detector Started')
        self.get_logger().info(f'   Orange HSV: {self.lower_orange} to {self.upper_orange}')
        self.get_logger().info(f'   Min area: {self.min_area}')
        
        # Subscribers
        self.create_subscription(
            Image, 
            'image_raw', 
            self.image_callback, 
            qos_profile_sensor_data
        )
        
        # Publishers
        self.detect_pub = self.create_publisher(Bool, '/gate/detected', 10)
        self.pos_pub = self.create_publisher(Float32, '/gate/frame_position', 10)
        self.dist_pub = self.create_publisher(Float32, '/gate/estimated_distance', 10)
        self.debug_pub = self.create_publisher(Image, '/gate/debug_image', 10)

    def image_callback(self, msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f'Bridge Error: {e}')
            return

        h, w = cv_img.shape[:2]
        hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
        
        # Create mask
        mask = cv2.inRange(hsv, self.lower_orange, self.upper_orange)
        
        # Clean noise (less aggressive)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Debug: Draw mask stats
        debug_img = cv_img.copy()
        mask_pixels = cv2.countNonZero(mask)
        cv2.putText(debug_img, f"Mask pixels: {mask_pixels}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(debug_img, f"Contours: {len(contours)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        gate_found = False
        target_x = 0.0
        distance = 999.0
        
        # Find valid gate parts
        valid_contours = [c for c in contours if cv2.contourArea(c) > self.min_area]
        
        if valid_contours:
            # Combine all gate parts
            all_pts = np.concatenate(valid_contours)
            x, y, bw, bh = cv2.boundingRect(all_pts)
            
            gate_found = True
            
            # Calculate center
            img_center_x = w // 2
            obj_center_x = x + (bw // 2)
            obj_center_y = y + (bh // 2)
            target_x = (obj_center_x - img_center_x) / (w / 2)
            
            # Estimate distance
            if bw > 20:
                distance = (self.gate_width * self.fx) / bw
                distance = max(0.5, min(distance, 15.0))
            
            # Draw detection
            cv2.rectangle(debug_img, (x, y), (x+bw, y+bh), (0, 255, 0), 3)
            cv2.circle(debug_img, (obj_center_x, obj_center_y), 10, (0, 0, 255), -1)
            cv2.line(debug_img, (obj_center_x, 0), (obj_center_x, h), (0, 255, 0), 2)
            
            cv2.putText(debug_img, f"GATE DETECTED", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            cv2.putText(debug_img, f"Dist: {distance:.1f}m", (x, y+bh+30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(debug_img, f"Pos: {target_x:+.2f}", (x, y+bh+60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Draw individual parts
            for i, cnt in enumerate(valid_contours):
                area = cv2.contourArea(cnt)
                cx, cy, cw, ch = cv2.boundingRect(cnt)
                cv2.putText(debug_img, f"A:{int(area)}", (cx, cy-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        else:
            cv2.putText(debug_img, "NO GATE DETECTED", (w//2-150, h//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
        # Draw center line
        cv2.line(debug_img, (w//2, 0), (w//2, h), (255, 255, 0), 2)
        
        # Temporal filtering
        self.history.append(gate_found)
        is_stable = sum(self.history) >= self.min_confirmations
        
        # Publish
        self.detect_pub.publish(Bool(data=is_stable))
        if is_stable:
            self.pos_pub.publish(Float32(data=float(target_x)))
            self.dist_pub.publish(Float32(data=float(distance)))
        
        # Always publish debug image
        try:
            # Show mask in corner
            mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            mask_small = cv2.resize(mask_colored, (w//4, h//4))
            debug_img[10:10+h//4, w-w//4-10:w-10] = mask_small
            
            self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug_img, "bgr8"))
        except:
            pass

def main(args=None):
    rclpy.init(args=args)
    node = QualificationDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()