#!/usr/bin/env python3
"""
FILE: qualification_detector.py
Purpose: Robust detection of the Orange/Red Qualification Gate.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Bool, Float32
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from collections import deque

class QualificationDetector(Node):
    def __init__(self):
        super().__init__('qualification_detector')
        self.bridge = CvBridge()
        self.fx = 300.0  # Default focal length, will update from camera_info
        
        # --- QUALIFICATION GATE COLOR (Orange/Red) ---
        # We use two ranges to cover both orange and the red wrap-around in HSV
        self.lower1 = np.array([0, 100, 50])
        self.upper1 = np.array([25, 255, 255])
        
        self.lower2 = np.array([160, 100, 50])
        self.upper2 = np.array([180, 255, 255])
        
        self.min_area = 500  # Minimum pixel area to count as a gate
        
        # Stability Buffer: Require 2 consecutive frames to confirm detection
        self.history = deque(maxlen=5)
        self.min_confirmations = 2
        
        # --- SUBSCRIBERS ---
        # We listen to 'image_raw' (relative) so we can remap it in the launch file
        self.create_subscription(Image, 'image_raw', self.image_callback, 10)
        self.create_subscription(CameraInfo, 'camera_info', self.info_callback, 10)
        
        # --- PUBLISHERS ---
        # These topics match what qualification_navigator.py listens for
        self.detect_pub = self.create_publisher(Bool, '/gate/detected', 10)
        self.pos_pub = self.create_publisher(Float32, '/gate/frame_position', 10)
        self.dist_pub = self.create_publisher(Float32, '/gate/estimated_distance', 10)
        self.debug_pub = self.create_publisher(Image, '/gate/debug_image', 10)
        
        self.get_logger().info('✅ Qualification Detector Initialized')

    def info_callback(self, msg):
        # Update focal length for accurate distance estimation
        if msg.k:
            k = np.array(msg.k).reshape((3, 3))
            self.fx = k[0, 0]

    def image_callback(self, msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError:
            return

        h, w = cv_img.shape[:2]
        hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
        
        # 1. Color Thresholding
        mask1 = cv2.inRange(hsv, self.lower1, self.upper1)
        mask2 = cv2.inRange(hsv, self.lower2, self.upper2)
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Clean up noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 2. Find Contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        gate_found = False
        frame_pos = 0.0
        distance = 999.0
        
        # Filter for significant objects only
        valid_cnts = [c for c in contours if cv2.contourArea(c) > self.min_area]
        
        if valid_cnts:
            gate_found = True
            
            # Combine all orange parts into one bounding box (handles interrupted legs)
            all_pts = np.concatenate(valid_cnts)
            x, y, bw, bh = cv2.boundingRect(all_pts)
            cx = x + bw // 2
            cy = y + bh // 2
            
            # Calculate Position: -1.0 (Left) to +1.0 (Right), 0.0 is Center
            frame_pos = (cx - w/2) / (w/2)
            
            # Estimate Distance: (Real Width * Focal Length) / Pixel Width
            # Assuming gate is approx 1.5m wide
            if bw > 0:
                distance = (1.5 * self.fx) / bw
                distance = max(0.5, min(distance, 15.0))
            
            # Visual Debugging
            cv2.rectangle(cv_img, (x, y), (x+bw, y+bh), (0, 255, 0), 3)
            cv2.circle(cv_img, (cx, cy), 10, (0, 0, 255), -1)
            cv2.putText(cv_img, f"GATE {distance:.1f}m", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(cv_img, "SEARCHING...", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # 3. Publish Data
        self.history.append(gate_found)
        # Require stable detection to avoid glitching
        is_stable = sum(self.history) >= self.min_confirmations
        
        self.detect_pub.publish(Bool(data=is_stable))
        
        if is_stable:
            self.pos_pub.publish(Float32(data=float(frame_pos)))
            self.dist_pub.publish(Float32(data=float(distance)))
            
        # Publish Debug Image
        try:
            self.debug_pub.publish(self.bridge.cv2_to_imgmsg(cv_img, "bgr8"))
        except:
            pass

def main(args=None):
    rclpy.init(args=args)
    node = QualificationDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()