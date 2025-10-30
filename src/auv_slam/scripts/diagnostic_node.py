#!/usr/bin/env python3
"""
Interactive HSV Color Tuner for Gate Detection
Use this to find the perfect HSV ranges for your Gazebo environment

Usage:
    ros2 run auv_slam hsv_tuner.py
    
Then use trackbars to adjust HSV ranges and see results in real-time
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np


class HSVTuner(Node):
    def __init__(self):
        super().__init__('hsv_tuner')
        self.bridge = CvBridge()
        self.current_image = None
        
        # Subscribe to camera
        self.image_sub = self.create_subscription(
            Image,
            '/camera_forward/image_raw',
            self.image_callback,
            10
        )
        
        # Create windows
        cv2.namedWindow('Original')
        cv2.namedWindow('HSV Tuning')
        cv2.namedWindow('Red Mask')
        cv2.namedWindow('Green Mask')
        
        # Initial HSV ranges (permissive defaults)
        self.create_trackbars()
        
        self.get_logger().info('='*70)
        self.get_logger().info('🎨 HSV Color Tuner Started')
        self.get_logger().info('   Adjust trackbars to find perfect HSV ranges')
        self.get_logger().info('   Press Q to quit')
        self.get_logger().info('='*70)
        
        # Processing timer
        self.timer = self.create_timer(0.1, self.process_image)
    
    def create_trackbars(self):
        """Create trackbars for HSV adjustment"""
        # Red channel (low H)
        cv2.createTrackbar('R1_H_min', 'HSV Tuning', 0, 180, lambda x: None)
        cv2.createTrackbar('R1_H_max', 'HSV Tuning', 30, 180, lambda x: None)
        cv2.createTrackbar('R1_S_min', 'HSV Tuning', 15, 255, lambda x: None)
        cv2.createTrackbar('R1_V_min', 'HSV Tuning', 30, 255, lambda x: None)
        
        # Red channel (high H)
        cv2.createTrackbar('R2_H_min', 'HSV Tuning', 140, 180, lambda x: None)
        cv2.createTrackbar('R2_H_max', 'HSV Tuning', 180, 180, lambda x: None)
        
        # Green channel
        cv2.createTrackbar('G_H_min', 'HSV Tuning', 20, 180, lambda x: None)
        cv2.createTrackbar('G_H_max', 'HSV Tuning', 120, 180, lambda x: None)
        cv2.createTrackbar('G_S_min', 'HSV Tuning', 10, 255, lambda x: None)
        cv2.createTrackbar('G_V_min', 'HSV Tuning', 25, 255, lambda x: None)
        
        # Morphology
        cv2.createTrackbar('Morph_kernel', 'HSV Tuning', 3, 15, lambda x: None)
        
        # Min area
        cv2.createTrackbar('Min_area', 'HSV Tuning', 15, 500, lambda x: None)
    
    def image_callback(self, msg):
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'Image conversion error: {e}')
    
    def process_image(self):
        """Process image with current trackbar values"""
        if self.current_image is None:
            return
        
        # Get trackbar values
        r1_h_min = cv2.getTrackbarPos('R1_H_min', 'HSV Tuning')
        r1_h_max = cv2.getTrackbarPos('R1_H_max', 'HSV Tuning')
        r1_s_min = cv2.getTrackbarPos('R1_S_min', 'HSV Tuning')
        r1_v_min = cv2.getTrackbarPos('R1_V_min', 'HSV Tuning')
        
        r2_h_min = cv2.getTrackbarPos('R2_H_min', 'HSV Tuning')
        r2_h_max = cv2.getTrackbarPos('R2_H_max', 'HSV Tuning')
        
        g_h_min = cv2.getTrackbarPos('G_H_min', 'HSV Tuning')
        g_h_max = cv2.getTrackbarPos('G_H_max', 'HSV Tuning')
        g_s_min = cv2.getTrackbarPos('G_S_min', 'HSV Tuning')
        g_v_min = cv2.getTrackbarPos('G_V_min', 'HSV Tuning')
        
        morph_size = max(1, cv2.getTrackbarPos('Morph_kernel', 'HSV Tuning'))
        min_area = cv2.getTrackbarPos('Min_area', 'HSV Tuning')
        
        # Convert to HSV
        hsv = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2HSV)
        
        # Create masks
        red_lower1 = np.array([r1_h_min, r1_s_min, r1_v_min])
        red_upper1 = np.array([r1_h_max, 255, 255])
        red_lower2 = np.array([r2_h_min, r1_s_min, r1_v_min])
        red_upper2 = np.array([r2_h_max, 255, 255])
        
        green_lower = np.array([g_h_min, g_s_min, g_v_min])
        green_upper = np.array([g_h_max, 255, 255])
        
        red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        
        # Morphology
        kernel = np.ones((morph_size, morph_size), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw on original
        result = self.current_image.copy()
        
        red_count = 0
        for cnt in red_contours:
            area = cv2.contourArea(cnt)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(result, f"R:{int(area)}", (x, y-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                red_count += 1
        
        green_count = 0
        for cnt in green_contours:
            area = cv2.contourArea(cnt)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(result, f"G:{int(area)}", (x, y-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                green_count += 1
        
        # Add stats
        stats_text = [
            f"Red: {cv2.countNonZero(red_mask)}px ({red_count} valid)",
            f"Green: {cv2.countNonZero(green_mask)}px ({green_count} valid)",
            f"Min area: {min_area}",
        ]
        
        y_offset = 30
        for text in stats_text:
            cv2.putText(result, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 30
        
        # Print current ranges to console
        if red_count > 0 or green_count > 0:
            self.get_logger().info(
                f"Red[{r1_h_min}-{r1_h_max}, {r2_h_min}-{r2_h_max}] "
                f"Green[{g_h_min}-{g_h_max}] | "
                f"Valid: R={red_count} G={green_count}",
                throttle_duration_sec=1.0
            )
        
        # Show results
        cv2.imshow('Original', self.current_image)
        cv2.imshow('HSV Tuning', result)
        cv2.imshow('Red Mask', red_mask)
        cv2.imshow('Green Mask', green_mask)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.print_final_values()
            cv2.destroyAllWindows()
            rclpy.shutdown()
    
    def print_final_values(self):
        """Print final HSV values for code"""
        r1_h_min = cv2.getTrackbarPos('R1_H_min', 'HSV Tuning')
        r1_h_max = cv2.getTrackbarPos('R1_H_max', 'HSV Tuning')
        r1_s_min = cv2.getTrackbarPos('R1_S_min', 'HSV Tuning')
        r1_v_min = cv2.getTrackbarPos('R1_V_min', 'HSV Tuning')
        
        r2_h_min = cv2.getTrackbarPos('R2_H_min', 'HSV Tuning')
        r2_h_max = cv2.getTrackbarPos('R2_H_max', 'HSV Tuning')
        
        g_h_min = cv2.getTrackbarPos('G_H_min', 'HSV Tuning')
        g_h_max = cv2.getTrackbarPos('G_H_max', 'HSV Tuning')
        g_s_min = cv2.getTrackbarPos('G_S_min', 'HSV Tuning')
        g_v_min = cv2.getTrackbarPos('G_V_min', 'HSV Tuning')
        
        min_area = cv2.getTrackbarPos('Min_area', 'HSV Tuning')
        
        self.get_logger().info('='*70)
        self.get_logger().info('📋 FINAL HSV VALUES - Copy to gate_detector_node.py:')
        self.get_logger().info('='*70)
        print(f"""
# RED channel
self.red_lower1 = np.array([{r1_h_min}, {r1_s_min}, {r1_v_min}])
self.red_upper1 = np.array([{r1_h_max}, 255, 255])
self.red_lower2 = np.array([{r2_h_min}, {r1_s_min}, {r1_v_min}])
self.red_upper2 = np.array([{r2_h_max}, 255, 255])

# GREEN channel
self.green_lower = np.array([{g_h_min}, {g_s_min}, {g_v_min}])
self.green_upper = np.array([{g_h_max}, 255, 255])

# Detection parameters
self.min_area_strict = {min_area}
""")
        self.get_logger().info('='*70)


def main(args=None):
    rclpy.init(args=args)
    node = HSVTuner()
    
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