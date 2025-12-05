#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

class QualificationDetector(Node):
    def __init__(self):
        super().__init__('qualification_detector')
        self.bridge = CvBridge()
        
        # --- COLOR TUNING FOR YOUR IMAGE ---
        # The gate in the image is bright Yellow-Orange.
        # Hue: 0-40 covers Red to Yellow.
        # Saturation: Lowered to 60 because water desaturates color.
        self.lower_orange = np.array([0, 60, 60])
        self.upper_orange = np.array([40, 255, 255])
        
        # --- SUBSCRIBERS ---
        # CRITICAL: qos_profile_sensor_data fixes the 'Blank Image' issue
        # by matching the simulator's "Best Effort" setting.
        self.create_subscription(
            Image, 
            'image_raw', 
            self.image_callback, 
            qos_profile_sensor_data
        )
        
        # --- PUBLISHERS ---
        self.detect_pub = self.create_publisher(Bool, '/gate/detected', 10)
        self.pos_pub = self.create_publisher(Float32, '/gate/frame_position', 10)
        self.debug_pub = self.create_publisher(Image, '/gate/debug_image', 10)
        
        self.get_logger().info('✅ Detector Started - Waiting for camera stream...')

    def image_callback(self, msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f'CV Bridge Error: {e}')
            return

        # 1. Convert to HSV
        hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
        
        # 2. Create Mask based on the new Orange/Yellow thresholds
        mask = cv2.inRange(hsv, self.lower_orange, self.upper_orange)
        
        # 3. Clean noise (remove small white speckles from water)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)

        # 4. Find Contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        gate_found = False
        target_x = 0.0
        
        # Find the largest object (The Gate)
        if contours:
            largest_cnt = max(contours, key=cv2.contourArea)
            
            if cv2.contourArea(largest_cnt) > 1000: # Filter out small noise
                gate_found = True
                x, y, w, h = cv2.boundingRect(largest_cnt)
                
                # Calculate center relative to image width (-1.0 to 1.0)
                img_center_x = cv_img.shape[1] // 2
                obj_center_x = x + (w // 2)
                target_x = (obj_center_x - img_center_x) / (cv_img.shape[1] / 2)
                
                # Draw Box and Center Point
                cv2.rectangle(cv_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
                cv2.circle(cv_img, (obj_center_x, y + h//2), 5, (0, 0, 255), -1)
                cv2.putText(cv_img, "GATE DETECTED", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # 5. Publish Status
        self.detect_pub.publish(Bool(data=gate_found))
        if gate_found:
            self.pos_pub.publish(Float32(data=float(target_x)))

        # 6. Publish Debug Image (ALWAYS publish this so you don't get a blank screen)
        try:
            self.debug_pub.publish(self.bridge.cv2_to_imgmsg(cv_img, "bgr8"))
        except CvBridgeError:
            pass

def main(args=None):
    rclpy.init(args=args)
    node = QualificationDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()