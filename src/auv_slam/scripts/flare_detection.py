#!/usr/bin/env python3
"""
Multi-Flare Detector Node - Detects Red, Yellow, and Blue flares for Task 4
Publishes individual detection topics for each flare color
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy


class FlareDetectorNode(Node):
    def __init__(self):
        super().__init__('flare_detector_node')
        self.bridge = CvBridge()
        self.camera_matrix = None
        self.image_width = None
        self.image_height = None
        
        # Parameters
        self.declare_parameter('min_contour_area', 300)
        self.declare_parameter('min_aspect_ratio', 3.0)  # Flares are tall and thin
        self.declare_parameter('publish_debug', True)
        
        self.min_area = self.get_parameter('min_contour_area').value
        self.min_aspect = self.get_parameter('min_aspect_ratio').value
        self.publish_debug = self.get_parameter('publish_debug').value
        
        # HSV color ranges for flares
        # Red flare (80cm tall, ~1.6cm diameter)
        self.red_lower1 = np.array([0, 120, 70])
        self.red_upper1 = np.array([10, 255, 255])
        self.red_lower2 = np.array([170, 120, 70])
        self.red_upper2 = np.array([180, 255, 255])
        
        # Yellow flare
        self.yellow_lower = np.array([20, 100, 100])
        self.yellow_upper = np.array([30, 255, 255])
        
        # Blue flare
        self.blue_lower = np.array([100, 150, 50])
        self.blue_upper = np.array([130, 255, 255])
        
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
        
        # Publishers for each flare color
        self.red_detected_pub = self.create_publisher(Bool, '/flare/red_detected_bool', 10)
        self.red_pose_pub = self.create_publisher(PoseStamped, '/flare/red_detected', 10)
        
        self.yellow_detected_pub = self.create_publisher(Bool, '/flare/yellow_detected_bool', 10)
        self.yellow_pose_pub = self.create_publisher(PoseStamped, '/flare/yellow_detected', 10)
        
        self.blue_detected_pub = self.create_publisher(Bool, '/flare/blue_detected_bool', 10)
        self.blue_pose_pub = self.create_publisher(PoseStamped, '/flare/blue_detected', 10)
        
        if self.publish_debug:
            self.debug_pub = self.create_publisher(Image, '/flare/debug_image', 10)
        
        self.get_logger().info('Multi-Flare Detector Node initialized')
    
    def cam_info_callback(self, msg: CameraInfo):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape((3, 3))
            self.image_width = msg.width
            self.image_height = msg.height
            self.get_logger().info(f'Camera info received: {self.image_width}x{self.image_height}')
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
        
        # Detect each flare color
        red_flare = self.detect_flare(
            hsv_image, self.red_lower1, self.red_upper1, 
            self.red_lower2, self.red_upper2, debug_img, (0, 0, 255), "RED"
        )
        
        yellow_flare = self.detect_flare(
            hsv_image, self.yellow_lower, self.yellow_upper,
            None, None, debug_img, (0, 255, 255), "YELLOW"
        )
        
        blue_flare = self.detect_flare(
            hsv_image, self.blue_lower, self.blue_upper,
            None, None, debug_img, (255, 0, 0), "BLUE"
        )
        
        # Publish detections
        self.publish_flare_detection(red_flare, 'red', 
            self.red_detected_pub, self.red_pose_pub, msg.header)
        self.publish_flare_detection(yellow_flare, 'yellow',
            self.yellow_detected_pub, self.yellow_pose_pub, msg.header)
        self.publish_flare_detection(blue_flare, 'blue',
            self.blue_detected_pub, self.blue_pose_pub, msg.header)
        
        # Publish debug image
        if self.publish_debug and debug_img is not None:
            try:
                debug_msg = self.bridge.cv2_to_imgmsg(debug_img, "bgr8")
                debug_msg.header = msg.header
                self.debug_pub.publish(debug_msg)
            except CvBridgeError as e:
                self.get_logger().error(f'Debug image error: {e}')
    
    def detect_flare(self, hsv_image, lower1, upper1, lower2, upper2, 
                     debug_img, color, label):
        """Detect a flare of specific color"""
        
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
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        flare_info = None
        max_area = 0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(h) / w if w > 0 else 0
            
            if aspect_ratio > self.min_aspect and area > max_area:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    max_area = area
                    flare_info = {
                        'center': (cx, cy),
                        'bbox': (x, y, w, h),
                        'area': area
                    }
                    
                    if debug_img is not None:
                        cv2.rectangle(debug_img, (x, y), (x + w, y + h), color, 2)
                        cv2.circle(debug_img, (cx, cy), 5, color, -1)
                        cv2.putText(debug_img, label, (x, y - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return flare_info
    
    def publish_flare_detection(self, flare_info, color_name, bool_pub, pose_pub, header):
        """Publish detection status and pose"""
        detected = Bool()
        detected.data = flare_info is not None
        bool_pub.publish(detected)
        
        if flare_info:
            pose_msg = PoseStamped()
            pose_msg.header = header
            pose_msg.header.frame_id = 'camera_forward'
            pose_msg.pose.position.x = float(flare_info['center'][0])
            pose_msg.pose.position.y = float(flare_info['center'][1])
            pose_msg.pose.orientation.w = 1.0
            pose_pub.publish(pose_msg)


def main(args=None):
    rclpy.init(args=args)
    node = FlareDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()