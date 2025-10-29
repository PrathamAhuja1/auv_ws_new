#!/usr/bin/env python3
"""
Drum Detector Node - Detects colored drums for Target Acquisition task
Detects blue drum and red drums, publishes their positions
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Bool, Float32
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, String
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy


class DrumDetectorNode(Node):
    def __init__(self):
        super().__init__('drum_detector_node')
        self.bridge = CvBridge()
        self.camera_matrix = None
        self.image_width = None
        self.image_height = None
        
        # Parameters
        self.declare_parameter('min_contour_area', 800)
        self.declare_parameter('drum_aspect_ratio_min', 0.8)
        self.declare_parameter('drum_aspect_ratio_max', 1.2)
        self.declare_parameter('publish_debug', True)
        
        self.min_area = self.get_parameter('min_contour_area').value
        self.aspect_min = self.get_parameter('drum_aspect_ratio_min').value
        self.aspect_max = self.get_parameter('drum_aspect_ratio_max').value
        self.publish_debug = self.get_parameter('publish_debug').value
        
        # HSV color ranges for drums (60cm diameter cylinders)
        # Blue drum
        self.blue_lower = np.array([100, 150, 50])
        self.blue_upper = np.array([130, 255, 255])
        
        # Red drum (handle red wraparound)
        self.red_lower1 = np.array([0, 120, 70])
        self.red_upper1 = np.array([10, 255, 255])
        self.red_lower2 = np.array([170, 120, 70])
        self.red_upper2 = np.array([180, 255, 255])
        
        # Known drum dimensions
        self.DRUM_DIAMETER = 0.6  # meters
        
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
            '/camera_down/image_raw',  # Downward camera for drums
            self.image_callback,
            qos_sensor
        )
        
        self.cam_info_sub = self.create_subscription(
            CameraInfo,
            '/camera_down/camera_info',
            self.cam_info_callback,
            qos_reliable
        )
        
        # Publishers
        self.blue_drum_detected_pub = self.create_publisher(
            Bool, '/drum/blue_detected', 10)
        self.blue_drum_pose_pub = self.create_publisher(
            PoseStamped, '/drum/blue_pose', 10)
        self.blue_drum_distance_pub = self.create_publisher(
            Float32, '/drum/blue_distance', 10)
        
        self.red_drum_detected_pub = self.create_publisher(
            Bool, '/drum/red_detected', 10)
        self.red_drum_pose_pub = self.create_publisher(
            PoseStamped, '/drum/red_pose', 10)
        
        self.closest_drum_pub = self.create_publisher(
            String, '/drum/closest_color', 10)
        
        if self.publish_debug:
            self.debug_pub = self.create_publisher(Image, '/drum/debug_image', 10)
        
        self.get_logger().info('Drum Detector Node initialized')
    
    def cam_info_callback(self, msg: CameraInfo):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape((3, 3))
            self.image_width = msg.width
            self.image_height = msg.height
            self.fx = self.camera_matrix[0, 0]
            self.fy = self.camera_matrix[1, 1]
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
        
        # Detect drums
        blue_drum = self.detect_drum(
            hsv_image, self.blue_lower, self.blue_upper,
            None, None, debug_img, (255, 0, 0), "BLUE DRUM"
        )
        
        red_drums = self.detect_drum(
            hsv_image, self.red_lower1, self.red_upper1,
            self.red_lower2, self.red_upper2, debug_img, (0, 0, 255), "RED DRUM",
            find_all=True  # Find all red drums
        )
        
        # Publish blue drum
        blue_detected = Bool()
        blue_detected.data = blue_drum is not None
        self.blue_drum_detected_pub.publish(blue_detected)
        
        if blue_drum:
            self.publish_drum_pose(blue_drum, 'blue', msg.header)
        
        # Publish red drums
        red_detected = Bool()
        red_detected.data = len(red_drums) > 0
        self.red_drum_detected_pub.publish(red_detected)
        
        if red_drums:
            # Publish closest red drum
            closest_red = min(red_drums, key=lambda d: d['distance'])
            self.publish_drum_pose(closest_red, 'red', msg.header)
        
        # Determine closest drum overall
        all_drums = []
        if blue_drum:
            all_drums.append(('blue', blue_drum))
        for red in red_drums:
            all_drums.append(('red', red))
        
        if all_drums:
            closest_color, _ = min(all_drums, key=lambda x: x[1]['distance'])
            closest_msg = String()
            closest_msg.data = closest_color
            self.closest_drum_pub.publish(closest_msg)
        
        # Publish debug image
        if self.publish_debug and debug_img is not None:
            try:
                debug_msg = self.bridge.cv2_to_imgmsg(debug_img, "bgr8")
                debug_msg.header = msg.header
                self.debug_pub.publish(debug_msg)
            except CvBridgeError as e:
                self.get_logger().error(f'Debug image error: {e}')
    
    def detect_drum(self, hsv_image, lower1, upper1, lower2, upper2, 
                    debug_img, color, label, find_all=False):
        """Detect drum(s) of specific color"""
        
        # Create mask
        mask1 = cv2.inRange(hsv_image, lower1, upper1)
        if lower2 is not None and upper2 is not None:
            mask2 = cv2.inRange(hsv_image, lower2, upper2)
            mask = mask1 | mask2
        else:
            mask = mask1
        
        # Morphological operations
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_drums = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue
            
            # Check circularity (drums are circular from above)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            if circularity < 0.6:  # Not circular enough
                continue
            
            # Get bounding circle
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            cx, cy = int(cx), int(cy)
            radius = int(radius)
            
            # Estimate distance using known drum diameter
            drum_diameter_pixels = radius * 2
            if drum_diameter_pixels > 0:
                distance = (self.DRUM_DIAMETER * self.fx) / drum_diameter_pixels
            else:
                distance = 999.0
            
            drum_info = {
                'center': (cx, cy),
                'radius': radius,
                'area': area,
                'circularity': circularity,
                'distance': distance
            }
            
            detected_drums.append(drum_info)
            
            if debug_img is not None:
                cv2.circle(debug_img, (cx, cy), radius, color, 3)
                cv2.circle(debug_img, (cx, cy), 5, color, -1)
                cv2.putText(debug_img, f"{label} {distance:.1f}m", 
                           (cx - 50, cy - radius - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        if find_all:
            return detected_drums
        else:
            return detected_drums[0] if detected_drums else None
    
    def publish_drum_pose(self, drum_info, color, header):
        """Publish drum detection as PoseStamped"""
        pose_msg = PoseStamped()
        pose_msg.header = header
        pose_msg.header.frame_id = 'camera_down'
        
        pose_msg.pose.position.x = float(drum_info['center'][0])
        pose_msg.pose.position.y = float(drum_info['center'][1])
        pose_msg.pose.position.z = float(drum_info['distance'])
        pose_msg.pose.orientation.w = 1.0
        
        if color == 'blue':
            self.blue_drum_pose_pub.publish(pose_msg)
            
            distance_msg = Float32()
            distance_msg.data = drum_info['distance']
            self.blue_drum_distance_pub.publish(distance_msg)
        elif color == 'red':
            self.red_drum_pose_pub.publish(pose_msg)


def main(args=None):
    rclpy.init(args=args)
    node = DrumDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()