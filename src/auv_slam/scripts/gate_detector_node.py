#!/usr/bin/env python3

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
        
        # HSV RANGES FOR PURE RED AND GREEN IN GAZEBO
        self.red_lower1 = np.array([0, 100, 100])
        self.red_upper1 = np.array([10, 255, 255])
        self.red_lower2 = np.array([170, 100, 100])
        self.red_upper2 = np.array([180, 255, 255])
        
        self.green_lower = np.array([40, 100, 100])
        self.green_upper = np.array([80, 255, 255])
        
        self.orange_lower = np.array([10, 120, 120])
        self.orange_upper = np.array([25, 255, 255])
        
        # Detection parameters
        self.min_area_strict = 300
        self.min_area_relaxed = 80
        self.aspect_threshold = 0.8
        self.gate_width = 1.5
        
        self.gate_detection_history = deque(maxlen=5)
        self.min_confirmations = 2
        
        self.frame_count = 0
        self.detection_stats = {
            'red_pixels': 0,
            'green_pixels': 0,
            'orange_pixels': 0,
            'red_contours': 0,
            'green_contours': 0,
            'orange_contours': 0
        }
        
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
        
        self.gate_detected_pub = self.create_publisher(Bool, '/gate/detected', 10)
        self.alignment_pub = self.create_publisher(Float32, '/gate/alignment_error', 10)
        self.distance_pub = self.create_publisher(Float32, '/gate/estimated_distance', 10)
        self.gate_center_pub = self.create_publisher(Point, '/gate/center_point', 10)
        self.debug_pub = self.create_publisher(Image, '/gate/debug_image', 10)
        self.status_pub = self.create_publisher(String, '/gate/status', 10)
        
        self.flare_detected_pub = self.create_publisher(Bool, '/flare/detected', 10)
        self.flare_direction_pub = self.create_publisher(Float32, '/flare/avoidance_direction', 10)
        self.flare_warning_pub = self.create_publisher(String, '/flare/warning', 10)
        
        self.get_logger().info('Gate Detector initialized - HSV optimized for Gazebo')
        self.get_logger().info('View debug: rqt_image_view /gate/debug_image')
    
    def cam_info_callback(self, msg: CameraInfo):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape((3, 3))
            self.image_width = msg.width
            self.image_height = msg.height
            self.fx = self.camera_matrix[0, 0]
            self.fy = self.camera_matrix[1, 1]
            self.cx = self.camera_matrix[0, 2]
            self.cy = self.camera_matrix[1, 2]
            self.get_logger().info(f'Camera: {self.image_width}x{self.image_height}')
    
    def image_callback(self, msg: Image):
        if self.camera_matrix is None:
            return
        
        self.frame_count += 1
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        except CvBridgeError as e:
            self.get_logger().error(f'CV Bridge error: {e}')
            return
        
        debug_img = cv_image.copy()
        h, w = cv_image.shape[:2]
        
        # Create masks
        red_mask1 = cv2.inRange(hsv_image, self.red_lower1, self.red_upper1)
        red_mask2 = cv2.inRange(hsv_image, self.red_lower2, self.red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        green_mask = cv2.inRange(hsv_image, self.green_lower, self.green_upper)
        orange_mask = cv2.inRange(hsv_image, self.orange_lower, self.orange_upper)
        
        red_pixels = cv2.countNonZero(red_mask)
        green_pixels = cv2.countNonZero(green_mask)
        orange_pixels = cv2.countNonZero(orange_mask)
        
        self.detection_stats['red_pixels'] = red_pixels
        self.detection_stats['green_pixels'] = green_pixels
        self.detection_stats['orange_pixels'] = orange_pixels
        
        # Morphological operations
        kernel = np.ones((3, 3), np.uint8)
        
        red_mask_clean = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        red_mask_clean = cv2.morphologyEx(red_mask_clean, cv2.MORPH_OPEN, kernel)
        
        green_mask_clean = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
        green_mask_clean = cv2.morphologyEx(green_mask_clean, cv2.MORPH_OPEN, kernel)
        
        orange_mask_clean = cv2.morphologyEx(orange_mask, cv2.MORPH_CLOSE, kernel)
        orange_mask_clean = cv2.morphologyEx(orange_mask_clean, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        red_contours, _ = cv2.findContours(red_mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        green_contours, _ = cv2.findContours(green_mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        orange_contours, _ = cv2.findContours(orange_mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        self.detection_stats['red_contours'] = len(red_contours)
        self.detection_stats['green_contours'] = len(green_contours)
        self.detection_stats['orange_contours'] = len(orange_contours)
        
        # Detect orange flare
        flare_detected = False
        flare_center_x = None
        
        if orange_contours:
            largest_orange = max(orange_contours, key=cv2.contourArea)
            orange_area = cv2.contourArea(largest_orange)
            
            if orange_area > 500:
                M = cv2.moments(largest_orange)
                if M["m00"] > 0:
                    flare_center_x = int(M["m10"] / M["m00"])
                    flare_center_y = int(M["m01"] / M["m00"])
                    
                    flare_detected = True
                    
                    cv2.circle(debug_img, (flare_center_x, flare_center_y), 30, (0, 140, 255), 5)
                    cv2.putText(debug_img, "FLARE", (flare_center_x - 50, flare_center_y - 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 140, 255), 3)
                    
                    image_center_x = w / 2
                    if flare_center_x < image_center_x:
                        avoidance_direction = -1.0
                    else:
                        avoidance_direction = 1.0
                    
                    flare_msg = Bool()
                    flare_msg.data = True
                    self.flare_detected_pub.publish(flare_msg)
                    
                    dir_msg = Float32()
                    dir_msg.data = avoidance_direction
                    self.flare_direction_pub.publish(dir_msg)
                    
                    warn_msg = String()
                    warn_msg.data = f"CRITICAL: Orange flare at X={flare_center_x}"
                    self.flare_warning_pub.publish(warn_msg)
        
        if not flare_detected:
            flare_msg = Bool()
            flare_msg.data = False
            self.flare_detected_pub.publish(flare_msg)
        
        # Detect gate stripes
        red_stripe = self.find_best_stripe(red_contours, debug_img, (0, 0, 255), 
                                           "RED", self.min_area_strict, strict=True)
        green_stripe = self.find_best_stripe(green_contours, debug_img, (0, 255, 0), 
                                             "GREEN", self.min_area_strict, strict=True)
        
        if not red_stripe:
            red_stripe = self.find_best_stripe(red_contours, debug_img, (100, 0, 200), 
                                               "RED*", self.min_area_relaxed, strict=False)
        if not green_stripe:
            green_stripe = self.find_best_stripe(green_contours, debug_img, (0, 200, 100), 
                                                 "GRN*", self.min_area_relaxed, strict=False)
        
        # Gate detection logic
        gate_detected = False
        alignment_error = 0.0
        estimated_distance = 999.0
        gate_center_x = w // 2
        gate_center_y = h // 2
        
        if red_stripe and green_stripe:
            gate_detected = True
            
            gate_center_x = (red_stripe['center'][0] + green_stripe['center'][0]) // 2
            gate_center_y = (red_stripe['center'][1] + green_stripe['center'][1]) // 2
            
            image_center_x = w / 2
            pixel_error = gate_center_x - image_center_x
            alignment_error = pixel_error / image_center_x
            
            stripe_distance = abs(red_stripe['center'][0] - green_stripe['center'][0])
            if stripe_distance > 20:
                estimated_distance = (self.gate_width * self.fx) / stripe_distance
                estimated_distance = max(0.5, min(estimated_distance, 50.0))
            
            cv2.circle(debug_img, (gate_center_x, gate_center_y), 25, (255, 0, 255), -1)
            cv2.line(debug_img, (gate_center_x, 0), (gate_center_x, h), (255, 0, 255), 4)
            cv2.line(debug_img, red_stripe['center'], green_stripe['center'], (0, 255, 255), 5)
            
            cv2.putText(debug_img, f"{estimated_distance:.1f}m", 
                       (gate_center_x - 60, gate_center_y - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 4)
        
        cv2.line(debug_img, (w//2, 0), (w//2, h), (0, 255, 255), 2)
        
        # Status overlay
        status_lines = [
            f"Frame {self.frame_count}",
            f"Red: {red_pixels}px ({len(red_contours)} cnt)",
            f"Green: {green_pixels}px ({len(green_contours)} cnt)",
            f"Orange: {orange_pixels}px ({len(orange_contours)} cnt)",
        ]
        
        if flare_detected:
            status_lines.append("FLARE DETECTED")
        if gate_detected:
            status_lines.append(f"GATE: {estimated_distance:.1f}m")
            status_lines.append(f"Align: {alignment_error:+.2f}")
        elif red_stripe or green_stripe:
            status_lines.append("PARTIAL GATE")
        else:
            status_lines.append("SEARCHING")
        
        box_height = len(status_lines) * 40 + 20
        cv2.rectangle(debug_img, (5, 5), (500, box_height), (0, 0, 0), -1)
        cv2.rectangle(debug_img, (5, 5), (500, box_height), 
                     (0, 255, 0) if gate_detected else (255, 140, 0) if flare_detected else (100, 100, 100), 3)
        
        for i, line in enumerate(status_lines):
            color = (0, 255, 0) if gate_detected else (255, 140, 0) if flare_detected else (200, 200, 200)
            cv2.putText(debug_img, line, (15, 40 + i*40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        info_text = f"HSV: R[{self.red_lower1[0]}-{self.red_upper1[0]},{self.red_lower2[0]}-{self.red_upper2[0]}] G[{self.green_lower[0]}-{self.green_upper[0]}]"
        cv2.putText(debug_img, info_text, (10, h - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        # Temporal filtering
        self.gate_detection_history.append(gate_detected)
        confirmed_gate = sum(self.gate_detection_history) >= self.min_confirmations
        
        # Publish
        gate_msg = Bool()
        gate_msg.data = confirmed_gate
        self.gate_detected_pub.publish(gate_msg)
        
        if confirmed_gate:
            align_msg = Float32()
            align_msg.data = float(alignment_error)
            self.alignment_pub.publish(align_msg)
            
            dist_msg = Float32()
            dist_msg.data = float(estimated_distance)
            self.distance_pub.publish(dist_msg)
            
            center_msg = Point()
            center_msg.x = float(gate_center_x)
            center_msg.y = float(gate_center_y)
            center_msg.z = float(estimated_distance)
            self.gate_center_pub.publish(center_msg)
        
        status_msg = String()
        status_msg.data = " | ".join(status_lines)
        self.status_pub.publish(status_msg)
        
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(debug_img, "bgr8")
            debug_msg.header = msg.header
            self.debug_pub.publish(debug_msg)
        except CvBridgeError as e:
            self.get_logger().error(f'Debug image error: {e}')
        
        if self.frame_count % 30 == 0:
            self.get_logger().info(
                f"R={red_pixels}px G={green_pixels}px O={orange_pixels}px | "
                f"Gate: {'YES' if confirmed_gate else 'NO'} | "
                f"Flare: {'WARNING' if flare_detected else 'OK'}"
            )
    


    def find_best_stripe(self, contours, debug_img, color, label, min_area, strict=True):
        if not contours:
            return None
        
        best_stripe = None
        best_score = 0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            if area < min_area:
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            if w == 0:
                continue
            
            aspect_ratio = float(h) / w
            
            # NEW: Check if stripe is at image edge (partial crop)
            image_width = debug_img.shape[1] if debug_img is not None else 1280
            at_edge = (x < 50 or (x + w) > image_width - 50)  # Within 50px of edge
            
            score = area
            
            if strict:
                # CHANGED: Accept lower aspect ratio OR edge position
                if aspect_ratio > self.aspect_threshold or at_edge:
                    score *= 2.0
                else:
                    continue
            else:
                if aspect_ratio > self.aspect_threshold or at_edge:
                    score *= 1.5
            
            # BONUS: Prefer stripes closer to center (but don't reject edges)
            center_x = x + w/2
            center_distance = abs(center_x - image_width/2) / (image_width/2)
            if center_distance < 0.3:  # Within center 60%
                score *= 1.2
            
            if score > best_score:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    best_stripe = {
                        'center': (cx, cy),
                        'bbox': (x, y, w, h),
                        'area': area,
                        'aspect': aspect_ratio,
                        'score': score
                    }
                    best_score = score
        
        if best_stripe and debug_img is not None:
            cx, cy = best_stripe['center']
            x, y, w, h = best_stripe['bbox']
            
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), color, 3)
            cv2.circle(debug_img, (cx, cy), 15, color, -1)
            cv2.circle(debug_img, (cx, cy), 17, (255, 255, 255), 2)
            
            label_text = f"{label} {int(best_stripe['area'])}"
            cv2.putText(debug_img, label_text, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return best_stripe


def main(args=None):
    rclpy.init(args=args)
    node = GateDetectorNode()
    
    node.get_logger().info('Gate detector ready')
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()