#!/usr/bin/env python3
"""
QUALIFICATION Gate Detector - Detects orange gate markings on port/starboard sides
Key features:
1. Detects two orange vertical stripes (port/starboard)
2. Handles partial views (single stripe)
3. Provides confidence metrics and frame position
4. Distinguishes gate stripes from orange flare obstacle
"""

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


class QualificationGateDetector(Node):
    def __init__(self):
        super().__init__('gate_detector_node')
        self.bridge = CvBridge()
        self.camera_matrix = None
        self.image_width = None
        self.image_height = None
        
        # HSV range for ORANGE (gate markings and flare)
        self.orange_lower = np.array([10, 120, 120])
        self.orange_upper = np.array([25, 255, 255])
        
        # Detection parameters
        self.min_stripe_area = 500
        self.min_flare_area = 2000
        self.aspect_threshold = 2.5  # Tall vertical stripes
        self.gate_width = 1.5  # 150cm in real world
        
        self.gate_detection_history = deque(maxlen=5)
        self.min_confirmations = 2
        
        self.frame_count = 0
        
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
        
        # Publishers
        self.gate_detected_pub = self.create_publisher(Bool, '/gate/detected', 10)
        self.alignment_pub = self.create_publisher(Float32, '/gate/alignment_error', 10)
        self.distance_pub = self.create_publisher(Float32, '/gate/estimated_distance', 10)
        self.gate_center_pub = self.create_publisher(Point, '/gate/center_point', 10)
        self.debug_pub = self.create_publisher(Image, '/gate/debug_image', 10)
        self.status_pub = self.create_publisher(String, '/gate/status', 10)
        
        # Navigation feedback
        self.frame_position_pub = self.create_publisher(Float32, '/gate/frame_position', 10)
        self.confidence_pub = self.create_publisher(Float32, '/gate/detection_confidence', 10)
        self.partial_gate_pub = self.create_publisher(Bool, '/gate/partial_detection', 10)
        self.pass_number_pub = self.create_publisher(String, '/gate/pass_number', 10)
        
        # Obstacle detection
        self.flare_detected_pub = self.create_publisher(Bool, '/flare/detected', 10)
        self.flare_direction_pub = self.create_publisher(Float32, '/flare/avoidance_direction', 10)
        self.flare_warning_pub = self.create_publisher(String, '/flare/warning', 10)
        
        self.get_logger().info('✅ QUALIFICATION Gate Detector initialized')
        self.get_logger().info('   - Detects orange port/starboard stripes')
        self.get_logger().info('   - Handles partial gate views')
        self.get_logger().info('   - Distinguishes gate from flare obstacle')
    
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
        
        # Create orange mask for both gate and flare
        orange_mask = cv2.inRange(hsv_image, self.orange_lower, self.orange_upper)
        
        # Morphological operations
        kernel = np.ones((5, 5), np.uint8)
        orange_mask_clean = cv2.morphologyEx(orange_mask, cv2.MORPH_CLOSE, kernel)
        orange_mask_clean = cv2.morphologyEx(orange_mask_clean, cv2.MORPH_OPEN, kernel)
        
        # Find all orange contours
        contours, _ = cv2.findContours(orange_mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Separate flare from gate stripes
        flare_detected = False
        gate_stripe_contours = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_stripe_area:
                continue
            
            x, y, w_bbox, h_bbox = cv2.boundingRect(cnt)
            aspect_ratio = float(h_bbox) / w_bbox if w_bbox > 0 else 0
            
            # Flare is large blob, gate stripes are tall/vertical
            if area > self.min_flare_area and aspect_ratio < 2.0:
                # This is likely the flare obstacle
                flare_detected = True
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    flare_center_x = int(M["m10"] / M["m00"])
                    cv2.circle(debug_img, (flare_center_x, int(h/2)), 35, (0, 140, 255), 6)
                    cv2.putText(debug_img, "FLARE", (flare_center_x - 60, int(h/2) - 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 140, 255), 4)
                    
                    avoidance_direction = -1.0 if flare_center_x < w/2 else 1.0
                    self.flare_detected_pub.publish(Bool(data=True))
                    self.flare_direction_pub.publish(Float32(data=avoidance_direction))
                    self.flare_warning_pub.publish(String(data=f"FLARE at X={flare_center_x}"))
            elif aspect_ratio > self.aspect_threshold:
                # This is likely a gate stripe
                gate_stripe_contours.append(cnt)
        
        if not flare_detected:
            self.flare_detected_pub.publish(Bool(data=False))
        
        # Find best two stripes for gate detection
        port_stripe = None
        starboard_stripe = None
        
        if len(gate_stripe_contours) >= 1:
            # Sort by x-position to separate port/starboard
            stripe_info = []
            for cnt in gate_stripe_contours:
                area = cv2.contourArea(cnt)
                x, y, w_bbox, h_bbox = cv2.boundingRect(cnt)
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    stripe_info.append({
                        'center': (cx, cy),
                        'bbox': (x, y, w_bbox, h_bbox),
                        'area': area,
                        'aspect': float(h_bbox) / w_bbox
                    })
            
            # Sort by x-position
            stripe_info.sort(key=lambda s: s['center'][0])
            
            # Assign port (left) and starboard (right)
            if len(stripe_info) >= 2:
                # Check if they're far enough apart to be gate edges
                stripe_distance = stripe_info[-1]['center'][0] - stripe_info[0]['center'][0]
                min_gate_pixels = 100  # Minimum pixels for gate width
                
                if stripe_distance > min_gate_pixels:
                    port_stripe = stripe_info[0]
                    starboard_stripe = stripe_info[-1]
                else:
                    # Might be same stripe detected twice, take largest two
                    stripe_info.sort(key=lambda s: s['area'], reverse=True)
                    port_stripe = stripe_info[0]
                    starboard_stripe = stripe_info[1] if len(stripe_info) > 1 else None
            else:
                # Only one stripe detected - partial view
                port_stripe = stripe_info[0]
        
        # IMPROVED GATE DETECTION - Handles partial views
        gate_detected = False
        partial_gate = False
        alignment_error = 0.0
        estimated_distance = 999.0
        gate_center_x = w // 2
        gate_center_y = h // 2
        frame_position = 0.0
        confidence = 0.0
        
        if port_stripe and starboard_stripe:
            # FULL GATE DETECTED - Both stripes visible
            gate_detected = True
            partial_gate = False
            confidence = 1.0
            
            gate_center_x = (port_stripe['center'][0] + starboard_stripe['center'][0]) // 2
            gate_center_y = (port_stripe['center'][1] + starboard_stripe['center'][1]) // 2
            
            image_center_x = w / 2
            pixel_error = gate_center_x - image_center_x
            alignment_error = pixel_error / image_center_x
            
            stripe_distance = abs(port_stripe['center'][0] - starboard_stripe['center'][0])
            if stripe_distance > 20:
                estimated_distance = (self.gate_width * self.fx) / stripe_distance
                estimated_distance = max(0.5, min(estimated_distance, 50.0))
            
            # Frame position (-1 = left edge, 0 = center, +1 = right edge)
            frame_position = (gate_center_x - w/2) / (w/2)
            
            # Visualization
            cv2.circle(debug_img, (gate_center_x, gate_center_y), 25, (255, 0, 255), -1)
            cv2.line(debug_img, (gate_center_x, 0), (gate_center_x, h), (255, 0, 255), 4)
            cv2.line(debug_img, port_stripe['center'], starboard_stripe['center'], (0, 255, 255), 5)
            
            cv2.putText(debug_img, f"FULL GATE {estimated_distance:.1f}m", 
                       (gate_center_x - 100, gate_center_y - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 4)
            
            # Draw stripes
            for i, stripe in enumerate([port_stripe, starboard_stripe]):
                x, y, w_bbox, h_bbox = stripe['bbox']
                color = (0, 140, 255) if i == 0 else (0, 200, 255)
                label = "PORT" if i == 0 else "STBD"
                cv2.rectangle(debug_img, (x, y), (x+w_bbox, y+h_bbox), color, 3)
                cv2.putText(debug_img, label, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        elif port_stripe:
            # PARTIAL GATE DETECTED - Only one stripe visible
            gate_detected = True
            partial_gate = True
            confidence = 0.6
            
            gate_center_x = port_stripe['center'][0]
            gate_center_y = port_stripe['center'][1]
            
            # Determine which side based on position
            is_port_side = gate_center_x < w/2
            stripe_name = "PORT" if is_port_side else "STBD"
            
            # Calculate desired position to keep gate in frame
            if is_port_side:
                # Port stripe visible - aim to keep it on left third
                desired_x = w * 0.30
            else:
                # Starboard stripe visible - aim to keep it on right third
                desired_x = w * 0.70
            
            pixel_error = gate_center_x - desired_x
            image_center_x = w / 2
            alignment_error = pixel_error / image_center_x
            
            estimated_distance = 999.0  # Unreliable with one stripe
            frame_position = (gate_center_x - w/2) / (w/2)
            
            # Visualization
            cv2.circle(debug_img, (gate_center_x, gate_center_y), 30, (255, 165, 0), 5)
            cv2.putText(debug_img, f"PARTIAL: {stripe_name}", 
                       (gate_center_x - 150, gate_center_y - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 165, 0), 3)
            cv2.putText(debug_img, "CENTERING...", 
                       (gate_center_x - 100, gate_center_y + 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        
        # Draw center line
        cv2.line(debug_img, (w//2, 0), (w//2, h), (0, 255, 255), 2)
        
        # Status overlay
        status_lines = [
            f"Frame {self.frame_count}",
            f"Confidence: {confidence:.2f}",
        ]
        
        if flare_detected:
            status_lines.append("FLARE!")
        if gate_detected:
            if partial_gate:
                status_lines.append(f"PARTIAL @ {frame_position:+.2f}")
            else:
                status_lines.append(f"FULL GATE {estimated_distance:.1f}m")
            status_lines.append(f"Align: {alignment_error:+.2f}")
        else:
            status_lines.append("SEARCHING")
        
        # Draw status box
        box_height = len(status_lines) * 35 + 20
        cv2.rectangle(debug_img, (5, 5), (450, box_height), (0, 0, 0), -1)
        box_color = (0, 255, 0) if (gate_detected and not partial_gate) else \
                    (255, 165, 0) if partial_gate else (100, 100, 100)
        cv2.rectangle(debug_img, (5, 5), (450, box_height), box_color, 3)
        
        for i, line in enumerate(status_lines):
            cv2.putText(debug_img, line, (15, 35 + i*35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Temporal filtering
        self.gate_detection_history.append(gate_detected)
        confirmed_gate = sum(self.gate_detection_history) >= self.min_confirmations
        
        # Publish all data
        self.publish_gate_data(confirmed_gate, partial_gate, alignment_error, 
                              estimated_distance, gate_center_x, gate_center_y, 
                              frame_position, confidence, msg.header)
        
        # Publish debug image
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(debug_img, "bgr8")
            debug_msg.header = msg.header
            self.debug_pub.publish(debug_msg)
        except CvBridgeError as e:
            self.get_logger().error(f'Debug image error: {e}')
    
    def publish_gate_data(self, confirmed_gate, partial_gate, alignment_error, 
                         estimated_distance, center_x, center_y, frame_position, 
                         confidence, header):
        """Publish all gate detection data"""
        self.gate_detected_pub.publish(Bool(data=confirmed_gate))
        self.partial_gate_pub.publish(Bool(data=partial_gate))
        self.confidence_pub.publish(Float32(data=confidence))
        
        if confirmed_gate:
            self.alignment_pub.publish(Float32(data=float(alignment_error)))
            self.distance_pub.publish(Float32(data=float(estimated_distance)))
            self.frame_position_pub.publish(Float32(data=float(frame_position)))
            
            center_msg = Point()
            center_msg.x = float(center_x)
            center_msg.y = float(center_y)
            center_msg.z = float(estimated_distance)
            self.gate_center_pub.publish(center_msg)
            
            status = f"{'PARTIAL' if partial_gate else 'FULL'} | Pos:{frame_position:+.2f} | Conf:{confidence:.2f}"
            self.status_pub.publish(String(data=status))


def main(args=None):
    rclpy.init(args=args)
    node = QualificationGateDetector()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()