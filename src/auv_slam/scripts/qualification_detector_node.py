#!/usr/bin/env python3
"""
QUALIFICATION Gate Detector
Includes 'Reverse Mode' to swap Red/Green logic when approaching from the back.
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
        
        # Mode State
        self.reverse_mode = False
        
        # HSV RANGES (Same as your working gate_params.yaml)
        self.red_lower1 = np.array([0, 100, 100])
        self.red_upper1 = np.array([10, 255, 255])
        self.red_lower2 = np.array([170, 100, 100])
        self.red_upper2 = np.array([180, 255, 255])
        self.green_lower = np.array([40, 100, 100])
        self.green_upper = np.array([80, 255, 255])
        
        # Detection parameters
        self.min_area_strict = 300
        self.min_area_relaxed = 80
        self.gate_width = 1.5
        self.fx = 0.0
        
        # History
        self.gate_detection_history = deque(maxlen=5)
        self.min_confirmations = 2
        self.frame_count = 0
        
        # QoS
        qos_sensor = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        qos_reliable = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)
        
        # Subscribers
        self.image_sub = self.create_subscription(Image, '/camera_forward/image_raw', self.image_callback, qos_sensor)
        self.cam_info_sub = self.create_subscription(CameraInfo, '/camera_forward/camera_info', self.cam_info_callback, qos_reliable)
        
        # NEW: Reverse Mode Subscriber
        self.reverse_mode_sub = self.create_subscription(Bool, '/mission/reverse_mode', self.reverse_mode_callback, 10)
        
        # Publishers
        self.gate_detected_pub = self.create_publisher(Bool, '/gate/detected', 10)
        self.alignment_pub = self.create_publisher(Float32, '/gate/alignment_error', 10)
        self.distance_pub = self.create_publisher(Float32, '/gate/estimated_distance', 10)
        self.frame_position_pub = self.create_publisher(Float32, '/gate/frame_position', 10)
        self.confidence_pub = self.create_publisher(Float32, '/gate/detection_confidence', 10)
        self.partial_gate_pub = self.create_publisher(Bool, '/gate/partial_detection', 10)
        self.debug_pub = self.create_publisher(Image, '/gate/debug_image', 10)
        
        self.get_logger().info('✅ QUALIFICATION Gate Detector Ready')

    def reverse_mode_callback(self, msg: Bool):
        if self.reverse_mode != msg.data:
            self.reverse_mode = msg.data
            state = "REVERSE (Expect Green-Left)" if self.reverse_mode else "FORWARD (Expect Red-Left)"
            self.get_logger().warn(f'🔄 SWITCHED DETECTOR MODE: {state}')

    def cam_info_callback(self, msg: CameraInfo):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape((3, 3))
            self.image_width = msg.width
            self.fx = self.camera_matrix[0, 0]

    def image_callback(self, msg: Image):
        if self.camera_matrix is None: return
        self.frame_count += 1
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        except CvBridgeError: return
        
        debug_img = cv_image.copy()
        h, w = cv_image.shape[:2]
        
        # Process Masks
        red_mask = cv2.bitwise_or(cv2.inRange(hsv_image, self.red_lower1, self.red_upper1),
                                  cv2.inRange(hsv_image, self.red_lower2, self.red_upper2))
        green_mask = cv2.inRange(hsv_image, self.green_lower, self.green_upper)
        
        # Find Contours
        red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Identify Stripes
        red_stripe = self.find_best_stripe(red_contours, debug_img, (0, 0, 255), "RED")
        green_stripe = self.find_best_stripe(green_contours, debug_img, (0, 255, 0), "GREEN")
        
        gate_detected = False
        partial_gate = False
        estimated_distance = 999.0
        gate_center_x = w // 2
        confidence = 0.0
        frame_position = 0.0

        # --- LOGIC HANDLING (Forward vs Reverse) ---
        
        if red_stripe and green_stripe:
            # FULL GATE: Center is always the average
            gate_detected = True
            confidence = 1.0
            gate_center_x = (red_stripe['center'][0] + green_stripe['center'][0]) // 2
            
            stripe_distance = abs(red_stripe['center'][0] - green_stripe['center'][0])
            if stripe_distance > 10:
                estimated_distance = (self.gate_width * self.fx) / stripe_distance
            
            cv2.line(debug_img, red_stripe['center'], green_stripe['center'], (255, 255, 0), 2)

        elif red_stripe or green_stripe:
            # PARTIAL GATE: Logic depends on MODE
            gate_detected = True
            partial_gate = True
            confidence = 0.5
            
            stripe = red_stripe if red_stripe else green_stripe
            cx = stripe['center'][0]
            
            # FORWARD: Red is Left Post (30% pos), Green is Right Post (70% pos)
            # REVERSE: Green is Left Post (30% pos), Red is Right Post (70% pos)
            
            target_pct = 0.5 # Default
            
            if not self.reverse_mode:
                # FORWARD MODE
                if red_stripe:   target_pct = 0.30 # Red is Left
                else:            target_pct = 0.70 # Green is Right
            else:
                # REVERSE MODE
                if green_stripe: target_pct = 0.30 # Green is Left
                else:            target_pct = 0.70 # Red is Right
            
            # If we see the Left Post (30%), the Gate Center (50%) is +20% (Right)
            # If we see the Right Post (70%), the Gate Center (50%) is -20% (Left)
            offset_from_center_pct = 0.5 - target_pct 
            gate_center_x = cx + (w * offset_from_center_pct)
            
            cv2.putText(debug_img, f"PARTIAL {'REV' if self.reverse_mode else 'FWD'}", (cx, stripe['center'][1]-20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)

        # Calculate Final Metrics
        if gate_detected:
            # frame_position: -1 (Left Edge) ... 0 (Center) ... +1 (Right Edge)
            frame_position = (gate_center_x - w/2) / (w/2)
            cv2.circle(debug_img, (int(gate_center_x), h//2), 20, (255, 0, 255), -1)

        # Publish
        self.gate_detection_history.append(gate_detected)
        confirmed = sum(self.gate_detection_history) >= self.min_confirmations
        
        self.gate_detected_pub.publish(Bool(data=confirmed))
        self.partial_gate_pub.publish(Bool(data=partial_gate))
        self.confidence_pub.publish(Float32(data=confidence))
        if confirmed:
            self.alignment_pub.publish(Float32(data=float(frame_position))) # Using frame_pos as align error
            self.distance_pub.publish(Float32(data=float(estimated_distance)))
            self.frame_position_pub.publish(Float32(data=float(frame_position)))

        # Overlay Status
        mode_str = "REVERSE MODE" if self.reverse_mode else "FORWARD MODE"
        cv2.putText(debug_img, mode_str, (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        try:
            self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug_img, "bgr8"))
        except CvBridgeError: pass

    def find_best_stripe(self, contours, debug_img, color, label):
        if not contours: return None
        best = None
        max_area = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > self.min_area_relaxed:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect = float(h)/w
                if aspect > 0.5: # Simple aspect check
                    if area > max_area:
                        max_area = area
                        M = cv2.moments(cnt)
                        if M["m00"] > 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            best = {'center': (cx, cy), 'bbox': (x,y,w,h)}
        
        if best:
            cv2.rectangle(debug_img, (best['bbox'][0], best['bbox'][1]), 
                         (best['bbox'][0]+best['bbox'][2], best['bbox'][1]+best['bbox'][3]), color, 2)
        return best

def main(args=None):
    rclpy.init(args=args)
    node = QualificationGateDetector()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: 
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()