#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TwistWithCovarianceStamped
from nav_msgs.msg import Odometry # Subscribing to EKF output for height
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import time

class OpticalFlowNode(Node):
    def __init__(self):
        super().__init__('optical_flow_node')

        # --- Parameters ---
        self.declare_parameter('flow_method', 'farneback') # 'farneback' or 'lucas_kanade'
        self.flow_method = self.get_parameter('flow_method').value
        self.declare_parameter('blur_kernel_size', 15) # Kernel size for Gaussian blur (must be odd)
        self.blur_ksize = self.get_parameter('blur_kernel_size').value
        if self.blur_ksize % 2 == 0 and self.blur_ksize > 0: self.blur_ksize +=1
        self.declare_parameter('publish_debug_image', False)
        self.publish_debug = self.get_parameter('publish_debug_image').value
        self.declare_parameter('min_quality_threshold', 0.5) # Min LK quality
        self.min_lk_quality = self.get_parameter('min_quality_threshold').value
        self.declare_parameter('max_flow_magnitude', 50.0) # Max reasonable pixel shift per frame
        self.max_flow_mag = self.get_parameter('max_flow_magnitude').value
        self.declare_parameter('height_source', 'odom') # 'odom' or 'fixed'
        self.height_source = self.get_parameter('height_source').value
        self.declare_parameter('fixed_height', 1.0) # Use if height_source is 'fixed'
        self.fixed_height = self.get_parameter('fixed_height').value
        self.declare_parameter('velocity_covariance', [0.1, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                      0.0, 0.1, 0.0, 0.0, 0.0, 0.0,
                                                      0.0, 0.0, 999.0, 0.0, 0.0, 0.0,
                                                      0.0, 0.0, 0.0, 999.0, 0.0, 0.0,
                                                      0.0, 0.0, 0.0, 0.0, 999.0, 0.0,
                                                      0.0, 0.0, 0.0, 0.0, 0.0, 999.0]) # Covariance for Twist msg (tune this!)


        # LK parameters
        self.feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        self.lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # Farneback parameters
        self.farneback_params = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.1, flags=0) # Adjusted sigma


        # --- State Variables ---
        self.bridge = CvBridge()
        self.prev_gray = None
        self.prev_time_stamp = None
        self.p0 = None # Previous points for LK
        self.camera_matrix = None
        self.fx = None
        self.fy = None
        self.current_height = None if self.height_source == 'odom' else self.fixed_height

        # --- Quality of Service ---
        qos_profile_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE # Sensor data is volatile
        )
        qos_profile_sensor_data = QoSProfile( # Best effort for high rate sensor data
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE
        )

        # --- Subscriptions ---
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw', # INPUT: Downward camera image topic
            self.image_callback,
            qos_profile_sensor_data) # Use best effort for images
        self.cam_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera_info', # INPUT: Camera calibration topic
            self.cam_info_callback,
            qos_profile_reliable) # Need reliable camera info
        if self.height_source == 'odom':
            self.odom_sub = self.create_subscription(
                Odometry,
                '/odometry/filtered', # INPUT: EKF output topic
                self.odom_callback,
                qos_profile_reliable) # Need reliable odometry

        # --- Publishers ---
        self.vo_pub = self.create_publisher(
            TwistWithCovarianceStamped,
            '/optical_flow/twist', # OUTPUT: Velocity estimate for EKF
            10)
        if self.publish_debug:
            self.debug_img_pub = self.create_publisher(Image, '/optical_flow/debug_image', 10)

        self.get_logger().info(f"Optical Flow Node started with method: {self.flow_method}, Blur: {self.blur_ksize}x{self.blur_ksize}")
        if self.height_source == 'fixed':
             self.get_logger().info(f"Using fixed height: {self.fixed_height} m")

    def cam_info_callback(self, msg: CameraInfo):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape((3, 3))
            self.fx = self.camera_matrix[0, 0]
            self.fy = self.camera_matrix[1, 1]
            self.get_logger().info(f"Camera intrinsics received: fx={self.fx}, fy={self.fy}")
            # Unsubscribe after receiving info once
            self.destroy_subscription(self.cam_info_sub)

    def odom_callback(self, msg: Odometry):
         # Assuming Z is height above pool floor (needs EKF configured correctly)
         # WARNING: EKF outputs pose in 'odom' or 'map' frame. If Z=0 is the surface,
         # you need pool_depth - msg.pose.pose.position.z
         # This needs careful frame management in your EKF config!
         # For now, let's assume EKF's Z is height above floor (positive up)
        estimated_height = msg.pose.pose.position.z
        if estimated_height <= 0.1: # Avoid division by zero or small numbers
             #self.get_logger().warn(f"Estimated height {estimated_height:.2f} too low, skipping update")
             self.current_height = None
        else:
            self.current_height = estimated_height


    def image_callback(self, msg: Image):
        # Check prerequisites
        if self.fx is None or self.current_height is None or self.current_height <= 0.1:
            # self.get_logger().warn("Waiting for camera info or valid height...", throttle_duration_sec=5.0)
            return

        current_time_stamp = msg.header.stamp
        if self.prev_time_stamp is None:
            self.prev_time_stamp = current_time_stamp
            return # Need previous frame

        dt_sec = (current_time_stamp.sec - self.prev_time_stamp.sec) + \
                 (current_time_stamp.nanosec - self.prev_time_stamp.nanosec) / 1e9

        if dt_sec <= 0:
             self.get_logger().warn(f"Negative or zero dt ({dt_sec:.4f}), skipping frame.")
             # Update prev_time_stamp to potentially recover? Maybe just reset state.
             self.prev_gray = None
             self.p0 = None
             self.prev_time_stamp = current_time_stamp
             return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # --- Apply Blur to potentially reduce caustics ---
            if self.blur_ksize > 0:
                gray = cv2.GaussianBlur(gray, (self.blur_ksize, self.blur_ksize), 0)

        except CvBridgeError as e:
            self.get_logger().error(f'CV Bridge error: {e}')
            return

        vel_x, vel_y = None, None
        debug_img = frame if self.publish_debug else None

        # --- Calculate Flow ---
        if self.prev_gray is not None:
            if self.flow_method == 'lucas_kanade':
                vel_x, vel_y, debug_img = self.calculate_lk_flow(gray, debug_img, dt_sec)
            elif self.flow_method == 'farneback':
                vel_x, vel_y, debug_img = self.calculate_farneback_flow(gray, debug_img, dt_sec)
            else:
                 self.get_logger().error(f"Unknown flow method: {self.flow_method}")
                 return # Or default to one

            # --- Publish Velocity ---
            if vel_x is not None and vel_y is not None:
                twist_msg = TwistWithCovarianceStamped()
                twist_msg.header = msg.header # Use current image timestamp
                twist_msg.header.frame_id = "base_link_optical_flow" # Make up a frame name

                # Velocities are calculated in camera frame (X right, Y down, Z forward)
                # Need to transform to AUV base_link frame (X forward, Y left, Z up)
                # Assuming downward camera: Cam X -> -AUV Y; Cam Y -> -AUV X
                twist_msg.twist.twist.linear.x = -vel_y
                twist_msg.twist.twist.linear.y = -vel_x
                twist_msg.twist.twist.linear.z = 0.0 # Cannot estimate Z velocity from downward cam
                # Cannot estimate angular velocity from flow alone
                twist_msg.twist.twist.angular.x = 0.0
                twist_msg.twist.twist.angular.y = 0.0
                twist_msg.twist.twist.angular.z = 0.0

                # Set covariance (TUNE THIS CAREFULLY!)
                twist_msg.twist.covariance = self.get_parameter('velocity_covariance').value

                self.vo_pub.publish(twist_msg)

                # --- Publish Debug Image ---
                if self.publish_debug and debug_img is not None:
                    try:
                        debug_msg = self.bridge.cv2_to_imgmsg(debug_img, "bgr8")
                        debug_msg.header = msg.header
                        self.debug_img_pub.publish(debug_msg)
                    except CvBridgeError as e:
                        self.get_logger().error(f'CV Bridge error for debug image: {e}')


        # --- Update state for next iteration ---
        self.prev_gray = gray.copy()
        self.prev_time_stamp = current_time_stamp
        # Update points if using LK - done inside calculate_lk_flow


    def calculate_lk_flow(self, gray, debug_img, dt_sec):
        vel_x, vel_y = None, None
        good_new = None

        if self.p0 is None or len(self.p0) < 5: # Find features if none exist or too few left
             self.p0 = cv2.goodFeaturesToTrack(self.prev_gray, mask=None, **self.feature_params)
             if self.p0 is None:
                 self.get_logger().warn("LK: No features found in previous frame.")
                 return None, None, debug_img # No features to track

        # Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.p0, None, **self.lk_params)

        # Select good points
        if p1 is not None and st is not None:
            good_new = p1[st==1]
            good_old = self.p0[st==1]
        else:
            self.get_logger().warn("LK: Tracking failed (p1 or st is None).")
            self.p0 = None # Force feature re-detection next time
            return None, None, debug_img

        # --- Calculate Average Velocity ---
        if len(good_new) > 4: # Need a minimum number of good tracks
            dx_pixels = good_new[:, 0] - good_old[:, 0]
            dy_pixels = good_new[:, 1] - good_old[:, 1]

            # --- Basic Outlier Rejection (Magnitude based) ---
            flow_mags = np.sqrt(dx_pixels**2 + dy_pixels**2)
            valid_indices = flow_mags < self.max_flow_mag
            if np.sum(valid_indices) < 5:
                self.get_logger().warn(f"LK: Too few points ({np.sum(valid_indices)}) after outlier rejection.")
                self.p0 = good_new.reshape(-1,1,2) if len(good_new)>0 else None # Update points for next time
                return None, None, debug_img


            avg_dx = np.mean(dx_pixels[valid_indices])
            avg_dy = np.mean(dy_pixels[valid_indices])

            # Convert pixel shift/dt to meters/sec using camera intrinsics and height
            # vel_x_cam = (dx_pixels / dt) * Z / fx
            # vel_y_cam = (dy_pixels / dt) * Z / fy
            # NOTE: Sign depends on coordinate system conventions! Check carefully.
            # Assuming standard OpenCV camera (X right, Y down) and Z = height (positive)
            vel_x = (avg_dx / dt_sec) * self.current_height / self.fx
            vel_y = (avg_dy / dt_sec) * self.current_height / self.fy

            # --- Draw tracks for debugging ---
            if self.publish_debug and debug_img is not None:
                for i, (new, old) in enumerate(zip(good_new[valid_indices], good_old[valid_indices])):
                    a, b = new.ravel().astype(int)
                    c, d = old.ravel().astype(int)
                    debug_img = cv2.line(debug_img, (a, b), (c, d), (0, 255, 0), 2)
                    debug_img = cv2.circle(debug_img, (a, b), 5, (0, 0, 255), -1)

            # Update previous points
            self.p0 = good_new[valid_indices].reshape(-1,1,2)
            if len(self.p0) < 0.5 * self.feature_params['maxCorners']: # Regenerate points if too few left
                 self.get_logger().info("LK: Feature count low, triggering regeneration.")
                 self.p0 = None # Force regeneration next frame


        else:
             self.get_logger().warn("LK: Not enough good tracks found.")
             self.p0 = None # Force feature re-detection next time

        return vel_x, vel_y, debug_img


    def calculate_farneback_flow(self, gray, debug_img, dt_sec):
        # Calculate dense optical flow
        flow = cv2.calcOpticalFlowFarneback(self.prev_gray, gray, None, **self.farneback_params)

        # --- Calculate Average Velocity ---
        # flow is (height, width, 2), where flow[y, x] = [dx, dy] pixel shift
        avg_dx = np.mean(flow[..., 0])
        avg_dy = np.mean(flow[..., 1])

        # --- Basic Outlier Rejection ---
        flow_mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        valid_mask = flow_mag < self.max_flow_mag
        if np.sum(valid_mask) < 0.1 * flow.shape[0] * flow.shape[1]: # Check if at least 10% are valid
            self.get_logger().warn(f"Farneback: Too few points ({np.sum(valid_mask)}) after outlier rejection.")
            return None, None, debug_img

        avg_dx_valid = np.mean(flow[valid_mask, 0])
        avg_dy_valid = np.mean(flow[valid_mask, 1])


        # Convert pixel shift/dt to meters/sec
        vel_x = (avg_dx_valid / dt_sec) * self.current_height / self.fx
        vel_y = (avg_dy_valid / dt_sec) * self.current_height / self.fy

        # --- Draw flow field for debugging ---
        if self.publish_debug and debug_img is not None:
             step = 16
             h, w = gray.shape[:2]
             y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2,-1).astype(int)
             fx, fy = flow[y,x][valid_mask[y,x]].T # Only show valid flow
             lines = np.vstack([x[valid_mask[y,x]], y[valid_mask[y,x]], x[valid_mask[y,x]]+fx, y[valid_mask[y,x]]+fy]).T.reshape(-1, 2, 2)
             lines = np.int32(lines + 0.5)
             cv2.polylines(debug_img, lines, 0, (0, 255, 0))
             for (x1, y1), (_x2, _y2) in lines:
                 cv2.circle(debug_img, (x1, y1), 1, (0, 255, 0), -1)

        return vel_x, vel_y, debug_img


def main(args=None):
    rclpy.init(args=args)
    node = OpticalFlowNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()