# auv_core/control_node.py

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Twist, Point, Quaternion
import math
import numpy as np
import tf_transformations
# from simple_pid import PID

class ControlNode(Node):
    def __init__(self):
        super().__init__('control_node')
        self.current_odom: Odometry = None
        self.target_pose: PoseStamped = None
        self.target_velocity: Twist = None

        # --- Parameters ---
        self.declare_parameter('max_linear_vel', 0.5) # m/s
        self.declare_parameter('max_angular_vel', 0.3) # rad/s
        self.declare_parameter('pos_p_gain', 0.5) # Proportional gain for position error
        self.declare_parameter('yaw_p_gain', 0.8) # Proportional gain for yaw error
        self.declare_parameter('depth_p_gain', 1.0) # Proportional gain for depth error

        self.max_lin_vel = self.get_parameter('max_linear_vel').value
        self.max_ang_vel = self.get_parameter('max_angular_vel').value
        self.pos_p = self.get_parameter('pos_p_gain').value
        self.yaw_p = self.get_parameter('yaw_p_gain').value
        self.depth_p = self.get_parameter('depth_p_gain').value


        # --- Subscriptions ---
        self.odom_sub = self.create_subscription(
            Odometry, '/odometry/filtered', self.odom_callback, 10)
        self.target_pose_sub = self.create_subscription(
            PoseStamped, '/control/target_pose', self.target_pose_callback, 10)
        self.target_vel_sub = self.create_subscription(
            Twist, '/control/target_velocity', self.target_vel_callback, 10)

        # --- Publisher ---
        self.cmd_vel_pub = self.create_publisher(Twist, '/rp2040/cmd_vel', 10) # Commands to serial bridge

        self.timer = self.create_timer(0.05, self.control_loop) # 20Hz control loop
        self.get_logger().info('Control Node started.')

    def odom_callback(self, msg: Odometry):
        self.current_odom = msg

    def target_pose_callback(self, msg: PoseStamped):
        # Only accept pose commands if frame_id is correct (e.g., 'odom')
        if msg.header.frame_id == self.current_odom.header.frame_id:
            self.target_pose = msg
            self.target_velocity = None # Prioritize pose commands
        else:
            self.get_logger().warn(f"Received target pose in frame '{msg.header.frame_id}' but expected '{self.current_odom.header.frame_id}'")


    def target_vel_callback(self, msg: Twist):
        self.target_velocity = msg
        self.target_pose = None # Prioritize velocity commands

    def control_loop(self):
        if self.current_odom is None:
            return

        cmd_vel = Twist() # Initialize command to zero

        current_pos = self.current_odom.pose.pose.position
        current_q = self.current_odom.pose.pose.orientation
        _, _, current_yaw = tf_transformations.euler_from_quaternion([current_q.x, current_q.y, current_q.z, current_q.w])

        if self.target_pose:
            # --- Simple P Controller for Pose Goal ---
            goal_pos = self.target_pose.pose.position
            goal_q = self.target_pose.pose.orientation
            _, _, goal_yaw = tf_transformations.euler_from_quaternion([goal_q.x, goal_q.y, goal_q.z, goal_q.w])

            # Calculate error in world frame
            error_x = goal_pos.x - current_pos.x
            error_y = goal_pos.y - current_pos.y
            error_z = goal_pos.z - current_pos.z
            error_yaw = goal_yaw - current_yaw
            # Handle yaw wrapping
            while error_yaw > math.pi: error_yaw -= 2 * math.pi
            while error_yaw <= -math.pi: error_yaw += 2 * math.pi

            # Transform world XY error to base_link frame velocity command
            # Rotate error vector by -current_yaw
            cmd_vel.linear.x = (error_x * math.cos(current_yaw) + error_y * math.sin(current_yaw)) * self.pos_p
            cmd_vel.linear.y = (-error_x * math.sin(current_yaw) + error_y * math.cos(current_yaw)) * self.pos_p # Y is left in base_link
            cmd_vel.linear.z = error_z * self.depth_p # Simple Z control
            cmd_vel.angular.z = error_yaw * self.yaw_p # Simple Yaw control


        elif self.target_velocity:
            # --- Pass through Velocity Command ---
            cmd_vel = self.target_velocity

        else:
            pass

        # --- Apply Velocity Limits ---
        lin_vel_mag = math.sqrt(cmd_vel.linear.x**2 + cmd_vel.linear.y**2 + cmd_vel.linear.z**2)
        if lin_vel_mag > self.max_lin_vel:
            scale = self.max_lin_vel / lin_vel_mag
            cmd_vel.linear.x *= scale
            cmd_vel.linear.y *= scale
            cmd_vel.linear.z *= scale

        if abs(cmd_vel.angular.z) > self.max_ang_vel:
            cmd_vel.angular.z = math.copysign(self.max_ang_vel, cmd_vel.angular.z)

        # --- Publish Command ---
        self.cmd_vel_pub.publish(cmd_vel)


def main(args=None):
    rclpy.init(args=args)
    node = ControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()