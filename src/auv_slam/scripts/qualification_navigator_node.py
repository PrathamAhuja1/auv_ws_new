#!/usr/bin/env python3
"""
FINAL QUALIFICATION NAVIGATOR (revised)

Key ideas:
- Hold ONE safe mission depth the whole time (no depth change near the gate)
- Use yaw only until 3 m
- Inside 3 m: use yaw + gentle lateral sway to center on the gate
- Pass straight through (no corrections) once well aligned
- U-turn fully submerged, then repeat in reverse, then surface
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import time
import math


class FinalQualificationNavigator(Node):
    def __init__(self):
        super().__init__('qualification_navigator')

        # STATE ENUM
        self.SUBMERGING = 0
        self.CRUISING = 1
        self.FINAL_ALIGNMENT = 2
        self.PASSING = 3
        self.CLEARING = 4
        self.UTURN = 5
        self.REVERSE_CRUISING = 6
        self.REVERSE_ALIGNMENT = 7
        self.REVERSE_PASSING = 8
        self.REVERSE_CLEARING = 9
        self.SURFACING = 10
        self.COMPLETED = 11

        self.state = self.SUBMERGING

        # ----------------- DEPTH / GEOMETRY PARAMETERS -----------------

        # One single safe mission depth used for the whole run
        # (must be comfortably above the deepest floor and below surface)
        self.mission_depth = -0.8          # [m] adjust if sim depth frame differs

        # For clarity keep the old names but point them to the same value
        self.cruise_depth = self.mission_depth
        self.gate_center_depth = self.mission_depth

        # Gate X position in world frame (used only for "cleared" checks)
        # Set to the map X coord of the gate if you know it; 0 is fine as
        # we only require "some distance past gate".
        self.gate_x_position = 0.0

        # Distances
        self.alignment_distance = 3.0   # start precise alignment inside 3 m
        self.passing_distance = 1.0     # commit to pass when closer than 1 m
        self.clearance_distance = 1.0   # distance the robot must travel during PASS

        # ----------------- SPEED GAINS -----------------
        self.cruise_speed = 0.8         # fast forward outside 3 m
        self.alignment_speed = 0.35     # slower forward inside 3 m
        self.passing_speed = 1.0        # straight pass through the gate

        # Yaw control
        self.yaw_gain_cruise = 1.0
        self.yaw_gain_align = 1.2

        # Lateral (sway) control during final alignment
        self.sway_gain = 0.35
        self.max_sway_speed = 0.3

        # ----------------------------------------------------------------

        # State tracking
        self.gate_detected = False
        self.alignment_error = 0.0        # typically in [-1, 1] from vision
        self.estimated_distance = 999.0
        self.current_depth = 0.0
        self.current_position = None
        self.current_yaw = 0.0

        self.passing_start_x = None
        self.uturn_start_yaw = 0.0
        self.state_start_time = time.time()
        self.mission_start_time = time.time()

        # Subscriptions
        self.create_subscription(Bool, '/qualification/gate_detected',
                                 self.gate_cb, 10)
        self.create_subscription(Float32, '/qualification/alignment_error',
                                 self.align_cb, 10)
        self.create_subscription(Float32, '/qualification/estimated_distance',
                                 self.dist_cb, 10)
        self.create_subscription(Odometry, '/ground_truth/odom',
                                 self.odom_cb, 10)

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/rp2040/cmd_vel', 10)
        self.state_pub = self.create_publisher(String, '/qualification/state', 10)
        self.reverse_mode_pub = self.create_publisher(Bool,
                                                      '/mission/reverse_mode', 10)

        # Control loop at 20 Hz
        self.create_timer(0.05, self.control_loop)

        # Log
        self.get_logger().info('=' * 70)
        self.get_logger().info('✅ FINAL QUALIFICATION NAVIGATOR (revised)')
        self.get_logger().info('=' * 70)
        self.get_logger().info(f' Mission depth: {self.mission_depth:.2f} m')
        self.get_logger().info(' Strategy: '
                               'Submerge → Cruise → Align@3m → Pass '
                               '→ U-turn → Align → Pass → Surface')
        self.get_logger().info('=' * 70)

    # ------------------------------------------------------------------ #
    #   CALLBACKS
    # ------------------------------------------------------------------ #

    def gate_cb(self, msg: Bool):
        self.gate_detected = msg.data

    def align_cb(self, msg: Float32):
        self.alignment_error = msg.data

    def dist_cb(self, msg: Float32):
        self.estimated_distance = msg.data

    def odom_cb(self, msg: Odometry):
        self.current_depth = msg.pose.pose.position.z
        self.current_position = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        )

        # yaw from quaternion
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

    # ------------------------------------------------------------------ #
    #   MAIN CONTROL LOOP
    # ------------------------------------------------------------------ #

    def control_loop(self):
        cmd = Twist()

        # Vertical control – keep single mission depth in all active states
        if self.state in [
            self.SUBMERGING,
            self.CRUISING,
            self.FINAL_ALIGNMENT,
            self.PASSING,
            self.CLEARING,
            self.UTURN,
            self.REVERSE_CRUISING,
            self.REVERSE_ALIGNMENT,
            self.REVERSE_PASSING,
            self.REVERSE_CLEARING,
        ]:
            cmd.linear.z = self.depth_control(self.mission_depth)

        elif self.state == self.SURFACING:
            # Gentle rise to surface
            cmd.linear.z = -0.5
        else:
            cmd.linear.z = 0.0

        # State-specific horizontal behaviour
        if self.state == self.SUBMERGING:
            cmd = self.submerge(cmd)
        elif self.state == self.CRUISING:
            cmd = self.cruise(cmd)
        elif self.state == self.FINAL_ALIGNMENT:
            cmd = self.final_align(cmd)
        elif self.state == self.PASSING:
            cmd = self.passing(cmd, "FORWARD")
        elif self.state == self.CLEARING:
            cmd = self.clearing(cmd, "FORWARD")
        elif self.state == self.UTURN:
            cmd = self.uturn(cmd)
        elif self.state == self.REVERSE_CRUISING:
            cmd = self.reverse_cruise(cmd)
        elif self.state == self.REVERSE_ALIGNMENT:
            cmd = self.reverse_align(cmd)
        elif self.state == self.REVERSE_PASSING:
            cmd = self.passing(cmd, "REVERSE")
        elif self.state == self.REVERSE_CLEARING:
            cmd = self.clearing(cmd, "REVERSE")
        elif self.state == self.SURFACING:
            cmd = self.surfacing(cmd)
        elif self.state == self.COMPLETED:
            cmd = self.completed(cmd)

        # Publish
        self.cmd_vel_pub.publish(cmd)
        self.state_pub.publish(String(data=self.get_state_name()))

        # Throttled log
        if self.current_position and int(time.time() * 2) % 4 == 0:
            self.get_logger().info(
                f'[{self.get_state_name()}] '
                f'X={self.current_position[0]:.2f}, '
                f'Y={self.current_position[1]:.2f}, '
                f'Z={self.current_depth:.2f}, '
                f'Dist={self.estimated_distance:.2f} m',
                throttle_duration_sec=1.9
            )

    # ------------------------------------------------------------------ #
    #   DEPTH CONTROL
    # ------------------------------------------------------------------ #

    def depth_control(self, target_depth: float) -> float:
        """
        Simple P control on depth with clamped output.
        The important part for your issue: we never ask for different
        depths in different states – target_depth is always mission_depth.
        """
        depth_error = target_depth - self.current_depth

        # Small deadband so we don't chatter
        DEADBAND = 0.15
        if abs(depth_error) < DEADBAND:
            return 0.0

        # Piecewise proportional gain for smoothness
        if abs(depth_error) < 0.4:
            z_cmd = depth_error * 0.4
        elif abs(depth_error) < 0.8:
            z_cmd = depth_error * 0.6
        else:
            z_cmd = depth_error * 0.8

        # Clamp vertical speed
        return max(-0.6, min(z_cmd, 0.6))

    # ------------------------------------------------------------------ #
    #   BEHAVIOURS
    # ------------------------------------------------------------------ #

    def submerge(self, cmd: Twist) -> Twist:
        """Wait until depth is close to mission depth, then start cruising"""
        if abs(self.mission_depth - self.current_depth) < 0.3:
            if time.time() - self.state_start_time > 3.0:
                self.get_logger().info('✅ Submerged at mission depth – starting cruise')
                self.reverse_mode_pub.publish(Bool(data=False))
                self.transition_to(self.CRUISING)
        return cmd

    def cruise(self, cmd: Twist) -> Twist:
        """
        Cruise towards gate.
        - If gate not detected: search with slow yaw scan
        - If gate detected: fast forward, yaw-only correction
        """
        if not self.gate_detected:
            cmd.linear.x = 0.3
            cmd.linear.y = 0.0
            cmd.angular.z = 0.4 if (time.time() % 8 < 4) else -0.4
            return cmd

        # Inside 3 m → switch to precise alignment
        if self.estimated_distance <= self.alignment_distance:
            self.get_logger().info(
                f'🎯 Gate within {self.alignment_distance:.1f} m '
                f'({self.estimated_distance:.2f} m) – switching to FINAL_ALIGNMENT'
            )
            self.transition_to(self.FINAL_ALIGNMENT)
            return cmd

        # Fast forward, yaw correction to keep gate centered
        cmd.linear.x = self.cruise_speed
        cmd.linear.y = 0.0
        cmd.angular.z = -self.alignment_error * self.yaw_gain_cruise
        return cmd

    def final_align(self, cmd: Twist) -> Twist:
        """
        Precise alignment in the last 3 m:
        - Forward speed reduced
        - Yaw + lateral sway from alignment_error
        - Only commit to PASS when well-centered and closer than passing_distance
        """
        if not self.gate_detected:
            # Gate lost – slow forward + gentle search
            cmd.linear.x = 0.2
            cmd.linear.y = 0.0
            cmd.angular.z = 0.3
            return cmd

        # Close enough – check if we can commit to pass
        if self.estimated_distance <= self.passing_distance:
            if abs(self.alignment_error) < 0.08:
                self.get_logger().info(
                    f'✅ Aligned at {self.estimated_distance:.2f} m '
                    f'(err={self.alignment_error:+.3f}) – COMMITTING TO PASS'
                )
                self.passing_start_x = self.current_position[0] if self.current_position else 0.0
                self.transition_to(self.PASSING)
                return cmd
            else:
                self.get_logger().warn(
                    f'⚠️ At {self.estimated_distance:.2f} m but misaligned '
                    f'(err={self.alignment_error:+.3f}) – still correcting'
                )

        # Alignment behaviour
        # Forward speed: slower near gate
        cmd.linear.x = self.alignment_speed

        # Lateral sway to center on gate
        sway = -self.alignment_error * self.sway_gain
        sway = max(-self.max_sway_speed, min(sway, self.max_sway_speed))
        cmd.linear.y = sway

        # Yaw correction (slightly stronger than cruise)
        cmd.angular.z = -self.alignment_error * self.yaw_gain_align

        return cmd

    def passing(self, cmd: Twist, direction: str) -> Twist:
        """
        Straight pass through the gate:
        - Full speed
        - NO yaw / lateral changes (reduce risk of clipping gate)
        """
        if self.passing_start_x is None and self.current_position:
            self.passing_start_x = self.current_position[0]

        cmd.linear.x = self.passing_speed
        cmd.linear.y = 0.0
        cmd.angular.z = 0.0

        # Measure distance travelled during PASS
        if self.current_position and self.passing_start_x is not None:
            distance_travelled = abs(self.current_position[0] - self.passing_start_x)
            if distance_travelled >= self.clearance_distance:
                self.get_logger().info(
                    f'✅ {direction} PASS distance travelled: {distance_travelled:.2f} m'
                )
                if direction == "FORWARD":
                    self.transition_to(self.CLEARING)
                else:
                    self.transition_to(self.REVERSE_CLEARING)

        return cmd

    def clearing(self, cmd: Twist, direction: str) -> Twist:
        """
        Keep moving forward a bit more after PASS to be sure the
        back of the AUV has fully cleared the gate plane.
        """
        current_x = self.current_position[0] if self.current_position else 0.0

        if direction == "FORWARD":
            if current_x > (self.gate_x_position + 1.5):
                elapsed = time.time() - self.mission_start_time
                self.get_logger().info('=' * 70)
                self.get_logger().info('✅ FORWARD PASS COMPLETE')
                self.get_logger().info(f'   X={current_x:.2f} m | t={elapsed:.1f} s')
                self.get_logger().info('   Starting submerged U-turn...')
                self.get_logger().info('=' * 70)

                self.uturn_start_yaw = self.current_yaw
                self.transition_to(self.UTURN)
        else:
            if current_x < (self.gate_x_position - 1.5):
                elapsed = time.time() - self.mission_start_time
                self.get_logger().info('=' * 70)
                self.get_logger().info('✅ REVERSE PASS COMPLETE')
                self.get_logger().info(f'   X={current_x:.2f} m | t={elapsed:.1f} s')
                self.get_logger().info('   QUALIFICATION COMPLETE – SURFACING')
                self.get_logger().info('=' * 70)

                self.transition_to(self.SURFACING)

        cmd.linear.x = self.passing_speed
        cmd.linear.y = 0.0
        cmd.angular.z = 0.0
        return cmd

    def uturn(self, cmd: Twist) -> Twist:
        """180° turn at mission depth."""
        angle_turned = abs(self.normalize_angle(self.current_yaw - self.uturn_start_yaw))

        if angle_turned > (math.pi - 0.15):  # ~165°
            self.get_logger().info('✅ U-turn complete – starting REVERSE_CRUISING')
            self.reverse_mode_pub.publish(Bool(data=True))
            self.transition_to(self.REVERSE_CRUISING)
            return cmd

        cmd.linear.x = 0.2
        cmd.angular.z = 0.7
        return cmd

    def reverse_cruise(self, cmd: Twist) -> Twist:
        """Cruise back towards gate in reverse direction (yaw-only)."""
        if not self.gate_detected:
            cmd.linear.x = 0.3
            cmd.linear.y = 0.0
            cmd.angular.z = -0.4 if (time.time() % 8 < 4) else 0.4
            return cmd

        if self.estimated_distance <= self.alignment_distance:
            self.get_logger().info('🎯 Reverse: within 3 m – switching to REVERSE_ALIGNMENT')
            self.transition_to(self.REVERSE_ALIGNMENT)
            return cmd

        cmd.linear.x = self.cruise_speed
        cmd.linear.y = 0.0
        cmd.angular.z = -self.alignment_error * self.yaw_gain_cruise
        return cmd

    def reverse_align(self, cmd: Twist) -> Twist:
        """Same as FINAL_ALIGNMENT but for reverse run."""
        if not self.gate_detected:
            cmd.linear.x = 0.2
            cmd.linear.y = 0.0
            cmd.angular.z = -0.3
            return cmd

        if self.estimated_distance <= self.passing_distance:
            if abs(self.alignment_error) < 0.08:
                self.get_logger().info(
                    f'✅ Reverse aligned (err={self.alignment_error:+.3f}) – COMMITTING TO PASS'
                )
                self.passing_start_x = self.current_position[0] if self.current_position else 0.0
                self.transition_to(self.REVERSE_PASSING)
                return cmd

        cmd.linear.x = self.alignment_speed

        sway = -self.alignment_error * self.sway_gain
        sway = max(-self.max_sway_speed, min(sway, self.max_sway_speed))
        cmd.linear.y = sway

        cmd.angular.z = -self.alignment_error * self.yaw_gain_align
        return cmd

    def surfacing(self, cmd: Twist) -> Twist:
        """Rise to surface and stop."""
        if self.current_depth > -0.2:
            self.get_logger().info('✅ SURFACED – mission complete')
            self.transition_to(self.COMPLETED)
            return cmd

        cmd.linear.x = 0.0
        cmd.linear.y = 0.0
        cmd.angular.z = 0.0
        # linear.z already set in control_loop
        return cmd

    def completed(self, cmd: Twist) -> Twist:
        """Zero all velocities."""
        cmd.linear.x = 0.0
        cmd.linear.y = 0.0
        cmd.linear.z = 0.0
        cmd.angular.x = 0.0
        cmd.angular.y = 0.0
        cmd.angular.z = 0.0
        return cmd

    # ------------------------------------------------------------------ #
    #   UTILS
    # ------------------------------------------------------------------ #

    def transition_to(self, new_state: int):
        self.state = new_state
        self.state_start_time = time.time()
        self.get_logger().info(f'🔄 → {self.get_state_name()}')

    def get_state_name(self) -> str:
        names = {
            self.SUBMERGING: 'SUBMERGING',
            self.CRUISING: 'CRUISING',
            self.FINAL_ALIGNMENT: 'FINAL_ALIGNMENT',
            self.PASSING: 'PASSING',
            self.CLEARING: 'CLEARING',
            self.UTURN: 'UTURN',
            self.REVERSE_CRUISING: 'REVERSE_CRUISING',
            self.REVERSE_ALIGNMENT: 'REVERSE_ALIGNMENT',
            self.REVERSE_PASSING: 'REVERSE_PASSING',
            self.REVERSE_CLEARING: 'REVERSE_CLEARING',
            self.SURFACING: 'SURFACING',
            self.COMPLETED: 'COMPLETED',
        }
        return names.get(self.state, 'UNKNOWN')

    @staticmethod
    def normalize_angle(angle: float) -> float:
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle


def main(args=None):
    rclpy.init(args=args)
    node = FinalQualificationNavigator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
