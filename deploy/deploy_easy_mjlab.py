#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np
import onnxruntime as ort
import yaml
import math
from collections import deque
from std_msgs.msg import Float32MultiArray, Bool
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import String
import argparse


CONTROLLER_JOINT_NAMES = [
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
]


class CustomRobotDeployEasyMjlabNode(Node):

    def __init__(self, config_file):
        super().__init__('custom_robot_deploy_easy_mjlab_node')
        self.jointPos = None
        self.jointVel = None
        self.imu_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.imu_angVel = None
        self.cmd_vals = None

        self.prev_action = None
        self.policy_step_count = 0

        self.filtered_target_q = None
        self.target_q_alpha = 0.2

        self.hist_obs = None

        self.decimation_counter = 0
        self.current_action = None
        self.load_config(config_file)
        self.init_policy()
        self.create_subscriptions()
        self.create_publishers()

        self.timer = self.create_timer(self.simulation_dt, self.control_loop)
        self.get_logger().info('Custom Robot Deploy Node initialized')
        self.get_logger().info(f'Control frequency: {1/self.simulation_dt:.1f}Hz')
        self.get_logger().info(f'Policy frequency: {1/(self.simulation_dt * self.decimation):.1f}Hz')
        self.get_logger().info(f'Decimation factor: {self.decimation}')

        self.hw_switch_enabled = False
        self.control_mode = 'WALK_12DOF'
        self.get_logger().info("HW switch default: OFF")
        self.get_logger().info("Control mode default: WALK_12DOF")

    def load_config(self, config_file):
        with open(config_file, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        self.policy_path = config["policy_path"]
        self.standup_policy_path = config["standup_policy_path"]
        self.simulation_dt = config["simulation_dt"]

        self.decimation = config["decimation"]
        control_freq = 1.0 / self.simulation_dt
        policy_freq = control_freq / self.decimation

        if abs(policy_freq - 100.0) > 1.0:
            self.get_logger().warn(f"Policy freq {policy_freq:.1f}Hz != target 100Hz")
            self.get_logger().info(f"Suggest: simulation_dt={1.0/(100.0*self.decimation):.6f}")

        self.frame_stack = config.get("frame_stack", 1)
        self.base_num_single_obs = 45

        phase_cfg = config.get("phase", {})
        self.phase_type = phase_cfg.get("type", "simple")
        self.phase_dim = 6 if self.phase_type == "gait" else 2
        self.phase_period = float(phase_cfg.get("period", 0.6))
        self.command_stand_threshold = float(phase_cfg.get("stand_threshold", 0.1))

        if self.phase_type == "gait":
            self.gait_air_ratio_l = float(phase_cfg.get("gait_air_ratio_l", 0.38))
            self.gait_air_ratio_r = float(phase_cfg.get("gait_air_ratio_r", 0.38))
            self.gait_phase_offset_l = float(phase_cfg.get("gait_phase_offset_l", 0.38))
            self.gait_phase_offset_r = float(phase_cfg.get("gait_phase_offset_r", 0.88))
            self.gait_cycle = float(phase_cfg.get("gait_cycle", 0.85))

        expected_single_obs = self.base_num_single_obs + self.phase_dim
        configured_single_obs = config.get("num_single_obs", expected_single_obs)
        if configured_single_obs != expected_single_obs:
            self.get_logger().warn(
                f"Config num_single_obs={configured_single_obs}, "
                f"but phase_type={self.phase_type} expects {expected_single_obs}; "
                f"using {expected_single_obs}"
            )
        self.num_single_obs = expected_single_obs
        self.num_observations = self.frame_stack * self.num_single_obs
        self.policy_dt = self.simulation_dt * self.decimation

        self.clip_observations = config["clip_observations"]
        self.clip_actions = config["clip_actions"]
        self.action_scale = config["action_scale"]
        self.target_q_alpha = float(config.get("target_filter_alpha", 0.2))
        self.num_actions = config["num_actions"]

        self.init_qpos_walk = np.array(config["default_angles"], dtype=np.float32)
        self.init_qpos_standup = np.array(config["default_angles_standup"], dtype=np.float32)
        self.kps = np.array(config["robot_config"]["kps"], dtype=np.float32)
        self.kds = np.array(config["robot_config"]["kds"], dtype=np.float32)

        limits = config["robot_config"].get("joint_pos_limits", {})
        if limits:
            self.joint_pos_low = np.array(limits.get("low"), dtype=np.float32)
            self.joint_pos_high = np.array(limits.get("high"), dtype=np.float32)
        else:
            self.joint_pos_low = None
            self.joint_pos_high = None

        self.joint_names = config.get("joint_names", [])
        self.controller_joint_names = config.get("controller_joint_names", CONTROLLER_JOINT_NAMES)
        self.policy_to_controller_indices = self.resolve_policy_to_controller_indices()

    def resolve_policy_to_controller_indices(self):
        if not self.joint_names:
            return np.arange(self.num_actions, dtype=np.int64)
        if len(self.joint_names) != self.num_actions:
            raise ValueError(
                f"joint_names length {len(self.joint_names)} != num_actions {self.num_actions}"
            )
        if len(self.controller_joint_names) != self.num_actions:
            raise ValueError(
                f"controller_joint_names length {len(self.controller_joint_names)} "
                f"!= num_actions {self.num_actions}"
            )

        missing = [name for name in self.controller_joint_names if name not in self.joint_names]
        if missing:
            raise ValueError(f"controller_joint_names missing from joint_names: {missing}")

        return np.array(
            [self.joint_names.index(name) for name in self.controller_joint_names],
            dtype=np.int64,
        )

    def to_controller_order(self, values):
        return np.asarray(values, dtype=np.float32)[self.policy_to_controller_indices]

    def init_policy(self):
        self.prev_action = np.zeros(self.num_actions, dtype=np.float32)
        self.current_action = np.zeros(self.num_actions, dtype=np.float32)

        self.hist_obs = deque(maxlen=self.frame_stack)
        for _ in range(self.frame_stack):
            self.hist_obs.append(np.zeros((1, self.num_single_obs), dtype=np.float32))

        self.policy = ort.InferenceSession(self.policy_path, providers=['CPUExecutionProvider'])
        self.standup_policy = ort.InferenceSession(self.standup_policy_path, providers=['CPUExecutionProvider'])
        self.policy_single_obs = self.resolve_policy_single_obs_dim(self.policy, self.policy_path)
        self.standup_policy_single_obs = self.resolve_policy_single_obs_dim(
            self.standup_policy, self.standup_policy_path
        )
        print("**********************************************************")
        print(f"Walk policy: {self.policy_path}")
        print(f"Stand policy: {self.standup_policy_path}")
        print("**********************************************************")
        self.get_logger().info(
            f"Node single_obs={self.num_single_obs}, "
            f"walk single_obs={self.policy_single_obs}, "
            f"stand single_obs={self.standup_policy_single_obs}, "
            f"frame_stack={self.frame_stack}"
        )
        self.get_logger().info(
            f"Publish joint order: {self.controller_joint_names}; "
            f"policy_to_controller_indices={self.policy_to_controller_indices.tolist()}"
        )

        self.start_time = self.get_clock().now()

    def resolve_policy_single_obs_dim(self, policy, policy_path):
        input_shape = policy.get_inputs()[0].shape
        if len(input_shape) >= 2 and isinstance(input_shape[1], int):
            input_dim = input_shape[1]
            if input_dim % self.frame_stack == 0:
                single_obs_dim = input_dim // self.frame_stack
                if single_obs_dim > self.num_single_obs:
                    self.get_logger().warn(
                        f"{policy_path} input_dim={input_dim}, single={single_obs_dim} "
                        f"> node {self.num_single_obs}"
                    )
                return single_obs_dim
            self.get_logger().warn(
                f"{policy_path} input_dim={input_dim} not divisible by frame_stack={self.frame_stack}"
            )
        return self.num_single_obs

    def create_subscriptions(self):
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.joint_subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            sensor_qos)
        self.imu_subscription = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            sensor_qos)
        self.cmd_vel_subscription = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            sensor_qos)
        self.hw_switch_subscription = self.create_subscription(
            Bool,
            '/hwswitch',
            self.hw_switch_callback,
            sensor_qos)
        self.control_mode_subscription = self.create_subscription(
            String,
            '/control_mode',
            self.control_mode_callback,
            sensor_qos)

    def create_publishers(self):
        control_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.target_torque_pub = self.create_publisher(Float32MultiArray, '/targetTorque', control_qos)
        self.target_pos_pub = self.create_publisher(Float32MultiArray, '/targetPos', control_qos)
        self.target_vel_pub = self.create_publisher(Float32MultiArray, '/targetVel', control_qos)
        self.target_kp_pub = self.create_publisher(Float32MultiArray, '/targetKp', control_qos)
        self.target_kd_pub = self.create_publisher(Float32MultiArray, '/targetKd', control_qos)
        self.policy_action = self.create_publisher(Float32MultiArray, '/policy_action', control_qos)

    def joint_state_callback(self, msg: JointState):
        if len(msg.position) != self.num_actions or len(msg.velocity) != self.num_actions:
            self.get_logger().error(
                f"JointState length wrong: pos={len(msg.position)}, vel={len(msg.velocity)}"
            )
            return
        try:
            if msg.name and self.joint_names:
                name_to_idx = {name: i for i, name in enumerate(msg.name)}
                indices = [name_to_idx[name] for name in self.joint_names]
            else:
                indices = list(range(self.num_actions))
            self.jointPos = np.array([msg.position[i] for i in indices], dtype=np.float32)
            self.jointVel = np.array([msg.velocity[i] for i in indices], dtype=np.float32)
        except Exception as e:
            self.get_logger().error(f"Parse joint state failed: {e}")
            return

    def imu_callback(self, msg):
        self.imu_quat = np.array([msg.orientation.w, msg.orientation.x,
                                  msg.orientation.y, msg.orientation.z], dtype=np.float32)
        self.imu_angVel = np.array([msg.angular_velocity.x, msg.angular_velocity.y,
                                   msg.angular_velocity.z], dtype=np.float32)

    def cmd_vel_callback(self, msg):
        self.cmd_vals = np.array([msg.linear.x, msg.linear.y, msg.angular.z], dtype=np.float32)

    def hw_switch_callback(self, msg):
        prev_state = self.hw_switch_enabled
        self.hw_switch_enabled = msg.data

        if prev_state != self.hw_switch_enabled:
            if self.hw_switch_enabled:
                self.start_time = self.get_clock().now()
                self.decimation_counter = 0
                self.get_logger().info("HW switch ON, reset policy state")

                self.prev_action = np.zeros(self.num_actions, dtype=np.float32)
                self.current_action = np.zeros(self.num_actions, dtype=np.float32)
                self.policy_step_count = 0
                self.filtered_target_q = None

                self.hist_obs.clear()
                for _ in range(self.frame_stack):
                    self.hist_obs.append(np.zeros((1, self.num_single_obs), dtype=np.float32))
            else:
                self.get_logger().info("HW switch OFF, stop publishing")

    def control_mode_callback(self, msg):
        requested_mode = msg.data
        self.control_mode = "WALK_12DOF"
        if requested_mode != self.control_mode:
            self.get_logger().info(
                f"Received control_mode={requested_mode}, forcing WALK_12DOF"
            )

    def pd_control(self, target_q, q, kp, target_dq, dq, kd):
        return (target_q - q) * kp + (target_dq - dq) * kd

    def quat_rotate_inverse(self, q_ros, v):
        q_sim2sim = np.array([q_ros[1], q_ros[2], q_ros[3], q_ros[0]], dtype=np.float32)
        q_w = q_sim2sim[-1]
        q_vec = q_sim2sim[:3]
        a = v * (2.0 * q_w**2 - 1.0)
        b = np.cross(q_vec, v) * q_w * 2.0
        c = q_vec * np.dot(q_vec, v) * 2.0
        return a - b + c

    def compute_phase_obs(self, command):
        if self.phase_type == "gait":
            t = self.policy_step_count * self.policy_dt / self.gait_cycle
            phase_offset = np.array(
                [self.gait_phase_offset_l, self.gait_phase_offset_r], dtype=np.float32
            )
            gait_phase = (t + phase_offset) % 1.0
            phase_ratio = np.array(
                [self.gait_air_ratio_l, self.gait_air_ratio_r], dtype=np.float32
            )
            return np.concatenate([
                np.sin(2.0 * math.pi * gait_phase),
                np.cos(2.0 * math.pi * gait_phase),
                phase_ratio,
            ]).astype(np.float32)

        phase = (self.policy_step_count * self.policy_dt) % self.phase_period
        phase = phase / self.phase_period
        obs = np.array(
            [math.sin(2.0 * math.pi * phase), math.cos(2.0 * math.pi * phase)],
            dtype=np.float32,
        )
        if np.linalg.norm(command) < self.command_stand_threshold:
            obs[:] = 0.0
        return obs

    def build_policy_input(self, single_obs_dim):
        if single_obs_dim > self.num_single_obs:
            self.get_logger().error(
                f"Model needs single_obs={single_obs_dim} > node={self.num_single_obs}"
            )
            return None
        return np.concatenate(
            [h[:, :single_obs_dim].reshape(1, -1) for h in self.hist_obs], axis=1
        )

    def run_policy_inference(self):
        command = (
            np.asarray(self.cmd_vals, dtype=np.float32)
            if self.cmd_vals is not None
            else np.zeros(3, dtype=np.float32)
        )
        current_policy = self.policy
        current_policy_single_obs = self.policy_single_obs
        self.init_qpos = self.init_qpos_walk

        project_gravity = self.quat_rotate_inverse(self.imu_quat, np.array([0, 0, -1], dtype=np.float32))
        obs = np.zeros((1, self.num_single_obs), dtype=np.float32)
        obs[0, 0:3] = np.asarray(self.imu_angVel, dtype=np.float32)
        obs[0, 3:6] = project_gravity
        obs[0, 6:9] = np.asarray(command, dtype=np.float32)
        obs[0, 9:9 + self.phase_dim] = self.compute_phase_obs(command)
        joint_pos_start = 9 + self.phase_dim
        joint_vel_start = joint_pos_start + self.num_actions
        action_start = joint_vel_start + self.num_actions
        obs[0, joint_pos_start:joint_vel_start] = (
            self.jointPos[:self.num_actions] - self.init_qpos
        ).astype(np.float32)
        obs[0, joint_vel_start:action_start] = self.jointVel[:self.num_actions].astype(np.float32)
        obs[0, action_start:action_start + self.num_actions] = np.asarray(
            self.prev_action, dtype=np.float32
        )

        obs = np.clip(obs, -self.clip_observations, self.clip_observations)
        self.hist_obs.append(obs.copy())

        policy_input = self.build_policy_input(current_policy_single_obs)
        if policy_input is None:
            return

        expected_obs_dim = self.frame_stack * current_policy_single_obs
        if policy_input.shape[1] < expected_obs_dim:
            pad_left = expected_obs_dim - policy_input.shape[1]
            pad = np.zeros((1, pad_left), dtype=np.float32)
            policy_input = np.concatenate((pad, policy_input), axis=1)

        input_name = current_policy.get_inputs()[0].name
        try:
            res = current_policy.run(None, {input_name: policy_input.astype(np.float32)})
        except Exception as e:
            self.get_logger().error(f"ONNX inference failed: {e}")
            return

        action = np.array(res[0]).squeeze()
        action = np.clip(action, -self.clip_actions, self.clip_actions)

        self.current_action = action.copy()
        self.policy_step_count += 1

    def control_loop(self):
        if self.jointPos is None or self.jointVel is None or self.imu_angVel is None:
            if self.imu_angVel is None:
                self.get_logger().warn("IMU data not ready, skip control")
            if self.jointPos is None or self.jointVel is None:
                self.get_logger().warn("Joint state not ready, skip control")
            return

        if not self.hw_switch_enabled:
            self.get_logger().warn("HW switch OFF, skip control")
            return

        if self.cmd_vals is None:
            self.get_logger().warn("No cmd_vel received, using default [0,0,0]")
            self.cmd_vals = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        if self.decimation_counter % self.decimation == 0:
            self.run_policy_inference()

        self.decimation_counter += 1
        action = self.current_action.copy()

        self.prev_action = action.copy()

        raw_target_q = action * self.action_scale + self.init_qpos

        if self.joint_pos_low is not None and self.joint_pos_high is not None:
            raw_target_q = np.clip(raw_target_q, self.joint_pos_low, self.joint_pos_high)

        if self.filtered_target_q is None:
            self.filtered_target_q = raw_target_q.copy()
        else:
            self.filtered_target_q = self.target_q_alpha * raw_target_q + (
                1.0 - self.target_q_alpha
            ) * self.filtered_target_q

        target_q = self.filtered_target_q

        policy_action_msg = Float32MultiArray()
        policy_action_msg.data = target_q.tolist()
        self.policy_action.publish(policy_action_msg)

        target_dq = np.zeros(self.num_actions, dtype=np.float32)

        torque_msg = Float32MultiArray()
        pos_msg = Float32MultiArray()
        vel_msg = Float32MultiArray()

        torque_msg.data = [0.0] * self.num_actions
        pos_msg.data = self.to_controller_order(target_q).tolist()
        vel_msg.data = self.to_controller_order(target_dq).tolist()

        kp_msg = Float32MultiArray()
        kd_msg = Float32MultiArray()
        kp_msg.data = self.to_controller_order(self.kps).tolist()
        kd_msg.data = self.to_controller_order(self.kds).tolist()

        if self.hw_switch_enabled:
            self.target_torque_pub.publish(torque_msg)
            self.target_pos_pub.publish(pos_msg)
            self.target_vel_pub.publish(vel_msg)
            self.target_kp_pub.publish(kp_msg)
            self.target_kd_pub.publish(kd_msg)


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
        default="/home/wangshun/X_2/rl_locomotion/humanoid_deploy/config/custom_robot_easy_mjlab.yaml",
        help="Deploy config YAML file path")
    parsed_args = parser.parse_args()

    rclpy.init(args=args)
    node = CustomRobotDeployEasyMjlabNode(parsed_args.config)
    try:
        node.get_logger().info('Starting custom_robot_deploy_easy_mjlab_node...')
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted by user')
    except Exception as e:
        if 'node' in locals():
            node.get_logger().error(f'Error: {str(e)}')
    finally:
        if 'node' in locals():
            node.destroy_node()
        if rclpy.ok():
            node.get_logger().info('Shutting down...')
            rclpy.shutdown()


if __name__ == '__main__':
    main()
