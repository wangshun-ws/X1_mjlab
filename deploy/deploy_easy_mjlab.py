#!/usr/bin/env python3
"""
强化学习策略部署节点 (mjlab版)

功能：将训练好的ONNX强化学习策略部署到真实机器人下肢，实现行走、站立等运动控制。
- 订阅关节状态、IMU数据、速度指令、硬件开关
- 通过ONNX Runtime进行策略推理，输出目标关节位置
- 发布目标位置、速度、Kp/Kd等控制指令给底层控制器
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np
import onnxruntime as ort
import yaml
from collections import deque
from std_msgs.msg import Float32MultiArray, Bool
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu, JointState
import argparse

HISTORY_LAYOUT_FRAME_MAJOR = "frame_major_oldest_to_newest"
EXPECTED_OBSERVATION_NAMES = [
    "base_ang_vel",
    "projected_gravity",
    "command",
    "phase",
    "joint_pos",
    "joint_vel",
    "actions",
]
GAIT_PHASE_PERIOD = 0.8
GAIT_COMMAND_THRESHOLD = 0.1
DEFAULT_CONFIG = "/home/wangshun/X1_mjlab/deploy/config/custom_robot_easy_mjlab.yaml"


# 控制器关节名称列表（下肢12自由度，按控制器期望的顺序排列）
# 顺序：左右交替排列 - yaw, roll, pitch, knee, ankle_pitch, ankle_roll
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


def _metadata_list(value):
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


class DigitalLowPassFilter:
    """Second-order digital low-pass filter used for target joint smoothing."""

    def __init__(self, wc: float, ts: float):
        self.wc = float(wc)
        self.ts = float(ts)
        self._input_prev = [0.0, 0.0]
        self._output_prev = [0.0, 0.0]
        self._output = 0.0
        self._update_coefficients()

    def _update_coefficients(self) -> None:
        den = 2500.0 * self.ts * self.ts * self.wc * self.wc + 7071.0 * self.ts * self.wc + 10000.0
        self._in1 = 2500.0 * self.ts * self.ts * self.wc * self.wc / den
        self._in2 = 5000.0 * self.ts * self.ts * self.wc * self.wc / den
        self._in3 = 2500.0 * self.ts * self.ts * self.wc * self.wc / den
        self._out1 = -(5000.0 * self.ts * self.ts * self.wc * self.wc - 20000.0) / den
        self._out2 = -(2500.0 * self.ts * self.ts * self.wc * self.wc - 7071.0 * self.ts * self.wc + 10000.0) / den

    def input(self, value: float) -> None:
        value = float(value)
        self._output = (
            self._in1 * value
            + self._in2 * self._input_prev[0]
            + self._in3 * self._input_prev[1]
            + self._out1 * self._output_prev[0]
            + self._out2 * self._output_prev[1]
        )
        self._input_prev[1] = self._input_prev[0]
        self._input_prev[0] = value
        self._output_prev[1] = self._output_prev[0]
        self._output_prev[0] = self._output

    def output(self) -> float:
        return self._output

    def init(self, value: float) -> None:
        value = float(value)
        self._input_prev[0] = value
        self._input_prev[1] = value
        self._output_prev[0] = value
        self._output_prev[1] = value
        self._output = value

    def reset(self, value: float) -> None:
        self.init(value)


class CustomRobotDeployEasyMjlabNode(Node):
    """
    强化学习策略部署节点

    将ONNX策略模型部署到真实机器人，通过ROS2话题接收传感器数据，
    进行策略推理，并将目标关节位置等控制指令发布给底层控制器。

    主要功能：
    - 加载YAML配置文件，解析策略参数、PD参数、关节限位等
    - 加载ONNX策略模型（支持frame_stack帧堆叠、decimation降采样）
    - 订阅关节状态(/joint_states)、IMU(/imu/data)、速度指令(/cmd_vel)、硬件开关(/hwswitch)
    - 构建观测向量（角速度、重力投影、指令、步态相位、关节位置/速度、上一步策略动作）
    - 运行ONNX推理，输出关节目标位置
    - 支持二阶数字低通滤波平滑目标位置
    - 支持策略关节排序与控制器关节排序的映射
    """

    def __init__(self, config_file):
        """
        初始化部署节点

        Args:
            config_file: YAML配置文件路径，包含策略路径、PD参数、关节限位等
        """
        super().__init__('custom_robot_deploy_easy_mjlab_node')

        # ---- 传感器数据缓存 ----
        self.jointPos = None                        # 关节位置（12维）
        self.jointVel = None                        # 关节速度（12维）
        self.imu_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # IMU四元数 [w, x, y, z]
        self.imu_angVel = None                      # IMU角速度（3维）
        self.cmd_vals = None                        # 速度指令 [vx, vy, wz]
        self.command_limits = np.full(3, np.inf, dtype=np.float32)  # 速度指令限幅

        # ---- 策略状态变量 ----
        self.current_target_q_raw = None            # 当前未限位、未滤波目标关节角
        self.prev_action = None                     # 上一步策略原始输出（用于 actions 观测）

        # ---- 目标位置滤波 ----
        self.filtered_target_q = None               # 滤波后的目标关节位置
        self.target_lpf_wc = 100.0                  # 二阶低通截止角频率
        self.target_lpf_ts = 0.001                  # 二阶低通采样周期
        self.target_q_filters = []                  # 每个关节对应一个二阶低通

        # ---- 帧堆叠历史观测 ----
        self.hist_obs = None                        # 历史观测队列（deque）

        # ---- Decimation（降采样）相关 ----
        self.decimation_counter = 0                 # 降采样计数器
        self.policy_step_count = 0                  # 策略步计数器（用于步态相位）
        self.current_action = None                  # 当前策略输出的动作

        # 加载配置文件
        self.load_config(config_file)
        # 初始化ONNX策略模型
        self.init_policy()
        # 创建ROS2订阅器和发布器
        self.create_subscriptions()
        self.create_publishers()

        # 创建控制定时器，按 simulation_dt 周期调用 control_loop
        self.timer = self.create_timer(self.simulation_dt, self.control_loop)
        self.get_logger().info('Custom Robot Deploy Node initialized')
        self.get_logger().info(f'Control frequency: {1/self.simulation_dt:.1f}Hz')
        self.get_logger().info(f'Policy frequency: {1/(self.simulation_dt * self.decimation):.1f}Hz')
        self.get_logger().info(f'Decimation factor: {self.decimation}')

        # 硬件开关默认关闭，确保安全
        self.hw_switch_enabled = False
        self.get_logger().info("HW switch default: OFF")

    def load_config(self, config_file):
        """
        加载YAML配置文件，解析所有策略部署参数

        配置项包括：策略路径、控制周期、降采样、帧堆叠、
        PD参数、关节限位、速度指令限幅、关节名映射等

        Args:
            config_file: YAML配置文件绝对路径
        """
        with open(config_file, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        # ---- 基础参数 ----
        self.policy_path = config["policy_path"]              # ONNX策略模型路径
        self.simulation_dt = config["simulation_dt"]          # 控制周期（秒）

        # ---- Decimation（降采样）设置 ----
        # 策略更新频率 = 控制频率 / decimation
        self.decimation = config["decimation"]
        if self.decimation < 1:
            raise ValueError(f"decimation must be >= 1, got {self.decimation}")

        # ---- 帧堆叠设置（历史观测帧数） ----
        self.frame_stack = config.get("frame_stack", 1)

        # ---- 观测维度计算 ----
        # 单帧观测 = ang_vel(3) + gravity(3) + command(3) + phase(2)
        #           + joint_pos(12) + joint_vel(12) + actions(12)
        # 输入观测 = frame_stack × 单帧观测（用于帧堆叠）
        expected_single_obs = 47
        configured_single_obs = config.get("num_single_obs", expected_single_obs)
        if configured_single_obs != expected_single_obs:
            self.get_logger().warn(
                f"Config num_single_obs={configured_single_obs}, "
                f"but current X1_flat policy layout expects {expected_single_obs}; "
                f"using {expected_single_obs}."
            )
        self.num_single_obs = expected_single_obs
        self.num_observations = self.frame_stack * self.num_single_obs  # 策略总输入维度
        self.policy_dt = self.simulation_dt * self.decimation           # 策略步长时间

        # ---- 裁剪与缩放参数 ----
        self.clip_observations = config["clip_observations"]  # 观测值裁剪上限
        self.clip_actions = config["clip_actions"]            # 动作输出裁剪上限
        self.num_actions = config["num_actions"]              # 动作维度（关节数=12）
        self.action_scale = self.as_scale_array(config["action_scale"], "action_scale")
        target_lpf = config.get("target_lpf")
        if target_lpf is None:
            raise ValueError("target_lpf config is required.")
        if "wc" not in target_lpf or "ts" not in target_lpf:
            raise ValueError("target_lpf must define both 'wc' and 'ts'.")
        self.target_lpf_wc = float(target_lpf["wc"])
        self.target_lpf_ts = float(target_lpf["ts"])
        if self.target_lpf_wc <= 0.0:
            raise ValueError(f"target_lpf.wc must be > 0, got {self.target_lpf_wc}")
        if self.target_lpf_ts <= 0.0:
            raise ValueError(f"target_lpf.ts must be > 0, got {self.target_lpf_ts}")
        self.target_q_filters = [
            DigitalLowPassFilter(self.target_lpf_wc, self.target_lpf_ts)
            for _ in range(self.num_actions)
        ]

        # ---- 速度指令限幅 ----
        command_limits = config.get("command_limits", {})
        self.command_limits = np.abs(np.array([
            float(command_limits.get("lin_vel_x", np.inf)),   # 前向线速度限幅
            float(command_limits.get("lin_vel_y", np.inf)),   # 侧向线速度限幅
            float(command_limits.get("ang_vel_z", np.inf)),   # 偏航角速度限幅
        ], dtype=np.float32))

        # ---- 默认关节角度与PD参数 ----
        self.init_qpos = np.array(config["default_angles"], dtype=np.float32)
        self.kps = np.array(config["robot_config"]["kps"], dtype=np.float32)  # 比例增益
        self.kds = np.array(config["robot_config"]["kds"], dtype=np.float32)  # 微分增益

        # ---- 关节位置限位 ----
        limits = config["robot_config"].get("joint_pos_limits", {})
        if limits:
            self.joint_pos_low = np.array(limits.get("low"), dtype=np.float32)   # 下限
            self.joint_pos_high = np.array(limits.get("high"), dtype=np.float32) # 上限
        else:
            self.joint_pos_low = None
            self.joint_pos_high = None

        # ---- 关节名映射（策略顺序 → 控制器顺序） ----
        self.joint_names = config.get("joint_names", [])                           # 策略中的关节名顺序
        self.controller_joint_names = config.get("controller_joint_names", CONTROLLER_JOINT_NAMES)  # 控制器期望的关节名顺序
        self.policy_to_controller_indices = self.resolve_policy_to_controller_indices()  # 建立索引映射

    def as_scale_array(self, value, name):
        scale = np.asarray(value, dtype=np.float32)
        if scale.ndim == 0:
            return np.full(self.num_actions, float(scale), dtype=np.float32)
        scale = scale.reshape(-1)
        if scale.shape[0] != self.num_actions:
            raise ValueError(
                f"{name} length {scale.shape[0]} != num_actions {self.num_actions}"
            )
        return scale.astype(np.float32)

    def resolve_policy_to_controller_indices(self):
        """
        解析策略关节顺序到控制器关节顺序的索引映射

        策略输出的关节顺序可能与控制器期望的顺序不同（如策略按左右分块，
        控制器按左右交替排列），需要通过索引映射进行重排。

        Returns:
            np.array: 索引映射数组，用于 to_controller_order() 重排

        Raises:
            ValueError: 关节名长度不匹配或控制器关节名不在策略关节名中
        """
        # 如果没有配置关节名，使用默认顺序（0, 1, 2, ...）
        if not self.joint_names:
            return np.arange(self.num_actions, dtype=np.int64)

        # 校验关节名数量
        if len(self.joint_names) != self.num_actions:
            raise ValueError(
                f"joint_names length {len(self.joint_names)} != num_actions {self.num_actions}"
            )
        if len(self.controller_joint_names) != self.num_actions:
            raise ValueError(
                f"controller_joint_names length {len(self.controller_joint_names)} "
                f"!= num_actions {self.num_actions}"
            )

        # 检查控制器关节名是否都存在于策略关节名中
        missing = [name for name in self.controller_joint_names if name not in self.joint_names]
        if missing:
            raise ValueError(f"controller_joint_names missing from joint_names: {missing}")

        # 构建索引映射：对于每个控制器关节名，找到它在策略关节名中的位置
        return np.array(
            [self.joint_names.index(name) for name in self.controller_joint_names],
            dtype=np.int64,
        )

    def reset_target_q_filters(self, target_q):
        target_q = np.asarray(target_q, dtype=np.float32)
        if target_q.shape != (self.num_actions,):
            raise ValueError(
                f"target_q shape must be ({self.num_actions},), got {target_q.shape}"
            )
        for lpf, value in zip(self.target_q_filters, target_q, strict=False):
            lpf.reset(float(value))
        self.filtered_target_q = target_q.copy()

    def filter_target_q(self, target_q):
        target_q = np.asarray(target_q, dtype=np.float32)
        if target_q.shape != (self.num_actions,):
            raise ValueError(
                f"target_q shape must be ({self.num_actions},), got {target_q.shape}"
            )
        filtered = np.empty(self.num_actions, dtype=np.float32)
        for idx, (lpf, value) in enumerate(zip(self.target_q_filters, target_q, strict=False)):
            lpf.input(float(value))
            filtered[idx] = lpf.output()
        self.filtered_target_q = filtered
        return filtered

    def to_controller_order(self, values):
        """
        将策略输出的值按控制器期望的关节顺序重排

        Args:
            values: 策略顺序的数组（如目标位置、PD参数等）

        Returns:
            按控制器关节顺序排列的数组
        """
        return np.asarray(values, dtype=np.float32)[self.policy_to_controller_indices]

    def clip_command(self, command):
        """
        对速度指令进行限幅

        Args:
            command: [vx, vy, wz] 速度指令

        Returns:
            限幅后的速度指令
        """
        command = np.asarray(command, dtype=np.float32)
        return np.clip(command, -self.command_limits, self.command_limits)

    def build_phase_observation(self, command):
        """
        构建步态相位观测，与训练侧 mdp.phase(period=0.8) 保持一致。

        低速或站立指令下输出零相位，避免站立时强制进入周期步态。
        """
        command = np.asarray(command, dtype=np.float32)
        phase = np.zeros(2, dtype=np.float32)
        if np.linalg.norm(command) < GAIT_COMMAND_THRESHOLD:
            return phase
        global_phase = (
            (self.policy_step_count * self.policy_dt) % GAIT_PHASE_PERIOD
        ) / GAIT_PHASE_PERIOD
        phase_angle = global_phase * np.pi * 2.0
        phase[0] = np.sin(phase_angle)
        phase[1] = np.cos(phase_angle)
        return phase

    def init_policy(self):
        """
        初始化ONNX策略模型及相关状态变量

        - 创建ONNX Runtime推理会话（CPU执行）
        - 初始化上一帧目标关节角为默认姿态
        - 初始化帧堆叠观测队列（填充全零）
        - 验证策略输入维度与配置是否一致
        """
        self.current_action = np.zeros(self.num_actions, dtype=np.float32)
        self.prev_action = np.zeros(self.num_actions, dtype=np.float32)
        self.policy_step_count = 0
        self.current_target_q_raw = self.init_qpos.copy()
        self.reset_target_q_filters(self.current_target_q_raw)

        # 初始化帧堆叠历史观测队列，填充全零
        self.hist_obs = deque(maxlen=self.frame_stack)
        for _ in range(self.frame_stack):
            self.hist_obs.append(np.zeros((1, self.num_single_obs), dtype=np.float32))

        # 加载ONNX策略模型，使用CPU推理
        self.policy = ort.InferenceSession(self.policy_path, providers=['CPUExecutionProvider'])
        self.policy_input_dim = self.resolve_policy_input_dim(self.policy, self.policy_path)
        self.validate_policy_metadata(self.policy, self.policy_path)
        print("**********************************************************")
        print(f"Policy: {self.policy_path}")
        print("**********************************************************")
        self.get_logger().info(
            f"single_obs={self.num_single_obs}, frame_stack={self.frame_stack}, "
            f"policy_input_dim={self.policy_input_dim}"
        )
        self.get_logger().info(
            f"Publish joint order: {self.controller_joint_names}; "
            f"policy_to_controller_indices={self.policy_to_controller_indices.tolist()}"
        )

    def resolve_policy_input_dim(self, policy, policy_path):
        """
        解析并校验ONNX策略模型的输入维度

        确保模型输入维度与配置中的 frame_stack × num_single_obs 一致

        Args:
            policy: ONNX InferenceSession 对象
            policy_path: 模型文件路径（用于错误提示）

        Returns:
            int: 策略输入维度

        Raises:
            ValueError: 输入维度无法解析或与配置不匹配
        """
        input_shape = policy.get_inputs()[0].shape
        # 确保具有静态的输入维度（shape[1] 为整数）
        if len(input_shape) < 2 or not isinstance(input_shape[1], int):
            raise ValueError(
                f"{policy_path} must expose a static input dim; got shape {input_shape}"
            )

        input_dim = input_shape[1]
        expected_dim = self.frame_stack * self.num_single_obs
        # 验证维度匹配
        if input_dim != expected_dim:
            raise ValueError(
                f"{policy_path} input_dim={input_dim}, expected {expected_dim} "
                f"(frame_stack={self.frame_stack}, single_obs={self.num_single_obs})"
            )
        return input_dim

    def validate_policy_metadata(self, policy, policy_path):
        metadata = policy.get_modelmeta().custom_metadata_map or {}

        obs_names = _metadata_list(metadata.get("observation_names"))
        if obs_names and obs_names != EXPECTED_OBSERVATION_NAMES:
            raise ValueError(
                f"{policy_path} observation_names metadata does not match deploy layout: "
                f"policy={obs_names}, deploy={EXPECTED_OBSERVATION_NAMES}"
            )

        policy_joint_names = _metadata_list(metadata.get("joint_names"))
        if policy_joint_names and policy_joint_names != self.joint_names:
            raise ValueError(
                f"{policy_path} joint_names metadata does not match deploy config:\n"
                f"  policy={policy_joint_names}\n"
                f"  config={self.joint_names}"
            )

        history_layout = metadata.get("history_layout")
        if history_layout != HISTORY_LAYOUT_FRAME_MAJOR:
            raise ValueError(
                f"{policy_path} history_layout must be '{HISTORY_LAYOUT_FRAME_MAJOR}', "
                f"got {history_layout!r}."
            )

        self.get_logger().info(
            "Policy metadata verified: observation_names, joint_names, and history_layout match"
        )

    def create_subscriptions(self):
        """
        创建ROS2订阅器

        订阅话题：
        - /joint_states: 关节状态（位置、速度），使用BEST_EFFORT传输保证实时性
        - /imu/data: IMU数据（四元数姿态、角速度）
        - /cmd_vel: 速度指令 [vx, vy, wz]
        - /hwswitch: 硬件安全开关
        """
        # 传感器数据使用BEST_EFFORT QoS，降低延迟、允许丢帧
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

    def create_publishers(self):
        """
        创建ROS2发布器

        发布话题（均使用RELIABLE传输保证控制指令不丢失）：
        - /targetTorque: 目标力矩（当前固定发布零力矩）
        - /targetPos: 目标关节位置
        - /targetVel: 目标关节速度（当前固定发布零速度）
        - /targetKp: 比例增益
        - /targetKd: 微分增益
        - /policy_action: 策略输出的原始目标位置（供调试监控使用）
        """
        # 控制指令使用RELIABLE QoS，保证指令可靠送达
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
        """
        关节状态回调：从 /joint_states 话题解析关节位置和速度

        支持两种匹配方式：
        1. 通过关节名匹配（如果配置了 joint_names）
        2. 通过索引顺序匹配（默认前 num_actions 个关节）

        Args:
            msg: JointState 消息，包含 position、velocity 和 name 字段
        """
        try:
            if msg.name and self.joint_names:
                # 通过关节名精确匹配
                name_to_idx = {name: i for i, name in enumerate(msg.name)}
                missing = [name for name in self.joint_names if name not in name_to_idx]
                if missing:
                    self.get_logger().error(f"JointState missing joint names: {missing}")
                    return
                indices = [name_to_idx[name] for name in self.joint_names]
            else:
                # 按索引顺序取前 num_actions 个
                if len(msg.position) < self.num_actions or len(msg.velocity) < self.num_actions:
                    self.get_logger().error(
                        f"JointState too short without names: "
                        f"pos={len(msg.position)}, vel={len(msg.velocity)}, need>={self.num_actions}"
                    )
                    return
                indices = list(range(self.num_actions))
            # 提取关节位置和速度
            self.jointPos = np.array([msg.position[i] for i in indices], dtype=np.float32)
            self.jointVel = np.array([msg.velocity[i] for i in indices], dtype=np.float32)
        except Exception as e:
            self.get_logger().error(f"Parse joint state failed: {e}")
            return

    def imu_callback(self, msg):
        """
        IMU数据回调：提取姿态四元数和角速度

        Args:
            msg: Imu 消息，包含 orientation（四元数）和 angular_velocity（角速度）
        """
        # 四元数存储为 [w, x, y, z] 顺序
        self.imu_quat = np.array([msg.orientation.w, msg.orientation.x,
                                  msg.orientation.y, msg.orientation.z], dtype=np.float32)
        # 角速度 [x, y, z]
        self.imu_angVel = np.array([msg.angular_velocity.x, msg.angular_velocity.y,
                                   msg.angular_velocity.z], dtype=np.float32)

    def cmd_vel_callback(self, msg):
        """
        速度指令回调：接收遥控器/上层发送的期望速度，并进行限幅

        Args:
            msg: Twist 消息，linear.x/y 为线速度，angular.z 为偏航角速度
        """
        self.cmd_vals = self.clip_command([msg.linear.x, msg.linear.y, msg.angular.z])

    def hw_switch_callback(self, msg):
        """
        硬件安全开关回调

        当开关从OFF→ON时：重置所有策略状态（动作、观测历史、滤波、相位计数器），
        确保策略从干净的初始状态开始运行。
        当开关从ON→OFF时：停止发布控制指令，机器人进入安全状态。

        Args:
            msg: Bool 消息，True=开启，False=关闭
        """
        prev_state = self.hw_switch_enabled
        self.hw_switch_enabled = msg.data

        # 仅在状态变化时执行重置/停止操作
        if prev_state != self.hw_switch_enabled:
            if self.hw_switch_enabled:
                # 开关开启：重置所有策略状态
                self.decimation_counter = 0
                self.policy_step_count = 0
                self.get_logger().info("HW switch ON, reset policy state")

                self.current_action = np.zeros(self.num_actions, dtype=np.float32)
                self.prev_action = np.zeros(self.num_actions, dtype=np.float32)
                self.current_target_q_raw = self.init_qpos.copy()
                self.reset_target_q_filters(self.current_target_q_raw)

                # 清空并重新填充帧堆叠历史
                self.hist_obs.clear()
                for _ in range(self.frame_stack):
                    self.hist_obs.append(np.zeros((1, self.num_single_obs), dtype=np.float32))
            else:
                self.get_logger().info("HW switch OFF, stop publishing")


    def quat_rotate_inverse(self, q_ros, v):
        """
        用四元数的逆旋转对向量进行变换（计算重力在机器人坐标系下的投影）

        将ROS格式四元数 [w, x, y, z] 转为仿真格式 [x, y, z, w]，
        然后计算 q⁻¹ * v * q（逆旋转），得到重力向量在机器人本体坐标系中的投影。

        公式：v' = v·(2w²-1) - 2w·(qv × v) + 2·qv·(qv · v)

        Args:
            q_ros: ROS格式四元数 [w, x, y, z]
            v:     待旋转向量（通常为重力方向 [0, 0, -1]）

        Returns:
            旋转后的向量
        """
        # ROS [w,x,y,z] → 仿真 [x,y,z,w]
        q_sim2sim = np.array([q_ros[1], q_ros[2], q_ros[3], q_ros[0]], dtype=np.float32)
        q_w = q_sim2sim[-1]    # 标量部分
        q_vec = q_sim2sim[:3]  # 向量部分
        # 逆旋转公式（等价于共轭四元数的旋转）
        a = v * (2.0 * q_w**2 - 1.0)
        b = np.cross(q_vec, v) * q_w * 2.0
        c = q_vec * np.dot(q_vec, v) * 2.0
        return a - b + c

    def build_policy_input(self):
        """
        构建策略输入：将帧堆叠历史观测拼接为一个向量

        从 deque 中取出 frame_stack 帧观测，每帧为 [1, num_single_obs]，
        沿列方向拼接为 [1, frame_stack × num_single_obs] 的输入。

        Returns:
            np.array: 策略输入向量，shape [1, policy_input_dim]

        Raises:
            ValueError: 拼接后的维度与预期不符
        """
        policy_input = np.concatenate(list(self.hist_obs), axis=1)
        if policy_input.shape[1] != self.policy_input_dim:
            raise ValueError(
                f"Built policy input dim {policy_input.shape[1]} != expected {self.policy_input_dim}"
            )
        return policy_input

    def run_policy_inference(self):
        """
        执行一次策略推理

        流程：
        1. 获取当前速度指令并限幅
        2. 计算重力在机器人坐标系下的投影
        3. 构建观测向量：
           [IMU角速度(3) | 重力投影(3) | 速度指令(3) | 步态相位(2) |
            关节位置偏差(12) | 关节速度(12) | 上一步策略动作(12)]
        4. 裁剪观测值，更新帧堆叠历史
        5. 运行ONNX推理
        6. 裁剪动作输出，更新当前动作
        """
        # 获取速度指令（无指令时默认零）
        command = (
            np.asarray(self.cmd_vals, dtype=np.float32)
            if self.cmd_vals is not None
            else np.zeros(3, dtype=np.float32)
        )
        command = self.clip_command(command)

        # 计算重力方向在机器人本体坐标系中的投影
        # 世界坐标系重力 [0, 0, -1] 通过IMU姿态逆旋转到机器人坐标系
        project_gravity = self.quat_rotate_inverse(self.imu_quat, np.array([0, 0, -1], dtype=np.float32))
        phase = self.build_phase_observation(command)

        joint_pos = self.jointPos[:self.num_actions].astype(np.float32)
        joint_vel = self.jointVel[:self.num_actions].astype(np.float32)
        joint_pos_rel = (joint_pos - self.init_qpos).astype(np.float32)
        parts = [
            np.asarray(self.imu_angVel, dtype=np.float32),
            project_gravity.astype(np.float32),
            np.asarray(command, dtype=np.float32),
            phase,
            joint_pos_rel,
            joint_vel,
            self.prev_action.astype(np.float32),
        ]
        obs = np.concatenate(parts).reshape(1, -1)
        if obs.shape[1] != self.num_single_obs:
            self.get_logger().error(
                f"Built single obs dim {obs.shape[1]} != expected {self.num_single_obs}"
            )
            return

        # 裁剪观测值防止数值溢出
        obs = np.clip(obs, -self.clip_observations, self.clip_observations)
        # 将当前观测加入帧堆叠队列（deque自动丢弃最旧帧）
        self.hist_obs.append(obs.copy())

        # 拼接帧堆叠历史为策略输入
        try:
            policy_input = self.build_policy_input()
        except ValueError as e:
            self.get_logger().error(str(e))
            return

        # ---- ONNX 推理 ----
        input_name = self.policy.get_inputs()[0].name
        try:
            res = self.policy.run(None, {input_name: policy_input.astype(np.float32)})
        except Exception as e:
            self.get_logger().error(f"ONNX inference failed: {e}")
            return

        # 提取并裁剪动作
        action = np.array(res[0]).squeeze()
        action = np.clip(action, -self.clip_actions, self.clip_actions)
        if action.shape != (self.num_actions,):
            self.get_logger().error(
                f"Policy action shape must be ({self.num_actions},), got {action.shape}"
            )
            return

        self.current_action = action.copy()
        self.policy_step_count += 1

    def control_loop(self):
        """
        主控制循环（由定时器以 simulation_dt 周期调用）

        流程：
        1. 检查传感器数据就绪状态，未就绪则跳过
        2. 检查硬件开关状态，关闭则跳过
        3. 按 decimation 周期执行策略推理
        4. 将动作映射为目标关节位置：target_q = action × scale + init_qpos
        5. 关节限位裁剪
        6. 二阶数字低通滤波平滑
        7. 按控制器关节顺序重排并发布控制指令
        """
        # ---- 传感器数据就绪检查 ----
        if self.jointPos is None or self.jointVel is None or self.imu_angVel is None:
            if self.imu_angVel is None:
                self.get_logger().warn("IMU data not ready, skip control")
            if self.jointPos is None or self.jointVel is None:
                self.get_logger().warn("Joint state not ready, skip control")
            return

        # ---- 硬件开关检查 ----
        if not self.hw_switch_enabled:
            self.get_logger().warn("HW switch OFF, skip control")
            return

        # ---- 速度指令检查 ----
        if self.cmd_vals is None:
            self.get_logger().warn("No cmd_vel received, using default [0,0,0]")
            self.cmd_vals = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # ---- Decimation：每 decimation 个控制周期执行一次策略推理 ----
        if self.decimation_counter % self.decimation == 0:
            self.run_policy_inference()

        self.decimation_counter += 1
        action = self.current_action.copy()

        # ---- 动作映射到目标关节位置 ----
        # target_q = action × action_scale + init_qpos
        raw_target_q = action * self.action_scale + self.init_qpos
        self.current_target_q_raw = raw_target_q.astype(np.float32).copy()

        # ---- 二阶数字低通滤波平滑目标位置 ----
        target_q = self.filter_target_q(raw_target_q)

        # ---- 关节位置限位 ----
        # Butterworth filters can overshoot, so clip the filtered target before publishing.
        if self.joint_pos_low is not None and self.joint_pos_high is not None:
            target_q = np.clip(target_q, self.joint_pos_low, self.joint_pos_high)

        # ---- 发布策略原始动作（供调试监控） ----
        policy_action_msg = Float32MultiArray()
        policy_action_msg.data = target_q.tolist()
        self.policy_action.publish(policy_action_msg)

        # 目标速度设为零（位置控制模式）
        target_dq = np.zeros(self.num_actions, dtype=np.float32)

        # ---- 构造控制消息并发布 ----
        torque_msg = Float32MultiArray()
        pos_msg = Float32MultiArray()
        vel_msg = Float32MultiArray()

        torque_msg.data = [0.0] * self.num_actions            # 零力矩（由底层控制器自行计算）
        pos_msg.data = self.to_controller_order(target_q).tolist()   # 目标位置（重排为控制器顺序）
        vel_msg.data = self.to_controller_order(target_dq).tolist()  # 目标速度（零）

        kp_msg = Float32MultiArray()
        kd_msg = Float32MultiArray()
        kp_msg.data = self.to_controller_order(self.kps).tolist()    # Kp增益（重排为控制器顺序）
        kd_msg.data = self.to_controller_order(self.kds).tolist()    # Kd增益（重排为控制器顺序）

        # 仅在硬件开关开启时发布控制指令
        if self.hw_switch_enabled:
            self.target_torque_pub.publish(torque_msg)
            self.target_pos_pub.publish(pos_msg)
            self.target_vel_pub.publish(vel_msg)
            self.target_kp_pub.publish(kp_msg)
            self.target_kd_pub.publish(kd_msg)

        self.prev_action = action.copy()


def main(args=None):
    """
    程序入口：解析命令行参数，初始化ROS2节点并运行主循环

    用法：
        python deploy_easy_mjlab.py --config /path/to/config.yaml
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="RL策略部署节点 (mjlab版)")
    parser.add_argument("--config", type=str,
        default=DEFAULT_CONFIG,
        help="Deploy config YAML file path")
    parsed_args = parser.parse_args()

    # 初始化ROS2
    rclpy.init(args=args)
    # 创建部署节点
    node = CustomRobotDeployEasyMjlabNode(parsed_args.config)
    try:
        node.get_logger().info('Starting custom_robot_deploy_easy_mjlab_node...')
        # 进入ROS2事件循环
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted by user')
    except Exception as e:
        if 'node' in locals():
            node.get_logger().error(f'Error: {str(e)}')
    finally:
        # 清理资源
        if 'node' in locals():
            node.destroy_node()
        if rclpy.ok():
            node.get_logger().info('Shutting down...')
            rclpy.shutdown()


if __name__ == '__main__':
    main()
