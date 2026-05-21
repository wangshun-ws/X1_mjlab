#!/usr/bin/env python3
"""Standalone MuJoCo ONNX joystick simulation.

This script runs the flat-terrain policy observation used by CustomRobot-Flat:
  ang_vel, projected_gravity, command, phase, joint_pos, joint_vel, last_action.
It does not depend on mjlab or ROS2.
"""

from __future__ import annotations

import argparse
import csv
import errno
import math
import os
import struct
import time
from collections import deque
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE_XML = ROOT / "sim" / "scene.xml"
TRAINING_JOINT_ARMATURE = 0.05
TRAINING_JOINT_FRICTIONLOSS = 0.0
HISTORY_LAYOUT_FRAME_MAJOR = "frame_major_oldest_to_newest"

JOINT_NAMES = [
  "left_hip_yaw_joint",
  "left_hip_roll_joint",
  "left_hip_pitch_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_yaw_joint",
  "right_hip_roll_joint",
  "right_hip_pitch_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
]

NUM_ACTIONS = len(JOINT_NAMES)
NUM_SINGLE_OBS = 3 + 3 + 3 + 2 + NUM_ACTIONS * 3

DEFAULT_Q = np.array([0.0, 0.0, 0.2, -0.4, 0.2, 0.0, 0.0, 0.0, 0.2, -0.4, 0.2, 0.0], dtype=np.float32)
ACTION_SCALE = np.full(NUM_ACTIONS, 0.22, dtype=np.float32)

KPS = np.array([100.0, 150.0, 150.0, 180.0, 75.0, 30.0, 100.0, 150.0, 150.0, 180.0, 75.0, 30.0], dtype=np.float32)
KDS = np.array([1.0, 4.0, 4.0, 4.0, 2.0, 2.0, 1.0, 4.0, 4.0, 4.0, 2.0, 2.0], dtype=np.float32)
TAU_LIMIT = np.array([80.0, 120.0, 212.0, 180.0, 80.0, 20.0, 80.0, 120.0, 212.0, 180.0, 80.0, 20.0], dtype=np.float32)

JOINT_POS_LOW = np.array([-0.518, -0.175, -0.524, -1.745, -1.082, -0.664, -0.785, -0.175, -0.524, -1.745, -1.082, -0.664], dtype=np.float32)
JOINT_POS_HIGH = np.array([0.518, 0.960, 1.047, 0.087, 1.082, 0.664, 0.785, 0.960, 1.047, 0.087, 1.082, 0.664], dtype=np.float32)

CMD_LIMIT = np.array([1.0, 0.5, 0.5], dtype=np.float32)

FRAME_STACK = 10
CLIP_OBSERVATIONS = 100.0
CLIP_ACTIONS = 100.0
PHASE_PERIOD = 0.8
COMMAND_STAND_THRESHOLD = 0.1
SIM_DT = 0.001
DECIMATION = 10
TARGET_Q_ALPHA = 0.2
DEFAULT_RENDER_DECIMATION = 20


def _metadata_list(value: str | None) -> list[str]:
  if not value:
    return []
  return [item.strip() for item in value.split(",") if item.strip()]


def _latest_policy() -> Path | None:
  root = ROOT / "logs" / "rsl_rl" / "custom_robot_velocity"
  candidates = sorted(root.glob("*/policy.onnx"), key=lambda p: p.stat().st_mtime)
  return candidates[-1] if candidates else None


class TorqueLogger:
  def __init__(self, path: Path, joint_names: list[str], log_decimation: int):
    self.path = path
    self.joint_names = list(joint_names)
    self.log_decimation = max(1, log_decimation)
    self._rows_written = 0
    self.path.parent.mkdir(parents=True, exist_ok=True)
    self._file = self.path.open("w", newline="", encoding="utf-8")
    self._writer = csv.writer(self._file)
    header = [
      "sim_time",
      "sim_step",
      "policy_step",
      "cmd_vx",
      "cmd_vy",
      "cmd_wz",
    ]
    header += [f"tau_cmd_{joint_name}" for joint_name in self.joint_names]
    header += [f"tau_applied_{joint_name}" for joint_name in self.joint_names]
    self._writer.writerow(header)

  def maybe_write(
    self,
    sim_time: float,
    sim_step: int,
    policy_step: int,
    command: np.ndarray,
    tau_cmd: np.ndarray,
    tau_applied: np.ndarray,
  ) -> None:
    if sim_step % self.log_decimation != 0:
      return
    row = [
      f"{sim_time:.6f}",
      sim_step,
      policy_step,
      f"{float(command[0]):.6f}",
      f"{float(command[1]):.6f}",
      f"{float(command[2]):.6f}",
    ]
    row += [f"{float(value):.6f}" for value in tau_cmd]
    row += [f"{float(value):.6f}" for value in tau_applied]
    self._writer.writerow(row)
    self._rows_written += 1
    if self._rows_written % 200 == 0:
      self._file.flush()

  def close(self) -> None:
    self._file.flush()
    self._file.close()


def quat_rotate_inverse(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
  q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float32)
  q_w = q_xyzw[-1]
  q_vec = q_xyzw[:3]
  a = v * (2.0 * q_w**2 - 1.0)
  b = np.cross(q_vec, v) * q_w * 2.0
  c = q_vec * np.dot(q_vec, v) * 2.0
  return a - b + c


class JoystickUnavailableError(RuntimeError):
  pass


class JoystickVelocitySource:
  def __init__(self, args: argparse.Namespace, name: str):
    self.args = args
    self.name = name
    self.enabled = args.enabled
    self.walking = args.walking
    self._last_enable_button = False
    self._last_mode_button = False
    self._command = np.zeros(3, dtype=np.float32)

    print(f"[INFO] joystick backend: {self.name}")
    print("[INFO] left stick=vx/vy, right stick X=yaw, LB=enable, RB=stand/walk")
    self._print_state()

  @property
  def command(self) -> np.ndarray:
    return self._command

  def poll(self) -> np.ndarray:
    self._poll_device()
    self._update_toggles()

    vx = -self._axis_with_deadzone(self.args.joy_lin_vel_x_axis)
    vy = -self._axis_with_deadzone(self.args.joy_lin_vel_y_axis)
    yaw = self._axis_with_deadzone(self.args.joy_ang_vel_z_axis)

    if not self.enabled or not self.walking:
      self._command[:] = 0.0
    else:
      self._command[:] = (
        vx * self.args.joy_max_lin_vel_x,
        vy * self.args.joy_max_lin_vel_y,
        yaw * self.args.joy_max_ang_vel_z,
      )
    return self._command

  def close(self) -> None:
    pass

  def _poll_device(self) -> None:
    raise NotImplementedError

  def _axis(self, axis: int) -> float:
    raise NotImplementedError

  def _button(self, button: int) -> bool:
    raise NotImplementedError

  def _axis_with_deadzone(self, axis: int) -> float:
    value = self._axis(axis)
    return 0.0 if abs(value) <= self.args.joy_deadzone else value

  def _print_state(self) -> None:
    print(
      f"[INFO] joystick state: {'enabled' if self.enabled else 'disabled'}, "
      f"{'walk' if self.walking else 'stand'}"
    )

  def _update_toggles(self) -> None:
    enable_button = self._button(self.args.joy_enable_button)
    if enable_button and not self._last_enable_button:
      self.enabled = not self.enabled
      self._print_state()
    self._last_enable_button = enable_button

    mode_button = self._button(self.args.joy_mode_button)
    if mode_button and not self._last_mode_button:
      self.walking = not self.walking
      self._print_state()
    self._last_mode_button = mode_button


class PygameJoystickVelocitySource(JoystickVelocitySource):
  def __init__(self, args: argparse.Namespace):
    try:
      import pygame
    except ModuleNotFoundError as exc:
      raise JoystickUnavailableError("pygame is not installed") from exc

    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
      os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    self._pygame = pygame
    pygame.init()
    pygame.joystick.init()
    joystick_count = pygame.joystick.get_count()
    if joystick_count <= args.joy_device_index:
      pygame.quit()
      raise JoystickUnavailableError(
        f"pygame found {joystick_count} joystick(s), index {args.joy_device_index} requested"
      )
    self._joystick = pygame.joystick.Joystick(args.joy_device_index)
    self._joystick.init()
    super().__init__(args, f"pygame:{self._joystick.get_name()}")

  def close(self) -> None:
    self._joystick.quit()
    self._pygame.quit()

  def _poll_device(self) -> None:
    self._pygame.event.pump()

  def _axis(self, axis: int) -> float:
    if axis < 0 or axis >= self._joystick.get_numaxes():
      return 0.0
    return float(self._joystick.get_axis(axis))

  def _button(self, button: int) -> bool:
    if button < 0 or button >= self._joystick.get_numbuttons():
      return False
    return bool(self._joystick.get_button(button))


class LinuxJoystickVelocitySource(JoystickVelocitySource):
  _EVENT = struct.Struct("IhBB")
  _JS_EVENT_BUTTON = 0x01
  _JS_EVENT_AXIS = 0x02
  _JS_EVENT_INIT = 0x80

  def __init__(self, args: argparse.Namespace):
    self._axes: dict[int, float] = {}
    self._buttons: dict[int, bool] = {}
    try:
      self._fd = os.open(args.joy_device, os.O_RDONLY | os.O_NONBLOCK)
    except OSError as exc:
      raise JoystickUnavailableError(
        f"cannot open joystick device {args.joy_device}: {exc.strerror}"
      ) from exc
    super().__init__(args, f"linuxjs:{args.joy_device}")
    self._poll_device()

  def close(self) -> None:
    os.close(self._fd)

  def _poll_device(self) -> None:
    while True:
      try:
        data = os.read(self._fd, self._EVENT.size)
      except BlockingIOError:
        return
      except OSError as exc:
        if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
          return
        raise
      if len(data) != self._EVENT.size:
        return
      _time_ms, value, event_type, number = self._EVENT.unpack(data)
      event_type &= ~self._JS_EVENT_INIT
      if event_type == self._JS_EVENT_AXIS:
        scale = 32767.0 if value >= 0 else 32768.0
        self._axes[number] = max(-1.0, min(1.0, value / scale))
      elif event_type == self._JS_EVENT_BUTTON:
        self._buttons[number] = bool(value)

  def _axis(self, axis: int) -> float:
    return self._axes.get(axis, 0.0)

  def _button(self, button: int) -> bool:
    return self._buttons.get(button, False)


def create_joystick(args: argparse.Namespace) -> JoystickVelocitySource:
  errors: list[str] = []
  if args.joy_backend in ("auto", "pygame"):
    try:
      return PygameJoystickVelocitySource(args)
    except JoystickUnavailableError as exc:
      errors.append(f"pygame: {exc}")
      if args.joy_backend == "pygame":
        raise
  if args.joy_backend in ("auto", "linuxjs"):
    try:
      return LinuxJoystickVelocitySource(args)
    except JoystickUnavailableError as exc:
      errors.append(f"linuxjs: {exc}")
      if args.joy_backend == "linuxjs":
        raise
  raise JoystickUnavailableError("no joystick backend is available:\n  " + "\n  ".join(errors))


class OnnxPolicySim:
  def __init__(self, args: argparse.Namespace):
    self.args = args
    self.policy_path = args.policy
    if self.policy_path is None:
      self.policy_path = _latest_policy()
    if self.policy_path is None:
      raise FileNotFoundError("No policy.onnx found. Pass --policy explicitly.")
    if not self.policy_path.exists():
      raise FileNotFoundError(f"policy not found: {self.policy_path}")

    self.num_actions = NUM_ACTIONS
    self.num_single_obs = NUM_SINGLE_OBS
    self.frame_stack = FRAME_STACK
    self.clip_observations = CLIP_OBSERVATIONS
    self.clip_actions = CLIP_ACTIONS
    self.phase_period = PHASE_PERIOD
    self.command_stand_threshold = COMMAND_STAND_THRESHOLD
    self.sim_dt = SIM_DT
    self.decimation = DECIMATION
    self.policy_dt = self.sim_dt * self.decimation
    self.target_q_alpha = TARGET_Q_ALPHA
    self.joint_armature = TRAINING_JOINT_ARMATURE
    self.joint_frictionloss = TRAINING_JOINT_FRICTIONLOSS

    self.joint_names = list(JOINT_NAMES)
    self.default_q = DEFAULT_Q.copy()
    self.action_scale = ACTION_SCALE.copy()
    self.kps = KPS.copy()
    self.kds = KDS.copy()
    self.tau_limit = TAU_LIMIT.copy()
    self.joint_pos_low = JOINT_POS_LOW.copy()
    self.joint_pos_high = JOINT_POS_HIGH.copy()
    self.cmd_limit = CMD_LIMIT.copy()
    self.last_tau_cmd = np.zeros(self.num_actions, dtype=np.float32)
    self.last_tau_applied = np.zeros(self.num_actions, dtype=np.float32)
    self.torque_logger: TorqueLogger | None = None

    self._load_model()
    self._load_policy()
    if self.args.torque_log is not None:
      self.torque_logger = TorqueLogger(
        path=self.args.torque_log,
        joint_names=self.joint_names,
        log_decimation=self.args.torque_log_decimation,
      )
      print(f"[INFO] torque log: {self.args.torque_log} (every {self.torque_logger.log_decimation} sim steps)")
    self._reset()
    self._initialize_history()

  def _load_model(self) -> None:
    self.model = mujoco.MjModel.from_xml_path(str(DEFAULT_SCENE_XML))
    self.data = mujoco.MjData(self.model)
    self.model.opt.timestep = self.sim_dt

    self.free_joint_id = -1
    for joint_name in ("floating_base_joint", "root_freejoint"):
      joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
      if joint_id >= 0:
        self.free_joint_id = int(joint_id)
        break
    if self.free_joint_id < 0:
      free_joint_ids = np.flatnonzero(self.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
      if len(free_joint_ids) > 0:
        self.free_joint_id = int(free_joint_ids[0])
      else:
        raise ValueError(
          "scene must contain a free joint (e.g. 'floating_base_joint' or 'root_freejoint')"
        )
    self.free_qpos_adr = int(self.model.jnt_qposadr[self.free_joint_id])

    self.joint_ids = np.asarray(
      [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.joint_names],
      dtype=np.int32,
    )
    if np.any(self.joint_ids < 0):
      missing = [n for n, jid in zip(self.joint_names, self.joint_ids) if jid < 0]
      raise ValueError(f"missing joints in scene: {missing}")
    self.qpos_adrs = self.model.jnt_qposadr[self.joint_ids]
    self.qvel_adrs = self.model.jnt_dofadr[self.joint_ids]

    self.actuator_ids = []
    for jid, joint_name in zip(self.joint_ids, self.joint_names):
      matches = np.flatnonzero(self.model.actuator_trnid[:, 0] == jid)
      if len(matches) == 0:
        raise ValueError(f"missing motor actuator for joint {joint_name}")
      self.actuator_ids.append(int(matches[0]))
    self.actuator_ids = np.asarray(self.actuator_ids, dtype=np.int32)

    for jid in self.joint_ids:
      dof_adr = self.model.jnt_dofadr[jid]
      self.model.dof_armature[dof_adr] = self.joint_armature
      self.model.dof_frictionloss[dof_adr] = self.joint_frictionloss

    self.gyro_sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
    if self.gyro_sid < 0:
      raise ValueError("scene must contain sensor 'imu_ang_vel'")

  def _load_policy(self) -> None:
    self.session = ort.InferenceSession(str(self.policy_path), providers=["CPUExecutionProvider"])
    self.input_name = self.session.get_inputs()[0].name
    self._validate_policy_metadata()
    shape = self.session.get_inputs()[0].shape
    self.policy_single_obs = self.num_single_obs
    if len(shape) >= 2 and isinstance(shape[1], int):
      if shape[1] % self.frame_stack != 0:
        raise ValueError(f"policy input dim {shape[1]} is not divisible by frame_stack")
      self.policy_single_obs = shape[1] // self.frame_stack
      if self.policy_single_obs > self.num_single_obs:
        raise ValueError(
          f"policy expects {self.policy_single_obs} obs per frame, "
          f"but simulator builds {self.num_single_obs}"
        )

    self.hist_obs: deque[np.ndarray] = deque(maxlen=self.frame_stack)
    for _ in range(self.frame_stack):
      self.hist_obs.append(np.zeros((1, self.num_single_obs), dtype=np.float32))

    self.prev_action = np.zeros(self.num_actions, dtype=np.float32)
    self.current_action = np.zeros(self.num_actions, dtype=np.float32)
    self.filtered_target_q: np.ndarray | None = None
    self.policy_step_count = 0
    self.sim_step_count = 0

    print(f"[INFO] scene: {DEFAULT_SCENE_XML}")
    print(f"[INFO] policy: {self.policy_path}")
    print(
      f"[INFO] obs: single={self.num_single_obs}, stack={self.frame_stack}, "
      f"policy_input={self.policy_single_obs * self.frame_stack}"
    )
    print(
      f"[INFO] control: sim_dt={self.sim_dt:g}, decimation={self.decimation}, "
      f"policy_dt={self.policy_dt:g}, history_layout={HISTORY_LAYOUT_FRAME_MAJOR}"
    )
    print(
      f"[INFO] joint dynamics override: armature={self.joint_armature:g}, "
      f"frictionloss={self.joint_frictionloss:g}"
    )

  def _validate_policy_metadata(self) -> None:
    metadata = self.session.get_modelmeta().custom_metadata_map or {}
    obs_names = _metadata_list(metadata.get("observation_names"))
    expected_obs_names = [
      "base_ang_vel",
      "projected_gravity",
      "command",
      "phase",
      "joint_pos",
      "joint_vel",
      "actions",
    ]
    if obs_names and obs_names != expected_obs_names:
      raise ValueError(
        "policy observation_names metadata does not match simulator layout: "
        f"policy={obs_names}, sim={expected_obs_names}"
      )

    policy_joint_names = _metadata_list(metadata.get("joint_names"))
    if policy_joint_names and policy_joint_names != self.joint_names:
      raise ValueError(
        "policy joint_names metadata does not match deploy config robot.joint_names:\n"
        f"  policy={policy_joint_names}\n"
        f"  config={self.joint_names}"
      )

    history_layout = metadata.get("history_layout")
    if history_layout != HISTORY_LAYOUT_FRAME_MAJOR:
      raise ValueError(
        "policy history_layout must be "
        f"'{HISTORY_LAYOUT_FRAME_MAJOR}', got {history_layout!r}. "
        "Use a policy trained with the frame-major history pipeline and "
        "exported by the current runner; legacy term-major ONNX policies are "
        "incompatible."
      )

    print(
      "[INFO] policy metadata verified: observation_names, joint_names, "
      "and history_layout match"
    )

  def _reset(self) -> None:
    self.data.qpos[:] = 0.0
    self.data.qvel[:] = 0.0
    self.data.qpos[self.free_qpos_adr : self.free_qpos_adr + 7] = (
      0.0,
      0.0,
      self.args.base_height,
      1.0,
      0.0,
      0.0,
      0.0,
    )
    self.data.qpos[self.qpos_adrs] = self.default_q
    self.data.ctrl[:] = 0.0
    self.last_tau_cmd[:] = 0.0
    self.last_tau_applied[:] = 0.0
    mujoco.mj_forward(self.model, self.data)

  def _initialize_history(self) -> None:
    self.hist_obs.clear()
    obs = self._single_obs(np.zeros(3, dtype=np.float32))
    for _ in range(self.frame_stack):
      self.hist_obs.append(obs.copy())

  def _sensor(self, sid: int) -> np.ndarray:
    adr = int(self.model.sensor_adr[sid])
    dim = int(self.model.sensor_dim[sid])
    return self.data.sensordata[adr : adr + dim].copy()

  def _phase_obs(self, command: np.ndarray) -> np.ndarray:
    phase = (self.policy_step_count * self.policy_dt) % self.phase_period
    phase = phase / self.phase_period
    obs = np.array(
      [math.sin(2.0 * math.pi * phase), math.cos(2.0 * math.pi * phase)],
      dtype=np.float32,
    )
    if np.linalg.norm(command) < self.command_stand_threshold:
      obs[:] = 0.0
    return obs

  def _single_obs(self, command: np.ndarray) -> np.ndarray:
    root_quat = self.data.qpos[self.free_qpos_adr + 3 : self.free_qpos_adr + 7].copy()
    projected_gravity = quat_rotate_inverse(root_quat, np.array([0.0, 0.0, -1.0], dtype=np.float32))
    joint_pos = self.data.qpos[self.qpos_adrs].astype(np.float32)
    joint_vel = self.data.qvel[self.qvel_adrs].astype(np.float32)
    parts = [
      self._sensor(self.gyro_sid).astype(np.float32),
      projected_gravity.astype(np.float32),
      command.astype(np.float32),
      self._phase_obs(command),
      (joint_pos - self.default_q).astype(np.float32),
      joint_vel,
      self.prev_action.astype(np.float32),
    ]
    obs = np.concatenate(parts).reshape(1, -1)
    return np.clip(obs, -self.clip_observations, self.clip_observations).astype(np.float32)

  def _policy_input(self) -> np.ndarray:
    history = list(self.hist_obs)
    history_array = np.concatenate(history, axis=0)[:, : self.policy_single_obs]
    return history_array.reshape(1, -1).astype(np.float32)

  def _run_policy(self, command: np.ndarray) -> None:
    obs = self._single_obs(command)
    self.hist_obs.append(obs.copy())
    policy_input = self._policy_input()
    action = np.asarray(self.session.run(None, {self.input_name: policy_input})[0]).squeeze()
    action = action.astype(np.float32)
    if self.clip_actions is not None:
      action = np.clip(action, -self.clip_actions, self.clip_actions)
    if action.shape != (self.num_actions,):
      raise RuntimeError(f"policy action shape must be ({self.num_actions},), got {action.shape}")
    self.current_action = action
    self.policy_step_count += 1

  def _target_q(self) -> np.ndarray:
    raw = self.current_action * self.action_scale + self.default_q
    raw = np.clip(raw, self.joint_pos_low, self.joint_pos_high)
    if self.filtered_target_q is None or self.target_q_alpha >= 1.0:
      self.filtered_target_q = raw.copy()
    else:
      a = self.target_q_alpha
      self.filtered_target_q = a * raw + (1.0 - a) * self.filtered_target_q
    return self.filtered_target_q.astype(np.float32)

  def step(self, command: np.ndarray) -> np.ndarray:
    command = np.clip(command, -self.cmd_limit, self.cmd_limit)
    if self.sim_step_count % self.decimation == 0:
      self._run_policy(command)
    target_q = self._target_q()
    joint_pos = self.data.qpos[self.qpos_adrs]
    joint_vel = self.data.qvel[self.qvel_adrs]
    tau = self.kps * (target_q - joint_pos) - self.kds * joint_vel
    tau = np.clip(tau, -self.tau_limit, self.tau_limit)
    self.data.ctrl[self.actuator_ids] = tau
    mujoco.mj_step(self.model, self.data)
    self.last_tau_cmd = tau.astype(np.float32, copy=True)
    self.last_tau_applied = self.data.actuator_force[self.actuator_ids].astype(np.float32, copy=True)
    if self.torque_logger is not None:
      self.torque_logger.maybe_write(
        sim_time=float(self.data.time),
        sim_step=self.sim_step_count,
        policy_step=self.policy_step_count,
        command=command,
        tau_cmd=self.last_tau_cmd,
        tau_applied=self.last_tau_applied,
      )
    self.prev_action = self.current_action.copy()
    self.sim_step_count += 1
    return command

  def close(self) -> None:
    if self.torque_logger is not None:
      self.torque_logger.close()
      self.torque_logger = None


def parse_args() -> argparse.Namespace:
  latest = _latest_policy()
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--policy",
    type=Path,
    default=latest,
    help="ONNX policy path. Defaults to latest logs/rsl_rl/custom_robot_velocity/*/policy.onnx.",
  )
  parser.add_argument("--base-height", type=float, default=1.03)
  parser.add_argument("--duration", type=float, default=0.0, help="Seconds to run; 0 means until viewer closes.")
  parser.add_argument("--print-rate", type=float, default=1.0, help="Status print period in seconds.")
  parser.add_argument(
    "--render-decimation",
    type=int,
    default=DEFAULT_RENDER_DECIMATION,
    help="Call viewer.sync() every N sim steps. Higher values reduce GUI overhead.",
  )
  parser.add_argument(
    "--realtime",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="When enabled, pace wall-clock time to sim.data.time.",
  )

  parser.add_argument("--joy-backend", choices=("auto", "pygame", "linuxjs"), default="auto")
  parser.add_argument("--joy-device-index", type=int, default=0)
  parser.add_argument("--joy-device", default="/dev/input/js0")
  parser.add_argument("--joy-deadzone", type=float, default=0.1)
  parser.add_argument("--joy-max-lin-vel-x", type=float, default=0.6)
  parser.add_argument("--joy-max-lin-vel-y", type=float, default=0.6)
  parser.add_argument("--joy-max-ang-vel-z", type=float, default=1.0)
  parser.add_argument("--joy-lin-vel-x-axis", type=int, default=1)
  parser.add_argument("--joy-lin-vel-y-axis", type=int, default=0)
  parser.add_argument("--joy-ang-vel-z-axis", type=int, default=2)
  parser.add_argument("--joy-enable-button", type=int, default=6)
  parser.add_argument("--joy-mode-button", type=int, default=7)
  parser.add_argument("--enabled", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument("--walking", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument(
    "--torque-log",
    type=Path,
    default=None,
    help="Optional CSV path for joint torque logs.",
  )
  parser.add_argument(
    "--torque-log-decimation",
    type=int,
    default=1,
    help="Write one torque log row every N sim steps.",
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  sim = OnnxPolicySim(args)
  joystick = create_joystick(args)
  start_wall = time.time()
  last_print = start_wall
  render_decimation = max(1, args.render_decimation)

  try:
    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
      while viewer.is_running():
        command = sim.step(joystick.poll())
        if sim.sim_step_count % render_decimation == 0:
          viewer.sync()

        now = time.time()
        if args.print_rate > 0 and now - last_print >= args.print_rate:
          wall_elapsed = max(now - start_wall, 1e-6)
          rtf = sim.data.time / wall_elapsed
          print(
            "[INFO] "
            f"t={sim.data.time:.2f}s cmd=[{command[0]:+.2f}, {command[1]:+.2f}, {command[2]:+.2f}] "
            f"base_z={sim.data.qpos[sim.free_qpos_adr + 2]:.3f} rtf={rtf:.2f}x"
          )
          last_print = now

        if args.duration > 0 and sim.data.time >= args.duration:
          break

        if args.realtime:
          target_wall = start_wall + sim.data.time
          sleep_time = target_wall - time.time()
          if sleep_time > 0:
            time.sleep(sleep_time)
  finally:
    joystick.close()
    sim.close()


if __name__ == "__main__":
  main()
