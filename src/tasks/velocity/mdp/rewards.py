from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.lab_api.string import (
  resolve_matching_names_values,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def track_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for tracking the commanded base linear velocity.

  The commanded z velocity is assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_lin_vel_b
  xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  z_error = torch.square(actual[:, 2])
  lin_vel_error = xy_error + (2 * z_error)
  return torch.exp(-lin_vel_error / std**2)


def track_angular_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward heading error for heading-controlled envs, angular velocity for others.

  The commanded xy angular velocities are assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_ang_vel_b
  z_error = torch.square(command[:, 2] - actual[:, 2])
  xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
  ang_vel_error = z_error + (0.05 * xy_error)
  return torch.exp(-ang_vel_error / std**2)


def body_orientation_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward flat base orientation (robot being upright).

  If asset_cfg has body_ids specified, computes the projected gravity
  for that specific body. Otherwise, uses the root link projected gravity.
  """
  asset: Entity = env.scene[asset_cfg.name]

  # If body_ids are specified, compute projected gravity for that body.
  if asset_cfg.body_ids:
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # [B, N, 4]
    body_quat_w = body_quat_w.squeeze(1)  # [B, 4]
    gravity_w = asset.data.gravity_vec_w  # [3]
    projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)  # [B, 3]
    xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)
  else:
    # Use root link projected gravity.
    xy_squared = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
  return xy_squared


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Penalize self-collisions.

  When the sensor provides force history (from ``history_length > 0``),
  counts substeps where any contact force exceeds *force_threshold*.
  Falls back to the instantaneous ``found`` count otherwise.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
    return hit.sum(dim=-1).float()  # [B]
  assert data.found is not None
  return data.found.squeeze(-1)


def body_angular_velocity_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize excessive body angular velocities."""
  asset: Entity = env.scene[asset_cfg.name]
  ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :]
  ang_vel = ang_vel.squeeze(1)
  ang_vel_xy = ang_vel[:, :2]  # Don't penalize z-angular velocity.
  return torch.sum(torch.square(ang_vel_xy), dim=1)


def angular_momentum_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Penalize whole-body angular momentum to encourage natural arm swing."""
  angmom_sensor: BuiltinSensor = env.scene[sensor_name]
  angmom = angmom_sensor.data
  angmom_magnitude_sq = torch.sum(torch.square(angmom), dim=-1)
  angmom_magnitude = torch.sqrt(angmom_magnitude_sq)
  env.extras["log"]["Metrics/angular_momentum_mean"] = torch.mean(angmom_magnitude)
  return angmom_magnitude_sq


class normalized_torque_l2:
  """Penalize squared actuator torque normalized by effort limits."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset_cfg = cfg.params["asset_cfg"]
    asset: Entity = env.scene[asset_cfg.name]

    self._asset_name = asset_cfg.name
    self._actuator_ids = asset_cfg.actuator_ids
    actuator_force = asset.data.actuator_force[:, self._actuator_ids]
    effort_limits = torch.as_tensor(
      cfg.params["effort_limits"], device=env.device, dtype=actuator_force.dtype
    )
    if effort_limits.numel() != actuator_force.shape[1]:
      raise ValueError(
        f"effort_limits length {effort_limits.numel()} does not match actuator "
        f"dimension {actuator_force.shape[1]}."
      )
    self._effort_limits = torch.clamp(torch.abs(effort_limits), min=1e-6)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    effort_limits: tuple[float, ...],
    command_name: str | None = None,
    command_threshold: float = 0.1,
  ) -> torch.Tensor:
    del asset_cfg, effort_limits  # Unused after initialization.

    asset: Entity = env.scene[self._asset_name]
    torque = asset.data.actuator_force[:, self._actuator_ids]
    normalized_torque = torque / self._effort_limits.unsqueeze(0)
    env.extras["log"]["Metrics/normalized_torque_mean"] = torch.mean(
      torch.abs(normalized_torque)
    )
    cost = torch.sum(torch.square(normalized_torque), dim=1)

    if command_name is not None:
      command = env.command_manager.get_command(command_name)
      assert command is not None, f"Command '{command_name}' not found."
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost *= active
    return cost


class subset_action_rate_l2:
  """Penalize the rate of change of a subset of raw policy actions."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    action_term_name = cfg.params["action_term_name"]
    joint_names = cfg.params["joint_names"]
    action_manager = env.action_manager
    action_term = action_manager.get_term(action_term_name)

    target_names = action_term.target_names
    name_to_local_idx = {name: idx for idx, name in enumerate(target_names)}
    missing = [name for name in joint_names if name not in name_to_local_idx]
    if missing:
      raise ValueError(
        f"Joint names {missing} are not controlled by action term "
        f"'{action_term_name}'."
      )

    action_offset = 0
    for name, dim in zip(action_manager.active_terms, action_manager.action_term_dim):
      if name == action_term_name:
        break
      action_offset += dim
    else:
      raise ValueError(f"Action term '{action_term_name}' not found.")

    subset_ids = [action_offset + name_to_local_idx[name] for name in joint_names]
    self._action_ids = torch.tensor(
      subset_ids, device=env.device, dtype=torch.long
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    action_term_name: str,
    joint_names: list[str],
  ) -> torch.Tensor:
    del action_term_name, joint_names  # Unused.
    delta_action = (
      env.action_manager.action[:, self._action_ids]
      - env.action_manager.prev_action[:, self._action_ids]
    )
    return torch.sum(torch.square(delta_action), dim=1)


class feet_air_time:
  """Penalize swing air time outside a target interval at touchdown."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    sensor: ContactSensor = env.scene[cfg.params["sensor_name"]]
    current_air_time = sensor.data.current_air_time
    assert current_air_time is not None
    self._last_air_time = torch.zeros_like(current_air_time)

    min_air_time = cfg.params["min_air_time"]
    max_air_time = cfg.params["max_air_time"]
    if min_air_time > max_air_time:
      raise ValueError(
        f"min_air_time ({min_air_time}) must be <= max_air_time ({max_air_time})."
      )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    min_air_time: float,
    max_air_time: float,
    command_threshold: float = 0.1,
  ) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    current_air_time = sensor.data.current_air_time
    assert current_air_time is not None
    first_contact = sensor.compute_first_contact(dt=env.step_dt)
    if first_contact.shape != current_air_time.shape:
      raise ValueError(
        f"first_contact shape {tuple(first_contact.shape)} does not match air "
        f"time shape {tuple(current_air_time.shape)}."
      )

    reset_mask = env.episode_length_buf == 0
    self._last_air_time[reset_mask] = 0.0
    self._last_air_time = torch.where(
      current_air_time > 0.0, current_air_time, self._last_air_time
    )
    landed_air_time = torch.maximum(current_air_time, self._last_air_time)
    first_contact_float = first_contact.float()

    low_error = torch.clamp(min_air_time - landed_air_time, min=0.0)
    high_error = torch.clamp(landed_air_time - max_air_time, min=0.0)
    cost = torch.sum(
      (torch.square(low_error) + torch.square(high_error)) * first_contact_float,
      dim=1,
    )

    num_touchdowns = torch.sum(first_contact_float)
    mean_air_time = torch.sum(landed_air_time * first_contact_float) / torch.clamp(
      num_touchdowns, min=1
    )
    env.extras["log"]["Metrics/air_time_mean"] = mean_air_time

    self._last_air_time = torch.where(
      first_contact, torch.zeros_like(self._last_air_time), self._last_air_time
    )

    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    active = (total_command > command_threshold).float()
    return cost * active


class feet_contact_balance:
  """Penalize long-window contact duty imbalance between left and right foot."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    sensor: ContactSensor = env.scene[cfg.params["sensor_name"]]
    found = sensor.data.found
    assert found is not None
    if found.shape[1] != 2:
      raise ValueError(
        f"feet_contact_balance expects exactly 2 feet, got found shape "
        f"{tuple(found.shape)}."
      )
    self._duty = torch.zeros_like(found, dtype=torch.float32)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    command_threshold: float = 0.1,
    window_s: float = 0.8,
    allowed_imbalance: float = 0.2,
  ) -> torch.Tensor:
    if window_s <= 0.0:
      raise ValueError(f"window_s must be positive, got {window_s}.")

    sensor: ContactSensor = env.scene[sensor_name]
    found = sensor.data.found
    assert found is not None
    if found.shape != self._duty.shape or found.shape[1] != 2:
      raise ValueError(
        f"feet_contact_balance expects found shape {tuple(self._duty.shape)}, "
        f"got {tuple(found.shape)}."
      )

    reset_mask = env.episode_length_buf == 0
    self._duty[reset_mask] = 0.0

    in_contact = (found > 0).float()
    alpha = 1.0 - torch.exp(
      torch.as_tensor(-env.step_dt / window_s, device=env.device)
    )
    self._duty = (1.0 - alpha) * self._duty + alpha * in_contact

    diff = torch.abs(self._duty[:, 0] - self._duty[:, 1])
    env.extras["log"]["Metrics/feet_contact_balance_diff"] = torch.mean(diff)
    env.extras["log"]["Metrics/left_contact_duty"] = torch.mean(self._duty[:, 0])
    env.extras["log"]["Metrics/right_contact_duty"] = torch.mean(self._duty[:, 1])

    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    command_speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    active = (command_speed > command_threshold).float()

    excess = torch.clamp(diff - allowed_imbalance, min=0.0)
    return torch.square(excess) * active


class feet_double_support:
  """Penalize excessive double-support duty during commanded locomotion."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    sensor: ContactSensor = env.scene[cfg.params["sensor_name"]]
    found = sensor.data.found
    assert found is not None
    if found.shape[1] != 2:
      raise ValueError(
        f"feet_double_support expects exactly 2 feet, got found shape "
        f"{tuple(found.shape)}."
      )
    self._double_support_duty = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.float32
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    command_threshold: float = 0.1,
    window_s: float = 0.8,
    allowed_double_support: float = 0.45,
  ) -> torch.Tensor:
    if window_s <= 0.0:
      raise ValueError(f"window_s must be positive, got {window_s}.")

    sensor: ContactSensor = env.scene[sensor_name]
    found = sensor.data.found
    assert found is not None
    if found.shape[1] != 2:
      raise ValueError(
        f"feet_double_support expects exactly 2 feet, got found shape "
        f"{tuple(found.shape)}."
      )

    reset_mask = env.episode_length_buf == 0
    self._double_support_duty[reset_mask] = 0.0

    in_contact = found > 0
    double_support = torch.logical_and(in_contact[:, 0], in_contact[:, 1]).float()
    alpha = 1.0 - torch.exp(
      torch.as_tensor(-env.step_dt / window_s, device=env.device)
    )
    self._double_support_duty = (
      (1.0 - alpha) * self._double_support_duty + alpha * double_support
    )

    env.extras["log"]["Metrics/double_support_duty"] = torch.mean(
      self._double_support_duty
    )

    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    command_speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    active = (command_speed > command_threshold).float()

    excess = torch.clamp(
      self._double_support_duty - allowed_double_support, min=0.0
    )
    return torch.square(excess) * active


def feet_clearance(
  env: ManagerBasedRlEnv,
  target_height: float,
  command_name: str | None = None,
  command_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize deviation from target clearance height, weighted by foot velocity."""
  asset: Entity = env.scene[asset_cfg.name]
  foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  delta = torch.abs(foot_z - target_height)  # [B, N]
  cost = torch.sum(delta * vel_norm, dim=1)  # [B]
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


def feet_gait(
        env: ManagerBasedRlEnv,
        period: float,
        offset: list[float],
        threshold: float,
        command_threshold: float,
        command_name: str,
        sensor_name: str,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    is_contact = sensor.data.current_contact_time > 0
    global_phase = ((env.episode_length_buf * env.step_dt) / period).unsqueeze(1)
    offsets = torch.as_tensor(offset, device=env.device, dtype=global_phase.dtype).view(1, -1)
    leg_phase = (global_phase + offsets) % 1.0
    is_stance = (leg_phase < threshold)
    reward = (is_stance == is_contact).float().mean(dim=1)
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
            scale = (total_command > command_threshold).float()
            reward *= scale
    return reward


class feet_swing_height:
  """Penalize deviation from target swing height, evaluated at landing."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.sensor_name = cfg.params["sensor_name"]
    self.site_names = cfg.params["asset_cfg"].site_names
    self.peak_heights = torch.zeros(
      (env.num_envs, len(self.site_names)), device=env.device, dtype=torch.float32
    )
    self.step_dt = env.step_dt

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    target_height: float,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    foot_heights = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
    in_air = contact_sensor.data.found == 0
    self.peak_heights = torch.where(
      in_air,
      torch.maximum(self.peak_heights, foot_heights),
      self.peak_heights,
    )
    first_contact = contact_sensor.compute_first_contact(dt=self.step_dt)
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    active = (total_command > command_threshold).float()
    error = self.peak_heights / target_height - 1.0
    cost = torch.sum(torch.square(error) * first_contact.float(), dim=1) * active
    num_landings = torch.sum(first_contact.float())
    peak_heights_at_landing = self.peak_heights * first_contact.float()
    mean_peak_height = torch.sum(peak_heights_at_landing) / torch.clamp(
      num_landings, min=1
    )
    env.extras["log"]["Metrics/peak_height_mean"] = mean_peak_height
    self.peak_heights = torch.where(
      first_contact,
      torch.zeros_like(self.peak_heights),
      self.peak_heights,
    )
    return cost


def feet_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize foot sliding (xy velocity while in contact)."""
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  linear_norm = torch.norm(command[:, :2], dim=1)
  angular_norm = torch.abs(command[:, 2])
  total_command = linear_norm + angular_norm
  active = (total_command > command_threshold).float()
  assert contact_sensor.data.found is not None
  in_contact = (contact_sensor.data.found > 0).float()  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  vel_xy_norm_sq = torch.square(vel_xy_norm)  # [B, N]
  cost = torch.sum(vel_xy_norm_sq * in_contact, dim=1) * active
  num_in_contact = torch.sum(in_contact)
  mean_slip_vel = torch.sum(vel_xy_norm * in_contact) / torch.clamp(
    num_in_contact, min=1
  )
  env.extras["log"]["Metrics/slip_velocity_mean"] = mean_slip_vel
  return cost


def feet_touchdown_velocity(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  base_threshold: float,
  threshold_gain: float,
  max_threshold: float | None = None,
  command_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize excessive downward foot velocity on first ground contact."""
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."

  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)
  foot_vel_z = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, 2]
  if first_contact.shape != foot_vel_z.shape:
    raise ValueError(
      f"first_contact shape {tuple(first_contact.shape)} does not match foot "
      f"velocity shape {tuple(foot_vel_z.shape)}."
    )

  command_speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
  velocity_threshold = base_threshold + threshold_gain * command_speed
  if max_threshold is not None:
    velocity_threshold = torch.clamp(velocity_threshold, max=max_threshold)

  downward_speed = torch.clamp(-foot_vel_z, min=0.0)
  first_contact_float = first_contact.float()
  touchdown_speed = downward_speed * first_contact_float
  num_touchdowns = torch.sum(first_contact_float)
  mean_touchdown_speed = torch.sum(touchdown_speed) / torch.clamp(
    num_touchdowns, min=1
  )
  env.extras["log"]["Metrics/touchdown_velocity_mean"] = mean_touchdown_speed
  env.extras["log"]["Metrics/touchdown_velocity_threshold_mean"] = torch.mean(
    velocity_threshold
  )

  excess = torch.clamp(
    downward_speed - velocity_threshold.unsqueeze(1), min=0.0
  )
  active = (command_speed > command_threshold).float()
  return torch.sum(torch.square(excess) * first_contact_float, dim=1) * active


def soft_landing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Penalize high impact forces at landing to encourage soft footfalls."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = contact_sensor.data
  assert sensor_data.force is not None
  forces = sensor_data.force  # [B, N, 3]
  force_magnitude = torch.norm(forces, dim=-1)  # [B, N]
  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)  # [B, N]
  landing_impact = force_magnitude * first_contact.float()  # [B, N]
  cost = torch.sum(landing_impact, dim=1)  # [B]
  num_landings = torch.sum(first_contact.float())
  mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
  env.extras["log"]["Metrics/landing_force_mean"] = mean_landing_force
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


class variable_posture:
  """Penalize deviation from default pose with speed-dependent tolerance.

  Uses per-joint standard deviations to control how much each joint can deviate
  from default pose. Smaller std = stricter (less deviation allowed), larger
  std = more forgiving. The reward is: exp(-mean(error² / std²))

  Three speed regimes (based on linear + angular command velocity):
    - std_standing (speed < walking_threshold): Tight tolerance for holding pose.
    - std_walking (walking_threshold <= speed < running_threshold): Moderate.
    - std_running (speed >= running_threshold): Loose tolerance for large motion.

  Tune std values per joint based on how much motion that joint needs at each
  speed. Map joint name patterns to std values, e.g. {".*knee.*": 0.35}.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    self.default_joint_pos = default_joint_pos

    _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)

    _, _, std_standing = resolve_matching_names_values(
      data=cfg.params["std_standing"],
      list_of_strings=joint_names,
    )
    self.std_standing = torch.tensor(
      std_standing, device=env.device, dtype=torch.float32
    )

    _, _, std_walking = resolve_matching_names_values(
      data=cfg.params["std_walking"],
      list_of_strings=joint_names,
    )
    self.std_walking = torch.tensor(std_walking, device=env.device, dtype=torch.float32)

    _, _, std_running = resolve_matching_names_values(
      data=cfg.params["std_running"],
      list_of_strings=joint_names,
    )
    self.std_running = torch.tensor(std_running, device=env.device, dtype=torch.float32)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std_standing,
    std_walking,
    std_running,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    walking_threshold: float = 0.5,
    running_threshold: float = 1.5,
  ) -> torch.Tensor:
    del std_standing, std_walking, std_running  # Unused.

    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None

    linear_speed = torch.norm(command[:, :2], dim=1)
    angular_speed = torch.abs(command[:, 2])
    total_speed = linear_speed + angular_speed

    standing_mask = (total_speed < walking_threshold).float()
    walking_mask = (
      (total_speed >= walking_threshold) & (total_speed < running_threshold)
    ).float()
    running_mask = (total_speed >= running_threshold).float()

    std = (
      self.std_standing * standing_mask.unsqueeze(1)
      + self.std_walking * walking_mask.unsqueeze(1)
      + self.std_running * running_mask.unsqueeze(1)
    )

    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
    error_squared = torch.square(current_joint_pos - desired_joint_pos)

    return torch.exp(-torch.mean(error_squared / (std**2), dim=1))


def stand_still(
        env: ManagerBasedRlEnv,
        command_name: str,
        command_threshold: float = 0.1,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    diff_angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.square(diff_angle), dim=1)
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
            scale = (total_command <= command_threshold).float()
            reward *= scale
    return reward
