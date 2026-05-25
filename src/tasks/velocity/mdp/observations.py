from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def foot_height(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # (num_envs, num_sites)


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  return current_air_time


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float()


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.force is not None
  forces_flat = sensor_data.force.flatten(start_dim=1)  # [B, N*3]
  return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))


def normalized_actuator_torque(
  env: ManagerBasedRlEnv,
  effort_limits: tuple[float, ...],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  torque = asset.data.actuator_force[:, asset_cfg.actuator_ids]
  limits = torch.as_tensor(effort_limits, device=torque.device, dtype=torque.dtype)
  if limits.numel() != torque.shape[1]:
    raise ValueError(
      f"effort_limits length {limits.numel()} does not match actuator force "
      f"dimension {torque.shape[1]}."
    )
  limits = torch.clamp(torch.abs(limits), min=1e-6)
  return torch.clamp(torque / limits.unsqueeze(0), min=-1.0, max=1.0)


def _action_term_prev_action(
  env: ManagerBasedRlEnv,
  action_term_name: str | None,
) -> tuple[torch.Tensor, object | None]:
  action_manager = env.action_manager
  prev_action = action_manager.prev_action
  if action_term_name is None:
    return prev_action, None

  action_offset = 0
  for name, dim in zip(action_manager.active_terms, action_manager.action_term_dim):
    if name == action_term_name:
      action_term = action_manager.get_term(action_term_name)
      return prev_action[:, action_offset : action_offset + dim], action_term
    action_offset += dim
  raise ValueError(f"Action term '{action_term_name}' not found.")


def _action_scale_tensor(
  action_scale: float | Sequence[float],
  *,
  dim: int,
  device: torch.device,
  dtype: torch.dtype,
) -> torch.Tensor:
  scale = torch.as_tensor(action_scale, device=device, dtype=dtype).flatten()
  if scale.numel() == 1:
    return scale.expand(dim)
  if scale.numel() != dim:
    raise ValueError(
      f"action_scale length {scale.numel()} does not match target error "
      f"dimension {dim}."
    )
  return scale


def target_joint_pos_error(
  env: ManagerBasedRlEnv,
  action_scale: float | Sequence[float] = 0.25,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  action_term_name: str | None = "joint_pos",
) -> torch.Tensor:
  """Return normalized previous target joint error for joint-position policies."""
  asset: Entity = env.scene[asset_cfg.name]
  joint_pos_rel = (
    asset.data.joint_pos[:, asset_cfg.joint_ids]
    - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  )

  prev_action, action_term = _action_term_prev_action(env, action_term_name)
  joint_names = getattr(asset_cfg, "joint_names", None)
  if (
    action_term is not None
    and getattr(asset_cfg, "preserve_order", False)
    and isinstance(joint_names, (list, tuple))
  ):
    target_names = action_term.target_names
    name_to_local_idx = {name: idx for idx, name in enumerate(target_names)}
    missing = [name for name in joint_names if name not in name_to_local_idx]
    if missing:
      raise ValueError(
        f"Joint names {missing} are not controlled by action term "
        f"'{action_term_name}'."
      )
    action_ids = torch.tensor(
      [name_to_local_idx[name] for name in joint_names],
      device=prev_action.device,
      dtype=torch.long,
    )
    prev_action = prev_action[:, action_ids]

  if prev_action.shape[1] != joint_pos_rel.shape[1]:
    raise ValueError(
      f"previous action dimension {prev_action.shape[1]} does not match "
      f"joint dimension {joint_pos_rel.shape[1]}."
    )

  scale = _action_scale_tensor(
    action_scale,
    dim=joint_pos_rel.shape[1],
    device=joint_pos_rel.device,
    dtype=joint_pos_rel.dtype,
  )
  scale_abs = torch.clamp(torch.abs(scale), min=1e-6)
  target_error = prev_action * scale.unsqueeze(0) - joint_pos_rel
  return target_error / scale_abs.unsqueeze(0)


def phase(env: ManagerBasedRlEnv, period: float, command_name: str) -> torch.Tensor:
    global_phase = (env.episode_length_buf * env.step_dt) % period / period
    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    stand_mask = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < 0.1
    phase = torch.where(stand_mask.unsqueeze(1), torch.zeros_like(phase), phase)
    return phase
