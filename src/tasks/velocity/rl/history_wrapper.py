"""RSL-RL wrapper for frame-major observation history flattening."""

from __future__ import annotations

from typing import Any

import torch
from tensordict import TensorDict

from mjlab.rl import RslRlVecEnvWrapper

HISTORY_LAYOUT_FRAME_MAJOR = "frame_major_oldest_to_newest"


class FrameMajorHistoryRslRlVecEnvWrapper(RslRlVecEnvWrapper):
  """Flatten group-level history as complete frames, oldest to newest.

  The mjlab observation manager emits `(B, H, D)` when a group keeps history
  with `flatten_history_dim=False`. RSL-RL's MLP expects `(B, H * D)`, so this
  wrapper performs that final flatten after all terms in the frame were joined.

  For this project we intentionally zero-fill newly reset history windows so
  the first policy input after reset matches deployment semantics:
  `[0, 0, ..., 0, current_frame]` instead of backfilling all slots with the
  current frame.
  """

  def get_observations(self) -> TensorDict:
    obs_dict = self.unwrapped.observation_manager.compute()
    return self._as_tensordict(obs_dict)

  def reset(self) -> tuple[TensorDict, dict]:
    obs_dict, extras = self.env.reset()
    env_ids = torch.arange(self.num_envs, device=self.unwrapped.device, dtype=torch.long)
    self._zero_fill_reset_history(obs_dict, env_ids)
    return self._as_tensordict(obs_dict), extras

  def step(
    self, actions: torch.Tensor
  ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
    if self.clip_actions is not None:
      actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
    obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
    term_or_trunc = terminated | truncated
    assert isinstance(rew, torch.Tensor)
    assert isinstance(term_or_trunc, torch.Tensor)
    dones = term_or_trunc.to(dtype=torch.long)
    reset_env_ids = term_or_trunc.nonzero(as_tuple=False).squeeze(-1)
    if reset_env_ids.numel() > 0:
      self._zero_fill_reset_history(obs_dict, reset_env_ids)
    if not self.cfg.is_finite_horizon:
      extras["time_outs"] = truncated
    return self._as_tensordict(obs_dict), rew, dones, extras

  def _zero_fill_reset_history(
    self, obs_dict: dict[str, Any], env_ids: torch.Tensor
  ) -> None:
    if env_ids.ndim == 0:
      env_ids = env_ids.unsqueeze(0)
    if env_ids.numel() == 0:
      return

    obs_manager = self.unwrapped.observation_manager
    for group_name, obs in obs_dict.items():
      if not isinstance(obs, torch.Tensor):
        continue

      group_cfg = self.cfg.observations.get(group_name)
      if group_cfg is None:
        continue
      if group_cfg.history_length is None or group_cfg.history_length <= 0:
        continue
      if group_cfg.flatten_history_dim:
        raise ValueError(
          f"Observation group '{group_name}' must keep history unflattened before "
          "FrameMajorHistoryRslRlVecEnvWrapper runs."
        )
      if obs.ndim != 3:
        raise ValueError(
          f"Observation group '{group_name}' must have shape (B, H, D), got {tuple(obs.shape)}."
        )

      latest_frame = obs[env_ids, -1, :].clone()
      obs[env_ids] = 0.0
      obs[env_ids, -1, :] = latest_frame

      term_names = obs_manager._group_obs_term_names[group_name]
      term_dims = obs_manager._group_obs_term_dim[group_name]
      history_buffers = obs_manager._group_obs_term_history_buffer[group_name]
      feature_offset = 0
      for term_name, term_dim in zip(term_names, term_dims, strict=False):
        if len(term_dim) < 2:
          feature_dim = 1
          feature_shape: tuple[int, ...] = ()
        else:
          feature_shape = tuple(term_dim[1:])
          feature_dim = int(torch.tensor(feature_shape, device="cpu").prod().item())

        circular_buffer = history_buffers.get(term_name)
        if circular_buffer is not None and circular_buffer._buffer is not None:
          latest_term = latest_frame[:, feature_offset : feature_offset + feature_dim]
          if feature_shape:
            latest_term = latest_term.reshape(latest_term.shape[0], *feature_shape)
          circular_buffer._buffer[:, env_ids] = 0.0
          circular_buffer._buffer[circular_buffer._pointer, env_ids] = latest_term.to(
            circular_buffer._buffer.dtype
          )
          circular_buffer._num_pushes[env_ids] = 1

        feature_offset += feature_dim

  def _as_tensordict(self, obs_dict: dict[str, Any]) -> TensorDict:
    flattened = {
      group_name: self._flatten_group_history(group_name, obs)
      for group_name, obs in obs_dict.items()
    }
    return TensorDict(flattened, batch_size=[self.num_envs])

  def _flatten_group_history(self, group_name: str, obs: Any) -> Any:
    if not isinstance(obs, torch.Tensor):
      return obs

    group_cfg = self.cfg.observations.get(group_name)
    if group_cfg is None:
      return obs
    if group_cfg.history_length is None or group_cfg.history_length <= 0:
      return obs
    if group_cfg.flatten_history_dim:
      raise ValueError(
        f"Observation group '{group_name}' must keep history unflattened before "
        "FrameMajorHistoryRslRlVecEnvWrapper runs."
      )
    if obs.ndim != 3:
      raise ValueError(
        f"Observation group '{group_name}' must have shape (B, H, D), got {tuple(obs.shape)}."
      )
    return obs.reshape(obs.shape[0], -1)
