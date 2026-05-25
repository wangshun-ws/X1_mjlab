"""Local action terms for velocity tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from mjlab.utils.buffers import DelayBuffer

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class DelayedJointPositionActionCfg(JointPositionActionCfg):
  """Joint-position action with an optional action-space delay buffer."""

  delay_min_lag: int = 0
  delay_max_lag: int = 0
  delay_per_env: bool = True
  delay_hold_prob: float = 0.0
  delay_update_period: int = 0
  delay_per_env_phase: bool = True

  def build(self, env: ManagerBasedRlEnv) -> DelayedJointPositionAction:
    return DelayedJointPositionAction(self, env)


class DelayedJointPositionAction(JointPositionAction):
  """Apply joint-position targets through a control-step delay buffer."""

  cfg: DelayedJointPositionActionCfg

  def __init__(self, cfg: DelayedJointPositionActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)
    self._delay_buffer: DelayBuffer | None = None
    if cfg.delay_max_lag > 0:
      self._delay_buffer = DelayBuffer(
        min_lag=cfg.delay_min_lag,
        max_lag=cfg.delay_max_lag,
        batch_size=self.num_envs,
        device=self.device,
        per_env=cfg.delay_per_env,
        hold_prob=cfg.delay_hold_prob,
        update_period=cfg.delay_update_period,
        per_env_phase=cfg.delay_per_env_phase,
      )

  def apply_actions(self) -> None:
    target = self._processed_actions
    if self._delay_buffer is not None:
      self._delay_buffer.append(target)
      target = self._delay_buffer.compute()

    encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
    self._entity.set_joint_position_target(
      target - encoder_bias, joint_ids=self._target_ids
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids=env_ids)
    if self._delay_buffer is not None:
      batch_ids = None if isinstance(env_ids, slice) else env_ids
      self._delay_buffer.reset(batch_ids=batch_ids)
