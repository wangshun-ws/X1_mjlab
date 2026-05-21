from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  custom_robot_flat_env_cfg,
  custom_robot_rough_env_cfg,
)
from .rl_cfg import custom_robot_ppo_runner_cfg

register_mjlab_task(
  task_id="CustomRobot-Rough",
  env_cfg=custom_robot_rough_env_cfg(),
  play_env_cfg=custom_robot_rough_env_cfg(play=True),
  rl_cfg=custom_robot_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="CustomRobot-Flat",
  env_cfg=custom_robot_flat_env_cfg(),
  play_env_cfg=custom_robot_flat_env_cfg(play=True),
  rl_cfg=custom_robot_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
