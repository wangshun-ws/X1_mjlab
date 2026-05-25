"""Custom robot velocity environment configurations."""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from src.assets.robots import (
  CUSTOM_ROBOT_ACTION_SCALE,
  CUSTOM_ROBOT_CONTROL_ACTUATOR_NAMES,
  CUSTOM_ROBOT_EFFORT_LIMITS,
  get_custom_robot_cfg,
)
from src.tasks.velocity.delayed_actions import DelayedJointPositionActionCfg
import src.tasks.velocity.mdp as project_mdp
from src.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

BASE_BODY_NAME = "pelvis_link"
FOOT_SITE_NAMES = ("left_foot", "right_foot")
FOOT_GEOM_NAMES = tuple(
  f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
)


def custom_robot_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create custom robot rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 48

  cfg.scene.entities = {"robot": get_custom_robot_cfg()}

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = BASE_BODY_NAME

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern=BASE_BODY_NAME, entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern=BASE_BODY_NAME, entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, DelayedJointPositionActionCfg)
  joint_pos_action.scale = CUSTOM_ROBOT_ACTION_SCALE
  joint_pos_action.delay_min_lag = 0
  joint_pos_action.delay_max_lag = 0
  joint_pos_action.delay_per_env = True
  joint_pos_action.delay_hold_prob = 0.0
  joint_pos_action.delay_update_period = (
    math.ceil(cfg.episode_length_s / cfg.sim.mujoco.timestep) + 1
  )
  joint_pos_action.delay_per_env_phase = False

  cfg.viewer.body_name = BASE_BODY_NAME
  cfg.viewer.distance = 3.0
  cfg.viewer.elevation = -5.0

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.15

  cfg.observations["critic"].terms["foot_height"].params[
    "asset_cfg"
  ].site_names = FOOT_SITE_NAMES
  cfg.observations["critic"].terms["actuator_torque"] = ObservationTermCfg(
    func=project_mdp.normalized_actuator_torque,
    params={
      "asset_cfg": SceneEntityCfg(
        "robot",
        actuator_names=list(CUSTOM_ROBOT_CONTROL_ACTUATOR_NAMES),
        preserve_order=True,
      ),
      "effort_limits": CUSTOM_ROBOT_EFFORT_LIMITS,
    },
  )

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = FOOT_GEOM_NAMES
  cfg.events["base_com"].params["asset_cfg"].body_names = ("waist_yaw_link",)
  cfg.events["base_com"].mode = "reset"

  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.5,
    r".*ankle_pitch.*": 0.15,
    r".*ankle_roll.*": 0.1,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.25,
    r".*hip_yaw.*": 0.25,
    r".*knee.*": 0.5,
    r".*ankle_pitch.*": 0.25,
    r".*ankle_roll.*": 0.1,
  }

  cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = (
    BASE_BODY_NAME,
  )
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("waist_yaw_link",)
  cfg.rewards["foot_clearance"].params[
    "asset_cfg"
  ].site_names = FOOT_SITE_NAMES
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = FOOT_SITE_NAMES
  cfg.rewards["foot_touchdown_velocity"].params[
    "asset_cfg"
  ].site_names = FOOT_SITE_NAMES
  cfg.rewards["actuator_torque_l2"] = RewardTermCfg(
    func=project_mdp.normalized_torque_l2,
    weight=-0.1,
    params={
      "asset_cfg": SceneEntityCfg(
        "robot",
        actuator_names=list(CUSTOM_ROBOT_CONTROL_ACTUATOR_NAMES),
        preserve_order=True,
      ),
      "effort_limits": CUSTOM_ROBOT_EFFORT_LIMITS,
      "command_name": "twist",
      "command_threshold": 0.1,
    },
  )
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=project_mdp.self_collision_cost,
    weight=-5.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

  cfg.curriculum = {}

  if play:
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    joint_pos_action.delay_min_lag = 0
    joint_pos_action.delay_max_lag = 0
    cfg.events.pop("push_robot", None)
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def custom_robot_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create custom robot flat terrain velocity configuration."""
  cfg = custom_robot_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-0.8, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.8, 0.8)
    twist_cmd.ranges.ang_vel_z = (-1.0, 1.0)

  return cfg
