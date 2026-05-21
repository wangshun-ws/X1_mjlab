"""X1 robot constants for velocity locomotion."""

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.actuator.delayed_actuator import DelayedActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg
from src import SRC_PATH

##
# MJCF and assets.
##

CUSTOM_ROBOT_XML: Path = (
  SRC_PATH / "assets" / "robots" / "custom_robot" / "xmls" / "custom_robot.xml"
)
assert CUSTOM_ROBOT_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  assets_dir = CUSTOM_ROBOT_XML.parent / "assets"
  if assets_dir.exists():
    update_assets(assets, assets_dir, meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(CUSTOM_ROBOT_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config.
##

CUSTOM_ROBOT_ACTUATOR_HIP_YAW = DelayedActuatorCfg(
  base_cfg=BuiltinPositionActuatorCfg(
    target_names_expr=(".*_hip_yaw_joint",),
    stiffness=100.0,
    damping=1.0,
    effort_limit=80.0,
    armature=0.05,
  ),
  delay_min_lag=0,
  delay_max_lag=2,
  delay_update_period=2001,
  delay_hold_prob=0.0,
)
CUSTOM_ROBOT_ACTUATOR_HIP_ROLL = DelayedActuatorCfg(
  base_cfg=BuiltinPositionActuatorCfg(
    target_names_expr=(".*_hip_roll_joint",),
    stiffness=150.0,
    damping=4.0,
    effort_limit=120.0,
    armature=0.05,
  ),
  delay_min_lag=0,
  delay_max_lag=2,
  delay_update_period=2001,
  delay_hold_prob=0.0,
)
CUSTOM_ROBOT_ACTUATOR_HIP_PITCH = DelayedActuatorCfg(
  base_cfg=BuiltinPositionActuatorCfg(
    target_names_expr=(".*_hip_pitch_joint",),
    stiffness=150.0,
    damping=4.0,
    effort_limit=212.0,
    armature=0.05,
  ),
  delay_min_lag=0,
  delay_max_lag=2,
  delay_update_period=2001,
  delay_hold_prob=0.0,
)
CUSTOM_ROBOT_ACTUATOR_KNEE = DelayedActuatorCfg(
  base_cfg=BuiltinPositionActuatorCfg(
    target_names_expr=(".*_knee_joint",),
    stiffness=180.0,
    damping=4.0,
    effort_limit=180.0,
    armature=0.05,
  ),
  delay_min_lag=0,
  delay_max_lag=2,
  delay_update_period=2001,
  delay_hold_prob=0.0,
)
CUSTOM_ROBOT_ACTUATOR_ANKLE_PITCH = DelayedActuatorCfg(
  base_cfg=BuiltinPositionActuatorCfg(
    target_names_expr=(".*_ankle_pitch_joint",),
    stiffness=75.0,
    damping=2.0,
    effort_limit=80,
    armature=0.05,
  ),
  delay_min_lag=0,
  delay_max_lag=2,
  delay_update_period=2001,
  delay_hold_prob=0.0,
)
CUSTOM_ROBOT_ACTUATOR_ANKLE_ROLL = DelayedActuatorCfg(
  base_cfg=BuiltinPositionActuatorCfg(
    target_names_expr=(".*_ankle_roll_joint",),
    stiffness=30.0,
    damping=2.0,
    effort_limit=20.0,
    armature=0.05,
  ),
  delay_min_lag=0,
  delay_max_lag=2,
  delay_update_period=2001,
  delay_hold_prob=0.0,
)


##
# Initial state.
##

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 1.03),
  joint_pos={
    ".*_hip_yaw_joint": 0.0,
    ".*_hip_roll_joint": 0.0,
    ".*_hip_pitch_joint": 0.2,
    ".*_knee_joint": -0.4,
    ".*_ankle_pitch_joint": 0.2,
    ".*_ankle_roll_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)


##
# Collision config.
##

_foot_regex = "^(left|right)_foot[0-9]+_collision$"

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={_foot_regex: 3, ".*_collision": 1},
  priority={_foot_regex: 1},
  friction={_foot_regex: (0.6,)},
  solimp={_foot_regex: (0.9, 0.95, 0.023)},
  contype=1,
  conaffinity=0,
)


##
# Final config.
##

CUSTOM_ROBOT_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    CUSTOM_ROBOT_ACTUATOR_HIP_YAW,
    CUSTOM_ROBOT_ACTUATOR_HIP_ROLL,
    CUSTOM_ROBOT_ACTUATOR_HIP_PITCH,
    CUSTOM_ROBOT_ACTUATOR_KNEE,
    CUSTOM_ROBOT_ACTUATOR_ANKLE_PITCH,
    CUSTOM_ROBOT_ACTUATOR_ANKLE_ROLL,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_custom_robot_cfg() -> EntityCfg:
  """Get a fresh custom robot configuration instance."""
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=CUSTOM_ROBOT_ARTICULATION,
  )


CUSTOM_ROBOT_ACTION_SCALE: dict[str, float] = {}
for actuator in CUSTOM_ROBOT_ARTICULATION.actuators:
  for name_expr in actuator.target_names_expr:
    CUSTOM_ROBOT_ACTION_SCALE[name_expr] = 0.25


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_custom_robot_cfg())

  viewer.launch(robot.spec.compile())
