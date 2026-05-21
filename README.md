# Custom Locomotion Mjlab

This is a MuJoCo, mujoco-warp, mjlab and rsl_rl workspace for custom robot locomotion. The repository has been trimmed down to a single velocity-tracking training path for a self-built robot.

The current built-in robot is X1. The task IDs intentionally stay generic:

- `CustomRobot-Flat`: flat-ground velocity tracking
- `CustomRobot-Rough`: rough-terrain velocity tracking

## Current Scope

This project is set up for:

1. Training locomotion policies with the integrated X1 model.
2. Replacing the `custom_robot` MJCF and parameters with your own robot.
3. Exporting `policy.onnx` from training and wiring it into your own deployment stack.

Vendor robot assets, vendor communication code, vendor deployment projects, mimic/tracking tasks and example motion datasets have been removed.

## Installation

See [doc/setup_en.md](doc/setup_en.md) for the full environment setup.

From the repository root, install the project in editable mode:

```bash
pip install -e .
```

Verify task registration:

```bash
python scripts/list_envs.py
```

Expected tasks:

```text
CustomRobot-Flat
CustomRobot-Rough
```

## Layout

```text
src/assets/robots/custom_robot/
  custom_robot_constants.py        # init state, actuators, collisions, action scale
  xmls/custom_robot.xml            # current X1 MJCF
  xmls/assets/                     # X1 mesh files

src/tasks/velocity/config/custom_robot/
  __init__.py                      # registers CustomRobot-Flat / CustomRobot-Rough
  env_cfgs.py                      # env, observations, contacts, rewards, terrain
  rl_cfg.py                        # PPO and network config

scripts/
  list_envs.py                     # list tasks
  train.py                         # train policies
  play.py                          # play or debug policies
  visualize_terrain.py             # terrain visualization helper
```

## Quick Check

Check that the MJCF, registry and environment path work:

```bash
python scripts/list_envs.py
python scripts/play.py CustomRobot-Flat --agent zero --num-envs 1
```

On headless machines, use the Viser viewer:

```bash
python scripts/play.py CustomRobot-Flat --agent zero --num-envs 1 --viewer viser
```

## Training

Start with the flat task:

```bash
python scripts/train.py CustomRobot-Flat --env.scene.num-envs=4096
```

After flat-ground walking is stable, move to rough terrain:

```bash
python scripts/train.py CustomRobot-Rough --env.scene.num-envs=4096
```

Multi-GPU training:

```bash
python scripts/train.py CustomRobot-Flat \
  --gpu-ids 0 1 \
  --env.scene.num-envs=4096
```

Training logs and checkpoints are written to:

```text
logs/rsl_rl/custom_robot_velocity/<date_time>/
```

Common outputs:

```text
model_<iteration>.pt
params/env.yaml
params/agent.yaml
policy.onnx
```

## Play

Run a trained checkpoint:

```bash
python scripts/play.py CustomRobot-Flat \
  --checkpoint-file=logs/rsl_rl/custom_robot_velocity/<run>/model_<iter>.pt
```

Record a play video:

```bash
python scripts/play.py CustomRobot-Flat \
  --checkpoint-file=logs/rsl_rl/custom_robot_velocity/<run>/model_<iter>.pt \
  --video
```

Use dummy agents to debug the model and viewer without a checkpoint:

```bash
python scripts/play.py CustomRobot-Flat --agent zero --num-envs 1
python scripts/play.py CustomRobot-Flat --agent random --num-envs 1
```

### Joystick Play

`play.py` can write a physical gamepad command into the policy `twist` observation, using the same mapping as the X1 simulation `joy_control.py`:

- left stick vertical: `lin_vel_x`
- left stick horizontal: `lin_vel_y`
- right stick X: `ang_vel_z`
- LB: control enable
- RB: stand/walk mode

The startup state is disabled and stand mode. Press LB once to enable control, then RB once to switch to walk mode.

```bash
python scripts/play.py CustomRobot-Flat \
  --checkpoint-file=logs/rsl_rl/custom_robot_velocity/<run>/model_<iter>.pt \
  --num-envs 1 \
  --joystick True
```

The default backend is `auto`: try `pygame` first, then fall back to Linux `/dev/input/js0`. If your right stick X axis is axis 3 instead of axis 2:

```bash
python scripts/play.py CustomRobot-Flat \
  --checkpoint-file=logs/rsl_rl/custom_robot_velocity/<run>/model_<iter>.pt \
  --num-envs 1 \
  --joystick True \
  --joy-ang-vel-z-axis 3
```

## Adding Your Robot

See [doc/custom_robot_zh.md](doc/custom_robot_zh.md) for more detailed notes.

Minimum edit scope:

1. Replace `src/assets/robots/custom_robot/xmls/custom_robot.xml` and `xmls/assets/`.
2. Update `custom_robot_constants.py`:
   - base height and default joint positions
   - actuator stiffness, damping, effort limit and armature
   - foot collision geom regex and friction parameters
   - action scale
3. Update `env_cfgs.py`:
   - base body name
   - foot site names
   - foot contact geom names
   - contact sensor subtree
   - gait offset
   - pose reward joint groups
4. Run `python scripts/list_envs.py` and confirm the tasks still register.
5. Bring up `CustomRobot-Flat` first, then switch to `CustomRobot-Rough`.

## Rewards

The current `CustomRobot` task uses velocity locomotion rewards:

- linear and angular velocity tracking
- body orientation
- default pose
- joint acceleration and action-rate penalties
- foot gait, clearance and slip
- soft landing
- stand still
- self-collision penalty

The X1 pose reward was adapted from a biped lower-body locomotion template and mainly constrains hip, knee and ankle joints. When replacing the robot, first check that those joint-name regexes match your MJCF.

## Deployment

The repository now includes a reference ROS2 deployment node:

```bash
python deploy/deploy_easy_mjlab.py \
  --config deploy/config/custom_robot_easy_mjlab.yaml
```

Before running it, edit the policy paths in the YAML:

```yaml
policy:
  walk_path: "logs/rsl_rl/custom_robot_velocity/<run>/policy.onnx"
  stand_path: "logs/rsl_rl/custom_robot_velocity/<run>/policy.onnx"
```

The node subscribes to `/joint_states`, `/imu/data`, `/cmd_vel`, `/hwswitch`, and `/control_mode`, and publishes `/targetPos`, `/targetVel`, `/targetKp`, `/targetKd`, and `/targetTorque`. The default config matches the current flat-task 47D training observation, not the older X1 AMP 51D gait observation.

History uses ten complete observation frames flattened frame-major as `[obs_t-9, ..., obs_t]`. Policies trained with the previous term-major history layout are incompatible and must be retrained with the current pipeline; re-exporting an old checkpoint does not change its input semantics.

Real robot deployment still requires you to verify:

- robot communication
- state estimation and frame conversion
- policy observation construction and normalization
- action-to-motor target mapping
- safety limits, emergency stop and protection logic

Bring up deployment with fixed seeds and low-speed commands in simulation before moving the same observation and action-scale path to hardware.

## Common Commands

```bash
# List tasks
python scripts/list_envs.py

# Flat training
python scripts/train.py CustomRobot-Flat --env.scene.num-envs=4096

# Rough-terrain training
python scripts/train.py CustomRobot-Rough --env.scene.num-envs=4096

# Zero-action debug
python scripts/play.py CustomRobot-Flat --agent zero --num-envs 1

# Checkpoint play
python scripts/play.py CustomRobot-Flat \
  --checkpoint-file=logs/rsl_rl/custom_robot_velocity/<run>/model_<iter>.pt
```

## Dependencies

- [mjlab](https://github.com/mujocolab/mjlab.git)
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl.git)
- [mujoco_warp](https://github.com/google-deepmind/mujoco_warp.git)
- [mujoco](https://github.com/google-deepmind/mujoco.git)
