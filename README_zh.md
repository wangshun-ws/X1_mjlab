# Custom Locomotion Mjlab

这是一个基于 MuJoCo、mujoco-warp、mjlab 和 rsl_rl 的自制机器人 locomotion 强化学习项目。当前仓库已经移除原来的厂商机器人示例，只保留一条面向自定义机器人的速度跟踪训练主线。

当前内置机器人是 X1，任务名称仍使用通用的 `CustomRobot`：

- `CustomRobot-Flat`：平地速度跟踪
- `CustomRobot-Rough`：崎岖地形速度跟踪

## 当前状态

项目现在适合做三件事：

1. 基于已接入的 X1 模型训练 locomotion policy。
2. 替换 `custom_robot` 目录下的 MJCF 和参数，接入自己的机器人。
3. 从训练结果导出 `policy.onnx`，再接到你自己的部署程序里。

项目现在不再包含厂商机器人资产、厂商通信代码、厂商部署工程、mimic/tracking 任务和示例 motion 数据。

## 安装

完整环境安装见 [doc/setup_zh.md](doc/setup_zh.md)。

进入项目根目录后，建议使用 editable 方式安装：

```bash
pip install -e .
```

安装完成后确认任务注册正常：

```bash
python scripts/list_envs.py
```

正常输出应只包含：

```text
CustomRobot-Flat
CustomRobot-Rough
```

## 目录结构

```text
src/assets/robots/custom_robot/
  custom_robot_constants.py        # 机器人初始姿态、actuator、碰撞、action scale
  xmls/custom_robot.xml            # 当前 X1 MJCF
  xmls/assets/                     # X1 mesh 文件

src/tasks/velocity/config/custom_robot/
  __init__.py                      # 注册 CustomRobot-Flat / CustomRobot-Rough
  env_cfgs.py                      # 环境、观测、接触、奖励、地形配置
  rl_cfg.py                        # PPO 和网络配置

scripts/
  list_envs.py                     # 查看任务
  train.py                         # 训练
  play.py                          # 回放或调试 policy
  visualize_terrain.py             # 地形可视化辅助脚本
```

## 快速验证

先确认 MJCF、任务注册和环境创建没有问题：

```bash
python scripts/list_envs.py
python scripts/play.py CustomRobot-Flat --agent zero --num-envs 1
```

如果没有显示器，可以使用 Viser viewer：

```bash
python scripts/play.py CustomRobot-Flat --agent zero --num-envs 1 --viewer viser
```

## 训练

建议先从平地任务开始：

```bash
python scripts/train.py CustomRobot-Flat --env.scene.num-envs=4096
```

平地能站稳并跟踪速度后，再训练崎岖地形：

```bash
python scripts/train.py CustomRobot-Rough --env.scene.num-envs=4096
```

多 GPU 训练：

```bash
python scripts/train.py CustomRobot-Flat \
  --gpu-ids 0 1 \
  --env.scene.num-envs=4096
```

训练日志和模型默认保存到：

```text
logs/rsl_rl/custom_robot_velocity/<date_time>/
```

常用输出包括：

```text
model_<iteration>.pt
params/env.yaml
params/agent.yaml
policy.onnx
```

## 回放

使用训练好的 checkpoint：

```bash
python scripts/play.py CustomRobot-Flat --checkpoint-file=logs/rsl_rl/custom_robot_velocity/<run>/model_<iter>.pt
```

录制回放视频：

```bash
python scripts/play.py CustomRobot-Flat \
  --checkpoint-file=logs/rsl_rl/custom_robot_velocity/<run>/model_<iter>.pt \
  --video
```

不用 checkpoint 时可以用零动作或随机动作快速检查模型和 viewer：

```bash
python scripts/play.py CustomRobot-Flat --agent zero --num-envs 1
python scripts/play.py CustomRobot-Flat --agent random --num-envs 1
```

### 手柄控制回放

`play.py` 可以把物理手柄输入写入策略观测里的 `twist` 速度命令，按键映射与 X1 仿真版 `joy_control.py` 保持一致：

- 左摇杆前后：`lin_vel_x`
- 左摇杆左右：`lin_vel_y`
- 右摇杆 X 轴：`ang_vel_z`
- LB：控制使能开关
- RB：站立/行走模式切换

启动后默认是关闭、站立状态，所以需要先按一次 LB 使能，再按一次 RB 切到行走模式。

```bash
python scripts/play.py CustomRobot-Flat \
  --checkpoint-file=logs/rsl_rl/custom_robot_velocity/<run>/model_<iter>.pt \
  --num-envs 1 \
  --joystick True
```

默认手柄后端为 `auto`：优先使用 `pygame`，如果当前环境没有安装 `pygame`，会退到 Linux joystick 设备 `/dev/input/js0`。如果你的手柄右摇杆 X 轴不是 axis 2，可以改成常见的 axis 3：

```bash
python scripts/play.py CustomRobot-Flat \
  --checkpoint-file=logs/rsl_rl/custom_robot_velocity/<run>/model_<iter>.pt \
  --num-envs 1 \
  --joystick True \
  --joy-ang-vel-z-axis 3
```

## 接入自己的机器人

详细说明见 [doc/custom_robot_zh.md](doc/custom_robot_zh.md)。

最小修改范围：

1. 替换 `src/assets/robots/custom_robot/xmls/custom_robot.xml` 和 `xmls/assets/`。
2. 在 `custom_robot_constants.py` 中更新：
   - 初始 base 高度和默认关节角
   - actuator stiffness、damping、effort limit、armature
   - foot collision geom 的正则表达式和摩擦参数
   - action scale
3. 在 `env_cfgs.py` 中更新：
   - base body 名称
   - foot site 名称
   - foot contact geom 名称
   - contact sensor subtree
   - gait offset
   - pose reward 的关节分组
4. 运行 `python scripts/list_envs.py` 确认任务还能注册。
5. 先用 `CustomRobot-Flat` 调通，再切换到 `CustomRobot-Rough`。

## 奖励配置

当前 `CustomRobot` 使用速度跟踪 locomotion 奖励，包含：

- 线速度和角速度跟踪
- body 姿态约束
- 默认姿态约束
- 关节加速度和动作变化惩罚
- 足端 gait、clearance、slip
- soft landing
- stand still
- self collision 惩罚

X1 的 pose reward 按双足下肢 locomotion 模板做了迁移，主要约束 hip、knee、ankle。后续换机器人时，优先检查这些关节正则表达式是否能匹配你的 MJCF 关节名。

## 部署

仓库现在提供一个参考 ROS2 部署节点：

```bash
python deploy/deploy_easy_mjlab.py \
  --config deploy/config/custom_robot_easy_mjlab.yaml
```

使用前先把 YAML 里的策略路径改成实际训练结果：

```yaml
policy:
  walk_path: "logs/rsl_rl/custom_robot_velocity/<run>/policy.onnx"
  stand_path: "logs/rsl_rl/custom_robot_velocity/<run>/policy.onnx"
```

这个节点订阅 `/joint_states`、`/imu/data`、`/cmd_vel`、`/hwswitch`、`/control_mode`，发布 `/targetPos`、`/targetVel`、`/targetKp`、`/targetKd`、`/targetTorque`。默认配置按当前 flat 任务训练观测写成 47 维，不是外部 X1 AMP 的 51 维 gait 观测。

历史输入固定为 10 个完整单帧观测，按 frame-major 的 `[obs_t-9, ..., obs_t]` 展平。旧版 term-major 历史拼接训练出的 policy 不兼容，需要按当前流程重新训练；只重新导出旧 checkpoint 不会改变它的输入语义。

真实机器人部署仍然需要你自己确认：

- 机器人通信接口
- 状态估计和坐标系转换
- policy 输入归一化和观测拼接
- action 到电机目标的映射
- 安全限幅、急停和保护逻辑

建议先在仿真中固定随机种子和低速命令调通，再把同一套观测和 action scale 迁移到真机。

## 常见命令

```bash
# 查看任务
python scripts/list_envs.py

# 平地训练
python scripts/train.py CustomRobot-Flat --env.scene.num-envs=4096

# 崎岖地形训练
python scripts/train.py CustomRobot-Rough --env.scene.num-envs=4096

# 零动作调试
python scripts/play.py CustomRobot-Flat --agent zero --num-envs 1

# checkpoint 回放
python scripts/play.py CustomRobot-Flat \
  --checkpoint-file=logs/rsl_rl/custom_robot_velocity/<run>/model_<iter>.pt
```

## 依赖项目

- [mjlab](https://github.com/mujocolab/mjlab.git)
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl.git)
- [mujoco_warp](https://github.com/google-deepmind/mujoco_warp.git)
- [mujoco](https://github.com/google-deepmind/mujoco.git)
