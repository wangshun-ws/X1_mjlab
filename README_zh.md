# X1_Mjlab

这是一个基于 MuJoCo、mujoco-warp、mjlab 和 rsl_rl 的X1机器人 locomotion 强化学习项目。
当前内置机器人是 X1，当前公开任务是：

- `X1_flat`：平地速度跟踪



## 安装


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
X1_flat
```

## 目录结构

```text
src/assets/robots/custom_robot/
  custom_robot_constants.py        # 机器人初始姿态、actuator、碰撞、action scale
  xmls/custom_robot.xml            # 当前 X1 MJCF
  xmls/assets/                     # X1 mesh 文件

src/tasks/velocity/config/custom_robot/
  __init__.py                      # 注册 X1_flat
  env_cfgs.py                      # 环境、观测、接触、奖励、地形配置
  rl_cfg.py                        # PPO 和网络配置

scripts/
  list_envs.py                     # 查看任务
  train.py                         # 训练
  play.py                          # 回放或调试 policy
  visualize_terrain.py             # 地形可视化辅助脚本
```

## 训练

训练入口：

```bash
python scripts/train.py X1_flat --env.scene.num-envs=4096
```

多 GPU 训练：

```bash
python scripts/train.py X1_flat \
  --gpu-ids 0 1 \
  --env.scene.num-envs=4096
```

训练日志和模型默认保存到：

```text
logs/rsl_rl/x1_flat_velocity/<date_time>/
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
python scripts/play.py X1_flat --checkpoint-file=logs/rsl_rl/x1_flat_velocity/<run>/model_<iter>.pt
```

录制回放视频：

```bash
python scripts/play.py X1_flat \
  --checkpoint-file=logs/rsl_rl/x1_flat_velocity/<run>/model_<iter>.pt \
  --video
```

不用 checkpoint 时可以用零动作或随机动作快速检查模型和 viewer：

```bash
python scripts/play.py X1_flat --agent zero --num-envs 1
python scripts/play.py X1_flat --agent random --num-envs 1
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
python scripts/play.py X1_flat \
  --checkpoint-file=logs/rsl_rl/x1_flat_velocity/<run>/model_<iter>.pt \
  --num-envs 1 \
  --joystick True
```

默认手柄后端为 `auto`：优先使用 `pygame`，如果当前环境没有安装 `pygame`，会退到 Linux joystick 设备 `/dev/input/js0`。如果你的手柄右摇杆 X 轴不是 axis 2，可以改成常见的 axis 3：

```bash
python scripts/play.py X1_flat \
  --checkpoint-file=logs/rsl_rl/x1_flat_velocity/<run>/model_<iter>.pt \
  --num-envs 1 \
  --joystick True \
  --joy-ang-vel-z-axis 3
```


## 依赖项目

- [mjlab](https://github.com/mujocolab/mjlab.git)
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl.git)
- [mujoco_warp](https://github.com/google-deepmind/mujoco_warp.git)
- [mujoco](https://github.com/google-deepmind/mujoco.git)
