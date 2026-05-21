"""Script to play RL agent with RSL-RL."""

import errno
import os
import struct
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro

import mjlab
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from src.tasks.velocity.rl import FrameMajorHistoryRslRlVecEnvWrapper

PROJECT_TASK_PREFIXES = ("CustomRobot-",)


@dataclass(frozen=True)
class PlayConfig:
  agent: Literal["zero", "random", "trained"] = "trained"
  checkpoint_file: str | None = None
  num_envs: int | None = None
  device: str | None = None
  video: bool = False
  video_length: int = 200
  video_height: int | None = None
  video_width: int | None = None
  camera: int | str | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"
  no_terminations: bool = False
  """Disable all termination conditions."""
  joystick: bool = False
  """Use a physical joystick/gamepad to override the twist velocity command."""
  joy_backend: Literal["auto", "pygame", "linuxjs"] = "auto"
  """Joystick backend. auto tries pygame first, then Linux /dev/input/js*."""
  joy_device_index: int = 0
  """Pygame joystick index."""
  joy_device: str = "/dev/input/js0"
  """Linux joystick device used by the linuxjs backend."""
  joy_deadzone: float = 0.1
  joy_max_lin_vel_x: float = 0.6
  joy_max_lin_vel_y: float = 0.6
  joy_max_ang_vel_z: float = 1.0
  joy_lin_vel_x_axis: int = 1
  joy_lin_vel_y_axis: int = 0
  joy_ang_vel_z_axis: int = 2
  joy_enable_button: int = 6
  """LB in the X1 pygame joystick script."""
  joy_mode_button: int = 7
  """RB in the X1 pygame joystick script."""
  joy_start_enabled: bool = False
  joy_start_walking: bool = False


class JoystickUnavailableError(RuntimeError):
  """Raised when the requested joystick backend cannot be used."""


class JoystickVelocitySource:
  """X1-style joystick mapping for velocity commands."""

  def __init__(self, cfg: PlayConfig, name: str):
    self.cfg = cfg
    self.name = name
    self.enabled = cfg.joy_start_enabled
    self.walking = cfg.joy_start_walking
    self._last_enable_button = False
    self._last_mode_button = False
    self._command = (0.0, 0.0, 0.0)

    print(f"[INFO] Joystick backend: {self.name}")
    print(
      "[INFO] Joystick mapping: left stick = vx/vy, right stick X = yaw, "
      "LB = enable, RB = stand/walk"
    )
    print(
      f"[INFO] Joystick state: {'enabled' if self.enabled else 'disabled'}, "
      f"{'walk' if self.walking else 'stand'}"
    )

  @property
  def command(self) -> tuple[float, float, float]:
    return self._command

  def poll(self) -> tuple[float, float, float]:
    self._poll_device()
    self._update_toggles()

    vx = -self._axis_with_deadzone(self.cfg.joy_lin_vel_x_axis)
    vy = -self._axis_with_deadzone(self.cfg.joy_lin_vel_y_axis)
    yaw = self._axis_with_deadzone(self.cfg.joy_ang_vel_z_axis)

    if not self.enabled or not self.walking:
      self._command = (0.0, 0.0, 0.0)
    else:
      self._command = (
        vx * self.cfg.joy_max_lin_vel_x,
        vy * self.cfg.joy_max_lin_vel_y,
        yaw * self.cfg.joy_max_ang_vel_z,
      )
    return self._command

  def close(self) -> None:
    pass

  def _poll_device(self) -> None:
    raise NotImplementedError

  def _axis(self, axis: int) -> float:
    raise NotImplementedError

  def _button(self, button: int) -> bool:
    raise NotImplementedError

  def _axis_with_deadzone(self, axis: int) -> float:
    value = self._axis(axis)
    if abs(value) <= self.cfg.joy_deadzone:
      return 0.0
    return value

  def _update_toggles(self) -> None:
    enable_button = self._button(self.cfg.joy_enable_button)
    if enable_button and not self._last_enable_button:
      self.enabled = not self.enabled
      print(f"[INFO] Joystick control: {'enabled' if self.enabled else 'disabled'}")
    self._last_enable_button = enable_button

    mode_button = self._button(self.cfg.joy_mode_button)
    if mode_button and not self._last_mode_button:
      self.walking = not self.walking
      print(f"[INFO] Joystick mode: {'walk' if self.walking else 'stand'}")
    self._last_mode_button = mode_button


class PygameJoystickVelocitySource(JoystickVelocitySource):
  def __init__(self, cfg: PlayConfig):
    try:
      import pygame
    except ModuleNotFoundError as exc:
      raise JoystickUnavailableError(
        "pygame is not installed. Install it or use --joy-backend linuxjs."
      ) from exc

    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
      os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    self._pygame = pygame
    pygame.init()
    pygame.joystick.init()

    joystick_count = pygame.joystick.get_count()
    if joystick_count <= cfg.joy_device_index:
      pygame.quit()
      raise JoystickUnavailableError(
        f"pygame found {joystick_count} joystick(s), "
        f"but index {cfg.joy_device_index} was requested."
      )

    self._joystick = pygame.joystick.Joystick(cfg.joy_device_index)
    self._joystick.init()
    super().__init__(cfg, f"pygame:{self._joystick.get_name()}")

  def close(self) -> None:
    self._joystick.quit()
    self._pygame.quit()

  def _poll_device(self) -> None:
    self._pygame.event.pump()

  def _axis(self, axis: int) -> float:
    if axis < 0 or axis >= self._joystick.get_numaxes():
      return 0.0
    return float(self._joystick.get_axis(axis))

  def _button(self, button: int) -> bool:
    if button < 0 or button >= self._joystick.get_numbuttons():
      return False
    return bool(self._joystick.get_button(button))


class LinuxJoystickVelocitySource(JoystickVelocitySource):
  _EVENT = struct.Struct("IhBB")
  _JS_EVENT_BUTTON = 0x01
  _JS_EVENT_AXIS = 0x02
  _JS_EVENT_INIT = 0x80

  def __init__(self, cfg: PlayConfig):
    self._axes: dict[int, float] = {}
    self._buttons: dict[int, bool] = {}
    try:
      self._fd = os.open(cfg.joy_device, os.O_RDONLY | os.O_NONBLOCK)
    except OSError as exc:
      raise JoystickUnavailableError(
        f"Cannot open joystick device {cfg.joy_device}: {exc.strerror}"
      ) from exc

    super().__init__(cfg, f"linuxjs:{cfg.joy_device}")
    self._poll_device()

  def close(self) -> None:
    os.close(self._fd)

  def _poll_device(self) -> None:
    while True:
      try:
        data = os.read(self._fd, self._EVENT.size)
      except BlockingIOError:
        return
      except OSError as exc:
        if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
          return
        raise

      if len(data) != self._EVENT.size:
        return

      _time_ms, value, event_type, number = self._EVENT.unpack(data)
      event_type &= ~self._JS_EVENT_INIT

      if event_type == self._JS_EVENT_AXIS:
        scale = 32767.0 if value >= 0 else 32768.0
        self._axes[number] = max(-1.0, min(1.0, value / scale))
      elif event_type == self._JS_EVENT_BUTTON:
        self._buttons[number] = bool(value)

  def _axis(self, axis: int) -> float:
    return self._axes.get(axis, 0.0)

  def _button(self, button: int) -> bool:
    return self._buttons.get(button, False)


def create_joystick_velocity_source(cfg: PlayConfig) -> JoystickVelocitySource:
  errors: list[str] = []

  if cfg.joy_backend in ("auto", "pygame"):
    try:
      return PygameJoystickVelocitySource(cfg)
    except JoystickUnavailableError as exc:
      errors.append(f"pygame: {exc}")
      if cfg.joy_backend == "pygame":
        raise

  if cfg.joy_backend in ("auto", "linuxjs"):
    try:
      return LinuxJoystickVelocitySource(cfg)
    except JoystickUnavailableError as exc:
      errors.append(f"linuxjs: {exc}")
      if cfg.joy_backend == "linuxjs":
        raise

  raise JoystickUnavailableError(
    "No joystick backend is available:\n  " + "\n  ".join(errors)
  )


def install_joystick_command_override(
  env: RslRlVecEnvWrapper,
  source: JoystickVelocitySource,
  command_name: str = "twist",
) -> None:
  """Override a velocity command term with joystick values every env step."""
  term = env.unwrapped.command_manager.get_term(command_name)
  if not hasattr(term, "vel_command_b"):
    raise TypeError(f"Command term '{command_name}' does not expose vel_command_b.")

  def write_command() -> None:
    vx, vy, yaw = source.poll()
    term.vel_command_b[:, 0] = vx
    term.vel_command_b[:, 1] = vy
    term.vel_command_b[:, 2] = yaw
    if hasattr(term, "is_standing_env"):
      term.is_standing_env[:] = False
    if hasattr(term, "is_heading_env"):
      term.is_heading_env[:] = False

  original_compute = term.compute
  original_reset = term.reset

  def compute_with_joystick(dt: float) -> None:
    original_compute(dt)
    write_command()

  def reset_with_joystick(env_ids: torch.Tensor | slice | None):
    extras = original_reset(env_ids)
    write_command()
    return extras

  term.compute = compute_with_joystick
  term.reset = reset_with_joystick
  write_command()


def run_play(task_id: str, cfg: PlayConfig):
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  DUMMY_MODE = cfg.agent in {"zero", "random"}
  TRAINED_MODE = not DUMMY_MODE

  # Disable terminations if requested.
  if cfg.no_terminations:
    env_cfg.terminations = {}
    print("[INFO]: Terminations disabled")

  log_dir: Path | None = None
  resume_path: Path | None = None
  if TRAINED_MODE:
    if cfg.checkpoint_file is not None:
      resume_path = Path(cfg.checkpoint_file)
      if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
      print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    else:
      raise ValueError("`checkpoint-file` is required when agent='trained'.")
    log_dir = resume_path.parent

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
  if cfg.video and DUMMY_MODE:
    print(
      "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
    )
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  if TRAINED_MODE and cfg.video:
    print("[INFO] Recording videos during play")
    assert log_dir is not None  # log_dir is set in TRAINED_MODE block
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos" / "play",
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  env = FrameMajorHistoryRslRlVecEnvWrapper(
    env, clip_actions=agent_cfg.clip_actions
  )
  if DUMMY_MODE:
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape
    if cfg.agent == "zero":

      class PolicyZero:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return torch.zeros(action_shape, device=env.unwrapped.device)

      policy = PolicyZero()
    else:

      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

      policy = PolicyRandom()
  else:
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
      str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)

  # Handle "auto" viewer selection.
  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
    del has_display
  else:
    resolved_viewer = cfg.viewer

  joystick_source: JoystickVelocitySource | None = None
  try:
    if cfg.joystick:
      joystick_source = create_joystick_velocity_source(cfg)
      install_joystick_command_override(env, joystick_source)

    if resolved_viewer == "native":
      NativeMujocoViewer(env, policy).run()
    elif resolved_viewer == "viser":
      ViserPlayViewer(env, policy).run()
    else:
      raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")
  finally:
    try:
      if joystick_source is not None:
        joystick_source.close()
    finally:
      env.close()


def main():
  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import src.tasks

  all_tasks = [
    task_id for task_id in list_tasks() if task_id.startswith(PROJECT_TASK_PREFIXES)
  ]
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  # Parse the rest of the arguments + allow overriding env_cfg and agent_cfg.
  agent_cfg = load_rl_cfg(chosen_task)

  args = tyro.cli(
    PlayConfig,
    args=remaining_args,
    default=PlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args, agent_cfg

  run_play(chosen_task, args)


if __name__ == "__main__":
  main()
