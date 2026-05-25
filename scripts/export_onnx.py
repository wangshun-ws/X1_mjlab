"""Export an ONNX policy from a saved RSL-RL checkpoint."""

import sys
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

import tyro

import mjlab
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from src.tasks.velocity.rl import (
  FrameMajorHistoryRslRlVecEnvWrapper,
  HISTORY_LAYOUT_FRAME_MAJOR,
)

PUBLIC_TASK_IDS = ("X1_flat",)


@dataclass(frozen=True)
class ExportOnnxConfig:
  checkpoint_file: str
  output_path: str | None = None
  device: str = "cpu"
  num_envs: int = 1
  verbose: bool = False
  overwrite: bool = False


def _default_output_path(checkpoint_path: Path) -> Path:
  if checkpoint_path.stem.startswith("model_"):
    suffix = checkpoint_path.stem.removeprefix("model_")
    return checkpoint_path.with_name(f"policy_{suffix}.onnx")
  return checkpoint_path.with_suffix(".onnx")


def _resolve_output_path(
  checkpoint_path: Path, output_path_arg: str | None
) -> Path:
  if output_path_arg is None:
    return _default_output_path(checkpoint_path)

  output_path = Path(output_path_arg).expanduser()
  if not output_path.is_absolute():
    output_path = checkpoint_path.parent / output_path
  return output_path


def run_export(task_id: str, cfg: ExportOnnxConfig) -> Path:
  configure_torch_backends()

  checkpoint_path = Path(cfg.checkpoint_file).expanduser()
  if not checkpoint_path.exists():
    raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
  if checkpoint_path.suffix not in {".pt", ".pth"}:
    raise ValueError(
      f"Checkpoint must be a PyTorch file (.pt/.pth), got: {checkpoint_path.name}"
    )

  output_path = _resolve_output_path(checkpoint_path, cfg.output_path)
  if output_path.exists() and not cfg.overwrite:
    raise FileExistsError(
      f"Output file already exists: {output_path}. Use --overwrite to replace it."
    )
  output_path.parent.mkdir(parents=True, exist_ok=True)

  env_cfg = load_env_cfg(task_id, play=True)
  env_cfg.scene.num_envs = cfg.num_envs
  agent_cfg = load_rl_cfg(task_id)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
  wrapped_env = FrameMajorHistoryRslRlVecEnvWrapper(
    env, clip_actions=agent_cfg.clip_actions
  )

  try:
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(
      wrapped_env, asdict(agent_cfg), str(output_path.parent), device=cfg.device
    )
    runner.load(
      str(checkpoint_path),
      load_cfg={"actor": True},
      strict=True,
      map_location=cfg.device,
    )
    runner.export_policy_to_onnx(
      str(output_path.parent), filename=output_path.name, verbose=cfg.verbose
    )

    metadata = get_base_metadata(wrapped_env.unwrapped, checkpoint_path.stem)
    metadata["history_layout"] = HISTORY_LAYOUT_FRAME_MAJOR
    attach_metadata_to_onnx(str(output_path), metadata)
  finally:
    wrapped_env.close()

  return output_path


def main():
  # Import tasks to populate the registry.
  import src.tasks

  registered_tasks = set(list_tasks())
  all_tasks = [task_id for task_id in PUBLIC_TASK_IDS if task_id in registered_tasks]
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  args = tyro.cli(
    ExportOnnxConfig,
    args=remaining_args,
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args

  output_path = run_export(chosen_task, args)
  print(f"[INFO] Exported ONNX: {output_path}")


if __name__ == "__main__":
  main()
