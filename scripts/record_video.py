"""Record a headless rollout video of a trained checkpoint.

mjlab's `play --video` records only while its viewer steps the environment, and
on a headless box that viewer (viser) never exits. This follows play's loading
path and steps the policy for a fixed number of steps instead, the same way the
viewer does (get_observations -> policy -> step).

    python scripts/record_video.py Mjlab-Velocity-Flat-MicroDuck \
        --checkpoint logs/rsl_rl/velocity/<run>/model_499.pt --out <dir>/rollout.mp4
"""

import argparse
import shutil
from dataclasses import asdict
from pathlib import Path

import torch

import mjlab  # noqa: F401  # loads the mjlab.tasks entry points, and with them the ROCm patch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("task")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="output .mp4")
    ap.add_argument("--steps", type=int, default=500, help="policy steps at 50 Hz")
    ap.add_argument("--num-envs", type=int, default=1)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    # mjlab's default camera sits 3 m out for full-size robots; the duck is 25 cm tall.
    ap.add_argument("--distance", type=float, default=0.8, help="m from the tracked trunk")
    ap.add_argument("--elevation", type=float, default=-15.0, help="degrees")
    ap.add_argument("--azimuth", type=float, default=135.0, help="degrees")
    args = ap.parse_args()

    configure_torch_backends()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env_cfg = load_env_cfg(args.task, play=True)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.viewer.width, env_cfg.viewer.height = args.width, args.height
    env_cfg.viewer.distance, env_cfg.viewer.elevation = args.distance, args.elevation
    env_cfg.viewer.azimuth = args.azimuth
    agent_cfg = load_rl_cfg(args.task)

    # Raw recordings go next to the checkpoint, as play's do; only the final
    # .mp4 is copied to --out.
    folder = args.checkpoint.parent / "videos" / "record"
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
    env = VideoRecorder(env, video_folder=folder, step_trigger=lambda step: step == 0,
                        video_length=args.steps, disable_logger=True)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(str(args.checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)

    with torch.no_grad():
        for _ in range(args.steps + 1):
            env.step(policy(env.get_observations()))
    env.close()

    videos = sorted(folder.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
    if not videos:
        raise SystemExit(f"no video written to {folder}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(videos[-1], args.out)
    print(f"Written {args.out}")


if __name__ == "__main__":
    main()
