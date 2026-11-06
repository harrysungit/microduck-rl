"""Time rsl_rl's PPO update in a task's real runner, and profile one update.

Builds the task's environment and runner as ``train`` does, runs a few
learning iterations with ``runner.alg.update`` timed (device synchronised on
both sides), and records the last update with torch.profiler: total device
busy time against wall time, and the operators that take the device time.
Kineto needs rocprofiler registration, so the script turns it back on, and
with it torch's exit handler: the exit code is always 0.

    python scripts/profile_ppo.py --num-envs 4096 --njmax 128 --iterations 4
"""

import argparse
import dataclasses
import os
import time

import torch
import warp as wp  # noqa: F401  # after torch, so the ROCm libraries bundled with torch load first

import mjlab  # noqa: F401  # loads the mjlab.tasks entry points, and with them the ROCm patch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", default="Mjlab-Velocity-Flat-MicroDuck")
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--njmax", type=int)
    ap.add_argument("--iterations", type=int, default=4)
    ap.add_argument("--rows", type=int, default=15)
    args = ap.parse_args()
    os.environ.pop("ROCPROFILER_REGISTER_ENABLED", None)  # read when HIP starts, which is below

    env_cfg = load_env_cfg(args.task)
    env_cfg.scene.num_envs = args.num_envs
    if args.njmax:
        env_cfg.sim.njmax = args.njmax
    agent_cfg = load_rl_cfg(args.task)
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0"), clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
    runner = runner_cls(env, dataclasses.asdict(agent_cfg), log_dir=None, device="cuda:0")

    update = runner.alg.update
    times: list[float] = []
    prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])

    def timed_update():
        last = len(times) == args.iterations - 1
        torch.cuda.synchronize()
        t = time.perf_counter()
        if last:
            prof.__enter__()
        result = update()
        torch.cuda.synchronize()
        if last:
            prof.__exit__(None, None, None)
        times.append(time.perf_counter() - t)
        return result

    runner.alg.update = timed_update
    runner.learn(num_learning_iterations=args.iterations)

    events = prof.key_averages()
    # Kernels only: an operator's self device time repeats the time of the kernels it launched.
    device_us = sum(e.self_device_time_total for e in events if e.device_type != DeviceType.CPU)
    print(f"PPO update wall time per iteration: {', '.join(f'{t * 1e3:.0f} ms' for t in times)}")
    print(f"profiled update: device busy {device_us / 1e3:.1f} ms of {times[-1] * 1e3:.0f} ms wall")
    print(events.table(sort_by="self_device_time_total", row_limit=args.rows, max_name_column_width=50))


if __name__ == "__main__":
    main()
