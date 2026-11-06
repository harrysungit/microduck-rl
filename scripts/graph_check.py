"""Check that replaying the captured HIP graphs computes what eager execution computes.

Brings a task's environment to a mid-episode state, snapshots every array of
the mujoco_warp Data, and runs the same physics (4 steps and a forward) three
times from that snapshot: eagerly, eagerly again, and through the captured
graphs. The two eager runs give the GPU's own run-to-run spread; the graph run
must stay within the same order. Differences are reported per world (median,
99th percentile, maximum), since a few worlds in stiff contact amplify the
GPU's spread far beyond the rest.

    MICRODUCK_HIP_GRAPHS=1 python scripts/graph_check.py --num-envs 4096 --njmax 128
"""

import argparse
import dataclasses
import json

import numpy as np
import torch
import warp as wp

import mjlab  # noqa: F401  # loads the mjlab.tasks entry points, and with them the ROCm patch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

OUTPUTS = ["qpos", "qvel", "qacc", "xpos", "cvel", "qfrc_constraint"]


def arrays(obj, prefix=""):
    """(path, wp.array) for every array in a mujoco_warp Data, nested structs included."""
    for f in dataclasses.fields(obj):
        value = getattr(obj, f.name)
        if isinstance(value, wp.array):
            yield prefix + f.name, value
        elif dataclasses.is_dataclass(value):
            yield from arrays(value, prefix + f.name + ".")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", default="Mjlab-Velocity-Flat-MicroDuck")
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--njmax", type=int)
    ap.add_argument("--settle-steps", type=int, default=20)
    ap.add_argument("--out")
    args = ap.parse_args()

    cfg = load_env_cfg(args.task)
    cfg.scene.num_envs = args.num_envs
    cfg.seed = 0
    if args.njmax:
        cfg.sim.njmax = args.njmax
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    sim = env.sim
    if sim.step_graph is None:
        raise SystemExit("no captured graphs; run with MICRODUCK_HIP_GRAPHS=1")
    env.reset()
    gen = torch.Generator(device="cuda:0").manual_seed(0)
    for _ in range(args.settle_steps):
        env.step(0.5 * torch.randn(env.action_space.shape, device="cuda:0", generator=gen))

    snapshot = {path: wp.clone(a) for path, a in arrays(sim.wp_data)}
    live = dict(arrays(sim.wp_data))

    def run(use_graph: bool) -> dict:
        for path, saved in snapshot.items():
            wp.copy(live[path], saved)
        sim.use_cuda_graph = use_graph
        for _ in range(4):
            sim.step()
        sim.forward()
        wp.synchronize_device(sim.wp_device)
        return {k: getattr(sim.wp_data, k).numpy().copy() for k in OUTPUTS}

    eager_a, eager_b, graph = run(False), run(False), run(True)
    sim.use_cuda_graph = True
    def per_world(a, b):
        d = np.abs(a - b).reshape(a.shape[0], -1).max(axis=1)
        return {"median": float(np.median(d)), "p99": float(np.percentile(d, 99)), "max": float(d.max())}

    result = {}
    for k in OUTPUTS:
        ee, ge = per_world(eager_b[k], eager_a[k]), per_world(graph[k], eager_a[k])
        result[k] = {"eager_vs_eager": ee, "graph_vs_eager": ge, "graph_finite": bool(np.isfinite(graph[k]).all())}
        print(f"{k:<16} per-world max diff, median / p99 / max:  eager vs eager "
              f"{ee['median']:.1e} / {ee['p99']:.1e} / {ee['max']:.1e}   graph vs eager "
              f"{ge['median']:.1e} / {ge['p99']:.1e} / {ge['max']:.1e}   finite {result[k]['graph_finite']}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
