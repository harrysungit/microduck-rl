"""Is the sitstand policy's sit started by its own exploration noise? (MuJoCo C, CPU)

PPO trains a stochastic policy: every action is the actor mean plus Gaussian noise of the learned std (0.15-0.26 rad per
joint at the end of the SitStand run). The robot runs the mean. If the mean policy holds a stand under the sit command and only
the noise (or a push) gets it started, the training reward was earned with noise the robot never has.

Runs sim2sim's clean L3 level (sensor observations, no observation noise, no pushes) with the actor mean of an rsl_rl
checkpoint plus Gaussian action noise of SCALE x the checkpoint's learned std, under the stand / sit / stand / sit / stand
schedule, and counts how often the trunk reaches the sit height.

    python scripts/sitstand_noise_sweep.py --run <rsl_rl run dir> --onnx <exported policy.onnx for the joint metadata> \
        --assets logs/sim2sim/assets_v2 --out-dir <dir>
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

import sim2sim

L3 = next(lv for lv in sim2sim.SITSTAND_LEVELS if lv.key == "L3")
ExportedPolicy = sim2sim.Policy  # main() replaces sim2sim.Policy; the export's metadata still comes from this one
ONNX_META: Path | None = None  # set in main; read by the workers (forked)


class CheckpointPolicy:
    """An rsl_rl checkpoint's actor mean (normalizer, 61-512-256-128-14 ELU MLP) plus Gaussian action noise."""

    def __init__(self, spec: str):
        path, scale, seed = spec.split("@")
        ck = torch.load(path, map_location="cpu", weights_only=False)["actor_state_dict"]
        torch.set_num_threads(1)
        self.mean, self.std = ck["obs_normalizer._mean"], ck["obs_normalizer._std"]
        self.layers = [(ck[f"mlp.{i}.weight"], ck[f"mlp.{i}.bias"]) for i in (0, 2, 4, 6)]
        self.noise = float(scale) * ck["distribution.std_param"].numpy().astype(float)
        self.rng = np.random.default_rng(int(seed))
        ref = ExportedPolicy(ONNX_META)  # the joint order, HOME and action scale of the export
        self.joint_names, self.home, self.scale = ref.joint_names, ref.home, ref.scale

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        h = (torch.from_numpy(obs.astype(np.float32))[None] - self.mean) / (self.std + 1e-2)
        for i, (w, b) in enumerate(self.layers):
            h = h @ w.T + b
            if i < 3:
                h = torch.nn.functional.elu(h)
        return h[0].double().numpy() + self.noise * self.rng.normal(size=len(self.noise))


def episode(args: tuple) -> dict:
    path, scale, seed, assets, servo = args
    r = sim2sim.run_episode(L3, seed, f"{path}@{scale}@{seed}", assets, servo)
    down = [s for s in r["segments"] if s["transition"] == "sit-down"]
    return {"path": path, "scale": scale, "seed": seed, "sit_down_ok": int(sum(s["ok"] for s in down)),
            "sit_down_tries": len(down), "rise_ok": int(sum(s["ok"] for s in r["segments"] if s["transition"] == "stand-up")),
            "rise_tries": sum(s["transition"] == "stand-up" for s in r["segments"]),
            "tipped": r["fell_at_s"] is not None}


def rate(rows: list[dict], key: str = "sit_down") -> float:
    return sum(r[f"{key}_ok"] for r in rows) / max(sum(r[f"{key}_tries"] for r in rows), 1)


def main() -> None:
    global ONNX_META
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--assets", type=Path, required=True)
    ap.add_argument("--final", default="model_2999.pt")
    ap.add_argument("--seeds", type=int, default=16, help="episodes per noisy point")
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ONNX_META = args.onnx
    sim2sim.Policy = CheckpointPolicy  # run_episode builds its policy by name; the workers are forked after this
    servo = sim2sim.SERVO_PARAMS

    final = args.run / args.final
    dose = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
    ckpts = sorted((p for p in args.run.glob("model_*.pt") if int(p.stem.split("_")[1]) >= 100),
                   key=lambda p: int(p.stem.split("_")[1]))
    jobs = [(str(final), s, seed, args.assets, servo) for s in dose for seed in (range(args.seeds) if s else [0])]
    jobs += [(str(p), s, seed, args.assets, servo) for p in ckpts for s in (0.0, 1.0)
             for seed in (range(8) if s else [0])]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(episode, jobs))
    (args.out_dir / "episodes.json").write_text(json.dumps(rows))

    fin = [r for r in rows if r["path"] == str(final)]
    lines = ["# SitStand: is the sit started by the policy's own exploration noise?", "",
             "MuJoCo C, clean L3 level (sensor observations; no observation noise, no pushes, no head command). Actor mean "
             "of the checkpoint plus Gaussian action noise of SCALE x the checkpoint's learned std. Schedule: stand 3 s, "
             "sit 5 s, stand 5 s, sit 5 s, stand 5 s; sit-down reached = trunk at most 7.5 cm and upright, held to the "
             "end of the sit segment.", "",
             f"## Final checkpoint `{args.final}`, noise scale", "",
             "| noise scale | mean action noise (rad) | sit-down reached | stand-up reached | tipped over |", "|---|---|---|---|---|"]
    ck = torch.load(final, map_location="cpu", weights_only=False)["actor_state_dict"]["distribution.std_param"]
    for s in dose:
        rr = [r for r in fin if r["scale"] == s]
        lines.append(f"| {s:g} | {s * float(ck.mean()):.3f} | {sum(r['sit_down_ok'] for r in rr)} of "
                     f"{sum(r['sit_down_tries'] for r in rr)} ({rate(rr):.0%}) | {rate(rr, 'rise'):.0%} | "
                     f"{sum(r['tipped'] for r in rr)} of {len(rr)} |")
    lines += ["", "## Every 100 iterations: noise 0 (the robot) against noise 1 (what training sampled)", "",
              "| iteration | sit-down, mean action | sit-down, with the training noise | tipped over (with noise) |",
              "|---|---|---|---|"]
    curve = []
    for p in ckpts:
        it = int(p.stem.split("_")[1])
        r0 = [r for r in rows if r["path"] == str(p) and r["scale"] == 0.0]
        r1 = [r for r in rows if r["path"] == str(p) and r["scale"] == 1.0]
        curve.append((it, rate(r0), rate(r1)))
        lines.append(f"| {it} | {sum(r['sit_down_ok'] for r in r0)} of {sum(r['sit_down_tries'] for r in r0)} | "
                     f"{sum(r['sit_down_ok'] for r in r1)} of {sum(r['sit_down_tries'] for r in r1)} ({rate(r1):.0%}) | "
                     f"{sum(r['tipped'] for r in r1)} of {len(r1)} |")
    lines += ["", "![sweep](noise_sweep.png)", ""]
    (args.out_dir / "README.md").write_text("\n".join(lines))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    a.plot(dose, [rate([r for r in fin if r["scale"] == s]) for s in dose], "o-")
    a.set_xlabel("action noise, x the learned std")
    a.set_ylabel("share of sit commands that reach the sit")
    a.set_title(f"final checkpoint ({args.final})", fontsize="small")
    its = [c[0] for c in curve]
    b.plot(its, [c[1] for c in curve], "o-", label="mean action (the robot)")
    b.plot(its, [c[2] for c in curve], "s-", label="with the training noise")
    b.set_xlabel("training iteration")
    b.set_title("share of sit commands that reach the sit", fontsize="small")
    b.legend(fontsize="small")
    for ax in (a, b):
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_dir / "noise_sweep.png", dpi=110)
    print("\n".join(lines[:16]))


if __name__ == "__main__":
    main()
