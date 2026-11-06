"""Physics parity probe: MJWarp on the HIP device against MJWarp on the CPU and MuJoCo C.

Runs one scene, initial state and control sequence on three backends and
compares the trajectories:

- MuJoCo C (float64), the reference implementation;
- MJWarp on Warp's CPU device (float32), the same kernels as on the GPU;
- MJWarp on the HIP device, with several identical worlds in one batch.

Solver options and contact budgets come from a training task. Scenarios:

- stand, drop, crouch: the walk scene the velocity task trains on (only the
  feet touch the ground), with the XML position actuators stiffened enough to
  hold the stance; holding, a 5 cm drop, and a 1 Hz crouch (leg targets
  scaled by ±20 %, symmetric so the stance survives);
- collapse: the all-collisions scene (every part collides) with the stock
  actuators and zero command, so the robot falls and lies on the floor, which
  is the contact-heavy case where a broken backend lets bodies sink through.

The point is to catch a broken HIP backend (a robot sinking through the floor
while training rewards still look healthy) before any training run is trusted.

    python physics_parity.py --out-dir logs/results/<TS>_physics_parity
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch  # noqa: F401  # before warp, so the ROCm libraries bundled with torch load first
import warp as wp
import mujoco
import mujoco_warp as mjw

import mjlab_microduck
from mjlab.tasks.registry import load_env_cfg

SCENES = Path(mjlab_microduck.__file__).parent / "robot/microduck"
# Position-servo gains that hold the walk stance at the task's 5 ms step (the
# stock XML gains, kp 0.55 and 0.96 N·m, let the robot fold within ~1 s).
STIFF = {"kp": 4.0, "kv": 0.1, "force": 1.5}

# Pass criteria. HIP and Warp CPU run the same float32 kernels, so they must
# agree closely; MuJoCo C runs in float64 with its own solver, so only gross
# disagreement there is a failure. Identical worlds are not bitwise identical
# on the GPU (accumulation order varies run to run, ~1e-5) and dynamic motion
# amplifies that through stick-slip at the feet, so the batch is held to a
# spread well below the task's own encoder-bias randomisation (±0.015 rad).
# Collapse is chaotic once parts hit the floor, so it is held to the floor and
# resting-height checks only.
TOL = {
    "world_spread": 1e-2,     # m or rad, across identical worlds in one HIP batch
    "trunk_z_vs_cpu": 2e-3,   # m, HIP vs Warp CPU, whole run
    "qpos_vs_cpu": 2e-2,      # rad, HIP vs Warp CPU joints at the end
    "trunk_z_vs_mjc": 1e-2,   # m, HIP vs MuJoCo C, stand and drop only
    "rest_z_vs_cpu": 1e-2,    # m, HIP vs Warp CPU trunk height at the end of collapse
    "floor": -5e-3,           # m, lowest body origin may not go below this
}
CHECKS = {
    "stand": ["world_spread", "trunk_z_vs_cpu", "qpos_vs_cpu", "trunk_z_vs_mjc", "floor"],
    "drop": ["world_spread", "trunk_z_vs_cpu", "qpos_vs_cpu", "trunk_z_vs_mjc", "floor"],
    "crouch": ["world_spread", "trunk_z_vs_cpu", "qpos_vs_cpu", "trunk_z_vs_mjc", "floor"],
    "collapse": ["rest_z_vs_cpu", "floor"],
}


def load(scene: str, sim_cfg, stiff: bool) -> mujoco.MjModel:
    mjm = mujoco.MjModel.from_xml_path(str(SCENES / scene))
    sim_cfg.mujoco.apply(mjm)
    if stiff:
        mjm.actuator_gainprm[:, 0] = STIFF["kp"]
        mjm.actuator_biasprm[:, 1] = -STIFF["kp"]
        mjm.actuator_biasprm[:, 2] = -STIFF["kv"]
        mjm.actuator_forcerange[:] = [-STIFF["force"], STIFF["force"]]
    return mjm


def scenarios(sim_cfg, seconds: float) -> dict:
    """name -> (model, initial qpos, control sequence)"""
    walk = load("scene_walk.xml", sim_cfg, stiff=True)
    steps = int(round(seconds / walk.opt.timestep))
    stand = walk.key("STAND")
    t = np.arange(steps)[:, None] * walk.opt.timestep
    legs = np.ones(walk.nu)
    legs[5:9] = 0.0  # neck_pitch, head_pitch, head_yaw, head_roll stay put
    crouch = stand.ctrl * (1.0 + 0.2 * legs * np.sin(2 * np.pi * 1.0 * t))
    drop_qpos = stand.qpos.copy()
    drop_qpos[2] += 0.05
    allc = load("scene_allcollisions.xml", sim_cfg, stiff=False)
    return {
        "stand": (walk, stand.qpos, np.tile(stand.ctrl, (steps, 1))),
        "drop": (walk, drop_qpos, np.tile(stand.ctrl, (steps, 1))),
        "crouch": (walk, stand.qpos, crouch),
        "collapse": (allc, allc.key("STAND").qpos, np.zeros((steps, allc.nu))),
    }


def qpos_name(mjm: mujoco.MjModel, index: int) -> str:
    """Joint (and component, for the free joint) that owns qpos[index]."""
    for j in range(mjm.njnt):
        adr = mjm.jnt_qposadr[j]
        size = {mujoco.mjtJoint.mjJNT_FREE: 7, mujoco.mjtJoint.mjJNT_BALL: 4}.get(mjm.jnt_type[j], 1)
        if adr <= index < adr + size:
            name = mjm.joint(j).name
            return f"{name}[{'xyzwxyz'[index - adr]}]" if size == 7 else name
    return f"qpos[{index}]"


def run_mujoco_c(mjm, qpos0, ctrl):
    mjd = mujoco.MjData(mjm)
    mjd.qpos[:] = qpos0
    mujoco.mj_forward(mjm, mjd)
    qpos, low = [], []
    for u in ctrl:
        mjd.ctrl[:] = u
        mujoco.mj_step(mjm, mjd)
        qpos.append(mjd.qpos.copy())
        low.append(mjd.xpos[1:, 2].min())
    return np.array(qpos)[:, None, :], np.array(low)[:, None]


def run_mjwarp(mjm, qpos0, ctrl, device, nworld, sim_cfg):
    mjd = mujoco.MjData(mjm)
    mjd.qpos[:] = qpos0
    mujoco.mj_forward(mjm, mjd)
    with wp.ScopedDevice(device):
        m = mjw.put_model(mjm)
        # Same Warp-side options as mjlab's Simulation, plus the HIP fallback loop.
        m.opt.ls_parallel = sim_cfg.ls_parallel
        m.opt.contact_sensor_maxmatch = sim_cfg.contact_sensor_maxmatch
        m.opt.graph_conditional = False
        d = mjw.put_data(mjm, mjd, nworld=nworld, nconmax=sim_cfg.nconmax, njmax=sim_cfg.njmax)
        qpos, low = [], []
        for u in ctrl:
            d.ctrl.assign(np.tile(u, (nworld, 1)).astype(np.float32))
            mjw.step(m, d)
            qpos.append(d.qpos.numpy().copy())
            low.append(d.xpos.numpy()[:, 1:, 2].min(axis=1))
    return np.array(qpos), np.array(low)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", default="Mjlab-Velocity-Flat-MicroDuck")
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--worlds", type=int, default=64)
    ap.add_argument("--allocator", choices=["caching", "plain"], default="caching",
                    help="Warp allocator on the HIP device; training uses the fork's caching allocator")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    sim_cfg = load_env_cfg(args.task).sim
    wp.config.log_level = wp.LOG_WARNING
    wp.init()
    hip = wp.get_device("cuda:0")
    assert hip.is_hip, f"{hip} is not a HIP device"
    if args.allocator == "caching":
        from microduck_rl.patches.rocm import CachingAllocator

        wp.set_device_allocator(hip, CachingAllocator(wp.get_device_allocator(hip)))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report, series, ok = {}, {}, True
    cases = scenarios(sim_cfg, args.seconds)
    mjm = cases["stand"][0]
    steps = len(cases["stand"][2])
    for name, (model, qpos0, ctrl) in cases.items():
        ref_q, ref_low = run_mujoco_c(model, qpos0, ctrl)
        cpu_q, cpu_low = run_mjwarp(model, qpos0, ctrl, "cpu", 1, sim_cfg)
        hip_q, hip_low = run_mjwarp(model, qpos0, ctrl, hip, args.worlds, sim_cfg)

        spread = np.abs(hip_q - hip_q[:, :1])
        t_i, _, q_i = np.unravel_index(spread.argmax(), spread.shape)
        r = {
            "world_spread": float(spread.max()),
            "world_spread_median": float(np.median(spread.max(axis=(0, 2)))),
            "world_spread_at": f"{qpos_name(model, q_i)} at {t_i * model.opt.timestep:.2f} s",
            "trunk_z_vs_cpu": float(np.abs(hip_q[:, 0, 2] - cpu_q[:, 0, 2]).max()),
            "qpos_vs_cpu": float(np.abs(hip_q[-1, 0, 7:] - cpu_q[-1, 0, 7:]).max()),
            "trunk_z_vs_mjc": float(np.abs(hip_q[:, 0, 2] - ref_q[:, 0, 2]).max()),
            "rest_z_vs_cpu": float(np.abs(hip_q[-1, :, 2] - cpu_q[-1, 0, 2]).max()),
            "floor": float(min(hip_low.min(), cpu_low.min(), ref_low.min())),
            "trunk_z_end": {"mujoco_c": float(ref_q[-1, 0, 2]), "warp_cpu": float(cpu_q[-1, 0, 2]),
                            "warp_hip": float(hip_q[-1, 0, 2])},
            "contacts_scene": "feet only" if model is mjm else "all parts",
        }
        r["failed"] = [
            k for k in CHECKS[name]
            if (r[k] < TOL[k] if k == "floor" else r[k] > TOL[k])
        ]
        ok &= not r["failed"]
        if name == "stand":
            # Where the batch spread comes from: CPU worlds are bitwise identical,
            # the GPU varies run to run.
            cpu4_q, _ = run_mjwarp(model, qpos0, ctrl, "cpu", 4, sim_cfg)
            hip2_q, _ = run_mjwarp(model, qpos0, ctrl, hip, args.worlds, sim_cfg)
            dev = np.abs(hip_q[-1] - hip_q[-1, :1]).max(axis=1)
            determinism = {
                "cpu_4_world_spread": float(np.abs(cpu4_q - cpu4_q[:, :1]).max()),
                "hip_run_to_run_world0": float(np.abs(hip_q[:, 0] - hip2_q[:, 0]).max()),
                "hip_world_dev_median": float(np.median(dev)),
                "hip_world_dev_max": float(dev.max()),
            }
        report[name] = r
        series[name] = {"mujoco_c": ref_q[:, 0, 2], "warp_cpu": cpu_q[:, 0, 2], "warp_hip": hip_q[:, 0, 2]}
        print(name, json.dumps(r))

    meta = {"task": args.task, "scenes": {"stand/drop/crouch": "scene_walk.xml (stiffened servos)",
                                          "collapse": "scene_allcollisions.xml"},
            "stiff_servo": STIFF, "steps": steps, "timestep": mjm.opt.timestep,
            "iterations": mjm.opt.iterations, "ls_iterations": mjm.opt.ls_iterations,
            "nconmax": sim_cfg.nconmax, "njmax": sim_cfg.njmax, "worlds": args.worlds,
            "device": f"{hip.name} ({hip.arch_str})", "allocator": args.allocator,
            "tolerances": TOL, "determinism": determinism,
            "pass": bool(ok)}
    (args.out_dir / "parity.json").write_text(json.dumps({"meta": meta, "scenarios": report}, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(steps) * mjm.opt.timestep
    fig, axes = plt.subplots(1, len(series), figsize=(4 * len(series), 3), sharey=True)
    for ax, (name, s) in zip(axes, series.items()):
        for label, z in s.items():
            ax.plot(t, z, label=label, linestyle="--" if label == "warp_cpu" else "-")
        ax.set_title(name)
        ax.set_xlabel("time (s)")
    axes[0].set_ylabel("trunk height (m)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(args.out_dir / "trunk_height.png", dpi=120)

    lines = [f"# Physics parity: MJWarp on {meta['device']} vs Warp CPU vs MuJoCo C", "",
             f"Task options from `{args.task}`: timestep {meta['timestep']} s, {steps} steps, "
             f"iterations {meta['iterations']}, ls_iterations {meta['ls_iterations']}, "
             f"nconmax {meta['nconmax']}, njmax {meta['njmax']}; {args.worlds} HIP worlds; "
             f"{args.allocator} Warp allocator on the HIP device.", "",
             f"stand/drop/crouch use `scene_walk.xml` (feet-only ground contact, as in training) with servos "
             f"stiffened to kp {STIFF['kp']}, kv {STIFF['kv']}, ±{STIFF['force']} N·m; collapse uses "
             "`scene_allcollisions.xml` (every part collides) with the stock servos at zero command.", "",
             f"**Result: {'PASS' if ok else 'FAIL'}**", "",
             "| Scenario | World spread max (median, where) | Trunk z vs CPU (m) | Joints vs CPU (rad) "
             "| Trunk z vs MuJoCo C (m) | Resting z vs CPU (m) | Lowest body z (m) | Failed |",
             "|---|---|---|---|---|---|---|---|"]
    for name, r in report.items():
        keys = ["world_spread", "trunk_z_vs_cpu", "qpos_vs_cpu", "trunk_z_vs_mjc", "rest_z_vs_cpu"]
        marked = [f"{r[k]:.2e}" if k in CHECKS[name] else f"({r[k]:.2e})" for k in keys]
        marked[0] += f" ({r['world_spread_median']:.1e}, {r['world_spread_at']})"
        lines.append(f"| {name} | {' | '.join(marked)} | {r['floor']:.4f} | {', '.join(r['failed']) or '-'} |")
    d = determinism
    lines += ["", f"Tolerances: `{json.dumps(TOL)}`. Values in parentheses are reported, not checked: collapse "
              "is chaotic once parts hit the floor, so it is held to the floor and resting-height checks.", "",
              f"Batch determinism (stand): 4 identical worlds on Warp CPU differ by {d['cpu_4_world_spread']:.1e}; "
              f"on the GPU the same world differs run to run by {d['hip_run_to_run_world0']:.1e}, and the "
              f"{args.worlds} worlds deviate from world 0 by a median {d['hip_world_dev_median']:.1e} "
              f"(max {d['hip_world_dev_max']:.1e}). The GPU's accumulation order is not fixed, so identical "
              "worlds are not bitwise identical; the spread stays bounded.", "",
              "![trunk height](trunk_height.png)"]
    (args.out_dir / "README.md").write_text("\n".join(lines) + "\n")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
