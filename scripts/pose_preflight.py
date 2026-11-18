"""Hold HOME/SIT targets for 3 s from noisy starts. Physics diagnostic, not policy validation."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import re
from pathlib import Path
import subprocess

import mujoco
import numpy as np
import mjlab  # noqa: F401; load task entry points before importing upstream robot constants
from mjlab_microduck.robot.microduck_constants import HOME_FRAME
import sim2sim as sim
from contact_audit import Geometry, HeadAudit
from runtime_targets import TargetTransform

SIT = {"left_hip_roll": 0., "left_hip_pitch": -.4079, "left_knee": 1.35, "left_ankle": 0.,
       "right_hip_roll": 0., "right_hip_pitch": .4079, "right_knee": -1.35, "right_ankle": 0.}
LIMITS = {"max_tilt_deg": 15., "tail_max_angular_speed_rad_s": .5,
          "tail_max_vertical_speed_m_s": .02, "tail_height_range_m": .005,
          "min_head_clearance_m": .002}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def home_targets(names):
    values = []
    for name in names:
        matches = [v for pattern, v in HOME_FRAME.joint_pos.items() if re.fullmatch(pattern, name)]
        if len(matches) != 1:
            raise ValueError(f"HOME must resolve exactly once for {name}: {matches}")
        values.append(matches[0])
    return np.array(values, dtype=float)


def run(args, pose, seed):
    rng = np.random.default_rng(seed)
    scene_name = getattr(args, "scene", "groundcontact")
    level = sim.Level("pose", "Fixed pose diagnostic", scene=scene_name,
                      collisions="groundcontact" if scene_name == "groundcontact" else "authored",
                      vin=args.vin, task="sitstand")
    scene = args.assets / f"scene_hd1910_{scene_name}.xml"
    raw = mujoco.MjModel.from_xml_path(str(scene))
    names = [raw.joint(int(raw.actuator_trnid[a, 0])).name for a in range(raw.nu)]
    home = home_targets(names)
    sit_target = np.array([SIT.get(n, home[i]) for i, n in enumerate(names)])
    target = sit_target.copy() if pose == "sit" else home.copy()
    p = sim.sample_params(level, rng, len(names))
    model = sim.build_model(level, args.assets, names, sim.SERVO_PARAMS, p)
    data = mujoco.MjData(model)
    servos = sim.Servos(model, names, sim.SERVO_PARAMS, p, level.latency_ms, rng)
    transform = TargetTransform(names, home, args.calibration) if args.calibration else None
    geometry = Geometry(model)
    audit = HeadAudit(model, geometry)
    data.qpos[:3], data.qpos[3:7] = (0, 0, .15), (1, 0, 0, 0)
    data.qpos[servos.qadr] = target
    if seed:
        data.qpos[servos.qadr] += rng.uniform(-.03, .03, len(names))
        axis = rng.normal(size=3)
        axis[2] = 0
        axis /= np.linalg.norm(axis)
        angle = math.radians(rng.uniform(-3, 3))
        data.qpos[3:7] = np.r_[math.cos(angle / 2), axis * math.sin(angle / 2)]
    mujoco.mj_forward(model, data)
    collision_geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] != 0 and
                       (model.geom_contype[g] or model.geom_conaffinity[g])]
    data.qpos[2] += .002 - min(geometry.minimum_z(data, g) for g in collision_geoms)
    mujoco.mj_forward(model, data)
    spawn = data.qpos.copy()
    servos.reset(data, target)
    audit.sample(data, 0., 0.)
    renderer = camera = encoder = None
    video = args.out_dir / f"{pose}_seed{seed}.mp4"
    if args.video:
        renderer = mujoco.Renderer(model, height=480, width=640)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [0, 0, .09]
        camera.distance, camera.azimuth, camera.elevation = .65, 135, -12
        # Text is authored by this script; poses and seeds are fixed identifiers.
        label = f"{'CONTROL' if pose == 'cycle' else 'STATIC'} {pose.upper()} | seed {seed} | 1x | t=%{{pts\\:hms}}"
        overlay = f"drawtext=text='{label}':x=12:y=12:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.6"
        if pose == "cycle":
            overlay += ",drawtext=text='CMD STAND':x=12:y=40:fontsize=18:fontcolor=white:box=1:enable='lt(t,3)+gte(t,8)'"
            overlay += ",drawtext=text='CMD SIT':x=12:y=40:fontsize=18:fontcolor=white:box=1:enable='between(t,3,7.999)'"
        encoder = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                    "-s", "640x480", "-r", "50", "-i", "-", "-vf",
                                    overlay,
                                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video)], stdin=subprocess.PIPE)
    rows = []
    try:
        for tick in range(650 if pose == "cycle" else 150):
            if pose == "cycle":
                t = tick * sim.CONTROL_DT
                blend = np.clip((t - 3.) / 2., 0., 1.) - np.clip((t - 8.) / 2., 0., 1.)
                target = home + blend * (sit_target - home)
                # Optional feasibility probes, never imposed as training waypoints.
                bump = math.sin(math.pi * blend)
                for joint, offset in (("left_hip_pitch", -args.mid_hip_offset),
                                      ("right_hip_pitch", args.mid_hip_offset),
                                      ("left_ankle", args.mid_ankle_offset),
                                      ("right_ankle", -args.mid_ankle_offset)):
                    target[names.index(joint)] += bump * offset
            supply = max(servos.vin - servos.drop_gain * abs(servos.prev_torque).sum(), sim.VIN_MIN)
            servos.queue.send(data.time, transform(target, supply) if transform else target)
            for _ in range(10):
                time_s = float(data.time)
                servos.step(model, data)
                mujoco.mj_step(model, data)
                audit.sample(data, time_s, model.opt.timestep)
            mujoco.mj_forward(model, data)
            trunk = model.body("trunk_base").id
            tilt = math.degrees(math.acos(np.clip(data.xmat[trunk].reshape(3, 3)[2, 2], -1, 1)))
            rows.append([float(data.time), float(data.xpos[trunk, 2]), tilt,
                         float(np.linalg.norm(data.qvel[3:6])), abs(float(data.qvel[2]))])
            if renderer:
                renderer.update_scene(data, camera=camera)
                encoder.stdin.write(renderer.render().tobytes())
    finally:
        if encoder:
            encoder.stdin.close()
            code = encoder.wait(timeout=30)
            renderer.close()
            if code:
                raise RuntimeError(f"Video encoding failed ({code})")
    trace = np.asarray(rows)
    np.savetxt(args.out_dir / f"{pose}_seed{seed}.csv", trace, delimiter=",",
               header="time_s,trunk_z_m,tilt_deg,angular_speed_rad_s,vertical_speed_m_s", comments="")
    tail = trace[trace[:, 0] >= (12.0 if pose == "cycle" else 2.0)]
    metrics = {"max_tilt_deg": float(trace[:, 2].max()), "tail_mean_height_m": float(tail[:, 1].mean()),
               "tail_height_range_m": float(np.ptp(tail[:, 1])),
               "tail_max_angular_speed_rad_s": float(tail[:, 3].max()),
               "tail_max_vertical_speed_m_s": float(tail[:, 4].max())}
    safety = audit.result()
    passed = not safety["head_contact"] and safety["min_clearance_m"] >= LIMITS["min_head_clearance_m"]
    passed = passed and all(metrics[k] <= v for k, v in LIMITS.items() if k != "min_head_clearance_m")
    segments = None
    if pose == "cycle":
        segments = sim.posture_segments({"t": trace[:, 0], "z": trace[:, 1], "tilt": trace[:, 2],
                                        "vx": np.zeros(len(trace)), "vy": np.zeros(len(trace))},
                                       sim.SIT_FROM_STAND[:3])
        passed = passed and all(s["ok"] for s in segments)
    return {"pose": pose, "seed": seed, "passed": bool(passed), "metrics": metrics, "safety": safety,
            "segments": segments,
            "level": asdict(level), "joint_names": names, "targets_rad": target.tolist(), "home_rad": home.tolist(),
            "spawn_qpos": spawn.tolist(), "total_mass_kg": float(model.body_mass.sum()),
            "video": video.name if args.video else None,
            "mass_inertia_targets_sha256": hashlib.sha256(model.body_mass.tobytes() + model.body_ipos.tobytes() +
                                                       model.body_inertia.tobytes() + target.tobytes()).hexdigest()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--assets", required=True, type=Path)
    ap.add_argument("--scene", choices=["groundcontact", "allcollisions"], default="groundcontact",
                    help="match the actual training collision model; HeadSafe uses allcollisions")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--cycle", action="store_true", help="diagnose a 2-second joint-target blend; this is not a learned policy")
    ap.add_argument("--mid-hip-offset", type=float, default=0., help="cycle diagnostic only: symmetric mid-transition hip offset, rad")
    ap.add_argument("--mid-ankle-offset", type=float, default=0., help="cycle diagnostic only: symmetric mid-transition ankle offset, rad")
    ap.add_argument("--vin", type=float, default=7.7)
    ap.add_argument("--calibration", type=Path, help="private device TOML: enable limits, lowpass .5/.7, voltage and encoder goals")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    report = {"kind": "joint_blend_cycle_diagnostic" if args.cycle else "static_pose_physics_diagnostic",
              "policy_used": False, "thresholds": LIMITS,
              "cycle_mid_offsets_rad": {"hip": args.mid_hip_offset, "ankle": args.mid_ankle_offset},
              "schedule": "stand 0-3s; blend to sit 3-5s; sit 5-8s; blend to stand 8-10s; stand 10-13s" if args.cycle else None,
              "human_review": "pending", "deployment_approved": False,
              "limitations": ["This diagnostic uses authored controls, not a learned policy; a failed blend does not prove no safe path exists.",
                              ("Uses the supplied calibration with voltage, lowpass .5/.7 and integer encoder goals."
                               if args.calibration else "Uses BAM without the device calibration/filter/voltage target pipeline."),
                              "A pass alone does not authorize training or deployment; runtime and reward audits remain required."],
              "servo_sha256": sha(sim.SERVO_PARAMS), "script_sha256": sha(__file__),
              "calibration_sha256": sha(args.calibration) if args.calibration else None,
              "runtime_target_source_sha256": sha(Path(__file__).with_name("runtime_targets.py")),
              "scene": args.scene,
              "assets_sha256": {p.name: sha(p) for p in sorted(args.assets.glob(f"*{args.scene}.xml"))},
              "rows": []}
    try:
        for pose in (("cycle",) if args.cycle else ("stand", "sit")):
            for seed in args.seeds:
                row = run(args, pose, seed)
                report["rows"].append(row)
                print(json.dumps({"pose": pose, "seed": seed, "pass": row["passed"], "metrics": row["metrics"],
                                  "head_contact": row["safety"]["head_contact"]}), flush=True)
        report["verdict"] = "PASS" if all(r["passed"] for r in report["rows"]) else "FAIL"
    except Exception as exc:
        report.update(verdict="FAIL" if any(not r["passed"] for r in report["rows"]) else "UNVERIFIED",
                      error=f"{type(exc).__name__}: {exc}")
    (args.out_dir / "preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"{report['kind']}: {report['verdict']}", flush=True)
    return {"PASS": 0, "FAIL": 1, "UNVERIFIED": 2}[report["verdict"]]


if __name__ == "__main__":
    raise SystemExit(main())
