"""Full-cycle CPU ONNX check, head/neck substep audit, and mandatory normal-speed video."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess

import sim2sim as sim


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--onnx", required=True, type=Path)
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--assets", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--calibration", type=Path)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest() if p else None
    report = {"policy_sha256": digest(args.onnx), "checkpoint_sha256": digest(args.checkpoint),
              "calibration_sha256": digest(args.calibration), "servo_sha256": digest(sim.SERVO_PARAMS),
              "source_sha256": {n: digest(Path(__file__).with_name(n)) for n in
                                ("sitstand_check.py", "sim2sim.py", "contact_audit.py", "runtime_targets.py")},
              "schedule": "stand 0-3s; sit 3-8s; stand 8-13s", "episodes": [],
              "human_review": "pending", "deployment_approved": False,
              "scope": "Requested seeds in L3 nominal and, when calibration is supplied, runtime-target conditions. "
                       "Not a held-out/randomization/walk-switch certification."}
    try:
        base = next(lv for lv in sim.SITSTAND_LEVELS if lv.key == "L3")
        cases = [("nominal", base, None)]
        if args.calibration:
            cases.append(("runtime", replace(base, lowpass=(.5, .7)), args.calibration))
        for label, level, calibration in cases:
            for seed in args.seeds:
                raw = args.out_dir / f"{label}_seed{seed}_raw.mp4"
                row = sim.run_episode(level, seed, args.onnx, args.assets, sim.SERVO_PARAMS,
                                      record=True, video=raw, calibration=calibration,
                                      schedule_override=sim.SIT_FROM_STAND[:3])
                (args.out_dir / f"{label}_seed{seed}_traces.json").write_text(json.dumps(row.pop("traces")))
                video = args.out_dir / f"{label}_seed{seed}.mp4"
                filters = ["drawtext=text='CMD STAND':x=24:y=24:fontsize=30:fontcolor=white:box=1:boxcolor=black@0.6:enable='lt(t,3)+gte(t,8)'",
                           "drawtext=text='CMD SIT':x=24:y=24:fontsize=30:fontcolor=white:box=1:boxcolor=black@0.6:enable='between(t,3,7.999)'",
                           f"drawtext=text='{label} seed {seed} | 1x | t=%{{pts\\:hms}}':x=24:y=68:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.6"]
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(raw), "-vf", ",".join(filters),
                                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video)], check=True)
                row.update(condition=label, seed=seed, video=video.name)
                first = row["safety"]["first_contact_s"]
                if first is not None:
                    slow = video.with_name(video.stem + "_impact_4x.mp4")
                    begin = max(0., first - .3)
                    caption = f"4x SLOW | {label} seed {seed} | source {begin:.3f}-{begin + .9:.3f}s"
                    slow_filter = (f"drawtext=text='{caption}':x=24:y=24:fontsize=28:fontcolor=white:box=1:"
                                   "boxcolor=black@0.6,setpts=4*PTS")
                    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(begin),
                                    "-t", "0.9", "-i", str(raw), "-vf", slow_filter, "-an",
                                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(slow)], check=True)
                    row["slow_video"] = slow.name
                report["episodes"].append(row)
        report["automatic_verdict"] = "PASS" if all(e["automatic_pass"] for e in report["episodes"]) else "FAIL"
    except Exception as exc:
        report.update(automatic_verdict="FAIL" if any(not e["automatic_pass"] for e in report["episodes"]) else "UNVERIFIED",
                      error=f"{type(exc).__name__}: {exc}")
    (args.out_dir / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"automatic_verdict": report["automatic_verdict"], "policy_sha256": report["policy_sha256"],
                      "episodes": len(report["episodes"]), "error": report.get("error")}), flush=True)
    return {"PASS": 0, "FAIL": 1, "UNVERIFIED": 2}[report["automatic_verdict"]]


if __name__ == "__main__":
    raise SystemExit(main())
