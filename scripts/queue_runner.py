"""Unattended multi-stage training queue.

Runs the stages of a queue file one after another. A stage trains one task from scratch, warm-started
from another task's checkpoint (MICRODUCK_WARM_START=1: curricula and iteration restart at 0), or
resumed from a checkpoint of the same task family (counters kept, so the curricula stay where they
were). For each stage:
- a time cap (its own `hours` and the queue deadline) and a stall watchdog (no console output for
  20 min); a failed segment resumes from the stage's last checkpoint, at most twice, with njmax or
  nconmax x1.5 after a budget overflow;
- convergence early stop: once the stage's curricula have ended and 1000 more iterations have run,
  training stops at the first checkpoint where the mean reward of the last 500 iterations is less
  than 1% above the mean of the 500 before;
- afterwards: summary (curve, ONNX, versions), deterministic evaluation of checkpoints at 25/50/75%
  (`eval.fractions`) and the last one, plus the stage's starting checkpoint and comparison models,
  a video and a README in the stage's result dir.
- an acceptance check (acceptance_core.py, scripts/acceptance.py): the stage's declared contract run on the policy that
  ships, in the training world, clean and as trained. The verdict is ACCEPTED, FAILED or UNVERIFIED and goes into the
  state, the summary table and the stage's README. A queue whose stages do not all declare an `acceptance` entry (a
  contract, or "none" on purpose) does not start; "done" means trained. ACCEPTED is an automatic result only;
  full-motion video and human review are still required before considering deployment.
Stages whose dependencies did not succeed (or whose acceptance FAILED) run their fallback or are skipped; a stage that cannot
reach its curriculum end before the deadline is deferred. The two-stage gate (stage `gate:
two_stage`) decides whether stages marked `needs_gate` run. State goes to <queue dir>/state.json,
which a restart picks up, and the queue's summary README is rewritten after every stage.

    nohup setsid python scripts/queue_runner.py --queue queues/hd1910_skills.json > /dev/null 2>&1 < /dev/null &
    python scripts/queue_runner.py --queue queues/hd1910_skills.json --dry-run   # 64 envs, a few iterations
"""

import argparse
import json
import os
import re
import signal
import statistics
import subprocess
import time
import traceback
from datetime import datetime
from pathlib import Path

import acceptance_core as ac

TRAIN = Path(__file__).resolve().parents[1]  # the repository root
ANSI = re.compile(r"\x1b\[[0-9;]*m")
ITER_RE = re.compile(r"Learning iteration (\d+)/(\d+)")
REWARD_RE = re.compile(r"Mean reward:\s+(-?[\d.]+(?:e[-+]?\d+)?)")
CKPT_RE = re.compile(r"model_(\d+)\.pt$")
EVAL_RE = re.compile(r"eval_(.+?)_s(\d+)(?:_compare)?\.json$")
POLL_S = 30
STALL_S = 20 * 60
WINDOW = 500
MIN_GAIN = 0.01


def ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def clock(t: float | None = None) -> str:
    return datetime.fromtimestamp(t if t is not None else time.time()).strftime("%Y-%m-%d %H:%M:%S")


def ckpt_iter(path) -> int:
    return int(CKPT_RE.search(str(path)).group(1))


def ckpts(dirs) -> list[Path]:
    """Checkpoints of the given run dirs, by iteration (a later segment wins a tie)."""
    found = {}
    for d in dirs:
        for p in Path(d).glob("model_*.pt"):
            found[ckpt_iter(p)] = p
    return [found[k] for k in sorted(found)]


def verdict_cell(rec: dict) -> str:
    if not rec.get("verdict"):
        return "-"
    n = len(rec.get("verdict_warnings") or [])
    return f"**{rec['verdict']}**" + (f" ({n} warning{'s' if n > 1 else ''})" if n else "")


def check_declarations(spec: dict) -> None:
    """Refuse a queue with a stage that does not say how its skill is accepted: nothing trains unchecked by omission."""
    found = []
    for stage in spec["stages"]:
        found += [f"{stage['id']}: {m}" for m in ac.problems(stage)]
        if stage.get("fallback"):
            found += [f"{stage['id']} fallback: {m}" for m in ac.problems({**stage, **stage["fallback"]})]
    if found:
        raise SystemExit("the queue was not started; fix the acceptance declarations:\n  " + "\n  ".join(found))


class Queue:
    def __init__(self, args):
        self.args = args
        self.spec = json.loads(Path(args.queue).read_text())
        check_declarations(self.spec)
        self.dry = args.dry_run
        name = self.spec["name"] + ("_dryrun" if self.dry else "")
        if args.state:
            self.qdir = Path(args.state).resolve().parent
            self.state = json.loads(Path(args.state).read_text())
        else:
            self.qdir = TRAIN / "logs/queue" / f"{ts()}_{name}"
            self.qdir.mkdir(parents=True)
            self.state = {"queue": name, "queue_file": str(Path(args.queue).resolve()), "started": clock(),
                          "gate": {}, "stages": {}, "order": []}
        root = self.qdir / "results" if self.dry else (TRAIN / self.spec.get("results_root", "logs/results"))
        self.results_root = root.resolve()
        self.summary_dir = Path(self.state.get("summary_dir") or self.results_root / f"{ts()}_hd1910_{self.spec['name']}")
        self.state["summary_dir"] = str(self.summary_dir)
        self.deadline = datetime.strptime(self.spec["deadline"], "%Y-%m-%d %H:%M").timestamp()
        if self.dry:
            self.deadline = time.time() + 4 * 3600
        self.save()

    # -- bookkeeping -------------------------------------------------------------------------------
    def log(self, msg: str) -> None:
        line = f"{clock()} {msg}"
        print(line, flush=True)
        with open(self.qdir / "queue.log", "a") as f:
            f.write(line + "\n")

    def save(self) -> None:
        self.state["heartbeat"] = clock()
        tmp = self.qdir / "state.json.tmp"
        tmp.write_text(json.dumps(self.state, indent=2))
        tmp.replace(self.qdir / "state.json")

    def resolve(self, value: str) -> str:
        """`@Q1` is stage Q1's last checkpoint; other values are paths relative to the repository root."""
        if value.startswith("@"):
            rec = self.state["stages"].get(value[1:], {})
            if not rec.get("final_checkpoint"):
                raise RuntimeError(f"{value}: stage has no checkpoint")
            return rec["final_checkpoint"]
        return str((TRAIN / value).resolve())

    def stage_env_value(self, value: str) -> str:
        """A stage's environment value: `@Q1` and paths are resolved, anything else (MICRODUCK_FIXED_STD=0.05) is as written."""
        return self.resolve(value) if value.startswith("@") or "/" in value else value

    def succeeded(self, sid: str) -> bool:
        """Trained to the end and not rejected: a stage whose acceptance FAILED does not feed the stages behind it."""
        rec = self.state["stages"].get(sid, {})
        return rec.get("status") == "done" and rec.get("verdict") != "FAILED"

    # -- one stage ---------------------------------------------------------------------------------
    def plan(self, spec: dict) -> dict:
        """The stage to run: its fallback when a dependency did not succeed. None to skip."""
        missing = [d for d in spec.get("needs", []) if not self.succeeded(d)]
        if missing:
            if spec.get("fallback"):
                fb = {**spec, **spec["fallback"], "id": spec["id"]}
                fb.pop("fallback", None)
                fb["note"] = f"fallback: {', '.join(missing)} did not succeed"
                return self.plan(fb)
            return {"skip": f"skipped: {', '.join(missing)} did not succeed"}
        gate = spec.get("needs_gate")
        if gate and self.state["gate"].get(gate, {}).get("passed") is not True:
            return {"skip": f"skipped: the {gate} gate did not pass"}
        if self.dry:
            spec = {**spec, "iters": min(spec["iters"], 6), "num_envs": 64, "save": 2, "curriculum_end": 0,
                    "hours": 0.5, "sec_per_iter": 1.0, "experiment": spec["experiment"] + "_dryrun"}
        return spec

    def run_stage(self, spec: dict) -> None:
        sid = spec["id"]
        prev = self.state["stages"].get(sid, {})
        exp_dir = TRAIN / "logs/rsl_rl" / spec["experiment"]
        exp_dir.mkdir(parents=True, exist_ok=True)
        restart = prev.get("status") == "running" and prev.get("segments")
        rec = prev if restart else {
            "id": sid, "skill": spec["skill"], "task": spec["task"], "status": "running", "started": clock(),
            "res": str(self.results_root / f"{ts()}_hd1910_{spec['skill']}"), "segments": [], "notes": [],
            "init": spec.get("init", {"mode": "scratch"}), "iters": spec["iters"], "num_envs": spec["num_envs"]}
        if spec.get("note"):
            rec["notes"].append(spec["note"])
        self.state["stages"][sid] = rec
        if sid not in self.state["order"]:
            self.state["order"].append(sid)
        res = Path(rec["res"])
        res.mkdir(parents=True, exist_ok=True)
        run_name = f"{sid.lower()}_{spec['skill']}"

        env = {k: v for k, v in os.environ.items() if k != "MICRODUCK_WARM_START"}
        for k, v in spec.get("env", {}).items():
            env[k] = self.stage_env_value(v)
        init = rec["init"]
        start_abs, resume_args = 0, []
        if init["mode"] in ("resume", "warm"):
            src = Path(self.resolve(init["checkpoint"]))
            rec["init_checkpoint"] = str(src)
            link = exp_dir / src.parent.name
            if src.parent.resolve() != link.resolve():
                if not link.exists():
                    link.symlink_to(src.parent.resolve())
                    rec["notes"].append(f"linked {src.parent} into {exp_dir.name}/ to load it")
            resume_args = ["--agent.resume", "True", "--agent.load-run", src.parent.name,
                           "--agent.load-checkpoint", src.name]
            if init["mode"] == "warm":
                env["MICRODUCK_WARM_START"] = "1"
            else:
                start_abs = ckpt_iter(src)
        rec["start_iteration"] = start_abs
        end_abs = start_abs + spec["iters"] - 1
        cur_end = spec.get("curriculum_end", 0)
        njmax, nconmax = spec["njmax"], spec.get("nconmax")
        stage_deadline = min(time.time() + spec["hours"] * 3600, self.deadline)
        if restart:
            last = ckpts(rec["segments"])[-1]
            resume_args = ["--agent.resume", "True", "--agent.load-run", last.parent.name,
                           "--agent.load-checkpoint", last.name]
            env.pop("MICRODUCK_WARM_START", None)
            remaining = end_abs - ckpt_iter(last) + 1
            rec["notes"].append(f"queue restarted; resumed from {last.parent.name}/{last.name}")
        else:
            remaining = spec["iters"]
        rewards: dict[int, float] = {int(k): v for k, v in rec.get("rewards", {}).items()}
        attempts = rec.get("attempts", 0)
        self.log(f"{sid} {spec['task']}: {init['mode']}, {spec['iters']} iterations, {spec['num_envs']} envs, "
                 f"njmax {njmax}, curricula end at +{cur_end}, cap {clock(stage_deadline)}")

        sampler_stop = self.qdir / f"{sid}.gpu_stop"
        sampler = subprocess.Popen(["python", "scripts/sample_gpu.py", "--out", str(res / "gpu.csv"),
                                    "--until", str(sampler_stop)], cwd=TRAIN, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, env=env)
        stop = None
        while True:
            seg = len(rec["segments"])
            seg_log = self.qdir / f"{sid}_seg{seg}.log"
            cmd = ["train", spec["task"], "--env.scene.num-envs", str(spec["num_envs"]), "--env.sim.njmax", str(njmax),
                   *(["--env.sim.nconmax", str(nconmax)] if nconmax else []),
                   "--agent.max-iterations", str(remaining), "--agent.save-interval", str(spec.get("save", 100)),
                   "--agent.logger", "tensorboard", "--agent.experiment-name", spec["experiment"],
                   "--agent.run-name", run_name, *resume_args]
            t0 = time.time()
            with open(seg_log, "w") as f:
                proc = subprocess.Popen(cmd, cwd=TRAIN, env=env, stdout=f, stderr=subprocess.STDOUT,
                                        start_new_session=True)
            self.log(f"{sid} segment {seg}: pid {proc.pid}, {remaining} iterations"
                     + (f", {' '.join(resume_args[3:])}" if resume_args else ""))
            rec["pid"] = proc.pid
            self.save()
            seg_dir, reason, parser = None, None, ConsoleParser(seg_log)
            seen = -1
            while proc.poll() is None:
                time.sleep(POLL_S if not self.dry else 5)
                parser.read(rewards)
                if seg_dir is None:
                    seg_dir = self.find_run(exp_dir, run_name, t0, rec["segments"])
                if time.time() >= stage_deadline:
                    reason = "deadline" if stage_deadline >= self.deadline else "time cap"
                    self.log(f"{sid}: {reason} reached; stopping")
                    stop_group(proc)
                    break
                if time.time() - seg_log.stat().st_mtime > STALL_S:
                    reason = "stalled"
                    self.log(f"{sid}: no console output for {STALL_S // 60} min; killing")
                    stop_group(proc)
                    break
                if seg_dir is not None:
                    found = ckpts([seg_dir])
                    if found and ckpt_iter(found[-1]) != seen:
                        seen = ckpt_iter(found[-1])
                        verdict = converged(rewards, seen, start_abs, cur_end)
                        rec["last_iteration"] = seen
                        rec["rewards"] = {str(k): v for k, v in rewards.items()}
                        self.save()
                        if verdict:
                            rec["convergence"] = verdict
                            reason = "converged"
                            self.log(f"{sid}: converged at {seen} (mean reward {verdict['prev_mean']:.2f} -> "
                                     f"{verdict['last_mean']:.2f}, {verdict['gain']:+.2%}); stopping")
                            stop_group(proc)
                            break
            code = proc.wait()
            parser.read(rewards)
            if seg_dir is None:
                seg_dir = self.find_run(exp_dir, run_name, t0, rec["segments"])
            if seg_dir is not None:
                rec["segments"].append(str(seg_dir))
            found = ckpts(rec["segments"])
            last = found[-1] if found else None
            rec["rewards"] = {str(k): v for k, v in rewards.items()}
            rec["last_iteration"] = ckpt_iter(last) if last else None
            self.log(f"{sid} segment {seg} ended: code {code}{', ' + reason if reason else ''}; "
                     f"last checkpoint {last.name if last else 'none'}")
            if reason in ("deadline", "time cap", "converged"):
                stop = reason
                break
            if reason is None and code == 0 and last is not None and ckpt_iter(last) >= end_abs - 1:
                stop = "completed"
                break
            if attempts >= 2 or time.time() >= stage_deadline:
                stop = f"failed (code {code}{', ' + reason if reason else ''})"
                break
            attempts += 1
            rec["attempts"] = attempts
            text = seg_log.read_text(errors="ignore")
            overflow = [ln for ln in text.splitlines() if "ConstraintBudgetExceeded" in ln]
            if overflow and "nconmax" in overflow[-1]:
                nconmax = int((nconmax or 200) * 1.5)
                rec["notes"].append(f"contact budget overflow in segment {seg}; nconmax {nconmax}")
            elif overflow:
                njmax = int(njmax * 1.5)
                rec["notes"].append(f"constraint budget overflow in segment {seg}; njmax {njmax}")
            if last is not None:
                resume_args = ["--agent.resume", "True", "--agent.load-run", last.parent.name,
                               "--agent.load-checkpoint", last.name]
                env.pop("MICRODUCK_WARM_START", None)  # the counters of this stage carry on
                remaining = end_abs - ckpt_iter(last) + 1
                rec["notes"].append(f"segment {seg} ended with code {code}{' (' + reason + ')' if reason else ''}; "
                                    f"resumed from {last.parent.name}/{last.name}")
            else:
                rec["notes"].append(f"segment {seg} ended with code {code} before a checkpoint; started again")
            if remaining <= 1:
                stop = "completed"
                break
        rec.update(stop_reason=stop, ended=clock(), njmax=njmax, nconmax=nconmax, pid=None)
        found = ckpts(rec["segments"])
        rec["final_checkpoint"] = str(found[-1].resolve()) if found else None
        progress = (ckpt_iter(found[-1]) - start_abs) if found else -1
        min_ok = spec.get("min_ok_iters", cur_end if cur_end else spec["iters"] // 2)
        ok = bool(found) and (progress >= min_ok - 1 or stop in ("completed", "converged"))
        if self.dry:
            ok = bool(found)
        rec["status"] = "post"
        self.save()
        self.log(f"{sid}: training {stop}; {progress + 1} iterations; post-processing")
        try:
            if found:  # the stage's environment: VelStand's runner loads its teachers even to evaluate
                post_env = {k: v for k, v in env.items() if k != "MICRODUCK_WARM_START"}
                self.post(spec, rec, found, start_abs, njmax, res, post_env)
        except Exception:
            rec["notes"].append("post-processing raised: " + traceback.format_exc(limit=3).replace("\n", " | "))
        sampler_stop.touch()
        try:
            sampler.wait(timeout=30)
        except subprocess.TimeoutExpired:
            sampler.kill()
        rec["status"] = "done" if ok else "failed"
        if not ok:
            self.set_verdict(rec, "FAILED", [f"training did not finish ({rec.get('stop_reason')})"])
        elif "verdict" not in rec:  # post-processing raised before the acceptance check
            self.set_verdict(rec, "UNVERIFIED", ["post-processing did not reach the acceptance check"])
        if spec.get("gate"):
            self.state["gate"][spec["gate"]] = self.gate(spec, rec, res) if ok else {
                "passed": False, "why": "the gate stage failed"}
            self.log(f"{sid}: {spec['gate']} gate {'passed' if self.state['gate'][spec['gate']]['passed'] else 'failed'}")
        self.write_stage_readme(spec, rec, res)
        self.save()
        self.log(f"{sid}: {rec['status']}")

    def find_run(self, exp_dir: Path, run_name: str, t0: float, known: list[str]):
        cands = [d for d in exp_dir.glob(f"*_{run_name}") if d.is_dir() and not d.is_symlink()
                 and str(d) not in known and d.stat().st_mtime >= t0 - 5]
        return sorted(cands)[-1] if cands else None

    # -- after training ----------------------------------------------------------------------------
    def run_logged(self, name: str, cmd: list[str], timeout: int, env=None) -> bool:
        with open(self.qdir / f"{name}.log", "w") as f:
            try:
                code = subprocess.run(cmd, cwd=TRAIN, stdout=f, stderr=subprocess.STDOUT, timeout=timeout,
                                      env=env).returncode
            except subprocess.TimeoutExpired:
                code = "timeout"
        if code != 0:
            self.log(f"{name}: exit {code}; see {self.qdir / (name + '.log')}")
        return code == 0

    def post(self, spec, rec, found, start_abs, njmax, res, env):
        sid, final = spec["id"], found[-1]
        if self.run_logged(f"{sid}_summarize", ["python", "scripts/summarize_run.py", spec["task"], str(final.parent),
                                                "--checkpoint", str(final), "--out-dir", str(res)], 1800, env):
            if (res / "README.md").exists():
                (res / "README.md").replace(res / "summary.md")
        # Checkpoints at 25/50/75% of the iterations this stage ran, and its last one.
        span = ckpt_iter(final) - start_abs
        ev = spec.get("eval", {})
        picks = []
        for f in ev.get("fractions", (0.25, 0.5, 0.75)):
            below = [c for c in found if ckpt_iter(c) <= start_abs + f * span]
            if below and below[-1] != final and below[-1] not in picks:
                picks.append(below[-1])
        picks.append(final)
        checkpoints, labels = [], []
        if rec.get("init_checkpoint") and ev.get("include_init", True):
            checkpoints.append(rec["init_checkpoint"])
            labels.append(ev.get("init_label", "init"))
        for c in picks:
            checkpoints.append(str(c))
            labels.append("final" if c == final else str(ckpt_iter(c)))
        # Comparison models in a call of their own: a model that fails to run must not take ours with it.
        compare = [(label, self.resolve(path)) for label, path in ev.get("compare", {}).items()]
        missing = [p for _, p in compare if not Path(p).exists()]
        if missing:
            rec["notes"].append(f"comparison models missing: {missing}")
        compare = [(label, p) for label, p in compare if Path(p).exists()]
        envs, seconds = (64, 2.0) if self.dry else (ev.get("num_envs", 1024), ev.get("seconds", 20.0))
        # Evaluation worlds are few, so a roomy constraint budget is cheap; other tasks can use more rows.
        eval_njmax = max(njmax, ev.get("njmax", 512))
        rec["evals"] = []
        for task in ev.get("tasks", [spec["task"]]):
            for seed in ev.get("seeds", [0]):
                for suffix, cks, lbs in (("", checkpoints, labels),
                                         ("_compare", [p for _, p in compare], [lb for lb, _ in compare])):
                    if not cks:
                        continue
                    out = res / f"eval_{task}_s{seed}{suffix}.json"
                    if self.run_logged(f"{sid}_eval_{task}_s{seed}{suffix}", [
                            "python", "scripts/eval_policy.py", "--task", task, "--checkpoints", *cks,
                            "--labels", *lbs, "--num-envs", str(envs), "--seconds", str(seconds), "--seed", str(seed),
                            "--njmax", str(eval_njmax), "--curriculum-step",
                            str(ev.get("curriculum_step", 240_000)), "--out", str(out)], 3600, env):
                        rec["evals"].append(str(out))
        venv = {**env, "MUJOCO_GL": "egl"}
        if not ac.contracts_of(spec["acceptance"]):
            # No contract, so no scenario to script: this rollout draws its commands at random and can miss the commanded
            # behaviour entirely (the SitStand video never sat down). Stages with a contract get a scripted video below.
            video = res / f"rollout_{ts()}.mp4"
            if self.run_logged(f"{sid}_video", ["python", "scripts/record_video.py", spec["task"], "--checkpoint",
                                                str(final), "--out", str(video),
                                                *(["--steps", "25"] if self.dry else [])], 1800, venv):
                rec["video"] = video.name
        self.accept(spec, rec, final, res, venv)

    def set_verdict(self, rec: dict, verdict: str, why: list[str], warnings=()) -> None:
        rec["verdict"], rec["verdict_why"], rec["verdict_warnings"] = verdict, why, list(warnings)
        self.log(f"{rec['id']}: acceptance {verdict}" + (f" — {'; '.join(why)}" if why else "")
                 + (f" (warnings: {'; '.join(warnings)})" if warnings else ""))

    def accept(self, spec, rec, final, res, env) -> None:
        """Run the stage's acceptance contracts on the exported policy; anything but a clean answer is UNVERIFIED."""
        sid = spec["id"]
        if not ac.contracts_of(spec["acceptance"]):
            return self.set_verdict(rec, "UNVERIFIED", ["no acceptance contract declared"])
        onnx = res / "policy.onnx"
        if not onnx.exists():
            return self.set_verdict(rec, "UNVERIFIED", ["no exported policy.onnx to check"])
        (res / "acceptance.json").unlink(missing_ok=True)  # an answer left by an earlier try must not be read as this one
        cmd = ["python", "scripts/acceptance.py", "--task", spec["task"], "--onnx", str(onnx), "--checkpoint", str(final),
               "--spec", json.dumps(spec["acceptance"]), "--out-dir", str(res), "--video",
               "--curriculum-step", str(spec.get("eval", {}).get("curriculum_step", 240_000)),
               *(["--num-envs", "32", "--no-noise-check"] if self.dry else [])]
        ran = self.run_logged(f"{sid}_acceptance", cmd, 7200, env)
        answer = res / "acceptance.json"
        if not (ran and answer.exists()):
            return self.set_verdict(rec, "UNVERIFIED", [f"the acceptance check did not finish; see {sid}_acceptance.log"])
        result = json.loads(answer.read_text())
        if result.get("video"):
            rec["video"] = result["video"]
        self.set_verdict(rec, result["verdict"], result["reasons"], result.get("warnings", []))

    def gate(self, spec, rec, res) -> dict:
        """Two-stage gate: falls after the fine-tune against before it, on each eval task, mean over seeds."""
        rules = spec["gate_rules"]  # {task: max ratio}
        out = {"passed": True, "tasks": {}}
        for task, ratio in rules.items():
            rows = eval_rows(rec, task)
            before = [r["falls_per_world_minute"] for r in rows if r["label"] == spec["eval"].get("init_label", "init")]
            after = [r["falls_per_world_minute"] for r in rows if r["label"] == "final"]
            if not before or not after:
                out["passed"] = False
                out["tasks"][task] = {"why": "no evaluation"}
                continue
            b, a = statistics.mean(before), statistics.mean(after)
            ok = a <= ratio * b + 0.01  # 0.01 falls per world-minute of slack for near-zero rates
            out["tasks"][task] = {"before": b, "after": a, "max_ratio": ratio, "ok": ok}
            out["passed"] &= ok
        rewards = {int(k): v for k, v in rec.get("rewards", {}).items()}
        its = sorted(rewards)
        if len(its) >= 100:
            first = statistics.mean(rewards[i] for i in its[:50])
            last = statistics.mean(rewards[i] for i in its[-100:])
            ok = last >= first - 0.2 * abs(first)
            out["reward"] = {"first_50": first, "last_100": last, "ok": ok}
            out["passed"] &= ok
        elif not self.dry:
            out["passed"] = False
            out["reward"] = {"why": f"only {len(its)} iterations logged"}
        if self.dry:  # a few iterations decide nothing; let the dry run reach the stages behind the gate
            out["dry_run_measured"] = out["passed"]
            out["passed"] = True
        return out

    # -- reports -----------------------------------------------------------------------------------
    def write_stage_readme(self, spec, rec, res):
        try:
            lines = stage_report(spec, rec, res, self.state["gate"].get(spec.get("gate", ""), None))
            (res / "README.md").write_text("\n".join(lines) + "\n")
        except Exception:
            rec["notes"].append("README failed: " + traceback.format_exc(limit=3).replace("\n", " | "))

    def write_summary(self):
        self.summary_dir.mkdir(parents=True, exist_ok=True)
        lines = [f"# Training queue `{self.state['queue']}`", "",
                 f"- Queue started {self.state['started']}; deadline {clock(self.deadline)}; this table as of {clock()}.",
                 f"- Queue file `{Path(self.state['queue_file']).name}`, logs `logs/queue/{self.qdir.name}/`.", ""]
        for name, g in self.state["gate"].items():
            lines.append(f"- Gate `{name}`: {'passed' if g.get('passed') else 'failed'} — `{json.dumps(g)}`")
        counts = {v: sum(r.get("verdict") == v for r in self.state["stages"].values()) for v in ac.VERDICTS}
        lines.append("- Verdicts: " + ", ".join(f"{n} {v}" for v, n in counts.items()) + ". ACCEPTED is automatic only; "
                     "full-motion video and human review are still required. UNVERIFIED means something was not checked.")
        lines += ["", "| stage | skill | task | status | verdict | iterations | wall time | last mean reward | "
                      "falls/world/min | upright | result |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for sid in self.state["order"]:
            r = self.state["stages"][sid]
            final = final_eval_row(r, r["task"])
            rw = last_mean(r)
            wall = "-"
            if r.get("ended") and r.get("started"):
                wall = f"{(datetime.strptime(r['ended'], '%Y-%m-%d %H:%M:%S') - datetime.strptime(r['started'], '%Y-%m-%d %H:%M:%S')).total_seconds() / 3600:.1f} h"
            its = "-" if r.get("last_iteration") is None else f"{r['last_iteration'] - r.get('start_iteration', 0) + 1}/{r.get('iters', '-')}"
            res = Path(r["res"]).name if r.get("res") else "-"
            link = f"[{res}](../{res}/README.md)" if r.get("res") else "-"
            lines.append(f"| {sid} | {r.get('skill', '-')} | `{r.get('task', '-')}` | {r['status']}"
                         f"{' (' + r['stop_reason'] + ')' if r.get('stop_reason') else ''} | "
                         f"{verdict_cell(r)} | {its} | {wall} | "
                         f"{'-' if rw is None else f'{rw:.2f}'} | {fmt(final, 'falls_per_world_minute', '.3f')} | "
                         f"{fmt(final, 'upright_share', '.1%')} | {link} |")
        (self.summary_dir / "README.md").write_text("\n".join(lines) + "\n")

    # -- main loop ---------------------------------------------------------------------------------
    def run(self):
        self.log(f"queue {self.state['queue']}: {len(self.spec['stages'])} stages, deadline {clock(self.deadline)}, "
                 f"state {self.qdir / 'state.json'}")
        for spec in self.spec["stages"]:
            sid = spec["id"]
            rec = self.state["stages"].get(sid, {})
            if rec.get("status") == "post":  # the queue died while post-processing this stage
                rec["status"] = "done" if rec.get("final_checkpoint") else "failed"
                rec.setdefault("notes", []).append("queue restarted during post-processing; evaluation may be missing")
                rec.setdefault("verdict", "UNVERIFIED")
                rec.setdefault("verdict_why", ["queue restarted during post-processing"])
            if rec.get("status") in ("done", "failed", "skipped", "deferred"):
                continue
            if self.args.only and sid not in self.args.only:
                continue
            try:
                stage = self.plan(spec)
                if "skip" in stage:
                    self.state["stages"][sid] = {"id": sid, "skill": spec["skill"], "task": spec["task"],
                                                 "status": "skipped", "stop_reason": stage["skip"]}
                    self.state["order"].append(sid)
                    self.log(f"{sid}: {stage['skip']}")
                    continue
                cur_end = stage.get("curriculum_end", 0)
                need = min(stage["iters"], cur_end + 2 * WINDOW) * stage.get("sec_per_iter", 0)
                if time.time() + need > self.deadline and self.state["stages"].get(sid, {}).get("status") != "running":
                    self.state["stages"][sid] = {"id": sid, "skill": spec["skill"], "task": stage["task"],
                                                 "status": "deferred",
                                                 "stop_reason": f"needs {need / 3600:.1f} h, deadline too close"}
                    self.state["order"].append(sid)
                    self.log(f"{sid}: deferred ({need / 3600:.1f} h needed)")
                    continue
                self.run_stage(stage)
            except Exception:
                rec = self.state["stages"].setdefault(sid, {"id": sid, "skill": spec["skill"], "task": spec["task"]})
                rec["status"] = "failed"
                rec.setdefault("notes", []).append("queue error: " + traceback.format_exc(limit=5).replace("\n", " | "))
                if sid not in self.state["order"]:
                    self.state["order"].append(sid)
                self.log(f"{sid}: queue error\n{traceback.format_exc()}")
            finally:
                self.save()
                try:
                    self.write_summary()
                except Exception:
                    self.log("summary failed\n" + traceback.format_exc())
        self.state["finished"] = clock()
        self.save()
        self.write_summary()
        (self.qdir / "done").touch()
        self.log("queue finished")


class ConsoleParser:
    """Reads rsl_rl's console incrementally: mean reward per (absolute) learning iteration."""

    def __init__(self, path: Path):
        self.path, self.pos, self.it = path, 0, None

    def read(self, rewards: dict) -> None:
        try:
            with open(self.path, "rb") as f:
                f.seek(self.pos)
                chunk = f.read()
        except FileNotFoundError:
            return
        end = chunk.rfind(b"\n")
        if end < 0:
            return
        self.pos += end + 1
        for line in ANSI.sub("", chunk[:end + 1].decode(errors="ignore")).splitlines():
            m = ITER_RE.search(line)
            if m:
                self.it = int(m.group(1))
                continue
            m = REWARD_RE.search(line)
            if m and self.it is not None:
                rewards[self.it] = float(m.group(1))


def converged(rewards: dict, at: int, start: int, cur_end: int) -> dict | None:
    if at - start < cur_end + 2 * WINDOW:
        return None
    last = [rewards[i] for i in range(at - WINDOW + 1, at + 1) if i in rewards]
    prev = [rewards[i] for i in range(at - 2 * WINDOW + 1, at - WINDOW + 1) if i in rewards]
    if len(last) < 0.8 * WINDOW or len(prev) < 0.8 * WINDOW:
        return None
    m1, m2 = statistics.mean(prev), statistics.mean(last)
    gain = (m2 - m1) / max(abs(m1), 1e-6)
    return {"at": at, "prev_mean": m1, "last_mean": m2, "gain": gain} if gain < MIN_GAIN else None


def stop_group(proc: subprocess.Popen) -> None:
    """SIGTERM the training's process group, SIGKILL after 30 s; it resumes from the last checkpoint."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    for _ in range(30):
        if proc.poll() is not None:
            return
        time.sleep(1)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def last_mean(rec: dict, n: int = 100):
    rewards = {int(k): v for k, v in rec.get("rewards", {}).items()}
    its = sorted(rewards)[-n:]
    return statistics.mean(rewards[i] for i in its) if its else None


def eval_rows(rec: dict, task: str) -> list[dict]:
    rows = []
    for p in rec.get("evals", []):
        m = EVAL_RE.match(Path(p).name)
        if m and m.group(1) == task and Path(p).exists():
            rows += json.loads(Path(p).read_text())["rows"]
    return rows


def final_eval_row(rec: dict, task: str) -> dict | None:
    rows = [r for r in eval_rows(rec, task) if r["label"] == "final"]
    if not rows:
        return None
    keys = ("falls_per_world_minute", "fell_share", "upright_share", "tip_overs_per_world_minute",
            "recoveries_per_world_minute")
    return {k: statistics.mean(r[k] for r in rows) for k in keys}


def fmt(row, key, spec) -> str:
    return "-" if not row or row.get(key) is None else format(row[key], spec)


def rel(text) -> str:
    """Paths under the repository root as relative paths, so reports read the same on every machine."""
    return str(text).replace(str(TRAIN) + "/", "")


def stage_report(spec, rec, res, gate) -> list[str]:
    init = rec.get("init", {})
    src = {"scratch": "from scratch", "warm": f"warm start from `{rel(rec.get('init_checkpoint', '-'))}` "
                      "(MICRODUCK_WARM_START=1: curricula and iteration restart)",
           "resume": f"resumed from `{rel(rec.get('init_checkpoint', '-'))}` (counters kept)"}[init.get("mode", "scratch")]
    lines = [f"# HD-1910 {rec['skill']}: `{rec['task']}`", "",
             f"- Queue stage {rec['id']}; {src}.",
             f"- Started {rec.get('started')}, ended {rec.get('ended')} ({rec.get('stop_reason')}); "
             f"iterations {rec.get('start_iteration', 0)}..{rec.get('last_iteration')} of a {rec.get('iters')}-iteration "
             f"budget at {rec.get('num_envs')} envs, njmax {rec.get('njmax')}"
             f"{', nconmax ' + str(rec['nconmax']) if rec.get('nconmax') else ''}.",
             "- Segments: " + ", ".join(f"`{Path(s).name}`" for s in rec.get("segments", [])) + "."]
    rw = {int(k): v for k, v in rec.get("rewards", {}).items()}
    if rw:
        its = sorted(rw)
        lines.append(f"- Mean reward: first 50 iterations {statistics.mean(rw[i] for i in its[:50]):.2f}, "
                     f"last 100 {statistics.mean(rw[i] for i in its[-100:]):.2f}.")
    if rec.get("convergence"):
        c = rec["convergence"]
        lines.append(f"- Converged at {c['at']}: mean reward of the last {WINDOW} iterations {c['last_mean']:.2f} against "
                     f"{c['prev_mean']:.2f} for the {WINDOW} before ({c['gain']:+.2%} < {MIN_GAIN:.0%}).")
    summary = json.loads((res / "summary.json").read_text()) if (res / "summary.json").exists() else {}
    if isinstance(summary.get("median_steps_per_s"), (int, float)):
        lines.append(f"- Median throughput {summary['median_steps_per_s']:.0f} env steps/s.")
    if gate is not None:
        lines.append(f"- Gate: {'passed' if gate.get('passed') else 'failed'} — `{json.dumps(gate)}`.")
    if rec.get("verdict"):
        reasons = rec.get("verdict_why") or []
        why = "; ".join(reasons[:3]) + (f" (and {len(reasons) - 3} more in acceptance.md)" if len(reasons) > 3 else "")
        lines.append(f"- Acceptance: **{rec['verdict']}**" + (f" — {why}" if why else "")
                     + (". Details: [acceptance.md](acceptance.md)." if (res / "acceptance.md").exists() else "."))
        if rec.get("verdict_warnings"):
            lines.append("- Acceptance warnings (they do not fail the stage): " + "; ".join(rec["verdict_warnings"]) + ".")
    for n in rec.get("notes", []):
        lines.append(f"- Note: {rel(n)}")
    lines += ["", "Curve: [curve.png](curve.png); ONNX, versions and commits: [summary.md](summary.md). "
              + (f"Video: [{rec['video']}]({rec['video']})." if rec.get("video") else "No video."), ""]
    tasks = []
    for p in rec.get("evals", []):
        t = EVAL_RE.match(Path(p).name).group(1)
        if t not in tasks:
            tasks.append(t)
    for task in tasks:
        rows = eval_rows(rec, task)
        seeds = len({EVAL_RE.match(Path(p).name).group(2) for p in rec["evals"]
                     if EVAL_RE.match(Path(p).name).group(1) == task})
        lines += [f"## Evaluation on `{task}`", "",
                  f"Actor mean, curricula at their final stage, {seeds} seed(s); mean over seeds. "
                  "Falls are terminations other than the time-out; upright is trunk tilt below 30 degrees, "
                  "a tip-over goes from upright to above 60 degrees.", "",
                  "| checkpoint | falls/world/min | fell share | upright | tip-overs/world/min | recoveries/world/min | "
                  "reward/step | |v_xy| err (m/s) |", "|---|---|---|---|---|---|---|---|"]
        labels = []
        for r in rows:
            if r["label"] not in labels:
                labels.append(r["label"])
        for label in labels:
            rs = [r for r in rows if r["label"] == label]
            m = lambda k: statistics.mean(r[k] for r in rs) if all(r.get(k) is not None for r in rs) else None  # noqa: E731
            rew = statistics.mean(sum(r["reward_per_step"].values()) for r in rs)
            lines.append(f"| {label} | {fmt({'v': m('falls_per_world_minute')}, 'v', '.3f')} | "
                         f"{fmt({'v': m('fell_share')}, 'v', '.1%')} | {fmt({'v': m('upright_share')}, 'v', '.1%')} | "
                         f"{fmt({'v': m('tip_overs_per_world_minute')}, 'v', '.3f')} | "
                         f"{fmt({'v': m('recoveries_per_world_minute')}, 'v', '.3f')} | {rew:.3f} | "
                         f"{fmt({'v': m('err_vxy_mps')}, 'v', '.3f')} |")
        final = [r for r in rows if r["label"] == "final"]
        if final:
            terms = final[0]["reward_per_step"]
            raw = final[0]["reward_raw_per_step"]
            lines += ["", "Final checkpoint, seed 0: reward terms per step (weighted / raw) and terminations "
                      "per world-minute.", "", "| reward term | weighted | raw |", "|---|---|---|"]
            for n in sorted(terms, key=lambda k: -abs(terms[k])):
                lines.append(f"| {n} | {terms[n]:.4f} | {fmt(raw, n, '.4f')} |")
            lines += ["", "| termination | per world-minute |", "|---|---|"]
            for n, v in final[0]["terminations_per_world_minute"].items():
                lines.append(f"| {n} | {v:.3f} |")
        lines.append("")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--queue", required=True, help="queue JSON")
    ap.add_argument("--state", help="state.json of an interrupted queue to continue")
    ap.add_argument("--only", nargs="+", help="run only these stage ids")
    ap.add_argument("--dry-run", action="store_true", help="64 envs, at most 6 iterations, results under the queue dir")
    ap.add_argument("--reviewed-upstream", help="exact upstream SHA reviewed before keeping the current pin")
    ap.add_argument("--upstream-review-note", help="reason for keeping the current pin / review report")
    ap.add_argument("--reviewed-policies", help="exact official policy-catalog revision reviewed")
    args = ap.parse_args()
    from check_upstream import check
    report = check(reviewed_sha=args.reviewed_upstream, note=args.upstream_review_note,
                   reviewed_policies=args.reviewed_policies)
    if not report["training_allowed"]:
        raise SystemExit("Upstream check needs review or failed; see logs/upstream-review/upstream.md. "
                         "No training was started.")
    Queue(args).run()


if __name__ == "__main__":
    main()
