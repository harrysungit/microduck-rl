"""Sit-stand gate: the exported ONNX in plain MuJoCo (CPU), on the training's physics with robotd's target chain.

The acceptance check of a sit-stand policy, judged the way Pollen's sit-stand task is judged:

- every posture command is an attempt. It succeeds when the trunk reaches the commanded posture within
  ``reach_s``, stays there to the end of the segment, the robot does not fall, and in the segment's last
  ``hold_window_s`` (the hold) the head does not stay off its commanded pose; the head never hits the ground
  (``head_impact_N``, about twice the robot's weight) and does not rest on it for ``head_rest_s`` once the posture is
  held. A fall or a broken simulation fails the attempt; nothing leaves the denominator. The first policy target fills the servo
  latency line, as the training's delay buffer does after a reset;
- light, brief head support during a transition is allowed: such contacts are recorded, not judged;
- every physics step (5 ms) records which non-foot bodies touch the ground and how hard: parts, start and end,
  peak normal force and impulse. Impact peaks are pointers for the video review, not a verdict;
- the official seated fall-back metrics (Pollen's docs/sitstand_policy.md): a seated start held for 30 s under
  seated head jerks, alone and with backward tip kicks; the share of time leaned back past 20 degrees, the share
  of scenes still leaned back at the end, the neck torque while leaned. Reported only, as Pollen reports them.

Scenes are fixed: four scenarios (stand_cycle, sit_cycle, stand_hold, sit_hold) of four 6 s segments, in a nominal
and a disturbed condition (the training's observation noise, latency, randomization and pushes), at two battery
voltages, 16 seeds each. The head command is zero, as robotd sends it in sit-stand; moving heads are the fall-back
battery's. Seated starts tilt at most seated_tilt_deg (the training spawns up to 8 degrees). A scene's random numbers
depend only on its id, so two policies meet the same starts and disturbances; the report records each scene's
initial-state hash so that a paired comparison (--reference) can check it.

Verdict, a stage automatic check: no fall, head failure or broken scene in the zero-tolerance conditions (nominal),
and each condition's success rate at least its threshold. PASSED, FAILED, or UNDECIDED while a threshold is unset.
The protocol and the criteria are in sitstand_gate.json; the report records its sha256. The verdict never replaces
the final review of full-motion videos or a robot test; --videos renders one scene per scenario by a fixed rule.

    python scripts/sitstand_gate.py --onnx policy.onnx --assets <sim2sim_assets dir> --out-dir <dir> [--videos]
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")  # headless rendering for --videos

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim2sim  # noqa: E402
from sim2sim import CONTROL_DT, Observer, Servos, build_model, robotd_chain, sample_params, seated_pose  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SPEC = Path(__file__).with_name("sitstand_gate.json")
NECK_JOINTS = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")  # the head_pose command's order
STANDING_Z = 0.125  # sim2sim's standing spawn
FORCE_EPS_N = 1e-6  # a contact that carries no normal force is a margin contact, not a touch
MERGE_GAP_S = CONTROL_DT  # a body that lets go for at most this long keeps one contact event
MAX_EVENTS = 300  # non-head contact events kept per scene; head events are always kept
# The posture levels of sim2sim: training-equivalent physics plus robotd's chain (ground contact, 5 ms, 30/50).
BASE_LEVEL = next(lv for lv in sim2sim.SITSTAND_LEVELS if lv.key == "L0-robotd")


# -- spec and scenes -----------------------------------------------------------------------------------------------
def load_spec(path: Path = SPEC) -> tuple[dict, str]:
    raw = Path(path).read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class Scene:
    kind: str  # posture | fallback
    scenario: str  # a posture scenario, or a fall-back variant
    condition: str  # nominal | disturbed
    vin: float
    seed: int
    head: bool  # random head commands

    @property
    def id(self) -> str:
        return f"{self.kind}/{self.scenario}/{self.condition}/{self.vin:.1f}V/{self.seed:03d}"


def posture_scenes(protocol: dict) -> list[Scene]:
    """Zero head command (robotd's in sit-stand); run_scene still drives one for a Scene built with head=True."""
    return [Scene("posture", name, cond, float(vin), seed, False)
            for name in protocol["scenarios"] for cond in protocol["conditions"]
            for vin in protocol["voltages"] for seed in range(protocol["seeds"])]


def fallback_scenes(protocol: dict) -> list[Scene]:
    fb, volts = protocol["fallback"], protocol["voltages"]
    return [Scene("fallback", variant, "nominal", float(volts[i % len(volts)]), i, True)
            for variant in fb["variants"] for i in range(fb["scenes_per_variant"])]


def streams(scene_id: str) -> tuple[np.random.Generator, np.random.Generator, np.random.Generator]:
    """Setup (randomization, start), commands (head commands, fall-back events) and runtime (noise, latency, pushes):
    seeded by the scene id alone, so every policy meets the same draws."""
    root = np.random.SeedSequence(int(hashlib.sha256(scene_id.encode()).hexdigest()[:16], 16))
    return tuple(np.random.default_rng(s) for s in root.spawn(3))


def start_of(scene: Scene, protocol: dict) -> str:
    return "seated" if scene.kind == "fallback" else protocol["scenarios"][scene.scenario]["start"]


def level_for(scene: Scene, protocol: dict) -> sim2sim.Level:
    level = replace(BASE_LEVEL, key=scene.id, title=protocol["physics"], vin=scene.vin, episodes=1,
                    start=start_of(scene, protocol))
    if scene.condition == "disturbed":
        d = protocol["disturbed"]
        level = replace(level, randomize=True, latency_ms=tuple(d["latency_ms"]), push_mps=float(d["push_mps"]))
    return level


def postures_of(scene: Scene, protocol: dict) -> list[str]:
    if scene.kind == "fallback":
        return ["sit"]
    return list(protocol["scenarios"][scene.scenario]["postures"])


def duration_of(scene: Scene, protocol: dict) -> float:
    if scene.kind == "fallback":
        return float(protocol["fallback"]["seconds"])
    return protocol["segment_s"] * len(postures_of(scene, protocol))


# -- policy and model ----------------------------------------------------------------------------------------------
class GatePolicy(sim2sim.Policy):
    """sim2sim's Policy on one thread: the gate runs many scenes side by side."""

    def __init__(self, path: Path):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
        meta = self.session.get_modelmeta().custom_metadata_map
        self.joint_names = meta["joint_names"].split(",")
        self.home = np.array([float(v) for v in meta["default_joint_pos"].split(",")])
        self.scale = float(meta.get("action_scale", "1.0"))
        self.input = self.session.get_inputs()[0].name


def neck_subtree(model: mujoco.MjModel) -> set[int]:
    """The head as the robot is built: every body under `neck` by parent links (not a list of names)."""
    root = model.body("neck").id
    out = set()
    for body in range(1, model.nbody):
        a = body
        while a and a != root:
            a = int(model.body_parentid[a])
        if a == root:
            out.add(body)
    return out


def foot_bodies(model: mujoco.MjModel) -> set[int]:
    return {int(model.geom_bodyid[model.geom(g).id]) for g in ("left_foot_collision", "right_foot_collision")}


class ContactAudit:
    """Ground contacts of every non-foot body at every physics step, with the solved normal force.

    A body's contact is one event from its first loaded step until it lets go for longer than MERGE_GAP_S: start,
    end, peak normal force and impulse. The scene's impact peak is the largest summed non-foot normal force.
    """

    def __init__(self, body_names: list[str], head: set[int], feet: set[int]):
        self.names, self.head, self.feet = body_names, head, feet
        self.open: dict[int, dict] = {}
        self.events: list[dict] = []
        self.peak_n, self.peak_t, self.peak_bodies = 0.0, None, []
        self._force = np.zeros(6)

    def sample(self, model: mujoco.MjModel, data: mujoco.MjData, t: float, dt: float) -> None:
        """After mj_step: its contacts and forces belong to the state at t, the time before the step."""
        floor = model.geom("floor").id
        loads: dict[int, float] = {}
        for c in range(data.ncon):
            con = data.contact[c]
            g1, g2 = int(con.geom1), int(con.geom2)
            if floor not in (g1, g2):
                continue
            body = int(model.geom_bodyid[g2 if g1 == floor else g1])
            if body in self.feet:
                continue
            mujoco.mj_contactForce(model, data, c, self._force)
            if self._force[0] > FORCE_EPS_N:
                loads[body] = loads.get(body, 0.0) + float(self._force[0])
        self.step(t, dt, loads)

    def step(self, t: float, dt: float, loads: dict[int, float]) -> None:
        t = round(t, 6)  # data.time sums 5 ms steps (18.78000000000055); the control clock is k * 20 ms (18.78)
        total = sum(loads.values())
        if total > self.peak_n:
            self.peak_n, self.peak_t = total, t
            self.peak_bodies = sorted(self.names[b] for b in loads)
        for body, f in loads.items():
            ev = self.open.get(body)
            if ev is None:
                ev = self.open[body] = {"body": self.names[body], "head": body in self.head, "start_s": t,
                                        "end_s": round(t + dt, 6), "peak_N": 0.0, "impulse_Ns": 0.0}
            ev["end_s"], ev["peak_N"] = round(t + dt, 6), max(ev["peak_N"], f)
            ev["impulse_Ns"] += f * dt
        for body in [b for b, ev in self.open.items() if b not in loads and t + dt - ev["end_s"] > MERGE_GAP_S + 1e-9]:
            self._close(body)

    def _close(self, body: int) -> None:
        ev = self.open.pop(body)
        ev["duration_s"] = ev["end_s"] - ev["start_s"]
        self.events.append(ev)

    def finish(self) -> list[dict]:
        for body in list(self.open):
            self._close(body)
        return sorted(self.events, key=lambda e: (e["start_s"], e["body"]))


def params_for(scene: Scene, protocol: dict, level: sim2sim.Level, rng: np.random.Generator, n: int) -> dict:
    """sim2sim's draws, then the scene's own battery and push: 0 in the spec means no pushes (sample_params reads a
    zero push as "unset" and pushes randomized levels at 0.3 m/s)."""
    p = sample_params(level, rng, n)
    p["vin"] = scene.vin  # fixed per scene; the torque cap still follows the task's 7.0-8.4 V range
    p["push_mps"] = float(protocol["disturbed"]["push_mps"]) if scene.condition == "disturbed" else 0.0
    p["pushes"] = p["push_mps"] > 0.0
    return p


def send_target(queue: sim2sim.CommandQueue, first: bool, t: float, target: np.ndarray) -> None:
    """A scene's first target fills the latency line, as mjlab's DelayBuffer backfills its history with the first
    command after a reset: the servos of a seated start never see HOME, the standing pose."""
    if first:
        queue.reset(target)
    else:
        queue.send(t, target)


class SeatedPerturbations:
    """The fall-back battery's head command and kicks, as the training's seated events fire them: the head command
    resampled in the live range, jerks (a sign flip or a fresh sample) and, with tips, backward pitch kicks. The
    battery's scenes are seated throughout, so every event applies."""

    def __init__(self, fb: dict, rng: np.random.Generator, draw_head, tips: bool):
        self.fb, self.rng, self.draw_head, self.tips = fb, rng, draw_head, tips
        self.next_resample = rng.uniform(*fb["head_resample_s"])
        self.next_jerk = rng.uniform(*fb["jerk_interval_s"])
        self.next_tip = rng.uniform(*fb["tip_interval_s"])
        self.counts = {"head_resamples": 0, "jerks": 0, "tips": 0}

    def step(self, t: float, head_cmd: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        fb, rng = self.fb, self.rng
        if t >= self.next_resample:
            head_cmd = self.draw_head()
            self.counts["head_resamples"] += 1
            self.next_resample += rng.uniform(*fb["head_resample_s"])
        if t >= self.next_jerk:
            if rng.random() < fb["jerk_prob"]:
                head_cmd = -head_cmd if rng.random() < fb["jerk_flip_frac"] else self.draw_head()
                self.counts["jerks"] += 1
            self.next_jerk += rng.uniform(*fb["jerk_interval_s"])
        if self.tips and t >= self.next_tip:
            if rng.random() < fb["tip_prob"]:
                mag = rng.uniform(*fb["tip_ang_vel_rad_s"])
                roll = rng.uniform(-1.0, 1.0) * fb["tip_lateral_frac"]
                qvel[3:6] += (roll * mag, -mag, 0.0)  # free joint: body-frame angular velocity; -y is nose up
                self.counts["tips"] += 1
            self.next_tip += rng.uniform(*fb["tip_interval_s"])
        return head_cmd


# -- one scene -----------------------------------------------------------------------------------------------------
def run_scene(scene: Scene, spec: dict, onnx: Path, assets: Path, servo_params: Path, policy=None,
              render: dict | None = None, keep_traces: bool = False) -> dict:
    try:
        return _run_scene(scene, spec, onnx, assets, servo_params, policy, render, keep_traces)
    except Exception as exc:  # noqa: BLE001 - a broken scene fails its attempts; it never leaves the denominator
        return {"id": scene.id, "scene": vars(scene), "error": repr(exc)}


def _run_scene(scene, spec, onnx, assets, servo_params, policy, render, keep_traces):
    protocol, criteria = spec["protocol"], spec["criteria"]
    setup, commands, runtime = streams(scene.id)
    policy = policy or GatePolicy(onnx)
    names, n = policy.joint_names, len(policy.joint_names)
    neck = [names.index(j) for j in NECK_JOINTS]
    level = level_for(scene, protocol)
    p = params_for(scene, protocol, level, setup, n)
    model = build_model(level, assets, names, servo_params, p)
    data = mujoco.MjData(model)
    servos = Servos(model, names, servo_params, p, level.latency_ms, runtime)
    observer = Observer(model, servos, policy.home, level, p, runtime)
    trunk = model.body("trunk_base").id

    if level.start == "seated":
        assert names[3] == "left_knee" and names[12] == "right_knee", "SIT_KEYFRAME needs the HD-1910 joint order"
        z, quat, joints = seated_pose(policy.home, setup, tilt=math.radians(protocol["seated_tilt_deg"]))
    else:
        z, quat, joints = STANDING_Z, (1.0, 0.0, 0.0, 0.0), policy.home.copy()
    data.qpos[0:3], data.qpos[3:7], data.qpos[servos.qadr] = (0.0, 0.0, z), quat, joints
    mujoco.mj_forward(model, data)
    init_hash = state_hash(data, p)
    servos.reset(data, policy.home.copy())
    observer.reset(data)

    postures = postures_of(scene, protocol)
    seg_s = protocol["segment_s"] if scene.kind == "posture" else duration_of(scene, protocol)
    head_range = np.array(protocol["head_command_range_rad"], dtype=float)
    draw_head = lambda: commands.uniform(head_range[:, 0], head_range[:, 1])  # noqa: E731
    seg_heads = [draw_head() if scene.head else np.zeros(4) for _ in postures]
    head_cmd = seg_heads[0].copy()
    perturb = SeatedPerturbations(protocol["fallback"], commands, draw_head,
                                  tips=scene.kind == "fallback" and scene.scenario == "head_jerks_and_tips")
    pushes = 0

    action_scale = policy.scale * float(robotd_chain.voltage_scale(np.float64(p["vin"])))
    alpha = np.array(robotd_chain.joint_alphas(names))
    audit = ContactAudit([model.body(b).name for b in range(model.nbody)], neck_subtree(model), foot_bodies(model))
    substeps = round(CONTROL_DT / model.opt.timestep)
    ticks = round(duration_of(scene, protocol) / CONTROL_DT)
    tr = {k: np.full(ticks, np.nan) for k in ("t", "flag", "z", "tilt", "gx", "yaw", "neck_tau")}
    tr["head_q"], tr["head_cmd"] = np.full((ticks, 4), np.nan), np.full((ticks, 4), np.nan)
    next_push = runtime.uniform(3.0, 6.0)
    last_action, filtered = np.zeros(n), None
    fell_at, error, done = None, None, 0
    renderer = Renderer(model, trunk, render) if render else None
    for k in range(ticks):
        t0 = k * CONTROL_DT
        seg = min(int(t0 // seg_s + 1e-9), len(postures) - 1)
        flag = 1.0 if postures[seg] == "sit" else 0.0
        if scene.kind == "posture":
            head_cmd = seg_heads[seg]
        else:  # the training's head command, plus the seated fall-back perturbations (always seated here)
            head_cmd = perturb.step(t0, head_cmd, data.qvel)
        obs = observer(data, last_action, np.array([flag, 0.0, 0.0]))
        obs[-10:-6] = head_cmd  # [twist(3), head_pose(4), body_pose(6)]: the head command slot
        action = policy(obs)
        target = robotd_chain.targets(action, policy.home, action_scale,
                                      policy.home if filtered is None else filtered, np.array([filtered is None]), alpha)
        filtered = target
        send_target(servos.queue, k == 0, data.time, target)
        if p["pushes"] and data.time >= next_push:
            data.qvel[0:2] += runtime.uniform(-p["push_mps"], p["push_mps"], 2)
            next_push += runtime.uniform(3.0, 6.0)
            pushes += 1
        tau = 0.0
        for _ in range(substeps):
            t_before = data.time
            torque = servos.step(model, data)
            mujoco.mj_step(model, data)
            audit.sample(model, data, t_before, model.opt.timestep)
            tau += float(np.abs(torque[neck]).mean()) / substeps
            if renderer:
                renderer.substep(data, t_before)
        mujoco.mj_forward(model, data)
        if not (np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))):
            error = f"non-finite state at {t0 + CONTROL_DT:.2f} s"
            break
        rot = data.xmat[trunk].reshape(3, 3)
        g = rot.T @ np.array([0.0, 0.0, -1.0])
        tilt = math.degrees(math.acos(np.clip(rot[2, 2], -1.0, 1.0)))
        q_head = data.qpos[servos.qadr[neck]] + servos.bl_mask[neck] * data.qpos[servos.bl_qadr[neck]]
        for key, value in (("t", t0 + CONTROL_DT), ("flag", flag), ("z", data.qpos[2]), ("tilt", tilt), ("gx", g[0]),
                           ("yaw", math.degrees(math.atan2(rot[1, 0], rot[0, 0]))), ("neck_tau", tau)):
            tr[key][k] = value
        tr["head_q"][k], tr["head_cmd"][k] = q_head - policy.home[neck], head_cmd
        last_action = action
        done = k + 1
        if renderer:
            renderer.tick(data)
        if tilt > criteria["fell_deg"] and fell_at is None:
            fell_at = t0 + CONTROL_DT
    if renderer:
        renderer.close()
    tr = {k: v[:done] for k, v in tr.items()}
    contacts = audit.finish()
    result = {"id": scene.id, "scene": vars(scene), "init_hash": init_hash, "vin": p["vin"], "fell_at_s": fell_at,
              "error": error, "events": {"pushes": pushes, **perturb.counts},
              "impact_peak_N": audit.peak_n, "impact_peak_s": audit.peak_t,
              "impact_peak_bodies": audit.peak_bodies}
    head_events = [e for e in contacts if e["head"]]
    other = [e for e in contacts if not e["head"]]
    result["contacts"] = head_events + sorted(other, key=lambda e: -e["peak_N"])[:MAX_EVENTS]
    result["contacts_dropped"] = max(0, len(other) - MAX_EVENTS)
    start_posture = "sit" if level.start == "seated" else "stand"
    if scene.kind == "posture":
        result["attempts"] = score_attempts(tr, postures, seg_s, start_posture, head_events, fell_at, error, criteria)
    else:
        result["fallback"] = fallback_metrics(tr, protocol["fallback"]["lean_deg"])
        result["fallback"]["error"] = error
    if keep_traces:
        result["traces"] = {k: v.tolist() for k, v in tr.items()}
    return result


def state_hash(data: mujoco.MjData, p: dict) -> str:
    """The scene's start: the robot's state and every randomized parameter."""
    h = hashlib.sha256(np.concatenate([data.qpos, data.qvel]).astype(np.float64).tobytes())
    for key in sorted(p):
        value = p[key]
        if isinstance(value, dict):
            value = [np.asarray(value[k], dtype=float) for k in sorted(value)]
        h.update(f"{key}={np.asarray(value, dtype=object if value is None else float).tolist()!r};".encode())
    return h.hexdigest()[:16]


# -- scoring -------------------------------------------------------------------------------------------------------
def score_attempts(tr: dict, postures: list[str], seg_s: float, start_posture: str, head_events: list[dict],
                   fell_at: float | None, error: str | None, criteria: dict) -> list[dict]:
    """One attempt per segment: reached in time, held to the end, no fall; no head impact (a head contact of at least
    head_impact_N) anywhere in the segment; no head resting on the ground for head_rest_s once the posture is held;
    the head on its command in the last hold_window_s. Lighter, shorter head support before the arrival is allowed."""
    t, z, tilt = (np.asarray(tr[k], dtype=float) for k in ("t", "z", "tilt"))
    dev = np.abs(np.asarray(tr["head_q"], dtype=float) - np.asarray(tr["head_cmd"], dtype=float)).reshape(len(t), 4)
    off = dev.max(axis=1) > criteria["head_deviation_rad"] if len(t) else np.zeros(0, bool)
    yaw = np.degrees(np.unwrap(np.radians(np.asarray(tr["yaw"], dtype=float)))) if len(t) else np.zeros(0)
    hold = criteria["hold_window_s"]
    out, previous = [], start_posture
    for i, posture in enumerate(postures):
        start, end = i * seg_s, (i + 1) * seg_s
        sel = (t > start + 1e-9) & (t <= end + 1e-9)
        row = {"segment": i, "posture": posture, "transition": "hold" if posture == previous else
               ("sit-down" if posture == "sit" else "stand-up"), "settled_s": None, "reasons": []}
        complete = sel.any() and t[sel][-1] >= end - 1e-6
        if error is not None and not complete:
            row["reasons"].append("error")
        if fell_at is not None and fell_at <= end + 1e-9:
            row["reasons"].append("fell")
        inside = [e for e in head_events if e["start_s"] < end - 1e-6 and e["end_s"] > start + 1e-6]
        hits = [e for e in inside if e["peak_N"] >= criteria["head_impact_N"]]
        row["head_impacts"] = [{k: e[k] for k in ("body", "start_s", "end_s", "peak_N")} for e in hits]
        if hits:
            row["reasons"].append("head_impact")
        if sel.any():
            good = ((z <= criteria["sit_z_max_m"]) if posture == "sit" else (z >= criteria["stand_z_min_m"])) & \
                   (tilt < criteria["upright_deg"])
            seg_t, seg_good = t[sel], good[sel]
            bad = np.flatnonzero(~seg_good)
            if bad.size == 0:
                row["settled_s"] = 0.0
            elif bad[-1] < len(seg_good) - 1:
                row["settled_s"] = float(seg_t[bad[-1] + 1] - start)
            win = sel & (t > end - hold + 1e-9)
            row["head_off_frac"] = float(off[win].mean()) if win.any() else None
            row["head_max_dev_rad"] = float(dev[win].max()) if win.any() else None
            idx = np.flatnonzero(sel)
            row["yaw_change_deg"] = float(yaw[idx[-1]] - yaw[max(idx[0] - 1, 0)])
        if complete:
            if row["settled_s"] is None:
                row["reasons"].append("not_in_posture_at_end")
            elif row["settled_s"] > criteria["reach_s"]:
                row["reasons"].append("reached_late")
            since = held_from(row, start, end, hold)
            row["held_from_s"] = since
            rest = [e for e in inside if min(e["end_s"], end) - max(e["start_s"], since) >= criteria["head_rest_s"] - 1e-6]
            row["head_rest_in_hold"] = [{k: e[k] for k in ("body", "start_s", "end_s", "peak_N")} for e in rest]
            if rest:
                row["reasons"].append("head_rest_in_hold")
            if row.get("head_off_frac") is not None and row["head_off_frac"] > criteria["head_deviation_frac_max"]:
                row["reasons"].append("head_off_command_in_hold")
        row["success"] = not row["reasons"]
        out.append(row)
        previous = posture
    return out


def held_from(row: dict, start: float, end: float, hold: float) -> float:
    """Where the posture is held: the whole segment when the posture does not change, else from the arrival (the
    last hold window if it never arrived). Head support before it is transition support."""
    if row["transition"] == "hold":
        return start
    return start + row["settled_s"] if row["settled_s"] is not None else end - hold


def fallback_metrics(tr: dict, lean_deg: float) -> dict:
    """Leaned back: the trunk tilted past lean_deg with its nose up (projected gravity x < 0, the training's sign)."""
    tilt, gx, tau = (np.asarray(tr[k], dtype=float) for k in ("tilt", "gx", "neck_tau"))
    leaned = (gx < 0.0) & (tilt > lean_deg)
    return {"ticks": int(len(tilt)), "leaned_ticks": int(leaned.sum()),
            "leaned_at_end": bool(leaned[-1]) if len(leaned) else None,
            "neck_tau_leaned_sum_Nm": float(tau[leaned].sum()), "max_tilt_deg": float(tilt.max()) if len(tilt) else None}


HEAD_REASONS = ("head_impact", "head_rest_in_hold", "head_off_command_in_hold")


def summarize(results: list[dict], spec: dict) -> dict:
    protocol = spec["protocol"]
    posture = [r for r in results if r["scene"]["kind"] == "posture"]
    expected = {s: len(protocol["scenarios"][s]["postures"]) for s in protocol["scenarios"]}
    cells, reasons, by_transition = {}, {}, {}
    falls = {c: [] for c in protocol["conditions"]}
    head_failures = {c: [] for c in protocol["conditions"]}
    head_assist = []
    for r in posture:
        sc = r["scene"]
        attempts = r.get("attempts") or [{"success": False, "reasons": ["error"], "transition": "?"}] * expected[sc["scenario"]]
        cell = cells.setdefault(f"{sc['scenario']}/{sc['condition']}", {"attempts": 0, "successes": 0})
        cell["attempts"] += len(attempts)
        cell["successes"] += sum(a["success"] for a in attempts)
        for a in attempts:
            tt = by_transition.setdefault(f"{a['transition']}/{sc['condition']}", {"attempts": 0, "successes": 0})
            tt["attempts"] += 1
            tt["successes"] += a["success"]
            for why in a["reasons"]:
                reasons[why] = reasons.get(why, 0) + 1
            if any(why in HEAD_REASONS for why in a["reasons"]):
                head_failures[sc["condition"]].append(f"{r['id']}#{a.get('segment')}")
        if r.get("fell_at_s") is not None:
            falls[sc["condition"]].append(r["id"])
        judged = {(c["start_s"], c["body"]) for a in attempts
                  for c in a.get("head_impacts", []) + a.get("head_rest_in_hold", [])}
        head_assist += [dict(c, id=r["id"]) for c in r.get("contacts", [])
                        if c["head"] and (c["start_s"], c["body"]) not in judged]
    for c in list(cells.values()) + list(by_transition.values()):
        c["rate"] = c["successes"] / c["attempts"] if c["attempts"] else None
    conditions = {}
    for cond in protocol["conditions"]:
        a = sum(c["attempts"] for k, c in cells.items() if k.endswith("/" + cond))
        s = sum(c["successes"] for k, c in cells.items() if k.endswith("/" + cond))
        conditions[cond] = {"attempts": a, "successes": s, "rate": s / a if a else None}
    impacts = {}
    for name in protocol["scenarios"]:
        rows = sorted(((r.get("impact_peak_N") or 0.0, r["id"], r.get("impact_peak_s"), r.get("impact_peak_bodies"))
                       for r in posture if r["scene"]["scenario"] == name), reverse=True)
        values = [v for v, *_ in rows]
        impacts[name] = {"median_N": float(np.median(values)) if values else None,
                         "max_N": values[0] if values else None,
                         "top": [{"id": i, "peak_N": v, "t_s": t, "bodies": b} for v, i, t, b in rows[:5]]}
    fallback = {}
    for variant in protocol["fallback"]["variants"]:
        rows = [r for r in results if r["scene"]["kind"] == "fallback" and r["scene"]["scenario"] == variant]
        ok = [r["fallback"] for r in rows if r.get("fallback") and not r["fallback"].get("error")]
        ticks, leaned = sum(m["ticks"] for m in ok), sum(m["leaned_ticks"] for m in ok)
        fallback[variant] = {
            "scenes": len(rows), "broken": len(rows) - len(ok),
            "time_leaned_back": leaned / ticks if ticks else None,
            "leaned_at_end": sum(bool(m["leaned_at_end"]) for m in ok) / len(ok) if ok else None,
            "neck_tau_while_leaned_Nm": sum(m["neck_tau_leaned_sum_Nm"] for m in ok) / leaned if leaned else None,
            "falls": sum(r.get("fell_at_s") is not None for r in rows),
            "head_contact_scenes": sum(any(c["head"] for c in r.get("contacts", [])) for r in rows)}
    errors = {c: [] for c in protocol["conditions"]}
    errors["fallback"] = []
    for r in results:
        if r.get("error"):
            errors["fallback" if r["scene"]["kind"] == "fallback" else r["scene"]["condition"]].append(r["id"])
    return {"cells": cells, "conditions": conditions, "by_transition": by_transition, "reasons": reasons,
            "falls": falls, "head_failures": head_failures, "broken_scenes": errors,
            "head_support": {"events": len(head_assist), "scenes": len({c["id"] for c in head_assist}),
                                           "total_s": float(sum(c["duration_s"] for c in head_assist)),
                                           "peak_N": max((c["peak_N"] for c in head_assist), default=0.0)},
            "impacts": impacts, "fallback": fallback}


def paired(results: list[dict], reference: dict, spec_sha256: str, overrides: dict) -> dict:
    """Scene by scene against another policy's gate report: same spec, same starts, and where one succeeds and the
    other not. A reference run under another spec or with overrides is not comparable; spec_matches says so."""
    ref = {r["id"]: r for r in reference["scenes"]}
    rp = reference["provenance"]
    mismatched, worse, better, common = [], [], [], 0
    for r in results:
        o = ref.get(r["id"])
        if o is None or r["scene"]["kind"] != "posture":
            continue
        common += 1
        if r.get("init_hash") != o.get("init_hash"):
            mismatched.append(r["id"])
        a = [x["success"] for x in r.get("attempts") or []]
        b = [x["success"] for x in o.get("attempts") or []]
        for i, (x, y) in enumerate(zip(a, b)):
            (worse if y and not x else better if x and not y else []).append(f"{r['id']}#{i}")
    return {"reference_onnx_sha256": rp["onnx_sha256"], "reference_spec_sha256": rp["spec_sha256"],
            "reference_overrides": rp.get("overrides") or {},
            "spec_matches": rp["spec_sha256"] == spec_sha256 and not rp.get("overrides") and not overrides,
            "common_scenes": common, "init_hash_mismatches": mismatched, "reference_succeeds_this_fails": worse,
            "this_succeeds_reference_fails": better,
            "reference_conditions": reference["summary"]["conditions"]}


def verdict(summary: dict, criteria: dict, pairing: dict | None) -> tuple[str, list[str]]:
    """Zero tolerance (falls, head failures, broken scenes) in the listed conditions; elsewhere those
    count against the success rate only, and the fall-back battery is reported only. Each condition's success rate
    against its own threshold."""
    fail, open_ = [], []
    for cond in criteria["zero_tolerance_conditions"]:
        falls, head = summary["falls"].get(cond, []), summary["head_failures"].get(cond, [])
        if len(falls) > criteria["max_falls"]:
            fail.append(f"{cond}: {len(falls)} scenes fell (max {criteria['max_falls']})")
        if len(head) > criteria["max_head_failures"]:
            fail.append(f"{cond}: {len(head)} attempts failed the head check (max {criteria['max_head_failures']})")
        if summary["broken_scenes"].get(cond):
            fail.append(f"{cond}: {len(summary['broken_scenes'][cond])} scenes broke")
    thresholds = criteria["min_success_rate"]
    unset = [cond for cond in summary["conditions"] if thresholds.get(cond) is None]
    for cond, c in summary["conditions"].items():
        threshold = thresholds.get(cond)
        if threshold is not None and (c["rate"] is None or c["rate"] < threshold):
            fail.append(f"{cond}: success {c['successes']}/{c['attempts']} below {threshold:.0%}")
    if criteria.get("not_below_reference"):
        if pairing is None:
            open_.append("the criteria compare against a reference policy; run with --reference")
        elif not pairing["spec_matches"]:
            fail.append("the reference ran another spec or overrides; it cannot be compared")
        else:
            if pairing["init_hash_mismatches"]:
                fail.append(f"{len(pairing['init_hash_mismatches'])} scenes start differently from the reference")
            for cond, c in summary["conditions"].items():
                ref = pairing["reference_conditions"].get(cond, {})
                if ref.get("rate") is not None and c["rate"] is not None and c["rate"] < ref["rate"]:
                    fail.append(f"{cond}: success {c['rate']:.1%} below the reference's {ref['rate']:.1%}")
    if unset:
        open_.append(f"no success-rate threshold is set for {', '.join(unset)} (criteria.min_success_rate)")
    broken_elsewhere = sum(len(v) for k, v in summary["broken_scenes"].items()
                           if k not in criteria["zero_tolerance_conditions"])
    if broken_elsewhere:  # counted against the rate or reported only, but a broken tool must not read as a pass
        open_.append(f"{broken_elsewhere} scenes outside {', '.join(criteria['zero_tolerance_conditions'])} broke; "
                     "check them before trusting the report")
    if fail:
        return "FAILED", fail + open_
    return ("UNDECIDED", open_) if open_ else ("PASSED", [])


# -- videos --------------------------------------------------------------------------------------------------------
def pick_videos(results: list[dict], protocol: dict) -> list[dict]:
    """One nominal scene per scenario, by a fixed rule: the first failed scene (lowest seed, then voltage), else the
    scene with the largest impact peak. A 4x slow-motion clip around the first head contact, else the impact peak."""
    picks = []
    for name in protocol["scenarios"]:
        rows = [r for r in results if r["scene"]["kind"] == "posture" and r["scene"]["scenario"] == name
                and r["scene"]["condition"] == "nominal"]
        failed = sorted((r for r in rows if not all(a["success"] for a in r.get("attempts") or [{"success": False}])),
                        key=lambda r: (r["scene"]["seed"], r["scene"]["vin"]))
        if failed:
            r, why = failed[0], "first failed scene"
        else:
            r, why = max(rows, key=lambda r: r.get("impact_peak_N") or 0.0), "largest impact peak (no failure)"
        heads = [c for c in r.get("contacts", []) if c["head"]]
        center = heads[0]["start_s"] if heads else r.get("impact_peak_s")
        picks.append({"scenario": name, "id": r["id"], "rule": why,
                      "slow_center_s": center, "slow_reason": "first head contact" if heads else "impact peak"})
    return picks


class Renderer:
    """Full scene at normal speed (one frame per 20 ms tick, 50 fps); a slow clip at 4x (one frame per 5 ms step)."""

    def __init__(self, model: mujoco.MjModel, trunk: int, render: dict, height: int = 540, width: int = 960):
        self.r = mujoco.Renderer(model, height=height, width=width)
        self.cam = mujoco.MjvCamera()
        self.cam.type, self.cam.trackbodyid = mujoco.mjtCamera.mjCAMERA_TRACKING, trunk
        self.cam.distance, self.cam.elevation, self.cam.azimuth = 0.7, -12.0, 135.0
        self.window = render.get("slow_window")
        self.full = self._ffmpeg(render["full"], width, height)
        self.slow = self._ffmpeg(render["slow"], width, height) if self.window and render.get("slow") else None

    @staticmethod
    def _ffmpeg(path, width, height):
        return subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                 "-s", f"{width}x{height}", "-r", str(round(1 / CONTROL_DT)), "-i", "-",
                                 "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)

    def _frame(self, data):
        self.r.update_scene(data, self.cam)
        return self.r.render().tobytes()

    def substep(self, data, t):
        if self.slow and self.window[0] <= t <= self.window[1]:
            self.slow.stdin.write(self._frame(data))

    def tick(self, data):
        self.full.stdin.write(self._frame(data))

    def close(self):
        for proc in (self.full, self.slow):
            if proc:
                proc.stdin.close()
                proc.wait()


# -- report --------------------------------------------------------------------------------------------------------
def _pct(c: dict) -> str:
    return "-" if not c or c.get("rate") is None else f"{c['successes']}/{c['attempts']} ({c['rate']:.1%})"


def write_markdown(out: Path, report: dict) -> None:
    pv, sm, spec = report["provenance"], report["summary"], report["spec"]
    pr, cr = spec["protocol"], spec["criteria"]
    conds = pr["conditions"]
    lines = [f"# Sit-stand gate: {pv['title']}", "",
             f"**Stage automatic check: {report['verdict']}**" + (": " + "; ".join(report["verdict_reasons"])
                                                                   if report["verdict_reasons"] else ""), "",
             "Not a robot candidate on its own: the final check is a review of full-motion videos and a robot test.", "",
             f"- Policy: `{pv['onnx']}` (sha256 `{pv['onnx_sha256']}`)",
             f"- Spec: `{pv['spec']}` (sha256 `{pv['spec_sha256']}`){' — PROTOCOL OVERRIDDEN, not a gate run' if pv.get('overrides') else ''}",
             f"- Physics: {pr['physics']}. Scenes: `{pv['assets']}` (all files sha256 "
             f"`{pv.get('assets_sha256', '?')[:16]}…`, compiled model `{pv.get('model_sha256', '?')[:16]}…`); "
             f"servo model sha256 `{pv['servo_sha256'][:16]}…`",
             f"- Code: {pv['code']}; {pv['scenes']} scenes in {pv['elapsed_s']:.0f} s on {pv['workers']} workers", "",
             "## Attempts", "",
             f"Each posture command is an attempt: reached within {cr['reach_s']:g} s, held to the end of the "
             f"{pr['segment_s']:g} s segment, no fall (tilt > {cr['fell_deg']:g}°), no head failure (see below), "
             f"and in the last {cr['hold_window_s']:g} s no head joint off its command by more than "
             f"{cr['head_deviation_rad']:g} rad for more than {cr['head_deviation_frac_max']:.0%} of the window. "
             f"Sit: trunk ≤ {cr['sit_z_max_m']:g} m; stand: ≥ {cr['stand_z_min_m']:g} m; tilt < {cr['upright_deg']:g}°. "
             f"Voltages {', '.join(f'{v:g} V' for v in pr['voltages'])}; {pr['seeds']} seeds per cell; head command "
             f"zero; seated starts tilted at most {pr['seated_tilt_deg']:g}°. Zero tolerance for falls and hold-window "
             f"head failures in: {', '.join(cr['zero_tolerance_conditions'])}; elsewhere they count against the "
             f"success rate. Thresholds: " + ", ".join(
                 f"{c} {'unset' if cr['min_success_rate'].get(c) is None else format(cr['min_success_rate'][c], '.0%')}"
                 for c in conds) + ".", "",
             "| scenario | " + " | ".join(conds) + " |", "|---|" + "---|" * len(conds)]
    for name in pr["scenarios"]:
        lines.append(f"| {name} | " + " | ".join(_pct(sm["cells"].get(f"{name}/{c}")) for c in conds) + " |")
    lines.append("| **all** | " + " | ".join(f"**{_pct(sm['conditions'][c])}**" for c in conds) + " |")
    lines += ["", "| transition | " + " | ".join(conds) + " |", "|---|" + "---|" * len(conds)]
    for tt in ("sit-down", "stand-up", "hold"):
        lines.append(f"| {tt} | " + " | ".join(_pct(sm["by_transition"].get(f"{tt}/{c}")) for c in conds) + " |")
    lines += ["", "Failure reasons (an attempt can have several): " + (", ".join(
        f"{k} {v}" for k, v in sorted(sm["reasons"].items(), key=lambda kv: -kv[1])) or "none") + ".",
              "Falls (scenes): " + ", ".join(f"{c} {len(sm['falls'][c])}" for c in conds) +
              ". Head failures (attempts): " + ", ".join(
                  f"{c} {len(sm['head_failures'][c])}" for c in conds) +
              ". Broken scenes: " + ", ".join(f"{c} {len(v)}" for c, v in sm["broken_scenes"].items()) + ".",
              f"Head failures: an impact of at least {cr['head_impact_N']:g} N anywhere in the segment; the head "
              f"resting on the ground for {cr['head_rest_s'] * 1000:g} ms or longer once the posture is held (the whole "
              "segment when the posture does not change, else from the arrival); the head off its command in the "
              "last window.",
              "",
              "## Contacts (every 5 ms physics step)", "",
              f"Head (the `neck` subtree) support that is not a failure: {sm['head_support']['events']} events in "
              f"{sm['head_support']['scenes']} scenes, {sm['head_support']['total_s']:.2f} s in all, peak "
              f"{sm['head_support']['peak_N']:.2f} N.", "",
              "Impact peak: the largest summed normal force of the non-foot bodies in a scene (the robot weighs about "
              "8 N). A pointer for the video review, not a verdict.", "",
              "| scenario | median (N) | max (N) | largest: scene, time, bodies |", "|---|---|---|---|"]
    for name, im in sm["impacts"].items():
        top = im["top"][0] if im["top"] and im["top"][0]["t_s"] is not None else None  # None: no non-foot contact
        lines.append(f"| {name} | {im['median_N']:.1f} | {im['max_N']:.1f} | " + (
            f"`{top['id']}` at {top['t_s']:.2f} s, {', '.join(top['bodies'])}" if top else "-") + " |")
    fb = pr["fallback"]
    lines += ["", "## Seated fall-back (Pollen's docs/sitstand_policy.md; reported, not judged)", "",
              f"Seated start, {fb['seconds']:g} s, {fb['scenes_per_variant']} scenes per variant; head command "
              f"resampled every {fb['head_resample_s'][0]:g}-{fb['head_resample_s'][1]:g} s; jerks every "
              f"{fb['jerk_interval_s'][0]:g}-{fb['jerk_interval_s'][1]:g} s with p {fb['jerk_prob']:g}; tips every "
              f"{fb['tip_interval_s'][0]:g}-{fb['tip_interval_s'][1]:g} s with p {fb['tip_prob']:g} at "
              f"{fb['tip_ang_vel_rad_s'][0]:g}-{fb['tip_ang_vel_rad_s'][1]:g} rad/s. Leaned back: nose up past "
              f"{fb['lean_deg']:g}°. Pollen's vfhxgrum@2000 (their robot, their training world): head jerks only 3.1% "
              "of the time and 3.1% at the end; with tips 21% of the time; neck 0.045 N·m while leaned.", "",
              "| variant | time leaned back | leaned at the end | neck torque while leaned (N·m) | falls | scenes with head contact |",
              "|---|---|---|---|---|---|"]
    f = lambda v, fmt: "-" if v is None else format(v, fmt)  # noqa: E731
    for name, m in sm["fallback"].items():
        lines.append(f"| {name} | {f(m['time_leaned_back'], '.1%')} | {f(m['leaned_at_end'], '.1%')} | "
                     f"{f(m['neck_tau_while_leaned_Nm'], '.3f')} | {m['falls']} | {m['head_contact_scenes']} |")
    if report.get("paired"):
        pa = report["paired"]
        lines += ["", "## Paired with the reference", "",
                  ("" if pa["spec_matches"] else "**The reference ran another spec or overrides: not comparable.** ") +
                  f"Reference `{pa['reference_onnx_sha256'][:16]}…`, {pa['common_scenes']} common scenes, "
                  f"{len(pa['init_hash_mismatches'])} with a different start. Reference: " + ", ".join(
                      f"{c} {_pct(pa['reference_conditions'].get(c))}" for c in conds) +
                  f". Attempts the reference wins and this policy loses: {len(pa['reference_succeeds_this_fails'])}; "
                  f"the other way: {len(pa['this_succeeds_reference_fails'])}."]
    if report.get("videos"):
        lines += ["", "## Videos (nominal condition, one per scenario, fixed rule)", "",
                  "| scenario | scene | rule | full | slow (4x) |", "|---|---|---|---|---|"]
        for v in report["videos"]:
            lines.append(f"| {v['scenario']} | `{v['id']}` | {v['rule']} | [{Path(v['full']).name}]"
                         f"(videos/{Path(v['full']).name}) | " + (
                             f"[{Path(v['slow']).name}](videos/{Path(v['slow']).name}) ({v['slow_reason']}, "
                             f"{v['slow_window'][0]:.2f}-{v['slow_window'][1]:.2f} s)" if v.get("slow") else "-") + " |")
    (out / "README.md").write_text("\n".join(lines) + "\n")


def code_identity() -> str:
    """The commit, and with local changes one sha256 over `git diff HEAD` and every untracked file's name and bytes."""
    git = lambda *a: subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True,  # noqa: E731
                                    check=True).stdout
    try:
        head, status = git("rev-parse", "HEAD").strip(), git("status", "--porcelain").strip()
        if not status:
            return f"microduck-rl {head[:12]}"
        untracked = sorted(git("ls-files", "--others", "--exclude-standard").split())
        h = hashlib.sha256(git("diff", "HEAD").encode())
        for name in untracked:
            h.update(name.encode() + b"\0" + (ROOT / name).read_bytes())
        return f"microduck-rl {head[:12]} with local changes (sha256 {h.hexdigest()[:16]}, untracked {untracked})"
    except Exception:  # noqa: BLE001
        return "microduck-rl (no git)"


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_sha256(root: Path) -> tuple[str, dict]:
    """Every file under the assets directory (the scene and the robot MJCF; its meshes may live elsewhere)."""
    files = {str(p.relative_to(root)): sha256(p) for p in sorted(Path(root).rglob("*")) if p.is_file()}
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(), files


def model_sha256(scene: Path) -> str:
    """The compiled scene as MuJoCo loads it: masses, collisions and the meshes wherever meshdir points."""
    model = mujoco.MjModel.from_xml_path(str(scene))
    buf = np.zeros(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, None, buf)
    return hashlib.sha256(buf.tobytes()).hexdigest()


def _job(args):
    return run_scene(*args)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--assets", type=Path, required=True, help="scenes from sim2sim_assets.py")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--spec", type=Path, default=SPEC)
    ap.add_argument("--servo-params", type=Path, default=sim2sim.SERVO_PARAMS)
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    ap.add_argument("--reference", type=Path, help="gate.json of another policy, for a scene-by-scene comparison")
    ap.add_argument("--videos", action="store_true", help="render one nominal scene per scenario by a fixed rule")
    ap.add_argument("--title", help="name in the report's title; default: the ONNX file's parent directory")
    ap.add_argument("--seeds", type=int, help="override the seed count (a quick look; the report is then not a gate run)")
    ap.add_argument("--fallback-scenes", type=int, help="override the fall-back scene count (same caveat)")
    args = ap.parse_args()

    spec, spec_sha = load_spec(args.spec)
    overrides = {}
    if args.seeds is not None:
        overrides["seeds"] = spec["protocol"]["seeds"] = args.seeds
    if args.fallback_scenes is not None:
        overrides["fallback_scenes"] = spec["protocol"]["fallback"]["scenes_per_variant"] = args.fallback_scenes
    scenes = posture_scenes(spec["protocol"]) + fallback_scenes(spec["protocol"])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    jobs = [(s, spec, args.onnx, args.assets, args.servo_params) for s in scenes]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(_job, jobs, chunksize=4))
    elapsed = time.time() - t0
    summary = summarize(results, spec)
    pairing = (paired(results, json.loads(args.reference.read_text()), spec_sha, overrides)
               if args.reference else None)
    status, reasons = verdict(summary, spec["criteria"], pairing)
    if overrides:
        status, reasons = "UNDECIDED", [f"protocol overridden ({overrides}): not a gate run"] + reasons
    assets_sha, assets_files = tree_sha256(args.assets)
    report = {"verdict": status, "verdict_reasons": reasons, "spec": spec, "summary": summary, "paired": pairing,
              "provenance": {"title": args.title or args.onnx.resolve().parent.name, "onnx": str(args.onnx.resolve()),
                             "onnx_sha256": sha256(args.onnx), "spec": str(args.spec.resolve()),
                             "spec_sha256": spec_sha, "overrides": overrides, "assets": str(args.assets.resolve()),
                             "assets_sha256": assets_sha, "assets_files": assets_files,
                             "model_sha256": model_sha256(args.assets / "scene_hd1910_groundcontact.xml"),
                             "servo_sha256": sha256(args.servo_params),
                             "code": code_identity(), "argv": sys.argv, "scenes": len(scenes),
                             "workers": args.workers, "elapsed_s": elapsed,
                             "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
              "scenes": results}
    if args.videos:
        vdir = args.out_dir / "videos"
        vdir.mkdir(exist_ok=True)
        by_id = {s.id: s for s in scenes}
        report["videos"] = []
        for v in pick_videos(results, spec["protocol"]):
            stem = v["id"].replace("/", "_")
            render = {"full": vdir / f"{stem}.mp4"}
            if v["slow_center_s"] is not None:
                render["slow"] = vdir / f"{stem}_slow4x.mp4"
                render["slow_window"] = (max(0.0, v["slow_center_s"] - 1.0), v["slow_center_s"] + 1.0)
            again = run_scene(by_id[v["id"]], spec, args.onnx, args.assets, args.servo_params, render=render)
            original = next(r for r in results if r["id"] == v["id"])
            report["videos"].append({**v, "full": str(render["full"]), "slow": str(render.get("slow") or "") or None,
                                     "slow_window": render.get("slow_window"),
                                     "rerun_matches": again.get("impact_peak_N") == original.get("impact_peak_N")
                                     and again.get("attempts") == original.get("attempts"),
                                     "error": again.get("error")})
    (args.out_dir / "gate.json").write_text(json.dumps(report, indent=1, default=float) + "\n")
    write_markdown(args.out_dir, report)
    c = summary["conditions"]
    print(f"{status}: " + ", ".join(f"{k} {_pct(v)}" for k, v in c.items()) + (f" — {'; '.join(reasons)}" if reasons else ""))


if __name__ == "__main__":
    main()
