"""sitstand_gate.py on CPU: scenes, scoring, the contact audit, the fall-back metrics, the verdict, and short scenes
on a stand-in robot with a stand-in policy."""

import copy
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import sim2sim  # noqa: E402
import sitstand_gate as g  # noqa: E402

SPEC, SPEC_SHA = g.load_spec()
PROTOCOL, CRIT = SPEC["protocol"], SPEC["criteria"]
DT = sim2sim.CONTROL_DT


# -- spec and scenes -----------------------------------------------------------------------------------------------
def test_the_spec_lists_the_approved_scenes():
    scenes = g.posture_scenes(PROTOCOL)
    assert len(scenes) == 4 * 2 * 2 * 16
    assert len({s.id for s in scenes}) == len(scenes)
    assert not any(s.head for s in scenes)  # robotd sends a zero head command in sit-stand
    assert {s.scenario for s in scenes} == {"stand_cycle", "sit_cycle", "stand_hold", "sit_hold"}
    fallback = g.fallback_scenes(PROTOCOL)
    assert len(fallback) == 2 * 128 and {s.vin for s in fallback} == {7.4, 8.0} and all(s.head for s in fallback)
    # set on the kissqy baseline (100% / 84.4%): a policy may not be clearly worse than the one the robot runs
    assert CRIT["min_success_rate"] == {"nominal": 0.98, "disturbed": 0.80}
    assert CRIT["zero_tolerance_conditions"] == ["nominal"]
    assert (CRIT["head_deviation_rad"], CRIT["head_deviation_frac_max"]) == (0.2, 0.2)


def test_seated_starts_tilt_less_than_the_trainings_with_the_same_draws():
    assert PROTOCOL["seated_tilt_deg"] == 4.0
    home = np.zeros(14)
    for seed in range(50):
        z8, q8, j8 = sim2sim.seated_pose(home, np.random.default_rng(seed))
        z4, q4, j4 = sim2sim.seated_pose(home, np.random.default_rng(seed), tilt=np.radians(4.0))
        assert z4 == z8 and np.array_equal(j4, j8)  # same draws: only the tilt shrinks
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, np.array(q4))
        assert np.degrees(np.arccos(rot[8])) <= 4.0 * np.sqrt(2) + 1e-9


def test_a_scenes_random_numbers_depend_on_its_id_only():
    a = [r.random() for r in g.streams("posture/x")]
    assert a == [r.random() for r in g.streams("posture/x")]
    assert a != [r.random() for r in g.streams("posture/y")]
    assert len(set(a)) == 3  # setup, commands and runtime are separate streams


def test_scenes_run_on_the_training_physics_with_robotds_chain():
    scenes = g.posture_scenes(PROTOCOL)
    nominal = g.level_for(scenes[0], PROTOCOL)
    assert (nominal.scene, nominal.collisions, nominal.robotd, nominal.sensors) == ("groundcontact", "groundcontact",
                                                                                     True, False)
    assert (nominal.timestep, nominal.iterations, nominal.ls_iterations, nominal.joint_vel_lag) == (0.005, 30, 50, 1)
    assert not nominal.randomize and nominal.push_mps == 0.0 and nominal.vin_range == sim2sim.ROBOTD_VIN_RANGE
    disturbed = g.level_for(next(s for s in scenes if s.condition == "disturbed"), PROTOCOL)
    assert disturbed.randomize and disturbed.push_mps == 0.2 and disturbed.latency_ms == (15.0, 30.0)
    assert g.level_for(next(s for s in scenes if s.scenario == "sit_hold"), PROTOCOL).start == "seated"
    assert g.level_for(g.fallback_scenes(PROTOCOL)[0], PROTOCOL).start == "seated"


# -- scoring -------------------------------------------------------------------------------------------------------
def make_traces(postures, z_of_t, seg_s=6.0, head_dev=None):
    t = np.arange(1, round(len(postures) * seg_s / DT) + 1) * DT
    z = np.array([z_of_t(x) for x in t])
    head_q = np.zeros((len(t), 4))
    if head_dev is not None:
        head_q[:, 0] = [head_dev(x) for x in t]
    return {"t": t, "z": z, "tilt": np.zeros(len(t)), "yaw": np.zeros(len(t)), "head_q": head_q,
            "head_cmd": np.zeros((len(t), 4))}


def sit_then_stand(sit_at=1.0, stand_at=7.0):
    """Stand start; sit command 0-6 s, stand command 6-12 s."""
    return lambda x: 0.06 if sit_at <= x < stand_at else 0.115


def score(tr, postures=("sit", "stand"), start="stand", head_events=(), fell_at=None, error=None):
    return g.score_attempts(tr, list(postures), 6.0, start, list(head_events), fell_at, error, CRIT)


def test_attempts_reached_in_time_and_held_succeed():
    a = score(make_traces(["sit", "stand"], sit_then_stand()))
    assert [x["transition"] for x in a] == ["sit-down", "stand-up"]
    assert [x["success"] for x in a] == [True, True]
    assert a[0]["settled_s"] == pytest.approx(1.0) and a[1]["settled_s"] == pytest.approx(1.0)


def test_a_late_arrival_and_a_lost_posture_fail():
    late = score(make_traces(["sit", "stand"], sit_then_stand(sit_at=3.5)))
    assert late[0]["reasons"] == ["reached_late"] and late[1]["success"]
    lost = score(make_traces(["sit", "stand"], sit_then_stand(stand_at=5.5)))  # stood up before the stand command
    assert lost[0]["reasons"] == ["not_in_posture_at_end"]


def test_a_head_off_its_command_in_the_hold_window_fails_above_the_share():
    def droop(share):
        return lambda x: 0.3 if 6.0 - 2.0 * share < x % 6.0 or x % 6.0 == 0.0 else 0.0
    ok = score(make_traces(["sit", "stand"], sit_then_stand(), head_dev=droop(0.1)))
    assert ok[0]["success"] and ok[0]["head_off_frac"] == pytest.approx(0.1, abs=0.02)
    bad = score(make_traces(["sit", "stand"], sit_then_stand(), head_dev=droop(0.3)))
    assert bad[0]["reasons"] == ["head_off_command_in_hold"]
    assert bad[0]["head_max_dev_rad"] == pytest.approx(0.3)


def test_head_contact_counts_in_the_hold_window_only():
    tr = make_traces(["sit", "stand"], sit_then_stand())
    assist = {"body": "jaw_soft", "head": True, "start_s": 1.0, "end_s": 1.4, "peak_N": 2.0}
    assert all(x["success"] for x in score(tr, head_events=[assist]))
    rest = {"body": "jaw_soft", "head": True, "start_s": 4.5, "end_s": 4.6, "peak_N": 0.5}
    a = score(tr, head_events=[rest])
    assert a[0]["reasons"] == ["head_contact_in_hold"] and a[0]["head_contact_in_hold"][0]["body"] == "jaw_soft"
    assert a[1]["success"]


def test_a_fall_fails_its_segment_and_every_later_one():
    a = score(make_traces(["sit", "stand", "sit"], lambda x: 0.06 if 1.0 <= x < 7.0 or x >= 13.0 else 0.115),
              postures=("sit", "stand", "sit"), fell_at=7.5)
    assert a[0]["success"] and "fell" in a[1]["reasons"] and "fell" in a[2]["reasons"]


def test_a_broken_scene_fails_the_segments_it_did_not_finish():
    tr = make_traces(["sit", "stand"], sit_then_stand())
    cut = {k: v[:400] for k, v in tr.items()}  # 8 s of 12
    a = score(cut, error="non-finite state")
    assert a[0]["success"] and a[1]["reasons"] == ["error"]


def test_a_seated_start_with_a_sit_command_is_a_hold():
    a = score(make_traces(["sit"], lambda x: 0.06), postures=("sit",), start="sit")
    assert a[0]["transition"] == "hold" and a[0]["settled_s"] == 0.0 and a[0]["success"]


# -- contact audit -------------------------------------------------------------------------------------------------
def test_contact_events_merge_short_gaps_and_add_up_impulse():
    audit = g.ContactAudit(["world", "trunk", "jaw"], head={2}, feet=set())
    dt = 0.005
    loads = [{1: 2.0}] * 3 + [{}] + [{1: 4.0}] * 2 + [{}] * 6 + [{2: 1.0, 1: 1.0}]
    for i, load in enumerate(loads):
        audit.step(i * dt, dt, load)
    events = audit.finish()
    trunk = [e for e in events if e["body"] == "trunk"]
    assert len(trunk) == 2  # a one-step gap stays one event; six steps close it
    assert trunk[0]["start_s"] == 0.0 and trunk[0]["end_s"] == pytest.approx(6 * dt)
    assert trunk[0]["peak_N"] == 4.0 and trunk[0]["impulse_Ns"] == pytest.approx((3 * 2.0 + 2 * 4.0) * dt)
    jaw = next(e for e in events if e["body"] == "jaw")
    assert jaw["head"] and jaw["duration_s"] == pytest.approx(dt)
    assert audit.peak_n == 4.0 and audit.peak_t == pytest.approx(4 * dt)


BOX = """
<mujoco>
  <option timestep="0.005"/>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.05"/>
    <body name="trunk_base" pos="0 0 0.05"><freejoint/><geom type="box" size="0.05 0.05 0.02" mass="0.8"/></body>
  </worldbody>
</mujoco>
"""


def test_the_audit_reads_the_solved_normal_force():
    model = mujoco.MjModel.from_xml_string(BOX)
    data = mujoco.MjData(model)
    for _ in range(400):
        mujoco.mj_step(model, data)
    audit = g.ContactAudit([model.body(b).name for b in range(model.nbody)], head=set(), feet=set())
    for _ in range(20):
        t = data.time
        mujoco.mj_step(model, data)
        audit.sample(model, data, t, model.opt.timestep)
    events = audit.finish()
    assert len(events) == 1 and events[0]["body"] == "trunk_base"
    weight = 0.8 * 9.81
    assert events[0]["impulse_Ns"] / events[0]["duration_s"] == pytest.approx(weight, rel=0.05)


# -- fall-back metrics ---------------------------------------------------------------------------------------------
def test_leaned_back_means_nose_up_past_the_threshold():
    tr = {"tilt": np.array([5.0, 25.0, 25.0, 30.0]), "gx": np.array([-0.1, -0.4, 0.4, -0.5]),
          "neck_tau": np.array([0.01, 0.10, 0.20, 0.30])}
    m = g.fallback_metrics(tr, 20.0)
    assert m["leaned_ticks"] == 2 and m["leaned_at_end"] is True  # the nose-down tilt does not count
    assert m["neck_tau_leaned_sum_Nm"] == pytest.approx(0.40)


# -- verdict -------------------------------------------------------------------------------------------------------
def summary(rate_nominal=0.97, rate_disturbed=0.96, falls=None, head=None, broken=()):
    conds = {c: {"attempts": 100, "successes": round(100 * r), "rate": r}
             for c, r in (("nominal", rate_nominal), ("disturbed", rate_disturbed))}
    empty = {"nominal": [], "disturbed": []}
    return {"conditions": conds, "falls": {**empty, **(falls or {})}, "hold_head_failures": {**empty, **(head or {})},
            "broken_scenes": list(broken)}


def crit(**kw):
    c = copy.deepcopy(CRIT)
    c.update(kw)
    return c


BOTH = {"nominal": 0.95, "disturbed": 0.80}


def test_without_thresholds_the_verdict_is_undecided_unless_something_failed():
    unset = crit(min_success_rate={"nominal": None, "disturbed": None})
    assert g.verdict(summary(), unset, None)[0] == "UNDECIDED"
    assert g.verdict(summary(falls={"nominal": ["a"]}), unset, None)[0] == "FAILED"
    assert g.verdict(summary(head={"nominal": ["a#1"]}), unset, None)[0] == "FAILED"
    assert g.verdict(summary(broken=["a"]), unset, None)[0] == "FAILED"
    status, why = g.verdict(summary(), crit(min_success_rate={"nominal": 0.95, "disturbed": None}), None)
    assert status == "UNDECIDED" and "disturbed" in why[0]


def test_the_frozen_thresholds_pass_a_kissqy_like_result_and_fail_a_worse_one():
    assert g.verdict(summary(1.0, 0.844), CRIT, None) == ("PASSED", [])
    assert g.verdict(summary(0.97, 0.90), CRIT, None)[0] == "FAILED"
    assert g.verdict(summary(1.0, 0.79), CRIT, None)[0] == "FAILED"


def test_falls_and_head_failures_are_zero_tolerance_in_nominal_only():
    disturbed_only = summary(falls={"disturbed": ["a", "b"]}, head={"disturbed": ["c#0"]})
    assert g.verdict(disturbed_only, crit(min_success_rate=BOTH), None) == ("PASSED", [])
    status, why = g.verdict(summary(falls={"nominal": ["a"]}), crit(min_success_rate=BOTH), None)
    assert status == "FAILED" and why[0].startswith("nominal: 1 scenes fell")


def test_each_condition_has_its_own_threshold():
    assert g.verdict(summary(), crit(min_success_rate=BOTH), None) == ("PASSED", [])
    status, why = g.verdict(summary(rate_disturbed=0.75), crit(min_success_rate=BOTH), None)
    assert status == "FAILED" and why == ["disturbed: success 75/100 below 80%"]
    status, why = g.verdict(summary(rate_nominal=0.94), crit(min_success_rate=BOTH), None)
    assert status == "FAILED" and why[0].startswith("nominal")


def test_not_below_the_reference_needs_one_and_compares_each_condition():
    c = crit(min_success_rate={"nominal": 0.5, "disturbed": 0.5}, not_below_reference=True)
    assert g.verdict(summary(), c, None)[0] == "UNDECIDED"
    ref = {"init_hash_mismatches": [], "reference_conditions": summary(0.99, 0.90)["conditions"]}
    status, why = g.verdict(summary(0.97, 0.96), c, ref)
    assert status == "FAILED" and "nominal" in why[0]
    assert g.verdict(summary(0.99, 0.96), c, ref)[0] == "PASSED"
    assert g.verdict(summary(0.99, 0.96), c, dict(ref, init_hash_mismatches=["x"]))[0] == "FAILED"


# -- short scenes on a stand-in robot ------------------------------------------------------------------------------
ROBOT = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.05"/>
    <body name="trunk_base" pos="0 0 0.125">
      <freejoint/>
      <geom name="trunk_collision" type="box" size="0.03 0.03 0.02" mass="0.4"/>
      <site name="imu"/>
      <body name="leg" pos="0 0 -0.03">
        <joint name="j1" axis="0 1 0" range="-1 1"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.04" size="0.008" mass="0.05"/>
        <body name="foot" pos="0 0 -0.05">
          <joint name="j2" axis="0 1 0" range="-1 1"/>
          <geom name="left_foot_collision" type="box" size="0.03 0.02 0.005" pos="0 0.02 0" mass="0.02"/>
          <geom name="right_foot_collision" type="box" size="0.03 0.02 0.005" pos="0 -0.02 0" mass="0.02"/>
        </body>
      </body>
      <body name="neck" pos="0.02 0 0.03">
        <joint name="neck_pitch" axis="0 1 0" range="-1.5 1.5"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.03" size="0.006" mass="0.02"/>
        <body name="head" pos="0 0 0.03">
          <joint name="head_pitch" axis="0 1 0" range="-1.5 1.5"/>
          <joint name="head_yaw" axis="0 0 1" range="-1.5 1.5"/>
          <joint name="head_roll" axis="1 0 0" range="-1.5 1.5"/>
          <geom name="jaw_soft" type="sphere" size="0.015" mass="0.03"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position joint="j1" kp="0.5"/><position joint="j2" kp="0.5"/><position joint="neck_pitch" kp="0.5"/>
    <position joint="head_pitch" kp="0.5"/><position joint="head_yaw" kp="0.5"/><position joint="head_roll" kp="0.5"/>
  </actuator>
  <sensor>
    <gyro name="imu_ang_vel" site="imu"/>
    <framequat name="orientation" objtype="site" objname="imu"/>
  </sensor>
</mujoco>
"""
JOINTS = ["j1", "j2", "neck_pitch", "head_pitch", "head_yaw", "head_roll"]


class StandIn:
    """A policy with the ONNX's interface: joint names, HOME, scale; records what it was shown."""

    def __init__(self, gain=0.0):
        self.joint_names, self.home, self.scale, self.gain = JOINTS, np.zeros(len(JOINTS)), 1.0, gain
        self.seen = []

    def __call__(self, obs):
        self.seen.append(obs.copy())
        return self.gain * np.ones(len(JOINTS))


@pytest.fixture
def assets(tmp_path):
    (tmp_path / "scene_hd1910_groundcontact.xml").write_text(ROBOT)
    return tmp_path


def short_spec():
    spec = copy.deepcopy(SPEC)
    spec["protocol"]["segment_s"] = 0.4
    spec["protocol"]["scenarios"] = {"stand_hold": {"start": "standing", "postures": ["stand", "sit"]}}
    spec["protocol"]["fallback"]["seconds"] = 0.4
    return spec


def test_the_head_command_reaches_the_policy_in_its_slot(assets):
    spec = short_spec()
    scene = g.Scene("posture", "stand_hold", "nominal", 7.4, 9, True)
    policy = StandIn()
    r = g.run_scene(scene, spec, None, assets, sim2sim.SERVO_PARAMS, policy=policy, keep_traces=True)
    assert r["error"] is None and len(r["attempts"]) == 2
    first, last = policy.seen[0], policy.seen[-1]
    cmd = np.array(r["traces"]["head_cmd"])
    np.testing.assert_allclose(first[-10:-6], cmd[0])
    np.testing.assert_allclose(last[-10:-6], cmd[-1])
    assert not np.allclose(cmd[0], cmd[-1])  # one head command per segment
    assert first[-13] == 0.0 and last[-13] == 1.0  # the posture flag: stand, then sit
    lo, hi = np.array(spec["protocol"]["head_command_range_rad"]).T
    assert np.all((cmd >= lo) & (cmd <= hi))


def test_a_scene_starts_the_same_for_every_policy_and_reruns_exactly(assets):
    spec = short_spec()
    scene = g.Scene("posture", "stand_hold", "disturbed", 8.0, 3, False)
    a = g.run_scene(scene, spec, None, assets, sim2sim.SERVO_PARAMS, policy=StandIn(0.0))
    b = g.run_scene(scene, spec, None, assets, sim2sim.SERVO_PARAMS, policy=StandIn(0.3))
    again = g.run_scene(scene, spec, None, assets, sim2sim.SERVO_PARAMS, policy=StandIn(0.0))
    assert a["error"] is None and b["error"] is None
    assert a["init_hash"] == b["init_hash"] == again["init_hash"]
    assert a["vin"] == 8.0
    assert again["impact_peak_N"] == a["impact_peak_N"] and again["attempts"] == a["attempts"]


def test_a_fallback_scene_reports_its_metrics_and_perturbations(assets):
    spec = short_spec()
    spec["protocol"]["fallback"].update(jerk_interval_s=[0.05, 0.05], jerk_prob=1.0)
    scene = g.Scene("fallback", "head_jerks", "nominal", 7.4, 0, True)
    with pytest.raises(AssertionError):  # the stand-in's joints are not the HD-1910's: no seated start for it
        r = g._run_scene(scene, spec, None, assets, sim2sim.SERVO_PARAMS, StandIn(), None, False)
    r = g.run_scene(scene, spec, None, assets, sim2sim.SERVO_PARAMS, policy=StandIn())
    assert "AssertionError" in r["error"]  # a broken scene is reported, not raised


def test_neck_subtree_follows_parent_links(assets):
    spec = mujoco.MjSpec.from_file(str(assets / "scene_hd1910_groundcontact.xml"))
    model = spec.compile()
    assert {model.body(b).name for b in g.neck_subtree(model)} == {"neck", "head"}
    assert {model.body(b).name for b in g.foot_bodies(model)} == {"foot"}


def test_the_gate_does_not_import_the_training_code():
    code = "import sys; sys.path.insert(0, %r); import sitstand_gate; " \
           "bad = [m for m in sys.modules if m.split('.')[0] in ('mjlab', 'mjlab_microduck', 'torch')]; " \
           "assert not bad, bad" % str(SCRIPTS)
    subprocess.run([sys.executable, "-c", code], check=True)


def test_summary_and_report_from_scene_results(tmp_path):
    spec = copy.deepcopy(SPEC)
    ok = {"success": True, "reasons": [], "transition": "sit-down", "segment": 0}
    bad = {"success": False, "reasons": ["head_contact_in_hold"], "transition": "hold", "segment": 1,
           "head_contact_in_hold": [{"body": "jaw_soft", "start_s": 11.0, "end_s": 11.2, "peak_N": 1.0}]}
    results = []
    for s in g.posture_scenes(spec["protocol"]):
        attempts = [dict(ok)] * 3 + [dict(bad) if s.seed == 0 and s.scenario == "sit_hold" else dict(ok)]
        contacts = [{"body": "jaw_soft", "head": True, "start_s": 11.0, "end_s": 11.2, "duration_s": 0.2,
                     "peak_N": 1.0, "impulse_Ns": 0.1}] if s.seed == 0 and s.scenario == "sit_hold" else []
        results.append({"id": s.id, "scene": vars(s), "init_hash": "h", "fell_at_s": None, "error": None,
                        "attempts": attempts, "contacts": contacts, "impact_peak_N": 5.0, "impact_peak_s": 1.0,
                        "impact_peak_bodies": ["trunk_base"]})
    for s in g.fallback_scenes(spec["protocol"]):
        results.append({"id": s.id, "scene": vars(s), "fell_at_s": None, "error": None, "contacts": [],
                        "fallback": {"ticks": 100, "leaned_ticks": 10, "leaned_at_end": s.seed % 2 == 0,
                                     "neck_tau_leaned_sum_Nm": 0.5, "error": None}})
    sm = g.summarize(results, spec)
    assert sm["conditions"]["nominal"]["attempts"] == 4 * 2 * 16 * 4
    # sit_hold seed 0 at two voltages, in each condition
    assert {c: len(v) for c, v in sm["hold_head_failures"].items()} == {"nominal": 2, "disturbed": 2}
    assert sm["head_contacts_outside_hold"]["events"] == 0  # the only head contacts were in hold windows
    fb = sm["fallback"]["head_jerks"]
    assert fb["time_leaned_back"] == pytest.approx(0.1) and fb["leaned_at_end"] == pytest.approx(0.5)
    assert fb["neck_tau_while_leaned_Nm"] == pytest.approx(0.05)
    status, _ = g.verdict(sm, spec["criteria"], None)
    assert status == "FAILED"
    picks = g.pick_videos(results, spec["protocol"])
    assert [p["scenario"] for p in picks] == list(spec["protocol"]["scenarios"])
    assert next(p for p in picks if p["scenario"] == "sit_hold")["rule"] == "first failed scene"
    report = {"verdict": status, "verdict_reasons": ["x"], "spec": spec, "summary": sm, "paired": None,
              "provenance": {"title": "t", "onnx": "p.onnx", "onnx_sha256": "0" * 64, "spec": "s", "spec_sha256": "1",
                             "overrides": {}, "assets": "a", "servo_sha256": "2" * 64, "code": "c", "scenes": 1,
                             "workers": 1, "elapsed_s": 1.0}}
    g.write_markdown(tmp_path, report)
    text = (tmp_path / "README.md").read_text()
    assert "Stage automatic check: FAILED" in text and "| sit_hold |" in text
    assert "Falls (scenes): nominal 0, disturbed 0" in text and "Thresholds: nominal 98%, disturbed 80%" in text
