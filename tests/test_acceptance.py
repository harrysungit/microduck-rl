"""acceptance_core.py without a GPU: declarations, verdicts and the metrics from recorded arrays."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import acceptance_core as ac  # noqa: E402
import sim2sim  # noqa: E402


# -- declaration
def test_a_stage_without_an_acceptance_entry_is_refused_but_none_is_allowed_on_purpose():
    assert ac.problems({"id": "Q1"})
    assert ac.problems({"id": "Q1", "acceptance": "none"}) == []
    assert ac.contracts_of("none") == []


def test_valid_declarations_in_all_three_forms():
    assert ac.problems({"acceptance": "posture"}) == []
    assert ac.problems({"acceptance": {"contract": "velocity", "params": {"num_envs": 128},
                                       "criteria": {"clean": {"forward_min": 0.6}}}}) == []
    assert ac.problems({"acceptance": ["velocity", {"contract": "getup", "criteria": {"trained": {"up_min": 0.7}}}]}) == []


@pytest.mark.parametrize("acceptance, text", [
    ("posutre", "unknown contract"),
    ({"contract": "posture", "criteria": {"clean": {"sit_dwn_min": 0.5}}}, "unknown criterion"),
    ({"contract": "posture", "criteria": {"clean": {"sit_down_mean": 0.5}}}, "unknown criterion"),
    ({"contract": "posture", "criteria": {"dirty": {"sit_down_min": 0.5}}}, "unknown variant"),
    ({"contract": "posture", "criteria": {"clean": {"sit_down_min": "high"}}}, "must be a number"),
    ({"contract": "posture", "criteria": {"clean": {"sit_down_min": float("nan")}}}, "must be a number"),
    ({"contract": "posture", "thresholds": {}}, "unknown field"),
    ({"contract": "posture", "params": {"worlds": 3}}, "unknown parameter"),
    ({"criteria": {}}, "unknown contract"),
])
def test_declaration_typos_are_caught(acceptance, text):
    found = ac.problems({"acceptance": acceptance})
    assert found and any(text in p for p in found), found


def test_criteria_overrides_merge_over_the_defaults():
    crit = ac.contracts_of({"contract": "posture", "criteria": {"clean": {"sit_down_min": 0.95}}})[0]["criteria"]
    assert crit["clean"]["sit_down_min"] == 0.95 and crit["clean"]["stand_up_min"] == 0.9
    assert crit["trained"] == ac.DEFAULT_CRITERIA["posture"]["trained"]
    assert ac.DEFAULT_CRITERIA["posture"]["clean"]["sit_down_min"] == 0.8  # the defaults are not mutated


# -- verdict
def check(passed, gate=True):
    return {"variant": "clean", "metric": "m", "op": ">=", "threshold": 1.0, "value": 0.5, "passed": passed, "gate": gate}


def test_verdict_priorities():
    assert ac.decide([check(True), check(True)]) == ("ACCEPTED", [])
    assert ac.decide([check(True), check(False)])[0] == "FAILED"
    assert ac.decide([check(None), check(False)])[0] == "FAILED"  # a known failure outranks an unknown
    assert ac.decide([check(True), check(None)])[0] == "UNVERIFIED"
    assert ac.decide([check(True)], errors=["the checker crashed"])[0] == "UNVERIFIED"
    assert ac.decide([])[0] == "UNVERIFIED"  # nothing ran: never a pass
    assert ac.decide([check(True), check(False, gate=False)])[0] == "ACCEPTED"  # warnings do not gate


def test_evaluate_reads_min_and_max_and_treats_missing_or_nan_as_unknown():
    metrics = {"clean": {"sit_down": 0.0, "tipped": 0.01, "stand_up": float("nan")}, "trained": {"sit_down": 0.7}}
    criteria = {"clean": {"sit_down_min": 0.8, "tipped_max": 0.02, "stand_up_min": 0.9},
                "trained": {"sit_down_min": 0.6, "tipped_max": 0.1}}
    by = {(c["variant"], c["metric"]): c["passed"] for c in ac.evaluate(metrics, criteria)}
    assert by == {("clean", "sit_down"): False, ("clean", "tipped"): True, ("clean", "stand_up"): None,
                  ("trained", "sit_down"): True, ("trained", "tipped"): None}


def test_the_onnx_check_and_the_noise_gap():
    assert ac.onnx_check(1e-6)["passed"] is True and ac.onnx_check(0.5)["passed"] is False
    assert ac.onnx_check(None)["passed"] is None
    assert ac.noise_gap(0.0, 1.0)["noise_started"] is True
    assert ac.noise_gap(0.9, 1.0)["noise_started"] is False
    assert ac.noise_gap(None, 1.0)["gap"] is None


# -- metrics from arrays
DT = 0.02


def sit_profile(t):
    """Stand; down over 1.5 s from 3 s; sit; up over 1 s from 8 s; again from 13 s and 18 s (SIT_FROM_STAND)."""
    z = np.full_like(t, 0.115)
    for down, up in ((3.0, 8.0), (13.0, 18.0)):
        d = np.clip((t - down) / 1.5, 0, 1)
        u = np.clip((t - up) / 1.0, 0, 1)
        z -= 0.055 * (d - u) * ((t > down) & (t <= up + 1.0))
    return z


def test_posture_counts_separate_a_policy_that_sits_from_one_that_does_not():
    t = (np.arange(1150) + 1) * DT
    z = np.stack([sit_profile(t), np.full_like(t, 0.115), sit_profile(t), sit_profile(t)], axis=1)
    tilt = np.full_like(z, 3.0)
    tilt[100:, 3] = 80.0  # the fourth world tips over and stays there
    run = {"z": z, "tilt": tilt, "reset": np.array([False, False, True, False]), "dt": DT}
    c = ac.posture_counts(run, sim2sim.SIT_FROM_STAND)
    assert c["worlds"] == 3 and c["reset_midway"] == 1 and c["tipped"] == 1
    assert c["sit_down_tries"] == 6  # two sit-downs in each of three worlds
    assert c["sit_down_ok"] == 2  # only the first world sits down; the second never does, the fourth is tipped
    m = ac.posture_metrics([c])
    assert math.isclose(m["sit_down"], 2 / 6) and math.isclose(m["tipped"], 1 / 3)
    assert 1.0 < m["sit_down_settle_median_s"] < 2.0


def test_posture_metrics_pool_over_scenarios():
    a = {"worlds": 10, "reset_midway": 0, "tipped": 1, "sit_down_ok": 5, "sit_down_tries": 10, "stand_up_ok": 10,
         "stand_up_tries": 10, "hold_ok": 10, "hold_tries": 10, "sit_down_settle_median_s": 2.0, "sit_z_end_median_cm": 6.3}
    b = {**a, "worlds": 30, "tipped": 3, "sit_down_ok": 30, "sit_down_tries": 30, "stand_up_ok": 15, "stand_up_tries": 30}
    m = ac.posture_metrics([a, b])
    assert math.isclose(m["sit_down"], 35 / 40) and math.isclose(m["stand_up"], 25 / 40) and math.isclose(m["tipped"], 4 / 40)


def velocity_run(scale_fwd=0.8, scale_turn=0.9, resets=(False, False, False)):
    t = (np.arange(round(sim2sim.DURATION / DT)) + 1) * DT
    n = len(resets)
    vx, vy, wz = (np.zeros((len(t), n)) for _ in range(3))
    start = 0.0
    for duration, cmd, label in sim2sim.SCHEDULE:
        sel = (t > start) & (t <= start + duration)
        vx[sel], vy[sel], wz[sel] = cmd[0] * scale_fwd, cmd[1] * 0.4, cmd[2] * scale_turn
        start += duration
    return {"vx": vx, "vy": vy, "wz": wz, "reset": np.array(resets), "dt": DT}


def test_velocity_metrics_are_ratios_to_the_command():
    m = ac.velocity_metrics(velocity_run(), sim2sim.SCHEDULE)
    assert math.isclose(m["forward"], 0.8) and math.isclose(m["backward"], 0.8)
    assert math.isclose(m["turn"], 0.9) and math.isclose(m["lateral"], 0.4) and m["reset"] == 0.0


def test_velocity_metrics_leave_out_worlds_that_reset_and_count_them():
    run = velocity_run(resets=(False, True, False))
    run["vx"][:, 1] = 99.0  # the world that reset must not leak into the ratios
    m = ac.velocity_metrics(run, sim2sim.SCHEDULE)
    assert math.isclose(m["forward"], 0.8) and math.isclose(m["reset"], 1 / 3)
    stuck = velocity_run(scale_fwd=0.0, scale_turn=0.0)
    m = ac.velocity_metrics(stuck, sim2sim.SCHEDULE)
    assert m["forward"] == 0.0 and m["turn"] == 0.0


def test_getup_metrics():
    t = (np.arange(400) + 1) * DT
    z = np.stack([np.where(t > 2.0, 0.115, 0.04), np.full_like(t, 0.04), np.where(t > 2.0, 0.115, 0.04), np.where(t > 7.5, 0.115, 0.04)], axis=1)
    tilt = np.where(z > 0.09, 3.0, 90.0)
    run = {"z": z, "tilt": tilt, "reset": np.array([False, False, True, False]), "dt": DT}
    m = ac.getup_metrics(run)
    assert math.isclose(m["up"], 1 / 4)  # world 0 only: 1 stays down, 2 reset, 3 is up for just half a second
    assert math.isclose(m["up_at_median_s"], 2.0, abs_tol=0.05) and m["worlds"] == 4


def test_getup_needs_the_trunk_up_for_a_full_second():
    t = (np.arange(400) + 1) * DT
    z = np.where((t > 2.0) & (t < 2.9), 0.115, 0.04)[:, None]
    run = {"z": z, "tilt": np.where(z > 0.09, 3.0, 90.0), "reset": np.array([False]), "dt": DT}
    assert ac.getup_metrics(run)["up"] == 0.0


# -- report
def test_report_names_the_verdict_and_the_failed_check():
    checks = ac.evaluate({"clean": {"sit_down": 0.0}}, {"clean": {"sit_down_min": 0.8}})
    for c in checks:
        c["contract"] = "posture"
    verdict, why = ac.decide(checks)
    result = {"verdict": verdict, "reasons": why, "task": "T", "policy": {"onnx": "policy.onnx", "sha256": "ab" * 32},
              "checks": checks, "contracts": [{"contract": "posture", "metrics": {"clean": {"sit_down": 0.0}},
                                               "noise_check": ac.noise_gap(0.0, 1.0)}],
              "learned_std": {"mean": 0.188, "min": 0.15, "max": 0.26}, "errors": []}
    text = ac.render(result)
    assert text.startswith("# Acceptance: FAILED") and "**FAIL**" in text
    assert "started by the noise" in text and "Learned exploration std" in text


def test_kick_metrics_count_worlds_whose_ball_goes_forward_fast_enough():
    ball = np.zeros((250, 4))
    ball[100:, 0] = 1.0  # world 0: forward at 1 m/s
    ball[:, 1] = 0.2  # world 1: too slow
    ball[:, 2] = -1.0  # world 2: backward
    ball[50, 3] = 0.6  # world 3: one fast moment is a kick
    m = ac.kick_metrics({"ball_fwd": ball, "reset": np.array([False, False, True, False])})
    assert m["kick"] == 0.5 and m["reset"] == 0.25 and m["worlds"] == 4


def test_roulade_metrics_need_the_trunk_over_and_standing_again():
    tilt, z = np.full((300, 4), 5.0), np.full((300, 4), 0.115)
    tilt[100:150, [0, 2, 3]] = 170.0  # three worlds turn over
    tilt[150:, 2], z[150:, 2] = 90.0, 0.04  # world 2 stays lying
    m = ac.roulade_metrics({"tilt": tilt, "z": z, "reset": np.array([False, False, False, True]), "dt": DT})
    assert m["roll"] == 0.25  # world 0 only: 1 never rolled, 2 did not land, 3 reset
    assert m["rolled"] == 0.75 and m["landed"] == 0.75 and m["worlds"] == 4
    assert m["tilt_max_median_deg"] == 170.0  # a half roll to 133 degrees would show here, below the 150 needed


def test_all_six_contracts_can_be_declared_and_have_defaults():
    assert ac.problems({"acceptance": ["posture", "velocity", "getup", "kick", "pick", "roulade"]}) == []
    assert set(ac.DEFAULT_CRITERIA) == {"posture", "velocity", "getup", "kick", "pick", "roulade"}


# -- warnings: criteria that do not gate
def test_warn_criteria_are_declarable_and_typos_are_still_caught():
    assert ac.problems({"acceptance": {"contract": "velocity", "criteria": {"trained": {"reset_warn_max": 0.05}}}}) == []
    assert ac.problems({"acceptance": {"contract": "velocity", "criteria": {"trained": {"nope_warn_max": 0.05}}}})
    assert ac.problems({"acceptance": {"contract": "velocity", "criteria": {"trained": {"reset_warn_mean": 0.05}}}})


def test_a_warn_criterion_never_fails_the_policy_but_is_listed():
    criteria = {"trained": {"reset_max": 0.25, "reset_warn_max": 0.10}}
    checks = ac.evaluate({"trained": {"reset": 0.129}}, criteria)
    assert [(c["metric"], c["gate"], c["passed"]) for c in checks] == [("reset", True, True), ("reset", False, False)]
    assert ac.decide(checks) == ("ACCEPTED", [])
    assert len(ac.warnings_of(checks)) == 1 and "0.129" in ac.warnings_of(checks)[0]
    failed = ac.evaluate({"trained": {"reset": 0.3}}, criteria)
    assert ac.decide(failed)[0] == "FAILED"
    assert ac.warnings_of(ac.evaluate({"trained": {"reset": 0.05}}, criteria)) == []
    assert ac.decide(ac.evaluate({"trained": {}}, criteria))[0] == "UNVERIFIED"  # the hard line has no value
    only_warn = ac.evaluate({"trained": {}}, {"trained": {"reset_warn_max": 0.1}})
    assert ac.decide(only_warn)[0] == "UNVERIFIED" and ac.warnings_of(only_warn) == []  # no value is not a warning


def test_the_velocity_defaults_have_two_lines_for_resets_in_the_trained_world():
    trained = ac.DEFAULT_CRITERIA["velocity"]["trained"]
    assert trained["reset_warn_max"] < trained["reset_max"]


def test_rejudge_moves_a_stored_result_to_the_current_criteria_without_a_run():
    stored = {"task": "T", "policy": {"onnx": "p.onnx", "sha256": "ab" * 32}, "errors": [], "notes": [],
              "verdict": "FAILED", "reasons": ["velocity trained reset 0.129 (needs <= 0.1)"],
              "contracts": [{"contract": "velocity", "metrics": {
                  "clean": {"reset": 0.0, "forward": 0.83, "turn": 0.85},
                  "trained": {"reset": 0.129, "forward": 0.6, "turn": 0.3}}}],
              "checks": [{"contract": "velocity", "variant": "trained", "metric": "reset", "op": "<=", "threshold": 0.1,
                          "value": 0.129, "passed": False, "gate": True}, ac.onnx_check(1e-7)]}
    new = ac.rejudge(stored)
    assert new["verdict"] == "ACCEPTED" and len(new["warnings"]) == 1
    assert new["rejudged_from"]["verdict"] == "FAILED" and stored["verdict"] == "FAILED"  # the original is untouched
    assert any(c["metric"] == "onnx_vs_checkpoint_max_abs_diff" for c in new["checks"])  # generic checks carried over
    text = ac.render(new)
    assert text.startswith("# Acceptance: ACCEPTED (with 1 warning)") and "| warning |" in text
    assert "the run's own verdict was FAILED" in text
    stored["errors"] = ["velocity: crashed"]
    assert ac.rejudge(stored)["verdict"] == "UNVERIFIED"  # a crash does not become a pass by re-judging


def test_pick_metrics_need_the_mouth_down_and_the_duck_standing_after_each_cycle():
    n, dt = 5, DT
    steps = round(8.0 / dt)
    t_ = (np.arange(steps) + 1) * dt
    mouth = np.full((steps, n), 0.21)
    tilt, z = np.full((steps, n), 3.0), np.full((steps, n), 0.115)
    for w in (0, 2, 3, 4):  # worlds 0, 2, 3, 4 reach down in the first descent (to 3 cm, or only 12 cm for world 3)
        mouth[(t_ > 0.5) & (t_ <= 1.8), w] = 0.03 if w != 3 else 0.12
    z[(t_ > 3.0) & (t_ <= 4.0), 2] = 0.05  # world 2 is still down when the first cycle ends
    tilt[(t_ > 7.0), 4] = 60.0  # world 4 is not upright when the second cycle ends
    m = ac.pick_metrics({"mouth_z": mouth, "tilt": tilt, "z": z, "reset": np.array([False, False, False, False, True]), "dt": dt})
    assert m["reach"] == 0.6 and m["back"] == 0.6  # reach: 0, 2, 4; back: 0, 1, 3
    assert m["pick"] == 0.2  # world 0 only: 2 did not stand again, 3 did not reach, 4 reset, 1 never went down
    assert m["reset"] == 0.2 and m["worlds"] == 5 and abs(m["mouth_min_median_cm"] - 3.0) < 1e-9


def test_kick_metrics_report_where_the_ball_went_without_gating_on_it():
    steps = 250
    fwd = np.zeros((steps, 3))
    lat = np.zeros((steps, 3))
    fwd[50:150, :] = 1.0  # 1 m/s for 2 s: 2 m along the kick direction in each world
    lat[50:150, 0] = 0.0  # straight
    lat[50:150, 1] = 0.5  # to the left: atan(0.5) = 26.6 degrees
    lat[50:150, 2] = -0.5  # to the right
    m = ac.kick_metrics({"ball_fwd": fwd, "ball_lat": lat, "reset": np.zeros(3, bool), "dt": DT})
    assert abs(m["angle_median_deg"]) < 1e-9 and abs(m["angle_abs_median_deg"] - 26.565) < 0.01
    assert m["kick"] == 1.0  # the angle is information, not a criterion
    assert "angle_median_deg" not in ac.kick_metrics({"ball_fwd": fwd, "reset": np.zeros(3, bool), "dt": DT})
