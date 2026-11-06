"""queue_runner.py without a GPU: console parsing, convergence, stage planning and the two-stage gate."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import queue_runner  # noqa: E402
from queue_runner import ConsoleParser, Queue, converged  # noqa: E402

BLOCK = "\x1b[1m Learning iteration {it}/{tot} \x1b[0m\n  Mean reward: {r}\n  Mean episode length: 10.0\n"


def test_console_parser_reads_whole_lines_only(tmp_path):
    log = tmp_path / "seg.log"
    log.write_text(BLOCK.format(it=7, tot=20, r=1.5) + "  Learning iteration 8/20\n  Mean rew")
    rewards = {}
    parser = ConsoleParser(log)
    parser.read(rewards)
    assert rewards == {7: 1.5}
    with open(log, "a") as f:
        f.write("ard: -2.25\n" + BLOCK.format(it=9, tot=20, r="3e-1"))
    parser.read(rewards)
    assert rewards == {7: 1.5, 8: -2.25, 9: 0.3}


def test_converged_only_after_the_curricula_and_two_windows():
    flat = {i: 100.0 for i in range(0, 3000)}
    assert converged(flat, 1999, 0, 1000) is None  # 999 iterations past the curricula
    v = converged(flat, 2099, 0, 1000)
    assert v and abs(v["gain"]) < 1e-9
    rising = {i: 100.0 + 0.01 * i for i in range(0, 3000)}  # +5 per 500 iterations: 4-5%
    assert converged(rising, 2099, 0, 1000) is None
    resumed = {i: 50.0 for i in range(10_000, 11_100)}
    assert converged(resumed, 10_999, 10_000, 0) is None  # 999 iterations into the stage
    assert converged(resumed, 11_000, 10_000, 0) is not None


def make_queue(tmp_path, stages, gate=None, done=()):
    qfile = tmp_path / "q.json"
    qfile.write_text(json.dumps({"name": "t", "deadline": "2099-01-01 08:00", "stages": stages}))
    q = Queue.__new__(Queue)
    q.args, q.spec, q.dry = SimpleNamespace(only=None), json.loads(qfile.read_text()), False
    q.state = {"gate": gate or {}, "stages": {s: {"status": "done"} for s in done}, "order": []}
    return q


def test_plan_uses_the_fallback_when_a_dependency_failed(tmp_path):
    q3 = {"id": "Q3", "skill": "walk", "task": "VelStand", "iters": 10, "needs": ["Q1", "Q2"], "needs_gate": "two_stage",
          "fallback": {"task": "Velocity-Rough-Backlash", "needs": []}}
    q = make_queue(tmp_path, [q3], gate={"two_stage": {"passed": True}}, done=("Q1",))
    plan = q.plan(q3)
    assert plan["task"] == "Velocity-Rough-Backlash" and "Q2" in plan["note"] and "fallback" not in plan
    q = make_queue(tmp_path, [q3], gate={"two_stage": {"passed": True}}, done=("Q1", "Q2"))
    assert q.plan(q3)["task"] == "VelStand"
    q = make_queue(tmp_path, [q3], gate={"two_stage": {"passed": False}}, done=("Q1", "Q2"))
    assert "gate" in q.plan(q3)["skip"]


def test_two_stage_gate(tmp_path):
    spec = {"gate_rules": {"Flat": 1.5, "Backlash": 1.2}, "eval": {"init_label": "before"}}
    evals = []
    for task, before, after in (("Flat", 0.04, 0.05), ("Backlash", 0.06, 0.05)):
        p = tmp_path / f"eval_{task}_s0.json"
        p.write_text(json.dumps({"rows": [{"label": "before", "falls_per_world_minute": before},
                                          {"label": "final", "falls_per_world_minute": after}]}))
        evals.append(str(p))
    rec = {"evals": evals, "rewards": {str(i): 100.0 for i in range(500)}}
    q = make_queue(tmp_path, [])
    assert q.gate(spec, rec, tmp_path)["passed"]
    rec["rewards"] = {str(i): 100.0 if i < 250 else 70.0 for i in range(500)}  # reward collapsed by 30%
    assert not q.gate(spec, rec, tmp_path)["passed"]
    rec["rewards"] = {str(i): 100.0 for i in range(500)}
    spec["gate_rules"]["Flat"] = 0.9  # 0.05 > 0.9 x 0.04 + 0.01
    assert not q.gate(spec, rec, tmp_path)["passed"]


def test_stage_report_renders(tmp_path):
    ev = tmp_path / "eval_T_s0.json"
    row = {"label": "final", "falls_per_world_minute": 0.1, "fell_share": 0.05, "upright_share": 0.9,
           "tip_overs_per_world_minute": 0.2, "recoveries_per_world_minute": 0.1, "err_vxy_mps": None,
           "reward_per_step": {"a": 0.5, "b": -0.1}, "reward_raw_per_step": {"a": 1.0},
           "terminations_per_world_minute": {"time_out": 3.0}}
    ev.write_text(json.dumps({"rows": [row]}))
    rec = {"id": "Q1", "skill": "standup", "task": "T", "init": {"mode": "scratch"}, "segments": ["x/run"],
           "rewards": {"0": 1.0, "1": 2.0}, "evals": [str(ev)], "notes": ["n"]}
    text = "\n".join(queue_runner.stage_report({}, rec, tmp_path, None))
    assert "| final | 0.100 |" in text and "| a | 0.5000 | 1.0000 |" in text and "| time_out | 3.000 |" in text


# -- acceptance
def test_a_queue_with_an_undeclared_acceptance_does_not_start():
    fallback = {"task": "T", "acceptance": {"contract": "velocity", "criteria": {"clean": {"nope_min": 1}}}}
    spec = {"stages": [{"id": "Q1", "skill": "a"}, {"id": "Q2", "acceptance": "none"}, {"id": "Q3", "acceptance": "posutre"},
                       {"id": "Q4", "acceptance": "posture", "fallback": fallback}]}
    with pytest.raises(SystemExit) as e:
        queue_runner.check_declarations(spec)
    msg = str(e.value)
    assert "Q1:" in msg and "Q3:" in msg and "Q4 fallback:" in msg and "Q2:" not in msg
    queue_runner.check_declarations({"stages": [{"id": "Q1", "acceptance": "none"},
                                                {"id": "Q2", "acceptance": ["velocity", "getup"]}]})


def test_a_stage_whose_acceptance_failed_does_not_feed_the_next_one(tmp_path):
    q = make_queue(tmp_path, [], done=("Q1", "Q2", "Q3"))
    for sid, verdict in (("Q1", "FAILED"), ("Q2", "UNVERIFIED"), ("Q3", "ACCEPTED")):
        q.state["stages"][sid]["verdict"] = verdict
    assert not q.succeeded("Q1") and q.succeeded("Q2") and q.succeeded("Q3")
    stage = {"id": "Q9", "skill": "s", "task": "T", "iters": 10, "needs": ["Q1"], "acceptance": "none"}
    assert "Q1" in q.plan(stage)["skip"]
    assert q.plan({**stage, "needs": ["Q3"]})["task"] == "T"


def acceptance_queue(tmp_path, answer=None, ran=True):
    q = make_queue(tmp_path, [])
    q.log, q.calls = (lambda m: None), []

    def run_logged(name, cmd, timeout, env=None):
        q.calls.append(cmd)
        if answer is not None:
            (tmp_path / "acceptance.json").write_text(json.dumps(answer))
        return ran

    q.run_logged = run_logged
    return q


def test_acceptance_none_or_a_missing_export_is_unverified(tmp_path):
    q = acceptance_queue(tmp_path)
    rec = {"id": "Q5"}
    q.accept({"id": "Q5", "task": "T", "acceptance": "none"}, rec, tmp_path / "m.pt", tmp_path, {})
    assert rec["verdict"] == "UNVERIFIED" and "no acceptance contract" in rec["verdict_why"][0] and not q.calls
    rec = {"id": "Q4"}
    q.accept({"id": "Q4", "task": "T", "acceptance": "posture"}, rec, tmp_path / "m.pt", tmp_path, {})
    assert rec["verdict"] == "UNVERIFIED" and "policy.onnx" in rec["verdict_why"][0] and not q.calls


def test_acceptance_reads_the_verdict_the_check_wrote(tmp_path):
    (tmp_path / "policy.onnx").write_bytes(b"x")
    q = acceptance_queue(tmp_path, answer={"verdict": "FAILED", "reasons": ["posture clean sit_down 0 (needs >= 0.8)"]})
    rec = {"id": "Q4"}
    q.accept({"id": "Q4", "task": "Task-T", "acceptance": {"contract": "posture"}}, rec, tmp_path / "m.pt", tmp_path, {})
    assert rec["verdict"] == "FAILED" and rec["verdict_why"] == ["posture clean sit_down 0 (needs >= 0.8)"]
    cmd = q.calls[0]
    assert cmd[cmd.index("--task") + 1] == "Task-T" and json.loads(cmd[cmd.index("--spec") + 1]) == {"contract": "posture"}
    assert cmd[cmd.index("--checkpoint") + 1].endswith("m.pt") and "--num-envs" not in cmd


def test_a_check_that_did_not_finish_is_unverified_and_never_reads_an_old_answer(tmp_path):
    (tmp_path / "policy.onnx").write_bytes(b"x")
    (tmp_path / "acceptance.json").write_text(json.dumps({"verdict": "ACCEPTED", "reasons": []}))  # left by an earlier try
    q = acceptance_queue(tmp_path, answer=None, ran=True)  # exits 0 but writes nothing
    rec = {"id": "Q4"}
    q.accept({"id": "Q4", "task": "T", "acceptance": "posture"}, rec, tmp_path / "m.pt", tmp_path, {})
    assert rec["verdict"] == "UNVERIFIED" and "did not finish" in rec["verdict_why"][0]
    q = acceptance_queue(tmp_path, answer={"verdict": "ACCEPTED", "reasons": []}, ran=False)  # crashed, left an answer
    rec = {"id": "Q4"}
    q.accept({"id": "Q4", "task": "T", "acceptance": "posture"}, rec, tmp_path / "m.pt", tmp_path, {})
    assert rec["verdict"] == "UNVERIFIED"


def test_the_summary_and_the_stage_report_show_the_verdict(tmp_path):
    q = make_queue(tmp_path, [])
    q.qdir, q.summary_dir, q.deadline = tmp_path / "queue", tmp_path / "summary", 4e9
    q.qdir.mkdir()
    q.state.update(queue="t", queue_file=str(tmp_path / "q.json"), started="2026-01-01 00:00:00")
    base = {"skill": "s", "task": "T", "status": "done", "stop_reason": "completed", "rewards": {}, "iters": 10}
    q.state["stages"] = {"Q4": {**base, "id": "Q4", "verdict": "FAILED"}, "Q5": {**base, "id": "Q5", "verdict": "UNVERIFIED"},
                         "Q6": {**base, "id": "Q6", "verdict": "ACCEPTED"}}
    q.state["order"] = ["Q4", "Q5", "Q6"]
    q.write_summary()
    text = (tmp_path / "summary" / "README.md").read_text()
    assert "| verdict |" in text and "**FAILED**" in text and "**ACCEPTED**" in text
    assert "1 ACCEPTED, 1 FAILED, 1 UNVERIFIED" in text
    (tmp_path / "acceptance.md").write_text("x")
    rec = {"id": "Q4", "skill": "sitstand", "task": "T", "init": {"mode": "scratch"}, "segments": ["x/run"],
           "rewards": {"0": 1.0}, "evals": [], "notes": [], "verdict": "FAILED", "verdict_why": ["posture clean sit_down 0"]}
    report = "\n".join(queue_runner.stage_report({}, rec, tmp_path, None))
    assert "Acceptance: **FAILED** — posture clean sit_down 0. Details: [acceptance.md](acceptance.md)." in report


def test_a_verdict_with_warnings_shows_in_the_summary_and_the_stage_report(tmp_path):
    q = make_queue(tmp_path, [])
    q.log = lambda m: None
    rec = {"id": "Q2"}
    q.set_verdict(rec, "ACCEPTED", [], ["velocity trained reset 0.129 (needs <= 0.1)"])
    assert rec["verdict"] == "ACCEPTED" and len(rec["verdict_warnings"]) == 1
    assert queue_runner.verdict_cell(rec) == "**ACCEPTED** (1 warning)"
    assert queue_runner.verdict_cell({"verdict": "FAILED", "verdict_warnings": []}) == "**FAILED**"
    assert queue_runner.verdict_cell({}) == "-"
    q.state["stages"] = {"Q2": {"id": "Q2", "skill": "walk", "task": "T", "status": "done", "stop_reason": "completed",
                                "rewards": {}, "iters": 10, **{k: v for k, v in rec.items() if k != "id"}}}
    q.state["order"] = ["Q2"]
    q.qdir, q.summary_dir, q.deadline = tmp_path / "queue", tmp_path / "summary", 4e9
    q.qdir.mkdir()
    q.state.update(queue="t", queue_file=str(tmp_path / "q.json"), started="2026-01-01 00:00:00")
    q.write_summary()
    assert "**ACCEPTED** (1 warning)" in (tmp_path / "summary" / "README.md").read_text()
    full = {"id": "Q2", "skill": "walk", "task": "T", "init": {"mode": "scratch"}, "segments": ["x/run"],
            "rewards": {"0": 1.0}, "evals": [], "notes": [], **rec}
    report = "\n".join(queue_runner.stage_report({}, full, tmp_path, None))
    assert "Acceptance: **ACCEPTED**" in report and "Acceptance warnings (they do not fail the stage): velocity trained reset" in report


def test_stage_env_values_are_resolved_only_when_they_name_a_stage_or_a_path(tmp_path):
    q = make_queue(tmp_path, [], done=("Q1",))
    q.state["stages"]["Q1"]["final_checkpoint"] = "/x/model_5.pt"
    assert q.stage_env_value("0.05") == "0.05" and q.stage_env_value("1") == "1"
    assert q.stage_env_value("@Q1") == "/x/model_5.pt"
    assert q.stage_env_value("logs/a/model_1.pt") == str(queue_runner.TRAIN / "logs/a/model_1.pt")
