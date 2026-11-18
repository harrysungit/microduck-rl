"""Regression: a transient head hit survives later recovery; neck geometry is not omitted."""
from pathlib import Path
import sys
import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import acceptance_core as ac
from contact_audit import Geometry, HeadAudit
import sim2sim


XML = """<mujoco><option timestep="0.002"/><worldbody>
<geom name="floor" type="plane" size="1 1 .1"/>
<body name="neck" pos="0 0 .2"><freejoint/>
 <geom name="neck_shell" type="box" size=".02 .02 .02" mass=".1"/>
 <body name="jaw_soft" pos="0 0 .08"><geom name="head_shell" type="sphere" size=".03" mass=".1"/></body>
</body></worldbody></mujoco>"""


def test_neck_clearance_and_collision_are_latched_after_recovery():
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    audit = HeadAudit(model, body_names=("neck", "jaw_soft"))
    mujoco.mj_forward(model, data)
    audit.sample(data, 0, 0)
    assert audit.result()["min_clearance_m"] == pytest.approx(.18)
    data.qpos[2] = .019
    mujoco.mj_forward(model, data)
    audit.sample(data, 1., .002)
    data.qpos[2] = .2
    mujoco.mj_forward(model, data)
    audit.sample(data, 2., .002)
    result = audit.result()
    assert result["head_contact"] and result["first_contact_s"] == 1.
    assert result["min_clearance_m"] == pytest.approx(-.001)
    assert result["min_clearance_geometry"] == "neck/neck_shell"
    assert result["head_contact_substeps"] == 1
    assert result["peak_single_contact_normal_N"] > 0


def test_missing_head_body_is_not_silently_a_safe_robot():
    model = mujoco.MjModel.from_xml_string(XML)
    with pytest.raises(KeyError):
        HeadAudit(model)


def test_endpoint_success_without_substep_audit_is_unverified_and_contact_fails():
    metrics = {v: {"sit_down": 1., "stand_up": 1., "tipped": 0.} for v in ac.VARIANTS}
    criteria = ac.effective_criteria("posture", {"clean": {"head_contact_max": 1., "head_audit_coverage_min": 0.}})
    assert criteria["clean"]["head_contact_max"] == 0 and criteria["clean"]["head_audit_coverage_min"] == 1
    assert ac.decide(ac.evaluate(metrics, criteria))[0] == "UNVERIFIED"
    for value in metrics.values():
        value.update(head_contact=0., head_audit_coverage=1.)
    metrics["clean"]["head_contact"] = .001
    assert ac.decide(ac.evaluate(metrics, criteria))[0] == "FAILED"
    assert ac.decide(ac.known_failure_checks("6f6c06e830d32023556f66d38c495e9b54ff6625448246896221875269e1dc20"))[0] == "FAILED"


def test_remaining_seated_is_not_a_successful_sit_down():
    t = np.arange(.02, 9.01, .02)
    trace = {"t": t, "z": np.full_like(t, .06), "tilt": np.zeros_like(t),
             "vx": np.zeros_like(t), "vy": np.zeros_like(t)}
    schedule = [(3, (0., 0, 0), "stand"), (3, (1., 0, 0), "sit"), (3, (0., 0, 0), "stand")]
    rows = sim2sim.posture_segments(trace, schedule)
    assert not rows[1]["ok"] and not rows[1]["origin_posture_ok"]


@pytest.mark.parametrize("bad", [np.float64(1.), np.float32(1.)])
def test_numpy_scalar_failure_is_a_real_gating_failure(bad):
    checks = ac.evaluate({"clean": {"head_contact": bad}}, {"clean": {"head_contact_max": 0.}})
    assert ac.decide(checks)[0] == "FAILED"


@pytest.mark.parametrize("bad", ["0", [], {}, True, np.array([0., 0.]), np.nan, np.inf])
def test_invalid_metric_never_becomes_accepted(bad):
    checks = ac.evaluate({"clean": {"head_contact": bad}}, {"clean": {"head_contact_max": 0.}})
    assert ac.decide(checks)[0] == "UNVERIFIED"


def test_truncated_stable_posture_is_not_a_full_segment():
    t = np.arange(.02, .51, .02)
    trace = {"t": t, "z": np.full_like(t, .12), "tilt": np.zeros_like(t),
             "vx": np.zeros_like(t), "vy": np.zeros_like(t)}
    rows = sim2sim.posture_segments(trace, [(3., (0., 0., 0.), "stand")])
    assert not rows[0]["ok"]


def test_mesh_clearance_accounts_for_compiler_mesh_frame_and_world_rotation():
    vertices = np.array([[-.02, -.01, -.03], [.04, 0, 0], [0, .03, 0], [0, 0, .05]])
    encoded = " ".join(str(v) for v in vertices.ravel())
    model = mujoco.MjModel.from_xml_string(f'''<mujoco><asset><mesh name="shell" vertex="{encoded}"/></asset>
      <worldbody><body pos="0 0 .2" euler="31 17 23"><geom name="head" type="mesh" mesh="shell"/>
      </body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    # The compiler recenters/reorients the mesh. Compare against the original authored vertices in body space.
    expected = .2 + (vertices @ data.xmat[1].reshape(3, 3)[2]).min()
    assert Geometry(model).minimum_z(data, 0) == pytest.approx(expected, abs=1e-8)
