"""Regression: a transient head hit survives later recovery; neck geometry is not omitted."""
from pathlib import Path
import sys
import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import acceptance_core as ac
from contact_audit import Geometry, HeadAudit, legal_sit_support
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


def test_sit_support_requires_loaded_non_foot_bodies_on_both_sides_or_torso():
    assert not legal_sit_support({"ankle_left": 4., "ankle_right": 4.})
    assert not legal_sit_support({"leg": 4., "ankle_right": 4.})
    assert not legal_sit_support({"leg": .05, "leg_2": .05})
    assert legal_sit_support({"leg": .051, "leg_2": .051})
    assert legal_sit_support({"upper_leg_left": 2., "hip_l_2": 2.})
    assert legal_sit_support({"trunk_base": 1.})


def test_substep_audit_records_loaded_support_not_merely_geometric_contact():
    model = mujoco.MjModel.from_xml_string(XML.replace('</worldbody>', '''
      <body name="leg" pos="-.05 0 .019"><freejoint/><geom type="box" size=".02 .02 .02" mass=".1"/></body>
      <body name="leg_2" pos=".05 0 .019"><freejoint/><geom type="box" size=".02 .02 .02" mass=".1"/></body>
      </worldbody>'''))
    data = mujoco.MjData(model)
    audit = HeadAudit(model, body_names=("neck", "jaw_soft"))
    mujoco.mj_step(model, data)
    audit.sample(data, 0., .002)
    result = audit.result()
    assert len(result["support_substep_trace"]) == 1
    time, loaded, legal = result["support_substep_trace"][0]
    assert time == 0. and loaded["leg"] > .05 and loaded["leg_2"] > .05
    assert legal


def test_posture_acceptance_requires_verified_legal_seat_and_cannot_relax_it():
    criteria = ac.effective_criteria('posture', {'clean': {'legal_sit_support_min': 0.}})
    assert criteria['clean']['legal_sit_support_min'] == 1.
    metrics = {variant: {'sit_down': 1., 'stand_up': 1., 'tipped': 0., 'head_contact': 0.,
                         'head_audit_coverage': 1.} for variant in ac.VARIANTS}
    assert ac.decide(ac.evaluate(metrics, criteria))[0] == 'UNVERIFIED'
    for row in metrics.values():
        row['legal_sit_support'] = 0.
    assert ac.decide(ac.evaluate(metrics, criteria))[0] == 'FAILED'


ANATOMY_XML='''<mujoco><option timestep=".002"/><worldbody>
<geom name="floor" type="plane" size="1 1 .1"/>
<body name="trunk_base" pos="0 0 .2"><freejoint/><geom type="sphere" size=".01" mass=".1"/>
 <body name="neck" pos="0 0 .15"><geom name="neck_shell" type="box" size=".02 .02 .02" mass=".1"/>
  <body name="neck_pitch" pos="0 0 .03"><geom type="sphere" size=".01" mass=".01"/>
   <body name="yaw_roll_motion" pos="0 0 .03"><geom type="sphere" size=".01" mass=".01"/>
    <body name="jaw_soft" pos="0 0 .03"><geom name="top_head_shell" type="sphere" size=".02" mass=".1"/>
     <body name="extra_head_part" pos="0 0 .1"><geom type="capsule" size=".01 .02" mass=".01"/></body>
    </body>
   </body>
  </body>
 </body>
 <body name="bearing_roll" pos=".1 0 -.181"><geom name="right_hip_shell" type="box" size=".02 .02 .02" mass=".1"/></body>
</body></worldbody></mujoco>'''


def test_neck_tree_covers_extra_descendant_but_right_hip_is_not_head():
    model=mujoco.MjModel.from_xml_string(ANATOMY_XML)
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    audit=HeadAudit(model);audit.sample(data,0.,.002)
    result=audit.result()
    assert set(result['head_bodies'])=={'neck','neck_pitch','yaw_roll_motion','jaw_soft','extra_head_part'}
    assert not result['head_contact'] and result['min_clearance_m']==pytest.approx(.33)
    assert result['legacy_right_hip']['contact'] and result['legacy_right_hip']['parent_body']=='trunk_base'
    assert result['legacy_right_hip']['min_clearance_m']==pytest.approx(-.001)
    assert result['legacy_superset']['contact'] and result['legacy_superset']['min_clearance_m']==pytest.approx(-.001)
    assert result['legacy_superset']['min_clearance_geometry']=='bearing_roll/right_hip_shell'
    assert 'bearing_roll' in result['non_head_ground_bodies']
    assert legal_sit_support({'bearing_roll':1.,'leg':1.})


def test_jaw_hit_still_fails_true_head_scope_after_later_recovery():
    model=mujoco.MjModel.from_xml_string(ANATOMY_XML)
    data=mujoco.MjData(model);audit=HeadAudit(model)
    model.body_pos[model.body('jaw_soft').id,2]=-.4
    mujoco.mj_forward(model,data);audit.sample(data,1.,.002)
    model.body_pos[model.body('jaw_soft').id,2]=.03
    mujoco.mj_forward(model,data);audit.sample(data,2.,.002)
    result=audit.result()
    assert result['head_contact'] and result['first_contact_s']==1.
    assert 'jaw_soft/top_head_shell' in result['contact_geometries']
    assert result['head_contact_substeps']==1


def test_legacy_hip_cannot_be_explicitly_selected_as_anatomical_head():
    model=mujoco.MjModel.from_xml_string(ANATOMY_XML)
    with pytest.raises(ValueError,match='right hip'):
        HeadAudit(model,body_names=('neck','bearing_roll'))
