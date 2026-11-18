"""Review-only packaging evidence and data privacy gates, no device operations."""
from pathlib import Path
from copy import deepcopy
import shutil
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
from package_review import DataLoader,actual_task,no_calibration_values,require_evidence,verify_pipeline_source
import yaml

POLICY='1'*64
CHECKPOINT='2'*64


def complete_summary():
    audit={'head_scope_version':'neck-subtree-v2','head_scope_source':'parent-link neck subtree',
           'head_bodies':['neck','neck_pitch','yaw_roll_motion','jaw_soft'],'head_geometry_count':52,
           'head_contact':False,'head_contact_substeps':0,'first_contact_s':None,
           'physics_substeps':6500,'expected_physics_substeps':6500,'coverage':1.,'min_clearance_m':.08}
    rows=[{'condition':condition,'seed':0,'automatic_pass':True,'legal_sit_support_pass':True,'safety':deepcopy(audit)}
          for condition in ('operating-standing','operating-seated','stress')]
    return {'policy_sha256':POLICY,'checkpoint_sha256':CHECKPOINT,'operating_pass':True,'automatic_pass':True,
            'automatic_verdict':'PASS','declared_checks':[f'{row["condition"]}:0' for row in rows],'rows':rows}

def test_evidence_is_model_bound_and_failed_or_pending_cannot_be_final():
    good=complete_summary()
    assert require_evidence(good,POLICY,CHECKPOINT)=='automatic_checks_complete_pending_human_video_review'
    for change in ({'policy_sha256':'3'*64},{'operating_pass':False},{'automatic_verdict':'FAIL'},
                   {'automatic_pass':False},{'head_contact':True},{'checkpoint_sha256':'4'*64},{'diagnostic_only':True}):
        with pytest.raises(ValueError):require_evidence({**good,**change},POLICY,CHECKPOINT)


@pytest.mark.parametrize('failure',[
    {'automatic_pass':False}, {'automatic_verdict':'FAIL'}, {'verdict':'FAILED'},
    {'status':'UNVERIFIED'}, {'head_contact':True}, {'head_contact':2},
    {'legal_sit_support_pass':False}, {'error':'sensor collection failed'}])
def test_nested_failure_cannot_hide_behind_top_level_operating_pass(failure):
    summary=complete_summary()
    summary['rows'][-1]['nested_result']={'stress_details':[failure]}
    with pytest.raises(ValueError):require_evidence(summary,POLICY,CHECKPOINT)


def test_actual_stress_head_hit_row_blocks_even_if_operating_is_true():
    summary=complete_summary()
    summary['rows'][-1]['automatic_pass']=False
    summary['rows'][-1]['safety']['head_contact']=True
    summary['rows'][-1]['safety']['head_contact_substeps']=1
    with pytest.raises(ValueError):require_evidence(summary,POLICY,CHECKPOINT)


@pytest.mark.parametrize('field',['automatic_pass','declared_checks','rows'])
def test_missing_top_level_audit_never_becomes_a_fake_pass(field):
    summary=complete_summary();summary.pop(field)
    with pytest.raises(ValueError):require_evidence(summary,POLICY,CHECKPOINT)


@pytest.mark.parametrize('field',['head_scope_version','head_scope_source','head_bodies','head_geometry_count',
    'head_contact','head_contact_substeps','first_contact_s','physics_substeps','expected_physics_substeps',
    'coverage','min_clearance_m'])
def test_required_row_missing_physics_evidence_is_rejected(field):
    summary=complete_summary();summary['rows'][0]['safety'].pop(field)
    with pytest.raises(ValueError):require_evidence(summary,POLICY,CHECKPOINT)


@pytest.mark.parametrize('changes',[{'physics_substeps':6499},{'coverage':.999},{'coverage':float('nan')},
    {'head_bodies':['neck','neck_pitch','yaw_roll_motion','jaw_soft','bearing_roll']},
    {'first_contact_s':3.8},{'min_clearance_m':-.001}])
def test_incomplete_or_legacy_row_cannot_be_claimed_complete(changes):
    summary=complete_summary();summary['rows'][0]['safety'].update(changes)
    with pytest.raises(ValueError):require_evidence(summary,POLICY,CHECKPOINT)


def test_declared_scope_cannot_omit_stress_or_duplicate_a_success():
    for change in ('omit_row','omit_declared','duplicate_row','empty_checks','missing_support'):
        summary=complete_summary()
        if change=='omit_row':summary['rows'].pop()
        elif change=='omit_declared':summary['declared_checks'].pop()
        elif change=='duplicate_row':summary['rows'].append(deepcopy(summary['rows'][0]))
        elif change=='empty_checks':summary['declared_checks']=[]
        else:summary['rows'][0].pop('legal_sit_support_pass')
        with pytest.raises(ValueError):require_evidence(summary,POLICY,CHECKPOINT)


def test_true_head_success_does_not_reclassify_a_legal_hip_contact_as_head_hit():
    summary=complete_summary()
    summary['rows'][0]['safety']['legacy_right_hip']={'contact':True,'first_contact_s':4.2}
    assert require_evidence(summary,POLICY,CHECKPOINT)=='automatic_checks_complete_pending_human_video_review'


def test_diagnostic_format_probe_explicitly_denies_candidate_status():
    diagnostic={'policy_sha256':POLICY,'operating_pass':False,'diagnostic_only':True}
    assert require_evidence(diagnostic,POLICY,CHECKPOINT,True)=='diagnostic_only_not_a_candidate'
    with pytest.raises(ValueError):require_evidence(diagnostic,POLICY,CHECKPOINT)
    with pytest.raises(ValueError):require_evidence({**diagnostic,'operating_pass':True},POLICY,CHECKPOINT,True)


def test_device_calibration_values_cannot_enter_contract():
    no_calibration_values({'actions':{'calibration_sha256':'a'*64}})
    for key in ('zero_raw','min_rad','max_rad','direction','calibration','device_calibration'):
        with pytest.raises(ValueError):no_calibration_values({'nested':[{'key':{key:1}}]})


def test_yaml_python_tags_are_inert_data(tmp_path):
    marker=tmp_path/'must-not-exist'
    value=yaml.load('!!python/object/apply:os.system ["touch '+str(marker)+'"]',Loader=DataLoader)
    assert value['type']=='object/apply:os.system' and not marker.exists()


def test_task_and_runtime_mapping_are_bound_to_archived_source(tmp_path):
    recipe=Path(__file__).resolve().parent
    assert actual_task(recipe/'headsafe_task.py').startswith('Mjlab-SitStand-')
    for name in ('runtime_actions.py','runtime_state.py'):shutil.copyfile(recipe/name,tmp_path/name)
    assert verify_pipeline_source(tmp_path)['state_ast_sha256']
    path=tmp_path/'runtime_state.py';path.write_text(path.read_text().replace('[.7]*5','[.8]*5'))
    with pytest.raises(ValueError,match='algorithm differs'):verify_pipeline_source(tmp_path)
