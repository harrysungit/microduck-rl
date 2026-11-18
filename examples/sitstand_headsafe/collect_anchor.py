"""13s GPU teacher consistency diagnostic, not independent CPU acceptance."""
import os
os.environ['MUJOCO_GL']='egl'
os.environ['MICRODUCK_WARM_START']='0'
os.environ['ROCPROFILER_REGISTER_ENABLED']='0'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import argparse
import subprocess
import json
import hashlib
import numpy as np
import torch
import onnxruntime as ort
if __package__:
    from . import headsafe_task as task
else:
    import headsafe_task as task



@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--teacher',type=Path,required=True)
    ap.add_argument('--calibration',type=Path,required=True)
    ap.add_argument('--profile',choices=['nominal','operating'],default='nominal')
    ap.add_argument('--initial-posture',choices=['stand','sit'],default='stand')
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    cfg,_=task.register(1,args.calibration)
    if __package__:
        from .operating_profile import apply
    else:
        from operating_profile import apply
    profile_record=apply(cfg,args.profile)
    cfg.auto_reset=False
    cfg.episode_length_s=13.1
    cfg.terminations.pop('head_contact')
    cfg.observations['actor'].enable_corruption=False
    cfg.viewer.width,cfg.viewer.height=720,480
    cfg.viewer.distance=.7
    cfg.viewer.elevation=-15.
    cfg.viewer.azimuth=120.
    env=task.HeadSafeEnv(cfg,'cuda:0',render_mode='rgb_array')
    env.initial_pose_override=float(args.initial_posture=='sit')
    obs,_=env.reset()
    session=ort.InferenceSession(str(args.teacher),providers=['CPUExecutionProvider'])
    name=session.get_inputs()[0].name
    raw=args.out/'full_raw.mp4'
    pipe=subprocess.Popen(['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','720x480','-r','50','-i','-','-an','-c:v','libx264','-preset','veryfast','-crf','20','-pix_fmt','yuv420p',str(raw)],stdin=subprocess.PIPE,stderr=subprocess.DEVNULL)
    rows=[]
    observations=[]
    teacher_actions=[]
    for step in range(650):
        action=session.run(None,{name:obs['actor'].cpu().numpy()})[0]
        observations.append(obs['actor'][0].cpu().numpy())
        teacher_actions.append(action[0])
        obs,reward,terminated,timeout,extra=env.step(torch.tensor(action,device=env.device))
        s=task.state(env)
        sensor=env.scene['head_ground']
        force=float(torch.linalg.vector_norm(sensor.data.force_history,dim=-1).max())
        support=env.scene['sit_support'].data
        q=env.scene['robot'].data.joint_pos[0,env.motor_ids]
        endpoint_error=max(abs(float(q[idx])-value) for idx,value in task.SIT.items())
        rows.append({'t':round((step+1)*.02,4),'command':float(s['command'][0]),'z':float(s['z'][0]),'upright':float(s['upright'][0]),'head_contact_latched':bool(sensor.ever_found[0]),'head_force_peak_N':force,'head_gap_control_sample_m':float(env.head_clearance(env)[0]),'success_count':int(s['success_count'][0]),'sensor_substeps':int(sensor.substep_count[0]),'sit_support_found':support.found[0].tolist(),'sit_support_force_N':support.force[0].norm(dim=-1).tolist(),'sit_endpoint_max_joint_error_rad':endpoint_error,'legacy_right_hip_contact_latched':bool(env.scene['legacy_hip_ground'].ever_found[0]),'legacy_right_hip_gap_control_sample_m':float(env.legacy_hip_clearance(env)[0]),'classified_posture':float(s['posture'][0]),'stable_s':float(s['stable_s'][0])})
        pipe.stdin.write(env.render().tobytes())
        # Continue the full failed episode rather than hide the failure.
        env._manual_reset_pending.zero_()
    pipe.stdin.close()
    assert pipe.wait()==0
    video=args.out/'full.mp4'
    schedule='STAND 0-3s - SIT 3-8s - STAND 8-13s' if args.initial_posture=='stand' else 'SIT 0-3s - STAND 3-8s - SIT 8-13s'
    text=f"drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text='GPU SitStand | 1x | CMD {schedule}':x=10:y=12:fontsize=15:fontcolor=white:box=1:boxcolor=black@0.7"
    subprocess.run(['ffmpeg','-y','-i',str(raw),'-vf',text,'-c:v','libx264','-crf','20','-pix_fmt','yuv420p',str(video)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    first=next((r['t'] for r in rows if r['head_contact_latched']),None)
    slow=None
    if first is not None:
        slow=args.out/'contact_4x_slow.mp4'
        subprocess.run(['ffmpeg','-y','-ss',str(max(0,first-.3)),'-t','1.2','-i',str(raw),'-vf',"setpts=4*PTS,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text='4x SLOW GPU head contact':x=10:y=12:fontsize=18:fontcolor=white",'-an',str(slow)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    np.savez(args.out/'teacher_rollout.npz',observations=np.array(observations),actions=np.array(teacher_actions))
    result={'onnx_sha256':hashlib.sha256(args.teacher.read_bytes()).hexdigest(),'condition':'GPU HD1910 allcollisions BAM + calibrated target pipeline; see profile_record','duration_s':13,'command_switch_s':[3,8],'head_contact':first is not None,'first_head_contact_control_end_s':first,'head_force_max_N':max(r['head_force_peak_N'] for r in rows),'min_full_head_gap_control_sample_m':min(r['head_gap_control_sample_m'] for r in rows),'valid_transition_count':rows[-1]['success_count'],'physics_sensor_update_count':rows[-1]['sensor_substeps'],'head_collision_geometries':env.head_clearance.names,'head_gap_coverage':'control_step_sample_only','contact_body_resolution':'true neck subtree aggregate; historical right hip reported separately','automatic_status':'consistency_diagnostic_only','human_review':'pending','deployment_approved':False,'video':video.name,'slow_video':slow.name if slow else None}
    (args.out/'trace.json').write_text(json.dumps(rows))
    result['head_scope_version']='neck-subtree-v2'
    result['legacy_right_hip_contact']=any(r['legacy_right_hip_contact_latched'] for r in rows)
    result['legacy_right_hip_min_gap_control_sample_m']=min(r['legacy_right_hip_gap_control_sample_m'] for r in rows)
    result['legacy_head_labelled_superset_contact']=result['head_contact'] or result['legacy_right_hip_contact']
    result['legacy_head_labelled_superset_min_gap_control_sample_m']=min(result['min_full_head_gap_control_sample_m'],result['legacy_right_hip_min_gap_control_sample_m'])
    result['profile_record']=profile_record
    result['initial_posture']=args.initial_posture
    result['command_schedule']='stand-sit-stand' if args.initial_posture=='stand' else 'sit-stand-sit'
    result['calibration_sha256']=hashlib.sha256(args.calibration.read_bytes()).hexdigest()
    result['anchor_data_sha256']=hashlib.sha256((args.out/'teacher_rollout.npz').read_bytes()).hexdigest()
    (args.out/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
    env.close()


if __name__=='__main__':
    main()
