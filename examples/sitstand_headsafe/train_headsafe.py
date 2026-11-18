"""Explicit, bounded teacher warm start or recipe checkpoint resume. Never deploys."""
import os
os.environ['MICRODUCK_WARM_START']='0'
os.environ['ROCPROFILER_REGISTER_ENABLED']='0'
os.environ['WANDB_MODE']='disabled'
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict
from copy import deepcopy
import numpy as np
if __package__:
    from .bootstrap import sha,bind_inputs,require_resume_identity,validate_anchor
    from .calibration import load_calibration
    from .goals import PostureGoals
else:
    from bootstrap import sha,bind_inputs,require_resume_identity,validate_anchor
    from calibration import load_calibration
    from goals import PostureGoals

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--calibration',type=Path,required=True)
    ap.add_argument('--teacher',type=Path,required=True,help='compatible deterministic ONNX; not a checkpoint resume')
    ap.add_argument('--anchor-data',type=Path,required=True,help='audited external NPZ observations/raw teacher labels')
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--resume',type=Path,help='trusted checkpoint produced by this exact recipe/input identity')
    ap.add_argument('--validate-only',action='store_true',help='CPU input binding/teacher-label check, no simulation or training')
    ap.add_argument('--iterations',type=int,default=5,help='additional updates, at most 200')
    ap.add_argument('--num-envs',type=int,default=64)
    ap.add_argument('--anchor-coef',type=float,default=10.)
    ap.add_argument('--online-anchor-coef',type=float,default=25.)
    ap.add_argument('--critic-warmup-updates',type=int,default=10)
    ap.add_argument('--profile',choices=['nominal','operating'],default='nominal')
    ap.add_argument('--exploration-std',type=float,default=.05)
    ap.add_argument('--review-every',type=int,default=0)
    ap.add_argument('--assets',type=Path,help='CPU sim2sim assets, required for boundary video reviews')
    ap.add_argument('--reviewed-upstream')
    ap.add_argument('--reviewed-policies')
    ap.add_argument('--upstream-review-note')
    args=ap.parse_args()
    if not (0<args.iterations<=200 and 0<args.num_envs<=1024):ap.error('Bounded recipe: 1..200 updates and 1..1024 environments')
    if not (np.isfinite(args.anchor_coef) and args.anchor_coef>0 and np.isfinite(args.exploration_std) and args.exploration_std>0):ap.error('Anchor coefficient and std must be positive finite')
    if not (np.isfinite(args.online_anchor_coef) and args.online_anchor_coef>=0 and args.critic_warmup_updates>=0):ap.error('Online anchor and warmup must be nonnegative')
    if args.review_every<0 or (args.iterations>5 and args.review_every<=0):ap.error('Runs longer than smoke require --review-every')
    if args.review_every and (args.assets is None or not args.assets.is_dir()):ap.error('--assets directory required for video reviews')
    load_calibration(args.calibration)
    anchor_obs,anchor_actions,label_check=validate_anchor(args.teacher,args.anchor_data)
    goals=PostureGoals()
    settings={'posture_goals':goals.record(),'num_envs':args.num_envs,'anchor_coef':args.anchor_coef,'exploration_std':args.exploration_std,
              'seed':181126,'height_tracking_weight':1.,'command_ramp_s':.4,'learning_rate':3e-5,
              'profile':args.profile,'online_anchor_coef':args.online_anchor_coef,'critic_warmup_updates':args.critic_warmup_updates}
    sources=sorted(HERE.glob('*.py'))
    binding=bind_inputs(args.teacher,args.anchor_data,args.calibration,sources,settings)
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'bootstrap.json').write_text(json.dumps({**binding,'label_check':label_check},indent=2))
    if args.validate_only:
        print(json.dumps({'input_validation':'pass','identity_sha256':binding['identity_sha256'],**label_check},indent=2))
        return
    sys.path.insert(0,str(REPO/'scripts'))
    from check_upstream import check
    upstream=check(root=REPO,out_dir=args.out/'upstream',reviewed_sha=args.reviewed_upstream,
                   reviewed_policies=args.reviewed_policies,note=args.upstream_review_note)
    if not upstream['training_allowed']:raise SystemExit('Upstream review is unresolved; no training started')
    os.environ['MICRODUCK_FIXED_STD']=str(args.exploration_std)
    import torch
    if __package__:
        from . import headsafe_task as task
        from .teacher_init import initialize,freeze_normalizer,assert_normalizer_unchanged,actor_warmup
        from .operating_profile import apply
        from .verify_geometry import verify
        from .runtime_identity import snapshot
    else:
        import headsafe_task as task
        from teacher_init import initialize,freeze_normalizer,assert_normalizer_unchanged,actor_warmup
        from operating_profile import apply
        from verify_geometry import verify
        from runtime_identity import snapshot
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_runner_cls
    from mjlab.utils.os import dump_yaml
    from mjlab.rl.exporter_utils import attach_metadata_to_onnx,get_base_metadata
    cfg,agent=task.register(args.num_envs,args.calibration,goals=goals)
    profile_record=apply(cfg,args.profile)
    profile_record['posture_goals']=goals.record()
    (args.out/'profile.json').write_text(json.dumps(profile_record,indent=2))
    cfg.rewards['height_tracking'].weight=1.
    cfg.commands['twist'].ramp_s=.4
    agent.algorithm.learning_rate=3e-5
    agent.algorithm.schedule='fixed'
    agent.actor.distribution_cfg['init_std']=args.exploration_std
    dump_yaml(args.out/'params/env.yaml',asdict(cfg))
    dump_yaml(args.out/'params/agent.yaml',asdict(agent))
    env=task.HeadSafeEnv(cfg,'cuda:0')
    try:
        obs,_=env.reset()
        assert obs['actor'].shape==(args.num_envs,61) and torch.isfinite(obs['actor']).all()
        gap=env.head_clearance(env)
        assert gap.min()>.012 and not env.scene['head_ground'].ever_found.any()
        geometry_check=verify(env)
        runtime_record=snapshot(env,task.TASK,profile_record)
        (args.out/'runtime_identity.json').write_text(json.dumps(runtime_record,indent=2))
        env.safety['failed'][0]=True
        env._safety_tick=env.common_step_counter
        for name in ('height_tracking','endpoint_pose','transition_bonus'):assert getattr(task,name)(env)[0]==0,name
        env.reset()
        wrapped=RslRlVecEnvWrapper(env,clip_actions=agent.clip_actions)
        runner=load_runner_cls(task.BASE)(wrapped,asdict(agent),str(args.out),'cuda:0')
        policy=runner.alg.get_policy()
        assert runner.current_learning_iteration==env.common_step_counter==0 and not runner.alg.optimizer.state
        imported=initialize(policy,args.teacher,actual_obs=obs['actor'])
        frozen_teacher=deepcopy(policy).eval()
        for parameter in frozen_teacher.parameters():parameter.requires_grad_(False)
        frozen=freeze_normalizer(policy)
        mode='warm_start_teacher_actor_fresh_critic_optimizer_counters'
        if args.resume:
            # Read only trusted local recipe checkpoints: torch checkpoint loading can execute Python.
            stored=torch.load(args.resume,map_location='cpu',weights_only=False)
            infos=require_resume_identity(stored.get('infos'),binding)
            runner.load(str(args.resume),map_location=env.device)
            freeze_normalizer(policy)
            assert_normalizer_unchanged(policy,frozen)
            runner.current_learning_iteration=infos['next_update']
            env.common_step_counter=infos['common_step_counter']
            env._sim_step_counter=env.common_step_counter*cfg.decimation
            mode='resume_actor_critic_optimizer_counters_new_episode_states'
        original_save=runner.save
        def save(path,infos=None):
            record={'recipe':binding,'common_step_counter':env.common_step_counter,
                    'next_update':env.common_step_counter//agent.num_steps_per_env}
            original_save(path,infos=record)
        runner.save=save
        obs_tensor=torch.tensor(anchor_obs,device=env.device)
        actions_tensor=torch.tensor(anchor_actions,device=env.device)
        anchor_losses=[];online_losses=[]
        def anchor_hook(optimizer,_args,_kwargs):
            with torch.enable_grad():
                index=torch.randint(len(obs_tensor),(256,),device=env.device)
                loss=(policy({'actor':obs_tensor[index]})-actions_tensor[index]).square().mean()
                storage_obs=runner.alg.storage.observations['actor'].reshape(-1,61)
                current_index=torch.randint(len(storage_obs),(256,),device=env.device)
                current={'actor':storage_obs[current_index].detach()}
                with torch.no_grad():teacher_mean=frozen_teacher(current)
                online_loss=(policy(current)-teacher_mean).square().mean()
                (args.anchor_coef*loss+args.online_anchor_coef*online_loss).backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(),runner.alg.max_grad_norm)
                anchor_losses.append(float(loss.detach()))
                online_losses.append(float(online_loss.detach()))
                # This must be LAST, after PPO and both BC gradients, before Adam.
                actor_warmup(policy,env.common_step_counter,args.critic_warmup_updates,agent.num_steps_per_env)
        handle=runner.alg.optimizer.register_step_pre_hook(anchor_hook)
        start_iteration=runner.current_learning_iteration
        start_steps=env.common_step_counter
        initial={**binding,**imported,'initialization':mode,'resume_checkpoint_sha256':sha(args.resume) if args.resume else None,
                 'optimizer_inherited':bool(args.resume),'critic_inherited':bool(args.resume),'counter_inherited':bool(args.resume),
                 'iteration_before':start_iteration,'common_step_counter_before':start_steps,
                 'geometry_check':geometry_check,'task_id':task.TASK,'runtime_identity':runtime_record,'posture_goals':goals.record(),'endpoint_spawns':env.endpoint_spawns,'human_review':'pending','deployment_approved':False}
        initial['profile_record']=profile_record
        (args.out/'initialization.json').write_text(json.dumps(initial,indent=2))
        (args.out/'source').mkdir()
        for source in sources:(args.out/'source'/source.name).write_bytes(source.read_bytes())
        def export(directory,checkpoint):
            runner.export_policy_to_onnx(str(directory),'policy.onnx')
            # Keep public identity relative; device paths never enter the ONNX run metadata.
            attach_metadata_to_onnx(str(directory/'policy.onnx'),get_base_metadata(env,run_path=checkpoint.name))
        reviews=[];done=0;start=time.monotonic();stop=None
        while done<args.iterations:
            block=min(args.review_every or args.iterations,args.iterations-done)
            runner.learn(num_learning_iterations=block,init_at_random_ep_len=False)
            done+=block
            runner.current_learning_iteration=start_iteration+done
            assert_normalizer_unchanged(policy,frozen)
            if args.review_every:
                boundary=args.out/f'update_{runner.current_learning_iteration:04d}'
                boundary.mkdir();checkpoint=boundary/'model.pt';runner.save(str(checkpoint));export(boundary,checkpoint)
                command=[sys.executable,str(REPO/'scripts/sitstand_check.py'),'--assets',str(args.assets.resolve()),
                         '--onnx',str((boundary/'policy.onnx').resolve()),'--checkpoint',str(checkpoint.resolve()),
                         '--calibration',str(args.calibration.resolve()),'--out-dir',str((boundary/'cpu_review').resolve())]
                with (boundary/'cpu_review.log').open('w') as log:
                    process=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'MUJOCO_GL':'egl'})
                report_path=boundary/'cpu_review/result.json'
                report=json.loads(report_path.read_text()) if report_path.exists() else {}
                review={'update':runner.current_learning_iteration,'returncode':process.returncode,
                        'automatic_verdict':report.get('automatic_verdict','UNVERIFIED'),
                        'onnx_sha256':sha(boundary/'policy.onnx'),'review_directory':boundary.name+'/cpu_review'}
                reviews.append(review);(args.out/'reviews.json').write_text(json.dumps(reviews,indent=2))
                if review['automatic_verdict']!='PASS':stop='boundary_review_not_pass';break
        final=args.out/'model_final.pt';runner.save(str(final));export(args.out,final)
        with torch.inference_mode():
            test_obs,_=env.reset();test_obs=test_obs['actor']
            expected=policy({'actor':test_obs}).cpu().numpy()
        import onnxruntime as ort
        session=ort.InferenceSession(str(args.out/'policy.onnx'),providers=['CPUExecutionProvider'])
        error=0.
        for row,prediction in zip(test_obs.cpu().numpy(),expected):
            error=max(error,float(np.abs(session.run(None,{session.get_inputs()[0].name:row[None]})[0][0]-prediction).max()))
        assert np.isfinite(expected).all() and error<=1e-3,error
        result={**initial,'updates_requested':args.iterations,'updates_completed':done,'stopped_reason':stop,
                'iteration_after':runner.current_learning_iteration,'common_step_counter_after':env.common_step_counter,
                'new_env_transitions':(env.common_step_counter-start_steps)*args.num_envs,'wall_seconds':time.monotonic()-start,
                'checkpoint_sha256':sha(final),'onnx_sha256':sha(args.out/'policy.onnx'),'actor_onnx_max_abs_error':error,'normalizer_count_after':policy.obs_normalizer.count.item(),
                'anchor_mse_mean':float(np.mean(anchor_losses)) if anchor_losses else None,'reviews':reviews,
                'online_anchor_mse_mean':float(np.mean(online_losses)) if online_losses else None,
                'automatic_smoke':'pass','action_validation':'pending_independent_checks_and_human_video_review'}
        (args.out/'run.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result,indent=2),flush=True)
        handle.remove()
    finally:env.close()

if __name__=='__main__':main()
