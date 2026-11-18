"""Run a skill's acceptance contracts on the policy that ships, in the training world.

The policy is the exported ONNX, run on the CPU with onnxruntime as on the robot, acting with its actor mean. Each
contract (acceptance_core.py: posture, velocity, getup) scripts its commands and spawn states instead of drawing them,
and runs twice: "clean", the training world without any disturbance the robot does not have (pushes and other interval
events, observation noise and delay, head and body pose commands, IMU misalignment, encoder bias), and "trained", the
world as trained. Both must pass. With the checkpoint the run also checks that the ONNX acts like the checkpoint's actor,
and runs the first scenario again with the training's exploration noise added to the actions: a behaviour that
only appears with that noise is one the robot will never show (a SitStand policy did exactly that; see acceptance_core.py).

    python scripts/acceptance.py --task <task> --onnx <policy.onnx> [--checkpoint <model.pt>] \
        --spec '{"contract": "posture"}' --out-dir <dir>

Writes acceptance.json and acceptance.md to the out dir; the verdict is in them, the exit code is 0 whenever one was
written.
"""

import argparse
import hashlib
import json
import re
import shutil
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch

import mjlab  # noqa: F401  # loads the mjlab.tasks entry points, and with them the ROCm patch
from eval_policy import onnx_policy
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

import acceptance_core as ac  # noqa: E402
import sim2sim  # noqa: E402  # the schedules, thresholds and the pass criterion of the MuJoCo C check


class NotApplicable(Exception):
    """The task lacks something the contract scripts (a spawn event, a command term)."""


@dataclass
class Scenario:
    name: str
    seconds: float
    patch: Callable  # cfg -> None: the scenario's spawn state and command settings
    drive: Callable  # (base, t) -> None: writes the scripted command before the control step at time t
    probes: dict = field(default_factory=dict)  # name -> (base -> tensor of shape (worlds,)), recorded every step


# -- the training world without what the robot does not have --------------------------------------------------------
def make_clean(cfg) -> None:
    cfg.observations["actor"].enable_corruption = False  # observation noise and delay
    removed = [name for name, event in cfg.events.items() if event.mode == "interval"]  # pushes, kicks, topples
    for name in removed:
        cfg.events.pop(name)
    for name, term in list(cfg.curriculum.items()):
        params = getattr(term, "params", None) or {}
        if params.get("event_name") in removed or params.get("command_name") in ("head_pose", "body_pose"):
            cfg.curriculum.pop(name)
    for name in ("head_pose", "body_pose"):
        if name in cfg.commands:
            cfg.commands[name].ranges = tuple((0.0, 0.0) for _ in cfg.commands[name].ranges)
    for term in cfg.observations["actor"].terms.values():  # IMU mounting misalignment
        if term.params and "max_angle_deg" in term.params:
            term.params = {**term.params, "max_angle_deg": 0.0}
    if "encoder_bias" in cfg.events and "bias_range" in cfg.events["encoder_bias"].params:
        cfg.events["encoder_bias"].params["bias_range"] = (0.0, 0.0)


def scripted_commands(cfg) -> None:
    twist = cfg.commands["twist"]
    twist.resampling_time_range = (1e9, 1e9)  # the command is written by the scenario
    twist.heading_command, twist.rel_standing_envs, twist.rel_heading_envs = False, 0.0, 0.0
    twist.ranges.heading = None  # the config refuses a heading range without heading control
    if hasattr(twist, "rel_turn_in_place_envs"):
        twist.rel_turn_in_place_envs = 0.0
    cfg.curriculum.pop("standing_envs", None)


def spawn_upright(cfg) -> None:
    events = cfg.events
    if "random_prone_init" in events:
        events["random_prone_init"].params.update(prone_prob=0.0, crouch_prob=0.0)
        cfg.curriculum.pop("prone_init_prob", None)
    if "set_ground_state" in events:
        events["set_ground_state"].params.update(face_down_prob=0.0, face_up_prob=0.0, sitting_prob=0.0, standing_prob=1.0)
        cfg.curriculum.pop("ground_state_mix", None)


def write_command(base, cmd) -> None:
    term = base.command_manager.get_term("twist")
    term.vel_command_b[:] = torch.tensor(cmd, device=term.vel_command_b.device, dtype=term.vel_command_b.dtype)
    if hasattr(term, "is_standing_env"):
        term.is_standing_env[:] = False


# -- contracts ---------------------------------------------------------------------------------------------------------
class Posture:
    """Commanded sit and stand (SitStand): the trunk must go down on a sit command and up on a stand command."""
    name, envs = "posture", 1024

    def scenarios(self, task):
        def patch(start):
            def f(cfg):
                if "set_ground_state" not in cfg.events:
                    raise NotApplicable("the task has no set_ground_state spawn")
                scripted_commands(cfg)
                cfg.events["set_ground_state"].params.update(
                    face_down_prob=0.0, face_up_prob=0.0, sitting_prob=float(start == "seated"),
                    standing_prob=float(start != "seated"))
            return f

        def drive(schedule):
            def f(base, t):
                term = base.command_manager.get_term("twist")
                if not hasattr(term, "alpha"):
                    raise NotApplicable("the twist command is not a posture flag")
                term.vel_command_b[:, 0] = float(sim2sim.command_at(t, schedule)[0])
            return f

        return [Scenario(name, sum(d for d, _, _ in schedule), patch(name), drive(schedule))
                for name, schedule in (("standing", sim2sim.SIT_FROM_STAND), ("seated", sim2sim.SIT_FROM_SEAT))]

    def analyze(self, runs):
        schedules = {"standing": sim2sim.SIT_FROM_STAND, "seated": sim2sim.SIT_FROM_SEAT}
        return ac.posture_metrics([ac.posture_counts(run, schedules[name]) for name, run in runs.items()])

    def primary(self, metrics):
        return metrics["sit_down"]


class Velocity:
    """Scripted velocity commands (Velocity, VelStand): the robot must stay up and move as commanded."""
    name, envs = "velocity", 256

    def scenarios(self, task):
        def patch(cfg):
            scripted_commands(cfg)
            spawn_upright(cfg)

        def drive(base, t):
            write_command(base, sim2sim.command_at(t, sim2sim.SCHEDULE))

        return [Scenario("schedule", sim2sim.DURATION, patch, drive)]

    def analyze(self, runs):
        return ac.velocity_metrics(runs["schedule"], sim2sim.SCHEDULE)

    def primary(self, metrics):
        return None if metrics["forward"] is None else min(max(metrics["forward"], 0.0), 1.0)


class Getup:
    """Spawn lying on the floor, zero command (StandUp, VelStand): the robot must get its trunk up and hold it."""
    name, envs = "getup", 512

    def scenarios(self, task):
        def patch(cfg):
            scripted_commands(cfg)
            events = cfg.events
            if "set_ground_state" in events:
                events["set_ground_state"].params.update(face_down_prob=0.5, face_up_prob=0.5, sitting_prob=0.0,
                                                         standing_prob=0.0)
                cfg.curriculum.pop("ground_state_mix", None)
            elif "random_prone_init" in events:
                events["random_prone_init"].params.update(prone_prob=1.0, face_down_prob=0.5, crouch_prob=0.0)
                cfg.curriculum.pop("prone_init_prob", None)
            else:
                raise NotApplicable("the task has neither set_ground_state nor random_prone_init")

        return [Scenario("lying", sim2sim.RECOVERY_S, patch, lambda base, t: write_command(base, (0.0, 0.0, 0.0)))]

    def analyze(self, runs):
        return ac.getup_metrics(runs["lying"])

    def primary(self, metrics):
        return metrics["up"]


class Kick:
    """The ball is placed in front of the kicking foot, which the policy cannot see: it must be kicked forward."""
    name, envs = "kick", 512

    def scenarios(self, task):
        def patch(cfg):
            if "reset_ball" not in cfg.events or "ball" not in cfg.scene.entities:
                raise NotApplicable("the task has no ball")
            scripted_commands(cfg)

        def ball_fwd(base):
            from mjlab_microduck.tasks import mdp as microduck_mdp
            velocity = base.scene["ball"].data.root_link_lin_vel_w[:, :2]
            return (velocity * microduck_mdp._ball_kick_dir(base)).sum(dim=1)  # along the kick direction set at reset

        def ball_lat(base):
            from mjlab_microduck.tasks import mdp as microduck_mdp
            velocity, kick = base.scene["ball"].data.root_link_lin_vel_w[:, :2], microduck_mdp._ball_kick_dir(base)
            return velocity[:, 1] * kick[:, 0] - velocity[:, 0] * kick[:, 1]  # across it, + = to the left

        return [Scenario("kick", 5.0, patch, lambda base, t: write_command(base, (0.0, 0.0, 0.0)),
                         {"ball_fwd": ball_fwd, "ball_lat": ball_lat})]

    def analyze(self, runs):
        return ac.kick_metrics(runs["kick"])

    def primary(self, metrics):
        return metrics["kick"]


class Pick:
    """The mouth goes down toward the ground and the duck stands up again, on a 4 s cycle started from standing."""
    name, envs = "pick", 512

    def scenarios(self, task):
        def patch(cfg):
            twist = cfg.commands.get("twist")
            if not hasattr(twist, "randomize_phase"):
                raise NotApplicable("the twist command is not a pick cycle")
            if abs(twist.period - ac.PICK_CYCLE_S) > 1e-6:
                raise NotApplicable(f"the contract assumes a {ac.PICK_CYCLE_S:g} s cycle, the task has {twist.period:g} s")
            twist.randomize_phase = False  # every episode starts the cycle standing, as the runtime's button does
            cfg.curriculum.pop("standing_envs", None)

        site = {}

        def mouth_z(base):
            from mjlab.managers.scene_entity_config import SceneEntityCfg
            if not site:
                found = SceneEntityCfg("robot", site_names=["mouth_tip"])
                found.resolve(base.scene)
                site["id"] = found.site_ids[0]
            return base.scene["robot"].data.site_pos_w[:, site["id"], 2]

        return [Scenario("cycles", 2 * ac.PICK_CYCLE_S, patch, lambda base, t: None, {"mouth_z": mouth_z})]

    def analyze(self, runs):
        return ac.pick_metrics(runs["cycles"])

    def primary(self, metrics):
        return metrics["pick"]


class Roulade:
    """Starting standing, roll forward over the head and land on the feet."""
    name, envs = "roulade", 512

    def scenarios(self, task):
        def patch(cfg):
            if "set_roulade_state" not in cfg.events:
                raise NotApplicable("the task has no set_roulade_state spawn")
            scripted_commands(cfg)
            cfg.events["set_roulade_state"].params.update(standing_prob=1.0, midroll_prob=0.0)
            cfg.curriculum.pop("roulade_spawn_mix", None)

        return [Scenario("roll", 8.0, patch, lambda base, t: write_command(base, (0.0, 0.0, 0.0)))]

    def analyze(self, runs):
        return ac.roulade_metrics(runs["roll"])

    def primary(self, metrics):
        return metrics["roll"]


CONTRACTS = {c.name: c for c in (Posture(), Velocity(), Getup(), Kick(), Pick(), Roulade())}
assert set(CONTRACTS) == set(ac.DEFAULT_CRITERIA), "every declarable contract needs an implementation"


# -- running ---------------------------------------------------------------------------------------------------------
def build_env(task, scenario, variant, envs, args, device):
    cfg = load_env_cfg(task)
    cfg.scene.num_envs = envs
    cfg.seed = args.seed
    cfg.sim.njmax = args.njmax
    cfg.episode_length_s = scenario.seconds + 5.0  # the schedule must not end on a time-out
    scenario.patch(cfg)
    if variant == "clean":
        make_clean(cfg)
    try:
        return ManagerBasedRlEnv(cfg=cfg, device=device)
    except ValueError as e:
        m = re.search(r"njmax must be >= (\d+)", str(e))
        if not m:
            raise
        cfg.sim.njmax = int(m.group(1)) + 64
        return ManagerBasedRlEnv(cfg=cfg, device=device)


def rollout(base, env, policy, scenario, args, device, n, compare=None) -> dict:
    """Trunk height and tilt, body-frame velocities of every world at every control step under the scripted command."""
    robot, dt = base.scene["robot"], base.step_dt
    steps = round(scenario.seconds / dt)
    rec = {k: torch.zeros(steps, n, device=device) for k in ("z", "tilt", "vx", "vy", "wz", *scenario.probes)}
    reset = torch.zeros(n, dtype=torch.bool, device=device)
    base.common_step_counter = args.curriculum_step
    torch.manual_seed(args.seed)
    env.reset()
    diff = 0.0
    with torch.no_grad():
        for k in range(steps):
            scenario.drive(base, k * dt)
            obs = env.get_observations()
            action = policy(obs)
            if compare is not None and k < 20:
                diff = max(diff, (compare(obs) - action).abs().max().item())
            _, _, dones, _ = env.step(action)
            reset |= dones.bool()
            rec["z"][k] = robot.data.root_link_pos_w[:, 2] - base.scene.terrain.env_origins[:, 2]
            rec["tilt"][k] = torch.rad2deg(torch.acos((-robot.data.projected_gravity_b[:, 2]).clamp(-1.0, 1.0)))
            rec["vx"][k], rec["vy"][k] = robot.data.root_link_lin_vel_b[:, 0], robot.data.root_link_lin_vel_b[:, 1]
            rec["wz"][k] = robot.data.root_link_ang_vel_b[:, 2]
            for name, probe in scenario.probes.items():
                rec[name][k] = probe(base)
    out = {k: v.cpu().numpy() for k, v in rec.items()}
    out.update(reset=reset.cpu().numpy(), dt=dt, onnx_diff=diff if compare is not None else None)
    return out


def record_video(task, contract, onnx, out_dir, args, device) -> str:
    """The contract's first scenario, one world, clean: what the robot would do when asked, as a video."""
    from mjlab.utils.wrappers import VideoRecorder

    sc = contract.scenarios(task)[0]
    cfg = load_env_cfg(task)
    cfg.scene.num_envs, cfg.seed, cfg.sim.njmax = 1, args.seed, args.njmax
    cfg.episode_length_s = sc.seconds + 5.0
    cfg.viewer.width, cfg.viewer.height = 1280, 720
    cfg.viewer.distance, cfg.viewer.elevation, cfg.viewer.azimuth = 0.8, -15.0, 135.0  # the duck is 25 cm tall
    sc.patch(cfg)
    make_clean(cfg)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    folder = Path(__file__).resolve().parents[1] / "logs" / "acceptance_videos" / stamp
    base = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode="rgb_array")
    steps = round(sc.seconds / base.step_dt)
    env = VideoRecorder(base, video_folder=folder, step_trigger=lambda step: step == 0, video_length=steps,
                        disable_logger=True)
    env = RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(task).clip_actions)
    base.common_step_counter = args.curriculum_step
    env.reset()
    with torch.no_grad():
        for k in range(steps + 1):
            sc.drive(base, k * base.step_dt)
            env.step(onnx(env.get_observations()))
    env.close()
    mp4 = sorted(folder.glob("*.mp4"), key=lambda p: p.stat().st_mtime)[-1]
    name = f"rollout_{contract.name}_{sc.name}_{stamp}.mp4"
    shutil.copy(mp4, out_dir / name)
    return name


def noisy(policy, std):
    """The training's stochastic policy: the policy's mean plus Gaussian noise of the learned std."""
    return lambda obs: (lambda a: a + std * torch.randn_like(a))(policy(obs))


def run_contract(contract, task, item, args, device, onnx, pt, std) -> dict:
    envs = args.num_envs or item["params"].get("num_envs") or contract.envs
    scenarios = contract.scenarios(task)
    out = {"contract": contract.name, "criteria": item["criteria"], "metrics": {}, "onnx_diff": None}
    first_mean = first_sampled = None
    for variant in ac.VARIANTS:
        runs = {}
        for i, sc in enumerate(scenarios):
            first = variant == "clean" and i == 0
            base = build_env(task, sc, variant, envs, args, device)
            env = RslRlVecEnvWrapper(base, clip_actions=load_rl_cfg(task).clip_actions)
            try:
                compare = None
                if first and args.checkpoint:
                    runner = (load_runner_cls(task) or MjlabOnPolicyRunner)(env, asdict(load_rl_cfg(task)), device=device)
                    runner.load(str(args.checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
                    compare = runner.get_inference_policy(device=device)
                runs[sc.name] = rollout(base, env, onnx, sc, args, device, envs, compare)
                if first:
                    out["onnx_diff"] = runs[sc.name].pop("onnx_diff")
                    first_mean = contract.analyze({sc.name: runs[sc.name]})
                    if std is not None and not args.no_noise_check:
                        sampled = rollout(base, env, noisy(onnx, std), sc, args, device, envs)
                        first_sampled = contract.analyze({sc.name: sampled})
            finally:
                env.close()
            print(f"  {contract.name} {variant} {sc.name}: done", flush=True)
        out["metrics"][variant] = contract.analyze(runs)
    if first_sampled is not None:  # like with like: the first scenario, mean action against sampled action
        out["noise_check"] = ac.noise_gap(contract.primary(first_mean), contract.primary(first_sampled))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", required=True)
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, help="the rsl_rl .pt the ONNX was exported from (enables the generic checks)")
    ap.add_argument("--spec", required=True, help="the stage's acceptance entry as JSON, or a contract name")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--num-envs", type=int, help="override every contract's world count")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--njmax", type=int, default=512)
    ap.add_argument("--curriculum-step", type=int, default=10_000 * 24)
    ap.add_argument("--no-noise-check", action="store_true")
    ap.add_argument("--video", action="store_true", help="also record the first contract's first scenario")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    spec = args.spec if args.spec.lstrip().startswith(("{", "[", '"')) else json.dumps(args.spec)
    declared = ac.contracts_of(json.loads(spec))
    result = {"task": args.task, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
              "policy": {"onnx": str(args.onnx), "sha256": hashlib.sha256(args.onnx.read_bytes()).hexdigest(),
                         "checkpoint": str(args.checkpoint) if args.checkpoint else None},
              "declared": declared, "contracts": [], "checks": [], "errors": [], "notes": [], "learned_std": None}

    configure_torch_backends()
    device = "cuda:0"
    onnx = onnx_policy(args.onnx, device)
    std = None
    if args.checkpoint:
        sd = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["actor_state_dict"]
        if "distribution.std_param" in sd:
            s = sd["distribution.std_param"].float()
            result["learned_std"] = {"mean": float(s.mean()), "min": float(s.min()), "max": float(s.max())}
            std = s.to(device)
    onnx_diff = None
    for item in declared:
        try:
            r = run_contract(CONTRACTS[item["contract"]], args.task, item, args, device, onnx, args.checkpoint, std)
        except NotApplicable as e:
            result["errors"].append(f"{item['contract']}: not applicable here: {e}")
            continue
        except Exception:
            result["errors"].append(f"{item['contract']}: crashed: " + traceback.format_exc(limit=4).replace("\n", " | "))
            continue
        if onnx_diff is None:
            onnx_diff = r["onnx_diff"]
        checks = ac.evaluate(r["metrics"], item["criteria"])
        for c in checks:
            c["contract"] = item["contract"]
        result["checks"] += checks
        result["contracts"].append(r)
    if args.checkpoint:
        result["checks"].append(ac.onnx_check(onnx_diff))
    else:  # not an error: a model that comes as ONNX only (a third party's) has no checkpoint to compare with
        result["notes"].append("no checkpoint given: the ONNX-against-checkpoint check and the noise check were skipped")
    if args.video and declared:
        try:
            result["video"] = record_video(args.task, CONTRACTS[declared[0]["contract"]], onnx, args.out_dir, args, device)
        except Exception:
            result["notes"].append("video failed: " + traceback.format_exc(limit=3).replace("\n", " | "))
    result["checks"] += ac.known_failure_checks(result["policy"]["sha256"])
    result["human_review"] = "pending"
    result["deployment_approved"] = False
    result["verdict"], result["reasons"] = ac.decide(result["checks"], result["errors"])
    result["warnings"] = ac.warnings_of(result["checks"])
    result["elapsed_s"] = round(time.time() - t0)
    (args.out_dir / "acceptance.json").write_text(json.dumps(result, indent=1, default=float))
    (args.out_dir / "acceptance.md").write_text(ac.render(result))
    print(f"{args.task}: {result['verdict']}" + (f" — {'; '.join(result['reasons'])}" if result["reasons"] else "")
          + f" ({result['elapsed_s']} s)", flush=True)


if __name__ == "__main__":
    main()
