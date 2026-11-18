# microduck-rl

[English](README.md) | [中文](README.zh.md)

Training for the [Microduck](https://github.com/pollen-robotics/microduck) on
more servos and GPUs, built on Pollen Robotics'
[microduck_rl](https://github.com/pollen-robotics/microduck_rl).

This is not Pollen's repository. It is a package that depends on theirs,
unchanged, and registers extra [mjlab](https://github.com/mujocolab/mjlab)
tasks next to the official ones through mjlab's `mjlab.tasks` entry point.
`train`, `play` and export work as they do upstream.

What it adds:

- **The Feetech HD-1910 servo.** An actuator model (BAM's Feetech control law
  with LuwuDynamics' HD-1910 M6 fit, firmware P = 5, supply sag), a mass model
  (21 g servos, a 0.82 kg build) and HD-1910 versions of the official tasks.
- **ROCm.** mjlab and MuJoCo Warp on AMD GPUs through
  [AMD's Warp HIP port](https://github.com/AMD-Ecosystem/warp/tree/amd-integration-halo),
  with the fixes that make it fast: 7.3x the throughput of a plain port on a
  Radeon 8060S (gfx1151). See [benchmark/](benchmark/README.md) and
  [docs/ROCM.md](docs/ROCM.md).
- **Checks before a policy counts.** An acceptance gate that declares what
  "works" means for a skill before training and gives a verdict after it, and
  an independent sim2sim in plain MuJoCo, away from the training world.

2026-11-18: SitStand lowstd hit its head in simulation and a hardware trial.
Its old acceptance is withdrawn; it is not a deployment candidate. Trained
policies are not published. Read [training rules](AGENTS.md),
[upstream version checks](docs/REPOSITORIES.zh.md) and the [new sitstand plan](docs/SITSTAND_PLAN.zh.md).

## Skills

Historical skill checks are retained below; SitStand acceptance is withdrawn.
Automatic metrics do not replace human review of full-motion videos.

| Skill | Task | Acceptance (training world) | Sim2sim (MuJoCo C) |
|---|---|---|---|
| [Walk and get up](examples/walk/README.md) | `Mjlab-VelStand-Flat-MicroDuck-HD1910` | ACCEPTED, 1 warning | pass |
| [Stand up](examples/standup/README.md) | `Mjlab-StandUp-Flat-MicroDuck-HD1910` | ACCEPTED | pass, 46/48 |
| [Sit and stand](examples/sitstand/README.md) | `Mjlab-SitStand-Flat-MicroDuck-HD1910` | withdrawn: head impact | legacy posture check missed impact |
| [Kick](examples/kick/README.md) | `Mjlab-BallKick[Left]-Flat[-Backlash]-MicroDuck-HD1910` | ACCEPTED (draft contract) | pass |
| [Pick from the ground](examples/ground_pick/README.md) | `Mjlab-GroundPick-Flat[-Backlash]-MicroDuck-HD1910` | ACCEPTED (draft contract) | not run yet |

The Roulade trains but does not pass its contract; see the
[roadmap](docs/ROADMAP.md).

## Tasks

15 HD-1910 tasks, each the official task on the HD-1910 robot:

- `Mjlab-Velocity-{Flat,Rough}[-Backlash]-MicroDuck-HD1910`
- `Mjlab-VelStand-Flat[-Backlash]-MicroDuck-HD1910`, `Mjlab-VelStand-Rough-Backlash-MicroDuck-HD1910`
- `Mjlab-StandUp-Flat-MicroDuck-HD1910`, `Mjlab-SitStand-Flat-MicroDuck-HD1910`
- `Mjlab-GroundPick-Flat[-Backlash]-MicroDuck-HD1910`
- `Mjlab-BallKick-Flat-MicroDuck-HD1910`, `Mjlab-BallKickLeft-Flat[-Backlash]-MicroDuck-HD1910`
- `Mjlab-Roulade-Flat-MicroDuck-HD1910`

Backlash variants use the dense Jacobian: with 34 DOFs, MuJoCo Warp would pick
its sparse path, which is slower here. VelStand's behaviour cloning takes its
teachers from local HD-1910 checkpoints (`HD1910_STAND_EXPERT`,
`HD1910_WALK_ANCHOR`).

## Install (ROCm, gfx1151)

Needs Linux, the host ROCm 7.2 at `/opt/rocm` (Warp builds against it) and
Python 3.12.

```bash
git clone https://github.com/harrysungit/microduck-rl.git
cd microduck-rl
bash env/setup.sh          # .venv/, .deps/ and .cache/, all inside the repository
source env/activate.sh
python env/check_env.py    # devices, a Warp kernel, the ROCm patch, the tasks
```

`env/pins.toml` pins every source: Pollen's microduck_rl, AMD's Warp branch,
the torch nightly for gfx1151 and the Python packages of microduck_rl's
`uv.lock`. `env/activate.sh` keeps all caches under `.cache/` and turns on HIP
graph capture and PyTorch TunableOp.

Other GPUs: the ROCm patches are no-ops on CUDA and CPU devices. Installing
this package into a working microduck_rl environment (`pip install --no-deps
-e .`) should be enough there, but it has not been tested.

## Train

```bash
source env/activate.sh
train Mjlab-Velocity-Flat-MicroDuck-HD1910 --env.scene.num-envs 4096 --env.sim.njmax 160 \
    --agent.max-iterations 10000 --agent.logger tensorboard
```

`--env.sim.njmax` matters: mjlab's default of 1500 constraint rows per world
is about 30 times what the walk uses, and a lower budget is much faster. The
budget is guarded: a world that would exceed it stops training with
`ConstraintBudgetExceeded` instead of dropping rows silently.

All five skills, from scratch, unattended (about 31 h on gfx1151):

```bash
nohup setsid python scripts/queue_runner.py --queue queues/hd1910_skills.json > /dev/null 2>&1 < /dev/null &
```

The queue runs each stage with a time cap and a stall watchdog, stops at
convergence, chains warm starts and resumes, then evaluates, records a video
and runs the acceptance gate.

## Check a policy

```bash
# Acceptance: a skill's contract on the exported ONNX, in a clean and in the trained world
python scripts/acceptance.py --task Mjlab-StandUp-Flat-MicroDuck-HD1910 --onnx policy.onnx \
    --checkpoint model_5000.pt --spec getup --out-dir logs/results/standup_acceptance

# Sim2sim: the same ONNX in plain MuJoCo, CPU, with its own servo loop
python scripts/sim2sim_assets.py --out-dir logs/sim2sim/assets
python scripts/sim2sim.py --onnx policy.onnx --assets logs/sim2sim/assets --out-dir logs/results/walk_sim2sim
```

The verdict is ACCEPTED, FAILED or UNVERIFIED. UNVERIFIED means nothing failed
but something could not be checked; it is not a pass. See
[scripts/acceptance_core.py](scripts/acceptance_core.py) for the contracts.

To deploy, export with Pollen's `scripts/export.py`, which bakes the
observation normalizer into the ONNX. [docs/DEPLOY.md](docs/DEPLOY.md) has the
interface and what differs on an HD-1910 robot.

## Layout

| Path | Contents |
|---|---|
| `src/microduck_rl/tasks/` | Task registration and the runtime patches it applies |
| `src/microduck_rl/actuators/hd1910/` | HD-1910 actuator, mass model, robot configs, BAM parameters |
| `src/microduck_rl/patches/` | `rocm.py`, `perf.py`, `fixed_std.py`: runtime patches to mjlab, Warp's allocator use and rsl_rl |
| `env/` | Pinned sources, setup, activation, environment check, two MuJoCo Warp fixes |
| `scripts/` | Evaluation, acceptance, sim2sim, the training queue, throughput and profiling |
| `queues/` | Training queues for `scripts/queue_runner.py` |
| `examples/` | One directory per skill: recipe, numbers, limits |
| `benchmark/` | Throughput on gfx1151 |
| `docs/` | Deployment interface, roadmap, ROCm notes |
| `tests/` | CPU tests: `python -m pytest tests` |

## License

Apache-2.0, see [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md). This
repository contains no 3D model files: the robot's meshes come with Pollen's
microduck_rl, under the licence stated there.
