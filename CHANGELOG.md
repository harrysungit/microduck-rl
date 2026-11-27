# Changelog

## Unreleased

- Withdrawn: the sit-stand acceptance of 0.1.0. The gate scored end states
  only; the policy's head hit the ground on the way down in simulation, and
  it sat down too hard on a robot.
- The other skills of 0.1.0 passed checks in simulation only. None of the
  policies trained so far has passed a robot test or a review of full-motion
  videos; none is a candidate for a robot.
- `scripts/check_upstream.py`: what changed in Pollen's microduck_rl and its
  released policies since the pin; it only reports.
- Every training run writes `CODE_IDENTITY.json`: commits, local changes,
  untracked sources, the launcher and package versions.
- `env/setup.sh` honours `MICRODUCK_RL_DEPS`, so one environment outside the
  checkout can serve several worktrees.
- `Mjlab-SitStand-Flat-MicroDuck-HD1910-Robotd`: sit-stand trained with the
  targets robotd sends (`7.4 V / battery` action scale, low-pass 0.5 head /
  0.7 legs) over a 7.0–8.4 V battery. sim2sim's robotd levels (`L0-robotd`,
  `L3-robotd*`) run the same arithmetic, and every sit-stand report gives each
  transition's net turn.
- `scripts/sitstand_gate.py`: the sit-stand gate, on the training's physics
  with robotd's chain. Each posture command is an attempt; a fall or a
  broken scene stays in the denominator. The head must not hit the ground
  (16 N, about twice the robot's weight) nor rest on it for 50 ms once the
  posture is held, and in the last 2 s be within 0.2 rad of its command;
  lighter head support during a transition is recorded, not judged.
  The first policy target fills the servo latency line, as the training's
  delay buffer does after a reset. Every 5 ms step records non-foot
  ground contacts with force and impulse. Falls, head failures and broken
  scenes fail it in the nominal condition; success thresholds are 98%
  nominal and 80% disturbed, set on kissqy's sit-stand policy (512/512 and
  85.5%). Pollen's seated fall-back metrics are reported, not judged.
- sim2sim's `seated_pose` takes the largest tilt; its draws do not change.
- Tests clear `MICRODUCK_RL_*` from the environment: under `env/activate.sh`
  the upstream-check test fetched its fixture into the shared review cache.

## 0.1.0 (2026-11-06)

First release.

- HD-1910 servo: actuator model, mass model, 15 HD-1910 versions of the
  official tasks.
- ROCm: runtime patches that make mjlab and MuJoCo Warp fast on gfx1151
  (7.3x a plain port at 4096 worlds), two MuJoCo Warp fixes, a pinned
  environment with one setup script.
- Acceptance gate, sim2sim in MuJoCo C, evaluation tools, an unattended
  training queue, throughput and profiling scripts.
- Five skills accepted in simulation: walk and get up, stand up, sit and
  stand, kick, pick from the ground.

### Known limitations

- No policy has run on a robot. The HD-1910 servo model is borrowed, not
  identified on our servos, and the robot's mass is a placeholder.
- Trained policies are not published yet.
- The Roulade does not pass its contract; the ground pick has no sim2sim yet;
  the kick and pick contracts are drafts.
- Tested on gfx1151 only. `env/setup.sh` has not been run end to end on a
  clean machine.
