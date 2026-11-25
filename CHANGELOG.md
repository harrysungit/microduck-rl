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
