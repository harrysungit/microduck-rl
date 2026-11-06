# Changelog

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
