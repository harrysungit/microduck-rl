# Roadmap

Each item says what "done" means. Simulation work is done for five skills;
the next step needs a robot.

## Hardware

- [ ] **Identify the HD-1910 on a bench.** Done: our own BAM fit from logged
  servo trajectories replaces the borrowed one, and the five skills pass their
  contracts again with it.
- [ ] **Weigh the robot.** Done: `TOTAL_MASS_KG` in
  `actuators/hd1910/mass.py` is our scale reading, and the skills are
  re-checked.
- [ ] **First stand and walk on the robot.** Done: the walk holds a zero
  command for a minute and follows 0.3 m/s forward on a flat floor, recorded
  on video, with the runtime's logs.

## Skills

- [ ] **Roulade.** It rolls, but lands upright in 14% of rolls; the contract
  asks for 80% in the clean world. Done: ACCEPTED.
- [ ] **Sim2sim for the ground pick.** Done: `scripts/sim2sim.py --task pick`
  exists and the two pick policies pass it.
- [ ] **Kick direction.** The kicks send the ball about 17° off straight
  ahead, in the training world and in sim2sim; the contract checks speed only.
  Done: a direction criterion in the kick contract, and policies that pass it.
- [ ] **Confirm the draft contracts.** The kick (ball speed at least 0.5 m/s)
  and pick (mouth tip within 8 cm of the floor, then standing) thresholds are
  drafts. Done: thresholds set from hardware runs.
- [ ] **SitStand margin.** Clean sit-down is 84% against a threshold of 80%,
  and while sitting the ankle and hip parts touch in the all-collision model
  for about 37% of the steps. Done: a margin of 10 points, and the contact
  checked on the real robot.
- [ ] **Walk recovery.** After the trained world's topple pushes, 12.9% of
  worlds stay down for 8 s (warning above 10%, failure above 25%). Done: under
  10%.

## Release

- [ ] **Publish trained policies.** Done: ONNX files with a manifest in the
  shape Pollen's runtime loads.
- [ ] **Fresh install.** `env/setup.sh` has not been run end to end on a
  clean machine. Done: a clean run, recorded.
- [ ] **CUDA.** Done: the HD-1910 tasks train on an NVIDIA GPU with this
  package installed into an upstream environment.

## Upstream

- [ ] **ROCm and throughput fixes.** Most of `patches/rocm.py` and
  `patches/perf.py` adjust mjlab's behaviour. Done: each fix proposed to the
  project it belongs to, and removed here once released there.
- [x] **MuJoCo Warp fixes.** Both patches in `env/patches/` follow changes
  already in MuJoCo Warp 3.10 or its main branch. They go away with the next
  MuJoCo Warp bump.
