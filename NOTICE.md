# Notice

microduck-rl is licensed under Apache-2.0 (see [LICENSE](LICENSE)). It builds
on the following work.

| Source | Used for | License |
|---|---|---|
| [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl) @ `8d0db749` | Installed as a dependency, unchanged: task configs, robot models, actuator code, ONNX export. Its README states that its 3D model files are under a Creative Commons BY-SA-NC license; this repository contains none of them | Apache-2.0 (code) |
| [LuwuDynamics/xgoduck_rl](https://github.com/LuwuDynamics/xgoduck_rl) @ `326d77a1` | `src/microduck_rl/actuators/hd1910/params/1910_m6.json`, copied byte for byte (see the README next to it) | Apache-2.0 |
| [fanhao375/microduck-replica](https://github.com/fanhao375/microduck-replica) @ `25791f0` | The approach of the HD-1910 actuator's goal-state reset (`software/training/.../feetech_bam.py`) | Apache-2.0 (the training code; the repository mixes licenses) |
| [qiaosiyi/microduck_rl_hd1910](https://github.com/qiaosiyi/microduck_rl_hd1910) | The approach of editing the robot spec at load time for the HD-1910 mass model | Apache-2.0 |
| [Rhoban/bam](https://github.com/Rhoban/bam) @ `62bd8ce1` | Actuator models (dependency) | Apache-2.0 |
| [AMD-Ecosystem/warp](https://github.com/AMD-Ecosystem/warp/tree/amd-integration-halo) @ `74014999` | Warp with a HIP backend (dependency, built from source) | Apache-2.0 |
| [google-deepmind/mujoco_warp](https://github.com/google-deepmind/mujoco_warp) 3.8.1 | Two local fixes in `env/patches/`, following what upstream did later | Apache-2.0 |
| [mujocolab/mjlab](https://github.com/mujocolab/mjlab) 1.3.0 | Training framework (dependency); `src/microduck_rl/patches/` adjusts it at runtime | Apache-2.0 |

Microduck is a robot by Pollen Robotics. This project is not affiliated with
or endorsed by Pollen Robotics.

`docs/upstream/AGENTS.develop.md` is an unmodified reference copy of Pollen
Robotics' AGENTS.md at `0daab160d3fb1c2cbaba16a4a19052f36709dd9e` (Apache-2.0).
The source URL and SHA256 are recorded in `docs/upstream/source.json`.
The root AGENTS.md adapts selected guidance for this extension and adds local
hardware and validation requirements; it is not an official Pollen document.
