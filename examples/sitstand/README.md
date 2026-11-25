# Sit and stand

`Mjlab-SitStand-Flat-MicroDuck-HD1910`: sits down and stands up on command.

**Status:** withdrawn. Passed the 0.1.0 gate in the training world and MuJoCo
C sim2sim, which scored end states only: on the way down its head shell hit
the ground in simulation, and on a robot (2026-11-18) it sat down too hard.
Not a candidate for a robot. Policy not published (sha256 `6f6c06e8…`). The
rest of this page is the record of 0.1.0.

## Recipe

Stages `Q4` and `Q4b` of [`queues/hd1910_skills.json`](../../queues/hd1910_skills.json):

| Stage | Start | Iterations | Notes | Wall time |
|---|---|---|---|---|
| Q4 | scratch | 3,000 | njmax 256, nconmax 200 | 3.2 h |
| Q4b | resume Q4 | 500 | `MICRODUCK_FIXED_STD=0.05` | 0.6 h |

Why two stages: after Q4 the policy sat down in training but almost never
from a standing start without exploration noise (0–2% of commanded sits).
`scripts/eval_posture.py` and `scripts/sitstand_noise_sweep.py` showed that
the noise, not the mean policy, started the descent. The learned std was
0.19; fine-tuning with the std fixed at 0.05 makes the mean policy do it.

## Acceptance

`posture` contract, exported ONNX, actor mean.

| World | Sit-down | Stand-up | Criterion |
|---|---|---|---|
| clean | 84.2% | 99.7% | sit ≥ 80%, stand ≥ 90% |
| trained | 99.1% | 98.1% | sit ≥ 60%, stand ≥ 60% |

Verdict: **ACCEPTED**.

## Sim2sim

Noise-free levels 5 of 5; randomized 310 of 320; with 0.3 m/s pushes 320 of
320; starting seated 64 of 64. With 0.6 m/s pushes, 18 of 64 tip over.

## Known limits

- The clean sit-down margin is thin (84% against 80%): about one sit command
  in six does not start without noise.
- While sitting, the ankle and hip parts touch in the all-collision model for
  about 37% of the steps. Whether they touch on the real robot is unknown.
