# Sit and stand

`Mjlab-SitStand-Flat-MicroDuck-HD1910`: sits down and stands up on command.

**Status (2026-11-18): WITHDRAWN / FAILED.** A hardware trial and a substep
simulation audit exposed head-shell ground impacts. The old endpoint-only
checks below missed them. Do not deploy ONNX SHA256
`6f6c06e830d32023556f66d38c495e9b54ff6625448246896221875269e1dc20` (`model_3498.pt`).
At L3 seed 0, top_head_shell contact occurred at 3.822–3.902 s after a sit
command at 3 s (peak single-contact simulated normal force 65.943 N).
The old posture checker still reported 5/5. These are simulation forces,
not hardware force measurements. The policy remains unpublished.

The following is historical evidence, not the current training recommendation.
See [the new plan](../../docs/SITSTAND_PLAN.zh.md).

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

Historical verdict: **ACCEPTED — withdrawn on 2026-11-18**.

## Sim2sim

Noise-free levels 5 of 5; randomized 310 of 320; with 0.3 m/s pushes 320 of
320; starting seated 64 of 64. With 0.6 m/s pushes, 18 of 64 tip over.

## Known limits

- The clean sit-down margin is thin (84% against 80%): about one sit command
  in six does not start without noise.
- While sitting, the ankle and hip parts touch in the all-collision model for
  about 37% of the steps. Whether they touch on the real robot is unknown.
