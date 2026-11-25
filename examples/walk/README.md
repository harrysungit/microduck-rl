# Walk and get up

`Mjlab-VelStand-Flat-MicroDuck-HD1910`: walking that follows a velocity
command, and gets up when it falls, in one policy.

**Status:** simulation only. Passed the 0.1.0 gate in the training world and
MuJoCo C sim2sim, with one warning. Has not passed a robot test or a review of
full-motion videos; not a candidate for a robot. Policy not published (sha256
`699e40e1…`).

## Recipe

Stages `W0` and `Q2` of [`queues/hd1910_skills.json`](../../queues/hd1910_skills.json):

| Stage | Task | Start | Iterations | Constraint budget | Wall time |
|---|---|---|---|---|---|
| W0 | `Mjlab-Velocity-Flat-MicroDuck-HD1910` | scratch | 10,000 | njmax 160 | 7.0 h, final mean reward 136.8 |
| Q2 | `Mjlab-VelStand-Flat-MicroDuck-HD1910` | warm start from W0 (`MICRODUCK_WARM_START=1`: curricula and iteration count restart) | 3,050 of 4,000, stopped at convergence | njmax 400, nconmax 200 | 6.1 h |

Q2's behaviour cloning takes its teachers from local checkpoints:
`HD1910_STAND_EXPERT` = the StandUp policy (stage Q1), `HD1910_WALK_ANCHOR` = W0.

## Acceptance

`velocity` and `getup` contracts, exported ONNX, actor mean.

| World | Metric | Value | Criterion |
|---|---|---|---|
| clean | forward tracking (achieved / commanded, 0.3 m/s) | 0.83 | ≥ 0.5 |
| clean | turn tracking (0.8 rad/s) | 0.85 | ≥ 0.4 |
| clean | gets up from lying within the time allowed | 98% | ≥ 90% |
| trained | worlds a termination ends after the topple pushes | **12.9%** | warning above 10%, failure above 25% |

Verdict: **ACCEPTED, with a warning** on recovery in the trained world.

## Sim2sim

All levels pass. No falls at the walking levels; gets up 48 of 48 times; with
0.3 m/s pushes 0 of 64 episodes fall, with 0.6 m/s pushes 37 of 64 fall.

## Known limits

- Recovery after a hard push is the weak point (the warning above).
- A rough-ground fine-tune with backlash was tried and made recovery worse
  (more worlds stay down); it is not used.
