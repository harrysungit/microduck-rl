# Kick

Right foot: `Mjlab-BallKick-Flat-MicroDuck-HD1910`. Left foot:
`Mjlab-BallKickLeft-Flat-MicroDuck-HD1910`, then fine-tuned with backlash on
`Mjlab-BallKickLeft-Flat-Backlash-MicroDuck-HD1910`.

**Status:** simulation only. Passed the 0.1.0 gate under a draft contract in
the training world and MuJoCo C sim2sim. Has not passed a robot test or a
review of full-motion videos; not a candidate for a robot. Policies not
published (sha256 right `e7ca843c…`, left flat `62f972b8…`, left with backlash
`54d2482b…`).

## Recipe

Stages `Q6`, `Q7a` and `Q7b` of [`queues/hd1910_skills.json`](../../queues/hd1910_skills.json):

| Stage | Task | Start | Iterations | Wall time |
|---|---|---|---|---|
| Q6 | BallKick (right) | scratch | 2,500 | 2.3 h |
| Q7a | BallKickLeft | scratch | 2,500 | 2.4 h |
| Q7b | BallKickLeft, backlash | resume Q7a | 500 | 1.0 h |

njmax 208 (224 with backlash), nconmax 50. The left foot uses Q7b.

## Acceptance

`kick` contract (draft), exported ONNX, actor mean: **100%** of worlds kick the
ball away faster than 0.5 m/s in the clean world, median ball speed
1.35–1.37 m/s (the training target is 1 m/s). Verdict: **ACCEPTED** for all
three.

## Sim2sim

All levels kick. Ball placed within the training range (±1.5 cm): 64 of 64;
within ±3 cm: about 80%.

## Known limits

- The ball leaves about 17° off straight ahead, in the training world and in
  sim2sim. The draft contract checks speed, not direction.
- The motion is large (action RMS about 4.6 rad, against 0.18 for the walk)
  and relies on joint limits and servo saturation. On a robot, check that the
  servo goals are not clipped differently.
