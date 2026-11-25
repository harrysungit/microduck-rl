# Stand up

`Mjlab-StandUp-Flat-MicroDuck-HD1910`: from lying on the back, front or side
to standing.

**Status:** simulation only. Passed the 0.1.0 gate in the training world and
MuJoCo C sim2sim. Has not passed a robot test or a review of full-motion
videos; not a candidate for a robot. Policy not published (sha256 `720da3f3…`).

## Recipe

Stage `Q1` of [`queues/hd1910_skills.json`](../../queues/hd1910_skills.json):
from scratch, converged at 5,000 of 6,000 iterations (mean reward of the last
500 iterations within 1% of the 500 before), njmax 256, nconmax 35, 4.2 h.

## Acceptance

`getup` contract, exported ONNX, actor mean: **100%** of worlds get up, in the
clean and in the trained world, median time 0.28 s. Verdict: **ACCEPTED**.

Training-world evaluation of the final checkpoint: upright 95.2% of the time,
0.30 tip-overs per world-minute.

## Sim2sim

46 of 48 starts get up: on the side 14 of 16 (median 1.1 s), on the front 16 of
16, on the back 16 of 16.

## Known limits

- Two of the sixteen side starts do not get up in sim2sim.
