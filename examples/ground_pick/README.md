# Pick from the ground

`Mjlab-GroundPick-Flat-MicroDuck-HD1910`, then fine-tuned with backlash on
`Mjlab-GroundPick-Flat-Backlash-MicroDuck-HD1910`: bends down to bring the
mouth to the floor and stands again, in 4 s cycles.

**Status:** accepted in the training world under a draft contract. **No
sim2sim yet.** Not run on a robot. Policies not published yet (sha256 flat
`4a0942c9…`, with backlash `3047efc9…`).

## Recipe

Stages `Q5a` and `Q5b` of [`queues/hd1910_skills.json`](../../queues/hd1910_skills.json):

| Stage | Task | Start | Iterations | Wall time |
|---|---|---|---|---|
| Q5a | GroundPick | scratch | 3,000 of 4,000, stopped at convergence | 2.3 h |
| Q5b | GroundPick, backlash | resume Q5a | 1,000 | 1.8 h |

njmax 208 (240 with backlash), nconmax 35.

## Acceptance

`pick` contract (draft), exported ONNX, actor mean.

| Policy | Clean | Trained | Lowest mouth tip, median |
|---|---|---|---|
| Q5a (flat) | 100% | 99% | 2.9 cm |
| Q5b (backlash) | 100% | 100% | 4.6 cm |

A pick counts when the mouth tip gets within 8 cm of the floor in the first
descent and the robot stands again for the last second of both cycles.
Verdict: **ACCEPTED** for both.

## Known limits

- No independent sim2sim; this is the skill checked least.
- The backlash fine-tune reaches about 1.7 cm less deep than the flat one.
- The 8 cm threshold is a draft.
