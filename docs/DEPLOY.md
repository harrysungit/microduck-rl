# Deploying a policy

The HD-1910 tasks keep the official policy interface, so a policy trained
here runs wherever an official Microduck policy runs, given a runtime that
drives Feetech servos.

## Interface

- **Export** with Pollen's `scripts/export.py` (in the microduck_rl checkout):

  ```bash
  python <microduck_rl>/scripts/export.py <task> --checkpoint-file model_<n>.pt --onnx-file policy.onnx
  ```

  It bakes the observation normalizer into the graph. Never convert a
  checkpoint by hand, or the policy sees unnormalized observations.
- **ONNX**: input `obs`, float32 `[1, 61]`; output `actions`, `[1, 14]`.
- **Rate**: 50 Hz, one observation in, one action out.
- **Observation**: 48 proprioception values, then the commands
  `[twist (3), head_pose (4), body_pose (6)]`. Tasks that do not use a command
  slot get zeros there. This shared layout is what lets the runtime swap
  policies.
- **Joints** (14 servos, for observations and actions): 0–4 left leg
  (hip_yaw, hip_roll, hip_pitch, knee, ankle), 5–8 neck and head (neck_pitch,
  head_pitch, head_yaw, head_roll), 9–13 right leg.

The action-to-goal mapping, default pose and limits are those of Pollen's task
configs, unchanged here.

## What differs on an HD-1910 robot

The policies were trained against these assumptions. A robot that differs
from them is outside what was simulated.

| Assumption | Value in training | Status |
|---|---|---|
| Servo control | Feetech firmware position loop, P = 5 | Not checked against our servos' settings |
| Servo dynamics | BAM M6 fit for the HD-1910 by LuwuDynamics; about 0.91 N·m stall at 7.4 V (datasheet 1.47 N·m) | Not identified on our servos |
| Supply | 7.4–8.0 V, sagging with load down to 7.0 V | Assumed |
| Command lag | 3–6 physics steps | Assumed |
| Mass | 21 g per servo, 0.82 kg for the robot | 0.82 kg is a placeholder until the robot is weighed |
| Backlash | Separate `-Backlash` tasks; the kick-left and pick policies are fine-tuned with it | Backlash size not measured |

The runtime must speak the Feetech bus protocol. Pollen's runtime drives
Dynamixel XL330 servos; HD-1910 builds of it exist in
[fanhao375/Microduck-kissqy](https://github.com/fanhao375/Microduck-kissqy).

## Before a policy goes on a robot

- It passes its acceptance contract (`scripts/acceptance.py`) and the
  sim2sim (`scripts/sim2sim.py`); see each skill's example for the numbers.
- Start with the motors limp and the robot held, and check every joint's
  sign and zero against the simulation before giving the policy control.

No policy from this repository has run on a robot yet.
