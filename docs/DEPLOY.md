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
configs, unchanged here, except on `Mjlab-SitStand-Flat-MicroDuck-HD1910-Robotd`,
which trains with robotd's target processing (below).

## What robotd does to the targets

robotd (the HD-1910 runtime; this page follows
[harrysungit/Microduck-kissqy](https://github.com/harrysungit/Microduck-kissqy),
branch `harry/hd1910-control48`, a fork of kissqy's) does not send
`home + action` to the servos. Every tick it scales the action by
`7.4 V / battery` (`voltage_adapt`, on by default) and low-pass filters the
targets against the previous ones: `0.5 × new + 0.5 × previous` on the head,
`0.7 × new + 0.3 × previous` on the legs; the first tick after the policy is
enabled goes through unfiltered. The policy still observes its raw action.
On the robot at about 8.0 V the readback was a scale of 0.92–0.93 and filters
0.5 / 0.7.

A policy from a plain task never saw this. `-Robotd` sit-stand trains with it
(`src/microduck_rl/actions/robotd.py`), over a 7.0–8.4 V battery, and the
sim2sim's robotd levels (`L0-robotd`, `L3-robotd*`) evaluate with the same
arithmetic. The exported
ONNX does not declare the filter: robotd applies its own defaults, so keep
them, or give the model a deployment contract that states them.

## What differs on an HD-1910 robot

The policies were trained against these assumptions. A robot that differs
from them is outside what was simulated.

| Assumption | Value in training | Status |
|---|---|---|
| Servo control | Feetech firmware position loop, P = 5 | Not checked against our servos' settings |
| Servo dynamics | BAM M6 fit for the HD-1910 by LuwuDynamics; about 0.91 N·m stall at 7.4 V (datasheet 1.47 N·m) | Not identified on our servos |
| Supply | 7.4–8.0 V (7.0–8.4 V on `-Robotd` sit-stand), sagging with load down to 7.0 V | Assumed; the robot read 7.6–8.0 V |
| Command lag | 3–6 physics steps | Assumed |
| Mass | 21 g per servo, 0.82 kg for the robot | Robot weighed at 0.821 kg; per-part masses and centre of mass not measured |
| Target processing | `home + action` (plain tasks); robotd's voltage scale and low-pass (`-Robotd` sit-stand) | robotd's defaults, read back on the robot |
| Servo gain in sit-stand | P = 5 throughout | robotd writes the deployment contract's firmware P (default 5) to Feetech servos and drops later gain requests, the stand-up gain ratio included |
| Sit-stand start | Each episode starts with an empty target filter | On the robot the filter carries the walk policy's last target into sit-stand |
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
