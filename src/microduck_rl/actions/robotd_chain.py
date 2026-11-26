"""robotd's runtime target processing, as plain arithmetic.

On the duck the servos do not receive ``home + action``. For each policy tick
robotd (github.com/harrysungit/Microduck-kissqy, a fork of kissqy's, branch
``harry/hd1910-control48`` @ 347007f; paths under ``microduck-hd1910/firmware/``):

1. scales the action by ``nominal_voltage / battery_v.clamp(6.0, 9.5)`` when
   ``voltage_adapt`` is on (``robotd/src/main.rs:3776-3777``; defaults
   ``voltage_adapt = true``, ``nominal_voltage = 7.4`` at
   ``robotd-params/src/lib.rs:1819-1820``). Sit-stand's own base scale is 1.0;
2. low-pass filters each target against the previous filtered target,
   ``alpha * new + (1 - alpha) * previous``, with alpha 0.5 on the head
   (neck_pitch, head_pitch, head_yaw, head_roll) and 0.7 on the other joints
   except the mouth (``robotd/src/control.rs:40``, ``:755-771``; defaults at
   ``robotd-params/src/lib.rs:1680-1681``).
   The first tick after ``Controller::reset()`` (policy re-enabled) has no
   previous target and goes through unfiltered (``control.rs:339-346``).

The policy's observed last action stays its raw output. The servo gain stays
put: on Feetech servos robotd writes the deployment contract's firmware P
(default 5, ``duck-control/src/deployment.rs:78-80``) and drops every later
gain request, the stand-up gain ratio included (``feetech.rs:2813-2823``).

The functions take torch tensors or numpy arrays alike, so the training's
action term (robotd.py) and the plain-MuJoCo evaluator (scripts/sim2sim.py)
run the same arithmetic. No mjlab or torch import here.
"""

from __future__ import annotations

NOMINAL_VOLTAGE = 7.4
HEAD_LOWPASS = 0.5
LEGS_LOWPASS = 0.7
HEAD_JOINTS = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")
UNFILTERED_JOINTS = ("mouth",)
BATTERY_CLAMP_V = (6.0, 9.5)  # robotd clamps the reading before dividing (main.rs:3777)


def voltage_scale(battery_v, nominal_voltage: float = NOMINAL_VOLTAGE):
    """robotd's voltage adaptation factor, ``nominal / clamp(battery, 6.0, 9.5)``."""
    return nominal_voltage / battery_v.clip(*BATTERY_CLAMP_V)


def joint_alphas(joint_names, head: float = HEAD_LOWPASS, legs: float = LEGS_LOWPASS) -> list[float]:
    """Per-joint weight of the new target, in ``joint_names`` order; 1.0 leaves a joint unfiltered."""
    missing = [n for n in HEAD_JOINTS if n not in joint_names]
    if missing:
        raise ValueError(f"robotd's head filter joints {missing} are not among {list(joint_names)}")
    for name, a in (("head", head), ("legs", legs)):
        if not 0.0 < a <= 1.0:
            raise ValueError(f"{name} low-pass must be in (0, 1], got {a}")
    return [1.0 if n in UNFILTERED_JOINTS else head if n in HEAD_JOINTS else legs for n in joint_names]


def targets(raw, offset, scale, previous, fresh, alpha):
    """One robotd tick: ``offset + scale * raw``, then the per-joint low-pass.

    Args:
      raw: (N, J) policy output.
      offset: HOME targets, (N, J), (J,) or a scalar.
      scale: total action scale, (N, 1), (N, J) or a scalar.
      previous: (N, J) filtered targets of the last tick (ignored where ``fresh``).
      fresh: (N, 1) bool, True where the filter has no previous target yet.
      alpha: (J,) weight of the new target.
    """
    target = offset + scale * raw
    # Exactly target where fresh, exactly the filtered value elsewhere (no where(): numpy and torch alike).
    return fresh * target + (~fresh) * (alpha * target + (1.0 - alpha) * previous)
