"""Feetech HD-1910 servo variant of the Microduck.

The official robot uses Dynamixel XL330 servos. The HD-1910 build swaps in
Feetech HD-1910 servos, which the policy feels in two places:

- actuator.py: BAM M6 parameters fitted for the HD-1910, driven by BAM's
  Feetech control law at the firmware P gain the robot writes to the servos;
- mass.py: 21 g per servo instead of 18 g, and a heavier build overall.

Joints, HOME pose, observations and actions are those of the XL330 tasks, so
an exported policy keeps the [1, 61] -> [1, 14] interface. Tasks:
Mjlab-Velocity-{Flat,Rough}[-Backlash]-MicroDuck-HD1910 (tasks/__init__.py).
"""
