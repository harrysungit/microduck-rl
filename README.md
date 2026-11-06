# microduck-rl

Training for the [Microduck](https://github.com/pollen-robotics/microduck) on
more servos and GPUs, built on Pollen Robotics'
[microduck_rl](https://github.com/pollen-robotics/microduck_rl).

This is not Pollen's repository. It is a package that depends on theirs,
unchanged, and registers extra [mjlab](https://github.com/mujocolab/mjlab)
tasks next to the official ones through mjlab's `mjlab.tasks` entry point.
