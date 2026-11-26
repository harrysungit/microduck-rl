"""Run an exported walking policy in plain MuJoCo (CPU), away from the training world.

The sim2sim check of the ONNX that goes to the robot. Nothing here imports
mjlab or mjlab_microduck: the scenes are standalone MJCF exported from the
robot description (sim2sim_assets.py), and the servos, the control loop and
the observations are rebuilt the way the robot runs them:

- servos: BAM's HD-1910 model (1910_m6.json: BAM's control law, torque and
  friction functions) at firmware P = 5, supply sag, command latency in
  milliseconds, the Feetech goal seeded from the measured position after a
  reset, and BAM's friction written to dof_frictionloss every step;
- observations from the IMU sensors and the joint encoders; actions
  HOME + scale * a at 50 Hz, or on the robotd levels the targets as robotd
  sends them (voltage scale and low-pass, microduck_rl.actions.robotd_chain:
  plain arithmetic shared with the training task that uses it).

A ladder of levels changes one thing at a time, from the training settings
to the robot's (LEVELS). Every level drives the same command schedule and is
scored on tracking, falls, contacts the training world never has, torque
saturation and smoothness. Push levels hit the trunk harder than training
does; recovery levels start the robot lying down and give it 8 s with a zero
command to get up. --reference adds the training world's own run of the
schedule (sim2sim_reference.py) for comparison with L0.

    python scripts/sim2sim.py --onnx <policy.onnx> --assets logs/sim2sim/assets \
        --reference <dir>/reference_traces.csv --out-dir <dir>

Sitstand policies: --task sitstand drives the posture flag (stand, sit, stand, ...) from a standing and from a seated
start, on the training's ground-contact model (scenes from sim2sim_assets.py), and scores whether the trunk reaches and
holds each posture:

    python scripts/sim2sim.py --task sitstand --onnx <policy.onnx> --assets logs/sim2sim/assets_v2 --out-dir <dir>
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")  # headless rendering for --video

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort
from bam.model import load_model

from microduck_rl.actions import robotd_chain  # plain arithmetic, no mjlab: the same chain the training task runs

ROOT = Path(__file__).resolve().parents[1]
SERVO_PARAMS = ROOT / "src/microduck_rl/actuators/hd1910/params/1910_m6.json"

KP_FW = 5.0  # firmware P the robot writes to the servos
VIN_RANGE = (7.4, 8.0)  # 2S battery
VIN_MIN = 7.0
ROBOTD_VIN_RANGE = (7.0, 8.4)  # the robotd-chain task's battery range (actuators/hd1910/robot.py)
# BAM's friction-constraint stiffening (bam/mjlab.py), part of its friction model.
STIFF_SOLREF_FRICTION = (-5.0e4, -2.0e2)
STIFF_SOLIMP_FRICTION = (0.99, 0.9999, 0.001, 0.5, 2.0)
CONTROL_DT = 0.02
FELL_DEG = 70.0  # the training's fell_over
ENCODER_STEP = 2 * math.pi / 4096
SETTLE_S = 1.0  # left out at the start of each segment
HEAD_BODIES = ("neck", "neck_pitch", "yaw_roll_motion", "jaw_soft", "bearing_roll")
# (duration s, (vx, vy, wz) in the body frame, label); within the training ranges.
SCHEDULE = [
    (3.0, (0.0, 0.0, 0.0), "stand"),
    (6.0, (0.3, 0.0, 0.0), "forward 0.3"),
    (5.0, (-0.2, 0.0, 0.0), "backward 0.2"),
    (5.0, (0.0, 0.2, 0.0), "left 0.2"),
    (5.0, (0.0, -0.2, 0.0), "right 0.2"),
    (5.0, (0.0, 0.0, 0.8), "turn left 0.8"),
    (5.0, (0.0, 0.0, -0.8), "turn right 0.8"),
    (5.0, (0.3, 0.0, 0.5), "forward 0.3 + turn 0.5"),
    (2.0, (0.0, 0.0, 0.0), "stand"),
]
DURATION = sum(d for d, _, _ in SCHEDULE)


def command_at(t: float, schedule: list = SCHEDULE) -> np.ndarray:
    for duration, cmd, _ in schedule:
        if t < duration:
            return np.array(cmd)
        t -= duration
    return np.zeros(3)


@dataclass(frozen=True)
class Level:
    key: str
    title: str
    scene: str = "allcollisions"  # walk | allcollisions | allcollisions_backlash
    collisions: str = "authored"  # "feet": only the feet collide, as in training
    timestep: float = 0.002
    iterations: int = 100
    ls_iterations: int = 50
    sensors: bool = True  # IMU sensors, quantized encoders, joint speeds from position differences
    joint_vel_lag: int = 0  # control steps; training feeds joint speeds one step late
    lowpass: tuple | None = None  # robot target low-pass (head, legs)
    robotd: bool = False  # robotd's whole target chain: voltage scale and low-pass (microduck_rl.actions.robotd_chain)
    vin_range: tuple | None = None  # battery drawn per episode from this range; also sets the servos' torque cap
    vin: float = 7.7
    drop_gain: float = 0.1
    latency_ms: tuple = (20.0, 20.0)
    randomize: bool = False
    episodes: int = 1
    push_mps: float = 0.0  # push the trunk every 3-6 s by up to this much (randomized levels: 0.3)
    start: str = "standing"  # recovery levels: "side", "front" or "back", with a zero command; sitstand: "seated"
    task: str = "walk"  # walk | sitstand | kick
    ball_noise: float = 0.0  # kick: the ball's placement noise, m (uniform, both axes); randomized levels use the training's
    kick_foot: str = "right"


_TRAINING_SOLVER = dict(timestep=0.005, iterations=10, ls_iterations=20)
LEVELS = [
    Level("L0", "training-equivalent", scene="walk", collisions="feet", sensors=False, joint_vel_lag=1,
          **_TRAINING_SOLVER),
    Level("L1", "+ all collisions", sensors=False, joint_vel_lag=1, **_TRAINING_SOLVER),
    Level("L2", "+ accurate solver (2 ms, 100/50)", sensors=False, joint_vel_lag=1),
    Level("L3", "+ sensor observations"),
    Level("L4", "+ robot low-pass 0.5 / 0.7", lowpass=(0.5, 0.7)),
    Level("L3-dr", "L3 + training randomization", randomize=True, episodes=64, latency_ms=(15.0, 30.0)),
    Level("L3-backlash", "L3 + 1 degree backlash", scene="allcollisions_backlash"),
    Level("L3-stress", "L3 at 7.0 V and 30 ms", vin=7.0, latency_ms=(30.0, 30.0)),
    Level("push-0.3", "L3 + pushes up to 0.3 m/s (as in training)", push_mps=0.3, episodes=64),
    Level("push-0.6", "L3 + pushes up to 0.6 m/s", push_mps=0.6, episodes=64),
    Level("push-0.9", "L3 + pushes up to 0.9 m/s", push_mps=0.9, episodes=64),
    Level("up-side", "get up from lying on its side", start="side", episodes=16),
    Level("up-front", "get up from lying face down", start="front", episodes=16),
    Level("up-back", "get up from lying on its back", start="back", episodes=16),
]
RECOVERY_S = 8.0
UPRIGHT_DEG = 30.0
# Lying poses: the trunk rolled or pitched by 90 degrees (w, x, y, z), dropped from a few cm.
LYING = {"side": (math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0),
         "front": (math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0),
         "back": (math.cos(math.pi / 4), 0.0, -math.sin(math.pi / 4), 0.0)}


# -- sitstand: one policy, commanded sit <-> stand. The command is a posture flag in the first command slot,
# (flag, 0, 0): 0 = stand, 1 = sit; the robot's runtime writes it.
SIT_FROM_STAND = [
    (3.0, (0.0, 0.0, 0.0), "stand"),
    (5.0, (1.0, 0.0, 0.0), "sit"),
    (5.0, (0.0, 0.0, 0.0), "stand"),
    (5.0, (1.0, 0.0, 0.0), "sit"),
    (5.0, (0.0, 0.0, 0.0), "stand"),
]
SIT_FROM_SEAT = [
    (3.0, (1.0, 0.0, 0.0), "sit"),
    (5.0, (0.0, 0.0, 0.0), "stand"),
    (5.0, (1.0, 0.0, 0.0), "sit"),
    (5.0, (0.0, 0.0, 0.0), "stand"),
]
# The training's SIT keyframe (SITTING_TARGET_OVERRIDES: servo index -> rad; neck and head stay at HOME) and its seated
# spawn: trunk 6-7.5 cm high, tilted up to 8 degrees in pitch and roll, 0.1 rad of noise on every servo. Copied, not
# imported: this runner must not touch the training code.
SIT_KEYFRAME = {1: 0.0, 2: -0.4079, 3: 1.35, 4: 0.0, 10: 0.0, 11: 0.4079, 12: -1.35, 13: 0.0}
SEAT_Z, SEAT_TILT, SEAT_JOINT_NOISE = (0.06, 0.075), math.radians(8.0), 0.10
# The training's own thresholds: SIT_UPRIGHT_Z (committed to the sit) and STAND_UPRIGHT_Z (fully up).
SIT_Z_MAX, STAND_Z_MIN = 0.075, 0.10
STAND_Z, SIT_Z = 0.115, 0.060  # the training's measured rest heights, for the plot

_SIT_SOLVER = dict(timestep=0.005, iterations=30, ls_iterations=50)  # the sitstand training's solver
SITSTAND_LEVELS = [
    Level("L0", "training-equivalent", scene="groundcontact", collisions="groundcontact", sensors=False,
          joint_vel_lag=1, task="sitstand", **_SIT_SOLVER),
    Level("L1", "+ all collisions", sensors=False, joint_vel_lag=1, task="sitstand", **_SIT_SOLVER),
    Level("L2", "+ accurate solver (2 ms, 100/50)", sensors=False, joint_vel_lag=1, task="sitstand"),
    Level("L3", "+ sensor observations", task="sitstand"),
    Level("L3-dr", "L3 + training randomization", randomize=True, episodes=64, latency_ms=(15.0, 30.0), task="sitstand"),
    Level("L3-backlash", "L3 + 1 degree backlash", scene="allcollisions_backlash", task="sitstand"),
    Level("L3-stress", "L3 at 7.0 V and 30 ms", vin=7.0, latency_ms=(30.0, 30.0), task="sitstand"),
    Level("push-0.3", "L3 + pushes up to 0.3 m/s (as in training)", push_mps=0.3, episodes=64, task="sitstand"),
    Level("push-0.6", "L3 + pushes up to 0.6 m/s", push_mps=0.6, episodes=64, task="sitstand"),
    Level("L3-seated", "L3, starting seated (SIT keyframe + noise)", start="seated", episodes=16, task="sitstand"),
    # The targets as robotd sends them (Mjlab-SitStand-Flat-MicroDuck-HD1910-Robotd trains with the same chain),
    # over a 2S battery from full to about 7.0 V: one episode per drawn voltage, per seated start, per randomization.
    # L0-robotd keeps the training-equivalent physics: the stand-up turn depends on the collision set and the solver as
    # much as on the chain, so a report shows both.
    Level("L0-robotd", "L0 + robotd's target chain, battery 7.0-8.4 V", scene="groundcontact", collisions="groundcontact",
          sensors=False, joint_vel_lag=1, robotd=True, vin_range=ROBOTD_VIN_RANGE, episodes=16, task="sitstand",
          **_SIT_SOLVER),
    Level("L3-robotd", "L3 + robotd's target chain, battery 7.0-8.4 V", robotd=True, vin_range=ROBOTD_VIN_RANGE,
          episodes=16, task="sitstand"),
    Level("L3-robotd-seated", "L3-robotd, starting seated", robotd=True, vin_range=ROBOTD_VIN_RANGE, start="seated",
          episodes=16, task="sitstand"),
    Level("L3-robotd-dr", "L3-dr + robotd's target chain, battery 7.0-8.4 V", robotd=True, vin_range=ROBOTD_VIN_RANGE,
          randomize=True, episodes=64, latency_ms=(15.0, 30.0), task="sitstand"),
]
# -- kick: the ball sits in front of the kicking foot, which the policy cannot see; it must go forward. Placement and ball
# as the training's (BallKick env cfg: offset 0.09 m ahead and 0.042 m to the side, noise 1.5 cm; ball 15 g, radius 3.5 cm).
BALL_OFFSET_X, BALL_OFFSET_ABS_Y, BALL_NOISE, BALL_RADIUS = 0.09, 0.042, 0.015, 0.035
KICK_MIN_SPEED = 0.5  # m/s along the kick direction; the same threshold as acceptance_core.KICK_MIN_SPEED
KICK_SCHEDULE = [(5.0, (0.0, 0.0, 0.0), "kick")]  # the training's 5 s episode, zero command
KICK_LEVELS = [
    Level("L0", "training-equivalent", scene="groundcontact_ball", collisions="groundcontact", sensors=False,
          joint_vel_lag=1, task="kick", **_TRAINING_SOLVER),
    Level("L1", "+ all collisions", scene="allcollisions_ball", sensors=False, joint_vel_lag=1, task="kick",
          **_TRAINING_SOLVER),
    Level("L2", "+ accurate solver (2 ms, 100/50)", scene="allcollisions_ball", sensors=False, joint_vel_lag=1,
          task="kick"),
    Level("L3", "+ sensor observations", scene="allcollisions_ball", task="kick"),
    Level("L3-ball", "L3, ball placed as in training (+-1.5 cm)", scene="allcollisions_ball", ball_noise=BALL_NOISE,
          episodes=64, task="kick"),
    Level("L3-ball-wide", "L3, ball placed +-3 cm (beyond training)", scene="allcollisions_ball", ball_noise=0.03,
          episodes=64, task="kick"),
    Level("L3-dr", "L3 + training randomization", scene="allcollisions_ball", randomize=True, episodes=64,
          latency_ms=(15.0, 30.0), task="kick"),
    Level("L3-backlash", "L3 + 1 degree backlash", scene="allcollisions_backlash_ball", task="kick"),
    Level("L3-stress", "L3 at 7.0 V and 30 ms", scene="allcollisions_ball", vin=7.0, latency_ms=(30.0, 30.0), task="kick"),
]
LEVEL_SETS = {"walk": LEVELS, "sitstand": SITSTAND_LEVELS, "kick": KICK_LEVELS}


def schedule_for(level: Level) -> list:
    if level.task == "sitstand":
        return SIT_FROM_SEAT if level.start == "seated" else SIT_FROM_STAND
    if level.task == "kick":
        return KICK_SCHEDULE
    return SCHEDULE


def ball_position(level: Level, rng: np.random.Generator) -> tuple[float, float]:
    """Where the ball starts, in the robot's frame at spawn (heading along +x): in front of the kicking foot, with noise."""
    noise = level.ball_noise or (BALL_NOISE if level.randomize else 0.0)
    dx, dy = rng.uniform(-noise, noise, 2) if noise else (0.0, 0.0)
    return BALL_OFFSET_X + float(dx), (-BALL_OFFSET_ABS_Y if level.kick_foot == "right" else BALL_OFFSET_ABS_Y) + float(dy)


def seated_pose(home: np.ndarray, rng: np.random.Generator) -> tuple[float, tuple, np.ndarray]:
    """Trunk height, orientation (w, x, y, z) and servo angles of a seated start, as the training spawns one."""
    yaw, pitch, roll = rng.uniform(-math.pi, math.pi), *rng.uniform(-SEAT_TILT, SEAT_TILT, 2)
    cy, sy, cp, sp, cr, sr = (f(a * 0.5) for a in (yaw, pitch, roll) for f in (math.cos, math.sin))
    quat = (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy)
    joints = home.copy()
    for i, angle in SIT_KEYFRAME.items():
        joints[i] = angle
    joints += rng.normal(0.0, SEAT_JOINT_NOISE, len(home))
    return float(rng.uniform(*SEAT_Z)), quat, joints


def sample_params(level: Level, rng: np.random.Generator, n: int) -> dict:
    """Per-episode physical parameters: nominal, or drawn like the training's randomization."""
    p = dict(vin=level.vin, drop_gain=level.drop_gain, foot_friction=None, trunk_mass_scale=1.0,
             trunk_com=np.zeros(3), head_com={}, friction_scale=1.0, armature_scale=np.ones(n),
             encoder_bias=np.zeros(n), imu_rotation=None, noise=False, pushes=False)
    if level.vin_range is not None and not level.randomize:
        p["vin"] = rng.uniform(*level.vin_range)
    if level.randomize:
        axis = rng.normal(size=3)
        p.update(vin=rng.uniform(*(level.vin_range or VIN_RANGE)), drop_gain=rng.uniform(0.0, 0.2),
                 foot_friction=rng.uniform(0.7, 1.3), trunk_mass_scale=rng.uniform(0.95, 1.05),
                 trunk_com=rng.uniform(-0.015, 0.015, 3),
                 head_com={b: rng.uniform(-0.01, 0.01, 3) for b in HEAD_BODIES},
                 friction_scale=rng.uniform(0.9, 1.1), armature_scale=rng.uniform(0.9, 1.1, n),
                 encoder_bias=rng.uniform(-0.015, 0.015, n),
                 imu_rotation=axis_angle(axis / np.linalg.norm(axis), math.radians(rng.uniform(0.0, 6.0))),
                 noise=True, pushes=True)
    p["push_mps"] = level.push_mps or (0.3 if level.randomize else 0.0)
    p["pushes"] = p["push_mps"] > 0
    return p


def axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * k @ k


class Policy:
    """The exported ONNX, run as on the robot; joint order, HOME and scale from its metadata."""

    def __init__(self, path: Path):
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        meta = self.session.get_modelmeta().custom_metadata_map
        self.joint_names = meta["joint_names"].split(",")
        self.home = np.array([float(v) for v in meta["default_joint_pos"].split(",")])
        self.scale = float(meta.get("action_scale", "1.0"))
        self.input = self.session.get_inputs()[0].name

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input: obs[None].astype(np.float32)})[0][0].astype(float)


def build_model(level: Level, assets: Path, joint_names: list[str], servo_params: Path, p: dict) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(assets / f"scene_hd1910_{level.scene}.xml"))
    opt = spec.option
    opt.timestep, opt.iterations, opt.ls_iterations = level.timestep, level.iterations, level.ls_iterations
    opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    opt.tolerance, opt.ls_tolerance, opt.impratio = 1e-8, 0.01, 1.0

    if level.collisions in ("feet", "groundcontact"):
        # The training's collision settings: feet condim 3, priority 1; other named *_collision geoms condim 1.
        # "feet" (the walk model): every other geom is switched off. "groundcontact": the other geoms keep the
        # contacts the model authors, which is how a seated duck rests on its trunk, hips, shanks and jaw.
        for geom in spec.geoms:
            if geom.name == "floor":
                continue
            if re.fullmatch(r"(left|right)_foot_collision", geom.name):
                geom.contype, geom.conaffinity, geom.condim, geom.priority = 1, 1, 3, 1
                geom.friction[0] = 1.0
            elif re.fullmatch(r".*_collision", geom.name):
                geom.contype, geom.conaffinity, geom.condim = 1, 1, 1
            elif level.collisions == "feet":
                geom.contype, geom.conaffinity = 0, 0

    # Servos: the XML position actuators become gain-1 torque motors, capped at the stall torque of the
    # highest supply voltage; MuJoCo's joint damping and friction give way to BAM's.
    bam = load_model(str(servo_params))
    force_limit = max(level.vin_range or VIN_RANGE) * bam.kt.value / bam.R.value
    servo_joints = set(joint_names)
    for actuator in spec.actuators:
        target = actuator.target if isinstance(actuator.target, str) else actuator.target.name
        if target in servo_joints:
            actuator.set_to_motor()
            actuator.forcelimited = True
            actuator.forcerange = (-force_limit, force_limit)
            actuator.gear = [1.0, 0, 0, 0, 0, 0]
    for joint in spec.joints:
        if joint.name in servo_joints:
            joint.armature = float(bam.actuator.get_extra_inertia())
            joint.damping = np.zeros((3, 1))
            joint.frictionloss = 0.0
            joint.solref_friction = STIFF_SOLREF_FRICTION
            joint.solimp_friction = STIFF_SOLIMP_FRICTION
    model = spec.compile()

    if p["foot_friction"] is not None:
        for name in ("left_foot_collision", "right_foot_collision"):
            model.geom_friction[model.geom(name).id, 0] = p["foot_friction"]
    trunk = model.body("trunk_base").id
    model.body_mass[trunk] *= p["trunk_mass_scale"]
    model.body_inertia[trunk] *= p["trunk_mass_scale"]
    model.body_ipos[trunk] += p["trunk_com"]
    for name, offset in p["head_com"].items():
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if b >= 0:
            model.body_ipos[b] += offset
    dofs = [model.jnt_dofadr[model.joint(n).id] for n in joint_names]
    model.dof_armature[dofs] *= p["armature_scale"]
    return model


class CommandQueue:
    """Targets as sent, released to the servos after a latency drawn every physics step."""

    def __init__(self, latency_s: tuple[float, float], rng: np.random.Generator):
        self.latency_s, self.rng = latency_s, rng
        self.items: deque = deque()

    def reset(self, targets: np.ndarray) -> None:
        self.items = deque([(-math.inf, targets.copy())])

    def send(self, t: float, targets: np.ndarray) -> None:
        self.items.append((t, targets.copy()))

    def at(self, t: float) -> np.ndarray:
        cutoff = t - self.rng.uniform(*self.latency_s)
        # Keep the newest target that is past the longest latency; older ones are never needed again.
        while len(self.items) > 1 and self.items[1][0] <= t - self.latency_s[1] + 1e-12:
            self.items.popleft()
        for sent, targets in reversed(self.items):
            if sent <= cutoff + 1e-12:
                return targets
        return self.items[0][1]


class Servos:
    """The robot's HD-1910 servos: BAM's model driving the MuJoCo motors every physics step."""

    def __init__(self, model: mujoco.MjModel, joint_names: list[str], servo_params: Path, p: dict,
                 latency_ms: tuple[float, float], rng: np.random.Generator):
        self.bam = load_model(str(servo_params))
        self.act = self.bam.actuator
        self.act.kp = KP_FW
        self.vin, self.drop_gain = p["vin"], p["drop_gain"]
        self.friction_scale, self.bias = p["friction_scale"], p["encoder_bias"]
        joints = [model.joint(n).id for n in joint_names]
        self.qadr, self.dadr = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
        motor_of = {int(model.actuator_trnid[a, 0]): a for a in range(model.nu)}
        self.ctrl = np.array([motor_of[j] for j in joints])
        self.force_limit = model.actuator_forcerange[self.ctrl, 1]
        # Backlash models: the encoder sits on the output side of the gear play.
        backlash = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"passive_{n}_backlash") for n in joint_names]
        self.bl_mask = np.array([b >= 0 for b in backlash], dtype=float)
        self.bl_qadr = np.array([model.jnt_qposadr[b] if b >= 0 else 0 for b in backlash])
        self.bl_dadr = np.array([model.jnt_dofadr[b] if b >= 0 else 0 for b in backlash])
        self.dt = model.opt.timestep
        self.queue = CommandQueue((latency_ms[0] / 1e3, latency_ms[1] / 1e3), rng)
        self.prev_torque = np.zeros(len(joints))

    def measured(self, data: mujoco.MjData) -> np.ndarray:
        """Encoder reading: motor angle, plus the gear play on backlash models, plus the encoder's bias."""
        return data.qpos[self.qadr] + self.bl_mask * data.qpos[self.bl_qadr] + self.bias

    def reset(self, data: mujoco.MjData, targets: np.ndarray) -> None:
        self.act.q_target_smooth = self.measured(data).copy()  # the Feetech goal starts where the servo is
        self.prev_torque[:] = 0.0
        self.queue.reset(targets)

    def step(self, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
        q, dq = self.measured(data), data.qvel[self.dadr]  # dq: rotor side, for back-EMF and friction
        vin = self.vin
        if self.drop_gain:
            vin = max(vin - self.drop_gain * np.abs(self.prev_torque).sum(), VIN_MIN)
        self.act.vin = vin
        control = self.act.compute_control(self.queue.at(data.time), q, dq, self.dt)
        torque = np.asarray(self.act.compute_torque(control, True, q, dq), dtype=float)
        self.prev_torque = torque
        data.ctrl[self.ctrl] = torque

        # BAM's friction: the external load on each servo excludes its own friction-loss force. Those
        # constraint rows carry the DOF index in efc_id (BAM's MujocoController compares joint ids,
        # six off here because of the free joint).
        external = -data.qfrc_bias[self.dadr] + data.qfrc_constraint[self.dadr]
        n = data.nefc
        rows = data.efc_type[:n] == mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF
        if rows.any():
            ids, forces = data.efc_id[:n][rows], data.efc_force[:n][rows]
            external -= np.array([forces[ids == d].sum() for d in self.dadr])
        frictionloss, damping = self.bam.compute_frictions(data.qfrc_actuator[self.dadr], external, dq)
        model.dof_frictionloss[self.dadr] = frictionloss * self.friction_scale
        model.dof_damping[self.dadr] = damping
        return torque


class Observer:
    """The policy's 61 inputs, built the way the robot builds them."""

    def __init__(self, model: mujoco.MjModel, servos: Servos, home: np.ndarray, level: Level, p: dict,
                 rng: np.random.Generator):
        self.servos, self.home, self.level, self.p, self.rng = servos, home, level, p, rng
        self.trunk = model.body("trunk_base").id
        self.gyro = model.sensor("imu_ang_vel").adr[0]
        self.orientation = model.sensor("orientation").adr[0]
        self.prev_q = None
        self.speeds: deque = deque()

    def encoders(self, data: mujoco.MjData) -> np.ndarray:
        q = self.servos.measured(data)
        return np.round(q / ENCODER_STEP) * ENCODER_STEP if self.level.sensors else q

    def joint_speeds(self, data: mujoco.MjData, q: np.ndarray) -> np.ndarray:
        if self.level.sensors:
            return (q - self.prev_q) / CONTROL_DT
        s = self.servos
        return data.qvel[s.dadr] + s.bl_mask * data.qvel[s.bl_dadr]

    def reset(self, data: mujoco.MjData) -> None:
        q = self.encoders(data)
        self.prev_q = q
        self.speeds = deque([self.joint_speeds(data, q)] * (self.level.joint_vel_lag + 1),
                            maxlen=self.level.joint_vel_lag + 1)

    def __call__(self, data: mujoco.MjData, last_action: np.ndarray, command: np.ndarray) -> np.ndarray:
        if self.level.sensors:
            gyro = data.sensordata[self.gyro:self.gyro + 3].copy()
            rot = np.zeros(9)
            mujoco.mju_quat2Mat(rot, data.sensordata[self.orientation:self.orientation + 4])
            rot = rot.reshape(3, 3)
        else:
            gyro = data.qvel[3:6].copy()  # free joint: body-frame angular velocity
            rot = data.xmat[self.trunk].reshape(3, 3)
        gravity = rot.T @ np.array([0.0, 0.0, -1.0])
        if self.p["imu_rotation"] is not None:
            gyro, gravity = self.p["imu_rotation"] @ gyro, self.p["imu_rotation"] @ gravity
        q = self.encoders(data)
        self.speeds.append(self.joint_speeds(data, q))
        self.prev_q = q
        q_rel, speed = q - self.home, self.speeds[0].copy()
        if self.p["noise"]:  # the training's observation noise
            u = lambda a, n: self.rng.uniform(-a, a, n)  # noqa: E731
            gyro, gravity = gyro + u(0.03, 3), gravity + u(0.01, 3)
            q_rel, speed = q_rel + u(0.001, len(q)), speed + u(0.25, len(q))
        return np.concatenate([gyro, gravity, q_rel, speed, last_action, command, np.zeros(4), np.zeros(6)])


def segment_metrics(traces: dict, schedule: list = SCHEDULE) -> list[dict]:
    """Mean body-frame velocities per schedule segment, after SETTLE_S, against the command."""
    t = np.asarray(traces["t"])
    out, start = [], 0.0
    for duration, cmd, label in schedule:
        sel = (t > start + SETTLE_S) & (t <= start + duration)
        row = {"segment": label, "command": list(cmd), "samples": int(sel.sum())}
        if sel.any():
            got = [float(np.mean(np.asarray(traces[k])[sel])) for k in ("vx", "vy", "wz")]
            row.update(achieved=got, err_vxy=float(math.hypot(got[0] - cmd[0], got[1] - cmd[1])),
                       err_wz=float(abs(got[2] - cmd[2])))
        out.append(row)
        start += duration
    return out


def posture_segments(traces: dict, schedule: list) -> list[dict]:
    """Per segment of a posture schedule: whether the trunk got to the commanded posture and stayed there to the end
    of the segment, and how long into the segment it last arrived there."""
    t, z, tilt = (np.asarray(traces[k]) for k in ("t", "z", "tilt"))
    speed = np.hypot(np.asarray(traces["vx"]), np.asarray(traces["vy"]))
    yaw = np.degrees(np.unwrap(np.radians(traces["yaw"]))) if "yaw" in traces else None
    out, start, previous = [], 0.0, None
    for duration, cmd, label in schedule:
        sit = cmd[0] > 0.5
        seg = (t > start + 1e-9) & (t <= start + duration + 1e-9)
        row = {"segment": label, "command": list(cmd), "samples": int(seg.sum()), "settled_s": None, "ok": False,
               "transition": "hold" if previous is None or previous == sit else ("sit-down" if sit else "stand-up")}
        if seg.any():
            good = ((z <= SIT_Z_MAX) if sit else (z >= STAND_Z_MIN)) & (tilt < UPRIGHT_DEG)
            seg_t, seg_good = t[seg], good[seg]
            bad = np.flatnonzero(~seg_good)
            if bad.size == 0:
                row["settled_s"] = 0.0
            elif bad[-1] < len(seg_good) - 1:
                row["settled_s"] = float(seg_t[bad[-1] + 1] - start)
            row["ok"] = row["settled_s"] is not None
            tail = seg & (t > start + duration - 1.0)
            row.update(z_end=float(z[tail].mean()), tilt_end=float(tilt[tail].mean()),
                       drift_end=float(speed[tail].mean()))
            if yaw is not None:  # net trunk turn over the segment, from the last sample before it
                idx = np.flatnonzero(seg)
                row["yaw_change_deg"] = float(yaw[idx[-1]] - yaw[max(idx[0] - 1, 0)])
        out.append(row)
        start += duration
        previous = sit
    return out


def kick_summary(traces: dict, start_x: float, start_y: float) -> dict:
    """The kick, from the ball's speed along +x and its position: peak speed, whether it counts, where the ball went."""
    peak = float(np.max(traces["ball_fwd"]))
    dx, dy = traces["ball_x"][-1] - start_x, traces["ball_y"][-1] - start_y
    travel = math.hypot(dx, dy)
    return {"kick_peak_mps": peak, "kick_ok": peak >= KICK_MIN_SPEED, "ball_travel_m": travel,
            "ball_angle_deg": math.degrees(math.atan2(dy, dx)) if travel > 0.02 else None}


def run_episode(level: Level, seed: int, onnx: Path, assets: Path, servo_params: Path,
                record: bool = False, video: Path | None = None) -> dict:
    rng = np.random.default_rng(seed)
    policy = Policy(onnx)
    n = len(policy.joint_names)
    p = sample_params(level, rng, n)
    model = build_model(level, assets, policy.joint_names, servo_params, p)
    data = mujoco.MjData(model)
    servos = Servos(model, policy.joint_names, servo_params, p, level.latency_ms, rng)
    observer = Observer(model, servos, policy.home, level, p, rng)

    recovery = level.start in LYING
    schedule = schedule_for(level)
    if recovery:
        data.qpos[0:3], data.qpos[3:7] = (0.0, 0.0, 0.1), LYING[level.start]
        # Small random yaw-free joint offsets so the 16 tries differ.
        data.qpos[servos.qadr] = policy.home + rng.uniform(-0.05, 0.05, n)
    elif level.start == "seated":
        assert policy.joint_names[3] == "left_knee" and policy.joint_names[12] == "right_knee"  # SIT_KEYFRAME's order
        z, quat, joints = seated_pose(policy.home, rng)
        data.qpos[0:3], data.qpos[3:7] = (0.0, 0.0, z), quat
        data.qpos[servos.qadr] = joints
    else:
        data.qpos[0:3], data.qpos[3:7] = (0.0, 0.0, 0.125), (1.0, 0.0, 0.0, 0.0)
        data.qpos[servos.qadr] = policy.home
    ball = None
    if level.task == "kick":
        bx, by = ball_position(level, rng)
        joint = model.joint("ball_free").id
        qadr, dadr = int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])
        data.qpos[qadr:qadr + 7] = (bx, by, BALL_RADIUS, 1.0, 0.0, 0.0, 0.0)
        ball = (qadr, dadr, bx, by)
    mujoco.mj_forward(model, data)
    servos.reset(data, policy.home.copy())
    observer.reset(data)

    floor = model.geom("floor").id
    # Feet are whole bodies: on the all-collision models a foot touches down with more than its sole.
    foot_bodies = [int(model.geom_bodyid[model.geom(g).id]) for g in ("left_foot_collision", "right_foot_collision")]
    body_name = [model.body(b).name for b in range(model.nbody)]
    ground_by_body: dict[str, int] = {}
    self_by_pair: dict[str, int] = {}
    joint_range = model.jnt_range[[model.joint(j).id for j in policy.joint_names]]
    near_limit = 0.02 * (joint_range[:, 1] - joint_range[:, 0])
    substeps = round(CONTROL_DT / model.opt.timestep)
    renderer = camera = ffmpeg = None
    if video:
        renderer = mujoco.Renderer(model, height=720, width=1280)
        camera = mujoco.MjvCamera()
        camera.type, camera.trackbodyid = mujoco.mjtCamera.mjCAMERA_TRACKING, observer.trunk
        camera.distance, camera.elevation, camera.azimuth = 0.8, -15.0, 135.0
        ffmpeg = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1280x720",
                                   "-r", str(round(1 / CONTROL_DT)), "-i", "-", "-pix_fmt", "yuv420p", str(video)],
                                  stdin=subprocess.PIPE)

    keys = ["t", "cmd_vx", "cmd_vy", "cmd_wz", "vx", "vy", "wz", "z", "tilt", "yaw"] + [f"a{i}" for i in range(n)]
    if ball:
        keys += ["ball_fwd", "ball_x", "ball_y"]  # the ball's speed along +x, the kick direction, and its position
    traces = {k: [] for k in keys}
    last_action, filtered = np.zeros(n), None
    action_scale, alpha = policy.scale, None  # alpha: the robot's target low-pass, head joints and legs
    if level.robotd:
        action_scale = policy.scale * float(robotd_chain.voltage_scale(np.float64(p["vin"])))
        alpha = np.array(robotd_chain.joint_alphas(policy.joint_names))
    elif level.lowpass is not None:
        alpha = np.array(robotd_chain.joint_alphas(policy.joint_names, *level.lowpass))
    next_push = rng.uniform(3.0, 6.0)
    ticks = round((RECOVERY_S if recovery else sum(d for d, _, _ in schedule)) / CONTROL_DT)
    upright_since = up_at = None
    self_ticks = ground_ticks = limit_ticks = saturated_steps = steps = 0
    peak_torque = np.zeros(n)
    fell_at, rate, power = None, [], []
    for k in range(ticks):
        command = np.zeros(3) if recovery else command_at(k * CONTROL_DT, schedule)
        action = policy(observer(data, last_action, command))
        if alpha is None:
            target = policy.home + action_scale * action
        else:
            fresh = np.array([filtered is None])
            target = robotd_chain.targets(action, policy.home, action_scale,
                                          policy.home if filtered is None else filtered, fresh, alpha)
            filtered = target
        servos.queue.send(data.time, target)
        if p["pushes"] and data.time >= next_push:
            data.qvel[0:2] += rng.uniform(-p["push_mps"], p["push_mps"], 2)
            next_push += rng.uniform(3.0, 6.0)
        ground_bodies, self_pairs = set(), set()
        for _ in range(substeps):
            torque = servos.step(model, data)
            mujoco.mj_step(model, data)
            steps += 1
            saturated_steps += bool(np.any(np.abs(torque) >= 0.99 * servos.force_limit))
            peak_torque = np.maximum(peak_torque, np.abs(torque))
            power.append(float(np.abs(torque * data.qvel[servos.dadr]).sum()))
            if data.ncon:
                b1 = model.geom_bodyid[np.array(data.contact.geom1)]
                b2 = model.geom_bodyid[np.array(data.contact.geom2)]
                g1 = np.array(data.contact.geom1)
                g2 = np.array(data.contact.geom2)
                for c in range(len(g1)):
                    if g1[c] == floor or g2[c] == floor:
                        body = b2[c] if g1[c] == floor else b1[c]
                        if body not in foot_bodies:
                            ground_bodies.add(body_name[body])
                    else:
                        self_pairs.add(" / ".join(sorted((body_name[b1[c]], body_name[b2[c]]))))
        mujoco.mj_forward(model, data)  # frames and sensors at the new state, as training does after its substeps
        self_ticks += bool(self_pairs)
        ground_ticks += bool(ground_bodies)
        for b in ground_bodies:
            ground_by_body[b] = ground_by_body.get(b, 0) + 1
        for pair in self_pairs:
            self_by_pair[pair] = self_by_pair.get(pair, 0) + 1
        q = data.qpos[servos.qadr]
        limit_ticks += bool(np.any((q - joint_range[:, 0] < near_limit) | (joint_range[:, 1] - q < near_limit)))
        rot = data.xmat[observer.trunk].reshape(3, 3)
        v = rot.T @ data.qvel[0:3]
        tilt = math.degrees(math.acos(np.clip(rot[2, 2], -1.0, 1.0)))  # trunk z axis against the vertical
        yaw = math.degrees(math.atan2(rot[1, 0], rot[0, 0]))  # heading of the trunk's x axis, wrapped
        row = [(k + 1) * CONTROL_DT, *command, v[0], v[1], data.qvel[5], data.qpos[2], tilt, yaw, *action]
        if ball:
            row += [data.qvel[ball[1]], data.qpos[ball[0]], data.qpos[ball[0] + 1]]
        for key, value in zip(keys, row):
            traces[key].append(float(value))
        rate.append(float(np.sqrt(np.mean((action - last_action) ** 2))))
        last_action = action
        if renderer:
            renderer.update_scene(data, camera)
            ffmpeg.stdin.write(renderer.render().tobytes())
        if recovery:  # up: trunk within 30 degrees of upright and off the floor, held for a second
            if tilt < UPRIGHT_DEG and data.qpos[2] > 0.09:
                upright_since = upright_since if upright_since is not None else (k + 1) * CONTROL_DT
                if up_at is None and (k + 1) * CONTROL_DT - upright_since >= 1.0:
                    up_at = upright_since
            else:
                upright_since = None
        elif tilt > FELL_DEG and fell_at is None:
            fell_at = (k + 1) * CONTROL_DT
            if level.task != "sitstand":  # a posture task plays the tip-over out: its training has no fall termination
                break
    if ffmpeg:
        ffmpeg.stdin.close()
        ffmpeg.wait()

    actions = np.array([[traces[f"a{i}"][j] for i in range(n)] for j in range(len(traces["t"]))])
    segments = (segment_metrics(traces) if level.task == "walk" else posture_segments(traces, schedule)
                if level.task == "sitstand" else [])
    moving = [s for s in segments if any(s["command"]) and "achieved" in s]
    result = {
        "level": level.key, "seed": seed, "fell_at_s": fell_at, "up_at_s": up_at,
        "final_tilt_deg": traces["tilt"][-1], "final_z": traces["z"][-1],
        "vin": p["vin"], "drop_gain": p["drop_gain"], "foot_friction": p["foot_friction"],
        "segments": segments,
        "err_vxy_mean": float(np.mean([s["err_vxy"] for s in moving])) if moving else None,
        "err_wz_mean": float(np.mean([s["err_wz"] for s in moving])) if moving else None,
        "self_contact_ticks": self_ticks / len(traces["t"]), "body_ground_ticks": ground_ticks / len(traces["t"]),
        "body_ground_by_body": {b: c / len(traces["t"]) for b, c in sorted(ground_by_body.items(), key=lambda kv: -kv[1])},
        "self_contact_by_pair": {p: c / len(traces["t"]) for p, c in sorted(self_by_pair.items(), key=lambda kv: -kv[1])},
        "joint_limit_ticks": limit_ticks / len(traces["t"]), "torque_saturated_steps": saturated_steps / steps,
        "peak_torque_nm": float(peak_torque.max()), "mean_power_w": float(np.mean(power)),
        "action_rms": float(np.sqrt(np.mean(actions ** 2))), "action_rate_rms": float(np.mean(rate)),
        "trunk_z_mean": float(np.mean(traces["z"])),
    }
    if level.task == "sitstand":
        result.update(posture_ok=sum(sgm["ok"] for sgm in segments), posture_total=len(segments))
    if ball:
        result.update(kick_summary(traces, ball[2], ball[3]))
    if record:
        result["traces"] = traces
    return result


def _fmt(v, f=".3f"):
    return "-" if v is None else format(v, f)


def _moving(values: list[float], n: int = 25) -> np.ndarray:
    """Mean over the last n control steps (0.5 s), to see through the gait's oscillation."""
    v = np.asarray(values)
    c = np.cumsum(np.insert(v, 0, 0.0))
    out = np.empty_like(v)
    for i in range(len(v)):
        lo = max(0, i + 1 - n)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


def read_traces(path: Path) -> dict:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {k: [float(r[k]) for r in rows] for k in rows[0]}


def write_report(out: Path, results: dict, provenance: dict) -> None:
    levels = {k: v for k, v in results.items() if k not in ("reference", "provenance")}
    task = next(iter(levels.values()))["level"].get("task", "walk")
    if task == "sitstand":
        return write_report_sitstand(out, levels, provenance)
    if task == "kick":
        return write_report_kick(out, levels, provenance)
    lines = [f"# HD-1910 {provenance.get('title', 'walk')}: sim2sim in MuJoCo C, away from the training world", "",
             "The exported ONNX that goes to the robot, run in plain MuJoCo on the CPU by `scripts/sim2sim.py`, which "
             "imports nothing from mjlab or the training code. The scene is a standalone MJCF of the robot description "
             "with the HD-1910 mass assumption ([mass_table.md](mass_table.md)); the servos are BAM's HD-1910 model "
             "driven by the script's own loop; observations and actions are built as the robot builds them.", "",
             f"- Policy: `{provenance['onnx']}` (sha256 `{provenance['onnx_sha256'][:16]}…`)",
             f"- Servo model: `1910_m6.json` (sha256 `{provenance['servo_sha256'][:16]}…`), firmware P {KP_FW:g}, "
             f"supply {VIN_RANGE[0]}–{VIN_RANGE[1]} V, floor {VIN_MIN} V",
             "- Command schedule, the same for every level: " + "; ".join(
                 f"{label} {d:g} s" for d, _, label in SCHEDULE) + f" ({DURATION:g} s). Segment means leave out the "
             f"first {SETTLE_S:g} s of each segment.", "",
             "## The ladder", "",
             "Each level adds one difference to the training world. Contacts are counted in control steps (20 ms) "
             "with at least one such contact; feet are whole bodies.", "",
             "| level | change | fell | planar speed error (m/s) | yaw rate error (rad/s) | self-contact steps | "
             "non-foot ground steps | torque at limit | peak torque (N·m) | action RMS | action rate RMS |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, r in levels.items():
        lv = r["level"]
        if lv.get("start", "standing") != "standing":
            continue  # getting-up levels have their own table
        if "episode" in r:
            e = r["episode"]
            fell = "no" if e["fell_at_s"] is None else f"at {e['fell_at_s']:.1f} s"
            lines.append(f"| {key} | {lv['title']} | {fell} | {_fmt(e['err_vxy_mean'])} | {_fmt(e['err_wz_mean'])} | "
                         f"{e['self_contact_ticks']:.1%} | {e['body_ground_ticks']:.1%} | {e['torque_saturated_steps']:.1%} | "
                         f"{e['peak_torque_nm']:.2f} | {e['action_rms']:.3f} | {e['action_rate_rms']:.3f} |")
        else:
            eps = r["episodes"]
            up = [e for e in eps if e["fell_at_s"] is None]
            mean = lambda k: float(np.mean([e[k] for e in up])) if up else None  # noqa: E731
            lines.append(f"| {key} | {lv['title']} | {len(eps) - len(up)} of {len(eps)} | {_fmt(mean('err_vxy_mean'))} | "
                         f"{_fmt(mean('err_wz_mean'))} | {_fmt(mean('self_contact_ticks'), '.1%')} | "
                         f"{_fmt(mean('body_ground_ticks'), '.1%')} | {_fmt(mean('torque_saturated_steps'), '.1%')} | "
                         f"{max(e['peak_torque_nm'] for e in eps):.2f} | {_fmt(mean('action_rms'))} | "
                         f"{_fmt(mean('action_rate_rms'))} |")
    lines += ["", "Randomized rows: falls out of all episodes; the other columns average the episodes that stayed up "
              "(peak torque over all)."]

    if "reference" in results and "L0" in levels:
        lines += ["", "## L0 against the training world", "",
                  "The same schedule in MJWarp with the same nominal settings (`scripts/sim2sim_reference.py`): "
                  "mean body-frame velocities per segment.", "",
                  "| segment | command (vx, vy, wz) | MJWarp | L0 (MuJoCo C) | L0 − MJWarp |", "|---|---|---|---|---|"]
        vec = lambda v: ", ".join(f"{x:+.3f}" for x in v)  # noqa: E731
        for ref, l0 in zip(results["reference"]["segments"], levels["L0"]["episode"]["segments"]):
            if "achieved" in ref and "achieved" in l0:
                lines.append(f"| {ref['segment']} | {vec(ref['command'])} | {vec(ref['achieved'])} | {vec(l0['achieved'])} | "
                             f"{vec([a - b for a, b in zip(l0['achieved'], ref['achieved'])])} |")
    for key in ("L3", "L3-backlash"):
        if key in levels and "episode" in levels[key]:
            lines += ["", f"## {key} per segment", "", "| segment | command (vx, vy, wz) | achieved | planar error | yaw error |",
                      "|---|---|---|---|---|"]
            for sgm in levels[key]["episode"]["segments"]:
                if "achieved" in sgm:
                    lines.append(f"| {sgm['segment']} | {', '.join(f'{x:+.2f}' for x in sgm['command'])} | "
                                 f"{', '.join(f'{x:+.3f}' for x in sgm['achieved'])} | {sgm['err_vxy']:.3f} | {sgm['err_wz']:.3f} |")
    contact_rows = []
    for key, r in levels.items():
        for e in ([r["episode"]] if "episode" in r else []):
            for b, f in e["body_ground_by_body"].items():
                contact_rows.append(f"| {key} | ground | {b} | {f:.1%} |")
            for pair, f in e["self_contact_by_pair"].items():
                contact_rows.append(f"| {key} | self | {pair} | {f:.1%} |")
    if contact_rows:
        lines += ["", "## Contacts the training world does not have", "", "| level | kind | bodies | control steps |",
                  "|---|---|---|---|"] + contact_rows
    for key, r in levels.items():
        if "episodes" in r:
            fallen = [e for e in r["episodes"] if e["fell_at_s"] is not None]
            if fallen:
                lines += ["", f"{key} falls: " + "; ".join(
                    f"seed {e['seed']} at {e['fell_at_s']:.1f} s (vin {e['vin']:.2f} V, foot friction "
                    f"{_fmt(e['foot_friction'], '.2f')})" for e in fallen) + "."]
    recovery = [(k, r) for k, r in levels.items() if r["level"].get("start", "standing") != "standing"]
    if recovery:
        lines += ["", "## Getting up", "",
                  f"Lying on the floor at the start, zero command for {RECOVERY_S:g} s, L3 settings. Up: trunk within "
                  f"{UPRIGHT_DEG:g} degrees of upright and above 9 cm for a second. Each try starts from slightly "
                  "different joint angles.", "",
                  "| start | got up | median time to get up (s) | final tilt, median (degrees) |", "|---|---|---|---|"]
        for key, r in recovery:
            eps = r["episodes"]
            ups = [e["up_at_s"] for e in eps if e["up_at_s"] is not None]
            lines.append(f"| {r['level']['title']} | {len(ups)} of {len(eps)} | "
                         f"{_fmt(float(np.median(ups)) if ups else None, '.1f')} | "
                         f"{np.median([e['final_tilt_deg'] for e in eps]):.0f} |")
    if any((out / f"traces_{key}.csv").exists() for key in ("L0", "L3", "L3-backlash")):  # a getting-up-only run has none
        lines += ["", "![velocities](velocity_traces.png)"]
    lines.append("")
    (out / "README.md").write_text("\n".join(lines))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = {}
    if (out / "reference_traces.csv").exists():
        series["MJWarp (training world)"] = read_traces(out / "reference_traces.csv")
    for key in ("L0", "L3", "L3-backlash"):
        if (out / f"traces_{key}.csv").exists():
            series[f"{key} (MuJoCo C)"] = read_traces(out / f"traces_{key}.csv")
    if not series:
        return
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    first = next(iter(series.values()))
    for ax, key, label in zip(axes, ("vx", "vy", "wz"), ("vx (m/s)", "vy (m/s)", "yaw rate (rad/s)")):
        ax.plot(first["t"], first[f"cmd_{key}"], "k--", lw=1, label="command")
        for name, tr in series.items():
            ax.plot(tr["t"], _moving(tr[key]), lw=1.2, label=name)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper right", fontsize="small", ncol=2)
    axes[-1].set_xlabel("time (s); 0.5 s moving mean")
    fig.tight_layout()
    fig.savefig(out / "velocity_traces.png", dpi=110)


def _settle_times(segments: list[dict], kind: str) -> list[float]:
    return [sgm["settled_s"] for sgm in segments if sgm["transition"] == kind and sgm["settled_s"] is not None]


def _median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def write_report_kick(out: Path, levels: dict, provenance: dict) -> None:
    lines = [f"# HD-1910 {provenance.get('title', 'kick')}: sim2sim in MuJoCo C, away from the training world", "",
             "The exported ONNX that goes to the robot, run in plain MuJoCo on the CPU by `scripts/sim2sim.py --task kick`, "
             "which imports nothing from mjlab or the training code. Same scene export, servo model and observation build "
             "as the walk check; the ball is the training's (15 g, radius 3.5 cm), placed in front of the kicking foot "
             f"(0.09 m ahead, {BALL_OFFSET_ABS_Y} m to the side, the kicking foot's side), the command is zero, the "
             "episode is the training's 5 s.", "",
             f"- Policy: `{provenance['onnx']}` (sha256 `{provenance['onnx_sha256'][:16]}…`)",
             f"- Servo model: `1910_m6.json` (sha256 `{provenance['servo_sha256'][:16]}…`), firmware P {KP_FW:g}, "
             f"supply {VIN_RANGE[0]}–{VIN_RANGE[1]} V, floor {VIN_MIN} V",
             f"- Kicked: the ball's speed along the kick direction (+x, the robot's heading at spawn) reaches "
             f"{KICK_MIN_SPEED:g} m/s at some moment. The training's target speed is 1 m/s. Fell: trunk tilt above "
             f"{FELL_DEG:g} degrees.",
             "- Ball angle: the direction of the ball's displacement over the episode; the table gives its distance from 0 = straight ahead.", "",
             "## The ladder", "",
             "Each level adds one difference to the training world (ground-contact model, the training's 5 ms / 10 / 20 "
             "solver, no sensors). The ball-placement levels draw the ball's position per episode.", "",
             "| level | change | fell | kicked | peak ball speed, median (min) m/s | ball angle off straight, median (p90) degrees | "
             "peak torque (N·m) | action RMS |", "|---|---|---|---|---|---|---|---|"]
    for key, r in levels.items():
        eps = [r["episode"]] if "episode" in r else r["episodes"]
        fell = [e for e in eps if e["fell_at_s"] is not None]
        fell_text = (f"at {fell[0]['fell_at_s']:.1f} s" if fell else "no") if len(eps) == 1 else f"{len(fell)} of {len(eps)}"
        peaks = [e["kick_peak_mps"] for e in eps]
        angles = [abs(e["ball_angle_deg"]) for e in eps if e["ball_angle_deg"] is not None]
        lines.append(f"| {key} | {r['level']['title']} | {fell_text} | {sum(e['kick_ok'] for e in eps)} of {len(eps)} | "
                     f"{np.median(peaks):.2f} ({min(peaks):.2f}) | "
                     f"{_fmt(float(np.median(angles)) if angles else None, '.0f')} "
                     f"({_fmt(float(np.percentile(angles, 90)) if angles else None, '.0f')}) | "
                     f"{max(e['peak_torque_nm'] for e in eps):.2f} | {float(np.mean([e['action_rms'] for e in eps])):.3f} |")
    lines += ["", "Randomized and ball-placement rows pool all episodes; peak torque is the largest of them."]
    for key, r in levels.items():
        if "episodes" in r:
            bad = [e for e in r["episodes"] if e["fell_at_s"] is not None or not e["kick_ok"]]
            if bad:
                lines += ["", f"{key} problems: " + "; ".join(
                    f"seed {e['seed']}" + (f" fell at {e['fell_at_s']:.1f} s" if e["fell_at_s"] is not None else "")
                    + ("" if e["kick_ok"] else f" ball peak {e['kick_peak_mps']:.2f} m/s") for e in bad[:20])
                    + (f" (+{len(bad) - 20} more)" if len(bad) > 20 else "") + "."]
    lines += ["", "![ball](ball_speed.png)", ""]
    (out / "README.md").write_text("\n".join(lines))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = {f"{key} (MuJoCo C)": read_traces(out / f"traces_{key}.csv") for key in ("L0", "L3", "L3-backlash")
              if (out / f"traces_{key}.csv").exists()}
    if not series:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    for name, tr in series.items():
        ax.plot(tr["t"], tr["ball_fwd"], lw=1.3, label=name)
    ax.axhline(KICK_MIN_SPEED, color="0.6", lw=0.8, ls=":")
    ax.set_xlabel("time (s); dotted: the speed that counts as a kick")
    ax.set_ylabel("ball speed along +x (m/s)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(out / "ball_speed.png", dpi=110)


def write_report_sitstand(out: Path, levels: dict, provenance: dict) -> None:
    def schedule_text(schedule: list) -> str:
        return "; ".join(f"{label} {d:g} s" for d, _, label in schedule) + f" ({sum(d for d, _, _ in schedule):g} s)"

    lines = ["# HD-1910 sitstand: sim2sim in MuJoCo C, away from the training world", "",
             "The exported ONNX that goes to the robot, run in plain MuJoCo on the CPU by `scripts/sim2sim.py --task "
             "sitstand`, which imports nothing from mjlab; the robotd levels share only the target arithmetic "
             "(`microduck_rl.actions.robotd_chain`) with the training task that uses it. Scene export, servo model and "
             "observations are the walk check's; the command is the posture flag the robot's runtime writes into the "
             "first command slot (0 = stand, 1 = sit).", "",
             f"- Policy: `{provenance['onnx']}` (sha256 `{provenance['onnx_sha256'][:16]}…`)",
             f"- Servo model: `1910_m6.json` (sha256 `{provenance['servo_sha256'][:16]}…`), firmware P {KP_FW:g}, "
             f"supply {VIN_RANGE[0]}–{VIN_RANGE[1]} V ({ROBOTD_VIN_RANGE[0]}–{ROBOTD_VIN_RANGE[1]} V on the robotd "
             f"levels), floor {VIN_MIN} V",
             "- robotd levels: the targets as robotd sends them, home + (7.4 V / battery) × action, low-passed "
             "0.5 on the head and 0.7 on the legs against the previous target; the policy still sees its raw action.",
             f"- Command schedule from standing: {schedule_text(SIT_FROM_STAND)}; from the seated start: "
             f"{schedule_text(SIT_FROM_SEAT)}.",
             f"- A posture is reached when the trunk is at most {SIT_Z_MAX * 100:g} cm high for a sit, at least "
             f"{STAND_Z_MIN * 100:g} cm for a stand, and within {UPRIGHT_DEG:g} degrees of upright, from some moment to "
             "the end of the segment. Settle time counts from the start of the segment. The heights are the training's "
             "own (SIT_UPRIGHT_Z, STAND_UPRIGHT_Z).",
             f"- Tipped over: trunk tilt above {FELL_DEG:g} degrees at any moment. The training has no fall "
             "termination, so the episode plays on.", "",
             "## The ladder", "",
             "Each level adds one difference to the training world: the ground-contact model (feet plus the trunk, hip, "
             "shank and jaw geoms the model authors), the training's 5 ms / 30 / 50 solver, no sensors. A sit-down is "
             "a sit after a stand, a stand-up a stand after a sit.", "",
             "| level | change | tipped over | postures reached | sit-down settle, median (s) | stand-up settle, "
             "median (s) | self-contact steps | torque at limit | peak torque (N·m) | action RMS | action rate RMS |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, r in levels.items():
        eps = [r["episode"]] if "episode" in r else r["episodes"]
        tipped = [e for e in eps if e["fell_at_s"] is not None]
        if len(eps) == 1:
            tip = f"at {tipped[0]['fell_at_s']:.1f} s" if tipped else "no"
        else:
            tip = f"{len(tipped)} of {len(eps)}"
        segs = [sgm for e in eps for sgm in e["segments"]]
        mean = lambda k: float(np.mean([e[k] for e in eps]))  # noqa: E731
        lines.append(f"| {key} | {r['level']['title']} | {tip} | {sum(e['posture_ok'] for e in eps)} of "
                     f"{sum(e['posture_total'] for e in eps)} | {_fmt(_median(_settle_times(segs, 'sit-down')), '.1f')} | "
                     f"{_fmt(_median(_settle_times(segs, 'stand-up')), '.1f')} | {mean('self_contact_ticks'):.1%} | "
                     f"{mean('torque_saturated_steps'):.1%} | {max(e['peak_torque_nm'] for e in eps):.2f} | "
                     f"{mean('action_rms'):.3f} | {mean('action_rate_rms'):.3f} |")
    lines += ["", "Randomized, push and seated-start rows pool all episodes; peak torque is the largest of them."]

    for key in ("L0", "L3", "L3-backlash"):
        if key in levels and "episode" in levels[key]:
            lines += ["", f"## {key} per segment", "",
                      "| segment | command | trunk height at the end (cm) | tilt at the end (degrees) | drift at the end "
                      "(cm/s) | settled after (s) |", "|---|---|---|---|---|---|"]
            for sgm in levels[key]["episode"]["segments"]:
                if "z_end" in sgm:
                    lines.append(f"| {sgm['segment']} | {sgm['command'][0]:g} | {100 * sgm['z_end']:.1f} | "
                                 f"{sgm['tilt_end']:.1f} | {100 * sgm['drift_end']:.2f} | {_fmt(sgm['settled_s'], '.1f')} |")
    if "L3-seated" in levels:
        eps = levels["L3-seated"]["episodes"]
        lines += ["", "## Starting seated", "",
                  f"{len(eps)} tries from the training's seated spawn, one per seed. The schedule is "
                  f"{schedule_text(SIT_FROM_SEAT)}.", "",
                  "| segment | reached | settle, median (s) | trunk height at the end, median (cm) |",
                  "|---|---|---|---|"]
        for i, (_, cmd, label) in enumerate(SIT_FROM_SEAT):
            col = [e["segments"][i] for e in eps]
            lines.append(f"| {label} ({cmd[0]:g}) | {sum(c['ok'] for c in col)} of {len(col)} | "
                         f"{_fmt(_median([c['settled_s'] for c in col if c['settled_s'] is not None]), '.1f')} | "
                         f"{100 * float(np.median([c['z_end'] for c in col if 'z_end' in c])):.1f} |")
    lines += ["", "## Heading per transition", "",
              "Net trunk turn over each sit-down and stand-up segment, in degrees, counter-clockwise positive, pooled "
              "over the level's episodes (the battery, and on the seated and randomized levels the start, differ by "
              "episode).", "",
              "| level | battery (V) | sit-down: n, min / median / max, mean abs | stand-up: n, min / median / max, "
              "mean abs |", "|---|---|---|---|"]
    for key, r in levels.items():
        eps = [r["episode"]] if "episode" in r else r["episodes"]
        vins = [e["vin"] for e in eps]
        cells = []
        for kind in ("sit-down", "stand-up"):
            turns = [s["yaw_change_deg"] for e in eps for s in e["segments"]
                     if s["transition"] == kind and "yaw_change_deg" in s]
            cells.append(f"{len(turns)}, {min(turns):+.1f} / {float(np.median(turns)):+.1f} / {max(turns):+.1f}, "
                         f"{float(np.mean(np.abs(turns))):.1f}" if turns else "-")
        lines.append(f"| {key} | {min(vins):.2f}–{max(vins):.2f} | {cells[0]} | {cells[1]} |")
    for key, r in levels.items():
        if "episodes" in r:
            bad = [e for e in r["episodes"] if e["fell_at_s"] is not None or e["posture_ok"] < e["posture_total"]]
            if bad:
                lines += ["", f"{key} problems: " + "; ".join(
                    f"seed {e['seed']}" + (f" tipped over at {e['fell_at_s']:.1f} s" if e["fell_at_s"] is not None else "")
                    + f" ({e['posture_ok']} of {e['posture_total']} postures, vin {e['vin']:.2f} V)" for e in bad) + "."]
    lines += ["", "![posture](posture_traces.png)", ""]
    (out / "README.md").write_text("\n".join(lines))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = {f"{key} (MuJoCo C)": read_traces(out / f"traces_{key}.csv") for key in ("L0", "L3", "L3-backlash")
              if (out / f"traces_{key}.csv").exists()}
    if not series:
        return
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    first = next(iter(series.values()))
    target = [SIT_Z if c > 0.5 else STAND_Z for c in first["cmd_vx"]]
    axes[0].plot(first["t"], target, "k--", lw=1, label="training rest height for the command")
    axes[0].axhline(SIT_Z_MAX, color="0.6", lw=0.8, ls=":")
    axes[0].axhline(STAND_Z_MIN, color="0.6", lw=0.8, ls=":")
    for name, tr in series.items():
        axes[0].plot(tr["t"], tr["z"], lw=1.2, label=name)
        axes[1].plot(tr["t"], tr["tilt"], lw=1.2, label=name)
    axes[0].set_ylabel("trunk height (m)")
    axes[1].set_ylabel("trunk tilt (degrees)")
    axes[1].axhline(UPRIGHT_DEG, color="0.6", lw=0.8, ls=":")
    axes[0].legend(loc="center right", fontsize="small")
    axes[-1].set_xlabel("time (s); dotted: the reach thresholds")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "posture_traces.png", dpi=110)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--assets", type=Path, required=True, help="scenes from sim2sim_assets.py")
    ap.add_argument("--servo-params", type=Path, default=SERVO_PARAMS)
    ap.add_argument("--task", choices=sorted(LEVEL_SETS), default="walk")
    ap.add_argument("--levels", nargs="+", help="default: every level of the task")
    ap.add_argument("--episodes", type=int, help="override the randomized levels' episode count")
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    ap.add_argument("--video", help="level to record a video of, e.g. L3")
    ap.add_argument("--reference", type=Path, help="reference_traces.csv from sim2sim_reference.py")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--title", help="the skill's name in the report's title; default: the task")
    ap.add_argument("--foot", choices=("right", "left"), default="right", help="the kick task's kicking foot")
    ap.add_argument("--report-only", action="store_true", help="rewrite the report from metrics.json")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.reference and args.task != "walk":
        ap.error("--reference is for the walk task")
    levels = {lv.key: replace(lv, kick_foot=args.foot) if lv.task == "kick" else lv for lv in LEVEL_SETS[args.task]}
    provenance = {"onnx": args.onnx.name, "onnx_sha256": hashlib.sha256(args.onnx.read_bytes()).hexdigest(),
                  "servo_sha256": hashlib.sha256(args.servo_params.read_bytes()).hexdigest(), "task": args.task,
                  "title": args.title or args.task}
    if args.report_only:
        write_report(args.out_dir, json.loads((args.out_dir / "metrics.json").read_text()), provenance)
        return

    results = {}
    for key in args.levels or levels:
        level = levels[key]
        if args.episodes and level.randomize:
            level = replace(level, episodes=args.episodes)
        if level.episodes == 1:
            video = args.out_dir / f"sim2sim_{key}.mp4" if args.video == key else None
            r = run_episode(level, 0, args.onnx, args.assets, args.servo_params, record=True, video=video)
            traces = r.pop("traces")
            with open(args.out_dir / f"traces_{key}.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(traces.keys())
                w.writerows([f"{v:.6g}" for v in row] for row in zip(*traces.values()))
            results[key] = {"level": asdict(level), "episode": r}
        else:
            with ProcessPoolExecutor(args.workers) as pool:
                eps = list(pool.map(run_episode, [level] * level.episodes, range(level.episodes),
                                    [args.onnx] * level.episodes, [args.assets] * level.episodes,
                                    [args.servo_params] * level.episodes))
            results[key] = {"level": asdict(level), "episodes": eps}
        summary = results[key].get("episode") or {
            "fell": sum(e["fell_at_s"] is not None for e in results[key]["episodes"]),
            **({"postures_reached": sum(e["posture_ok"] for e in results[key]["episodes"])}
               if level.task == "sitstand" else {})}
        print(f"{key:<12} {level.title:<36} " + json.dumps(
            {k: (round(v, 4) if isinstance(v, float) else v) for k, v in summary.items() if k != "segments"}), flush=True)

    if args.reference:
        with open(args.reference) as f:
            rows = list(csv.DictReader(f))
        ref = {k: [float(r[k]) for r in rows] for k in rows[0]}
        results["reference"] = {"segments": segment_metrics(ref), "fell_at_s": None if max(ref["tilt"]) <= FELL_DEG
                                else next(t for t, a in zip(ref["t"], ref["tilt"]) if a > FELL_DEG)}
    results["provenance"] = provenance
    (args.out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    write_report(args.out_dir, {k: v for k, v in results.items() if k != "provenance"}, provenance)


if __name__ == "__main__":
    main()
