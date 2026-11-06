"""Export standalone MuJoCo scenes of the HD-1910 duck for the sim2sim runner.

The sim2sim runner (sim2sim.py) must not touch the training world, so its
scenes are plain MJCF files built here from the robot description alone: the
fork's robot MJCF (geometry, joints, meshes, sensors) with the HD-1910 mass
assumption applied (hd1910/mass.py: 21 g servos, a 0.82 kg build), a floor and
a light. The XL330 position actuators stay as authored; the runner turns them
into torque motors for its own servo model. Also writes a mass table.

    python scripts/sim2sim_assets.py --out-dir logs/sim2sim/assets --table <dir>/mass_table
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

import mjlab_microduck
from microduck_rl.actuators.hd1910.mass import apply_hd1910_mass

ROBOT_DIR = Path(mjlab_microduck.__file__).parent / "robot" / "microduck"
ROBOTS = {
    "walk": "robot_walk.xml",  # the training model: feet are its only contacts in training
    "allcollisions": "robot_allcollisions.xml",  # every part collides
    "allcollisions_backlash": "robot_allcollisions_backlash.xml",  # and +-1 degree play per servo
    "groundcontact": "robot_groundcontact.xml",  # the training model of StandUp, SitStand, GroundPick, BallKick, Roulade
    "groundcontact_backlash": "robot_groundcontact_backlash.xml",  # and +-1 degree play per servo
}
SCENE = """<mujoco model="hd1910_{name}">
  <include file="robot_hd1910_{name}.xml"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <global azimuth="160" elevation="-20" offwidth="1280" offheight="720"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>
  <worldbody>
    <light pos="0 0 3.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
{extra}  </worldbody>
</mujoco>
"""


# The kick tasks' ball (robot/microduck/ball.xml): 70 mm, 15 g, hollow, low rolling resistance; its body, as authored.
BALL_ROBOTS = ("groundcontact", "groundcontact_backlash", "allcollisions", "allcollisions_backlash")
BALL = (ROBOT_DIR / "ball.xml").read_text().split("<worldbody>")[1].split("</worldbody>")[0].strip("\n") + "\n"
SCENE_BALL = SCENE.replace('model="hd1910_{name}"', 'model="hd1910_{name}_ball"')


def body_table(model: mujoco.MjModel) -> dict[str, dict]:
    return {model.body(b).name: {"mass_kg": float(model.body_mass[b]), "com_m": model.body_ipos[b].round(6).tolist()}
            for b in range(1, model.nbody)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--table", type=Path, help="write <table>.md and <table>.json")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tables = {}
    for name, robot in ROBOTS.items():
        base = mujoco.MjSpec.from_file(str(ROBOT_DIR / robot))
        spec = apply_hd1910_mass(mujoco.MjSpec.from_file(str(ROBOT_DIR / robot)))
        spec.meshdir = str((ROBOT_DIR / "assets").resolve())
        (args.out_dir / f"robot_hd1910_{name}.xml").write_text(spec.to_xml())
        (args.out_dir / f"scene_hd1910_{name}.xml").write_text(SCENE.format(name=name, extra=""))
        if name in BALL_ROBOTS:
            (args.out_dir / f"scene_hd1910_{name}_ball.xml").write_text(SCENE_BALL.format(name=name, extra=BALL))
        model = mujoco.MjModel.from_xml_path(str(args.out_dir / f"scene_hd1910_{name}.xml"))
        colliding = int(np.sum((model.geom_contype != 0) | (model.geom_conaffinity != 0))) - 1  # minus the floor
        print(f"{name}: total mass {model.body_mass.sum():.6f} kg, nq {model.nq}, nv {model.nv}, nu {model.nu}, "
              f"colliding robot geoms {colliding}")
        tables[name] = {"before": body_table(base.compile()), "after": body_table(model)}

    if args.table:
        t = tables["walk"]
        rows = ["# HD-1910 mass model of the sim2sim scenes", "",
                "Robot MJCF with the HD-1910 mass assumption (21 g per servo housing instead of 18 g, the rest of a "
                "0.82 kg build in trunk_base); the three scenes share it.", "",
                "| body | XL330 model (g) | HD-1910 (g) | change (g) | COM before (mm) | COM after (mm) |",
                "|---|---|---|---|---|---|"]
        for body, before in t["before"].items():
            after = t["after"][body]
            mm = lambda v: ", ".join(f"{1e3 * x:.1f}" for x in v)  # noqa: E731
            rows.append(f"| {body} | {1e3 * before['mass_kg']:.1f} | {1e3 * after['mass_kg']:.1f} | "
                        f"{1e3 * (after['mass_kg'] - before['mass_kg']):+.1f} | {mm(before['com_m'])} | {mm(after['com_m'])} |")
        total_b = sum(v["mass_kg"] for v in t["before"].values())
        total_a = sum(v["mass_kg"] for v in t["after"].values())
        rows += [f"| **total** | {1e3 * total_b:.1f} | {1e3 * total_a:.1f} | {1e3 * (total_a - total_b):+.1f} | | |", ""]
        args.table.parent.mkdir(parents=True, exist_ok=True)
        args.table.with_suffix(".md").write_text("\n".join(rows))
        args.table.with_suffix(".json").write_text(json.dumps(tables, indent=2))


if __name__ == "__main__":
    main()
