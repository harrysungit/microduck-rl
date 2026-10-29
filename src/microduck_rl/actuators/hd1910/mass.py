"""HD-1910 mass model: heavier servos, heavier build.

The MJCF bodies carry the XL330 build's masses: 18 g per servo housing,
0.737 kg in all. An HD-1910 weighs 21 g, so each of the 15 housings gains 3 g
at its centroid, and its body's mass, centre of mass and inertia follow by the
parallel-axis theorem. The rest of the gap to the built robot's weight
(electronics, wiring, heavier printed parts) goes into trunk_base, with its
inertia scaled alike so the centre of mass and radii of gyration stay put.

TOTAL_MASS_KG is the weight reported for an HD-1910 build with the same
electronics; replace it with our own scale reading once the duck is assembled.
Editing the spec at load time leaves the MJCF and meshes untouched; the idea
follows qiaosiyi/microduck_rl_hd1910 (Apache-2.0).
"""

import mujoco
import numpy as np

SERVO_MESH = "xl330"
SERVO_HOUSINGS = 15
SERVO_DELTA_KG = 0.021 - 0.018
TOTAL_MASS_KG = 0.82
PAYLOAD_BODY = "trunk_base"


def _matrix(six) -> np.ndarray:
    """MuJoCo fullinertia (ixx, iyy, izz, ixy, ixz, iyz) as a 3x3 matrix."""
    xx, yy, zz, xy, xz, yz = six
    return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])


def _point(mass: float, r: np.ndarray) -> np.ndarray:
    """Inertia of a point mass at r about the origin."""
    return mass * (r @ r * np.eye(3) - np.outer(r, r))


def servo_housings(spec: mujoco.MjSpec) -> dict[str, list[np.ndarray]]:
    """Body name -> centroids of its XL330 housing meshes, in the body frame."""
    # Compiling moves each mesh geom to its mesh's centroid.
    model = spec.compile()
    mesh = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, SERVO_MESH)
    housings: dict[str, list[np.ndarray]] = {}
    for g in range(model.ngeom):
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or model.geom_dataid[g] != mesh:
            continue
        if model.geom_contype[g] or model.geom_conaffinity[g]:
            continue  # count each housing once, by its visual geom
        body = model.body(model.geom_bodyid[g]).name
        housings.setdefault(body, []).append(model.geom_pos[g].copy())
    return housings


def apply_hd1910_mass(
    spec: mujoco.MjSpec,
    servo_delta_kg: float = SERVO_DELTA_KG,
    total_mass_kg: float = TOTAL_MASS_KG,
) -> mujoco.MjSpec:
    housings = servo_housings(spec)
    found = sum(len(points) for points in housings.values())
    if found != SERVO_HOUSINGS:
        raise ValueError(f"expected {SERVO_HOUSINGS} {SERVO_MESH} housings, found {found}")

    for name, points in housings.items():
        body = spec.body(name)
        mass, com = body.mass, np.asarray(body.ipos, dtype=float)
        new_mass = mass + servo_delta_kg * len(points)
        new_com = (mass * com + servo_delta_kg * np.sum(points, axis=0)) / new_mass
        inertia = _matrix(body.fullinertia) + _point(mass, com - new_com)
        for p in points:
            inertia += _point(servo_delta_kg, p - new_com)
        body.mass, body.ipos = new_mass, new_com
        body.fullinertia = [inertia[0, 0], inertia[1, 1], inertia[2, 2], inertia[0, 1], inertia[0, 2], inertia[1, 2]]

    payload = total_mass_kg - sum(body.mass for body in spec.bodies)
    if payload < -1e-9:
        raise ValueError(f"total mass {total_mass_kg} kg is below the servo-corrected model's")
    payload = max(payload, 0.0)
    body = spec.body(PAYLOAD_BODY)
    scale = (body.mass + payload) / body.mass
    body.mass *= scale
    body.fullinertia = [v * scale for v in body.fullinertia]
    return spec
