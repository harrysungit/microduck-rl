"""Head/neck contact latch and exact geometry-to-horizontal-floor clearance on CPU MuJoCo."""
import math
import mujoco
import numpy as np
from scipy.spatial import ConvexHull, QhullError

HEAD_BODIES = ("neck", "neck_pitch", "yaw_roll_motion", "jaw_soft", "bearing_roll")


class Geometry:
    def __init__(self, model):
        self.model = model
        self.vertices = {}
        for mesh in range(model.nmesh):
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            points = model.mesh_vert[start:start + count].astype(float)
            try:
                points = points[ConvexHull(points).vertices]
            except QhullError:
                pass  # planar/degenerate mesh: original vertices still give an exact minimum
            self.vertices[mesh] = points

    def minimum_z(self, data, geom):
        model = self.model
        kind, size = model.geom_type[geom], model.geom_size[geom]
        normal = data.geom_xmat[geom].reshape(3, 3)[2]
        center = float(data.geom_xpos[geom, 2])
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            return center + float((self.vertices[int(model.geom_dataid[geom])] @ normal).min())
        if kind == mujoco.mjtGeom.mjGEOM_BOX:
            return center - float(abs(normal) @ size)
        if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            return center - float(size[0])
        if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            return center - float(size[0] + size[1] * abs(normal[2]))
        if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
            return center - float(size[0] * np.linalg.norm(normal[:2]) + size[1] * abs(normal[2]))
        if kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            return center - float(np.linalg.norm(size * normal))
        raise ValueError(f"Unsupported finite geometry {geom}: type={kind}")

    def label(self, geom):
        model = self.model
        mesh = int(model.geom_dataid[geom])
        name = model.geom(geom).name or (model.mesh(mesh).name if mesh >= 0 else f"geom_{geom}")
        return f"{model.body(int(model.geom_bodyid[geom])).name}/{name}"


class HeadAudit:
    def __init__(self, model, geometry=None, body_names=HEAD_BODIES):
        self.model, self.geometry = model, geometry or Geometry(model)
        self.bodies = {model.body(n).id for n in body_names}  # missing body is an error, never an implicit pass
        self.geoms = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in self.bodies]
        if not self.geoms:
            raise ValueError("No head/neck geometry was selected")
        self.floor = model.geom("floor").id
        if model.geom_type[self.floor] != mujoco.mjtGeom.mjGEOM_PLANE:
            raise ValueError("Clearance audit requires a horizontal plane named floor")
        self.steps = 0
        self.contact_steps = 0
        self.min_gap = math.inf
        self.min_gap_time = self.min_gap_geom = None
        self.first_contact = None
        self.max_force = self.impulse = 0.0
        self.hits = {}
        self.support = set()

    def sample(self, data, time_s, dt):
        """Sample solved contacts and matching kinematics; call after each mj_step with time_s=t_before_step.

        mj_step integrates qpos last; its geom_xpos/contact/constraint force refer to the preceding solved state.
        dt=0 can inspect the initial forward state without adding impulse or a physics step.
        """
        if not np.allclose(data.geom_xmat[self.floor].reshape(3, 3)[2], (0, 0, 1), atol=1e-8):
            raise ValueError("Tilted floors require a different clearance calculation")
        floor_z = float(data.geom_xpos[self.floor, 2])
        gap, geom = min((self.geometry.minimum_z(data, g) - floor_z, g) for g in self.geoms)
        if not np.isfinite(gap):
            raise ValueError("Non-finite head geometry")
        if gap < self.min_gap:
            self.min_gap, self.min_gap_time, self.min_gap_geom = gap, float(time_s), self.geometry.label(geom)
        active, total_normal = False, 0.0
        for c in range(data.ncon):
            contact = data.contact[c]
            a, b = int(contact.geom1), int(contact.geom2)
            if self.floor not in (a, b):
                continue
            g = b if a == self.floor else a
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, data, c, force)
            if contact.dist > 0 and force[0] <= 1e-6:
                continue  # margin-only contact is not a physical hit
            body = int(self.model.geom_bodyid[g])
            if body not in self.bodies:
                self.support.add(self.model.body(body).name)
                continue
            active = True
            label = self.geometry.label(g)
            hit = self.hits.setdefault(label, {"first_s": float(time_s), "last_s": float(time_s),
                                               "peak_single_contact_normal_N": 0.0})
            hit["last_s"] = float(time_s)
            hit["peak_single_contact_normal_N"] = max(hit["peak_single_contact_normal_N"], float(force[0]))
            total_normal += max(0.0, float(force[0]))
            self.max_force = max(self.max_force, float(force[0]))
        if active and self.first_contact is None:
            self.first_contact = float(time_s)
        if dt > 0:
            self.steps += 1
            self.contact_steps += int(active)
            self.impulse += total_normal * dt

    def result(self):
        return {"head_contact": bool(self.hits), "first_contact_s": self.first_contact,
                "head_contact_substeps": self.contact_steps, "physics_substeps": self.steps,
                "min_clearance_m": self.min_gap, "min_clearance_s": self.min_gap_time,
                "min_clearance_geometry": self.min_gap_geom,
                "peak_single_contact_normal_N": self.max_force,
                "summed_normal_impulse_Ns": self.impulse, "contact_geometries": self.hits,
                "non_head_ground_bodies": sorted(self.support),
                "clearance_scope": "all geometries on named head/neck bodies, including non-colliding geometry",
                "head_bodies": sorted(self.model.body(b).name for b in self.bodies),
                "head_geometry_count": len(self.geoms),
                "units_note": "Simulation contact force/impulse, not measured hardware impact."}
