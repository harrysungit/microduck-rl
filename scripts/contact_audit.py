"""Head/neck contact latch and exact geometry-to-horizontal-floor clearance on CPU MuJoCo."""
import math
import mujoco
import numpy as np
from scipy.spatial import ConvexHull, QhullError

HEAD_BODIES = ("neck", "neck_pitch", "yaw_roll_motion", "jaw_soft")
LEGACY_HIP_BODY = "bearing_roll"  # right hip yaw, historically mislisted in upstream DR head names
# Diagnostic load threshold, not a hardware safety limit. CAD bodies: lower legs
# exist in both scenes; upper legs and hips additionally collide in allcollisions.
SUPPORT_NORMAL_THRESHOLD_N = 0.05
LEFT_SIT_BODIES = frozenset(("leg", "upper_leg_left", "hip_l"))
RIGHT_SIT_BODIES = frozenset(("leg_2", "upper_leg_right", "hip_l_2", LEGACY_HIP_BODY))


def neck_subtree(model, prefix=""):
    """Resolve actual anatomy from parent links; the historical DR list is not anatomy."""
    root = model.body(prefix + "neck").id
    descendants = set()
    for body in range(1, model.nbody):
        ancestor = body
        while ancestor and ancestor != root:
            ancestor = int(model.body_parentid[ancestor])
        if ancestor == root:
            descendants.add(body)
    for name in HEAD_BODIES:
        if model.body(prefix + name).id not in descendants:
            raise ValueError(f"Required head body {name} is outside the neck subtree")
    legacy = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + LEGACY_HIP_BODY)
    if legacy in descendants:
        raise ValueError("Legacy right-hip body unexpectedly occurs in the neck subtree")
    return descendants


def legal_sit_support(normal_by_body, threshold=SUPPORT_NORMAL_THRESHOLD_N):
    """Require a loaded torso or loaded leg/hip support on both sides; feet never qualify."""
    loaded = {name for name, force in normal_by_body.items() if np.isfinite(force) and force > threshold}
    return bool("trunk_base" in loaded or (loaded & LEFT_SIT_BODIES and loaded & RIGHT_SIT_BODIES))


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
    def __init__(self, model, geometry=None, body_names=None):
        self.model, self.geometry = model, geometry or Geometry(model)
        self.dynamic_scope = body_names is None
        self.bodies = neck_subtree(model) if body_names is None else {model.body(n).id for n in body_names}
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
        self.support_samples = []
        self.last_support_normal_N = {}
        self.last_legal_sit_support = False
        self.legacy_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, LEGACY_HIP_BODY)
        if self.legacy_body in self.bodies:
            raise ValueError("The right hip bearing_roll cannot be selected as anatomical head")
        self.legacy_geoms = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == self.legacy_body]
        self.legacy_gap = math.inf
        self.legacy_gap_time = self.legacy_gap_geom = None
        self.legacy_hits = {}
        self.legacy_first = None
        self.legacy_contact_steps = self.superset_contact_steps = 0
        self.legacy_max_force = self.legacy_impulse = 0.

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
        if self.legacy_geoms:
            old_gap, old_geom = min((self.geometry.minimum_z(data, g) - floor_z, g) for g in self.legacy_geoms)
            if not np.isfinite(old_gap):
                raise ValueError("Non-finite legacy right-hip geometry")
            if old_gap < self.legacy_gap:
                self.legacy_gap, self.legacy_gap_time = old_gap, float(time_s)
                self.legacy_gap_geom = self.geometry.label(old_geom)
        active, total_normal = False, 0.0
        legacy_active, legacy_normal = False, 0.
        support_normal = {}
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
                name = self.model.body(body).name
                self.support.add(name)
                support_normal[name] = support_normal.get(name, 0.0) + max(0.0, float(force[0]))
                if body == self.legacy_body:
                    legacy_active = True
                    label = self.geometry.label(g)
                    hit = self.legacy_hits.setdefault(label, {"first_s": float(time_s), "last_s": float(time_s),
                                                              "peak_single_contact_normal_N": 0.0})
                    hit["last_s"] = float(time_s)
                    hit["peak_single_contact_normal_N"] = max(hit["peak_single_contact_normal_N"], float(force[0]))
                    legacy_normal += max(0., float(force[0]))
                    self.legacy_max_force = max(self.legacy_max_force, float(force[0]))
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
        if legacy_active and self.legacy_first is None:
            self.legacy_first = float(time_s)
        self.last_support_normal_N = support_normal
        self.last_legal_sit_support = legal_sit_support(support_normal)
        if dt > 0:
            self.steps += 1
            self.contact_steps += int(active)
            self.impulse += total_normal * dt
            self.legacy_contact_steps += int(legacy_active)
            self.superset_contact_steps += int(active or legacy_active)
            self.legacy_impulse += legacy_normal * dt
            loaded = {name: force for name, force in support_normal.items() if force > SUPPORT_NORMAL_THRESHOLD_N}
            self.support_samples.append([float(time_s), loaded, self.last_legal_sit_support])

    def result(self):
        legacy = {"body": LEGACY_HIP_BODY, "anatomy": "right hip yaw, not head", "present": bool(self.legacy_geoms),
                  "parent_body": self.model.body(int(self.model.body_parentid[self.legacy_body])).name if self.legacy_body >= 0 else None,
                  "contact": bool(self.legacy_hits), "first_contact_s": self.legacy_first,
                  "contact_substeps": self.legacy_contact_steps,
                  "min_clearance_m": self.legacy_gap if self.legacy_geoms else None,
                  "min_clearance_s": self.legacy_gap_time, "min_clearance_geometry": self.legacy_gap_geom,
                  "peak_single_contact_normal_N": self.legacy_max_force,
                  "summed_normal_impulse_Ns": self.legacy_impulse, "contact_geometries": self.legacy_hits}
        use_legacy = self.legacy_gap < self.min_gap
        first = [t for t in (self.first_contact, self.legacy_first) if t is not None]
        superset = {"scope": "historical head-labelled superset: neck subtree plus right hip bearing_roll; not anatomical head",
                    "contact": bool(self.hits or self.legacy_hits), "first_contact_s": min(first) if first else None,
                    "contact_substeps": self.superset_contact_steps,
                    "min_clearance_m": min(self.min_gap, self.legacy_gap),
                    "min_clearance_s": self.legacy_gap_time if use_legacy else self.min_gap_time,
                    "min_clearance_geometry": self.legacy_gap_geom if use_legacy else self.min_gap_geom,
                    "peak_single_contact_normal_N": max(self.max_force, self.legacy_max_force),
                    "summed_normal_impulse_Ns": self.impulse + self.legacy_impulse,
                    "contact_geometries": {**self.hits, **self.legacy_hits}}
        return {"head_scope_version": "neck-subtree-v2", "head_contact": bool(self.hits), "first_contact_s": self.first_contact,
                "head_contact_substeps": self.contact_steps, "physics_substeps": self.steps,
                "min_clearance_m": self.min_gap, "min_clearance_s": self.min_gap_time,
                "min_clearance_geometry": self.min_gap_geom,
                "peak_single_contact_normal_N": self.max_force,
                "summed_normal_impulse_Ns": self.impulse, "contact_geometries": self.hits,
                "non_head_ground_bodies": sorted(self.support),
                "support_normal_threshold_N": SUPPORT_NORMAL_THRESHOLD_N,
                "legal_sit_support_rule": "trunk_base or both left and right leg/upper_leg/hip bodies loaded; feet excluded",
                "support_substep_trace_fields": ["solved_state_time_s", "non_head_body_normal_N_above_threshold", "legal_sit_support"],
                "support_substep_trace": self.support_samples,
                "clearance_scope": "all mesh/primitive geometry on dynamically resolved neck subtree, including non-colliding geometry",
                "head_scope_source": "parent-link neck subtree" if self.dynamic_scope else "explicit anatomical test/custom scope",
                "legacy_right_hip": legacy, "legacy_superset": superset,
                "head_bodies": sorted(self.model.body(b).name for b in self.bodies),
                "head_geometry_count": len(self.geoms),
                "units_note": "Simulation contact force/impulse, not measured hardware impact."}
