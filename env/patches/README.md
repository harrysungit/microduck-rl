# env/patches · local fixes to locked third-party packages

`env/setup.sh` applies these to the venv's site-packages after the locked
dependencies are installed, skipping any already applied.

- **Naming**: `<package>-<version>/NNNN-<what>.patch`.
- **Paths**: relative to site-packages.
- **On a version bump**: check each patch is still needed.

| Patch | Why | Upstream |
|---|---|---|
| `mujoco_warp-3.8.1/0001-sensor-frame-axis-unknown-objtype.patch` | The UNKNOWN branch of `_frame_axis` uses an undefined `xmat`. Warp 1.12 compiles it; Warp 1.17 (the HIP build) rejects it, so the sensor module fails to load | Rewritten in 3.10 with `_get_mat`, returning the identity for unknown types; this patch does the same |
| `mujoco_warp-3.8.1/0002-sparse-qfrc-constraint-converged-worlds.patch` | The sparse path (nv > 32, e.g. a model with backlash) zeroes `qfrc_constraint` for every world but recomputes it only for worlds still iterating, so converged worlds keep 0. The physics is unaffected, but BAM's load-dependent friction reads that field: contact and limit loads are lost and the servo's own friction is subtracted twice | Upstream main zeroes per world (`_zero_qfrc_constraint_sparse`); this patch likewise zeroes only the worlds that have not converged |
