"""Throughput fixes for mjlab at thousands of worlds; none of them is specific to ROCm.

Applied from ``microduck_rl.tasks`` next to the ROCm adjustments.

- mjlab's terrain adds one marker site per environment origin to the shared
  model (``env_origin_<i>``, debug-visualisation group 4). Every world then
  carries every environment's marker, so kinematics does num_envs**2 site
  transforms: at 4096 worlds, 16.8 M per call and about 0.8 GB of site arrays,
  which cost 3.5 ms in each of the five kinematics passes of an environment
  step on gfx1151. Nothing reads these sites, so they are skipped;
  MICRODUCK_ENV_ORIGIN_SITES=1 keeps them.
- mujoco_warp drops constraint rows beyond ``njmax`` and contacts beyond its
  contact budget without a warning. mjlab's velocity default of 1500 rows per
  world sizes the solver's scratch buffers; the Microduck walk tasks use about
  50, so lowering ``--env.sim.njmax`` pays off, but only with a guard. After
  every physics step the largest per-world row count and the contact count are
  folded into a device-side peak, which is read every ``CHECK_INTERVAL`` steps;
  reaching either budget stops the run with ``ConstraintBudgetExceeded``.
"""

import functools
import os

import warp as wp
from mjlab.sim import sim as _sim
from mjlab.terrains import terrain_entity as _terrain

CHECK_INTERVAL = 250  # physics steps between reads of the device-side peak

_applied = False


class ConstraintBudgetExceeded(RuntimeError):
    pass


@wp.kernel
def _track_peaks(nefc: wp.array[int], nacon: wp.array[int], peak: wp.array[int]):
    worldid = wp.tid()
    wp.atomic_max(peak, 0, nefc[worldid])
    if worldid == 0:
        wp.atomic_max(peak, 1, nacon[0])


def check_budget(peak, njmax: int, naconmax: int) -> None:
    """Raise if the tracked peaks reached the constraint-row or contact budget."""
    rows, contacts = (int(v) for v in peak.numpy()[:2])
    if rows >= njmax:
        raise ConstraintBudgetExceeded(
            f"a world used {rows} constraint rows with njmax={njmax}; rows past the budget are dropped. "
            "Raise --env.sim.njmax."
        )
    if contacts >= naconmax:
        raise ConstraintBudgetExceeded(
            f"{contacts} contacts with a budget of {naconmax}; contacts past it are dropped. "
            "Raise --env.sim.nconmax."
        )


def apply() -> None:
    """Patch mjlab's terrain and ``Simulation``. Safe to call repeatedly."""
    global _applied
    if _applied:
        return
    _applied = True

    if os.environ.get("MICRODUCK_ENV_ORIGIN_SITES") != "1":
        _terrain.TerrainEntity._add_env_origin_sites = lambda self: None

    step = _sim.Simulation.step

    @functools.wraps(step)
    def guarded_step(self) -> None:
        step(self)
        d = self._wp_data
        if getattr(self, "_budget_peak", None) is None:
            self._budget_peak = wp.zeros(2, dtype=int, device=self.wp_device)
            self._budget_steps = 0
        wp.launch(_track_peaks, dim=d.nworld, inputs=[d.nefc, d.nacon, self._budget_peak], device=self.wp_device)
        self._budget_steps += 1
        if self._budget_steps % CHECK_INTERVAL == 0:
            check_budget(self._budget_peak, d.njmax, d.naconmax)

    guarded_step._microduck_perf = True
    _sim.Simulation.step = guarded_step
