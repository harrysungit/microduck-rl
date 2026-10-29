import numpy as np
import pytest
import warp as wp
from mjlab.sim.sim import Simulation
from mjlab.terrains.terrain_entity import TerrainEntity

from microduck_rl.patches.perf import ConstraintBudgetExceeded, _track_peaks, apply, check_budget


def test_apply_skips_origin_sites_and_guards_step():
    apply()
    assert TerrainEntity._add_env_origin_sites(object()) is None
    assert Simulation.step._microduck_perf


def _peaks(nefc, nacon):
    peak = wp.zeros(2, dtype=int, device="cpu")
    wp.launch(_track_peaks, dim=len(nefc), device="cpu",
              inputs=[wp.array(np.array(nefc, dtype=np.int32), dtype=int, device="cpu"),
                      wp.array(np.array([nacon], dtype=np.int32), dtype=int, device="cpu"), peak])
    return peak


def test_peaks_track_the_largest_world_and_the_contact_count():
    assert list(_peaks([3, 51, 7], 30).numpy()) == [51, 30]


def test_budget_within_limits_passes():
    check_budget(_peaks([40, 51], 100), njmax=128, naconmax=1000)


def test_constraint_rows_at_the_budget_stop_the_run():
    with pytest.raises(ConstraintBudgetExceeded, match="njmax=128"):
        check_budget(_peaks([40, 128], 100), njmax=128, naconmax=1000)


def test_contacts_at_the_budget_stop_the_run():
    with pytest.raises(ConstraintBudgetExceeded, match="nconmax"):
        check_budget(_peaks([40], 1000), njmax=128, naconmax=1000)
