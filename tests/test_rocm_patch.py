from types import SimpleNamespace

from mjlab.sim.sim import Simulation

from microduck_rl.patches.rocm import apply


def test_hip_device_runs_eagerly():
    apply()
    hip = SimpleNamespace(wp_device=SimpleNamespace(is_hip=True, is_cuda=True))
    assert Simulation._should_use_cuda_graph(hip) is False


def test_cpu_device_keeps_upstream_decision():
    apply()
    cpu = SimpleNamespace(wp_device=SimpleNamespace(is_hip=False, is_cuda=False))
    assert Simulation._should_use_cuda_graph(cpu) is False


def test_apply_is_idempotent():
    apply()
    patched = Simulation._should_use_cuda_graph
    apply()
    assert Simulation._should_use_cuda_graph is patched
    assert patched._microduck_rocm
