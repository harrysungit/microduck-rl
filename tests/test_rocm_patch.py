from types import SimpleNamespace

import pytest
import torch
from mjlab.sim.sim import Simulation

from microduck_rl.patches.rocm import CachingAllocator, apply


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


class _Driver:
    memory_kind = None
    deallocate_requires_context_guard = True

    def __init__(self):
        self.next_ptr = 0x1000

    def allocate(self, size_in_bytes):
        self.next_ptr += 0x1000
        return self.next_ptr

    def deallocate(self, ptr, size_in_bytes):
        pytest.fail("a cached block must not be returned to the driver")


def test_caching_allocator_reuses_a_freed_block_of_the_same_size():
    alloc = CachingAllocator(_Driver())
    ptr = alloc.allocate(256)
    alloc.deallocate(ptr, 256)
    assert alloc.allocate(256) == ptr


def test_caching_allocator_does_not_hand_out_a_block_of_another_size():
    alloc = CachingAllocator(_Driver())
    ptr = alloc.allocate(256)
    alloc.deallocate(ptr, 256)
    assert alloc.allocate(512) != ptr


def test_caching_allocator_never_hands_out_a_live_block():
    alloc = CachingAllocator(_Driver())
    live = alloc.allocate(256)
    assert alloc.allocate(256) != live


@pytest.mark.skipif(not torch.version.hip, reason="only HIP builds of torch turn TorchScript fusion off")
def test_torchscript_fusion_is_off_on_hip():
    apply()
    assert not torch._C._jit_texpr_fuser_enabled()


class _Device:
    is_capturing = False


def test_a_block_taken_during_capture_stays_with_the_graphs():
    device = _Device()
    alloc = CachingAllocator(_Driver(), device)
    warm = alloc.allocate(256)
    alloc.deallocate(warm, 256)
    device.is_capturing = True
    ptr = alloc.allocate(256)
    alloc.deallocate(ptr, 256)
    device.is_capturing = False
    assert ptr == warm and alloc.capture_misses == 0
    assert alloc.allocate(256) != ptr


def test_a_capture_reuses_blocks_it_freed_and_graphs_share_scratch():
    device = _Device()
    alloc = CachingAllocator(_Driver(), device)
    warm = alloc.allocate(256)
    alloc.deallocate(warm, 256)
    device.is_capturing = True
    first = alloc.allocate(256)
    alloc.deallocate(first, 256)
    again = alloc.allocate(256)  # later in the same capture
    alloc.deallocate(again, 256)
    device.is_capturing = False
    device.is_capturing = True
    other_graph = alloc.allocate(256)  # a second capture
    device.is_capturing = False
    assert first == again == other_graph == warm
    assert alloc.capture_misses == 0


def test_a_cache_miss_during_capture_is_counted():
    device = _Device()
    alloc = CachingAllocator(_Driver(), device)
    device.is_capturing = True
    alloc.allocate(512)
    assert alloc.capture_misses == 1
