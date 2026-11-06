"""fixed_std.py: the exploration std stays where it is put, and nothing changes without the variable."""

import importlib.util
from pathlib import Path

import pytest
import torch

PATH = Path(__file__).resolve().parents[1] / "src/microduck_rl/patches/fixed_std.py"
distribution = pytest.importorskip("rsl_rl.modules.distribution")


def load():
    spec = importlib.util.spec_from_file_location("fixed_std_under_test", PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def restore():
    original = distribution.GaussianDistribution.update
    yield
    distribution.GaussianDistribution.update = original


def test_without_the_variable_nothing_is_patched(monkeypatch, restore):
    monkeypatch.delenv("MICRODUCK_FIXED_STD", raising=False)
    before = distribution.GaussianDistribution.update
    load().apply()
    assert distribution.GaussianDistribution.update is before


def test_the_std_is_fixed_frozen_and_survives_a_resume_with_another_value(monkeypatch, restore):
    monkeypatch.setenv("MICRODUCK_FIXED_STD", "0.05")
    load().apply()
    dist = distribution.GaussianDistribution(4, init_std=1.0)
    dist.std_param.data.fill_(0.19)  # what loading a checkpoint writes
    mean = torch.zeros(3, 4, requires_grad=True)
    dist.update(mean)
    assert torch.allclose(dist.std_param, torch.full((4,), 0.05)) and not dist.std_param.requires_grad
    assert torch.allclose(dist.std, torch.full((3, 4), 0.05))
    # the loss reaches the mean but not the std, entropy bonus included
    loss = -dist._distribution.log_prob(torch.ones(3, 4)).sum() - 1.0 * dist.entropy.sum()
    loss.backward()
    assert mean.grad is not None and dist.std_param.grad is None
    optimizer = torch.optim.Adam(dist.parameters(), lr=1.0)
    optimizer.step()
    assert torch.allclose(dist.std_param, torch.full((4,), 0.05))


def test_samples_use_the_fixed_std(monkeypatch, restore):
    monkeypatch.setenv("MICRODUCK_FIXED_STD", "0.05")
    load().apply()
    dist = distribution.GaussianDistribution(2, init_std=1.0)
    dist.update(torch.zeros(20000, 2))
    assert abs(float(dist.sample().std()) - 0.05) < 0.003


def test_a_bad_value_or_a_log_std_is_refused(monkeypatch, restore):
    monkeypatch.setenv("MICRODUCK_FIXED_STD", "-1")
    with pytest.raises(ValueError):
        load().apply()
    monkeypatch.setenv("MICRODUCK_FIXED_STD", "0.05")
    load().apply()
    log_dist = distribution.GaussianDistribution(2, init_std=0.5, std_type="log")
    with pytest.raises(ValueError):
        log_dist.update(torch.zeros(1, 2))
