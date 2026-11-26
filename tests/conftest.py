import pytest


@pytest.fixture(autouse=True)
def _no_runtime_env(monkeypatch):
    """Tests never touch the shared runtime: env/activate.sh exports these, and check_upstream's
    default cache follows MICRODUCK_RL_CACHE, so a test run under it fetched its fixture repository
    into the real review cache."""
    for name in ("MICRODUCK_RL_CACHE", "MICRODUCK_RL_VENV", "MICRODUCK_RL_DEPS"):
        monkeypatch.delenv(name, raising=False)
