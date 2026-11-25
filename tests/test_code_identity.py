"""identity.py: a run records the commit, local changes, untracked sources and launcher it ran with."""

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "src/microduck_rl/patches/identity.py"


def load():
    spec = importlib.util.spec_from_file_location("identity_under_test", PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git(root, *args):
    subprocess.run(["git", "-C", str(root), "-c", "user.email=t@example.com", "-c", "user.name=t", *args],
                   check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "code"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "__init__.py").write_text("A = 1\n")
    (root / "pkg" / "task.py").write_text("B = 1\n")
    git(root, "init", "-q")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "base")
    (root / "pkg" / "task.py").write_text("B = 2\n")  # modified, tracked
    (root / "pkg" / "new_recipe.py").write_text("C = 3\n")  # untracked source
    (root / "big.bin").write_bytes(b"\0" * 10)  # untracked, not a source suffix
    return root


def test_records_commit_diff_untracked_and_entry(tmp_path, repo, monkeypatch):
    module = load()
    entry = tmp_path / "launch.py"
    entry.write_text("print('go')\n")
    loose = tmp_path / "loose" / "mod.py"
    loose.parent.mkdir()
    loose.write_text("x = 1\n")
    monkeypatch.setattr(module, "_targets", lambda: [
        ("microduck_rl", repo / "pkg" / "__init__.py"),
        ("mjlab_microduck", None),
        ("entry", entry),
        ("other", loose),
    ])
    monkeypatch.setenv("MICRODUCK_FIXED_STD", "0.05")
    out = Path(module.write_identity(str(tmp_path / "run")))
    rec = json.loads(out.read_text())
    by_role = {r["role"]: r for r in rec["repos"]}

    code = by_role["microduck_rl"]
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    assert code["commit"] == head and code["dirty"] is True
    assert "B = 2" in (out.parent / code["diff_file"]).read_text()
    assert code["untracked_copied"] == ["pkg/new_recipe.py"] and code["untracked_skipped"] == ["big.bin"]
    assert (out.parent / "code_identity/untracked" / f"microduck_rl__{repo.name}" / "pkg/new_recipe.py").read_text() == "C = 3\n"

    assert by_role["mjlab_microduck"]["note"] == "not importable"
    assert by_role["other"]["git"] is None  # outside any repository: recorded, not fatal
    assert rec["entry_script"]["sha256"] == hashlib.sha256(entry.read_bytes()).hexdigest()
    assert (out.parent / rec["entry_script"]["copy"]).read_text() == "print('go')\n"
    assert rec["env"]["MICRODUCK_FIXED_STD"] == "0.05"
    assert rec["errors"] == []
    assert "torch" in rec["runtime"] or "error" in rec["runtime"]


def test_a_second_write_keeps_the_first(tmp_path, repo, monkeypatch):
    module = load()
    monkeypatch.setattr(module, "_targets", lambda: [("microduck_rl", repo / "pkg" / "__init__.py")])
    first = Path(module.write_identity(str(tmp_path / "run")))
    before = first.read_text()
    second = Path(module.write_identity(str(tmp_path / "run")))
    assert first.name == "CODE_IDENTITY.json" and second != first and second.name.startswith("CODE_IDENTITY.")
    assert first.read_text() == before


logger_mod = pytest.importorskip("rsl_rl.utils.logger")


@pytest.fixture
def restore():
    original = logger_mod.Logger._store_code_state
    yield
    logger_mod.Logger._store_code_state = original


def make_logger(log_dir, rank=0, distributed=False):
    cfg = {"algorithm": {"rnd_cfg": None}, "logger": "tensorboard"}
    return logger_mod.Logger(log_dir=str(log_dir), cfg=cfg, env_cfg={}, num_envs=1, is_distributed=distributed,
                             gpu_world_size=2 if distributed else 1, gpu_global_rank=rank, device="cpu")


def test_logger_stores_rsl_rl_diff_and_identity(tmp_path, repo, monkeypatch, restore):
    module = load()
    monkeypatch.setattr(module, "_targets", lambda: [("microduck_rl", repo / "pkg" / "__init__.py")])
    module.apply()
    module.apply()  # idempotent
    log = make_logger(tmp_path / "run")
    files = log._store_code_state()
    assert str(repo / "pkg" / "__init__.py") in log.git_status_repos
    assert (tmp_path / "run" / "git" / f"{repo.name}.diff").is_file()  # rsl_rl's own diff now covers it
    assert any(f.endswith("CODE_IDENTITY.json") for f in files)


def test_non_logging_rank_writes_nothing(tmp_path, repo, monkeypatch, restore):
    module = load()
    monkeypatch.setattr(module, "_targets", lambda: [("microduck_rl", repo / "pkg" / "__init__.py")])
    module.apply()
    log = make_logger(tmp_path / "run", rank=1, distributed=True)
    log._store_code_state()
    assert not (tmp_path / "run" / "CODE_IDENTITY.json").exists()
