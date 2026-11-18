"""A moved upstream ref invalidates a previous review; failures never look current."""
import json
import hashlib
import io
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from check_upstream import check, pins, policy_updates, review_allows
import pytest


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def test_real_fetch_diff_review_and_network_failure(tmp_path, monkeypatch):
    monkeypatch.setattr("check_upstream.policy_updates", lambda root: {"status": "CURRENT", "revision": "b" * 40, "changes": {}})
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    git(upstream, "init", "-b", "develop")
    git(upstream, "config", "user.email", "test@example.invalid")
    git(upstream, "config", "user.name", "Test")
    git(upstream, "config", "uploadpack.allowFilter", "true")
    (upstream / "AGENTS.md").write_text("old\n")
    git(upstream, "add", ".")
    git(upstream, "commit", "-m", "initial")
    pin = git(upstream, "rev-parse", "HEAD")
    root = tmp_path / "extension"
    (root / "env").mkdir(parents=True)
    manual = root / "docs/upstream"
    manual.mkdir(parents=True)
    (manual / "AGENTS.develop.md").write_bytes(b"old\n")
    (manual / "source.json").write_text(json.dumps({"commit": pin, "sha256": hashlib.sha256(b"old\n").hexdigest(),
                                                    "url": "fixture"}))
    url = upstream.as_uri()
    (root / "env/pins.toml").write_text(f'[repos.microduck_rl]\nurl = "{url}"\ncommit = "{pin}"\n')
    (root / "pyproject.toml").write_text(f'[project]\ndependencies = ["mjlab-microduck @ git+{url}@{pin}"]\n')
    assert check(root)["status"] == "CURRENT"
    (upstream / "AGENTS.md").write_text("new safety rule\n")
    git(upstream, "commit", "-am", "new rule")
    new = git(upstream, "rev-parse", "HEAD")
    report = check(root)
    assert report["status"] == "UPDATE_AVAILABLE" and not report["training_allowed"]
    assert "M\tAGENTS.md" in report["changed_files"]
    assert "new safety rule" in (root / "logs/upstream-review/focused.diff").read_text()
    assert check(root, reviewed_sha=new, note="Keep pin for controlled comparison")["training_allowed"]
    assert not review_allows(report, pin, "stale review")
    assert not review_allows(report, new, " ")
    # Reuse a populated filtered cache after upstream adds a new directory/blob.
    (upstream / "src/mjlab_microduck/tasks").mkdir(parents=True)
    (upstream / "src/mjlab_microduck/tasks/new.py").write_text("new task safety rule\n")
    git(upstream, "add", ".")
    git(upstream, "commit", "-m", "new task directory")
    report = check(root)
    assert report["status"] == "UPDATE_AVAILABLE"
    assert "new task safety rule" in (root / "logs/upstream-review/focused.diff").read_text()
    assert report["manual_snapshot"]["verified_against_git"]
    (manual / "AGENTS.develop.md").write_text("tampered\n")
    assert check(root)["status"] == "UNKNOWN"
    provenance = json.loads((manual / "source.json").read_text())
    provenance["sha256"] = hashlib.sha256(b"tampered\n").hexdigest()
    (manual / "source.json").write_text(json.dumps(provenance))
    assert check(root)["status"] == "UNKNOWN"  # changing the manifest too cannot forge the Git blob
    provenance["sha256"] = hashlib.sha256(b"old\n").hexdigest()
    (manual / "source.json").write_text(json.dumps(provenance))
    (manual / "AGENTS.develop.md").write_bytes(b"old\n")
    upstream.rename(tmp_path / "offline")
    failed = check(root, reviewed_sha=new, note="previous review")
    assert failed["status"] == "UNKNOWN" and not failed["training_allowed"]
    assert not (root / "logs/upstream-review/focused.diff").exists()
    assert json.loads((root / "logs/upstream-review/upstream.json").read_text())["status"] == "UNKNOWN"


def test_mismatched_dependency_pin_is_rejected(tmp_path):
    (tmp_path / "env").mkdir()
    (tmp_path / "env/pins.toml").write_text('[repos.microduck_rl]\nurl="official"\ncommit="' + "a" * 40 + '"\n')
    (tmp_path / "pyproject.toml").write_text('[project]\ndependencies=[]\n')
    with pytest.raises(ValueError, match="disagree"):
        pins(tmp_path)


def test_new_model_weights_require_their_own_exact_review():
    report = {"status": "UPDATE_AVAILABLE", "code_status": "CURRENT", "remote_commit": "a" * 40,
              "policies": {"status": "UPDATE_AVAILABLE", "revision": "b" * 40}}
    assert not review_allows(report, "a" * 40, "code only")
    assert not review_allows(report, "a" * 40, "outdated model review", "c" * 40)
    assert review_allows(report, note="Review official model update; keep our current model", reviewed_policies="b" * 40)


def test_policy_catalog_detects_replacement_and_missing_hash(tmp_path, monkeypatch):
    root = tmp_path / "docs/upstream"
    root.mkdir(parents=True)
    (root / "policies.json").write_text(json.dumps({"source": "https://example.invalid/catalog", "repository": "official/test",
                                                    "revision": "a" * 40, "sha256": {"sit.onnx": "1" * 64}}))
    info = {"id": "official/test", "sha": "b" * 40,
            "siblings": [{"rfilename": "sit.onnx", "lfs": {"sha256": "2" * 64}}]}
    monkeypatch.setattr("check_upstream.urllib.request.urlopen", lambda *a, **kw: io.BytesIO(json.dumps(info).encode()))
    result = policy_updates(tmp_path)
    assert result["status"] == "UPDATE_AVAILABLE" and result["changes"]["sit.onnx"]["remote"] == "2" * 64
    info["siblings"][0]["lfs"]["sha256"] = "1" * 64
    assert policy_updates(tmp_path)["status"] == "CURRENT"  # catalog README changed, model bytes did not
    del info["siblings"][0]["lfs"]
    with pytest.raises(ValueError, match="Missing LFS SHA256"):
        policy_updates(tmp_path)
