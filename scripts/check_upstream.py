"""Discover upstream changes without updating the installed dependency (stdlib + git)."""

import argparse
import hashlib
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import tomllib
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
FOCUS = ["AGENTS.md", "pyproject.toml", "uv.lock", "docs/sitstand_policy.md",
         "src/mjlab_microduck/tasks", "src/mjlab_microduck/robot/microduck_constants.py",
         "scripts/export.py", "src/mjlab_microduck/export.py", "src/mjlab_microduck/train_hook.py",
         "src/mjlab_microduck/checkpoints.py", "src/mjlab_microduck/publish"]


def git(repo, *args):
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=240)
    if result.returncode:
        raise RuntimeError(result.stderr.strip()[-2000:] or f"git failed ({result.returncode})")
    return result.stdout.strip()


def pins(root):
    config = tomllib.loads((root / "env/pins.toml").read_text(encoding="utf-8"))["repos"]["microduck_rl"]
    dependencies = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]["dependencies"]
    expected = f"mjlab-microduck @ git+{config['url']}@{config['commit']}"
    if expected not in dependencies:
        raise ValueError("env/pins.toml and pyproject.toml disagree on the official dependency")
    if not re.fullmatch(r"[0-9a-f]{40}", config["commit"]):
        raise ValueError("Upstream must be pinned to a full commit SHA")
    return config


def policy_updates(root):
    baseline = json.loads((root / "docs/upstream/policies.json").read_text(encoding="utf-8"))
    request = urllib.request.Request(baseline["source"], headers={"User-Agent": "microduck-rl-upstream-review"})
    with urllib.request.urlopen(request, timeout=30) as response:
        info = json.load(response)
    if info.get("id") != baseline["repository"] or not re.fullmatch(r"[0-9a-f]{40}", info.get("sha", "")):
        raise ValueError("Unexpected official policy catalog")
    observed = {}
    for item in info["siblings"]:
        if item["rfilename"].endswith(".onnx"):
            digest = item.get("lfs", {}).get("sha256", "")
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(f"Missing LFS SHA256 for {item['rfilename']}")
            observed[item["rfilename"]] = digest
    if not observed:
        raise ValueError("Official policy catalog returned no ONNX files")
    changes = {n: {"baseline": baseline["sha256"].get(n), "remote": observed.get(n)}
               for n in sorted(set(baseline["sha256"]) | set(observed))
               if baseline["sha256"].get(n) != observed.get(n)}
    return {"status": "UPDATE_AVAILABLE" if changes else "CURRENT", "revision": info["sha"],
            "repository": baseline["repository"], "last_modified": info.get("lastModified"),
            "baseline_revision": baseline["revision"], "changes": changes,
            "remote_sha256": observed, "weights_downloaded": False}


def review_allows(report, reviewed_sha=None, note=None, reviewed_policies=None):
    if report["status"] == "UNKNOWN":
        return False
    noted = bool(note and note.strip())
    code_ok = (report.get("code_status", report["status"]) == "CURRENT" or
               (reviewed_sha == report.get("remote_commit") and noted))
    models = report.get("policies", {"status": "CURRENT"})
    models_ok = (models["status"] == "CURRENT" or
                 (models["status"] == "UPDATE_AVAILABLE" and reviewed_policies == models["revision"] and noted))
    return code_ok and models_ok


def snapshot_source(root, cache, repository):
    """Bind the reference manual to its recorded Git object, not only a self-reported hash."""
    source = json.loads((root / "docs/upstream/source.json").read_text(encoding="utf-8"))
    commit, expected = source["commit"], source["sha256"]
    if not re.fullmatch(r"[0-9a-f]{40}", commit) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("Invalid upstream manual provenance")
    local = (root / "docs/upstream/AGENTS.develop.md").read_bytes()
    if hashlib.sha256(local).hexdigest() != expected:
        raise ValueError("Upstream manual snapshot SHA256 mismatch")
    # A previous snapshot can predate the current pin, or survive a rewritten develop.
    git(cache, "fetch", "--no-tags", "review", f"+{commit}:refs/review/manual")
    result = subprocess.run(["git", "-C", str(cache), "show", f"{commit}:AGENTS.md"],
                            capture_output=True, timeout=240)
    if result.returncode or hashlib.sha256(result.stdout).hexdigest() != expected:
        raise ValueError("Upstream manual snapshot does not match its recorded Git source")
    if repository == "https://github.com/pollen-robotics/microduck_rl.git":
        url = f"https://raw.githubusercontent.com/pollen-robotics/microduck_rl/{commit}/AGENTS.md"
        if source["url"] != url:
            raise ValueError("Upstream manual source URL does not match its recorded commit")
    return {**source, "verified_against_git": True}


def check(root=ROOT, out_dir=None, cache=None, reviewed_sha=None, note=None, reviewed_policies=None):
    root = Path(root)
    out_dir = Path(out_dir or root / "logs/upstream-review")
    cache = Path(cache or root / ".cache/upstream-review.git").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "branch": "develop",
              "status": "UNKNOWN", "training_allowed": False, "installed_dependency_changed": False}
    try:
        source = pins(root)
        report.update(repository=source["url"], pinned_commit=source["commit"])
        if not cache.exists():
            cache.mkdir(parents=True)
            git(cache, "init", "--bare")
        if git(cache, "rev-parse", "--is-bare-repository") != "true":
            raise ValueError("The review cache must be a bare repository, not an installed checkout")
        # Name the remote explicitly so filtered object fetches can retrieve missing blobs
        # when diff/show accesses a directory added by a later upstream update.
        git(cache, "config", "remote.review.url", source["url"])
        git(cache, "config", "remote.review.promisor", "true")
        git(cache, "config", "remote.review.partialclonefilter", "blob:none")
        # Fetch explicit refs together. Never trust a previous cached ref if this fetch fails.
        git(cache, "fetch", "--no-tags", "--filter=blob:none", "review",
            f"+{source['commit']}:refs/review/pinned", "+refs/heads/develop:refs/review/develop")
        current = git(cache, "rev-parse", "refs/review/develop")
        fixed = source["commit"]
        report["remote_commit"] = current
        report["manual_snapshot"] = snapshot_source(root, cache, source["url"])
        report["code_status"] = "CURRENT" if current == fixed else "UPDATE_AVAILABLE"
        report["policies"] = policy_updates(root)
        report["status"] = ("CURRENT" if report["code_status"] == report["policies"]["status"] == "CURRENT"
                            else "UPDATE_AVAILABLE")
        report["commits_since_pin"] = git(cache, "log", "--format=%h %s", f"{fixed}..{current}").splitlines()
        report["changed_files"] = git(cache, "diff", "--name-status", fixed, current).splitlines()
        report["compare_url"] = (f"https://github.com/pollen-robotics/microduck_rl/compare/{fixed}...{current}"
                                 if source["url"].startswith("https://github.com/pollen-robotics/") else None)
        (out_dir / "focused.diff").write_text(git(cache, "diff", fixed, current, "--", *FOCUS) + "\n",
                                               encoding="utf-8")
        report["review"] = {"remote_commit": reviewed_sha, "policies_revision": reviewed_policies, "note": note}
        report["training_allowed"] = review_allows(report, reviewed_sha, note, reviewed_policies)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.TimeoutExpired) as exc:
        report["status"] = "UNKNOWN"
        report["error"] = str(exc)
        # Remove only this tool's stale diff, so an error report cannot look like a fresh review.
        (out_dir / "focused.diff").unlink(missing_ok=True)
    (out_dir / "upstream.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# Official upstream check", "", f"Status: **{report['status']}**",
             f"Checked: {report['checked_at_utc']}", f"Pinned: `{report.get('pinned_commit', 'unknown')}`",
             f"Develop: `{report.get('remote_commit', 'unknown')}`",
             f"Training review satisfied: {report['training_allowed']}", "",
             "This check does not update installed code or models. It is run on demand, not a background subscription.", ""]
    if report.get("error"):
        lines += ["Check failed; upstream state is unknown.", report["error"], ""]
    if note:
        lines += [f"Review note: {note}", ""]
    if "policies" in report:
        models = report["policies"]
        lines += ["## Official policy files", "", f"Status: {models['status']}; revision `{models['revision']}`.",
                  "Only catalog hashes were fetched. No weights were downloaded or deployed.", "",
                  "```json", json.dumps(models["changes"], indent=2), "```", ""]
    lines += ["## Commits", "", *(f"- {x}" for x in report.get("commits_since_pin", [])), "",
              "## Changed files", "", "```text", *report.get("changed_files", []), "```", "",
              "Full focused changes: focused.diff (only present after a successful check).", ""]
    (out_dir / "upstream.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "logs/upstream-review")
    ap.add_argument("--cache", type=Path)
    ap.add_argument("--reviewed-upstream", help="exact remote SHA already reviewed; does not update the pin")
    ap.add_argument("--upstream-review-note", help="why this run keeps the old pin / where the review is recorded")
    ap.add_argument("--reviewed-policies", help="exact official policy-catalog revision already reviewed")
    args = ap.parse_args()
    result = check(out_dir=args.out_dir, cache=args.cache, reviewed_sha=args.reviewed_upstream,
                   note=args.upstream_review_note, reviewed_policies=args.reviewed_policies)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 2 if result["status"] == "UNKNOWN" else 0 if result["training_allowed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
