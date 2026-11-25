"""Record which code a training run actually ran: CODE_IDENTITY.json in the run directory.

rsl_rl stores a git diff of the repositories in ``Logger.git_status_repos`` when ``learn`` starts,
but only of rsl_rl and of the script that registered itself, and its diff leaves out untracked
files. A run started from this extension depends on three more places: this package, the
upstream task package ``mjlab_microduck`` and the launching script. This patch wraps
``Logger._store_code_state`` so that, in the same place and only on the logging rank:

- all three are added to ``git_status_repos``, so rsl_rl's own ``git/<repo>.diff`` covers them;
- ``CODE_IDENTITY.json`` records, per repository, the commit, branch, dirty state and the list of
  untracked files, plus the command line, ``MICRODUCK_*`` and GPU-runtime environment variables,
  installed package versions with editable locations, and torch's device runtime;
- ``code_identity/`` keeps a copy of each repository's ``git diff HEAD``, its untracked source
  files (text files up to 1 MB) and the launching script, so code that was never committed can
  still be recovered from the run.

A failure in any part is written into the file and printed, and never stops training. A resumed
run that logs into a directory that already has the file gets a time-stamped one next to it.
Applied from ``microduck_rl.tasks``.
"""

import datetime
import functools
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

_applied = False
_ENV_PREFIXES = ("MICRODUCK_", "MJLAB_", "HIP_", "ROCM_", "ROCR_", "HSA_", "PYTORCH_", "WARP_", "CUDA_")
_SOURCE_SUFFIXES = {".py", ".toml", ".json", ".yaml", ".yml", ".cfg", ".ini", ".sh", ".md", ".txt", ".xml", ".csv"}
_MAX_COPY_BYTES = 1_000_000
_MAX_COPY_FILES = 2000


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True).stdout


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _targets() -> list[tuple[str, Path | None]]:
    """(role, a file inside it) for every place the run's code comes from; None if not importable."""
    out: list[tuple[str, Path | None]] = []
    for role in ("microduck_rl", "mjlab_microduck"):
        try:
            out.append((role, Path(__import__(role).__file__).resolve()))
        except Exception:  # noqa: BLE001 - recorded as missing by the caller
            out.append((role, None))
    entry = Path(sys.argv[0]).resolve() if sys.argv and sys.argv[0] else None
    if entry is not None and entry.is_file():
        out.append(("entry", entry))
    return out


def _repo_record(role: str, path: Path, out_dir: Path) -> dict:
    rec: dict = {"role": role, "path": str(path)}
    try:
        root = Path(_git(path.parent, "rev-parse", "--show-toplevel").strip())
    except Exception as exc:  # noqa: BLE001
        rec["git"] = None
        rec["note"] = f"not in a git repository: {exc.__class__.__name__}"
        return rec
    rec["root"] = str(root)
    rec["commit"] = _git(root, "rev-parse", "HEAD").strip()
    rec["branch"] = _git(root, "rev-parse", "--abbrev-ref", "HEAD").strip()
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all").splitlines()
    rec["dirty"] = bool(status)
    tag = f"{role}__{root.name}"
    diff = _git(root, "diff", "HEAD")
    if diff:
        (out_dir / f"{tag}.diff").write_text(diff, encoding="utf-8")
        rec["diff_file"] = f"{out_dir.name}/{tag}.diff"
    untracked = [line[3:] for line in status if line.startswith("?? ")]
    rec["untracked"] = untracked
    copied, skipped = [], []
    for rel in untracked:
        src = root / rel
        if (len(copied) < _MAX_COPY_FILES and src.is_file() and src.suffix in _SOURCE_SUFFIXES
                and src.stat().st_size <= _MAX_COPY_BYTES):
            dst = out_dir / "untracked" / tag / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied.append(rel)
        else:
            skipped.append(rel)
    rec["untracked_copied"] = copied
    rec["untracked_skipped"] = skipped
    return rec


def _packages() -> dict:
    versions, editable = {}, {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"]
        if not name:
            continue
        versions[name] = dist.version
        direct = dist.read_text("direct_url.json")
        if direct:
            try:
                info = json.loads(direct)
                if info.get("dir_info", {}).get("editable"):
                    editable[name] = info.get("url", "")
            except ValueError:
                pass
    return {"versions": dict(sorted(versions.items())), "editable": editable}


def _torch_runtime() -> dict:
    try:
        import torch

        return {"torch": torch.__version__, "hip": getattr(torch.version, "hip", None), "cuda": torch.version.cuda}
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc)}


def write_identity(log_dir: str) -> str:
    """Write CODE_IDENTITY.json (and code_identity/) into ``log_dir``; return the JSON path."""
    log = Path(log_dir)
    log.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name, files = "CODE_IDENTITY.json", "code_identity"
    if (log / name).exists():
        name, files = f"CODE_IDENTITY.{stamp}.json", f"code_identity.{stamp}"
    out_dir = log / files
    out_dir.mkdir(parents=True, exist_ok=True)
    record: dict = {
        "schema": 1,
        "created_utc": stamp,
        "argv": list(sys.argv),
        "cwd": os.getcwd(),
        "python": sys.version,
        "executable": sys.executable,
        "env": {k: v for k, v in sorted(os.environ.items()) if k.startswith(_ENV_PREFIXES)},
        "repos": [],
        "errors": [],
    }
    for role, path in _targets():
        if path is None:
            record["repos"].append({"role": role, "path": None, "note": "not importable"})
            continue
        try:
            record["repos"].append(_repo_record(role, path, out_dir))
        except Exception as exc:  # noqa: BLE001
            record["errors"].append(f"{role}: {exc!r}")
        if role == "entry" and path.is_file():
            try:
                dst = out_dir / "entry" / path.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dst)
                record["entry_script"] = {"path": str(path), "sha256": _sha256(path), "copy": f"{files}/entry/{path.name}"}
            except Exception as exc:  # noqa: BLE001
                record["errors"].append(f"entry copy: {exc!r}")
    try:
        record["packages"] = _packages()
    except Exception as exc:  # noqa: BLE001
        record["errors"].append(f"packages: {exc!r}")
    record["runtime"] = _torch_runtime()
    path = log / name
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if record["errors"]:
        print(f"[microduck_rl] code identity written with errors: {record['errors']}", file=sys.stderr)
    return str(path)


def apply() -> None:
    """Wrap ``rsl_rl.utils.logger.Logger._store_code_state``. Safe to call repeatedly."""
    global _applied
    if _applied:
        return
    try:
        from rsl_rl.utils.logger import Logger
    except ImportError:
        return
    _applied = True
    store = Logger._store_code_state

    @functools.wraps(store)
    def store_with_identity(self) -> list[str]:
        for _, path in _targets():
            if path is not None and path.exists() and str(path) not in self.git_status_repos:
                self.git_status_repos.append(str(path))
        files = store(self)
        if self.log_dir is not None and not self.disable_logs:
            try:
                files = [*files, write_identity(self.log_dir)]
            except Exception as exc:  # noqa: BLE001 - never stop training over provenance
                print(f"[microduck_rl] could not write CODE_IDENTITY.json: {exc!r}", file=sys.stderr)
        return files

    Logger._store_code_state = store_with_identity
