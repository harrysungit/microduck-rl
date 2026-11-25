#!/usr/bin/env bash
# Build the ROCm (gfx1151) training environment from env/pins.toml.
#
# Idempotent: re-running skips what is already in place. Writes inside the
# repository (.venv/, .cache/, .deps/), or wherever MICRODUCK_RL_VENV,
# MICRODUCK_RL_CACHE and MICRODUCK_RL_DEPS point; no sudo, no system packages.
# Needs the host ROCm at /opt/rocm for the Warp build.
#
#   bash env/setup.sh                 # full setup
#   bash env/setup.sh --rebuild-warp  # force a Warp rebuild
set -euo pipefail

ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$ENV_DIR")"
DEPS="${MICRODUCK_RL_DEPS:-$ROOT/.deps}"
REBUILD_WARP=0
[ "${1:-}" = "--rebuild-warp" ] && REBUILD_WARP=1

# Redirect caches before anything downloads or compiles.
# shellcheck disable=SC1091
source "$ENV_DIR/activate.sh"

pin() {
    /usr/bin/python3 -c 'import sys, tomllib
d = tomllib.load(open(sys.argv[1], "rb"))
for key in sys.argv[2].split("."):
    d = d[key]
print(d)' "$ENV_DIR/pins.toml" "$1"
}

# Clone <repo> into .deps/<name> once, then check out its pinned commit.
checkout() {
    local name="$1" dir="$DEPS/$1"
    [ -d "$dir/.git" ] || git clone -q "$(pin "repos.$name.url")" "$dir"
    git -C "$dir" fetch -q origin "$(pin "repos.$name.commit")" 2>/dev/null || true
    git -C "$dir" checkout -q "$(pin "repos.$name.commit")"
}

VENV="${MICRODUCK_RL_VENV:-$ROOT/.venv}"
if [ ! -x "$VENV/bin/python" ]; then
    echo "== creating venv"
    /usr/bin/python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$ENV_DIR/activate.sh"
python -m pip install -q --upgrade pip
python -m pip install -q numpy setuptools packaging wheel

# 1. PyTorch for gfx1151; the wheel brings its own rocm-sdk libraries.
TORCH_VER="$(pin pip.torch.version)"
if ! python -c "import sys, torch; sys.exit(torch.__version__ != '$TORCH_VER')" 2>/dev/null; then
    echo "== installing torch $TORCH_VER"
    python -m pip install -q --index-url "$(pin pip.torch.index)" \
        --extra-index-url https://pypi.org/simple "torch==$TORCH_VER"
fi

# 2. Warp for HIP, built against the host ROCm (/opt/rocm).
mkdir -p "$DEPS"
checkout warp
if [ "$REBUILD_WARP" = 1 ] || [ ! -f "$DEPS/warp/warp/bin/warp.so" ]; then
    echo "== building Warp for gfx1151"
    (cd "$DEPS/warp" && HIP_ARCH=gfx1151 ROCM_PATH=/opt/rocm PYTHON="$(command -v python)" ./build_amd.sh)
fi
python -m pip install -q --no-deps -e "$DEPS/warp"

# 3. microduck_rl's locked dependencies, minus the CUDA-only packages.
checkout microduck_rl
echo "== installing locked dependencies"
python "$ENV_DIR/lock_to_requirements.py" "$DEPS/microduck_rl/uv.lock" \
    --pins "$ENV_DIR/pins.toml" -o "$ENV_DIR/requirements-rocm.txt"
python -m pip install -q --no-deps -r "$ENV_DIR/requirements-rocm.txt"

# Local fixes to locked packages (see patches/README.md); skip any already applied.
SITE="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
for p in "$ENV_DIR"/patches/*/*.patch; do
    [ -e "$p" ] || continue
    if patch -p1 -d "$SITE" -R -s -f --dry-run < "$p" >/dev/null 2>&1; then
        continue
    fi
    echo "== applying $(basename "$(dirname "$p")")/$(basename "$p")"
    patch -p1 -d "$SITE" -s -f < "$p"
done

# 4. BAM from its pinned commit, Pollen's microduck_rl, this package, and pytest.
python -m pip install -q --no-deps "better-actuator-models @ $(pin pip.better-actuator-models)"
python -m pip install -q --no-deps -e "$DEPS/microduck_rl"
python -m pip install -q --no-deps --no-build-isolation -e "$ROOT"
python -m pip install -q pytest

python "$ENV_DIR/check_env.py"
