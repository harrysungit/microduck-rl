# Source this file: `source env/activate.sh`
#
# Activates the training venv (once it exists) and keeps every cache the stack
# writes inside .cache/ at the repository root: pip, Warp kernels, packman's
# LLVM toolchain, ROCm comgr/MIOpen, matplotlib, TunableOp.
#
# MICRODUCK_RL_VENV=<dir> uses an existing venv instead of .venv/, and
# MICRODUCK_RL_CACHE=<dir> an existing cache directory instead of .cache/.

MDRL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MDRL_ROOT

export XDG_CACHE_HOME="${MICRODUCK_RL_CACHE:-$MDRL_ROOT/.cache}"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export WARP_CACHE_PATH="$XDG_CACHE_HOME/warp"
export PM_PACKAGES_ROOT="$XDG_CACHE_HOME/packman"
export AMD_COMGR_CACHE_DIR="$XDG_CACHE_HOME/comgr"
export MIOPEN_USER_DB_PATH="$XDG_CACHE_HOME/miopen"
export MIOPEN_CUSTOM_CACHE_DIR="$XDG_CACHE_HOME/miopen"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
mkdir -p "$XDG_CACHE_HOME"

# mjlab trains on the CPU when CUDA_VISIBLE_DEVICES is empty; ROCm honours it.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

# torch's bundled Kineto attaches to rocprofiler-sdk when HIP starts and
# registers an exit handler that calls _exit(0): every process that touched the
# GPU exits 0, failed or not. Off by default; torch.profiler then sees no GPU
# activity, so the profiling scripts turn it back on for themselves.
export ROCPROFILER_REGISTER_ENABLED="${ROCPROFILER_REGISTER_ENABLED:-0}"

# Capture mjlab's simulation graphs on HIP (patches/rocm.py); 0 runs eagerly.
export MICRODUCK_HIP_GRAPHS="${MICRODUCK_HIP_GRAPHS:-1}"

# PyTorch TunableOp: rocBLAS falls back to 32x32 tiles for PPO's GEMM shapes on
# gfx1151, and tuning picks kernels 1.5x faster. Tuned shapes are cached here
# and reused by later runs; new shapes are tuned once, when first seen.
export PYTORCH_TUNABLEOP_ENABLED="${PYTORCH_TUNABLEOP_ENABLED:-1}"
export PYTORCH_TUNABLEOP_TUNING="${PYTORCH_TUNABLEOP_TUNING:-1}"
export PYTORCH_TUNABLEOP_FILENAME="${PYTORCH_TUNABLEOP_FILENAME:-$XDG_CACHE_HOME/tunableop/results%d.csv}"
mkdir -p "$XDG_CACHE_HOME/tunableop"

MDRL_VENV="${MICRODUCK_RL_VENV:-$MDRL_ROOT/.venv}"
if [ -f "$MDRL_VENV/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$MDRL_VENV/bin/activate"
fi
