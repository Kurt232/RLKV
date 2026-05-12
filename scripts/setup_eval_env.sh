#!/bin/bash
#
# Minimal install for running scripts/run_sweep_lossless.sh (i.e.
# eval/efficiency/pred_sweep.py via the RLKV-patched SGLang fork).
#
# Targets a python binary directly — no `conda activate` needed, so this
# works on compute nodes where conda isn't on PATH. By default it uses
# /mnt/afs/duwenjie/.conda/envs/rlkv-eval/bin/python (shared FS, pre-created
# with `conda create -n rlkv-eval python=3.12 -y`).
#
# What gets installed (~5-8 GB on disk):
#   - sglang fork in editable mode — its pyproject pins torch==2.8.0 plus
#     transformers 4.56.1, sgl-kernel, flashinfer, triton, datasets, ...
#     so pip pulls everything in one resolve. Default PyPI torch 2.8.0 ships
#     cu128 wheels, which are forward-compatible with the cu12.9 driver.
#   - eval-time extras: latex2sympy2, partial-json-parser
#
# What is NOT installed (training-only / not used by pred_sweep.py):
#   - flash-attn, Block-Sparse-Attention, AReaL, the local `base` package
#
# Idempotent — re-runs are cheap (pip skips already-installed packages).
#
# Usage:
#   bash scripts/setup_eval_env.sh
#   PYTHON=/path/to/other/env/bin/python bash scripts/setup_eval_env.sh
#
# After install, run the sweep with the same python:
#   PYTHON=/mnt/afs/duwenjie/.conda/envs/rlkv-eval/bin/python \
#       bash scripts/run_sweep_lossless.sh 0 Llama-3.1-8B-R1
# or just `conda activate rlkv-eval` if conda is on PATH.

set -e
set -o pipefail

PYTHON="${PYTHON:-/mnt/afs/duwenjie/.conda/envs/rlkv-eval/bin/python}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SGLANG_DIR="${REPO_ROOT}/sglang"

# Disable user-site so ~/.local/lib/python3.12/site-packages/ doesn't
# (a) shadow env packages at runtime, or
# (b) trick pip into reporting "already satisfied" without installing
#     into the env. Affects all child python/pip invocations below.
export PYTHONNOUSERSITE=1

# Clear NGC container's PIP_CONSTRAINT, which pins torch to NGC's
# pre-built nv-torch (e.g. 2.8.0a0+...nv25.6). That version string doesn't
# satisfy sglang's exact `torch==2.8.0` pin → ResolutionImpossible. We
# want the stock PyPI torch 2.8.0 wheel instead.
unset PIP_CONSTRAINT PIP_BUILD_CONSTRAINT

echo "============================================="
echo "RLKV minimal eval install"
echo "Python:      $PYTHON"
echo "Repo:        $REPO_ROOT"
echo "SGLang fork: $SGLANG_DIR"
echo "============================================="

# ---- 0. Sanity checks ----
if [ ! -x "$PYTHON" ]; then
    echo "[error] python binary not found: $PYTHON" >&2
    echo "        create the env first, e.g.:" >&2
    echo "          conda create -n rlkv-eval python=3.12 -y" >&2
    echo "        or override with: PYTHON=/path/to/python bash $0" >&2
    exit 1
fi
echo "[python] $($PYTHON --version 2>&1) at $PYTHON"

if [ ! -f "$SGLANG_DIR/python/pyproject.toml" ]; then
    echo "[error] $SGLANG_DIR/python/pyproject.toml not found." >&2
    echo "        Populate the submodule first (public, no auth):" >&2
    echo "          git -C $REPO_ROOT config submodule.sglang.url https://github.com/Kurt232/rlkv-sglang-v0.5.2.git" >&2
    echo "          git -C $REPO_ROOT submodule update --init --depth 1 sglang" >&2
    exit 1
fi

# ---- 1. Bake user-site isolation into the env (if it's a conda env) ----
# Derive env prefix from the python path: <prefix>/bin/python -> <prefix>.
ENV_PREFIX="$(dirname "$(dirname "$PYTHON")")"
if [ -d "$ENV_PREFIX/conda-meta" ]; then
    mkdir -p "$ENV_PREFIX/etc/conda/activate.d" "$ENV_PREFIX/etc/conda/deactivate.d"
    cat > "$ENV_PREFIX/etc/conda/activate.d/00_no_user_site.sh" <<'EOF'
export PYTHONNOUSERSITE=1
EOF
    cat > "$ENV_PREFIX/etc/conda/deactivate.d/00_no_user_site.sh" <<'EOF'
unset PYTHONNOUSERSITE
EOF
    echo "[isolation] activate hook → $ENV_PREFIX/etc/conda/activate.d/"
fi

PIP="$PYTHON -m pip"

# ---- 2. sglang fork (editable, srt extras) ----
# Pulls torch==2.8.0 + transformers 4.56.1 + sgl-kernel + flashinfer + ...
# in one resolve. No need to pre-pin torch separately.
if $PYTHON -c "
import sys, dataclasses
from sglang.srt.server_args import ServerArgs
fields = {f.name for f in dataclasses.fields(ServerArgs)}
sys.exit(0 if 'enable_rlkv_inference' in fields else 1)
" 2>/dev/null; then
    echo "[sglang] RLKV-patched fork already installed; skipping"
else
    echo "[sglang] installing $SGLANG_DIR/python[srt] (editable)"
    $PIP install -e "$SGLANG_DIR/python[srt]"
fi

# ---- 3. Eval-time extras ----
echo "[extras] latex2sympy2, partial-json-parser"
$PIP install latex2sympy2 partial-json-parser

# ---- 4. Smoke test ----
echo ""
echo "============================================="
echo "Smoke test"
echo "============================================="
REPO_ROOT="$REPO_ROOT" $PYTHON - <<'PY'
import importlib, dataclasses, os, sys
mods = ["torch", "transformers", "datasets", "sglang", "sympy",
        "latex2sympy2", "partial_json_parser"]
for m in mods:
    try:
        v = getattr(importlib.import_module(m), "__version__", "?")
        print(f"  {m:25s} {v}")
    except Exception as e:
        print(f"  {m:25s} FAIL: {e}")
        sys.exit(1)

from sglang.srt.server_args import ServerArgs
fields = {f.name for f in dataclasses.fields(ServerArgs)}
needed = ["enable_rlkv_inference", "rlkv_sparsity", "adapter_load_path",
          "sink_window_size", "recent_window_size"]
missing = [n for n in needed if n not in fields]
if missing:
    print(f"\n[FAIL] RLKV server args missing: {missing}")
    sys.exit(1)
print("\n[OK] RLKV server args present:", needed)

repo_root = os.environ["REPO_ROOT"]
sys.path.insert(0, os.path.join(repo_root, "eval", "src"))
import metrics  # noqa: F401
print("[OK] eval/src/metrics.py imports clean")
PY

echo ""
echo "============================================="
echo "Done. Run the sweep with the same python:"
echo "  PYTHON=$PYTHON bash scripts/run_sweep_lossless.sh 0 Llama-3.1-8B-R1"
echo "============================================="
