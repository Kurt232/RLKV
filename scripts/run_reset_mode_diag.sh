#!/bin/bash
#
# Diagnostic: isolate which sglang state layer is corrupted in cross-task
# RLKV sweeps. Runs the same 2-task / sp=0.6 / 30-sample workload 3 times,
# differing only in inter-task reset behavior:
#
#   none:     shared engine, no reset between tasks      → reproduces bug
#   flush:    shared engine, engine.flush_cache() between → tests allocator
#   shutdown: destroy+recreate engine per task            → known-good
#
# Expected outcomes:
#   - all 3 produce math_500 ≈ full (it's the FIRST task)
#   - none produces aime24 ≪ full        (cross-task bug visible)
#   - shutdown produces aime24 ≈ full    (per-task fix; known-good)
#   - flush is the diagnostic:
#       * if aime24 ≈ full → bug lives in allocator state (full_to_comp_mapping
#         or _comp_free_chunks staleness), fix is to call flush_cache between
#         tasks inside sglang
#       * if aime24 ≪ full → bug lives in req_to_token tensor staleness or
#         HeadReallocAttnBackend's private buffers / CG captures
#
# Usage:
#   bash scripts/run_reset_mode_diag.sh [GPU_ID]
#
# Outputs go to eval/efficiency/pred_sweep_diag/<mode>/<MODEL>/ and the
# final summary is printed to stdout.

set -o pipefail

GPU_ID=${1:-0}
MODEL=${MODEL:-Llama-3.1-8B-R1}
SPARSITY=${SPARSITY:-0.6}
NUM_SAMPLES=${NUM_SAMPLES:-30}
TASKS=(math_500 aime24)
ADAPTER_DIR="head_dist/rlkv/Llama-3.1-8B-R1/llama_lr1e-2_ep2_bs32_reg1e-3_tau0.5"
DIAG_ROOT="eval/efficiency/pred_sweep_diag"
LOG_DIR="eval/efficiency/logs/${MODEL}/diag"
mkdir -p "$LOG_DIR"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_TOKEN="${HF_TOKEN:-}"
# Default to online via the hf-mirror endpoint (matches run_sweep_lossless.sh).
# Forcing offline here breaks the first run on nodes that haven't cached the
# Kurt232/bench dataset builder yet. Set HF_HUB_OFFLINE=1 explicitly to opt in.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"
: "${HF_DATASETS_CACHE:=/tmp/hf_datasets_$(whoami)}"
mkdir -p "$HF_DATASETS_CACHE"
export HF_DATASETS_CACHE
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::DeprecationWarning}"

PYTHON="${PYTHON:-python}"

run_mode() {
    local mode=$1
    local pred_root="${DIAG_ROOT}/${mode}"
    local log="${LOG_DIR}/${mode}.log"
    rm -rf "$pred_root"
    mkdir -p "$pred_root"
    echo ""
    echo "=============================================="
    echo ">>> reset-mode=${mode}"
    echo "    pred_root=$pred_root"
    echo "    log=$log"
    echo "=============================================="
    CUDA_VISIBLE_DEVICES=$GPU_ID "$PYTHON" -u eval/efficiency/pred_sweep.py \
        --model "$MODEL" \
        --adapter-load-path "$ADAPTER_DIR" \
        --sink-size 16 --recent-size 64 \
        --tasks "${TASKS[@]}" \
        --sparsity "$SPARSITY" \
        --num-samples "$NUM_SAMPLES" \
        --pred-root "$pred_root" \
        --reset-mode "$mode" \
        --is-rerun \
        2>&1 | tee "$log"
}

score_mode() {
    local mode=$1
    local pred_root="${DIAG_ROOT}/${mode}"
    echo ""
    echo "=============================================="
    echo ">>> scoring reset-mode=${mode}"
    echo "=============================================="
    "$PYTHON" -u eval/src/eval.py \
        --model "$MODEL" \
        --results_path "$pred_root" \
        2>&1 | tee "${LOG_DIR}/${mode}.eval.log"
}

# Run all three modes (sequential, single GPU).
for mode in none flush shutdown; do
    run_mode "$mode"
done

# Score each.
for mode in none flush shutdown; do
    score_mode "$mode"
done

# Print compact comparison table by parsing result.json.
echo ""
echo "=============================================================="
echo "Reset-mode diagnostic summary (sp=${SPARSITY}, n=${NUM_SAMPLES})"
echo "=============================================================="
"$PYTHON" - <<'PY'
import json, os, sys
modes = ["none", "flush", "shutdown"]
tasks = ["math_500", "aime24"]
model = os.environ.get("MODEL", "Llama-3.1-8B-R1")
diag_root = "eval/efficiency/pred_sweep_diag"
rows = {}
for mode in modes:
    p = f"{diag_root}/{mode}/{model}/result.json"
    if not os.path.exists(p):
        rows[mode] = {t: None for t in tasks}
        continue
    scores = json.load(open(p))
    rows[mode] = {}
    for t in tasks:
        # match any jsonl that starts with the task name (rlkv-... variants)
        match = [v for k, v in scores.items() if k.startswith(f"{t}-")]
        rows[mode][t] = match[0] if match else None

# Header
col_w = 14
print(f"{'task':<14}" + "".join(f"{m:<{col_w}}" for m in modes))
print("-" * (14 + col_w * len(modes)))
for t in tasks:
    line = f"{t:<14}"
    for mode in modes:
        v = rows[mode][t]
        line += f"{v if v is not None else '—':<{col_w}}"
    print(line)

print()
none_aime    = rows["none"]["aime24"]
flush_aime   = rows["flush"]["aime24"]
shutdown_aime = rows["shutdown"]["aime24"]
if None in (none_aime, flush_aime, shutdown_aime):
    print("[diag] one or more runs missing; cannot interpret.")
    sys.exit(0)
shutdown_band = shutdown_aime * 0.85  # within 15% of shutdown counts as recovered
print(f"shutdown aime24 (known-good): {shutdown_aime}")
print(f"none aime24 (bug baseline):   {none_aime}")
print(f"flush aime24 (test):          {flush_aime}")
print()
if flush_aime >= shutdown_band:
    print("[verdict] flush_cache RECOVERS aime24 → bug lives in allocator state")
    print("          (likely full_to_comp_mapping / _comp_free_chunks / tree_cache).")
    print("          Upstream fix: add the missing reset to engine.flush_cache or")
    print("          to a per-generate hook in sglang scheduler.")
elif flush_aime <= none_aime * 1.2:
    print("[verdict] flush_cache does NOT recover aime24 → bug lives in:")
    print("          (a) req_to_token tensor staleness (ReqToTokenPool.free does")
    print("              not zero req_to_token), OR")
    print("          (b) HeadReallocAttnBackend private buffers (_kv_indices_buf,")
    print("              _window_kv_indices_buf, captured CUDA graphs).")
    print("          Next step: add req_to_token.zero_() to flush_cache and re-test.")
else:
    print(f"[verdict] partial recovery (flush {flush_aime} between none {none_aime}")
    print(f"          and shutdown {shutdown_aime}) → bug is multi-layer.")
PY
