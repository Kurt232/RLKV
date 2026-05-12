#!/bin/bash
#
# Sink/local sweep for RLKV's lossless compression frontier.
# One invocation = one (GPU, MODEL, sink+local), looping over a sparsity
# list. Each sparsity = one fresh pred_sweep.py process = one engine init.
# Cached (model, sp, task) outputs are skipped, so re-runs are cheap.
#
# Aggregate results into the paper table with:
#   python eval/efficiency/tab_sweep.py --models Llama-3.1-8B-R1 Qwen-2.5-7B-R1 Qwen-3-4B-Thinking
#
# Usage:
#   bash scripts/run_sweep_lossless.sh [GPU_ID] [MODEL] [WIN_CONFIG] [SP_LIST]
#
# SP_LIST is a space-separated list (quote it). Default: "0.0 0.4 0.6 0.8".
# 0.0 = full-attention baseline (no adapter); >0 = RLKV at that sparsity.
#
# Examples:
#   bash scripts/run_sweep_lossless.sh 0 Llama-3.1-8B-R1                            # default sparsities
#   bash scripts/run_sweep_lossless.sh 1 Qwen-2.5-7B-R1 64:128
#   bash scripts/run_sweep_lossless.sh 2 Qwen-3-4B-Thinking 128:256 "0.4 0.5 0.6"   # custom list
#
# Fan-out across 3 GPUs (one model per GPU, run in parallel):
#   bash scripts/run_sweep_lossless.sh 0 Llama-3.1-8B-R1     16:64 &
#   bash scripts/run_sweep_lossless.sh 1 Qwen-2.5-7B-R1      16:64 &
#   bash scripts/run_sweep_lossless.sh 2 Qwen-3-4B-Thinking  16:64 &
#   wait

set -o pipefail

cleanup() {
    echo "Terminating background processes..."
    kill $(jobs -p) 2>/dev/null
    wait
    exit 1
}
trap cleanup SIGINT SIGTERM

GPU_ID=${1:-0}
MODEL=${2:-Llama-3.1-8B-R1}
WIN=${3:-16:64}
SP_LIST=${4:-"0.0 0.4 0.6"}

SINK="${WIN%%:*}"
RECENT="${WIN##*:}"

export HF_HOME=${HF_HOME:-/mnt/afs/duwenjie/.cache/huggingface}
# HF_TOKEN must come from the environment (e.g. ~/.bashrc or
# `huggingface-cli login`). Don't hardcode it here — this file is in git.
if [ -z "$HF_TOKEN" ]; then
    echo "[warn] HF_TOKEN not set; gated models/datasets will 401" >&2
fi

# Compute node can't reach huggingface.co cleanly (firewall + MITM cert).
# Route through the hf-mirror.com endpoint, which most CN HPC clusters
# allow and which has a valid cert.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# Datasets cache must live on a filesystem with working flock. quarkfs
# (where $HF_HOME lives) returns ENOENT on LOCK_UN release, breaking HF
# datasets. Point datasets cache at node-local /tmp (small footprint, fine
# to re-download per node). Model weights still cache under $HF_HOME.
: "${HF_DATASETS_CACHE:=/tmp/hf_datasets_$(whoami)}"
mkdir -p "$HF_DATASETS_CACHE"
export HF_DATASETS_CACHE

case "$MODEL" in
    Llama-3.1-8B-R1)
        ADAPTER_DIR="head_dist/rlkv/Llama-3.1-8B-R1/llama_lr1e-2_ep2_bs32_reg1e-3_tau0.5"
        ;;
    Qwen-2.5-7B-R1)
        ADAPTER_DIR="head_dist/rlkv/Qwen-2.5-7B-R1/qwen_lr1e-2_ep2_bs32_reg1e-3_tau0.55"
        ;;
    Qwen-3-4B-Thinking)
        ADAPTER_DIR="head_dist/rlkv/Qwen-3-4B-Thinking/qwen3_lr1e-2_ep2_bs32_reg2.5e-3_tau0.5"
        ;;
    *)
        echo "Unknown model: $MODEL"
        exit 1
        ;;
esac

TASKS=(
    "math_500"
    "aime24"
    "gsm8k"
    "mbpp"
    "mmlu_pro_che"
    "mmlu_pro_com"
    "mmlu_pro_law"
    "mmlu_pro_phy"
)

: "${HF_HOME:=$HOME/.cache/huggingface}"
export HF_HOME

# Pin to the rlkv-eval env's python so this runs without `conda activate`
# (e.g. on compute nodes where conda isn't on PATH). Override via env var:
#   PYTHON=/path/to/other/python bash scripts/run_sweep_lossless.sh ...
PYTHON="${PYTHON:-/mnt/afs/duwenjie/.conda/envs/rlkv-eval/bin/python}"
if [ ! -x "$PYTHON" ]; then
    echo "[error] python binary not found: $PYTHON" >&2
    echo "        run scripts/setup_eval_env.sh first, or override with PYTHON=..." >&2
    exit 1
fi

LOG_DIR="eval/efficiency/logs/${MODEL}/sweep_lossless"
mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/${MODEL}_s${SINK}r${RECENT}.log"

echo "============================================="
echo "RLKV sink/local sweep"
echo "GPU:          $GPU_ID"
echo "Model:        $MODEL"
echo "Sink/Recent:  ${SINK} / ${RECENT}"
echo "Adapter:      $ADAPTER_DIR"
echo "Tasks:        ${TASKS[*]}"
echo "Sparsities:   $SP_LIST"
echo "Log dir:      $LOG_DIR"
echo "Python:       $PYTHON"
echo "============================================="

for sp in $SP_LIST; do
    LOG="${LOG_DIR}/${MODEL}_s${SINK}r${RECENT}_sp${sp}.log"
    echo ""
    echo ">>> [$MODEL] sp=$sp → $LOG"

    # Adapter is only required when sp > 0. Pass it always; pred_sweep.py
    # ignores it for sp=0.0 (method=full).
    CUDA_VISIBLE_DEVICES=$GPU_ID "$PYTHON" -u eval/efficiency/pred_sweep.py \
        --model "$MODEL" \
        --adapter-load-path "$ADAPTER_DIR" \
        --sink-size "$SINK" --recent-size "$RECENT" \
        --tasks "${TASKS[@]}" \
        --sparsity "$sp" \
        2>&1 | tee "$LOG"
done

# Score all jsonl outputs at once. Reads eval/efficiency/pred_sweep/<MODEL>/
# and writes result.json / result_lengths.json / result_evals.json there.
EVAL_LOG="${LOG_DIR}/${MODEL}_s${SINK}r${RECENT}_eval.log"
echo ""
echo ">>> [$MODEL] scoring → $EVAL_LOG"
"$PYTHON" -u eval/src/eval.py \
    --model "$MODEL" \
    --results_path eval/efficiency/pred_sweep \
    2>&1 | tee "$EVAL_LOG"

echo ""
echo "============================================="
echo "Sweep complete for ($MODEL, sink=$SINK, recent=$RECENT) over sp={$SP_LIST}."
echo "Per-file scores: eval/efficiency/pred_sweep/${MODEL}/result.json"
echo "Aggregate to paper table with:"
echo "  python eval/efficiency/tab_sweep.py --models <models...>"
echo "============================================="
