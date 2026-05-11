#!/bin/bash
#
# Adaptive sink/local sweep for RLKV's lossless compression frontier.
# Loops over all 3 models sequentially. For each (model, sink+local), runs
# one long Python process that:
#   1. full-attention baseline on all benchmarks (1 engine init; cached
#      results from previous runs are reused)
#   2. sparsity walk 0.2 → 0.2+step → ... per benchmark with early stop
#      when accuracy drops more than `--tolerance` (default 3.0) points
#   3. fine-grained backup at (last_sp - 0.1) for any benchmark that broke
#
# Designed to be submitted as a long compute-node job — the model stays in
# GPU memory across all benchmark loops within a sparsity, and across the
# full + RLKV stages. Models are processed serially to share a single GPU.
#
# Aggregate results into the paper table with:
#   python eval/efficiency/tab_sweep.py --models Llama-3.1-8B-R1 Qwen-2.5-7B-R1 Qwen-3-4B-Thinking
#
# Usage:
#   bash scripts/run_sweep_lossless.sh [GPU_ID] [WIN_CONFIG] [SP_START]
#
# Examples:
#   bash scripts/run_sweep_lossless.sh                      # GPU=0, win=16:64, sp_start=0.2
#   bash scripts/run_sweep_lossless.sh 0 64:128
#   bash scripts/run_sweep_lossless.sh 0 128:256 0.4        # resume / skip first
#   # full multi-config sweep:
#   for w in 16:64 64:128 128:256; do
#       bash scripts/run_sweep_lossless.sh 0 "$w"
#   done

set -e

cleanup() {
    echo "Terminating background processes..."
    kill $(jobs -p) 2>/dev/null
    wait
    exit 1
}
trap cleanup SIGINT SIGTERM

GPU_ID=${1:-0}
WIN=${2:-16:64}
SP_START=${3:-0.4}

SINK="${WIN%%:*}"
RECENT="${WIN##*:}"

# Models to sweep (in order). Comment out any you don't want for this run.
MODELS=(
    "Llama-3.1-8B-R1"
    "Qwen-2.5-7B-R1"
    "Qwen-3-4B-Thinking"
)

adapter_dir_for() {
    case "$1" in
        Llama-3.1-8B-R1)      echo "head_dist/rlkv/Llama-3.1-8B-R1/llama_lr1e-2_ep2_bs32_reg1e-3_tau0.5" ;;
        Qwen-2.5-7B-R1)       echo "head_dist/rlkv/Qwen-2.5-7B-R1/qwen_lr1e-2_ep2_bs32_reg1e-3_tau0.55" ;;
        Qwen-3-4B-Thinking)   echo "head_dist/rlkv/Qwen-3-4B-Thinking/qwen3_lr1e-2_ep2_bs32_reg2.5e-3_tau0.5" ;;
        *) echo ""; return 1 ;;
    esac
}

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

export HF_HOME=${HF_HOME:-/mnt/raid10/wjdu/huggingface}

echo "============================================="
echo "RLKV adaptive sink/local sweep (all models)"
echo "GPU:          $GPU_ID"
echo "Sink/Recent:  ${SINK} / ${RECENT}"
echo "Start sp:     $SP_START"
echo "Models:       ${MODELS[*]}"
echo "Tasks:        ${TASKS[*]}"
echo "============================================="

for MODEL in "${MODELS[@]}"; do
    ADAPTER_DIR="$(adapter_dir_for "$MODEL")"
    if [ -z "$ADAPTER_DIR" ]; then
        echo "[skip] $MODEL: no adapter mapping; skipping"
        continue
    fi
    LOG_DIR="eval/efficiency/logs/${MODEL}/sweep_lossless"
    mkdir -p "$LOG_DIR"
    LOG="${LOG_DIR}/${MODEL}_s${SINK}r${RECENT}.log"

    echo ""
    echo ">>> [$MODEL] sink=$SINK recent=$RECENT sp_start=$SP_START → $LOG"
    CUDA_VISIBLE_DEVICES=$GPU_ID python -u eval/efficiency/pred_sweep.py \
        --model "$MODEL" \
        --adapter-load-path "$ADAPTER_DIR" \
        --sink-size "$SINK" --recent-size "$RECENT" \
        --tasks "${TASKS[@]}" \
        --sp-start "$SP_START" \
        2>&1 | tee "$LOG"
done

echo ""
echo "============================================="
echo "Sweep complete across all models. Aggregate with:"
echo "  python eval/efficiency/tab_sweep.py --models ${MODELS[*]}"
echo "============================================="
