#!/bin/bash
#
# Run sglang batch inference for RLKV efficiency evaluation.
#
# Usage:
#   bash scripts/bench_sglang.sh [GPU_ID] [BATCH_SIZES]
#
# Examples:
#   bash scripts/bench_sglang.sh 5            # default batch sizes: 1,4,16,32
#   bash scripts/bench_sglang.sh 5 "1 4"      # only batch sizes 1 and 4
#
# Runs Llama-3.1-8B-R1 on math_500 with both full and rlkv methods.

set -e

GPU_ID=${1:-5}
BATCH_SIZES=${2:-"1 4 16 32"}
MODEL="Llama-3.1-8B-R1"
TASK="math_500"
ADAPTER_DIR="head_dist/rlkv/Llama-3.1-8B-R1/llama_lr1e-2_ep2_bs32_reg1e-3_tau0.5"
SPARSITY=0.5
SINK_SIZE=16
RECENT_SIZE=64

# Use HF_HOME for model cache
export HF_HOME=${HF_HOME:-/mnt/raid10/wjdu/huggingface}

LOG_DIR="eval/efficiency/logs/${MODEL}"
mkdir -p "$LOG_DIR"

echo "============================================="
echo "RLKV sglang Efficiency Benchmark"
echo "Model: $MODEL | Task: $TASK | GPU: $GPU_ID"
echo "Batch sizes: $BATCH_SIZES"
echo "============================================="

# --- Full attention baseline ---
for BS in $BATCH_SIZES; do
    echo ""
    echo ">>> full attention, batch_size=$BS"
    CUDA_VISIBLE_DEVICES=$GPU_ID python -u eval/efficiency/pred.py \
        --model "$MODEL" \
        --task "$TASK" \
        --method full \
        --batch-size $BS \
        --is-rerun \
        2>&1 | tee "${LOG_DIR}/${TASK}_full_bs${BS}.log"
done

# --- RLKV head reallocation ---
for BS in $BATCH_SIZES; do
    echo ""
    echo ">>> rlkv (sparsity=$SPARSITY), batch_size=$BS"
    CUDA_VISIBLE_DEVICES=$GPU_ID python -u eval/efficiency/pred.py \
        --model "$MODEL" \
        --task "$TASK" \
        --method rlkv \
        --sparsity $SPARSITY \
        --adapter-load-path "$ADAPTER_DIR" \
        --sink-size $SINK_SIZE \
        --recent-size $RECENT_SIZE \
        --batch-size $BS \
        --is-rerun \
        2>&1 | tee "${LOG_DIR}/${TASK}_rlkv_sp${SPARSITY}_bs${BS}.log"
done

echo ""
echo "============================================="
echo "All benchmarks completed."
echo "Logs saved to: $LOG_DIR"
echo "============================================="
