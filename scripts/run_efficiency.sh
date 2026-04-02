#!/bin/bash
#
# Run RLKV efficiency evaluation: full vs rlkv sp=0.5, 100 samples, CUDA graph enabled.
#
# Usage:
#   bash scripts/run_efficiency.sh [GPU_ID]
#
# Example:
#   bash scripts/run_efficiency.sh 5

set -e

GPU_ID=${1:-5}
MODEL="Llama-3.1-8B-R1"
TASK="math_500"
ADAPTER_DIR="head_dist/rlkv/Llama-3.1-8B-R1/llama_lr1e-2_ep2_bs32_reg1e-3_tau0.5"
SPARSITY=0.5
NUM_SAMPLES=100

export HF_HOME=${HF_HOME:-/mnt/raid10/wjdu/huggingface}
export PATH="/home/wjdu/miniconda3/envs/rlkv/bin:$PATH"

PRED_DIR="eval/efficiency/pred/${MODEL}"
mkdir -p "$PRED_DIR"

echo "============================================="
echo "RLKV Efficiency Benchmark"
echo "Model: $MODEL | Task: $TASK | GPU: $GPU_ID"
echo "Samples: $NUM_SAMPLES | CUDA Graph: enabled"
echo "============================================="

# --- Full attention baseline ---
echo ""
echo ">>> [1/2] Full attention"
CUDA_VISIBLE_DEVICES=$GPU_ID python -u eval/efficiency/pred.py \
    --model "$MODEL" \
    --task "$TASK" \
    --method full \
    --num-samples $NUM_SAMPLES \
    --is-rerun \
    2>&1 | tee "${PRED_DIR}/${TASK}-full.log"

# --- RLKV sp=0.5 ---
echo ""
echo ">>> [2/2] RLKV sparsity=$SPARSITY"
CUDA_VISIBLE_DEVICES=$GPU_ID python -u eval/efficiency/pred.py \
    --model "$MODEL" \
    --task "$TASK" \
    --method rlkv \
    --sparsity $SPARSITY \
    --adapter-load-path "$ADAPTER_DIR" \
    --num-samples $NUM_SAMPLES \
    --is-rerun \
    2>&1 | tee "${PRED_DIR}/${TASK}-rlkv-sp${SPARSITY}.log"

# --- Summary ---
echo ""
echo "============================================="
echo "Done. Results in ${PRED_DIR}/"
echo "============================================="
ls -lh ${PRED_DIR}/${TASK}*.stats.json 2>/dev/null
