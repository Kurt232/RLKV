#!/bin/bash
#
# Run RLKV efficiency evaluation: full baseline + multiple sparsities sequentially.
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
cfg="llama_lr1e-2_ep2_bs32_reg1e-3_tau0.5"
ADAPTER_DIR="head_dist/rlkv/Llama-3.1-8B-R1/${cfg}"
FULL_MAX_RUNNING_REQUESTS=150
sparsities=(      0.2  0.4  0.6  0.8)
max_running_reqs=(200  250  375  750)

export HF_HOME=${HF_HOME:-/mnt/raid10/wjdu/huggingface}
export PATH="/home/wjdu/miniconda3/envs/rlkv/bin:$PATH"

PRED_DIR="eval/efficiency/pred/${MODEL}"
LOG_DIR="eval/efficiency/logs/${MODEL}"
mkdir -p "$PRED_DIR" "$LOG_DIR"

echo "============================================="
echo "RLKV Efficiency Benchmark"
echo "Model: $MODEL | Task: $TASK | GPU: $GPU_ID"
echo "CUDA Graph: enabled"
echo "Sparsities: ${sparsities[*]}"
echo "============================================="

--- Full attention baseline ---
echo ""
echo ">>> [1/$((${#sparsities[@]}+1))] Full attention"
CUDA_VISIBLE_DEVICES=$GPU_ID python -u eval/efficiency/pred.py \
    --model "$MODEL" \
    --task "$TASK" \
    --method full \
    --max-running-requests $FULL_MAX_RUNNING_REQUESTS \
    2>&1 | tee "${LOG_DIR}/${TASK}-full.log"

# --- RLKV with multiple sparsities sequentially ---
for i in "${!sparsities[@]}"; do
    sparsity="${sparsities[i]}"
    mr="${max_running_reqs[i]}"
    echo ""
    echo ">>> [$((i+2))/$((${#sparsities[@]}+1))] RLKV sparsity=$sparsity (max_running=$mr)"
    CUDA_VISIBLE_DEVICES=$GPU_ID python -u eval/efficiency/pred.py \
        --model "$MODEL" \
        --task "$TASK" \
        --method rlkv \
        --sparsity $sparsity \
        --adapter-load-path "$ADAPTER_DIR" \
        --max-running-requests $mr \
        2>&1 | tee "${LOG_DIR}/${TASK}-rlkv-sp${sparsity}.log"
done

# --- Eval ---
echo ""
echo ">>> Evaluating results..."
python -u eval/src/eval.py --model "$MODEL" --results_path eval/efficiency/pred

# --- Summary ---
echo ""
echo "============================================="
echo "Done. Results in ${PRED_DIR}/"
echo "============================================="
ls -lh ${PRED_DIR}/${TASK}*.stats.json 2>/dev/null
