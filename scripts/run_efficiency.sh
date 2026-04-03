#!/bin/bash
#
# Run RLKV efficiency evaluation: full baseline + multiple sparsities in parallel.
#
# Usage:
#   bash scripts/run_efficiency.sh [GPU_IDs...]
#
# Example:
#   bash scripts/run_efficiency.sh 4 5 6 7    # 4 GPUs for sp=0.2,0.4,0.6,0.8
#   bash scripts/run_efficiency.sh 5 6        # 2 GPUs for sp=0.2,0.4 (first N sparsities)

set -e

cleanup() {
    echo "Terminating background processes..."
    kill $(jobs -p) 2>/dev/null
    wait
    exit 1
}
trap cleanup SIGINT SIGTERM

MODEL="Llama-3.1-8B-R1"
TASK="math_500"
ADAPTER_DIR="head_dist/rlkv/Llama-3.1-8B-R1/llama_lr1e-2_ep2_bs32_reg1e-3_tau0.5"
MAX_RUNNING_REQUESTS=300
sparsities=(0.2 0.4 0.5 0.6 0.8)

# GPUs from args, default to (4 5 6 7)
if [ $# -gt 0 ]; then
    gpus=("$@")
else
    gpus=(4 5 6 7)
fi

export HF_HOME=${HF_HOME:-/mnt/raid10/wjdu/huggingface}
export PATH="/home/wjdu/miniconda3/envs/rlkv/bin:$PATH"

PRED_DIR="eval/efficiency/pred/${MODEL}"
LOG_DIR="eval/efficiency/logs/${MODEL}"
mkdir -p "$PRED_DIR" "$LOG_DIR"

echo "============================================="
echo "RLKV Efficiency Benchmark"
echo "Model: $MODEL | Task: $TASK"
echo "CUDA Graph: enabled"
echo "GPUs: ${gpus[*]}"
echo "Sparsities: ${sparsities[*]:0:${#gpus[@]}}"
echo "============================================="

# --- Full attention baseline (on first GPU) ---
echo ""
echo ">>> [1] Full attention (GPU ${gpus[0]})"
CUDA_VISIBLE_DEVICES=${gpus[0]} python -u eval/efficiency/pred.py \
    --model "$MODEL" \
    --task "$TASK" \
    --method full \
    --max-running-requests $MAX_RUNNING_REQUESTS \
    --is-rerun \
    2>&1 | tee "${LOG_DIR}/${TASK}-full.log"

# --- RLKV with multiple sparsities in parallel ---
echo ""
echo ">>> [2] RLKV sparsities in parallel..."
for i in "${!gpus[@]}"; do
    if [ "$i" -ge "${#sparsities[@]}" ]; then
        break
    fi
    sparsity="${sparsities[i]}"
    gpu="${gpus[i]}"
    (
        echo ">>> GPU $gpu: RLKV sparsity=$sparsity"
        CUDA_VISIBLE_DEVICES=$gpu python -u eval/efficiency/pred.py \
            --model "$MODEL" \
            --task "$TASK" \
            --method rlkv \
            --sparsity $sparsity \
            --adapter-load-path "$ADAPTER_DIR" \
            --max-running-requests $MAX_RUNNING_REQUESTS \
            2>&1 | tee "${LOG_DIR}/${TASK}-rlkv-sp${sparsity}.log"
    ) &
done

wait

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
