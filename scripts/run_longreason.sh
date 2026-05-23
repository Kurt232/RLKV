#!/bin/bash
#
# Run LongReason (64K-input subset, 70K context) sweep for all 3 models
# under the same configuration used in the rebuttal (s16r64, 4 sparsities,
# 3 baselines + RLKV).
#
# Outer loop = model (serial — 70K context is memory-heavy, share GPUs
# across configs within a model rather than across models).
# Inner parallel = (method, sparsity) across 4 GPUs.
#
# Score on the login node afterwards:
#   python -u eval/src/eval.py --model <MODEL> --results_path eval/longreason/pred

set -e

cleanup() {
    echo "Terminating background processes..."
    kill $(jobs -p) 2>/dev/null
    wait
    exit 1
}
trap cleanup SIGINT SIGTERM

root="eval/longreason"
task="longreason"

models=(
    "Llama-3.1-8B-R1"
    "Qwen-2.5-7B-R1"
    "Qwen-3-4B-Thinking"
)

# Per-model RLKV adapter (laser / canonical step185 checkpoints).
rlkv_cfg_for() {
    case "$1" in
        Llama-3.1-8B-R1)
            echo "laser_llama_r_v4__lr1e-2_ep2_bs32_reg1e-3__step185" ;;
        Qwen-2.5-7B-R1)
            echo "laser_qwen_rt_v4_t0.55__lr1e-2_ep2_bs32_reg1e-3__step185" ;;
        Qwen-3-4B-Thinking)
            echo "qwen3_cfg2_mcr16_t0.5__lr1e-2_ep2_bs32_reg2.5e-3__step185" ;;
        *) echo ""; return 1 ;;
    esac
}

sparsities=(0.2 0.4 0.6 0.8)
gpus=(0 1 2 3)

methods=(
    "rlkv"
    "duo_attn"
    "kvzip"
    "rkv"
    # "h2o"      # OOM at 70K context
)

log_dir=$root/logs
mkdir -p $log_dir

for model in "${models[@]}"; do
    echo "============================================="
    echo "[$(date +%H:%M:%S)] LongReason sweep: $model"
    echo "============================================="

    rlkv_cfg="$(rlkv_cfg_for "$model")"
    if [ -z "$rlkv_cfg" ]; then
        echo "[skip] $model: no adapter mapping" >&2
        continue
    fi

    cfgs=(
        "$rlkv_cfg"
        "lr=0.02-reg=0.05-ctx=1000_32000-multi_passkey10"
        "scbench_repoqa-0"
        "lr=0.02-reg=0.05-ctx=1000_32000-multi_passkey10"
        # "lr=0.02-reg=0.05-ctx=1000_32000-multi_passkey10"
    )
    attn_dirs=(
        "head_dist/rlkv"
        "head_dist/duo_attn"
        "head_dist/kvzip"
        "head_dist/duo_attn"
        # "head_dist/duo_attn"
    )

    # --- Full baseline (sparsity-independent; run once per model) ---
    CUDA_VISIBLE_DEVICES=${gpus[0]} bash scripts/longreason.sh \
        "$model" "$task" "" "0.5" "full" \
        > ${log_dir}/${model}_${task}_full.log 2>&1 || true

    # --- 4-way (sparsity, method) sweep across 4 GPUs ---
    for i in "${!sparsities[@]}"; do
        sparsity="${sparsities[i]}"
        gpu="${gpus[i]}"
        (
            for j in "${!methods[@]}"; do
                method="${methods[j]}"
                cfg="${cfgs[j]}"
                attn_dir="${attn_dirs[j]}"
                CUDA_VISIBLE_DEVICES=$gpu bash scripts/longreason.sh \
                    "$model" "$task" "$attn_dir/${model}/${cfg}" \
                    "$sparsity" "$method" \
                    > ${log_dir}/${model}_${task}_${method}_${sparsity}.log 2>&1
            done
        ) &
    done
    wait
    echo "[$(date +%H:%M:%S)] $model: sweep done"

    # Score immediately (CPU-bound, doesn't tie up GPU).
    python -u eval/src/eval.py --model "$model" --results_path "$root/pred" || true
done

echo "============================================="
echo "All LongReason sweeps complete."
echo "============================================="
