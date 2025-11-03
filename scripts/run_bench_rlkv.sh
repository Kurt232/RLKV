#!/bin/bash

set -e

cleanup() {
    echo "Terminating background processes..."
    kill $(jobs -p) 2>/dev/null
    wait
    exit 1
}
trap cleanup SIGINT SIGTERM

root="eval/bench"
# model="Llama-3.1-8B-R1"
# cfgs=(
#     llama_lr1e-2_ep2_bs32_reg1e-3_tau0.5
# )
# model="Qwen-2.5-7B-R1"
# cfgs=(
#     qwen_lr1e-2_ep2_bs32_reg1e-3_tau0.55
# )
model="Qwen-3-4B-Thinking"
cfgs=(
    qwen3_lr1e-2_ep2_bs32_reg2.5e-3_tau0.5
)

log_dir=$root/logs/$model
mkdir -p $log_dir

tasks=(
    "math_500"
    "gsm8k"
    "aime24"
    "mbpp"
)
sparsities=(0.2 0.4 0.6 0.8)
gpus=(4 5 6 7)
method="rlkv"
for i in "${!sparsities[@]}"; do
    sparsity="${sparsities[i]}"
    gpu="${gpus[i]}"
    (   
        for cfg in "${cfgs[@]}"; do
            for task in "${tasks[@]}"; do
                CUDA_VISIBLE_DEVICES=$gpu bash scripts/bench.sh "$model" "$task" "head_dist/rlkv/${model}/${cfg}" "$sparsity" "$method" > ${log_dir}/${model}_${task}_${method}_${sparsity}.log
            done
        done
    ) &
done

wait
echo "Benchmarks completed."

python -u eval/src/eval.py --model $model --results_path $root/pred

wait
