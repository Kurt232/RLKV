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
mkdir -p $root/logs

models=(
    "Llama-3.1-8B-R1"
    "Llama-3.1-8B-Inst"
    "Qwen-2.5-7B-R1"
    "Qwen-2.5-7B-Inst"
    Qwen-3-4B-Instruct
    Qwen-3-4B-Thinking
)
tasks=(
    "mbpp"
    "aime24"
    "gsm8k"
    "math_500"
)

sparsities=(0.2 0.4 0.6 0.8)
gpus=(0 1 2 3)
methods=(
    # "duo_attn"
    # "h2o"
    # "rkv"
)
cfg="lr=0.02-reg=0.05-ctx=1000_32000-multi_passkey10"
for i in "${!sparsities[@]}"; do
    sparsity="${sparsities[i]}"
    gpu="${gpus[i]}"
    (
        for method in "${methods[@]}"; do
            for task in "${tasks[@]}"; do
                for model in "${models[@]}"; do
                    CUDA_VISIBLE_DEVICES=$gpu bash scripts/bench.sh "$model" "$task" "head_dist/duo_attn/${model}/${cfg}" "$sparsity" "$method" > $root/logs/${model}_${task}_${method}_${sparsity}.log
                done
            done
        done
    ) &
done

# for model in "${models[@]}"; do
#     for task in "${tasks[@]}"; do
#         CUDA_VISIBLE_DEVICES=1 bash scripts/bench.sh "$model" "$task" "0" "0" "full" > $root/logs/${model}_${task}_full.log
#     done
# done

wait
echo "Benchmarks completed."

for model in "${models[@]}"; do
    python -u eval/src/eval.py --model $model --results_path $root/pred
done

wait
