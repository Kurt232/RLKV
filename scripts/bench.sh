model=$1
task=$2
attn_pattern=$3
sparsity=$4
method=$5
python -u eval/bench/pred.py \
    --model $model --task $task \
    --method $method \
    --sparsity $sparsity \
    --attn_load_dir ${attn_pattern} \
    --sink_size 16 \
    --recent_size 64
