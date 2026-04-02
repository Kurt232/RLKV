"""
Batch inference with sglang Engine for RLKV efficiency evaluation.

Supports:
- full: standard full attention (baseline)
- rlkv: head reallocation with dual KV pool

Usage:
    python eval/efficiency/pred.py \
        --model Llama-3.1-8B-R1 \
        --task math_500 \
        --method rlkv \
        --sparsity 0.5 \
        --adapter-load-path head_dist/rlkv/Llama-3.1-8B-R1/... \
        --batch-size 32
"""

# Must set event loop before uvloop patches asyncio
import asyncio
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

import argparse
import dataclasses
import json
import os
import subprocess
import time

import torch
import sglang as sgl
from datasets import load_dataset
from sglang.srt.server_args import ServerArgs
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--method", type=str, default="full", choices=["full", "rlkv"])
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--adapter-load-path", type=str, default=None)
    parser.add_argument("--sink-size", type=int, default=16)
    parser.add_argument("--recent-size", type=int, default=64)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--model-path", type=str, default=None,
                        help="Override model path (default: from model2path.json)")
    parser.add_argument("--is-rerun", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="Override max new tokens (default: dataset max_len)")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Limit number of samples (default: all)")
    parser.add_argument("--disable-cuda-graph", action="store_true",
                        help="Disable CUDA graph (default: enabled)")
    return parser.parse_args()


def format_prompts(tokenizer, data):
    """Apply chat template to all prompts."""
    formatted = []
    for item in data:
        prompt = item["prompt"]
        if tokenizer.chat_template is not None:
            text = tokenizer.apply_chat_template(
                conversation=[{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
        else:
            text = prompt
        formatted.append(text)
    return formatted


def main():
    args = parse_args()

    # Load configs
    config_dir = os.path.join(os.path.dirname(__file__), "..", "config")
    model2path = json.load(open(os.path.join(config_dir, "model2path.json")))
    dataset2maxlen = json.load(open(os.path.join(config_dir, "dataset2maxlen.json")))

    # HuggingFace model IDs for downloading if local path doesn't exist
    hf_model_ids = {
        "Llama-3.1-8B-R1": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "Qwen-2.5-7B-R1": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "Qwen-3-4B-Thinking": "Qwen/Qwen3-4B",
    }

    model_name = args.model
    model_path = args.model_path or model2path.get(model_name)
    # Fallback to HF model ID if local path doesn't exist
    if model_path and not os.path.exists(model_path):
        hf_id = hf_model_ids.get(model_name)
        if hf_id:
            print(f"Local model path '{model_path}' not found, using HF ID: {hf_id}")
            model_path = hf_id
        else:
            raise FileNotFoundError(
                f"Model path '{model_path}' not found and no HF ID configured for '{model_name}'"
            )
    max_length = dataset2maxlen[args.task]
    max_new_tokens = args.max_new_tokens or max_length

    # Output path
    root = "eval/efficiency/pred"
    os.makedirs(f"{root}/{model_name}", exist_ok=True)
    if args.method == "rlkv":
        adapter_tag = args.adapter_load_path.rstrip("/").split("/")[-1]
        out_path = (
            f"{root}/{model_name}/{args.task}-rlkv-{adapter_tag}"
            f"-sp-{args.sparsity}.jsonl"
        )
    else:
        out_path = (
            f"{root}/{model_name}/{args.task}-full.jsonl"
        )

    if os.path.exists(out_path) and not args.is_rerun:
        print(f"Predictions already exist at {out_path}, skipping...")
        return

    # Load dataset
    data = load_dataset("Kurt232/bench", name=args.task, split="test")
    if args.num_samples is not None:
        data = data.select(range(min(args.num_samples, len(data))))
    print(f"Loaded {len(data)} examples from {args.task}")

    # Format prompts
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    prompts = format_prompts(tokenizer, data)
    print(f"Formatted {len(prompts)} prompts (first prompt length: {len(prompts[0])} chars)")

    # Build sglang ServerArgs
    server_kwargs = dict(
        model_path=model_path,
        tp_size=args.tp_size,
        disable_radix_cache=True,
        context_length=max_length + max_new_tokens,
        mem_fraction_static=0.85,
    )

    if args.method == "rlkv":
        assert args.adapter_load_path is not None, (
            "--adapter-load-path required for rlkv method"
        )
        server_kwargs.update(
            attention_backend="triton",
            enable_rlkv_inference=True,
            rlkv_sparsity=args.sparsity,
            adapter_load_path=args.adapter_load_path,
            sink_window_size=args.sink_size,
            recent_window_size=args.recent_size,
            disable_cuda_graph=args.disable_cuda_graph,
        )
    else:
        # full attention baseline
        server_kwargs.update(
            attention_backend="triton",
            disable_cuda_graph=args.disable_cuda_graph,
        )

    server_args = ServerArgs(**server_kwargs)

    # Create engine
    print(f"Starting sglang engine ({args.method})...")
    engine = sgl.Engine(**dataclasses.asdict(server_args))
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    sampling_params = {
        "max_new_tokens": max_new_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
    }

    # Snapshot GPU memory after engine init (before inference)
    gpu_mem_after_init = {}
    for i in range(torch.cuda.device_count()):
        gpu_mem_after_init[f"gpu_{i}"] = {
            "allocated_MiB": round(torch.cuda.memory_allocated(i) / 1024**2, 1),
            "reserved_MiB": round(torch.cuda.memory_reserved(i) / 1024**2, 1),
            "total_MiB": round(torch.cuda.get_device_properties(i).total_memory / 1024**2, 1),
        }

    # Inference - let sglang handle batching internally
    print(f"Running inference on {len(prompts)} prompts...")
    t_start = time.time()
    outputs = engine.generate(prompts, sampling_params)
    total_time = time.time() - t_start

    total_input_tokens = 0
    total_output_tokens = 0
    preds = []
    for output, item in zip(outputs, data):
        input_tokens = output.get("meta_info", {}).get("prompt_tokens", 0)
        output_tokens = output.get("meta_info", {}).get("completion_tokens", 0)
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        preds.append({
            "prompt": item["prompt"],
            "pred": output["text"],
            "answers": item["answer"],
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        })

    # Snapshot GPU memory after inference
    gpu_mem_after_infer = {}
    for i in range(torch.cuda.device_count()):
        gpu_mem_after_infer[f"gpu_{i}"] = {
            "allocated_MiB": round(torch.cuda.memory_allocated(i) / 1024**2, 1),
            "reserved_MiB": round(torch.cuda.memory_reserved(i) / 1024**2, 1),
            "peak_allocated_MiB": round(torch.cuda.max_memory_allocated(i) / 1024**2, 1),
            "peak_reserved_MiB": round(torch.cuda.max_memory_reserved(i) / 1024**2, 1),
        }

    # nvidia-smi snapshot
    nvidia_smi = {}
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.strip().split("\n"):
            parts = [x.strip() for x in line.split(",")]
            if len(parts) == 4:
                nvidia_smi[f"gpu_{parts[0]}"] = {
                    "mem_used_MiB": int(parts[1]),
                    "mem_total_MiB": int(parts[2]),
                    "gpu_util_pct": int(parts[3]),
                }
    except Exception:
        pass

    # Shutdown engine
    engine.shutdown()

    # Save predictions
    with open(out_path, "w", encoding="utf-8") as f:
        for pred in preds:
            json.dump(pred, f, ensure_ascii=False)
            f.write("\n")

    # Build stats
    output_tokens_list = [p["output_tokens"] for p in preds]
    stats = {
        "config": {
            "model": model_name,
            "task": args.task,
            "method": args.method,
            "sparsity": args.sparsity,
            "max_new_tokens": max_new_tokens,
            "tp_size": args.tp_size,
            "num_samples": len(preds),
            "sink_size": args.sink_size,
            "recent_size": args.recent_size,
            "adapter_load_path": args.adapter_load_path,
        },
        "throughput": {
            "total_time_s": round(total_time, 2),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "avg_gen_throughput_tok_s": round(total_output_tokens / total_time, 1),
            "avg_req_throughput_req_s": round(len(preds) / total_time, 3),
            "avg_latency_per_req_s": round(total_time / len(preds), 2),
        },
        "token_stats": {
            "avg_input_tokens": round(total_input_tokens / len(preds), 1),
            "avg_output_tokens": round(total_output_tokens / len(preds), 1),
            "min_output_tokens": min(output_tokens_list),
            "max_output_tokens": max(output_tokens_list),
            "hit_max_tokens": sum(1 for t in output_tokens_list if t >= max_new_tokens),
        },
        "gpu_memory": {
            "after_init": gpu_mem_after_init,
            "after_inference": gpu_mem_after_infer,
            "nvidia_smi": nvidia_smi,
        },
    }

    # Save stats JSON — same naming as pred file but .stats.json
    stats_path = out_path.replace(".jsonl", ".stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Method: {args.method} | Model: {model_name} | Task: {args.task}")
    print(f"Sparsity: {args.sparsity}")
    print(f"Total examples: {len(preds)}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Gen throughput: {total_output_tokens/total_time:.1f} tok/s")
    print(f"Req throughput: {len(preds)/total_time:.2f} req/s")
    print(f"Total tokens: {total_input_tokens} in / {total_output_tokens} out")
    print(f"Predictions: {out_path}")
    print(f"Stats: {stats_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
