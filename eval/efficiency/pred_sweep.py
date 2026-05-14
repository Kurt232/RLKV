"""
Single-sparsity benchmark runner on SGLang.

One invocation = one engine init at one sparsity. sparsity=0.0 means the
full-attention baseline (no adapter needed); sparsity>0 means RLKV mode
with the supplied adapter. All tasks listed in --tasks are batched into one
engine.generate() call. Per-task outputs are written to:

  pred/{model}/{task}-full.jsonl
  pred/{model}/{task}-rlkv-{adapter_tag}-s{sink}r{recent}-sp-{sp}.jsonl

Already-existing (model, sp, task) outputs are skipped unless --is-rerun
is set. Scoring is offline: run eval/src/eval.py separately.

Loop multiple sparsities from the shell (each call re-inits the engine —
which is necessary because RLKV's dual KV pool sizing is baked at init).
scripts/run_sweep_lossless.sh does this for you.

Usage:
  python eval/efficiency/pred_sweep.py \\
      --model Llama-3.1-8B-R1 \\
      --tasks math_500 aime24 gsm8k mbpp mmlu_pro_che mmlu_pro_com mmlu_pro_law mmlu_pro_phy \\
      --sparsity 0.0                                                # full baseline

  python eval/efficiency/pred_sweep.py \\
      --model Llama-3.1-8B-R1 \\
      --adapter-load-path head_dist/rlkv/Llama-3.1-8B-R1/llama_lr1e-2_ep2_bs32_reg1e-3_tau0.5 \\
      --tasks math_500 aime24 gsm8k mbpp mmlu_pro_che mmlu_pro_com mmlu_pro_law mmlu_pro_phy \\
      --sparsity 0.4                                                # one RLKV sp
"""

# Must set event loop before uvloop patches asyncio.
import asyncio
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

import argparse
import dataclasses
import gc
import json
import os
import sys
import time
from collections import defaultdict
from typing import Optional

import torch
import sglang as sgl
from datasets import load_dataset
from sglang.srt.server_args import ServerArgs
from transformers import AutoTokenizer

SUPPORTED_TASKS = {
    "gsm8k", "math_500", "aime24", "mbpp",
    "mmlu_pro_che", "mmlu_pro_com", "mmlu_pro_law", "mmlu_pro_phy",
}

HF_MODEL_IDS = {
    "Llama-3.1-8B-R1": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "Qwen-2.5-7B-R1":  "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "Qwen-3-4B-Thinking": "Qwen/Qwen3-4B",
}

PRED_ROOT = "eval/efficiency/pred_sweep"

# Sparsity → max_running_requests. Capped so every in-flight decode batch
# fits inside a captured CUDA graph (cuda_graph_max_bs below). Batches that
# overflow fall back to eager kernels, which under RLKV compression have
# been observed to push the model into degenerate repetition loops
# ("So the picketers are not allowed..." repeated to max_new_tokens) —
# i.e. CUDA-graph capture appears to be load-bearing for numerical
# behavior, not just performance. Keep max_running ≤ CUDA_GRAPH_MAX_BS.
# Higher sparsity shrinks the comp KV pool and frees memory, so we can
# afford more concurrency, but never above the graph cap.
# For sparsities not in the table, we use the largest table key ≤ the
# requested sparsity (i.e. the conservative side).
MAX_REQ_BY_SPARSITY = {
    0.0: 128,    # full attention baseline
    0.1: 128,
    0.2: 160,
    0.3: 192,
    0.4: 224,
    0.5: 256,
    0.6: 256,
    0.7: 256,
    0.8: 256,
}

# CUDA graph batch-size cap. Set to the largest max_running_requests in
# the table above so all decode batches replay from graph. Bumping this
# costs capture time + memory; lowering it risks eager fallback.
CUDA_GRAPH_MAX_BS = 256


def max_running_requests_for(sparsity: float, fallback: int) -> int:
    """Floor-lookup into MAX_REQ_BY_SPARSITY. `fallback` is used only when
    the table is empty (i.e. someone wiped it) or sparsity is below every
    table key — which shouldn't happen with sparsity=0.0 in the table."""
    if not MAX_REQ_BY_SPARSITY:
        return fallback
    eligible = [s for s in MAX_REQ_BY_SPARSITY if s <= sparsity + 1e-9]
    if not eligible:
        return fallback
    return MAX_REQ_BY_SPARSITY[max(eligible)]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--adapter-load-path", default=None,
                   help="RLKV adapter dir (required when --sparsity > 0)")
    p.add_argument("--sink-size", type=int, default=16)
    p.add_argument("--recent-size", type=int, default=64)
    p.add_argument("--tasks", nargs="+", required=True,
                   help="Benchmark task names; must be in SUPPORTED_TASKS")
    p.add_argument("--sparsity", type=float, default=0.0,
                   help="Single sparsity to run. 0.0 → full attention baseline "
                        "(adapter not needed); >0 → RLKV at that sparsity.")
    p.add_argument("--max-running-requests", type=int, default=300,
                   help="Fallback when --sparsity is not in MAX_REQ_BY_SPARSITY")
    p.add_argument("--tp-size", type=int, default=1)
    p.add_argument("--model-path", type=str, default=None)
    p.add_argument("--num-samples", type=int, default=None,
                   help="Limit number of samples per task (for debugging)")
    p.add_argument("--disable-cuda-graph", action="store_true")
    p.add_argument("--is-rerun", action="store_true",
                   help="Re-run even when an output jsonl already exists")
    args = p.parse_args()
    if args.sparsity > 0 and not args.adapter_load_path:
        p.error("--adapter-load-path is required when --sparsity > 0")
    return args


def resolve_model_path(model_name: str, override: Optional[str]) -> str:
    config_dir = os.path.join(os.path.dirname(__file__), "..", "config")
    model2path = json.load(open(os.path.join(config_dir, "model2path.json")))
    path = override or model2path.get(model_name)
    if path and not os.path.exists(path):
        hf_id = HF_MODEL_IDS.get(model_name)
        if not hf_id:
            raise FileNotFoundError(f"No local path or HF ID for {model_name}")
        print(f"[setup] local path '{path}' missing, falling back to HF id {hf_id}")
        path = hf_id
    return path


def load_task_data(task: str, num_samples: Optional[int]):
    if task.startswith("mmlu_pro"):
        data = load_dataset(f"Kurt232/{task}", split="test")
        if len(data) > 500:
            data = data.shuffle(seed=42).take(500)
    else:
        data = load_dataset("Kurt232/bench", name=task, split="test")
    if num_samples is not None:
        data = data.select(range(min(num_samples, len(data))))
    return data


def format_prompts(tokenizer, data):
    out = []
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
        out.append(text)
    return out


def build_engine(model_path: str, *, method: str, args, sparsity: float = 0.0,
                 max_total_len: int):
    """Construct an SGLang Engine for either `full` or `rlkv` mode."""
    # Concurrency budget is sparsity-dependent (more sparsity → more headroom
    # for concurrent reqs). Look up the table; the CLI arg is the fallback.
    max_running = max_running_requests_for(sparsity, args.max_running_requests)
    print(f"[engine] max_running_requests={max_running} (sparsity={sparsity})")
    kwargs = dict(
        model_path=model_path,
        tp_size=args.tp_size,
        disable_radix_cache=True,
        context_length=max_total_len,
        mem_fraction_static=0.85,
        max_running_requests=max_running,
        attention_backend="triton",
        disable_cuda_graph=args.disable_cuda_graph,
        cuda_graph_max_bs=CUDA_GRAPH_MAX_BS,
    )
    if method == "rlkv":
        kwargs.update(
            enable_rlkv_inference=True,
            rlkv_sparsity=sparsity,
            adapter_load_path=args.adapter_load_path,
            sink_window_size=args.sink_size,
            recent_window_size=args.recent_size,
        )
    server_args = ServerArgs(**kwargs)
    print(f"[engine] starting ({method}, sparsity={sparsity})")
    return sgl.Engine(**dataclasses.asdict(server_args))


def out_path_for(model: str, task: str, *, method: str, adapter_tag: str,
                 sink: int, recent: int, sparsity: float) -> str:
    """Mirror eval/efficiency/pred.py naming so tab_sweep.py picks them up."""
    root = f"{PRED_ROOT}/{model}"
    os.makedirs(root, exist_ok=True)
    if method == "full":
        return f"{root}/{task}-full.jsonl"
    return (f"{root}/{task}-rlkv-{adapter_tag}"
            f"-s{sink}r{recent}-sp-{sparsity}.jsonl")


def run_batched(engine, prompts_per_task: dict[str, list[str]],
                max_new_tokens_per_task: dict[str, int]) -> dict[str, list[dict]]:
    """Run all (task, prompt) pairs in one engine.generate call, return per-task outputs."""
    # Build flat list, track origin.
    flat_prompts = []
    flat_max_new = []
    origins = []  # (task, idx_within_task)
    for task, prompts in prompts_per_task.items():
        for i, p in enumerate(prompts):
            flat_prompts.append(p)
            flat_max_new.append(max_new_tokens_per_task[task])
            origins.append((task, i))

    if not flat_prompts:
        return {}

    # SGLang accepts per-prompt sampling params as a list.
    sampling_params = [
        {"max_new_tokens": mn, "temperature": 0.0, "top_p": 1.0}
        for mn in flat_max_new
    ]

    print(f"[generate] {len(flat_prompts)} prompts across {len(prompts_per_task)} tasks")
    t0 = time.time()
    outputs = engine.generate(flat_prompts, sampling_params)
    elapsed = time.time() - t0
    print(f"[generate] done in {elapsed:.1f}s")

    grouped: dict[str, list[dict]] = defaultdict(list)
    for (task, _), out in zip(origins, outputs):
        grouped[task].append(out)
    return dict(grouped), elapsed


def write_jsonl(*, data, outputs, jsonl_path: str):
    """Write per-prompt records as JSONL. No stats sidecar."""
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for out, item in zip(outputs, data):
            in_tok = out.get("meta_info", {}).get("prompt_tokens", 0)
            out_tok = out.get("meta_info", {}).get("completion_tokens", 0)
            rec = {
                "prompt": item["prompt"],
                "pred": out["text"],
                "answers": item["answer"],
                # Field names eval/src/eval.py expects.
                "input_length": in_tok,
                "output_length": out_tok,
                # SGLang stops on EOS or max_new_tokens; no repeat-window
                # early stop like pred.py. Emit constant False so eval.py's
                # strict `data["is_early_stop"]` lookup doesn't KeyError.
                "is_early_stop": False,
            }
            json.dump(rec, f, ensure_ascii=False)
            f.write("\n")


def shutdown(engine):
    try:
        engine.shutdown()
    except Exception as e:
        print(f"[engine] shutdown warning: {e}")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_at_sparsity(*, args, model_path, adapter_tag,
                    sparsity, method, active_tasks, all_data, all_prompts,
                    max_total_len, max_new_tokens_per_task) -> list[str]:
    """Run one engine init at the given sparsity over all `active_tasks`.
    Writes per-task jsonl + stats; scoring is offline (eval/src/eval.py).
    Returns the list of task names actually written."""
    if not active_tasks:
        return []

    engine = build_engine(model_path, method=method, args=args,
                          sparsity=sparsity, max_total_len=max_total_len)
    # sglang Engine init can clear/replace the main-thread event loop (uvloop
    # takes over). Re-bootstrap before any engine.generate() call. Same
    # workaround as in eval/efficiency/pred.py.
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
    try:
        # Warmup so triton JIT happens once, not amortized into our timing.
        engine.generate("Hello", {"max_new_tokens": 1, "temperature": 0.0})

        prompts_per_task = {t: all_prompts[t] for t in active_tasks}
        outputs_by_task, total_time = run_batched(
            engine, prompts_per_task, max_new_tokens_per_task,
        )

        written: list[str] = []
        for task in active_tasks:
            outs = outputs_by_task.get(task, [])
            if not outs:
                continue
            jsonl_path = out_path_for(
                args.model, task, method=method,
                adapter_tag=adapter_tag,
                sink=args.sink_size, recent=args.recent_size,
                sparsity=sparsity,
            )
            write_jsonl(data=all_data[task], outputs=outs, jsonl_path=jsonl_path)
            print(f"[write] {task} method={method} sp={sparsity} → {jsonl_path}")
            written.append(task)
        return written
    finally:
        shutdown(engine)


def main():
    args = parse_args()
    for t in args.tasks:
        assert t in SUPPORTED_TASKS, f"unknown task: {t}"

    sparsity = round(args.sparsity, 2)
    method = "full" if sparsity == 0.0 else "rlkv"
    adapter_tag = (args.adapter_load_path.rstrip("/").split("/")[-1]
                   if args.adapter_load_path else "none")

    model_path = resolve_model_path(args.model, args.model_path)

    # Load configs (max-length per task).
    config_dir = os.path.join(os.path.dirname(__file__), "..", "config")
    dataset2maxlen = json.load(open(os.path.join(config_dir, "dataset2maxlen.json")))
    max_new_tokens_per_task = {t: dataset2maxlen[t] for t in args.tasks}
    max_total_len = max(max_new_tokens_per_task.values()) * 2

    # Cache check: skip tasks whose jsonl already exists (unless --is-rerun).
    tasks_to_run: list[str] = []
    cached: list[str] = []
    for t in args.tasks:
        jsonl_path = out_path_for(
            args.model, t, method=method, adapter_tag=adapter_tag,
            sink=args.sink_size, recent=args.recent_size, sparsity=sparsity,
        )
        if not args.is_rerun and os.path.exists(jsonl_path):
            cached.append(t)
            print(f"[cache] {t} sp={sparsity} → reusing {jsonl_path}")
        else:
            tasks_to_run.append(t)

    if not tasks_to_run:
        print(f"\n[done] all tasks already have outputs at sp={sparsity}; "
              f"nothing to run. Score with eval/src/eval.py.")
        return

    # Load datasets and tokenize prompts for the tasks we actually need.
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    all_data = {t: load_task_data(t, args.num_samples) for t in tasks_to_run}
    all_prompts = {t: format_prompts(tokenizer, all_data[t]) for t in tasks_to_run}
    for t in tasks_to_run:
        print(f"[data] {t}: {len(all_data[t])} samples, "
              f"max_new_tokens={max_new_tokens_per_task[t]}")

    print(f"\n{'='*60}\n[run] sparsity={sparsity} method={method} "
          f"active={tasks_to_run}\n{'='*60}")
    run_at_sparsity(
        args=args, model_path=model_path, adapter_tag=adapter_tag,
        sparsity=sparsity, method=method, active_tasks=tasks_to_run,
        all_data=all_data, all_prompts=all_prompts,
        max_total_len=max_total_len,
        max_new_tokens_per_task=max_new_tokens_per_task,
    )

    print(f"\n[done] sp={sparsity}: wrote {len(tasks_to_run)} new task(s); "
          f"{len(cached)} cached. Score with eval/src/eval.py "
          f"--model {args.model} --results_path {PRED_ROOT}")


if __name__ == "__main__":
    main()
