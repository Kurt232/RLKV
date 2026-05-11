"""
Adaptive sink/local sweep on SGLang for the lossless-frontier table.

For a fixed (model, sink+local) config, this script:
  1. Loads the model once into SGLang in full-attention mode and evaluates
     all benchmarks → records full_acc[task].
  2. Walks sparsity s = 0.2, 0.4, 0.6, ... (step = 0.2). At each s, re-inits
     the engine in RLKV mode at that sparsity and runs all *still-active*
     benchmarks in a single batched engine.generate() call.
  3. Per benchmark, when accuracy drops more than `--tolerance` (default 3.0)
     points below full, schedules one more run at s - 0.1 (the fine-grained
     backup) and marks the benchmark "done".
  4. After the main walk, runs the backup sparsities (grouped by sp value
     to share engine inits across benchmarks that broke at the same s).

Outputs match eval/efficiency/pred.py naming:
  pred/{model}/{task}-full.jsonl                            +.stats.json
  pred/{model}/{task}-rlkv-{adapter_tag}-s{sink}r{recent}-sp-{sp}.jsonl  +.stats.json

so the existing eval/efficiency/tab_sweep.py aggregator picks them up
without modification.

Usage:
  python eval/efficiency/pred_sweep.py \\
      --model Llama-3.1-8B-R1 \\
      --adapter-load-path head_dist/rlkv/Llama-3.1-8B-R1/llama_lr1e-2_ep2_bs32_reg1e-3_tau0.5 \\
      --sink-size 16 --recent-size 64 \\
      --tasks math_500 aime24 gsm8k mbpp mmlu_pro_che mmlu_pro_com mmlu_pro_law mmlu_pro_phy

Resume / skip-ahead:
  --sp-start 0.4   # skip 0.2, begin sweep at sparsity 0.4 (useful for resuming
                   # an interrupted run or starting from a known-safe value)
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

# Reuse the scoring metrics from eval/src.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from metrics import math_eval, mbpp_eval, simple_eval_match  # noqa: E402


DATASET2METRIC = {
    "gsm8k": math_eval,
    "math_500": math_eval,
    "aime24": math_eval,
    "mbpp": mbpp_eval,
    "mmlu_pro_che": simple_eval_match,
    "mmlu_pro_com": simple_eval_match,
    "mmlu_pro_law": simple_eval_match,
    "mmlu_pro_phy": simple_eval_match,
}

HF_MODEL_IDS = {
    "Llama-3.1-8B-R1": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "Qwen-2.5-7B-R1":  "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "Qwen-3-4B-Thinking": "Qwen/Qwen3-4B",
}

# Default sparsity grid.
DEFAULT_STEP = 0.2
DEFAULT_BACKUP_OFFSET = 0.1
DEFAULT_TOLERANCE = 3.0  # absolute accuracy points

PRED_ROOT = "eval/efficiency/pred_sweep"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--adapter-load-path", required=True)
    p.add_argument("--sink-size", type=int, default=16)
    p.add_argument("--recent-size", type=int, default=64)
    p.add_argument("--tasks", nargs="+", required=True,
                   help="Benchmark task names; must be keys in DATASET2METRIC")
    p.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE,
                   help="Stop a benchmark once acc drops by more than this many points vs full")
    p.add_argument("--sp-step", type=float, default=DEFAULT_STEP)
    p.add_argument("--backup-offset", type=float, default=DEFAULT_BACKUP_OFFSET,
                   help="When a task drops, also run at (last_sp - backup_offset) for finer frontier")
    p.add_argument("--sp-start", type=float, default=0.2)
    p.add_argument("--sp-max", type=float, default=0.8,
                   help="Upper bound on sparsity; sweep terminates when reached")
    p.add_argument("--max-running-requests", type=int, default=300)
    p.add_argument("--tp-size", type=int, default=1)
    p.add_argument("--model-path", type=str, default=None)
    p.add_argument("--num-samples", type=int, default=None,
                   help="Limit number of samples per task (for debugging)")
    p.add_argument("--disable-cuda-graph", action="store_true")
    return p.parse_args()


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


def score_predictions(task: str, preds: list[str], answers: list) -> float:
    score_list, _ = DATASET2METRIC[task](answers, preds)
    return round(100 * sum(score_list) / len(score_list), 2)


def build_engine(model_path: str, *, method: str, args, sparsity: float = 0.0,
                 max_total_len: int):
    """Construct an SGLang Engine for either `full` or `rlkv` mode."""
    kwargs = dict(
        model_path=model_path,
        tp_size=args.tp_size,
        disable_radix_cache=True,
        context_length=max_total_len,
        mem_fraction_static=0.85,
        max_running_requests=args.max_running_requests,
        attention_backend="triton",
        disable_cuda_graph=args.disable_cuda_graph,
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


def out_paths_for(model: str, task: str, *, method: str, adapter_tag: str,
                  sink: int, recent: int, sparsity: float):
    """Mirror eval/efficiency/pred.py naming so tab_sweep.py picks them up."""
    root = f"{PRED_ROOT}/{model}"
    os.makedirs(root, exist_ok=True)
    if method == "full":
        base = f"{root}/{task}-full"
    else:
        base = (f"{root}/{task}-rlkv-{adapter_tag}"
                f"-s{sink}r{recent}-sp-{sparsity}")
    return f"{base}.jsonl", f"{base}.stats.json"


def load_cached_accuracy(jsonl_path: str, task: str) -> Optional[float]:
    """If a prediction JSONL already exists for this task, score it from disk
    and return the accuracy; otherwise return None."""
    if not os.path.exists(jsonl_path):
        return None
    preds, answers = [], []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                return None
            preds.append(obj["pred"])
            answers.append(obj["answers"])
    if not preds:
        return None
    return score_predictions(task, preds, answers)


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


def write_jsonl_and_stats(*, data, outputs, jsonl_path: str,
                          stats_path: str, config: dict, total_time: float):
    preds_records = []
    out_tokens_list = []
    in_tokens_total = 0
    out_tokens_total = 0
    for out, item in zip(outputs, data):
        in_tok = out.get("meta_info", {}).get("prompt_tokens", 0)
        out_tok = out.get("meta_info", {}).get("completion_tokens", 0)
        in_tokens_total += in_tok
        out_tokens_total += out_tok
        out_tokens_list.append(out_tok)
        preds_records.append({
            "prompt": item["prompt"],
            "pred": out["text"],
            "answers": item["answer"],
            "input_tokens": in_tok,
            "output_tokens": out_tok,
        })
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in preds_records:
            json.dump(r, f, ensure_ascii=False)
            f.write("\n")
    stats = {
        "config": config,
        "throughput": {
            "total_time_s": round(total_time, 2),
            "total_input_tokens": in_tokens_total,
            "total_output_tokens": out_tokens_total,
            "avg_gen_throughput_tok_s": round(out_tokens_total / total_time, 1)
                if total_time > 0 else 0.0,
        },
        "token_stats": {
            "n": len(preds_records),
            "avg_input_tokens": round(in_tokens_total / max(1, len(preds_records)), 1),
            "avg_output_tokens": round(out_tokens_total / max(1, len(preds_records)), 1),
            "max_output_tokens": max(out_tokens_list) if out_tokens_list else 0,
        },
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    return preds_records


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
                    max_total_len, max_new_tokens_per_task) -> dict[str, float]:
    """Run one engine init at the given sparsity over all `active_tasks`.
    Returns {task: accuracy}."""
    if not active_tasks:
        return {}

    engine = build_engine(model_path, method=method, args=args,
                          sparsity=sparsity, max_total_len=max_total_len)
    try:
        # Warmup so triton JIT happens once, not amortized into our timing.
        engine.generate("Hello", {"max_new_tokens": 1, "temperature": 0.0})

        prompts_per_task = {t: all_prompts[t] for t in active_tasks}
        outputs_by_task, total_time = run_batched(
            engine, prompts_per_task, max_new_tokens_per_task,
        )

        accs = {}
        for task in active_tasks:
            outs = outputs_by_task.get(task, [])
            if not outs:
                continue
            jsonl_path, stats_path = out_paths_for(
                args.model, task, method=method,
                adapter_tag=adapter_tag,
                sink=args.sink_size, recent=args.recent_size,
                sparsity=sparsity,
            )
            cfg = dict(
                model=args.model, task=task, method=method,
                sparsity=sparsity, sink_size=args.sink_size,
                recent_size=args.recent_size, max_new_tokens=max_new_tokens_per_task[task],
                adapter_load_path=args.adapter_load_path if method == "rlkv" else None,
                max_running_requests=args.max_running_requests,
                tp_size=args.tp_size,
            )
            preds = write_jsonl_and_stats(
                data=all_data[task], outputs=outs,
                jsonl_path=jsonl_path, stats_path=stats_path,
                config=cfg, total_time=total_time,
            )
            answers = [p["answers"] for p in preds]
            pred_strs = [p["pred"] for p in preds]
            acc = score_predictions(task, pred_strs, answers)
            print(f"[score] {task} method={method} sp={sparsity}: acc={acc}")
            accs[task] = acc
        return accs
    finally:
        shutdown(engine)


def main():
    args = parse_args()
    for t in args.tasks:
        assert t in DATASET2METRIC, f"unknown task: {t}"

    model_path = resolve_model_path(args.model, args.model_path)
    adapter_tag = args.adapter_load_path.rstrip("/").split("/")[-1]

    # Load configs (max-length per task).
    config_dir = os.path.join(os.path.dirname(__file__), "..", "config")
    dataset2maxlen = json.load(open(os.path.join(config_dir, "dataset2maxlen.json")))
    max_new_tokens_per_task = {t: dataset2maxlen[t] for t in args.tasks}
    max_total_len = max(max_new_tokens_per_task.values()) * 2

    # Load datasets and tokenize prompts once (engine reload doesn't invalidate these).
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    all_data = {t: load_task_data(t, args.num_samples) for t in args.tasks}
    all_prompts = {t: format_prompts(tokenizer, all_data[t]) for t in args.tasks}
    for t in args.tasks:
        print(f"[data] {t}: {len(all_data[t])} samples, "
              f"max_new_tokens={max_new_tokens_per_task[t]}")

    # ---------- Stage 1: Full-attention baseline ----------
    print(f"\n{'='*60}\n[stage 1] full-attention baseline\n{'='*60}")
    # Reuse any existing {task}-full.jsonl from a previous run; only invoke
    # the engine on tasks whose full result is missing.
    full_accs: dict[str, float] = {}
    tasks_to_run_full: list[str] = []
    for t in args.tasks:
        jsonl_path, _ = out_paths_for(
            args.model, t, method="full", adapter_tag=adapter_tag,
            sink=args.sink_size, recent=args.recent_size, sparsity=0.0,
        )
        cached = load_cached_accuracy(jsonl_path, t)
        if cached is not None:
            full_accs[t] = cached
            print(f"[cache] {t}: reusing existing full result → acc={cached}")
        else:
            tasks_to_run_full.append(t)
    if tasks_to_run_full:
        fresh = run_at_sparsity(
            args=args, model_path=model_path, adapter_tag=adapter_tag,
            sparsity=0.0, method="full", active_tasks=tasks_to_run_full,
            all_data=all_data, all_prompts=all_prompts,
            max_total_len=max_total_len,
            max_new_tokens_per_task=max_new_tokens_per_task,
        )
        full_accs.update(fresh)
    else:
        print("[cache] all full results present; skipping engine init")

    # ---------- Stage 2: Adaptive RLKV walk ----------
    state = {t: "active" for t in args.tasks}
    backup_sp: dict[str, float] = {}  # task -> sparsity to run as backup
    history: dict[str, list[tuple[float, float]]] = defaultdict(list)  # task -> [(sp, acc)]

    sp = args.sp_start
    while sp <= args.sp_max + 1e-9:
        active = [t for t in args.tasks if state[t] == "active"]
        if not active:
            break
        print(f"\n{'='*60}\n[stage 2] sparsity={sp:.2f} | active tasks: {active}\n{'='*60}")
        accs = run_at_sparsity(
            args=args, model_path=model_path, adapter_tag=adapter_tag,
            sparsity=round(sp, 2), method="rlkv", active_tasks=active,
            all_data=all_data, all_prompts=all_prompts,
            max_total_len=max_total_len,
            max_new_tokens_per_task=max_new_tokens_per_task,
        )
        for t in active:
            acc = accs.get(t)
            if acc is None:
                continue
            history[t].append((round(sp, 2), acc))
            full_acc = full_accs.get(t)
            if full_acc is None:
                continue
            drop = full_acc - acc
            if drop > args.tolerance:
                # Schedule fine-grained backup at sp - backup_offset (if positive
                # and not already covered). If backup would land at a sparsity we
                # already ran, skip.
                cand = round(sp - args.backup_offset, 2)
                already_ran = any(abs(s - cand) < 1e-9 for s, _ in history[t])
                if cand > 0 and not already_ran:
                    backup_sp[t] = cand
                    print(f"[track] {t} dropped {drop:.2f} > {args.tolerance} at sp={sp:.2f}; "
                          f"will backup at sp={cand:.2f}")
                else:
                    print(f"[track] {t} dropped at sp={sp:.2f}; "
                          f"no new backup (cand={cand}, already_ran={already_ran})")
                state[t] = "done"
            else:
                print(f"[track] {t} acc={acc} vs full={full_acc} (drop {drop:.2f}); continue")
        sp = round(sp + args.sp_step, 2)

    # ---------- Stage 3: Run pending backups, batched by sp value ----------
    backup_groups: dict[float, list[str]] = defaultdict(list)
    for t, s in backup_sp.items():
        backup_groups[s].append(t)
    if backup_groups:
        print(f"\n{'='*60}\n[stage 3] backups: {dict(backup_groups)}\n{'='*60}")
        for s, tasks in sorted(backup_groups.items()):
            accs = run_at_sparsity(
                args=args, model_path=model_path, adapter_tag=adapter_tag,
                sparsity=s, method="rlkv", active_tasks=tasks,
                all_data=all_data, all_prompts=all_prompts,
                max_total_len=max_total_len,
                max_new_tokens_per_task=max_new_tokens_per_task,
            )
            for t in tasks:
                if t in accs:
                    history[t].append((s, accs[t]))
    else:
        print("\n[stage 3] no backups scheduled")

    # ---------- Summary ----------
    print(f"\n{'='*60}\n[summary] per-task frontier\n{'='*60}")
    print(f"{'task':16} {'full':>6} | {'(sp, acc) walk'}")
    for t in args.tasks:
        walk_str = ", ".join(f"({sp}, {acc})" for sp, acc in sorted(history[t]))
        print(f"{t:16} {full_accs.get(t, '--'):>6} | {walk_str}")


if __name__ == "__main__":
    main()
