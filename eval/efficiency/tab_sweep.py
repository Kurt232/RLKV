"""
Aggregate sink/local sweep results into the paper's lossless-frontier table.

For each (model, sink, local, benchmark), find the highest sparsity at which
accuracy stays within LOSSLESS_DROP points of the full-attention baseline,
and report that sparsity together with its accuracy. The LaTeX output matches
\\cref{tab:expr_lossless_window} in paper/tex/4_expr.tex (one row per (model,
sink+local), columns are benchmarks, cells are "sp / acc").

Also dumps a long CSV (one row per (model, sink, local, sparsity, task))
for ad-hoc inspection.

Usage:
    cd RLKV
    python eval/efficiency/tab_sweep.py --models Llama-3.1-8B-R1 Qwen-2.5-7B-R1 Qwen-3-4B-Thinking
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict

# Re-use the scoring metrics defined in eval/src.
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

TASK_ORDER = [
    "gsm8k", "math_500", "aime24", "mbpp",
    "mmlu_pro_che", "mmlu_pro_com", "mmlu_pro_law", "mmlu_pro_phy",
]
TASK_DISPLAY = {
    "gsm8k": "GSM8K",
    "math_500": "Math500",
    "aime24": "AIME24",
    "mbpp": "MBPP",
    "mmlu_pro_che": "Chem",
    "mmlu_pro_com": "CS",
    "mmlu_pro_law": "Law",
    "mmlu_pro_phy": "Phy",
}

# Near-lossless margin (in absolute accuracy points) relative to full attention.
LOSSLESS_DROP = 2.0


# File patterns produced by eval/efficiency/pred.py:
#   {task}-full.jsonl
#   {task}-rlkv-{adapter_tag}-s{sink}r{recent}-sp-{sparsity}.jsonl
RLKV_RE = re.compile(
    r"^(?P<task>[a-z0-9_]+)-rlkv-(?P<tag>[^-]+(?:-[^-]+)*?)-s(?P<sink>\d+)r(?P<recent>\d+)-sp-(?P<sp>[\d.]+)\.jsonl$"
)
FULL_RE = re.compile(r"^(?P<task>[a-z0-9_]+)-full\.jsonl$")


def score_file(path, dataset):
    with open(path, "r", encoding="utf-8") as f:
        preds, ans = [], []
        for line in f:
            obj = json.loads(line)
            preds.append(obj["pred"])
            ans.append(obj["answers"])
    if not preds:
        return None
    score_list, _ = DATASET2METRIC[dataset](ans, preds)
    return round(100 * sum(score_list) / len(score_list), 1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True,
                   help="One or more model names (subdirs under results-path)")
    p.add_argument("--results-path", default="eval/efficiency/pred_sweep")
    p.add_argument("--lossless-drop", type=float, default=LOSSLESS_DROP)
    p.add_argument("--cache", default=None,
                   help="Optional path to cache per-file scores as JSON")
    return p.parse_args()


def collect_model(root, model, cache):
    """Walk one model's results dir, return (full_scores, rlkv_scores)."""
    mp = os.path.join(root, model)
    if not os.path.isdir(mp):
        print(f"# WARN: {mp} missing, skipping", file=sys.stderr)
        return None, None

    full_scores = {}
    rlkv_scores = defaultdict(dict)  # (sink, recent, sparsity) -> {task: acc}

    for fname in sorted(os.listdir(mp)):
        if not fname.endswith(".jsonl"):
            continue
        fpath = os.path.join(mp, fname)
        m_full = FULL_RE.match(fname)
        m_rlkv = RLKV_RE.match(fname)
        if m_full:
            task = m_full.group("task")
            if task not in DATASET2METRIC:
                continue
            cache_key = f"{model}/{fname}"
            score = cache.get(cache_key)
            if score is None:
                score = score_file(fpath, task)
                cache[cache_key] = score
            full_scores[task] = score
        elif m_rlkv:
            task = m_rlkv.group("task")
            if task not in DATASET2METRIC:
                continue
            sink = int(m_rlkv.group("sink"))
            recent = int(m_rlkv.group("recent"))
            sp = float(m_rlkv.group("sp"))
            cache_key = f"{model}/{fname}"
            score = cache.get(cache_key)
            if score is None:
                score = score_file(fpath, task)
                cache[cache_key] = score
            rlkv_scores[(sink, recent, sp)][task] = score

    return full_scores, rlkv_scores


def max_lossless(rlkv_scores, full_scores, task, drop):
    """Return (best_sp, acc_at_best_sp) where best_sp is the largest sparsity
    such that acc >= full - drop, or None if no sparsity meets the bar."""
    full = full_scores.get(task)
    if full is None:
        return None
    best = None  # (sp, acc)
    for (sink, recent, sp), row in rlkv_scores.items():
        acc = row.get(task)
        if acc is None:
            continue
        if acc >= full - drop:
            if best is None or sp > best[0]:
                best = (sp, acc)
    return best


def main():
    args = parse_args()
    cache = {}
    if args.cache and os.path.exists(args.cache):
        cache = json.load(open(args.cache))

    per_model = {}  # model -> (full_scores, rlkv_scores)
    for model in args.models:
        fs, rs = collect_model(args.results_path, model, cache)
        if fs is None:
            continue
        per_model[model] = (fs, rs)

    if args.cache:
        with open(args.cache, "w") as f:
            json.dump(cache, f, indent=2)

    # --- CSV dump ---
    print("# CSV")
    print("model,sink,recent,sparsity,task,full_acc,rlkv_acc,delta,is_lossless")
    for model, (full_scores, rlkv_scores) in per_model.items():
        for (sink, recent, sp) in sorted(rlkv_scores.keys()):
            for task in TASK_ORDER:
                rlkv = rlkv_scores[(sink, recent, sp)].get(task)
                full = full_scores.get(task)
                if rlkv is None or full is None:
                    continue
                delta = round(rlkv - full, 2)
                is_lossless = int(delta >= -args.lossless_drop)
                print(f"{model},{sink},{recent},{sp},{task},{full},{rlkv},{delta},{is_lossless}")

    # --- LaTeX table (frontier view: one row per (model, window),
    #     cells = "sp / acc" for max lossless sparsity per benchmark). ---
    ncols = 2 + len(TASK_ORDER)
    print()
    print("% LaTeX — rows: (model, sink+local); cols: benchmarks; cell = max lossless sp / acc")
    print("\\begin{tabular}{l l " + "c " * len(TASK_ORDER) + "}")
    print("\\toprule")
    print("Model & Sink + Local & " +
          " & ".join(TASK_DISPLAY[t] for t in TASK_ORDER) + " \\\\")

    for model, (full_scores, rlkv_scores) in per_model.items():
        windows = sorted({(s, r) for (s, r, _) in rlkv_scores.keys()})
        rows_total = 1 + len(windows)
        print("\\midrule")
        # Full row
        full_cells = []
        for t in TASK_ORDER:
            v = full_scores.get(t)
            full_cells.append(
                f"\\cellcolor{{lightgray}}{{{v}}}" if v is not None
                else "\\cellcolor{lightgray}{--}"
            )
        print(f"\\multirow{{{rows_total}}}{{*}}{{{model}}} & -- (Full) & "
              + " & ".join(full_cells) + " \\\\")
        # One row per window
        for (sink, recent) in windows:
            cells = []
            for t in TASK_ORDER:
                best = max_lossless(
                    {k: v for k, v in rlkv_scores.items() if k[0] == sink and k[1] == recent},
                    full_scores, t, args.lossless_drop,
                )
                if best is None:
                    cells.append("--")
                else:
                    sp, acc = best
                    cells.append(f"sp~{sp} / {acc}")
            label = f"${sink} + {recent}$"
            print(f" & {label} & " + " & ".join(cells) + " \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")


if __name__ == "__main__":
    main()
