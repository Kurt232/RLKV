#!/bin/bash
#
# Score the prediction jsonls produced by scripts/run_sweep_lossless.sh.
# Pure CPU work (latex2sympy + mbpp sandbox + regex), no GPU needed —
# meant to run on the login node so compute nodes aren't tied up scoring.
#
# Incremental by default: reads result.json and only re-scores jsonls not
# already in it. Force a full re-score with FORCE=1.
#
# Usage:
#   bash scripts/run_sweep_eval.sh [MODEL] [PRED_ROOT]
#   FORCE=1 bash scripts/run_sweep_eval.sh [MODEL]            # rescore everything
#
# Defaults:
#   MODEL     = Llama-3.1-8B-R1
#   PRED_ROOT = eval/efficiency/pred_sweep
#
# Examples:
#   bash scripts/run_sweep_eval.sh                                # default model
#   bash scripts/run_sweep_eval.sh Qwen-3-4B-Thinking
#   bash scripts/run_sweep_eval.sh Llama-3.1-8B-R1 eval/bench/pred  # different pred root
#
# Score all three models in the background:
#   for m in Llama-3.1-8B-R1 Qwen-2.5-7B-R1 Qwen-3-4B-Thinking; do
#       bash scripts/run_sweep_eval.sh "$m" &
#   done; wait

set -o pipefail

MODEL=${1:-Llama-3.1-8B-R1}
PRED_ROOT=${2:-eval/efficiency/pred_sweep}

PYTHON="${PYTHON:-python}"
if [ ! -x "$PYTHON" ]; then
    echo "[error] python binary not found: $PYTHON" >&2
    echo "        override with PYTHON=/path/to/python bash scripts/run_sweep_eval.sh ..." >&2
    exit 1
fi

PRED_DIR="${PRED_ROOT}/${MODEL}"
if [ ! -d "$PRED_DIR" ]; then
    echo "[error] no predictions found at $PRED_DIR" >&2
    echo "        run scripts/run_sweep_lossless.sh first." >&2
    exit 1
fi

LOG_DIR="eval/efficiency/logs/${MODEL}/sweep_lossless"
mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/${MODEL}_eval.log"

echo "============================================="
echo "Scoring $MODEL"
echo "Pred root: $PRED_ROOT"
echo "Log:       $LOG"
echo "Python:    $PYTHON"
echo "============================================="

# Silence latex2sympy's `equations=True in NormalizationConfig is deprecated`
# warning — printed once per math sample, otherwise floods the log.
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::DeprecationWarning}"

"$PYTHON" -u eval/src/eval.py \
    --model "$MODEL" \
    --results_path "$PRED_ROOT" \
    2>&1 | tee "$LOG"

echo ""
echo "============================================="
echo "Done. Scores written to:"
echo "  $PRED_DIR/result.json          (per-file accuracy)"
echo "  $PRED_DIR/result_lengths.json  (avg output_length per file)"
echo "  $PRED_DIR/result_evals.json    (per-sample raw + error breakdown)"
echo "============================================="
