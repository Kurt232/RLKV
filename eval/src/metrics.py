import re

from math_comparison import compare_gold_target
from math_extraction import normalize_math_expression
from mbpp_eval.evaluation import evaluate_functional_correctness


def simple_eval_match(golds_list: list[list[str]], predictions_list: list[list[str]], regex=r"(?i)Answer:\s*([A-Z])") -> float:
    """Computes the metric over a list of golds and predictions for one single sample.

    Args:
        golds (list[str]): Reference targets
        predictions (list[str]): Predicted strings

    Returns:
        float: Aggregated score over the current sample's items.
    """
    assert len(golds_list) == len(predictions_list)
    assert len(golds_list) > 0 

    scores = []
    if isinstance(predictions_list[0], list):
        k = len(predictions_list[0])
    else:
        k = 1
    for golds, predictions in zip(golds_list, predictions_list):
        if not isinstance(predictions, list):
            predictions = [predictions]
            golds = [golds]
        extracted_preds = []
        for pred in predictions:
            extracted_answer = None
            pred = (
                pred.replace("**", "")
                .replace("$\\boxed{", "")
                .replace("}$", "")
                .replace("\\$", "")
                .replace("$\\text{", "")
                .replace("$", "")
                .replace("\\mathrm{", "")
                .replace("\\{", "")
                .replace("\\text", "")
                .replace("\\(", "")
                .replace("\\mathbf{", "")
                .replace("{", "")
                .replace("\\boxed", "")
            )
            match = re.search(regex, pred)
            if match:
                extracted_answer = match.group(1).strip()
            extracted_preds.append(extracted_answer)
        results = []
        # We might need to flatten golds if they are a list of lists
        for gold in golds:
            for pred in extracted_preds:
                score = 0.0
                if pred:
                    score = 1.0 if gold.strip() == pred else 0.0
                results.append(score)
        scores.append(max(results))

    return scores if scores else [0.0], k


def prefix_exact_match(golds_list: list[list[str]], predictions_list: list[list[str]]) -> float:
    """Computes the prefix exact match metric over lists of golds and predictions for multiple samples.

    Args:
        golds_list (list[list[str]]): List of reference targets for each sample
        predictions_list (list[list[str]]): List of predicted strings for each sample

    Returns:
        float: Average accuracy across all samples (pass@k evaluation).
    """
    assert len(golds_list) == len(predictions_list)
    assert len(golds_list) > 0 

    scores = []
    if isinstance(predictions_list[0], list):
        k = len(predictions_list[0])
    else:
        k = 1
    for golds, predictions in zip(golds_list, predictions_list):
        if not isinstance(predictions, list):
            predictions = [predictions]
            golds = [golds]
        results = []
        # We might need to flatten golds if they are a list of lists
        for gold in golds:
            for pred in predictions:
                if not pred:
                    results.append(0.0)
                    continue
                gold = gold.strip()
                pred = pred.strip()
                results.append(1.0 if pred.startswith(gold) else 0.0)
        scores.append(max(results))

    return scores if scores else [0.0], k


def math_eval(golds_list: list[list[str]], predictions_list: list[list[str]]) -> float:
    """Computes the math evaluation metric over lists of golds and predictions for multiple samples.

    Args:
        golds_list (list[list[str]]): List of reference targets for each sample
        predictions_list (list[list[str]]): List of predicted strings for each sample

    Returns:
        float: Average accuracy across all samples (pass@k evaluation).
    """
    assert len(golds_list) == len(predictions_list)
    assert len(golds_list) > 0 

    scores = []
    if isinstance(predictions_list[0], list):
        k = len(predictions_list[0])
    else:
        k = 1
    for golds, predictions in zip(golds_list, predictions_list):
        if not isinstance(predictions, list):
            predictions = [predictions]
            golds = [golds]
        assert len(predictions) == k, f"Number of predictions {len(predictions)} is not {k}."
        results = []
        for gold in golds:
            for pred in predictions:
                try:
                    gold_normalized = gold.strip()
                    gold_normalized = normalize_math_expression(gold_normalized)

                    pred_normalized = pred.strip()
                    pred_normalized = normalize_math_expression(pred_normalized)

                    score = 1.0 if compare_gold_target(gold_normalized, pred_normalized) else 0.0
                    results.append(score)
                except Exception:
                    # If there's an error in normalization or comparison, treat as incorrect
                    results.append(0.0)
        
        scores.append(max(results) if results else 0.0)

    return scores if scores else [0.0], k


def mbpp_eval(task_idx_list: list[list[int]], predictions_list: list[list[str]]) -> float:
    """Computes the MBPP evaluation metric over lists of golds and predictions for multiple samples.

    Args:
        golds_list (list[list[str]]): List of reference targets for each sample
        predictions_list (list[list[str]]): List of predicted strings for each sample

    Returns:
        float: Average accuracy across all samples (pass@k evaluation).
    """
    assert len(task_idx_list) == len(predictions_list)
    assert len(predictions_list) > 0

    if isinstance(predictions_list[0], list):
        _k = len(predictions_list[0])
    else:
        _k = 1
    _task_idx_list = []
    _predictions_list = []
    for task_idx, predictions in zip(task_idx_list, predictions_list):
        if not isinstance(predictions, list):
            predictions = [predictions]
            task_idx = [task_idx]
        _task_idx_list.append(task_idx)
        _predictions_list.append(predictions)

    scores, k = evaluate_functional_correctness(_task_idx_list, _predictions_list, problem_file="eval/src/mbpp_eval/mbpp_test.jsonl")
    assert k == _k, f"Number of predictions {k} is not {_k}."
    return scores, k