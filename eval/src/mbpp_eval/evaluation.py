import ast
import pandas as pd
import numpy as np
import itertools

from typing import *
from tqdm.auto import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from .data import stream_jsonl
from .execution import check_correctness

import re

def read_dataset(
    data_file: str = None,
) -> Dict:
    """
    Reads a dataset and returns a dictionary of tasks.
    """
    dataset = {task["task_id"]: task for task in stream_jsonl(data_file)}

    return dataset


def evaluate_functional_correctness(
        task_idx_list: list[list[int]],
        predictions_list: list[list[str]],
        n_workers: int = 32,
        timeout: float = 10.0,
        problem_file: str = "./mbpp_test.jsonl",
):
    """
    Evaluates the functional correctness of a model.
    """

    code_pattern = re.compile(r"```python(.*?)```", re.DOTALL)
    
    def extract_last_code_block(text):
        matches = code_pattern.findall(text)
        return matches[-1] if matches else None

    problems = read_dataset(problem_file)

    with ThreadPoolExecutor(max_workers=n_workers) as executor:

        futures = []
        completion_id = Counter()
        n_samples = 0
        results = defaultdict(list)

        for task_idx, predictions in zip(task_idx_list, predictions_list):
            assert len(task_idx) == len(predictions)

            for task_id, prediction in zip(task_idx, predictions):
                assert task_id in problems, f"Task ID {task_idx} not found in problems."
                assert isinstance(prediction, str), f"Prediction must be a string, got {type(prediction)}."

                code = extract_last_code_block(prediction)
                if code is None:
                    code = ""
                sample = code + "\n" + "\n".join(problems[task_id]["test"])

                completion_id_ = completion_id[task_id]
                args = (task_id, sample, timeout, completion_id_)
                future = executor.submit(check_correctness, *args)
                futures.append(future)
                completion_id[task_id] += 1
                n_samples += 1

        for future in as_completed(futures):
            result = future.result()
            results[result["task_id"]].append((result["completion_id"], result))

    assert len(results) > 0, "No results found."
    k = len(list(results.values())[0])
    scores = []
    for task_id, result in results.items():
        passed = [r[1]["passed"] for r in result]
        assert len(passed) == k, f"Number of completions for task {task_id} is not {k}."
        scores.append(max(passed))

    return scores if scores else [0.0], k
