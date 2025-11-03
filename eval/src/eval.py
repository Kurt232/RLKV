import argparse
import json
import os

import numpy as np
from metrics import math_eval, mbpp_eval

dataset2metric = {
    "gsm8k": math_eval,
    "math_500": math_eval,
    "aime24": math_eval,
    "mbpp": mbpp_eval,
}


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--results_path", type=str, default=None)
    return parser.parse_args(args)


if __name__ == "__main__":
    args = parse_args()
    scores = dict()
    output_dict = dict()
    evals = dict()
    wrongs = dict()
    path = os.path.join(args.results_path, args.model)
    if not os.path.exists(path):
        raise ValueError(f"Path {path} does not exist")

    all_files = os.listdir(path)
    all_files.sort()

    dataset2maxlen = json.load(open("eval/config/dataset2maxlen.json", "r"))

    print("Evaluating on:", all_files)
    for filename in all_files:
        if not filename.endswith("jsonl"):
            continue
        max_length = dataset2maxlen[filename.split("-")[0]]
        predictions, answers = [], []
        output_lengths = []  # 新增：收集output_length
        input_lengths = []
        is_early_stops = []
        dataset = filename.split("-")[0]
        with open(os.path.join(path, filename), "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                predictions.append(data["pred"])
                answers.append(data["answers"])
                output_lengths.append(data["output_length"])
                input_lengths.append(data["input_length"])
                is_early_stops.append(data["is_early_stop"])
            
        assert len(predictions) == len(output_lengths)

        if len(predictions) == 0:
            continue
        try:
            score_list, k = dataset2metric[dataset](answers, predictions)
        except AssertionError as e:
            print(f"Error in {filename}: {e}")
            raise e
        score = sum(score_list) / len(score_list)
        score = round(100 * score, 2)
        scores[filename] = score

        evals[filename] = [(is_correct, output_length + input_length >= max_length, is_early_stop) for is_correct, output_length, input_length, is_early_stop in zip(score_list, output_lengths, input_lengths, is_early_stops)]
        # 计算output_length平均值
        if output_lengths:
            avg_length = round(np.mean(output_lengths), 2)
            output_dict[filename] = avg_length
            print(f"{filename}: {score}, pass@{k}, avg_output_length: {avg_length}")
        else:
            print(f"{filename}: {score}, pass@{k}, no output_length data")
        
        # 计算统计信息
        is_overlengths = []
        is_repeats = []

        for (s, o, r) in evals[filename]:
            if s < 1:
                is_overlengths.append(o)
                is_repeats.append(r)
        
        error_rate = len(is_overlengths) / len(evals[filename])
        overlength_rate = np.mean(is_overlengths) * error_rate
        repeat_rate = np.mean(is_repeats) * error_rate
        incorrect_rate = error_rate - overlength_rate - repeat_rate

        wrongs[filename] = (round(error_rate * 100, 2), round(incorrect_rate * 100, 2), round(overlength_rate * 100, 2), round(repeat_rate * 100, 2))

    # 保存scores
    out_path = os.path.join(path, "result.json")
    length_out_path = os.path.join(path, "result_lengths.json")
    eval_out_path = os.path.join(path, "result_evals.json")

    with open(out_path, "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)

    # 保存output_lengths平均值
    if output_dict:
        with open(length_out_path, "w") as f:
            json.dump(output_dict, f, ensure_ascii=False, indent=4)
        print(f"\nOutput lengths saved to: {length_out_path}")

    # 保存evals
    if evals:
        with open(eval_out_path, "w") as f:
            json.dump({
                "results": wrongs,
                "raw": evals
            }, f, ensure_ascii=False, indent=4)
        print(f"\nEval results saved to: {eval_out_path}")
    else:
        print("\nNo eval data found in any files")