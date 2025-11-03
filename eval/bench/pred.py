import argparse
import json
import os
import random

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from datasets import load_dataset
from base.patch import enable_duo_attention_eval, enable_h2o_eval, enable_rkv_eval
from base.tuple_kv_cache import enable_tuple_kv_cache
from base.duo_attn.utils import load_attn_pattern, sparsify_attention_heads, to_device


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=None,
    )

    parser.add_argument("--task", type=str, help="task name", required=True)

    parser.add_argument(
        "--method",
        type=str,
        default="full",
    )

    # duo attention
    parser.add_argument(
        "--attn_load_dir", type=str, default=None, help="attention pattern directory"
    )
    parser.add_argument("--sink_size", type=int, default=None)
    parser.add_argument("--recent_size", type=int, default=None)

    parser.add_argument("--sparsity", type=float, default=0.5)

    parser.add_argument("--decoding_simulation_length", type=int, default=50)

    parser.add_argument(
        "--repeat_win", type=int, default=400, help="window size for repeat detection"
    )

    parser.add_argument(
        "--is_rerun", action="store_true", help="whether to rerun the predictions"
    )

    return parser.parse_args(args)


def get_pred(
    model,
    tokenizer,
    eos_token_ids,
    data,
    max_length,
    max_gen,
    method,
    decoding_simulation_length,
    repeat_win,
):
    preds = []
    pbar = tqdm(data)
    for idx, json_obj in enumerate(pbar):
        prompt = json_obj["prompt"]
        if tokenizer.chat_template is not None:
            tokenized_prompt = tokenizer.apply_chat_template(
                conversation=[{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                truncation=False,
                return_tensors="pt",
            )
            input_ids = tokenized_prompt
        else:
            tokenized_prompt = tokenizer(
                prompt,
                truncation=False,
                return_tensors="pt",
            )
            input_ids = tokenized_prompt.input_ids

        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        if len(input_ids[0]) > max_length:
            half = int(max_length / 2)
            truncated_prompt = tokenizer.decode(
                input_ids[0][:half], skip_special_tokens=True
            ) + tokenizer.decode(input_ids[0][-half:], skip_special_tokens=True)
            # Re-tokenize the truncated prompt
            input_ids = tokenizer(
                truncated_prompt, truncation=False, return_tensors="pt"
            ).input_ids
            is_truncated = True
        else:
            is_truncated = False

        input_ids = input_ids.to("cuda")
        input_length = len(input_ids[0])
        if max_gen is None:
            max_gen = max_length - input_length

        is_early_stop = False

        pbar.set_description(f"Generating for {idx}, len = {input_length}")
        simulation_start_idx = input_length - decoding_simulation_length
        with torch.no_grad():
            output = model(
                input_ids=input_ids[:, :simulation_start_idx],
                past_key_values=None,
                use_cache=True,
            )
            past_key_values = output.past_key_values
            if decoding_simulation_length > 0:
                for token_idx, input_id in enumerate(
                    input_ids[0, simulation_start_idx:]
                ):
                    output = model(
                        input_ids=input_id.unsqueeze(0).unsqueeze(0),
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    past_key_values = output.past_key_values
            pred_token_idx = output.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            generated_content = [pred_token_idx.item()]
            for _ in range(max_gen - 1):
                outputs = model(
                    input_ids=pred_token_idx,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

                past_key_values = outputs.past_key_values
                pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                generated_content += [pred_token_idx.item()]
                if pred_token_idx.item() in eos_token_ids:
                    break

                # early stop if existing repetitive content
                if (
                    repeat_win != 0
                    and len(generated_content) > repeat_win
                    and len(generated_content) % repeat_win == 0
                ):
                    if len(set(generated_content[-repeat_win:])) < repeat_win * 0.1:
                        is_early_stop = True
                        break

        if method in ["h2o", "rkv"]:
            # H2O method requires clear the cache after each generation
            for layer in model.model.layers:
                layer.self_attn.kv_sampler.reset()

        output_length = len(generated_content)
        pred = tokenizer.decode(generated_content, skip_special_tokens=True)
        print(f"Prediction: {pred}")
        preds.append(
            {
                "prompt": prompt,
                "pred": pred,
                "answers": json_obj["answer"],
                "output_length": output_length,
                "input_length": input_length,
                "is_truncated": is_truncated,
                "is_early_stop": is_early_stop,
            }
        )
    return preds


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(path, model_name):
    tokenizer = AutoTokenizer.from_pretrained(
        path, trust_remote_code=True, use_fast=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        path,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )

    generation_config = GenerationConfig.from_pretrained(path)
    eos_token_ids = generation_config.eos_token_id
    if not isinstance(eos_token_ids, list):
        eos_token_ids = [eos_token_ids]

    model = model.eval()

    if args.method == "duo_attn":
        assert args.attn_load_dir is not None, "attn_load_dir must be provided"
        print(
            f"Loading attention pattern from {args.attn_load_dir} with sparsity {args.sparsity}"
        )
        full_attention_heads, sink_size, recent_size = load_attn_pattern(
            args.attn_load_dir
        )

        if args.sink_size is not None:
            sink_size = args.sink_size
        if args.recent_size is not None:
            recent_size = args.recent_size

        full_attention_heads, sparsity = sparsify_attention_heads(
            full_attention_heads, None, sparsity=args.sparsity
        )
        print(f"True sparsity: {sparsity}")

        enable_duo_attention_eval(
            model,
            full_attention_heads,
            sink_size,
            recent_size,
        )
    elif args.method == "full":
        enable_tuple_kv_cache(model)
    elif args.method == "h2o":
        budget_ratio = round(1 - args.sparsity, 5)
        enable_h2o_eval(
            model, budget_ratio, args.sink_size + args.recent_size
        )  # keep same REAL sparsity as duo_attn
    elif args.method == "rkv":
        budget_ratio = round(1 - args.sparsity, 5)
        enable_rkv_eval(model, budget_ratio, args.sink_size + args.recent_size) # keep same REAL sparsity as duo_attn
    elif args.method == "rlkv":
        assert args.attn_load_dir is not None, "attn_load_dir must be provided"
        print(
            f"Loading attention pattern from {args.attn_load_dir} with sparsity {args.sparsity}"
        )
        adapter_weight = np.loadtxt(
            os.path.join(args.attn_load_dir, "adapter_weights.tsv"),
            dtype=float,
            delimiter="\t",
        )
        adapter_weight = np.clip(adapter_weight, 0, 1)
        sink_size = args.sink_size
        recent_size = args.recent_size

        adapter_weight, sparsity = sparsify_attention_heads(
            adapter_weight, None, sparsity=args.sparsity
        )
        print(f"True sparsity: {sparsity}")

        enable_duo_attention_eval(
            model,
            adapter_weight,
            sink_size,
            recent_size,
        )
    else:
        raise ValueError(f"Unknown method: {args.method}")

    return model, tokenizer, eos_token_ids


if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()
    model2path = json.load(open("eval/config/model2path.json", "r"))
    dataset2maxlen = json.load(open("eval/config/dataset2maxlen.json", "r"))
    device_list = [i for i in range(torch.cuda.device_count())]
    model_name = args.model
    # define your model
    model, tokenizer, eos_token_ids = load_model_and_tokenizer(
        model2path[model_name], model_name
    )
    model = to_device(model, device_list, enable_tp=True)

    dataset = args.task
    max_length = dataset2maxlen[dataset]
    # predict on each dataset
    root = "eval/bench/pred"
    if not os.path.exists(root):
        os.makedirs(root)

    data = load_dataset("Kurt232/bench", name=dataset, split="test")
    if not os.path.exists(f"{root}/{model_name}"):
        os.makedirs(f"{root}/{model_name}")
    if args.method == "duo_attn":
        out_path = f"{root}/{model_name}/{dataset}-duo_attn-pattern-{args.attn_load_dir.split('/')[-1]}-sp-{args.sparsity}.jsonl"
    elif args.method == "full":
        out_path = f"{root}/{model_name}/{dataset}-full.jsonl"
    elif args.method == "h2o":
        out_path = (
            f"{root}/{model_name}/{dataset}-h2o-sp-{args.sparsity}.jsonl"
        )
    elif args.method == "rkv":
        out_path = (
            f"{root}/{model_name}/{dataset}-rkv-sp-{args.sparsity}.jsonl"
        )
    elif args.method == "rlkv":
        out_path = f"{root}/{model_name}/{dataset}-rlkv-{args.attn_load_dir.split('/')[-1]}-sp-{args.sparsity}.jsonl"
    if os.path.exists(out_path) and not args.is_rerun:
        print(f"Predictions already exist at {out_path}, skipping...")
        exit(0)

    preds = get_pred(
        model,
        tokenizer,
        eos_token_ids,
        data,
        max_length,
        None,
        args.method,
        args.decoding_simulation_length,
        args.repeat_win,
    )
    with open(out_path, "w", encoding="utf-8") as f:
        for pred in preds:
            json.dump(pred, f, ensure_ascii=False)
            f.write("\n")
