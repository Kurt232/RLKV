import os

import numpy as np
import torch

from base.h2o_kv_cache import (
    enable_llama_h2o_eval,
    enable_llama_rkv_eval,
    enable_qwen_h2o_eval,
    enable_qwen_rkv_eval,
    enable_qwen3_h2o_eval,
    enable_qwen3_rkv_eval,
)
from base.patch.llama import (
    enable_llama_duo_attention_eval,
    enable_llama_duo_attention_training,
    get_llama_full_attention_heads,
    map_llama_full_attention_heads,
    set_llama_full_attention_heads,
)
from base.patch.qwen import (
    enable_qwen_duo_attention_eval,
    enable_qwen_duo_attention_training,
    get_qwen_full_attention_heads,
    map_qwen_full_attention_heads,
    set_qwen_full_attention_heads,
)
from base.patch.qwen3 import (
    enable_qwen3_duo_attention_eval,
    enable_qwen3_duo_attention_training,
    get_qwen3_full_attention_heads,
    map_qwen3_full_attention_heads,
    set_qwen3_full_attention_heads,
)


def enable_duo_attention_training(
    model,
    sink_size,
    recent_size,
    max_length,
    initial_value=1.0,
    enable_ulysses_attention=False,
    streaming_attn_implementation="blocksparse",
):
    print(
        f"Enabling DuoAttention training using {streaming_attn_implementation} implementation"
    )
    if "llama" in model.config.model_type:
        enable_llama_duo_attention_training(
            model,
            sink_size,
            recent_size,
            max_length,
            initial_value=initial_value,
            enable_ulysses_attention=enable_ulysses_attention,
            streaming_attn_implementation=streaming_attn_implementation,
        )
    elif "qwen2" in model.config.model_type:
        enable_qwen_duo_attention_training(
            model,
            sink_size,
            recent_size,
            max_length,
            initial_value=initial_value,
            enable_ulysses_attention=enable_ulysses_attention,
            streaming_attn_implementation=streaming_attn_implementation,
        )
    elif "qwen3" in model.config.model_type:
        enable_qwen3_duo_attention_training(
            model,
            sink_size,
            recent_size,
            max_length,
            initial_value=initial_value,
            enable_ulysses_attention=enable_ulysses_attention,
            streaming_attn_implementation=streaming_attn_implementation,
        )
    else:
        raise ValueError(f"Model type {model.config.model_type} not supported")


def enable_duo_attention_eval(
    model,
    full_attention_heads,
    sink_size,
    recent_size,
):
    print(
        f"Enabling DuoAttention evaluation using sink size {sink_size} and recent size {recent_size}"
    )
    if "llama" in model.config.model_type:
        enable_llama_duo_attention_eval(
            model,
            full_attention_heads,
            sink_size,
            recent_size,
        )
    elif "qwen2" in model.config.model_type:
        enable_qwen_duo_attention_eval(
            model,
            full_attention_heads,
            sink_size,
            recent_size,
        )
    elif "qwen3" in model.config.model_type:
        enable_qwen3_duo_attention_eval(
            model,
            full_attention_heads,
            sink_size,
            recent_size,
        )
    else:
        raise ValueError(f"Model type {model.config.model_type} not supported")


def get_full_attention_heads(model):
    if "llama" in model.config.model_type:
        return get_llama_full_attention_heads(model)
    elif "qwen2" in model.config.model_type:
        return get_qwen_full_attention_heads(model)
    elif "qwen3" in model.config.model_type:
        return get_qwen3_full_attention_heads(model)
    else:
        raise ValueError(f"Model type {model.config.model_type} not supported")


def set_full_attention_heads(model, full_attention_heads):
    if "llama" in model.config.model_type:
        model = set_llama_full_attention_heads(model, full_attention_heads)
    elif "qwen2" in model.config.model_type:
        model = set_qwen_full_attention_heads(model, full_attention_heads)
    elif "qwen3" in model.config.model_type:
        model = set_qwen3_full_attention_heads(model, full_attention_heads)
    else:
        raise ValueError(f"Model type {model.config.model_type} not supported")
    return model


def map_full_attention_heads(model, func):
    if "llama" in model.config.model_type:
        return map_llama_full_attention_heads(model, func)
    elif "qwen2" in model.config.model_type:
        return map_qwen_full_attention_heads(model, func)
    elif "qwen3" in model.config.model_type:
        return map_qwen3_full_attention_heads(model, func)
    else:
        raise ValueError(f"Model type {model.config.model_type} not supported")


def load_full_attention_heads(load_dir, filename="full_attention_heads.tsv"):
    full_attention_heads = np.loadtxt(
        os.path.join(load_dir, filename),
        dtype=float,
        delimiter="\t",
    )
    full_attention_heads = np.clip(full_attention_heads, 0, 1)
    full_attention_heads = torch.tensor(full_attention_heads, dtype=torch.float32)
    return full_attention_heads


def enable_h2o_eval(
    model,
    budget_ratio,
    window_size,
):
    print(
        f"Enabling H2O evaluation using budget ratio {budget_ratio} and window size {window_size}"
    )
    if "llama" in model.config.model_type:
        enable_llama_h2o_eval(
            model,
            budget_ratio,
            window_size,
        )
    elif "qwen2" in model.config.model_type:
        enable_qwen_h2o_eval(
            model,
            budget_ratio,
            window_size,
        )
    elif "qwen3" in model.config.model_type:
        enable_qwen3_h2o_eval(
            model,
            budget_ratio,
            window_size,
        )
    else:
        raise ValueError(f"Model type {model.config.model_type} not supported")


def enable_rkv_eval(
    model,
    budget_ratio,
    window_size,
):
    print(f"Enabling RKV evaluation with budget ratio {budget_ratio}")
    if "llama" in model.config.model_type:
        enable_llama_rkv_eval(
            model,
            budget_ratio,
            window_size,
        )
    elif "qwen2" in model.config.model_type:
        enable_qwen_rkv_eval(
            model,
            budget_ratio,
            window_size,
        )
    elif "qwen3" in model.config.model_type:
        enable_qwen3_rkv_eval(
            model,
            budget_ratio,
            window_size,
        )
    else:
        raise ValueError(f"Model type {model.config.model_type} not supported")
