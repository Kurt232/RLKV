import os
import types

import torch
from flash_attn import flash_attn_func
from tensor_parallel.pretrained_model import TensorParallelPreTrainedModel
from torch import nn
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM, Qwen2Model

from base.duo_attn.ulysses import UlyssesAttention

from base.patch.llama import (
    llama_duo_attention_forward_one_way_reordered,
    llama_duo_attention_forward_two_way,
)

from base.patch.streaming_attn import (
    generate_streaming_info_blocksparse_flash_attn,
    generate_streaming_mask,
    streaming_attn_blocksparse_flash_attn,
    streaming_attn_sdpa,
)
from base.tuple_kv_cache import enable_tuple_kv_cache_for_qwen
from base.patch.utils import reorder_full_attn_heads, reorder_linear_weights


def enable_qwen_duo_attention_training(
    model: Qwen2ForCausalLM,
    sink_size,
    recent_size,
    max_length,
    initial_value=1.0,
    enable_ulysses_attention=False,
    streaming_attn_implementation="blocksparse",
):
    enable_tuple_kv_cache_for_qwen(model)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    if streaming_attn_implementation == "blocksparse":
        num_sink_blocks = (sink_size + 127) // 128
        num_recent_blocks = (recent_size + 127) // 128
        num_heads_per_device = model.config.num_attention_heads // int(
            os.environ["WORLD_SIZE"]
        )
        print(
            f"Using blocksparse implementation with {num_sink_blocks} sink blocks, {num_recent_blocks} recent blocks, and {num_heads_per_device} heads per device"
        )
        streaming_mask = generate_streaming_info_blocksparse_flash_attn(
            num_sink_blocks, num_recent_blocks, num_heads_per_device, device
        )
        streaming_attn_func = streaming_attn_blocksparse_flash_attn
    elif streaming_attn_implementation == "sdpa":
        streaming_mask = generate_streaming_mask(
            max_length, sink_size, recent_size, device
        )
        streaming_attn_func = streaming_attn_sdpa
    else:
        raise ValueError(
            f"Unsupported streaming attention implementation: {streaming_attn_implementation}"
        )

    for layer in model.model.layers:
        module = layer.self_attn
        module.forward = types.MethodType(llama_duo_attention_forward_two_way, module)
        module.sink_size = sink_size
        module.recent_size = recent_size
        module.register_parameter(
            "full_attention_heads",
            nn.Parameter(
                torch.ones(
                    module.config.num_key_value_heads,
                    device=device,
                    dtype=dtype,
                    requires_grad=True,
                )
                * initial_value
            ),
        )

        module.register_buffer("streaming_mask", streaming_mask)
        if not enable_ulysses_attention:
            module.streaming_attn_func = streaming_attn_func
            module.full_attn_func = flash_attn_func
        else:
            module.streaming_attn_func = UlyssesAttention(
                attn_func=streaming_attn_func,
            )
            module.full_attn_func = UlyssesAttention(
                attn_func=flash_attn_func,
            )


def enable_qwen_duo_attention_eval(
    model: Qwen2ForCausalLM,
    full_attention_heads,
    sink_size,
    recent_size,
):
    enable_tuple_kv_cache_for_qwen(model)

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    for idx, layer in enumerate(model.model.layers):
        module = layer.self_attn
        layer_full_attention_heads = torch.tensor(
            full_attention_heads[idx], device=device, dtype=dtype
        )

        module.forward = types.MethodType(
            llama_duo_attention_forward_one_way_reordered, module
        )
        module.q_proj = reorder_linear_weights(
            module.q_proj,
            layer_full_attention_heads,
            module.num_key_value_groups * module.head_dim,
            "out",
        )
        module.k_proj = reorder_linear_weights(
            module.k_proj,
            layer_full_attention_heads,
            module.head_dim,
            "out",
        )
        module.v_proj = reorder_linear_weights(
            module.v_proj,
            layer_full_attention_heads,
            module.head_dim,
            "out",
        )
        module.o_proj = reorder_linear_weights(
            module.o_proj,
            layer_full_attention_heads,
            module.num_key_value_groups * module.head_dim,
            "in",
        )
        layer_full_attention_heads = reorder_full_attn_heads(layer_full_attention_heads)

        module.sink_size = sink_size
        module.recent_size = recent_size
        module.register_buffer(
            "full_attention_heads",
            layer_full_attention_heads,
        )


def get_qwen_full_attention_heads(model):
    full_attention_heads = []
    if isinstance(model, TensorParallelPreTrainedModel):
        for shard in model.wrapped_model.module_shards:
            sharded_full_attention_heads = []
            for layer in shard.model.layers:
                module = layer.self_attn.tp_wrapped_module
                if not hasattr(module, "full_attention_heads"):
                    continue
                sharded_full_attention_heads.append(module.full_attention_heads)
            full_attention_heads.append(sharded_full_attention_heads)
        # concatenate the full_attention_heads from all shards, getting a list of tensors with len = num_layers
        device = full_attention_heads[0][0].device
        full_attention_heads = [
            torch.cat(
                [
                    sharded_heads[layer_idx].to(device)
                    for sharded_heads in full_attention_heads
                ]
            )
            for layer_idx in range(len(full_attention_heads[0]))
        ]
    elif isinstance(model, Qwen2ForCausalLM):
        for layer in model.model.layers:
            module = layer.self_attn
            if not hasattr(module, "full_attention_heads"):
                continue
            full_attention_heads.append(module.full_attention_heads)
    elif isinstance(model, Qwen2Model):
        for layer in model.layers:
            module = layer.self_attn
            if not hasattr(module, "full_attention_heads"):
                continue
            full_attention_heads.append(module.full_attention_heads)
    else:
        raise ValueError("Model type not supported")

    return full_attention_heads


def set_qwen_full_attention_heads(model, full_attention_heads):
    if isinstance(model, TensorParallelPreTrainedModel):
        for shard in model.wrapped_model.module_shards:
            for layer_idx, layer in enumerate(shard.model.layers):
                module = layer.self_attn.tp_wrapped_module
                if not hasattr(module, "full_attention_heads"):
                    continue
                module.full_attention_heads.data = full_attention_heads[layer_idx].to(
                    module.full_attention_heads.device,
                    module.full_attention_heads.dtype,
                )
    elif isinstance(model, Qwen2ForCausalLM):
        for layer_idx, layer in enumerate(model.model.layers):
            module = layer.self_attn
            if not hasattr(module, "full_attention_heads"):
                continue
            module.full_attention_heads.data = full_attention_heads[layer_idx].to(
                module.full_attention_heads.device, module.full_attention_heads.dtype
            )
    elif isinstance(model, Qwen2Model):
        for layer_idx, layer in enumerate(model.layers):
            module = layer.self_attn
            if not hasattr(module, "full_attention_heads"):
                continue
            module.full_attention_heads.data = full_attention_heads[layer_idx].to(
                module.full_attention_heads.device, module.full_attention_heads.dtype
            )
    else:
        raise ValueError("Model type not supported")


def map_qwen_full_attention_heads(model, func):
    if isinstance(model, TensorParallelPreTrainedModel):
        for shard in model.wrapped_model.module_shards:
            for layer in shard.model.layers:
                module = layer.self_attn.tp_wrapped_module
                if not hasattr(module, "full_attention_heads"):
                    continue
                func(module.full_attention_heads)
    elif isinstance(model, Qwen2ForCausalLM):
        for layer in model.model.layers:
            module = layer.self_attn
            if not hasattr(module, "full_attention_heads"):
                continue
            func(module.full_attention_heads)
    elif isinstance(model, Qwen2Model):
        for layer in model.layers:
            module = layer.self_attn
            if not hasattr(module, "full_attention_heads"):
                continue
            func(module.full_attention_heads)
    else:
        raise ValueError("Model type not supported")
