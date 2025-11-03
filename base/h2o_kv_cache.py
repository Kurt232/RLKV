import types
from typing import Optional, Tuple

import torch
from torch import nn
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from transformers.models.llama.modeling_llama import (
    repeat_kv, 
    apply_rotary_pos_emb,
)

from base.tuple_kv_cache import (
    enable_tuple_kv_cache_for_llama,
    enable_tuple_kv_cache_for_qwen,
    enable_tuple_kv_cache_for_qwen3,
)


class BaseSampler:
    def __init__(
        self,
        budget_ratio: float,
        window_size: int,
    ):
        self.seq_len = 0
        self.budget_ratio = budget_ratio
        self.window_size = window_size

    @property
    def budget(self):
        return int(self.seq_len * self.budget_ratio) + self.window_size

    def reset(self):
        self.seq_len = 0

    def update_kv(self, *args, **kwargs):
        raise NotImplementedError("Subclasses should implement this method.")


class H2O(BaseSampler):
    def __init__(
        self,
        budget_ratio: float,
        window_size: int,
        num_key_value_groups: int,
        scaling: float,
    ):
        super().__init__(budget_ratio, window_size)
        self.num_key_value_groups = num_key_value_groups
        self.scaling = scaling
        self.hh_score = None  # (bsz, kv_heads, kv_len)

    def update_kv(self, key_states, query_states, value_states, attention_mask=None):
        key_states_rep = repeat_kv(key_states, self.num_key_value_groups)

        attn_weights = (
            torch.matmul(query_states, key_states_rep.transpose(2, 3)) * self.scaling
        )  # (bsz, heads, q_len, kv_len)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)

        q_len = attn_weights.shape[2]

        self.seq_len += q_len
        cache_size = self.budget
        hh_size = int(cache_size / 2)
        recent_size = int(cache_size / 2)

        bsz, kv_heads, kv_len, head_dim = key_states.shape

        # update hh_score
        attn_score = attn_weights.sum(2)  # (bsz, heads, kv_len)
        if self.num_key_value_groups != 1:
            attn_score = attn_score.reshape(bsz, kv_heads, -1, kv_len).sum(
                2
            )  # (bsz, kv_heads, kv_len)
        if self.hh_score is None:
            # prefill
            self.hh_score = attn_score
        else:
            # generate
            attn_score[..., :-q_len] += self.hh_score
            self.hh_score = attn_score

        if kv_len <= cache_size:
            return (key_states, value_states)

        # hh-selection
        select_hh_scores = self.hh_score[
            ..., : kv_len - recent_size
        ]  # (bsz, kv_heads, kv_len - recent_size)
        _, keep_topk = torch.topk(
            select_hh_scores, hh_size, dim=-1
        )  # (bsz, kv_heads, hh_size)
        keep_topk = keep_topk.sort().values

        # recent
        keep_recent = torch.arange(
            kv_len - recent_size, kv_len, device=key_states.device
        )[None].expand(
            bsz, kv_heads, recent_size
        )  # (recent_size, )

        # gather
        keep_idx = torch.cat(
            [keep_topk, keep_recent], dim=-1
        )  # (bsz, kv_heads, cache_size)
        keep_idx = keep_idx.unsqueeze(-1).expand(
            -1, -1, -1, head_dim
        )  # (bsz, kv_heads, cache_size, head_dim)

        k_crop = key_states.gather(2, keep_idx)  # (bsz, kv_heads, cache_size, head_dim)
        v_crop = value_states.gather(
            2, keep_idx
        )  # (bsz, kv_heads, cache_size, head_dim)

        # update hh_score
        self.hh_score = self.hh_score.gather(
            2, keep_idx[..., 0]
        )  # (bsz, kv_heads, cache_size)

        return (k_crop, v_crop)

    def reset(self):
        super().reset()
        self.hh_score = None


def llama_h2o_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    # LlamaFlashAttention2 attention does not support output_attentions
    output_attentions = False

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # Flash attention requires the input to have the shape
    # batch_size x seq_length x head_dime x hidden_dim
    # therefore we just need to keep the original shape
    query_states = query_states.view(
        bsz, q_len, self.config.num_attention_heads, self.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
        bsz, q_len, self.config.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
        bsz, q_len, self.config.num_key_value_heads, self.head_dim
    ).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin
    )

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    if use_cache:
        past_key_value = self.kv_sampler.update_kv(
            key_states, query_states, value_states, attention_mask
        )
    else:
        past_key_value = None

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        query_states = query_states.to(torch.float16)
        key_states = key_states.to(torch.float16)
        value_states = value_states.to(torch.float16)

    attn_output = self._flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        q_len,
        dropout=0.0 if not self.training else self.attention_dropout,
    )

    attn_output = attn_output.reshape(bsz, q_len, self.config.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def enable_llama_h2o_eval(
    model: LlamaForCausalLM,
    budget_ratio: float,
    window_size: int,
):
    enable_tuple_kv_cache_for_llama(model)

    for _, layer in enumerate(model.model.layers):
        module = layer.self_attn

        module.kv_sampler = H2O(
            budget_ratio=budget_ratio,
            window_size=window_size,
            num_key_value_groups=module.num_key_value_groups,
            scaling=module.scaling,
        )

        module.forward = types.MethodType(llama_h2o_attention_forward, module)


def enable_qwen_h2o_eval(
    model: Qwen2ForCausalLM,
    budget_ratio: float,
    window_size: int,
):
    enable_tuple_kv_cache_for_qwen(model)

    for _, layer in enumerate(model.model.layers):
        module = layer.self_attn

        module.kv_sampler = H2O(
            budget_ratio=budget_ratio,
            window_size=window_size,
            num_key_value_groups=module.num_key_value_groups,
            scaling=module.scaling,
        )

        module.forward = types.MethodType(llama_h2o_attention_forward, module)


def qwen3_h2o_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    # LlamaFlashAttention2 attention does not support output_attentions
    output_attentions = False

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # Flash attention requires the input to have the shape
    # batch_size x seq_length x head_dime x hidden_dim
    # therefore we just need to keep the original shape
    query_states = self.q_norm(query_states.view(
        bsz, q_len, self.config.num_attention_heads, self.head_dim
    )).transpose(1, 2)
    key_states = self.k_norm(key_states.view(
        bsz, q_len, self.config.num_key_value_heads, self.head_dim
    )).transpose(1, 2)
    value_states = value_states.view(
        bsz, q_len, self.config.num_key_value_heads, self.head_dim
    ).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin
    )

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    if use_cache:
        past_key_value = self.kv_sampler.update_kv(
            key_states, query_states, value_states, attention_mask
        )
    else:
        past_key_value = None

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        query_states = query_states.to(torch.float16)
        key_states = key_states.to(torch.float16)
        value_states = value_states.to(torch.float16)

    attn_output = self._flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        q_len,
        dropout=0.0 if not self.training else self.attention_dropout,
    )

    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def enable_qwen3_h2o_eval(
    model: Qwen3ForCausalLM,
    budget_ratio: float,
    window_size: int,
):
    enable_tuple_kv_cache_for_qwen3(model)

    for _, layer in enumerate(model.model.layers):
        module = layer.self_attn

        module.kv_sampler = H2O(
            budget_ratio=budget_ratio,
            window_size=window_size,
            num_key_value_groups=module.num_key_value_groups,
            scaling=module.scaling,
        )

        module.forward = types.MethodType(qwen3_h2o_attention_forward, module)


import torch.nn.functional as F


class RKV(BaseSampler):
    def __init__(
        self,
        budget_ratio: float,
        window_size: int,
        num_key_value_groups: int,
        scaling: float,
    ):
        self.rkv_window_size = 8  # official alpha
        self.rkv_buf_size = 128  # official
        self.rkv_fixed_size = (
            512  # ! to avoid OOM, it takes 1GB extra memory when computing cosine
        )

        current_window_size = int(
            window_size - num_key_value_groups * self.rkv_window_size / 2
        )
        assert (
            current_window_size > 0
        ), f"current_window_size must be greater than 0, got {current_window_size}"
        super().__init__(
            budget_ratio, current_window_size
        )  # to calculate correct budget

        self.num_key_value_groups = num_key_value_groups
        self.scaling = scaling

        self.kernel_size = 7  # official
        self.mix_lambda = 0.1  # official
        self.retain_ratio = 0.1
        self.retain_direction = "last"

        self.query_cache = None  # ! extra query cache for RKV, I need to reduce this part for fair comparison.
        # each layer
        # all window_cache: window_size * num_kv_heads * 2
        # all query_cache: rkv_window_size * num_q_heads
        # rest window_cache = window_size * num_kv_heads * 2 - self.rkv_window_size * num_q_heads
        # current_window_size to enlarge budget:
        # current_window_size = int(window_size - num_key_value_groups * self.rkv_window_size / 2)

    def update_kv(self, key_states, query_states, value_states, attention_mask=None):
        bsz, _, q_len, head_dim = query_states.shape

        # =============== Enable Query Cache ============
        self.seq_len += q_len  #! Update sequence length
        if self.seq_len == q_len:
            # prefill stage
            self.query_cache = query_states[:, :, -self.rkv_window_size :, :]
        else:
            # Add current query to cache
            self.query_cache = torch.cat(
                (
                    self.query_cache[:, :, -(self.rkv_window_size - q_len) :, :],
                    query_states,
                ),
                dim=2,
            )
        # =============== Enable Query Cache end =========

        # =============== decoding-time compression start ===============
        kv_cache_len = key_states.shape[-2]
        if kv_cache_len <= self.budget:
            return (key_states, value_states)

        key_cache = key_states[:, :, -self.rkv_fixed_size :, :]
        value_cache = value_states[:, :, -self.rkv_fixed_size :, :]
        # compute attention scores
        query_cache = self.query_cache
        key_cache_rep = repeat_kv(key_cache, self.num_key_value_groups)

        attn_weights = (
            torch.matmul(query_cache, key_cache_rep.transpose(2, 3)) * self.scaling
        )  # (bsz, num_heads, q_len, kv_len)

        attn_weights_recent = attn_weights[:, :, :, : -self.rkv_window_size]
        # upcast attention to fp32
        attn_weights_recent = nn.functional.softmax(
            attn_weights_recent, dim=-1, dtype=torch.float32
        ).to(
            query_cache.dtype
        )  # (bsz, num_heads, rkv_window_size, kv_len - rkv_window_size)

        attn_weights_sum = attn_weights_recent.mean(dim=-2).to(
            query_cache.dtype
        )  # (bsz, num_kv_heads, kv_len - rkv_window_size)

        if self.num_key_value_groups != 1:
            attn_weights_sum = (
                attn_weights_sum.reshape(
                    bsz, key_states.shape[1], self.num_key_value_groups, -1
                )
                .max(dim=2)
                .values
            )  # (bsz, num_kv_heads, kv_len - rkv_window_size)
        # TODO: Softmax then reduce head

        attn_cache = F.max_pool1d(
            attn_weights_sum,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            stride=1,
        )

        #! allocate temporary memory (bsz, num_kv_heads, budget, budget), leading to high peaking memory
        similarity_cos = self.cal_similarity(
            key_cache,
        )[:, : -self.rkv_window_size]

        final_score = attn_cache * self.mix_lambda - similarity_cos * (
            1 - self.mix_lambda
        )

        # shape: (bsz, num_kv_heads, budget - window_size)
        if kv_cache_len > self.rkv_fixed_size:
            length = self.rkv_fixed_size - self.rkv_buf_size - self.rkv_window_size
        else:
            length = kv_cache_len - self.rkv_window_size  # keep > 0 under high sparsity
        indices = final_score.topk(length, dim=-1).indices

        indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

        k_past_compress = key_cache[:, :, : -self.rkv_window_size, :].gather(
            dim=2, index=indices
        )
        v_past_compress = value_cache[:, :, : -self.rkv_window_size, :].gather(
            dim=2, index=indices
        )
        k_cur = key_cache[:, :, -self.rkv_window_size :, :]
        v_cur = value_cache[:, :, -self.rkv_window_size :, :]
        if kv_cache_len > self.rkv_fixed_size:
            k_crop = torch.cat(
                [key_states[:, :, : -self.rkv_fixed_size, :], k_past_compress, k_cur],
                dim=2,
            )
            v_crop = torch.cat(
                [value_states[:, :, : -self.rkv_fixed_size, :], v_past_compress, v_cur],
                dim=2,
            )
        else:
            k_crop = torch.cat([k_past_compress, k_cur], dim=2)
            v_crop = torch.cat([v_past_compress, v_cur], dim=2)
        return (k_crop, v_crop)

    def reset(self):
        super().reset()

    def cal_similarity(
        self,
        key_states,
        threshold=0.5,
    ):
        k = key_states[0]  # ! bsz=1
        num_heads = k.shape[0]

        k_norm = k / (k.norm(dim=-1, keepdim=True) + 1e-8)
        similarity_cos = torch.matmul(k_norm, k_norm.transpose(-1, -2))

        similarity_cos.diagonal(dim1=-2, dim2=-1).zero_()

        # shape: [num_heads, seq_len, seq_len]
        similarity_mask = similarity_cos > threshold

        seq_len = similarity_mask.size(-1)
        k = int(seq_len * self.retain_ratio)

        indices = torch.where(
            similarity_mask,
            torch.arange(similarity_mask.size(-1), device=similarity_mask.device),
            torch.zeros_like(similarity_mask, dtype=torch.long),
        )

        # find the last True index in each row
        if self.retain_direction == "last":
            similarity_retain = torch.max(indices, dim=-1)[0]

        # find the first True index in each row
        elif self.retain_direction == "first":
            similarity_retain = torch.min(indices, dim=-1)[0]

        # keep the last_percent% elements
        elif self.retain_direction == "last_percent":
            similarity_retain = torch.topk(indices, k=k, dim=-1)[0][:, :, 0]

        # keep the first_percent% elements
        elif self.retain_direction == "first_percent":
            similarity_retain = torch.topk(indices, k=k, dim=-1, largest=False)[0][
                :, :, -1
            ]

        # create indices for zeroing
        batch_idx = (
            torch.arange(num_heads).unsqueeze(1).repeat(1, similarity_retain.size(1))
        )
        seq_idx = (
            torch.arange(similarity_retain.size(1)).unsqueeze(0).repeat(num_heads, 1)
        )

        # zero the specified positions in similarity_cos
        similarity_cos[batch_idx, seq_idx, similarity_retain] = 0

        return similarity_cos.mean(dim=1).softmax(dim=-1)


def enable_llama_rkv_eval(
    model: LlamaForCausalLM,
    budget_ratio: float,
    window_size: int,
):
    enable_tuple_kv_cache_for_llama(model)

    for _, layer in enumerate(model.model.layers):
        module = layer.self_attn

        module.kv_sampler = RKV(
            budget_ratio=budget_ratio,
            window_size=window_size,
            num_key_value_groups=module.num_key_value_groups,
            scaling=module.scaling,
        )

        module.forward = types.MethodType(llama_h2o_attention_forward, module)


def enable_qwen_rkv_eval(
    model: Qwen2ForCausalLM,
    budget_ratio: float,
    window_size: int,
):
    enable_tuple_kv_cache_for_qwen(model)

    for _, layer in enumerate(model.model.layers):
        module = layer.self_attn

        module.kv_sampler = RKV(
            budget_ratio=budget_ratio,
            window_size=window_size,
            num_key_value_groups=module.num_key_value_groups,
            scaling=module.scaling,
        )

        module.forward = types.MethodType(llama_h2o_attention_forward, module)


def enable_qwen3_rkv_eval(
    model: Qwen3ForCausalLM,
    budget_ratio: float,
    window_size: int,
):
    enable_tuple_kv_cache_for_qwen3(model)

    for _, layer in enumerate(model.model.layers):
        module = layer.self_attn

        module.kv_sampler = RKV(
            budget_ratio=budget_ratio,
            window_size=window_size,
            num_key_value_groups=module.num_key_value_groups,
            scaling=module.scaling,
        )

        module.forward = types.MethodType(qwen3_h2o_attention_forward, module)