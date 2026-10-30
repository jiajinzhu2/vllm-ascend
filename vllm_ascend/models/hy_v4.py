# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Ascend implementation of the public ``HYV4ForCausalLM`` architecture.

The model intentionally reuses vLLM 0.20.2's MLA/indexer modules and the
vLLM-Ascend sparse-MLA and MoE implementations.  Only the architecture-specific
pieces live here: full/shared indexer scheduling, gated MLA, the public iHC
residual topology and checkpoint name mapping.
"""

import typing
from collections.abc import Callable, Iterable, Sequence
from itertools import islice

import torch
from torch import nn
from transformers import PreTrainedConfig
from vllm.config import VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ColumnParallelLinear, ReplicatedLinear, RowParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mla import MLAModules
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader, maybe_remap_kv_scale_name
from vllm.model_executor.models.deepseek_v2 import (
    DeepSeekV2FusedQkvAProjLinear,
    Indexer,
    _try_load_fp8_indexer_wk,
)
from vllm.model_executor.models.interfaces import SupportsEagle, SupportsLoRA, SupportsPP
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

from vllm_ascend.models.deepseek_v4 import (
    DeepseekV2MixtureOfExperts,
    DeepseekV2MLP,
    DeepseekV4MoE,
)
from vllm_ascend.ops.mla import AscendMultiHeadLatentAttention

_SPARSE_LAYER_TYPES = {
    "sparse_attention",
    "sparse",
    "indexed_attention",
    "deepseek_sparse_attention",
}


def normalize_hyv4_config(config: PreTrainedConfig) -> PreTrainedConfig:
    """Populate aliases consumed by vLLM 0.20.2 and Ascend shared layers."""
    config.router_scaling_factor = getattr(config, "router_scaling_factor", config.routed_scaling_factor)
    config.num_experts = getattr(config, "num_experts", config.n_routed_experts)
    config.expert_hidden_dim = getattr(config, "expert_hidden_dim", config.moe_intermediate_size)
    config.num_shared_experts = getattr(config, "num_shared_experts", config.n_shared_experts)
    config.route_norm = getattr(config, "route_norm", config.norm_topk_prob)
    config.num_hash_layers = getattr(config, "num_hash_layers", 0)
    config.scoring_func = getattr(config, "scoring_func", "sigmoid")
    config.n_group = getattr(config, "n_group", 1)
    config.topk_group = getattr(config, "topk_group", 1)
    config.indexer_rope_interleave = True
    config.indexer_rope_at_end = True
    config.use_index_cache = True
    rope = getattr(config, "rope_parameters", None)
    if rope is None:
        rope = {"rope_type": "default", "rope_theta": getattr(config, "rope_theta", 10000.0)}
        config.rope_parameters = rope
    rope.setdefault("rope_type", "default")
    return config


def compute_skip_topk_layers(config: PreTrainedConfig) -> set[int]:
    """Return sparse backbone layers that reuse the preceding full indexer."""
    indexer_types = getattr(config, "indexer_types", None)
    if indexer_types is not None:
        return {idx for idx, kind in enumerate(indexer_types) if kind == "shared"}
    pattern = getattr(config, "index_topk_pattern", None)
    if pattern is not None:
        return {idx for idx, kind in enumerate(pattern) if kind == "S"}
    frequency = max(int(getattr(config, "index_topk_freq", 1)), 1)
    offset = int(getattr(config, "index_skip_topk_offset", 2))
    return {idx for idx in range(config.num_hidden_layers) if max(idx - offset + 1, 0) % frequency != 0}


class HYV4HCPreLayer(nn.Module):
    """Reduce ``hc_mult`` independent residual streams to one block input."""

    def __init__(self, config: PreTrainedConfig, prefix: str = ""):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult
        self.magnitude = config.hc_magnitude
        self.hc_eps = config.hc_eps
        self.norm_eps = config.rms_norm_eps
        self.hc_fn = ReplicatedLinear(
            config.hc_mult * config.hidden_size,
            2 * config.hc_mult,
            params_dtype=torch.float32,
            bias=False,
            prefix=f"{prefix}.hc_fn",
        )
        self.hc_scale = nn.Parameter(torch.empty(2, dtype=torch.float32))
        self.hc_base = nn.Parameter(torch.empty(2 * config.hc_mult, dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.constant_(self.hc_scale, 0.01)
        with torch.no_grad():
            base = -torch.log(torch.tensor(self.hc_mult - 1.0, dtype=self.hc_base.dtype))
            self.hc_base[: self.hc_mult].fill_(base)
            self.hc_base[self.hc_mult :].zero_()

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shape = hidden_states.shape
        flat = hidden_states.flatten(1).float()
        rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.norm_eps)
        gates = self.hc_fn(flat)[0] * rsqrt
        pre_raw, post_raw = gates.split(self.hc_mult, dim=-1)
        pre = torch.sigmoid(pre_raw * self.hc_scale[0] + self.hc_base[: self.hc_mult]) + self.hc_eps
        post = self.magnitude * torch.sigmoid(post_raw * self.hc_scale[1] + self.hc_base[self.hc_mult :]) + self.hc_eps
        reduced = torch.sum(pre.unsqueeze(-1) * flat.view(shape), dim=1)
        return reduced.to(hidden_states.dtype), post


class HYV4HCPostLayer(nn.Module):
    def forward(self, output: torch.Tensor, residual: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        dtype = output.dtype
        return (post.float().unsqueeze(-1) * output.float().unsqueeze(-2) + residual.float()).to(dtype)


class HYV4HCLayer(nn.Module):
    """One public iHC boundary around attention or FFN."""

    def __init__(self, config: PreTrainedConfig, prefix: str = ""):
        super().__init__()
        self.config = config
        self.enabled = bool(getattr(config, "enable_ihc", False))
        if self.enabled:
            self.hc_pre = HYV4HCPreLayer(config, f"{prefix}.hc_pre")
            self.hc_post = HYV4HCPostLayer()

    def prepare_input(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.enabled or hidden_states.ndim == 3:
            return hidden_states
        if hidden_states.shape[-1] == self.config.hidden_size:
            return hidden_states.unsqueeze(1).expand(-1, self.config.hc_mult, -1)
        return hidden_states.view(-1, self.config.hc_mult, self.config.hidden_size)

    def pre(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        if not self.enabled:
            return hidden_states, None, hidden_states
        reduced, post = self.hc_pre(hidden_states)
        return reduced, post, hidden_states

    def post(
        self,
        output: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self.enabled:
            return output + residual
        assert post is not None
        return self.hc_post(output, residual, post)


class HYV4HCHeadLayer(nn.Module):
    """Merge the independent residual streams before final RMSNorm."""

    def __init__(self, config: PreTrainedConfig, prefix: str = ""):
        super().__init__()
        self.config = config
        self.hc_head_fn = ReplicatedLinear(
            config.hc_mult * config.hidden_size,
            config.hc_mult,
            params_dtype=torch.float32,
            bias=False,
            prefix=f"{prefix}.hc_head_fn",
        )
        self.hc_head_base = nn.Parameter(torch.empty(config.hc_mult, dtype=torch.float32))
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))
        nn.init.constant_(self.hc_head_scale, 0.01)
        with torch.no_grad():
            base = -torch.log(torch.tensor(config.hc_mult - 1.0, dtype=self.hc_head_base.dtype))
            self.hc_head_base.fill_(base)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape, dtype = hidden_states.shape, hidden_states.dtype
        flat = hidden_states.flatten(1).float()
        rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.config.rms_norm_eps)
        gates = torch.sigmoid(self.hc_head_fn(flat)[0] * rsqrt * self.hc_head_scale + self.hc_head_base)
        gates = gates + self.config.hc_eps
        return torch.sum(gates.unsqueeze(-1) * flat.view(shape), dim=1).to(dtype)


class HYV4MLAAttention(nn.Module):
    """Public gated MLA wired to Ascend sparse MLA and lightning indexer."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        config: PreTrainedConfig,
        prefix: str,
        topk_indices_buffer: torch.Tensor | None,
        topk_capture_buffer: torch.Tensor | None = None,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        tp_size = get_tensor_model_parallel_world_size()
        self.num_local_heads = self.num_heads // tp_size
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.scaling = self.qk_head_dim**-0.5
        self.layer_idx = int(prefix.split(".")[-2])
        layer_types = getattr(config, "layer_types", ())
        self.is_sparse = bool(
            hasattr(config, "index_topk")
            and self.layer_idx < len(layer_types)
            and layer_types[self.layer_idx] in _SPARSE_LAYER_TYPES
        )
        self.skip_topk = self.is_sparse and self.layer_idx in compute_skip_topk_layers(config)
        quant_config = vllm_config.quant_config
        cache_config = vllm_config.cache_config

        self.fused_qkv_a_proj = DeepSeekV2FusedQkvAProjLinear(
            self.hidden_size,
            [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
            quant_config=quant_config,
            prefix=f"{prefix}.fused_qkv_a_proj",
        )
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = ColumnParallelLinear(
            self.q_lora_rank,
            self.num_heads * self.qk_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_b_proj",
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.rotary_emb = get_rope(
            self.qk_rope_head_dim,
            max_position=getattr(config, "max_position_embeddings", 8192),
            rope_parameters=config.rope_parameters,
            is_neox_style=False,
        )

        self.indexer = None
        self.indexer_rope_emb = None
        if self.is_sparse and not self.skip_topk:
            self.indexer_rope_emb = get_rope(
                self.qk_rope_head_dim,
                max_position=getattr(config, "max_position_embeddings", 8192),
                rope_parameters=config.rope_parameters,
                is_neox_style=False,
            )
            self.indexer = Indexer(
                vllm_config,
                config,
                self.hidden_size,
                self.q_lora_rank,
                quant_config,
                cache_config,
                topk_indices_buffer,
                f"{prefix}.indexer",
            )

        self.gated_mla = bool(getattr(config, "gated_mla", False))
        self.gating_type = getattr(config, "gating_type", "elementwise")
        gate_per_head = 1 if self.gating_type == "headwise" else self.v_head_dim
        self.linear_gate = (
            ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * gate_per_head,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.linear_gate",
            )
            if self.gated_mla
            else None
        )
        self.learnable_sink_param = (
            nn.Parameter(torch.empty(self.num_local_heads, dtype=torch.float32))
            if getattr(config, "learnable_sink", False)
            else None
        )

        modules = MLAModules(
            kv_a_layernorm=self.kv_a_layernorm,
            kv_b_proj=self.kv_b_proj,
            rotary_emb=self.rotary_emb,
            o_proj=self.o_proj,
            fused_qkv_a_proj=self.fused_qkv_a_proj,
            kv_a_proj_with_mqa=None,
            q_a_layernorm=self.q_a_layernorm,
            q_b_proj=self.q_b_proj,
            q_proj=None,
            indexer=self.indexer,
            indexer_rotary_emb=self.indexer_rope_emb,
            is_sparse=self.is_sparse,
            topk_indices_buffer=topk_indices_buffer,
        )
        self.mla_attn = AscendMultiHeadLatentAttention(
            self.hidden_size,
            self.num_local_heads,
            self.scaling,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim,
            self.q_lora_rank,
            self.kv_lora_rank,
            modules,
            cache_config,
            quant_config,
            prefix,
            skip_topk=self.skip_topk,
            linear_gate=self.linear_gate,
            gating_type=self.gating_type,
            sinks=self.learnable_sink_param,
            topk_capture_buffer=topk_capture_buffer,
        )

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.mla_attn(positions, hidden_states)

    def draft_project(self, hidden_states: torch.Tensor):
        """Projection view reused by causal DFlash; no target weights are copied."""
        leading = hidden_states.shape[:-1]
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        qkv = self.fused_qkv_a_proj(flat)[0]
        q_c, kv = qkv.split([self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1)
        q = self.q_b_proj(self.q_a_layernorm(q_c))[0]
        q = q.view(*leading, self.num_local_heads, self.qk_head_dim)
        latent, k_rope = kv.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        return (
            q[..., : self.qk_nope_head_dim],
            q[..., self.qk_nope_head_dim :],
            self.kv_a_layernorm(latent).view(*leading, self.kv_lora_rank),
            k_rope.view(*leading, self.qk_rope_head_dim),
        )

    def draft_rotary(self, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor):
        if positions.ndim == 3:
            positions = positions[0]
        positions = positions.reshape(-1)
        q_shape, k_shape = query.shape, key.shape
        q, k = self.rotary_emb(
            positions,
            query.reshape(-1, self.num_local_heads, self.qk_rope_head_dim),
            key.reshape(-1, 1, self.qk_rope_head_dim),
        )
        return q.view(q_shape), k.view(k_shape)

    def draft_finish(self, hidden_states: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        flat_output = output.reshape(-1, self.num_local_heads * self.v_head_dim)
        if self.linear_gate is not None:
            gate = torch.sigmoid(self.linear_gate(flat_hidden)[0])
            if self.gating_type == "headwise":
                gated = flat_output.view(-1, self.num_local_heads, self.v_head_dim) * gate.unsqueeze(-1)
                flat_output = gated.flatten(-2)
            else:
                flat_output = flat_output * gate
        result = self.o_proj(flat_output)[0]
        return result.view(*hidden_states.shape[:-1], self.hidden_size)

    def draft_kv_up_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        impl = self.mla_attn.mla_attn.impl
        if hasattr(impl, "W_UK_T") and hasattr(impl, "W_UV"):
            return impl.W_UK_T, impl.W_UV.transpose(-1, -2)
        weight = self.kv_b_proj.weight.view(
            self.num_local_heads,
            self.qk_nope_head_dim + self.v_head_dim,
            self.kv_lora_rank,
        )
        return weight[:, : self.qk_nope_head_dim], weight[:, self.qk_nope_head_dim :]


class HYV4DecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        topk_indices_buffer: torch.Tensor | None,
        topk_capture_buffer: torch.Tensor | None = None,
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.layer_idx = int(prefix.split(".")[-1])
        self.self_attn = HYV4MLAAttention(
            vllm_config,
            config,
            f"{prefix}.self_attn",
            topk_indices_buffer,
            topk_capture_buffer,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if config.mlp_layer_types[self.layer_idx] == "dense":
            self.mlp = DeepseekV2MLP(
                config.hidden_size,
                config.intermediate_size,
                config.hidden_act,
                vllm_config.quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = DeepseekV4MoE(
                config,
                vllm_config.parallel_config,
                vllm_config.quant_config,
                prefix=f"{prefix}.mlp",
            )
        self.hc_attn_layer = HYV4HCLayer(config, f"{prefix}.hc_attn_layer")
        self.hc_mlp_layer = HYV4HCLayer(config, f"{prefix}.hc_mlp_layer")

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if getattr(self.config, "enable_ihc", False):
            hidden_states = self.hc_attn_layer.prepare_input(hidden_states)
            hidden_states, post, residual_hc = self.hc_attn_layer.pre(hidden_states)
            hidden_states = self.self_attn(positions, self.input_layernorm(hidden_states))
            hidden_states = self.hc_attn_layer.post(hidden_states, residual_hc, post)
            hidden_states = self.hc_mlp_layer.prepare_input(hidden_states)
            hidden_states, post, residual_hc = self.hc_mlp_layer.pre(hidden_states)
            hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
            return self.hc_mlp_layer.post(hidden_states, residual_hc, post), None

        if residual is not None:
            hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.self_attn(positions, self.input_layernorm(hidden_states))
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, residual


class HYV4Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = normalize_hyv4_config(vllm_config.model_config.hf_config)
        self.config = config
        self.device = current_platform.device_type
        self.enable_ihc = bool(getattr(config, "enable_ihc", False))
        self.topk_indices_buffer = None
        if hasattr(config, "index_topk"):
            self.topk_indices_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                config.index_topk,
                dtype=torch.int32,
                device=self.device,
            )

        dflash_config = {}
        speculative = vllm_config.speculative_config
        if speculative is not None and speculative.draft_model_config is not None:
            dflash_config = getattr(speculative.draft_model_config.hf_config, "dflash_config", {}) or {}
        capture_layer_ids = tuple(dflash_config.get("target_layer_ids", ()))
        self.hy_dflash_topk_buffers: dict[int, torch.Tensor] = {}
        if self.topk_indices_buffer is not None:
            for layer_id in capture_layer_ids:
                buffer = torch.empty_like(self.topk_indices_buffer)
                self.register_buffer(f"hy_dflash_topk_{layer_id}", buffer, persistent=False)
                self.hy_dflash_topk_buffers[layer_id] = buffer

        self.embed_tokens = (
            VocabParallelEmbedding(config.vocab_size, config.hidden_size, prefix=f"{prefix}.embed_tokens")
            if get_pp_group().is_first_rank or (config.tie_word_embeddings and get_pp_group().is_last_rank)
            else PPMissingLayer()
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda layer_prefix: HYV4DecoderLayer(
                vllm_config,
                layer_prefix,
                self.topk_indices_buffer,
                self.hy_dflash_topk_buffers.get(int(layer_prefix.split(".")[-1])),
            ),
            prefix=f"{prefix}.layers",
        )
        if get_pp_group().is_last_rank:
            self.hc_head = HYV4HCHeadLayer(config, f"{prefix}.hc_head") if self.enable_ihc else None
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.hc_head = self.norm = PPMissingLayer()
        intermediate_size = config.hc_mult * config.hidden_size if self.enable_ihc else config.hidden_size
        keys = ["hidden_states"] if self.enable_ihc else ["hidden_states", "residual"]
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(keys, intermediate_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = None if self.enable_ihc else intermediate_tensors["residual"]
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(positions, hidden_states, residual)
        if not get_pp_group().is_last_rank:
            values = {"hidden_states": hidden_states.flatten(1) if self.enable_ihc else hidden_states}
            if not self.enable_ihc:
                values["residual"] = residual
            return IntermediateTensors(values)
        if self.enable_ihc:
            hidden_states = self.hc_head(hidden_states)
        else:
            hidden_states = hidden_states + residual
        return self.norm(hidden_states)


class AscendHYV4ForCausalLM(
    nn.Module,
    SupportsPP,
    DeepseekV2MixtureOfExperts,
    SupportsLoRA,
    SupportsEagle,
):
    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
        "wk_weights_proj": ["wk", "weights_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = normalize_hyv4_config(vllm_config.model_config.hf_config)
        self.quant_config = vllm_config.quant_config
        if self.quant_config is not None:
            self.quant_config.packed_modules_mapping = self.packed_modules_mapping
        self.model = HYV4Model(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        self.lm_head = (
            ParallelLMHead(
                self.config.vocab_size,
                self.config.hidden_size,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if get_pp_group().is_last_rank
            else PPMissingLayer()
        )
        if get_pp_group().is_last_rank and self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors
        self.set_moe_parameters()

    def set_moe_parameters(self):
        self.expert_weights = []
        self.num_expert_groups = getattr(self.config, "n_group", 1)
        self.moe_layers = []
        self.moe_mlp_layers = []
        example = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer) or not isinstance(layer.mlp, DeepseekV4MoE):
                continue
            example = layer.mlp
            self.moe_mlp_layers.append(layer.mlp)
            self.moe_layers.append(layer.mlp.experts)
        self.extract_moe_parameters(example)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        if getattr(self.config, "soft_logits_capping", False):
            soft_cap = self.config.soft_logits_capping_logits
            logits = soft_cap * torch.nn.functional.tanh(logits / soft_cap)
        return logits

    def get_hy_dflash_layer_names(self, layer_ids: Sequence[int]) -> list[str]:
        return [self.model.layers[layer_id].self_attn.mla_attn.mla_attn.layer_name for layer_id in layer_ids]

    def get_hy_dflash_attention(self, layer_id: int) -> HYV4MLAAttention:
        return self.model.layers[layer_id].self_attn

    def get_hy_dflash_topk_buffer(self, layer_id: int) -> torch.Tensor:
        return self.model.hy_dflash_topk_buffers[layer_id]

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return FusedMoE.make_expert_params_mapping(
            self.model,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
            num_redundant_experts=self.num_redundant_experts,
        )

    @staticmethod
    def _load_fused_experts(
        name: str,
        weight: torch.Tensor,
        params: dict[str, nn.Parameter],
        expert_mapping: Sequence[tuple[str, str, int, str]],
        loaded: set[str],
    ) -> bool:
        """Load public 3-D expert tensors through 0.20.2's expert loader."""
        gate_up = ".experts.gate_up_proj" in name
        down = ".experts.down_proj" in name
        if not gate_up and not down:
            return False
        layer_prefix = name.split(".experts.", 1)[0]
        for param_name, weight_name, expert_id, shard_id in expert_mapping:
            if gate_up and shard_id not in ("w1", "w3"):
                continue
            if down and shard_id != "w2":
                continue
            logical_id = int(weight_name.split("experts.", 1)[1].split(".", 1)[0])
            expert_weight = weight[logical_id]
            if gate_up:
                expert_weight = expert_weight.chunk(2, dim=-2)[0 if shard_id == "w1" else 1]
            candidate = f"{layer_prefix}.{param_name}weight"
            if candidate not in params:
                continue
            param = params[candidate]
            loader = typing.cast(Callable[..., bool], param.weight_loader)
            if loader(
                param,
                expert_weight,
                candidate,
                shard_id=shard_id,
                expert_id=expert_id,
                return_success=True,
            ):
                loaded.add(candidate)
        return True

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        pending_wk_fp8: dict[str, dict[str, torch.Tensor]] = {}
        skip_topk = compute_skip_topk_layers(self.config)
        expert_mapping = self.get_expert_mapping()
        stacked = (
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            ("fused_qkv_a_proj", "q_a_proj", 0),
            ("fused_qkv_a_proj", "kv_a_proj_with_mqa", 1),
            ("wk_weights_proj", "wk", 0),
            ("wk_weights_proj", "weights_proj", 1),
        )
        tp_rank = get_tensor_model_parallel_rank()
        heads_per_rank = self.config.num_attention_heads // get_tensor_model_parallel_world_size()

        for name, weight in weights:
            if name.startswith("model.mtp_layers."):
                continue
            if ".indexer." in name:
                pieces = name.split(".layers.", 1)
                if len(pieces) == 2 and int(pieces[1].split(".", 1)[0]) in skip_topk:
                    continue
            if not name.startswith(("model.", "lm_head.")):
                name = f"model.{name}"
            name = name.replace(".mlp.router.gate.", ".mlp.gate.")
            name = name.replace(".mlp.gate.wg.", ".mlp.gate.")
            name = name.replace(".mlp.shared_mlp.", ".mlp.shared_experts.")
            name = name.replace(".mlp.expert_bias", ".mlp.gate.e_score_correction_bias")
            name = name.replace(".mlp.e_score_correction_bias", ".mlp.gate.e_score_correction_bias")
            if _try_load_fp8_indexer_wk(name, weight, pending_wk_fp8, params, loaded):
                continue
            if "hc_fn" in name and not name.endswith(".weight"):
                name = name.replace("hc_fn", "hc_fn.weight")
            if "hc_head_fn" in name and not name.endswith(".weight"):
                name = name.replace("hc_head_fn", "hc_head_fn.weight")
            if name.endswith(".sink"):
                name = name[: -len(".sink")] + ".learnable_sink_param"
            if name.endswith(".learnable_sink"):
                name = name[: -len(".learnable_sink")] + ".learnable_sink_param"
            if name.endswith(".attn_sink"):
                name = name[: -len(".attn_sink")] + ".learnable_sink_param"
            if name.endswith("learnable_sink_param"):
                param = params[name]
                param.data.copy_(weight.narrow(0, tp_rank * heads_per_rank, heads_per_rank))
                loaded.add(name)
                continue

            if self._load_fused_experts(name, weight, params, expert_mapping, loaded):
                continue

            mapped = False
            for param_name, weight_name, shard_id in stacked:
                if weight_name not in name or ".experts." in name:
                    continue
                candidate = name.replace(weight_name, param_name)
                if candidate not in params:
                    continue
                param = params[candidate]
                param.weight_loader(param, weight, shard_id)
                loaded.add(candidate)
                mapped = True
                break
            if mapped:
                continue

            is_expert = False
            for param_name, weight_name, expert_id, shard_id in expert_mapping:
                if weight_name not in name:
                    continue
                is_expert = True
                candidate = name.replace(weight_name, param_name)
                if is_pp_missing_parameter(candidate, self):
                    break
                param = params[candidate]
                loader = typing.cast(Callable[..., bool], param.weight_loader)
                if loader(param, weight, candidate, shard_id=shard_id, expert_id=expert_id, return_success=True):
                    loaded.add(candidate)
                break
            if is_expert:
                continue

            if name.endswith(".bias") and name not in params:
                continue
            name = maybe_remap_kv_scale_name(name, params)
            if name is None or is_pp_missing_parameter(name, self) or name not in params:
                continue
            param = params[name]
            loader = getattr(param, "weight_loader", default_weight_loader)
            loader(param, weight)
            loaded.add(name)
        return loaded


__all__ = [
    "AscendHYV4ForCausalLM",
    "HYV4DecoderLayer",
    "HYV4HCLayer",
    "HYV4MLAAttention",
    "HYV4Model",
    "compute_skip_topk_layers",
]
