# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""BF16 HY4DflyModels checkpoint wrapper for the 0.20.2 model loader.

Uses the backported HY4 projections without constructing draft KV-cache or
indexer-cache layers. No parameters are initialized from quantized target
attention weights: target_init_attn_layers describes training initialization.
"""

from collections.abc import Iterable
from copy import copy

import torch
from torch import nn
from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_ascend.models.deepseek_v4 import DeepseekV2MLP
from vllm_ascend.models.hy_v4 import HYV4MLAAttention
from vllm_ascend.patch.hy4_config import get_hy_dflash_config
from vllm_ascend.patch.worker.hy_dflash import HYDflashAdapter, HYMTPFusion


class HYDflashDenseLayer(nn.Module):
    def __init__(self, vllm_config, config, prefix):
        super().__init__()
        self.self_attn = HYV4MLAAttention(vllm_config, config, f"{prefix}.self_attn", None, draft_only=True)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = DeepseekV2MLP(
            config.hidden_size, config.intermediate_size, config.hidden_act, None, prefix=f"{prefix}.mlp"
        )


class HY4DflyModels(nn.Module):
    """Dense causal DFlash, slot-0 fusion, full vocabulary, no auxiliary heads.

    Supported canonical checkpoint keys are documented in hy_dflash_prototype.md.
    ``dflash_config.weight_name_mapping`` can explicitly translate vendor keys;
    unknown/missing trained tensors fail loading instead of leaving random data.
    """

    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
    }

    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        self.config = config = copy(vllm_config.model_config.hf_config)
        options = get_hy_dflash_config(config)
        if vllm_config.quant_config is not None or vllm_config.model_config.dtype != torch.bfloat16:
            raise ValueError("HY4DflyModels must be loaded with BF16 parameters and quant_config=None")
        if config.num_hidden_layers != 5:
            raise ValueError("This HY4DflyModels adapter expects the five-layer dense checkpoint")
        if not options.get("mtp_fusion_slot0", True):
            raise ValueError("This checkpoint adapter requires slot-0 MTP fusion")
        for flag in ("use_markov_head", "use_confidence_head", "use_hidden_correction"):
            if getattr(config, flag, False) or options.get(flag, False):
                raise ValueError(f"This HY4DflyModels checkpoint does not implement {flag}")
        config.layer_types = ["sparse_attention"] * config.num_hidden_layers
        config.indexer_types = ["full"] * config.num_hidden_layers
        config.hidden_act = getattr(config, "hidden_act", "silu")
        config.gated_mla = getattr(config, "gated_mla", True)
        config.rope_parameters = dict(
            getattr(config, "rope_parameters", None) or getattr(config, "rope_scaling", None) or {}
        )
        config.rope_parameters.setdefault("rope_type", "default")
        config.rope_parameters.setdefault("rope_theta", getattr(config, "rope_theta", 10000.0))
        model_prefix = f"{prefix}.model" if prefix else "model"
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [
                HYDflashDenseLayer(vllm_config, config, f"{model_prefix}.layers.{i}")
                for i in range(config.num_hidden_layers)
            ]
        )
        self.model.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.model.fusion = nn.Module()
        self.model.fusion.embedding_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.model.fusion.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.model.fusion.eh_proj = ReplicatedLinear(
            2 * config.hidden_size,
            config.hidden_size,
            bias=False,
            quant_config=None,
            prefix=f"{model_prefix}.fusion.eh_proj",
        )
        self.model.fusion.gate = nn.Parameter(torch.empty(config.hidden_size))
        # These may be replaced by target modules after strict backbone loading.
        self.model.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size, prefix=f"{model_prefix}.embed_tokens"
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size, config.hidden_size, prefix=f"{prefix}.lm_head" if prefix else "lm_head"
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self._has_embedding = self._has_lm_head = False
        self._weights_loaded = False

    def embed_input_ids(self, input_ids):
        return self.model.embed_tokens(input_ids)

    def compute_logits(self, hidden_states):
        return self.logits_processor(self.lm_head, hidden_states)

    def forward(self, *args, **kwargs):
        raise RuntimeError("HY4DflyModels executes through AscendHYDflashProposer with target context")

    def to_hy_dflash_adapter(self, target_model):
        if not self._weights_loaded:
            raise RuntimeError("HY draft weights must load completely before sharing target modules")
        if (
            target_model.config.hidden_size != self.config.hidden_size
            or target_model.config.vocab_size != self.config.vocab_size
        ):
            raise ValueError("HY target and draft must have the same hidden size and vocabulary")
        options = get_hy_dflash_config(self.config)
        for layer_id in (*options["target_kv_layers"], *options["target_index_layers"]):
            if layer_id >= target_model.config.num_hidden_layers:
                raise ValueError(f"HY target layer {layer_id} is outside the target backbone")
        for layer_id in options["target_kv_layers"]:
            target_attn = target_model.get_hy_dflash_attention(layer_id)
            if (target_attn.kv_lora_rank, target_attn.qk_rope_head_dim) != (
                self.config.kv_lora_rank,
                self.config.qk_rope_head_dim,
            ):
                raise ValueError("HY target/draft latent KV and RoPE dimensions must match")
        if not self._has_embedding:
            self.model.embed_tokens = target_model.model.embed_tokens
        if not self._has_lm_head:
            self.lm_head = target_model.lm_head
        fusion = self.model.fusion
        return HYDflashAdapter.from_hyv4_backbone(
            self.model.layers,
            embed_tokens=self.model.embed_tokens,
            final_norm=self.model.norm,
            fusion=HYMTPFusion(fusion.embedding_norm, fusion.hidden_norm, fusion.eh_proj, fusion.gate),
            correction=None,
            confidence_head=None,
            # Preserve target capping/quantization semantics for the shared head.
            compute_logits=self.compute_logits if self._has_lm_head else target_model.compute_logits,
            hidden_size=self.config.hidden_size,
            target_layer_ids=options["target_kv_layers"],
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        self._weights_loaded = False
        params = dict(self.named_parameters())
        loaded = set()
        shards = {}
        mapping = get_hy_dflash_config(self.config).get("weight_name_mapping", {})
        stacked = (
            (".mlp.gate_proj.", ".mlp.gate_up_proj.", 0),
            (".mlp.up_proj.", ".mlp.gate_up_proj.", 1),
            (".self_attn.q_a_proj.", ".self_attn.fused_qkv_a_proj.", 0),
            (".self_attn.kv_a_proj_with_mqa.", ".self_attn.fused_qkv_a_proj.", 1),
        )
        tp_rank = get_tensor_model_parallel_rank()
        local_heads = self.config.num_attention_heads // get_tensor_model_parallel_world_size()
        for original, weight in weights:
            name = mapping.get(original, original)
            if not name.startswith(("model.", "lm_head.")):
                name = "model." + name
            if ".indexer." in name or name.endswith(".rotary_emb.inv_freq"):
                # Draft selection comes from target_index_layers at anchor-1.
                continue
            if weight.dtype not in (torch.bfloat16, torch.float16, torch.float32):
                raise ValueError(f"HY draft weight {original} is {weight.dtype}; expected an unquantized checkpoint")
            shard = None
            for source, dest, shard_id in stacked:
                if source in name:
                    name = name.replace(source, dest)
                    shard = shard_id
                    break
            if name not in params:
                raise ValueError(f"Unknown HY draft weight {original}; provide an explicit weight_name_mapping")
            param = params[name]
            loader = getattr(param, "weight_loader", default_weight_loader)
            if shard is not None:
                loader(param, weight, shard)
                shards.setdefault(name, set()).add(shard)
            elif name.endswith("learnable_sink_param"):
                default_weight_loader(param, weight.narrow(0, tp_rank * local_heads, local_heads))
            else:
                loader(param, weight)
            loaded.add(name)
        optional = {"model.embed_tokens.weight", "lm_head.weight"}
        missing = set(params) - loaded - optional
        missing.update(name for name, parts in shards.items() if parts != {0, 1})
        if missing:
            raise ValueError("Missing HY draft weights/shards: " + ", ".join(sorted(missing)))
        self._has_embedding = "model.embed_tokens.weight" in loaded
        self._has_lm_head = "lm_head.weight" in loaded
        self._weights_loaded = True
        # Validation above handles optional shared parameters explicitly. Return
        # None to opt out of DefaultModelLoader's all-parameters-loaded check.
