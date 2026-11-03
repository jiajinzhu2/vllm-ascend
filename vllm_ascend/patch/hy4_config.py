# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""HY4 configuration compatibility for vLLM 0.20.2.

Layer numbers are checkpoint indices. In particular, target_layer_ids is NOT
an alias for target_kv_layers: a draft may select hidden features from layer 60
while reading the full indexer and KV of layer 61.
"""

from transformers import PretrainedConfig

HY_DFLASH_ARCHITECTURES = (
    "hy4dflymodels",
    "HY4DflyModels",
)


def is_hy_dflash(config) -> bool:
    names = getattr(config, "architectures", ()) or ()
    return any(name.removeprefix("DFlash").lower() == "hy4dflymodels" for name in names) or (
        (getattr(config, "dflash_config", None) or {}).get("ascend_proposer") == "hy"
    )


def get_hy_dflash_config(config) -> dict:
    """Read both top-level checkpoint fields and the DFlash sub-configuration."""
    options = dict(getattr(config, "dflash_config", None) or {})
    for key in (
        "target_layer_ids",
        "target_kv_layers",
        "target_index_layers",
        "target_init_attn_layers",
        "context_topk",
        "context_local_window",
        "mask_token_id",
        "dflash_causal_attn",
        "mtp_fusion_slot0",
    ):
        value = getattr(config, key, None)
        if value is not None:
            if key in options and options[key] != value:
                raise ValueError(f"Conflicting HY DFlash config field: {key}")
            options[key] = value
    if is_hy_dflash(config):
        options.setdefault("ascend_proposer", "hy")
        options.setdefault("context_topk", 2048)
        options.setdefault("context_local_window", 8)
        options.setdefault("mask_token_id", 120009)
        options.setdefault("dflash_causal_attn", True)
        options.setdefault("mtp_fusion_slot0", True)
    # Legacy adapters used one explicitly supplied source-layer list.
    layers = options.get("target_layer_ids", ())
    options.setdefault("target_kv_layers", layers)
    options.setdefault("target_index_layers", options["target_kv_layers"])
    options.setdefault("target_init_attn_layers", options["target_kv_layers"])
    options.setdefault("ascend_attention_backend", "auto")
    if options["ascend_attention_backend"] not in ("auto", "sfa", "reference"):
        raise ValueError("HY ascend_attention_backend must be auto, sfa or reference")
    count = getattr(config, "num_hidden_layers", len(layers))
    for key in ("target_layer_ids", "target_kv_layers", "target_index_layers", "target_init_attn_layers"):
        values = options[key]
        if len(values) != count or any(type(i) is not int or i < 0 for i in values):
            raise ValueError(f"HY requires {count} nonnegative checkpoint indices in {key}")
    if not layers or options.get("context_topk", 0) <= 0 or options.get("context_local_window", -1) < 0:
        raise ValueError("HY requires source layers, positive context_topk and nonnegative context_local_window")
    if not options.get("dflash_causal_attn", True):
        raise ValueError("HY KV-share draft requires dflash_causal_attn=true")
    return options


class HYDflashConfig(PretrainedConfig):
    model_type = "dflash"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if is_hy_dflash(self):
            self.dflash_config = get_hy_dflash_config(self)


class HYV4Config(PretrainedConfig):
    """Public HYV4 fields with the pre-dataclass Transformers config API.

    Dimensions and quantization metadata must come from the actual checkpoint;
    this class only supplies architecture defaults used by the public model.
    """

    model_type = "hy_v4"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        n = getattr(self, "num_hidden_layers", 0)
        defaults = dict(
            mlp_layer_types=["dense"] + ["sparse"] * max(n - 1, 0),
            layer_types=["deepseek_sparse_attention"] * n,
            indexer_types=["full" if i == 0 or (i - 1) % 4 == 0 else "shared" for i in range(n)],
            index_topk=2048,
            index_head_dim=128,
            index_n_heads=16,
            enable_ihc=True,
            hc_mult=4,
            hc_magnitude=2.0,
            hc_eps=1e-6,
            gated_mla=True,
            gating_type="elementwise",
            learnable_sink=True,
            rms_norm_eps=1e-5,
            hidden_act="silu",
            attention_bias=False,
        )
        for key, value in defaults.items():
            if getattr(self, key, None) is None:
                setattr(self, key, value)


def register_hy4_configs() -> None:
    # This runs with the general model plugin, before ModelConfig is built.
    # Keep vendor/installed upstream registrations when they already exist.
    from vllm.transformers_utils.config import _CONFIG_REGISTRY
    from vllm.transformers_utils.model_arch_config_convertor import (
        MODEL_ARCH_CONFIG_CONVERTORS,
        ModelArchConfigConvertorBase,
    )

    class HYV4ModelArchConfigConvertor(ModelArchConfigConvertorBase):
        def is_deepseek_mla(self) -> bool:
            return getattr(self.hf_text_config, "kv_lora_rank", None) is not None

    _CONFIG_REGISTRY.setdefault("hy_v4", HYV4Config)
    _CONFIG_REGISTRY.setdefault("dflash", HYDflashConfig)
    MODEL_ARCH_CONFIG_CONVERTORS.setdefault("hy_v4", HYV4ModelArchConfigConvertor)
