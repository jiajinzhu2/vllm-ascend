# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint namespace conversion, independent of the model/runtime imports."""

import re


def remap_dspark_weight(name: str, first_layer: int, num_layers: int) -> str | None:
    if first_layer < 0 or num_layers <= 0:
        raise ValueError("Invalid DSpark layer range")
    match = re.fullmatch(r"mtp\.(\d+)\.(.+)", name)
    if match is None:
        return None
    stage, rest = int(match[1]), match[2]
    if stage >= num_layers:
        raise ValueError(f"DSpark checkpoint stage {stage} exceeds {num_layers} layers")
    if stage == num_layers - 1 and rest.startswith("confidence_head."):
        mapped = f"model.{rest}"
    elif stage == 0 and rest == "embed.weight":
        mapped = "model.embed_tokens.weight"
    elif stage == num_layers - 1 and rest == "head.weight":
        mapped = "lm_head.weight"
    elif rest.startswith(("hc_head_fn", "hc_head_base", "hc_head_scale")):
        mapped = f"model.{rest}"
    else:
        if rest.startswith(("main_proj.", "main_norm.")):
            layer = first_layer
        elif rest.startswith(("norm.", "markov_head.")):
            layer = first_layer + num_layers - 1
        else:
            layer = first_layer + stage
        mapped = f"model.layers.{layer}.{rest}"
        for old, new in (
            (".attn.", ".self_attn."),
            (".ffn_norm.", ".post_attention_layernorm."),
            (".attn_norm.", ".input_layernorm."),
            (".ffn.", ".mlp."),
            (".w1.", ".gate_proj."),
            (".w2.", ".down_proj."),
            (".w3.", ".up_proj."),
            (".mlp.gate.bias", ".mlp.gate.e_score_correction_bias"),
        ):
            mapped = mapped.replace(old, new)
    return (
        mapped.replace(".markov_head.embed.weight", ".markov_head.markov_w1.weight")
        .replace(".markov_head.head.weight", ".markov_head.markov_w2.weight")
        .replace("model.confidence_head.weight", "model.confidence_head.proj.weight")
    )
