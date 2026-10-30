# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Composition adapter for an already-loaded HY DFlash draft, not a new loader.

No global patch or fictitious checkpoint architecture is registered. The vendor
draft wrapper supplies loaded modules via to_hy_dflash_adapter(target_model).
"""

from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from vllm_ascend.attention.hy_dflash import HY_BLOCK_SIZE, HY_TARGET_LAYERS, HYContext, HYDflashAttention


def linear_output(module: nn.Module, value: torch.Tensor) -> torch.Tensor:
    """Accept PyTorch Linear and vLLM's (output, deferred_bias) linear API."""
    result = module(value)
    if isinstance(result, tuple):
        output, bias = result
        return output if bias is None else output + bias
    return result


class HYMTPFusion(nn.Module):
    def __init__(self, embedding_norm: nn.Module, hidden_norm: nn.Module, eh_proj: nn.Module, gate: torch.Tensor):
        super().__init__()
        self.embedding_norm, self.hidden_norm, self.eh_proj = embedding_norm, hidden_norm, eh_proj
        self.register_buffer("gate", gate, persistent=False)

    def forward(self, anchor_embedding: torch.Tensor, target_final_hidden: torch.Tensor) -> torch.Tensor:
        if self.gate.shape != (anchor_embedding.shape[-1],):
            raise ValueError("MTP fusion requires a per-channel gate")
        joined = torch.cat((self.embedding_norm(anchor_embedding), self.hidden_norm(target_final_hidden)), dim=-1)
        return anchor_embedding + self.gate * linear_output(self.eh_proj, joined)


class HYHiddenCorrection(nn.Module):
    def __init__(
        self,
        hidden_norm: nn.Module,
        embedding_norm: nn.Module,
        gate_proj: nn.Module,
        up_proj: nn.Module,
        down_proj: nn.Module,
    ):
        super().__init__()
        self.hidden_norm, self.embedding_norm = hidden_norm, embedding_norm
        self.gate_proj, self.up_proj, self.down_proj = gate_proj, up_proj, down_proj

    def forward(self, hidden: torch.Tensor, previous_embedding: torch.Tensor) -> torch.Tensor:
        joined = torch.cat((self.hidden_norm(hidden), self.embedding_norm(previous_embedding)), dim=-1)
        update = F.silu(linear_output(self.gate_proj, joined)) * linear_output(self.up_proj, joined)
        return hidden + linear_output(self.down_proj, update)


class HYDenseDraftLayer(nn.Module):
    def __init__(
        self, input_norm: nn.Module, attention: HYDflashAttention, post_attention_norm: nn.Module, mlp: nn.Module
    ):
        super().__init__()
        self.input_norm, self.attention = input_norm, attention
        self.post_attention_norm, self.mlp = post_attention_norm, mlp

    def forward(self, hidden: torch.Tensor, positions: torch.Tensor, context: HYContext) -> torch.Tensor:
        shape = hidden.shape
        normalized = self.input_norm(hidden.flatten(0, 1)).view(shape)
        hidden = hidden + self.attention(normalized, positions, context)
        update = self.mlp(self.post_attention_norm(hidden.flatten(0, 1)))
        return hidden + update.view(shape)


class HYDflashAdapter(nn.Module):
    """Five loaded dense layers, slot-0 fusion, correction and confidence.

    The target supplies embeddings/LM head through the checkpoint's normal
    sharing policy. final_hidden is already target-final-normalized. The fusion
    hidden_norm is the *additional* normalization in the specified MTP formula.
    Checkpoint names and the concrete HY4 projection/rotary/finish wrappers must
    come from the existing model implementation, not guesses in this adapter.
    """

    hy_dflash_contract_version = 2

    def __init__(
        self,
        layers: Sequence[HYDenseDraftLayer],
        embed_tokens: nn.Module,
        final_norm: nn.Module,
        fusion: HYMTPFusion,
        correction: HYHiddenCorrection,
        confidence_head: nn.Module,
        compute_logits: Callable[[torch.Tensor], torch.Tensor],
        hidden_size: int = 6144,
    ):
        super().__init__()
        if len(layers) != len(HY_TARGET_LAYERS):
            raise ValueError("HY KV-share draft requires five dense layers")
        self.layers = nn.ModuleList(layers)
        self.embed_tokens, self.final_norm = embed_tokens, final_norm
        self.fusion, self.correction, self.confidence_head = fusion, correction, confidence_head
        self.logits_callback = compute_logits
        self.hidden_size = hidden_size

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        target_final_hidden: torch.Tensor,
        contexts: tuple[HYContext, ...],
    ) -> torch.Tensor:
        batch = input_ids.shape[0]
        if input_ids.shape != (batch, HY_BLOCK_SIZE) or positions.shape != (3, batch, HY_BLOCK_SIZE):
            raise ValueError("HY DFlash requires [B,8] tokens and [3,B,8] M-RoPE positions")
        if target_final_hidden.shape != (batch, self.hidden_size):
            raise ValueError("Expected target final-norm hidden at anchor-1")
        if tuple(context.layer_id for context in contexts) != HY_TARGET_LAYERS:
            raise ValueError("HY context layers must be 61,65,69,73,77 in order")
        hidden = self.embed_tokens(input_ids.reshape(-1)).view(batch, HY_BLOCK_SIZE, self.hidden_size)
        anchor = self.fusion(hidden[:, 0], target_final_hidden)
        hidden = torch.cat((anchor[:, None], hidden[:, 1:]), dim=1)
        for layer, context in zip(self.layers, contexts):
            hidden = layer(hidden, positions, context)
        return self.final_norm(hidden.flatten(0, 1)).view(batch, HY_BLOCK_SIZE, self.hidden_size)

    def correct_hidden_states(self, hidden: torch.Tensor, previous_token_ids: torch.Tensor) -> torch.Tensor:
        return self.correction(hidden, self.embed_tokens(previous_token_ids))

    def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.logits_callback(hidden)

    def predict_confidence(self, corrected_hidden: torch.Tensor) -> torch.Tensor:
        logits = linear_output(self.confidence_head, corrected_hidden)
        if logits.shape != (*corrected_hidden.shape[:-1], 1):
            raise ValueError("HY confidence head must output one acceptance logit per position")
        return logits.squeeze(-1)

    def dummy_contexts(self, batch: int, capacity: int) -> tuple[HYContext, ...]:
        contexts = []
        for layer_id, layer in zip(HY_TARGET_LAYERS, self.layers):
            attn = layer.attention
            reference = attn.key_up
            indices = torch.full((batch, capacity), -1, dtype=torch.int64, device=reference.device)
            kv = reference.new_zeros(batch, capacity, reference.shape[-1] + attn.rope_dim)
            contexts.append(HYContext(layer_id, indices, kv, indices >= 0))
        return tuple(contexts)
