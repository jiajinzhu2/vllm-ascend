# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Read-only target KV selection and absorbed, dual-source MLA for HY DFlash.

This is a PyTorch correctness path, not a tuned Ascend sparse-attention kernel.
Logical document/token positions are deliberately separate from 3-D M-RoPE.
"""

from dataclasses import dataclass

import torch
from torch import nn

HY_TARGET_LAYERS = (61, 65, 69, 73, 77)
HY_BLOCK_SIZE = 8
HY_MASK_TOKEN_ID = 120009
HY_CONTEXT_TOPK = 2048
HY_CONTEXT_LOCAL_WINDOW = 8


@dataclass(frozen=True)
class HYTargetLayerState:
    """One target layer, in current request order; all tensors are read-only.

    cache: [physical_blocks, block_size, latent_rank + rope_dim], BF16/FP.
    The last rope_dim channels contain target's already-rotated key component.
    A target-specific bridge must expose this view (dequantize if necessary).
    topk_indices: logical token offsets selected by this layer at anchor - 1,
    NOT physical slots, M-RoPE coordinates, or another layer's scratch buffer.
    """

    layer_id: int
    cache: torch.Tensor
    block_table: torch.Tensor
    topk_indices: torch.Tensor


@dataclass(frozen=True)
class HYTargetState:
    layers: tuple[HYTargetLayerState, ...]
    # Scalar document start offsets, in the same sequence coordinates as anchor.
    document_starts: torch.Tensor
    # Target final-norm output at anchor - 1, one row per request (not aux hidden).
    final_hidden: torch.Tensor
    # Correct positions for this block, including target's multimodal deltas.
    query_positions: torch.Tensor  # [3, batch, 8]


@dataclass(frozen=True)
class HYContext:
    layer_id: int
    indices: torch.Tensor  # [batch, capacity], -1 padding
    kv: torch.Tensor  # gathered latent + already-rotated rope key
    valid: torch.Tensor


def select_context_indices(
    topk_indices: torch.Tensor,
    anchor_positions: torch.Tensor,
    document_starts: torch.Tensor,
    *,
    topk: int = HY_CONTEXT_TOPK,
    local_window: int = HY_CONTEXT_LOCAL_WINDOW,
) -> torch.Tensor:
    """Unique(layer top-k at anchor-1, local window), fixed capacity, no host sync.

    Short documents explicitly use the complete prefix, as required by the full
    indexer contract. Long documents use the supplied layer-specific selection.
    Padding is compacted to the end; no positions from previous documents or
    rejected/speculative future rows survive the bounds mask.
    """
    batch = anchor_positions.shape[0]
    if topk <= 0 or local_window < 0:
        raise ValueError("Invalid HY context selection capacity")
    if document_starts.shape != (batch,) or topk_indices.ndim != 2 or topk_indices.shape[0] != batch:
        raise ValueError("HY context selection requires request-aligned inputs")
    if topk_indices.shape[1] > topk:
        raise ValueError("Target indexer returned more than the configured top-k")
    device = anchor_positions.device
    selected = torch.full((batch, topk), -1, dtype=torch.int64, device=device)
    selected[:, : topk_indices.shape[1]] = topk_indices
    prefix = document_starts[:, None] + torch.arange(topk, device=device)
    short = (anchor_positions - document_starts) <= topk
    selected = torch.where(short[:, None], prefix, selected)
    local = anchor_positions[:, None] - torch.arange(local_window, 0, -1, device=device)
    candidates = torch.cat((selected, local), dim=1)
    valid = (candidates >= document_starts[:, None]) & (candidates >= 0)
    valid &= candidates < anchor_positions[:, None]
    sentinel = torch.iinfo(torch.int64).max
    ordered = candidates.masked_fill(~valid, sentinel).sort(dim=1).values
    unique = torch.cat((torch.ones_like(ordered[:, :1], dtype=torch.bool), ordered[:, 1:] != ordered[:, :-1]), dim=1)
    compact = ordered.masked_fill(~unique, sentinel).sort(dim=1).values
    return compact.masked_fill(compact == sentinel, -1)


def gather_context(layer: HYTargetLayerState, indices: torch.Tensor) -> HYContext:
    """Read selected target slots without writes or target KV reprojection."""
    cache, table = layer.cache, layer.block_table
    if cache.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("HY cache bridge must expose dequantized floating-point latent KV")
    if cache.ndim != 3 or table.ndim != 2 or table.shape[0] != indices.shape[0]:
        raise ValueError("Expected paged HY cache and request-aligned block table")
    if cache.shape[0] == 0 or cache.shape[1] == 0 or table.shape[1] == 0:
        raise ValueError("HY target cache/block table is empty")
    block_size = cache.shape[1]
    logical_blocks = indices.clamp_min(0) // block_size
    addressable = (indices >= 0) & (logical_blocks < table.shape[1])
    physical_blocks = table.long().gather(1, logical_blocks.clamp_max(table.shape[1] - 1))
    addressable &= (physical_blocks >= 0) & (physical_blocks < cache.shape[0])
    # The provider must supply allocated blocks for every selected valid token.
    # Fail rather than silently dropping missing context. This one validation
    # synchronizes in the eager reference path, outside the attention forward.
    if not torch.equal(addressable, indices >= 0):
        raise ValueError("HY context points to an unallocated target KV block")
    slots = physical_blocks.clamp(0, cache.shape[0] - 1) * block_size + indices.clamp_min(0) % block_size
    kv = cache.flatten(0, 1)[slots]
    kv = kv.masked_fill(~addressable[..., None], 0)
    return HYContext(layer.layer_id, indices, kv, addressable)


def absorbed_mla(
    query_nope: torch.Tensor,  # [B, Q, heads, nope_dim]
    query_rope: torch.Tensor,  # [B, Q, heads, rope_dim], already rotated
    block_latent: torch.Tensor,  # [B, Q, rank], after kv_a layer norm
    block_rope: torch.Tensor,  # [B, Q, rope_dim], already rotated
    context: HYContext,
    key_up: torch.Tensor,  # [heads, nope_dim, rank]
    value_up: torch.Tensor,  # [heads, value_dim, rank]
    scale: float,
) -> torch.Tensor:
    """One joint softmax over selected context + causal current-block keys.

    Absorb kv_b into Q/output to avoid materializing [context, heads, K/V].
    The BF16 draft uses bias-free MLA expansion. All positions in context are
    already filtered to < anchor; every block key remains unsparsified.
    """
    if query_nope.ndim != 4 or key_up.ndim != 3 or value_up.ndim != 3:
        raise ValueError("Invalid HY MLA query/weight ranks")
    batch, query_len, heads, _ = query_nope.shape
    rank, rope_dim = block_latent.shape[-1], block_rope.shape[-1]
    if context.kv.shape[:2] != context.valid.shape or context.kv.shape[0] != batch:
        raise ValueError("HY context batch/valid mask mismatch")
    if context.kv.shape[-1] != rank + rope_dim:
        raise ValueError("Target and draft MLA latent/RoPE layouts are incompatible")
    if key_up.shape != (heads, query_nope.shape[-1], rank) or value_up.shape[0::2] != (heads, rank):
        raise ValueError("Incompatible HY MLA expansion weights")
    if query_rope.shape != (batch, query_len, heads, rope_dim):
        raise ValueError("Incompatible HY MLA query RoPE shape")
    if block_latent.shape != (batch, query_len, rank) or block_rope.shape != (batch, query_len, rope_dim):
        raise ValueError("Incompatible HY current-block KV shape")
    latent = torch.cat((context.kv[..., :rank], block_latent), dim=1)
    rope = torch.cat((context.kv[..., rank:], block_rope), dim=1)
    absorbed_query = torch.einsum("bqhd,hdr->bqhr", query_nope.float(), key_up.float())
    scores = torch.einsum("bqhr,bsr->bhqs", absorbed_query, latent.float())
    scores += torch.einsum("bqhd,bsd->bhqs", query_rope.float(), rope.float())
    causal = torch.ones((query_len, query_len), dtype=torch.bool, device=scores.device).tril()
    allowed = torch.cat(
        (context.valid[:, None, :].expand(-1, query_len, -1), causal[None].expand(batch, -1, -1)), dim=-1
    )
    probabilities = (scores * scale).masked_fill(~allowed[:, None], float("-inf")).softmax(dim=-1)
    latent_output = torch.einsum("bhqs,bsr->bqhr", probabilities, latent.float())
    return torch.einsum("bqhr,hvr->bqhv", latent_output, value_up.float()).to(query_nope.dtype)


class HYDflashAttention(nn.Module):
    """Compose loaded HY4 projections with a read-only, dual-source MLA forward.

    project(hidden) -> (q_nope, q_rope, normalized_kv_latent, k_rope).
    rotary(positions[3,B,Q], q_rope, k_rope) -> rotated (q_rope, k_rope).
    finish(hidden, attention_output[B,Q,heads,value_dim]) applies HY4's existing
    output gate/o_proj in its native order. No gate activation/layout is guessed.
    key_up/value_up are views of the loaded bias-free BF16 kv_b weight.
    """

    def __init__(
        self,
        project: nn.Module,
        rotary: nn.Module,
        finish: nn.Module,
        key_up: torch.Tensor,
        value_up: torch.Tensor,
        rope_dim: int,
        scale: float,
    ):
        super().__init__()
        self.project, self.rotary, self.finish = project, rotary, finish
        # These are views of already-loaded weights, not new trained parameters.
        self.register_buffer("key_up", key_up, persistent=False)
        self.register_buffer("value_up", value_up, persistent=False)
        self.rope_dim, self.scale = rope_dim, scale

    def forward(self, hidden: torch.Tensor, positions: torch.Tensor, context: HYContext) -> torch.Tensor:
        q_nope, q_rope, latent, k_rope = self.project(hidden)
        q_rope, k_rope = self.rotary(positions, q_rope, k_rope)
        output = absorbed_mla(q_nope, q_rope, latent, k_rope, context, self.key_up, self.value_up, self.scale)
        return self.finish(hidden, output)
