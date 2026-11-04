# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Read-only target KV selection and absorbed, dual-source MLA for HY DFlash.

The A5 path uses native SFA; the PyTorch path remains a numerical reference.
Logical document/token positions are deliberately separate from 3-D M-RoPE.
"""

from dataclasses import dataclass, replace

import torch
from torch import nn

SFA_LATENT_DIM = 512
SFA_ROPE_DIM = 64
SFA_MAX_QUERY_HEADS = 128


def get_hy_sfa_op(*, has_sink: bool):
    """Resolve the torch_npu operator once, before warmup or graph capture.

    The old _C_ascend binding has neither sinks nor attention_mode. Never drop
    a checkpoint's sink or fall back after a native kernel/capture failure.
    """
    try:
        op = torch.ops.npu.npu_sparse_flash_attention.default
    except AttributeError as error:
        raise RuntimeError("HY SFA requires torch_npu.npu_sparse_flash_attention; update torch_npu/CANN") from error
    required = {"attention_mode", "return_softmax_lse"}
    if has_sink:
        required.add("sinks")
    missing = required - {arg.name for arg in op._schema.arguments}
    if missing:
        raise RuntimeError(
            f"HY SFA operator is missing {sorted(missing)}. Install a torch_npu/CANN build with "
            "aclnnSparseFlashAttentionV2 for sinks, or explicitly select ascend_attention_backend=reference."
        )
    return op


def hy_sfa_indices(context_valid: torch.Tensor, query_len: int) -> torch.Tensor:
    """Pack valid sparse indices first, including only causal block positions.

    K/V stays [context capacity, current block], even for holes/empty contexts.
    Both halves of the scatter destination are unique: invalid slots cannot
    overwrite a valid slot. All lengths are device tensors and shapes are fixed
    for a captured request bucket; no CPU seqlens or task updates are needed.
    """
    batch, capacity = context_valid.shape
    valid = torch.cat((context_valid, context_valid.new_ones((batch, query_len))), dim=1)
    valid_rank = valid.to(torch.int64).cumsum(dim=1)
    invalid_rank = (~valid).to(torch.int64).cumsum(dim=1)
    count = valid_rank[:, -1:]
    destination = torch.where(valid, valid_rank - 1, count + invalid_rank - 1)
    positions = torch.arange(capacity + query_len, device=valid.device).expand(batch, -1)
    packed = torch.empty_like(positions).scatter_(1, destination, positions)
    # Valid context precedes all Q current-block keys in the packed list.
    lengths = count - query_len + torch.arange(1, query_len + 1, device=valid.device)
    allowed = positions[:, None, :] < lengths[:, :, None]
    return packed[:, None, :].expand(-1, query_len, -1).masked_fill(~allowed, -1).to(torch.int32).unsqueeze(2)


def sfa_absorbed_mla(
    query_nope, query_rope, block_latent, block_rope, context, key_up, value_up, scale, sinks, *, sfa_op
):
    """BF16 absorbed MLA with one joint native softmax over context and block.

    sparse_mode=0 is intentional: sparse_indices already contains the exact
    per-query causal set. The compact buffer's slots are NOT original RoPE
    positions. Keep the model's original scale after absorbing the K weights.
    """
    batch, query_len, heads, _ = query_nope.shape
    if (
        block_latent.shape[-1] != SFA_LATENT_DIM
        or query_rope.shape[-1] != SFA_ROPE_DIM
        or not 1 <= heads <= SFA_MAX_QUERY_HEADS
    ):
        raise ValueError("HY A5 SFA requires latent=512, RoPE=64 and 1..128 local query heads")
    if any(
        t.dtype != torch.bfloat16
        for t in (query_nope, query_rope, block_latent, block_rope, context.kv, key_up, value_up)
    ):
        raise ValueError("HY SFA requires BF16 projections and dequantized BF16 target context")
    # Native BF16 matmuls avoid repeatedly expanding weights/activations to FP32.
    query = torch.einsum("bqhd,hdr->bqhr", query_nope, key_up).contiguous()
    key = torch.cat((context.kv[..., :SFA_LATENT_DIM], block_latent), dim=1).unsqueeze(2).contiguous()
    key_rope = torch.cat((context.kv[..., SFA_LATENT_DIM:], block_rope), dim=1).unsqueeze(2).contiguous()
    kwargs = dict(
        query_rope=query_rope.contiguous(),
        key_rope=key_rope,
        sparse_block_size=1,
        layout_query="BSND",
        layout_kv="BSND",
        sparse_mode=0,
        attention_mode=2,
        return_softmax_lse=False,
    )
    if sinks is not None:
        kwargs["sinks"] = sinks.float().contiguous()
    # Tensor V aliases K, including on torch_npu builds predating value=None.
    latent_output, _, _ = sfa_op(query, key, key, hy_sfa_indices(context.valid, query_len), scale, **kwargs)
    return torch.einsum("bqhr,hvr->bqhv", latent_output, value_up)


def apply_hy_rotary(rotary, positions, query, key, *, sections=None):
    """Preserve HY's interleaved rotary pairs across all three M-RoPE axes."""
    if positions.ndim == 1:
        return rotary(positions, query, key)
    coordinates = positions.reshape(3, -1)
    if sections is None:
        if not torch.equal(coordinates, coordinates[:1].expand_as(coordinates)):
            raise ValueError("Distinct HY M-RoPE axes require checkpoint mrope_section")
        return rotary(coordinates[0], query, key)
    if len(sections) != 3 or sum(sections) * 2 != query.shape[-1]:
        raise ValueError("HY mrope_section must partition the rotary pairs into three axes")
    rotated = [rotary(axis, query.clone(), key.clone()) for axis in coordinates]
    query_parts, key_parts = [], []
    start = 0
    for axis, pairs in enumerate(sections):
        end = start + 2 * pairs
        query_parts.append(rotated[axis][0][..., start:end])
        key_parts.append(rotated[axis][1][..., start:end])
        start = end
    return torch.cat(query_parts, dim=-1), torch.cat(key_parts, dim=-1)


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
    # Ascend SFA stores latent KV and rotated RoPE keys in separate pages.
    # When omitted, ``cache`` is the legacy concatenated [latent, rope] view.
    rope_cache: torch.Tensor | None = None


@dataclass(frozen=True)
class HYTargetState:
    layers: tuple[HYTargetLayerState, ...]
    # Scalar document start offsets, in the same sequence coordinates as anchor.
    document_starts: torch.Tensor
    # Target final-norm output at anchor - 1, one row per request (not aux hidden).
    final_hidden: torch.Tensor
    # Correct positions for this block, including target's multimodal deltas.
    query_positions: torch.Tensor  # [3, batch, block_size]


@dataclass(frozen=True)
class HYContext:
    layer_id: int
    indices: torch.Tensor  # [batch, capacity], -1 padding
    kv: torch.Tensor  # gathered latent + already-rotated rope key
    valid: torch.Tensor


@dataclass(frozen=True)
class HYPagedContext:
    """Live target pages plus per-batch scratch pages excluded from its allocator.

    ``indices`` address the target's logical sequence. The last block-table
    columns map an artificial logical suffix to private scratch pages. RoPE
    still uses the true positions; that suffix never becomes accepted history.
    """

    layer_id: int
    indices: torch.Tensor
    valid: torch.Tensor
    cache: torch.Tensor
    rope_cache: torch.Tensor
    block_table: torch.Tensor
    block_slots: torch.Tensor
    block_positions: torch.Tensor
    seq_lens: torch.Tensor
    target_columns: int
    target_blocks: int


def slice_hy_context(context, batch):
    if isinstance(context, HYPagedContext):
        return replace(
            context,
            indices=context.indices[:batch],
            valid=context.valid[:batch],
            block_table=context.block_table[:batch],
            block_slots=context.block_slots[:batch],
            block_positions=context.block_positions[:batch],
            seq_lens=context.seq_lens[:batch],
        )
    return HYContext(context.layer_id, context.indices[:batch], context.kv[:batch], context.valid[:batch])


def make_paged_context(layer_id, cache, rope_cache, *, target_blocks, table_columns, batch, capacity, query_len):
    """Bind once after cache allocation; never concatenate/copy the target pool."""
    if (
        cache.ndim != 4
        or rope_cache.ndim != 4
        or cache.shape[-2:] != (1, SFA_LATENT_DIM)
        or rope_cache.shape != (*cache.shape[:-1], SFA_ROPE_DIM)
        or cache.dtype != torch.bfloat16
        or rope_cache.dtype != torch.bfloat16
    ):
        raise ValueError("HY paged SFA requires separate BF16 [blocks, block_size, 1, 512/64] caches")
    block_size = cache.shape[1]
    if not 16 <= block_size <= 1024 or block_size % 16:
        raise ValueError("HY paged SFA block size must be a multiple of 16 in [16, 1024]")
    if min(batch, table_columns, query_len) <= 0 or capacity < 0:
        raise ValueError("HY paged SFA requires positive batch/table/query dimensions and nonnegative capacity")
    per_request = (query_len + block_size - 1) // block_size
    if target_blocks <= 0 or target_blocks + batch * per_request > cache.shape[0]:
        raise ValueError("HY target KV allocation has insufficient/aligned private scratch pages")
    device = cache.device
    scratch = target_blocks + torch.arange(batch * per_request, device=device).view(batch, per_request)
    table = torch.zeros((batch, table_columns + per_request), dtype=torch.int32, device=device)
    table[:, table_columns:].copy_(scratch)
    offsets = torch.arange(query_len, device=device)
    slots = scratch[:, offsets // block_size] * block_size + offsets % block_size
    positions = (table_columns * block_size + offsets).expand(batch, -1)
    indices = torch.full((batch, capacity), -1, dtype=torch.int64, device=device)
    lengths = torch.full((batch,), table_columns * block_size + query_len, dtype=torch.int32, device=device)
    return HYPagedContext(
        layer_id,
        indices,
        indices >= 0,
        cache,
        rope_cache,
        table,
        slots,
        positions,
        lengths,
        table_columns,
        target_blocks,
    )


def update_paged_context(destination, layer, indices):
    """Validate outside capture; update metadata only, leaving historical KV read-only."""
    batch = indices.shape[0]
    if (
        batch > destination.indices.shape[0]
        or indices.shape[1:] != destination.indices.shape[1:]
        or layer.cache.data_ptr() != destination.cache.data_ptr()
        or layer.rope_cache is None
        or layer.rope_cache.data_ptr() != destination.rope_cache.data_ptr()
        or layer.block_table.shape != (batch, destination.target_columns)
    ):
        raise ValueError("HY target paged-cache topology changed after binding")
    logical = indices.clamp_min(0) // layer.cache.shape[1]
    physical = layer.block_table.long().gather(1, logical.clamp_max(destination.target_columns - 1))
    valid = indices >= 0
    allocated = (logical < destination.target_columns) & (physical >= 0) & (physical < destination.target_blocks)
    if not torch.equal(valid & allocated, valid):
        raise ValueError("HY selected context is outside the target's historical KV pool")
    destination.indices.fill_(-1)
    destination.indices[:batch].copy_(indices)
    destination.valid.zero_()
    destination.valid[:batch].copy_(valid)
    destination.block_table[:, : destination.target_columns].zero_()
    # The artificial suffix makes the logical length span unused history
    # columns too. Keep their physical IDs in range even though sparse_indices
    # never selects them; stale/unallocated table entries are not valid pages.
    table = layer.block_table.masked_fill((layer.block_table < 0) | (layer.block_table >= destination.target_blocks), 0)
    destination.block_table[:batch, : destination.target_columns].copy_(table)


def paged_sfa_absorbed_mla(
    query_nope, query_rope, block_latent, block_rope, context, key_up, value_up, scale, sinks, *, sfa_op
):
    """Write only reserved current-block slots and attend directly to target pages."""
    batch, query_len = query_nope.shape[:2]
    if any(t.dtype != torch.bfloat16 for t in (query_nope, query_rope, block_latent, block_rope, key_up, value_up)):
        raise ValueError("HY paged SFA requires BF16 draft projections")
    keys = block_latent.reshape(-1, 1, SFA_LATENT_DIM).contiguous()
    rope = block_rope.reshape(-1, 1, SFA_ROPE_DIM).contiguous()
    slots = context.block_slots.reshape(-1)
    if keys.device.type == "npu":
        from vllm_ascend.device.device_op import DeviceOperator

        DeviceOperator.reshape_and_cache(keys, rope, context.cache, context.rope_cache, slots)
    else:
        # Portable oracle only. NPU uses the native cache writer above.
        context.cache.flatten(0, 1).index_copy_(0, slots, keys)
        context.rope_cache.flatten(0, 1).index_copy_(0, slots, rope)
    packed = hy_sfa_indices(context.valid, query_len)
    logical = torch.cat((context.indices, context.block_positions), dim=-1)
    logical = logical[:, None, None, :].expand(batch, query_len, 1, -1)
    selected = logical.gather(-1, packed.clamp_min(0).long()).masked_fill(packed < 0, -1).int()
    kwargs = dict(
        block_table=context.block_table,
        actual_seq_lengths_kv=context.seq_lens,
        query_rope=query_rope.contiguous(),
        key_rope=context.rope_cache,
        sparse_block_size=1,
        layout_query="BSND",
        layout_kv="PA_BSND",
        sparse_mode=0,
        attention_mode=2,
        return_softmax_lse=False,
    )
    if sinks is not None:
        kwargs["sinks"] = sinks.float().contiguous()
    query = torch.einsum("bqhd,hdr->bqhr", query_nope, key_up).contiguous()
    result, _, _ = sfa_op(query, context.cache, context.cache, selected, scale, **kwargs)
    return torch.einsum("bqhr,hvr->bqhv", result, value_up)


def select_context_indices(
    topk_indices: torch.Tensor,
    anchor_positions: torch.Tensor,
    document_starts: torch.Tensor,
    *,
    topk: int,
    local_window: int,
) -> torch.Tensor:
    """Unique(layer top-k at anchor-1, local window), fixed capacity, no host sync.

    Short documents explicitly use the complete prefix, as required by the full
    indexer contract. Long documents use the supplied layer-specific selection.
    Padding is compacted to the end; no positions from previous documents or
    rejected/speculative future rows survive the bounds mask.
    """
    batch = anchor_positions.shape[0]
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
    if kv.ndim == 4 and kv.shape[-2] == 1:
        kv = kv.squeeze(-2)
    if layer.rope_cache is not None:
        rope = layer.rope_cache.flatten(0, 1)[slots]
        if rope.ndim == 4 and rope.shape[-2] == 1:
            rope = rope.squeeze(-2)
        kv = torch.cat((kv, rope), dim=-1)
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
    sinks: torch.Tensor | None = None,
) -> torch.Tensor:
    """One joint softmax over selected context + causal current-block keys.

    Absorb kv_b into Q/output to avoid materializing [context, heads, K/V].
    The BF16 draft uses bias-free MLA expansion. All positions in context are
    already filtered to < anchor; every block key remains unsparsified.
    """
    batch, query_len = query_nope.shape[:2]
    rank = block_latent.shape[-1]
    latent = torch.cat((context.kv[..., :rank], block_latent), dim=1)
    rope = torch.cat((context.kv[..., rank:], block_rope), dim=1)
    absorbed_query = torch.einsum("bqhd,hdr->bqhr", query_nope.float(), key_up.float())
    scores = torch.einsum("bqhr,bsr->bhqs", absorbed_query, latent.float())
    scores += torch.einsum("bqhd,bsd->bhqs", query_rope.float(), rope.float())
    causal = torch.ones((query_len, query_len), dtype=torch.bool, device=scores.device).tril()
    allowed = torch.cat(
        (context.valid[:, None, :].expand(-1, query_len, -1), causal[None].expand(batch, -1, -1)), dim=-1
    )
    scores = (scores * scale).masked_fill(~allowed[:, None], float("-inf"))
    if sinks is not None:
        sink_scores = sinks.float()[None, :, None, None].expand(batch, -1, query_len, 1)
        probabilities = torch.cat((scores, sink_scores), dim=-1).softmax(dim=-1)[..., :-1]
    else:
        probabilities = scores.softmax(dim=-1)
    latent_output = torch.einsum("bhqs,bsr->bqhr", probabilities, latent.float())
    return torch.einsum("bqhr,hvr->bqhv", latent_output, value_up.float()).to(query_nope.dtype)


def sparse_mla_with_sink(query, query_rope, cache, rope_cache, indices, table, query_ends, seq_lens, scale, sinks):
    """Bounded PyTorch fallback for HY4's learned sink on 0.20.2 SFA.

    The old kernel has no sink argument or LSE output. Ignoring the sink changes
    the target distribution. Process at most 16 queries at once, retaining the
    sink in the softmax denominator with zero value. This path needs NPU
    performance validation and is not a replacement for a sink-capable kernel.
    """
    indices = indices.reshape(indices.shape[0], -1)
    ends = torch.as_tensor(query_ends, device=query.device, dtype=torch.long)
    lengths = torch.as_tensor(seq_lens, device=query.device, dtype=torch.long)
    starts = torch.cat((ends.new_zeros(1), ends[:-1]))
    output = torch.empty_like(query)
    for start in range(0, query.shape[0], 16):
        stop = min(start + 16, query.shape[0])
        rows = torch.arange(start, stop, device=query.device)
        requests = torch.searchsorted(ends, rows, right=True).clamp_max(ends.numel() - 1)
        positions = lengths[requests] - (ends[requests] - starts[requests]) + rows - starts[requests]
        selected = indices[start:stop].long()
        valid = (selected >= 0) & (selected <= positions[:, None]) & (rows[:, None] < ends[-1])
        selected = selected.masked_fill(~valid, -1)
        context = gather_context(HYTargetLayerState(-1, cache, table[requests], selected, rope_cache), selected)
        rank = query.shape[-1]
        latent, rope = context.kv[..., :rank], context.kv[..., rank:]
        scores = torch.einsum("thr,tkr->thk", query[start:stop].float(), latent.float())
        scores += torch.einsum("thd,tkd->thk", query_rope[start:stop].float(), rope.float())
        scores = (scores * scale).masked_fill(~context.valid[:, None], float("-inf"))
        sink_logits = sinks.float()[None, :, None].expand(stop - start, -1, 1)
        probabilities = torch.cat((scores, sink_logits), dim=-1).softmax(dim=-1)[..., :-1]
        output[start:stop] = torch.einsum("thk,tkr->thr", probabilities, latent.float()).to(query.dtype)
    return output


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
        source_attention: nn.Module | None = None,
        attention_backend: str = "auto",
    ):
        super().__init__()
        if source_attention is not None:
            self.source_attention = source_attention
        self.project, self.rotary, self.finish = project, rotary, finish
        # These are views of already-loaded weights, not new trained parameters.
        self.register_buffer("key_up", key_up, persistent=False)
        self.register_buffer("value_up", value_up, persistent=False)
        self.rope_dim, self.scale = rope_dim, scale
        if attention_backend not in ("auto", "sfa", "reference"):
            raise ValueError("HY ascend_attention_backend must be auto, sfa or reference")
        self.attention_backend = "reference"
        self._sfa_op = None
        if attention_backend != "reference" and key_up.device.type == "npu":
            # Worker-only capability discovery; CPU reference tests do not load
            # torch_npu or initialize any NPU runtime.
            from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

            if get_ascend_device_type() == AscendDeviceType.A5:
                if (
                    key_up.shape[-1] != SFA_LATENT_DIM
                    or rope_dim != SFA_ROPE_DIM
                    or not 1 <= key_up.shape[0] <= SFA_MAX_QUERY_HEADS
                ):
                    raise ValueError("HY A5 SFA requires latent=512, RoPE=64 and 1..128 local query heads")
                sinks = getattr(source_attention, "learnable_sink_param", None)
                self._sfa_op = get_hy_sfa_op(has_sink=sinks is not None)
                self.attention_backend = "sfa"
        if attention_backend == "sfa" and self._sfa_op is None:
            raise ValueError("HY SFA backend requires an Ascend A5 NPU")

    @classmethod
    def from_hyv4(cls, attention: nn.Module, *, attention_backend: str = "auto") -> "HYDflashAttention":
        """Reuse a loaded public main-model attention implementation."""
        key_up, value_up = attention.draft_kv_up_weights()
        return cls(
            attention.draft_project,
            attention.draft_rotary,
            attention.draft_finish,
            key_up,
            value_up,
            attention.qk_rope_head_dim,
            attention.scaling,
            source_attention=attention,
            attention_backend=attention_backend,
        )

    def forward(self, hidden: torch.Tensor, positions: torch.Tensor, context: HYContext) -> torch.Tensor:
        q_nope, q_rope, latent, k_rope = self.project(hidden)
        q_rope, k_rope = self.rotary(positions, q_rope, k_rope)
        sinks = getattr(getattr(self, "source_attention", None), "learnable_sink_param", None)
        if isinstance(context, HYPagedContext):
            if self._sfa_op is None:
                raise ValueError("HY paged target KV requires the native SFA backend")
            output = paged_sfa_absorbed_mla(
                q_nope,
                q_rope,
                latent,
                k_rope,
                context,
                self.key_up,
                self.value_up,
                self.scale,
                sinks,
                sfa_op=self._sfa_op,
            )
        elif self._sfa_op is None:
            output = absorbed_mla(
                q_nope, q_rope, latent, k_rope, context, self.key_up, self.value_up, self.scale, sinks
            )
        else:
            output = sfa_absorbed_mla(
                q_nope,
                q_rope,
                latent,
                k_rope,
                context,
                self.key_up,
                self.value_up,
                self.scale,
                sinks,
                sfa_op=self._sfa_op,
            )
        return self.finish(hidden, output)
