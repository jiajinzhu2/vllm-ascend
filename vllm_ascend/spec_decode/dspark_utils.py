# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-independent DSpark input preparation and greedy Markov sampling."""

from dataclasses import dataclass
from typing import Any

import torch


def get_dspark_config(vllm_config: Any) -> Any:
    draft = vllm_config.speculative_config.draft_model_config
    return getattr(draft, "hf_text_config", None) or draft.hf_config


def is_dspark_draft(vllm_config: Any) -> bool:
    spec = vllm_config.speculative_config
    if spec is None or spec.draft_model_config is None:
        return False
    draft = spec.draft_model_config
    configs = (draft.hf_config, getattr(draft, "hf_text_config", None))
    return any(
        "dspark" in architecture.lower()
        for config in configs
        if config is not None
        for architecture in (getattr(config, "architectures", None) or ())
    )


def get_dspark_aux_layers(vllm_config: Any) -> tuple[int, ...]:
    config = get_dspark_config(vllm_config)
    # The explicit EAGLE3 field already uses the target's capture convention.
    capture_ids = getattr(config, "eagle_aux_hidden_state_layer_ids", None)
    if capture_ids:
        return tuple(capture_ids)
    # DSpark checkpoint ids name backbone layers; capture is after that layer.
    layer_ids = getattr(config, "dspark_target_layer_ids", None)
    if layer_ids is None:
        layer_ids = getattr(config, "target_layer_ids", None)
    if not layer_ids:
        raise ValueError("DSpark draft must declare its target auxiliary hidden-state layers")
    return tuple(layer + 1 for layer in layer_ids)


@dataclass
class DSparkInputs:
    input_ids: torch.Tensor
    positions: torch.Tensor
    context_slots: torch.Tensor
    query_slots: torch.Tensor
    sample_indices: torch.Tensor
    seq_lens: torch.Tensor


def prepare_dspark_inputs(
    query_start_loc: torch.Tensor,
    target_positions: torch.Tensor,
    seq_lens: torch.Tensor,
    next_token_ids: torch.Tensor,
    block_table: torch.Tensor,
    context_slots: torch.Tensor,
    *,
    block_size: int,
    num_speculative_tokens: int,
    noise_token_id: int,
    max_model_len: int,
    num_rejected_tokens: torch.Tensor | None = None,
    sample_from_anchor: bool = True,
) -> DSparkInputs:
    """Prepare one cache group's block draft, including rejection rollback.

    Context slots come from the runner, preserving its physical cache layout.
    Query slots use this group's kernel block size and block table. This eager
    implementation uses tensor operations on the input device, without reading
    device scalars on the host.
    """
    if num_speculative_tokens < 1 or block_size < 1 or max_model_len < 1:
        raise ValueError("DSpark token count, block size and model length must be positive")
    if target_positions.ndim != 1 or query_start_loc.ndim != 1:
        raise ValueError("DSpark currently requires one-dimensional positions and query starts")
    batch_size = next_token_ids.numel()
    num_context = target_positions.numel()
    if next_token_ids.ndim != 1 or query_start_loc.numel() != batch_size + 1:
        raise ValueError("DSpark request and query-start shapes disagree")
    if seq_lens.numel() < batch_size or context_slots.numel() < num_context:
        raise ValueError("DSpark context metadata is shorter than the input batch")
    if block_table.ndim != 2 or block_table.shape[0] < batch_size or block_table.shape[1] == 0:
        raise ValueError("DSpark requires a two-dimensional block table for every request")
    if batch_size and not num_context:
        raise ValueError("DSpark requests require target context")
    if num_rejected_tokens is not None and num_rejected_tokens.shape != next_token_ids.shape:
        raise ValueError("DSpark rejection counts must have one entry per request")

    device = target_positions.device
    num_queries = num_speculative_tokens + (not sample_from_anchor)
    rejected = torch.zeros_like(next_token_ids) if num_rejected_tokens is None else num_rejected_tokens
    valid_ends = query_start_loc[1 : batch_size + 1] - rejected
    effective_seq_lens = seq_lens[:batch_size] - rejected
    has_context = valid_ends > query_start_loc[:batch_size]
    last_indices = (valid_ends - 1).clamp(min=0, max=max(0, num_context - 1)).long()
    last_positions = torch.where(
        has_context, target_positions[last_indices], effective_seq_lens - 1
    )
    offsets = torch.arange(num_queries, device=device)
    positions = last_positions[:, None] + 1 + offsets
    valid_positions = (positions >= 0) & (positions < max_model_len) & (next_token_ids[:, None] >= 0)
    valid_queries = valid_positions.clone()
    logical_blocks = torch.div(positions, block_size, rounding_mode="floor")
    valid_queries &= (logical_blocks >= 0) & (logical_blocks < block_table.shape[1])
    physical_blocks = block_table[:batch_size].gather(
        1, logical_blocks.clamp(0, block_table.shape[1] - 1).long()
    )
    valid_queries &= physical_blocks >= 0
    slots = physical_blocks.long() * block_size + positions.remainder(block_size)
    slots = torch.where(valid_queries, slots, -1).flatten().to(torch.int32)

    input_ids = torch.full(
        (batch_size, num_queries), noise_token_id, dtype=next_token_ids.dtype, device=device
    )
    input_ids[:, 0] = torch.where(next_token_ids >= 0, next_token_ids, noise_token_id)
    # Safe RoPE positions for masked/padded query slots.
    positions = torch.where(valid_positions, positions, 0).flatten().to(torch.int32)

    query_lens = query_start_loc[1 : batch_size + 1] - query_start_loc[:batch_size]
    request_ids = torch.repeat_interleave(
        torch.arange(batch_size, device=device), query_lens.long(), output_size=num_context
    )
    valid_context = torch.arange(num_context, device=device) < valid_ends[request_ids]
    context_slots = torch.where(valid_context, context_slots[:num_context], -1).to(torch.int32)
    sample_offsets = torch.arange(num_speculative_tokens, device=device) + (not sample_from_anchor)
    sample_indices = (
        torch.arange(batch_size, device=device)[:, None] * num_queries + sample_offsets
    ).flatten().to(torch.int32)
    return DSparkInputs(
        input_ids.flatten(),
        positions,
        context_slots,
        slots,
        sample_indices,
        (effective_seq_lens + num_queries).clamp(max=max_model_len),
    )


def sample_dspark_greedy(
    model: Any,
    hidden_states: torch.Tensor,
    seed_token_ids: torch.Tensor,
    num_speculative_tokens: int,
) -> torch.Tensor:
    """Sample a whole block from one backbone pass and a sequential Markov head.

    The adapter supplies full draft-vocabulary logits on every TP rank. Each
    selected draft id is mapped to a target id before the next Markov lookup.
    Target ids are returned for the existing vLLM rejection sampler.
    """
    if seed_token_ids.ndim != 1 or num_speculative_tokens < 1:
        raise ValueError("DSpark sampling requires one seed per request and a positive block length")
    batch_size = seed_token_ids.numel()
    if hidden_states.shape[0] != batch_size * num_speculative_tokens:
        raise ValueError("DSpark hidden-state count must equal batch_size * num_speculative_tokens")
    if batch_size == 0:
        return torch.empty((0, num_speculative_tokens), dtype=torch.int64, device=seed_token_ids.device)
    logits = model.compute_draft_logits(hidden_states)
    if logits is None or logits.ndim != 2 or logits.shape[0] != hidden_states.shape[0]:
        raise ValueError("DSpark adapter must return full draft-vocabulary logits on every rank")
    logits = logits.reshape(batch_size, num_speculative_tokens, -1)
    result = torch.empty(
        (batch_size, num_speculative_tokens), dtype=torch.int64, device=seed_token_ids.device
    )
    previous = seed_token_ids.long()
    for step in range(num_speculative_tokens):
        bias = model.markov_bias(model.markov_embed(previous))
        if bias.shape != logits[:, step].shape:
            raise ValueError("DSpark Markov bias and draft logits must use the same vocabulary")
        draft_ids = (logits[:, step] + bias).argmax(dim=-1)
        previous = model.map_draft_to_target(draft_ids)
        result[:, step] = previous
    return result
