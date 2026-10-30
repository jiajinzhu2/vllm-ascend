# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""HY4 five-layer dense KV-share draft orchestration.

Target context KV is read-only. Only current-block KV is computed by the draft.
The target/draft integration bridge is documented in hy_dflash_prototype.md.
"""

from copy import copy
from typing import Any

import torch
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.v1.attention.backends.utils import CommonAttentionMetadata

from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.attention.hy_dflash import (
    HY_BLOCK_SIZE,
    HY_CONTEXT_LOCAL_WINDOW,
    HY_CONTEXT_TOPK,
    HY_MASK_TOKEN_ID,
    HY_TARGET_LAYERS,
    HYTargetState,
    gather_context,
    select_context_indices,
)
from vllm_ascend.patch.worker.hy_dflash import HYDflashAdapter
from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer
from vllm_ascend.utils import lmhead_tp_enable


class AscendHYDflashProposer(AscendDflashProposer):
    """Eager correctness path: 8 slots, 5 dense layers, 7 serial corrections."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device, runner=None):
        self._validate_config(vllm_config)
        super().__init__(vllm_config, device, runner=runner)
        if lmhead_tp_enable():
            raise NotImplementedError("HY KV-share draft does not support separate LM-head TP")
        if self.parallel_drafting_token_id != HY_MASK_TOKEN_ID:
            raise ValueError("HY KV-share checkpoint requires mask_token_id=120009")
        self.uses_mrope = True
        self.max_query_tokens = self.max_batch_size * HY_BLOCK_SIZE
        self.input_ids = torch.zeros(self.max_query_tokens, dtype=torch.int32, device=device)
        self.mrope_positions = torch.zeros((3, self.max_query_tokens), dtype=torch.int64, device=device)
        self.token_indices_to_sample = torch.zeros(
            self.max_batch_size * (HY_BLOCK_SIZE - 1), dtype=torch.int32, device=device
        )
        # These parent buffers serve projected DFlash context, not KV sharing.
        del self._dflash_hidden_states
        del self._context_positions_buffer
        del self._context_slot_mapping_buffer
        self._contexts = ()
        self._draft_probs = None
        self.confidence_logits = None
        self.confidence_request_ids: tuple[str, ...] = ()
        self._target_model = None

    @staticmethod
    def _validate_config(vllm_config: VllmConfig) -> None:
        spec = vllm_config.speculative_config
        if spec.method != "dflash" or not spec.parallel_drafting:
            raise ValueError("HY KV-share draft requires method='dflash' with parallel_drafting=True")
        if spec.num_speculative_tokens != HY_BLOCK_SIZE - 1:
            raise ValueError("HY KV-share checkpoint requires exactly 7 speculative tokens")
        if not spec.enforce_eager:
            raise NotImplementedError("HY KV-share attention is eager-only pending Ascend graph validation")
        for name in (
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "data_parallel_size",
            "prefill_context_parallel_size",
            "decode_context_parallel_size",
        ):
            if getattr(vllm_config.parallel_config, name, 1) != 1:
                raise NotImplementedError(f"HY KV-share draft requires {name}=1")
        if spec.draft_tensor_parallel_size != 1:
            raise NotImplementedError("HY KV-share draft requires draft_tensor_parallel_size=1")
        if vllm_config.lora_config is not None:
            raise NotImplementedError("HY KV-share draft does not support LoRA")

    def _get_eagle3_use_aux_hidden_state_from_config(self) -> bool:
        # Fusion consumes target final-norm hidden at anchor-1, not EAGLE aux
        # features. Attention consumes each target layer's existing latent KV.
        return False

    def load_model(self, model: torch.nn.Module) -> None:
        """Load through vLLM, then compose the vendor's already-loaded modules.

        No draft persistent context cache is discovered/allocated by this path.
        A vendor wrapper can return HYDflashAdapter directly or implement
        to_hy_dflash_adapter(target_model). This is not a checkpoint-name guess.
        """
        for name in ("get_hy_dflash_layer_names", "get_hy_dflash_context"):
            if not callable(getattr(model, name, None)):
                raise TypeError(f"Adapted HY4 target must expose {name}()")
        self._target_model = model
        with self.maybe_eager_context:
            draft = self._get_model()
        if not isinstance(draft, HYDflashAdapter):
            factory = getattr(draft, "to_hy_dflash_adapter", None)
            if not callable(factory):
                raise TypeError("Loaded draft must implement to_hy_dflash_adapter(target_model)")
            draft = factory(model)
        if not isinstance(draft, HYDflashAdapter) or draft.hidden_size != self.hidden_size:
            raise TypeError("Draft bridge must return a matching HYDflashAdapter")
        self.model = draft
        self.attn_layer_names = list(model.get_hy_dflash_layer_names(HY_TARGET_LAYERS))
        if len(self.attn_layer_names) != len(HY_TARGET_LAYERS) or len(set(self.attn_layer_names)) != len(
            HY_TARGET_LAYERS
        ):
            raise ValueError("Target must identify five distinct KV-source attention layers")
        self._draft_attn_layer_names = set()
        self.draft_attn_groups = []

    def initialize_attn_backend(self, kv_cache_config, kernel_block_sizes=None) -> None:
        """Bind target KV groups, not ordinary DFlash's new context-cache group.

        The runner uses attn_layer_names[0] to obtain request sequence metadata.
        Per-layer block tables/cache views come from the target provider, so
        source layers may belong to different target KV groups.
        """
        groups = []
        for name in self.attn_layer_names:
            matches = [i for i, group in enumerate(kv_cache_config.kv_cache_groups) if name in group.layer_names]
            if len(matches) != 1:
                raise ValueError(f"HY target KV source {name} must belong to exactly one cache group")
            groups.append(matches[0])
        self.target_kv_cache_group_ids = tuple(groups)
        self.kv_cache_gid = groups[0]

    def prepare_context(self, anchor_positions: torch.Tensor) -> None:
        """Fetch per-layer anchor-1 indexer rows and gather target KV read-only."""
        batch = anchor_positions.shape[0]
        request_ids = tuple(self.runner.input_batch.req_ids[:batch])
        if len(request_ids) != batch:
            raise ValueError("HY target context requires one request ID per block")
        state = self._target_model.get_hy_dflash_context(
            request_ids=request_ids,
            anchor_positions=anchor_positions,
            layer_ids=HY_TARGET_LAYERS,
            block_size=HY_BLOCK_SIZE,
        )
        if not isinstance(state, HYTargetState):
            raise TypeError("Target bridge must return HYTargetState")
        if tuple(layer.layer_id for layer in state.layers) != HY_TARGET_LAYERS:
            raise ValueError("Target context must be layer-aligned: 61,65,69,73,77")
        if state.final_hidden.shape != (batch, self.hidden_size):
            raise ValueError("Target must return final-normalized hidden at anchor-1")
        if state.query_positions.shape != (3, batch, HY_BLOCK_SIZE):
            raise ValueError("Target must return [3,B,8] M-RoPE positions, including multimodal deltas")
        if state.document_starts.shape != (batch,):
            raise ValueError("Target must return one document start per request")
        contexts = []
        for layer in state.layers:
            indices = select_context_indices(layer.topk_indices, anchor_positions, state.document_starts)
            contexts.append(gather_context(layer, indices))
        self._contexts = tuple(contexts)
        self._target_final_hidden = state.final_hidden
        self.mrope_positions[:, : batch * HY_BLOCK_SIZE].copy_(state.query_positions.reshape(3, -1))
        self.confidence_request_ids = request_ids

    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        cad: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
        req_scheduled_tokens=None,
        long_seq_metadata=None,
        num_prefill_reqs=0,
        num_decode_reqs=0,
    ) -> tuple[int, torch.Tensor, CommonAttentionMetadata, None]:
        """Prepare anchor/masks and accepted context coordinates, not context KV.

        cad.seq_lens counts logical tokens; M-RoPE coordinates must NOT be used
        as sequence offsets. The target provider maps request/anchor-1 to its
        original forward row, even after compacted/padded verification batches.
        """
        batch = cad.num_reqs
        if not 0 < batch <= self.max_batch_size or next_token_ids.shape != (batch,):
            raise ValueError("HY draft requires a nonempty request-aligned anchor batch")
        if cad.seq_lens.shape != (batch,):
            raise ValueError("HY draft requires unpadded request sequence lengths")
        accepted_lengths = cad.seq_lens
        if num_rejected_tokens_gpu is not None:
            if num_rejected_tokens_gpu.shape != (batch,):
                raise ValueError("HY draft requires one rejection count per request")
            accepted_lengths = accepted_lengths - num_rejected_tokens_gpu
        num_tokens = batch * HY_BLOCK_SIZE
        self.input_ids[:num_tokens].fill_(HY_MASK_TOKEN_ID)
        self.input_ids[:num_tokens:HY_BLOCK_SIZE].copy_(next_token_ids)
        self.prepare_context(accepted_lengths)
        indices = torch.arange(num_tokens, device=self.device, dtype=torch.int32).view(batch, HY_BLOCK_SIZE)[:, 1:]
        indices = indices.reshape(-1)
        self.token_indices_to_sample[: indices.numel()].copy_(indices)

        query = copy(cad)
        query.query_start_loc = self.arange_dflash[: batch + 1] * HY_BLOCK_SIZE
        query.query_start_loc_cpu = torch.arange(batch + 1, dtype=torch.int32) * HY_BLOCK_SIZE
        query.seq_lens = accepted_lengths + HY_BLOCK_SIZE
        query.num_actual_tokens = query.num_input_tokens = num_tokens
        query.max_query_len = HY_BLOCK_SIZE
        query.max_seq_len = cad.max_seq_len + HY_BLOCK_SIZE
        query.positions = self.mrope_positions[:, :num_tokens]
        query.positions_cpu = None
        query.causal = True
        query.slot_mapping = self._slot_mapping_buffer[:num_tokens]
        query.slot_mapping.fill_(-1)  # transient block KV; NEVER write target slots
        query.seq_lens_cpu = query._seq_lens_cpu = None
        query.num_computed_tokens_cpu = query._num_computed_tokens_cpu = None
        query._num_computed_tokens_cache = None
        query.seq_lens_cpu_upper_bound = None
        return num_tokens, indices, query, None

    def build_model_inputs_first_pass(self, num_input_tokens: int) -> dict[str, Any]:
        if not 0 < num_input_tokens <= self.max_query_tokens or num_input_tokens % HY_BLOCK_SIZE:
            raise ValueError("HY query must contain complete 8-slot blocks")
        batch = num_input_tokens // HY_BLOCK_SIZE
        return {
            "input_ids": self.input_ids[:num_input_tokens].view(batch, HY_BLOCK_SIZE),
            "positions": self.mrope_positions[:, :num_input_tokens].view(3, batch, HY_BLOCK_SIZE),
            "target_final_hidden": self._target_final_hidden,
            "contexts": self._contexts,
        }

    @torch.inference_mode()
    def _propose(
        self,
        target_token_ids,
        target_positions,
        target_hidden_states,
        next_token_ids,
        token_indices_to_sample,
        common_attn_metadata,
        target_model_batch_desc,
        sampling_metadata,
        mm_embed_inputs=None,
        num_rejected_tokens_gpu=None,
        **kwargs,
    ) -> torch.Tensor:
        # Bypass the common single-attention-metadata path: each draft layer has
        # its own selected target context and a transient causal block source.
        num_tokens, indices, metadata, _ = self.set_inputs_first_pass(
            target_token_ids,
            next_token_ids,
            target_positions,
            target_hidden_states,
            token_indices_to_sample,
            common_attn_metadata,
            num_rejected_tokens_gpu,
        )
        with set_ascend_forward_context(
            None,
            self.vllm_config,
            num_tokens=num_tokens,
            num_actual_tokens=num_tokens,
            aclgraph_runtime_mode=CUDAGraphMode.NONE,
            is_draft_model=True,
        ):
            return self._runnable(
                num_input_tokens=num_tokens,
                batch_size=metadata.num_reqs,
                token_indices_to_sample=indices,
                target_positions=target_positions,
                inputs_embeds=None,
                multi_steps_attn_metadata=[],
                num_tokens=num_tokens,
            )

    def _run_merged_draft(
        self,
        num_input_tokens,
        batch_size,
        token_indices_to_sample,
        target_positions,
        inputs_embeds,
        multi_steps_attn_metadata,
        num_tokens,
        is_prefill=None,
    ) -> torch.Tensor:
        if num_input_tokens != num_tokens or num_tokens != batch_size * HY_BLOCK_SIZE:
            raise ValueError("HY eager query must contain exactly B * 8 tokens")
        hidden = self.model(**self.build_model_inputs_first_pass(num_tokens))
        if hidden.shape != (batch_size, HY_BLOCK_SIZE, self.hidden_size):
            raise ValueError("HY backbone must return [B,8,H] final-normalized hidden")
        ids = torch.empty((batch_size, HY_BLOCK_SIZE - 1), dtype=torch.int64, device=self.device)
        confidence = torch.empty_like(ids, dtype=torch.float32)
        previous_ids = self.input_ids[:num_tokens:HY_BLOCK_SIZE].long()
        for step in range(HY_BLOCK_SIZE - 1):
            corrected = self.model.correct_hidden_states(hidden[:, step + 1], previous_ids)
            logits = self.model.compute_logits(corrected)
            if logits is None or logits.ndim != 2 or logits.shape[0] != batch_size:
                raise ValueError("HY LM head must return full-vocabulary [B,V] logits")
            ids[:, step] = logits.argmax(dim=-1)
            # This is a diagnostic prediction, not p/q, not a path selector.
            confidence[:, step] = self.model.predict_confidence(corrected).float()
            previous_ids = ids[:, step]
        self.confidence_logits = confidence
        return self.commit_draft_probs(ids, None)

    def commit_draft_probs(self, draft_ids: torch.Tensor, draft_probs: torch.Tensor | None) -> torch.Tensor:
        if draft_probs is not None:
            raise NotImplementedError("Random draft sampling needs request-aligned q storage")
        self._draft_probs = None  # deterministic delta q, not softmax(greedy logits)
        return draft_ids

    def verification_probs(self) -> torch.Tensor | None:
        """Only actual draft q; confidence must never be passed to the sampler."""
        return self._draft_probs

    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        num_reqs: int = 0,
        num_tokens_across_dp=None,
        aclgraph_runtime_mode=CUDAGraphMode.NONE,
        batch_descriptor=None,
        dummy_compute_logits=lambda hidden_states: None,
        is_profile=False,
        **kwargs,
    ) -> None:
        if num_reqs == 0:
            return
        if not 0 < num_reqs <= self.max_batch_size:
            raise ValueError("HY dummy batch exceeds allocated query buffers")
        # No live target read/write and no context projection. Allocate worst-case
        # gathered context for memory profiling, even though all entries are masked.
        self._contexts = self.model.dummy_contexts(num_reqs, HY_CONTEXT_TOPK + HY_CONTEXT_LOCAL_WINDOW)
        self._target_final_hidden = torch.zeros((num_reqs, self.hidden_size), dtype=self.dtype, device=self.device)
        count = num_reqs * HY_BLOCK_SIZE
        self.input_ids[:count].fill_(HY_MASK_TOKEN_ID)
        self.input_ids[:count:HY_BLOCK_SIZE].zero_()
        self.mrope_positions[:, :count].copy_(self.arange_dflash[:HY_BLOCK_SIZE].repeat(num_reqs)[None].expand(3, -1))
        with set_ascend_forward_context(
            None,
            self.vllm_config,
            num_tokens=count,
            num_actual_tokens=count,
            aclgraph_runtime_mode=CUDAGraphMode.NONE,
            is_draft_model=True,
            in_profile_run=is_profile,
        ):
            self._runnable(count, num_reqs, None, None, None, [], count)
        # Profiling does not publish confidence for any live request.
        self.confidence_logits = None
        self.confidence_request_ids = ()
