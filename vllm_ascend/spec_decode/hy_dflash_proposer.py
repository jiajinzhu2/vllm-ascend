# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Experimental HY DFlash orchestration, not a HY checkpoint implementation.

The model contract is intentionally explicit: context projection/cache insertion,
causal MLA/DSA, MoE, hidden correction and path selection belong to the model.
This proposer owns their ordering and the context/query boundary. See
docs/source/developer_guide/Design_Documents/hy_dflash_prototype.md for the
version-1 contract.
"""

from copy import copy
from typing import Any

import torch
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import get_forward_context
from vllm.v1.attention.backends.utils import CommonAttentionMetadata

from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer
from vllm_ascend.utils import lmhead_tp_enable


class AscendHYDflashProposer(AscendDflashProposer):
    """Single-device, eager, deterministic-draft prototype without IHC streams.

    Phase A writes accepted target context to the draft's own layer caches.
    Phase B runs one causal anchor+mask block, then samples in position order.
    No checkpoint architecture, projection weights or trained correction path
    is inferred from the name of the target model.
    """

    def __init__(self, vllm_config: VllmConfig, device: torch.device, runner=None):
        self._validate_config(vllm_config)
        super().__init__(vllm_config, device, runner=runner)
        if self.uses_mrope or self.uses_xdrope_dim or self.supports_mm_inputs:
            raise NotImplementedError("HY DFlash prototype supports text with one-dimensional positions only")
        if lmhead_tp_enable():
            raise NotImplementedError("HY DFlash prototype does not support separate LM-head TP")

        # A query block can be larger than the target's scheduled-token buffer.
        self.input_ids = torch.zeros(self.max_query_tokens, dtype=torch.int32, device=device)
        self.token_indices_to_sample = torch.zeros(
            self.max_batch_size * self.num_speculative_tokens, dtype=torch.int32, device=device
        )
        self.slot_mapping_group[0] = torch.full((self.max_query_tokens,), -1, dtype=torch.int32, device=device)
        self._draft_probs: torch.Tensor | None = None

    @staticmethod
    def _validate_config(vllm_config: VllmConfig) -> None:
        spec = vllm_config.speculative_config
        if spec.method != "dflash" or not spec.parallel_drafting:
            raise ValueError("HY DFlash requires method='dflash' and parallel_drafting=True")
        if not spec.enforce_eager:
            raise NotImplementedError(
                "HY DFlash is an eager prototype; set speculative_config.enforce_eager=True. "
                "ACL graph replay needs real HY MLA/DSA operator validation."
            )
        parallel = vllm_config.parallel_config
        for name in (
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "data_parallel_size",
            "prefill_context_parallel_size",
            "decode_context_parallel_size",
        ):
            if getattr(parallel, name, 1) != 1:
                raise NotImplementedError(f"HY DFlash prototype requires {name}=1")
        if spec.draft_tensor_parallel_size != 1:
            raise NotImplementedError("HY DFlash prototype requires draft_tensor_parallel_size=1")
        if vllm_config.lora_config is not None:
            raise NotImplementedError("HY DFlash prototype does not support LoRA")

    def load_model(self, model: torch.nn.Module) -> None:
        super().load_model(model)
        self._validate_model_contract()

    def _validate_model_contract(self) -> None:
        model = self.get_model()
        if getattr(model, "hy_dflash_contract_version", None) != 1:
            raise ValueError(
                "HY DFlash requires an external model implementing hy_dflash_contract_version=1; "
                "a normal Qwen DFlash or HY MTP checkpoint is not interchangeable."
            )
        for name in (
            "combine_hidden_states",
            "precompute_and_store_context_kv",
            "correct_hidden_states",
            "select_hidden_path",
            "compute_logits",
        ):
            if not callable(getattr(model, name, None)):
                raise TypeError(f"HY DFlash model must implement {name}()")

    def _maybe_share_topk_indices(self, target_language_model: torch.nn.Module) -> None:
        # The target's sparse indices describe different query positions. HY's
        # anchor+mask block and context indexer cache must be draft-owned.
        pass

    def _combine_target_hidden_states(self, target_hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.model.combine_hidden_states(target_hidden_states)
        if hidden_states.ndim != 2 or hidden_states.shape != (target_hidden_states.shape[0], self.hidden_size):
            raise ValueError("HY DFlash requires [tokens, hidden_size] context; IHC streams are not supported")
        return hidden_states

    def context_slot_mapping(
        self,
        query_start_loc: torch.Tensor,
        num_rejected_tokens_gpu: torch.Tensor | None,
    ) -> torch.Tensor:
        """Mask rejected target rows in the inherited physical-slot mapping.

        The parent packs slots as block_id * block_size + position % block_size.
        Rejected verification rows are present in padded batches but are NOT
        committed context. -1 must be ignored by both MLA and indexer writers.
        """
        slots = self._context_slot_mapping_buffer[: self._dflash_num_context]
        if num_rejected_tokens_gpu is not None:
            lengths = query_start_loc[1:] - query_start_loc[:-1]
            accepted_ends = query_start_loc[1:] - num_rejected_tokens_gpu
            accepted_end_per_token = torch.repeat_interleave(
                accepted_ends, lengths, output_size=self._dflash_num_context
            )
            token_offsets = self.arange_dflash[: self._dflash_num_context]
            slots.masked_fill_(token_offsets >= accepted_end_per_token, -1)
        return slots

    def prepare_context(
        self,
        query_start_loc: torch.Tensor,
        num_rejected_tokens_gpu: torch.Tensor | None,
    ) -> None:
        """Finish staging accepted context and execute phase A outside query forward."""
        self.context_slot_mapping(query_start_loc, num_rejected_tokens_gpu)
        self.prefill_context()

    def prefill_context(self, *, is_profile: bool = False) -> None:
        """Project/pack context and write every draft layer's MLA + DSA cache.

        The model owns the actual projection layout and per-layer cache writes.
        Passing None slots during profiling computes intermediates without
        writing live KV. This is also used by dummy_run before query warmup.
        """
        num_context = self._dflash_num_context
        if num_context == 0:
            return
        with set_ascend_forward_context(
            None,
            self.vllm_config,
            num_tokens=num_context,
            num_actual_tokens=num_context,
            aclgraph_runtime_mode=CUDAGraphMode.NONE,
            is_draft_model=True,
            in_profile_run=is_profile,
        ):
            self.model.precompute_and_store_context_kv(
                self._dflash_hidden_states[:num_context],
                self._context_positions_buffer[:num_context],
                None if is_profile else self._context_slot_mapping_buffer[:num_context],
            )

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
        """Pack [anchor, mask * K], preserve accepted context, enable causality."""
        if not 0 < cad.num_reqs <= self.max_batch_size:
            raise ValueError("HY DFlash requires a nonempty batch within max_batch_size")
        if target_token_ids.shape[0] > self.max_num_tokens:
            raise ValueError("HY DFlash context exceeds the allocated context buffer")
        if target_token_ids.ndim != 1 or target_token_ids.shape[0] < cad.num_reqs:
            raise ValueError("HY DFlash requires at least one context token per request")
        if cad.query_start_loc.shape != (cad.num_reqs + 1,):
            raise ValueError("HY DFlash requires unpadded context query_start_loc")
        if target_positions.ndim != 1 or target_positions.shape != target_token_ids.shape:
            raise ValueError("HY DFlash requires one position per target token")
        if target_hidden_states.shape != (target_token_ids.shape[0], self.hidden_size):
            raise ValueError("HY DFlash context must be [tokens, hidden_size]")
        if next_token_ids.shape != (cad.num_reqs,):
            raise ValueError("HY DFlash requires one anchor per request")
        if num_rejected_tokens_gpu is not None and num_rejected_tokens_gpu.shape != (cad.num_reqs,):
            raise ValueError("HY DFlash requires one rejected-token count per request")

        # Keep target metadata intact; the parent rewrites its argument in place.
        context_query_start_loc = cad.query_start_loc
        num_tokens, indices, query_metadata, _ = super().set_inputs_first_pass(
            target_token_ids,
            next_token_ids,
            target_positions,
            target_hidden_states,
            token_indices_to_sample,
            copy(cad),
            num_rejected_tokens_gpu,
            req_scheduled_tokens,
            long_seq_metadata,
            num_prefill_reqs,
            num_decode_reqs,
        )
        query_metadata.causal = True
        query_metadata.positions = self.positions[:num_tokens]
        query_metadata.positions_cpu = None
        # Do not hand MLA/DSA builders the target's stale CPU sequence lengths.
        # Eager builders materialize the corrected GPU seq_lens when needed.
        query_metadata.seq_lens_cpu = None
        query_metadata._seq_lens_cpu = None
        query_metadata.num_computed_tokens_cpu = None
        query_metadata._num_computed_tokens_cpu = None
        query_metadata._num_computed_tokens_cache = None
        if query_metadata.seq_lens_cpu_upper_bound is not None:
            query_metadata.seq_lens_cpu_upper_bound = cad.seq_lens_cpu_upper_bound + 1 + self.num_speculative_tokens

        self.prepare_context(context_query_start_loc, num_rejected_tokens_gpu)
        return num_tokens, indices, query_metadata, None

    def build_model_inputs_first_pass(self, num_input_tokens: int) -> dict[str, Any]:
        """Return query-only inputs; phase A must not be replayed in this region."""
        if not 0 <= num_input_tokens <= self.max_query_tokens:
            raise ValueError("HY DFlash query exceeds the allocated query buffer")
        return {
            "input_ids": self.input_ids[:num_input_tokens],
            "positions": self.positions[:num_input_tokens],
            "inputs_embeds": None,
        }

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
        """One three-layer query forward, then prefix-conditioned sampling.

        The public signature matches the parent's runnable. The model contract
        uses block order directly: column 0 is anchor, columns 1..K are masks.
        Hidden correction only receives hidden positions through the current
        mask and the already sampled prefix, never a future position or token.
        """
        query_length = 1 + self.num_speculative_tokens
        if num_tokens != batch_size * query_length or num_input_tokens != num_tokens:
            raise ValueError("HY DFlash eager query must contain exactly B * (K + 1) tokens")
        get_forward_context().moe_layer_index = 0
        hidden_states = self.model(**self.build_model_inputs_first_pass(num_input_tokens))
        if not isinstance(hidden_states, torch.Tensor) or hidden_states.shape != (num_tokens, self.hidden_size):
            raise ValueError("HY DFlash query forward must return [query_tokens, hidden_size], without IHC streams")
        block_hidden_states = hidden_states.view(batch_size, query_length, self.hidden_size)
        draft_ids = torch.empty((batch_size, self.num_speculative_tokens), dtype=torch.int64, device=self.device)
        for step in range(self.num_speculative_tokens):
            # The adapter decides how to combine three-layer output with the
            # sampled prefix and how its trained selector chooses a path.
            corrected = self.model.correct_hidden_states(
                block_hidden_states[:, : step + 2, :], draft_ids[:, :step], step
            )
            selected = self.model.select_hidden_path(corrected, step)
            if selected.shape != (batch_size, self.hidden_size):
                raise ValueError("HY DFlash path selector must return [batch_size, hidden_size]")
            logits = self.model.compute_logits(selected)
            if logits is None or logits.ndim != 2 or logits.shape[0] != batch_size:
                raise ValueError("HY DFlash LM head must return full-vocabulary [batch_size, vocab_size] logits")
            draft_ids[:, step] = logits.argmax(dim=-1)
        return self.commit_draft_probs(draft_ids, draft_probs=None)

    def commit_draft_probs(self, draft_ids: torch.Tensor, draft_probs: torch.Tensor | None) -> torch.Tensor:
        """Publish a deterministic proposal; this does NOT commit query KV.

        v0.20.2 uses None for a delta proposal q(token)=1 at the drafted ID.
        A softmax of greedy logits is NOT the distribution that generated it.
        Stochastic drafting needs request/step-aligned probability storage and
        scheduler reordering support before accepting a non-None tensor here.
        """
        if draft_probs is not None:
            raise NotImplementedError("Stochastic HY draft probabilities are not wired to verification yet")
        self._draft_probs = None
        return draft_ids

    def verification_probs(self) -> torch.Tensor | None:
        """Return q for the runner's rejection sampler, not target probability p.

        None is stateless across request reordering/partial verification: the
        sampler derives the delta q from the current speculative token IDs.
        Target logits and sampling transforms remain owned by the runner.
        """
        return self._draft_probs

    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        num_reqs: int = 0,
        num_tokens_across_dp: torch.Tensor | None = None,
        aclgraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        batch_descriptor=None,
        dummy_compute_logits=lambda hidden_states: None,
        is_profile=False,
        **kwargs,
    ) -> None:
        """Exercise context projection AND the complete query/sampling path.

        No live context cache slots are written. Like the parent with
        enforce_eager=True, this ignores the target's requested graph mode.
        This is profiling/warmup, NOT a claim of ACL graph support.
        """
        if num_reqs == 0:
            return
        if not 0 < num_reqs <= self.max_batch_size or not 0 <= num_tokens <= self.max_num_tokens:
            raise ValueError("HY DFlash dummy batch exceeds allocated buffers")
        self._validate_model_contract()
        self._dflash_num_context = num_tokens
        self._dflash_hidden_states[:num_tokens].zero_()
        self._context_positions_buffer[:num_tokens].zero_()
        self.prefill_context(is_profile=True)

        query_length = 1 + self.num_speculative_tokens
        num_query_tokens = num_reqs * query_length
        self.input_ids[:num_query_tokens].fill_(self.parallel_drafting_token_id)
        self.input_ids[:num_query_tokens:query_length].zero_()
        self.positions[:num_query_tokens].copy_(self.arange_dflash[:query_length].repeat(num_reqs))
        self._slot_mapping_buffer[:num_query_tokens].fill_(-1)
        with set_ascend_forward_context(
            None,
            self.vllm_config,
            num_tokens=num_query_tokens,
            num_actual_tokens=num_query_tokens,
            aclgraph_runtime_mode=CUDAGraphMode.NONE,
            is_draft_model=True,
            in_profile_run=is_profile,
        ):
            self._runnable(
                num_input_tokens=num_query_tokens,
                batch_size=num_reqs,
                token_indices_to_sample=None,
                target_positions=None,
                inputs_embeds=None,
                multi_steps_attn_metadata=[],
                num_tokens=num_query_tokens,
            )
