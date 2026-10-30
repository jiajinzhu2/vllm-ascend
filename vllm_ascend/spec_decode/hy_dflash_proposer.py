# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""HY4 dense KV-share draft orchestration.

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
    gather_context,
    select_context_indices,
)
from vllm_ascend.patch.worker.hy_dflash import HYDflashAdapter
from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer
from vllm_ascend.utils import lmhead_tp_enable


class AscendHYDflashProposer(AscendDflashProposer):
    """Eager KV-share backbone followed by serial hidden correction."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device, runner=None):
        self._validate_config(vllm_config)
        super().__init__(vllm_config, device, runner=runner)
        config = self.speculative_config.draft_model_config.hf_config.dflash_config
        self.target_layer_ids = tuple(config["target_layer_ids"])
        self.context_topk = config["context_topk"]
        self.context_local_window = config["context_local_window"]
        self.query_block_size = self.num_speculative_tokens + 1
        if not self.target_layer_ids or self.context_topk <= 0 or self.context_local_window < 0:
            raise ValueError("HY requires target_layer_ids, positive context_topk and nonnegative context_local_window")
        self.uses_mrope = True
        self.max_query_tokens = self.max_batch_size * self.query_block_size
        self.input_ids = torch.zeros(self.max_query_tokens, dtype=torch.int32, device=device)
        self.mrope_positions = torch.zeros((3, self.max_query_tokens), dtype=torch.int64, device=device)
        self.token_indices_to_sample = torch.zeros(
            self.max_batch_size * (self.query_block_size - 1), dtype=torch.int32, device=device
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
        # This implementation has no graph/distributed/LoRA execution path yet.
        # Check capabilities once; model dimensions belong to the checkpoint.
        spec = vllm_config.speculative_config
        unsupported = [
            name
            for name in (
                "tensor_parallel_size",
                "pipeline_parallel_size",
                "data_parallel_size",
                "prefill_context_parallel_size",
                "decode_context_parallel_size",
            )
            if getattr(vllm_config.parallel_config, name, 1) != 1
        ]
        unsupported += [
            name
            for name, enabled in (
                ("ACL graphs", not spec.enforce_eager),
                ("draft TP", spec.draft_tensor_parallel_size != 1),
                ("LoRA", vllm_config.lora_config is not None),
                ("LM-head TP", lmhead_tp_enable()),
            )
            if enabled
        ]
        if unsupported:
            raise NotImplementedError("HY eager KV-share does not implement: " + ", ".join(unsupported))

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
        self._target_model = model
        with self.maybe_eager_context:
            draft = self._get_model()
        if not isinstance(draft, HYDflashAdapter):
            draft = draft.to_hy_dflash_adapter(model)
        self.model = draft
        self.attn_layer_names = list(model.get_hy_dflash_layer_names(self.target_layer_ids))
        if (
            draft.hidden_size != self.hidden_size
            or draft.target_layer_ids != self.target_layer_ids
            or len(draft.layers) != len(self.target_layer_ids)
            or len(self.attn_layer_names) != len(self.target_layer_ids)
            or len(set(self.attn_layer_names)) != len(self.attn_layer_names)
        ):
            raise ValueError("HY draft dimensions and target KV layer mapping must match the model configuration")
        self._draft_attn_layer_names = set()
        self.draft_attn_groups = []

    def initialize_attn_backend(self, kv_cache_config, kernel_block_sizes=None) -> None:
        """Bind target KV groups, not ordinary DFlash's new context-cache group.

        The runner uses attn_layer_names[0] to obtain request sequence metadata.
        Per-layer block tables/cache views come from the target provider, so
        source layers may belong to different target KV groups.
        """
        layer_groups = {
            name: gid for gid, group in enumerate(kv_cache_config.kv_cache_groups) for name in group.layer_names
        }
        self.target_kv_cache_group_ids = tuple(layer_groups[name] for name in self.attn_layer_names)
        self.kv_cache_gid = self.target_kv_cache_group_ids[0]

    def prepare_context(self, anchor_positions: torch.Tensor) -> None:
        """Fetch per-layer anchor-1 indexer rows and gather target KV read-only."""
        batch = anchor_positions.shape[0]
        request_ids = tuple(self.runner.input_batch.req_ids[:batch])
        state = self._target_model.get_hy_dflash_context(
            request_ids=request_ids,
            anchor_positions=anchor_positions,
            layer_ids=self.target_layer_ids,
            block_size=self.query_block_size,
        )
        if tuple(layer.layer_id for layer in state.layers) != self.target_layer_ids:
            raise ValueError("Target context is not in the configured draft layer order")
        contexts = []
        for layer in state.layers:
            indices = select_context_indices(
                layer.topk_indices,
                anchor_positions,
                state.document_starts,
                topk=self.context_topk,
                local_window=self.context_local_window,
            )
            contexts.append(gather_context(layer, indices))
        self._contexts = tuple(contexts)
        self._target_final_hidden = state.final_hidden
        self.mrope_positions[:, : batch * self.query_block_size].copy_(state.query_positions.reshape(3, -1))
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
        accepted_lengths = cad.seq_lens
        if num_rejected_tokens_gpu is not None:
            accepted_lengths = accepted_lengths - num_rejected_tokens_gpu.reshape(batch)
        num_tokens = batch * self.query_block_size
        self.input_ids[:num_tokens].fill_(self.parallel_drafting_token_id)
        self.input_ids[: num_tokens : self.query_block_size].copy_(next_token_ids.reshape(batch))
        self.prepare_context(accepted_lengths)
        indices = torch.arange(num_tokens, device=self.device, dtype=torch.int32).view(batch, self.query_block_size)[
            :, 1:
        ]
        indices = indices.reshape(-1)
        self.token_indices_to_sample[: indices.numel()].copy_(indices)

        query = copy(cad)
        query.query_start_loc = self.arange_dflash[: batch + 1] * self.query_block_size
        query.query_start_loc_cpu = torch.arange(batch + 1, dtype=torch.int32) * self.query_block_size
        query.seq_lens = accepted_lengths + self.query_block_size
        query.num_actual_tokens = query.num_input_tokens = num_tokens
        query.max_query_len = self.query_block_size
        query.max_seq_len = cad.max_seq_len + self.query_block_size
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
        batch = num_input_tokens // self.query_block_size
        return {
            "input_ids": self.input_ids[:num_input_tokens].view(batch, self.query_block_size),
            "positions": self.mrope_positions[:, :num_input_tokens].view(3, batch, self.query_block_size),
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
        hidden = self.model(**self.build_model_inputs_first_pass(num_tokens))
        ids = torch.empty((batch_size, self.query_block_size - 1), dtype=torch.int64, device=self.device)
        confidence = torch.empty_like(ids, dtype=torch.float32)
        previous_ids = self.input_ids[: num_tokens : self.query_block_size].long()
        for step in range(self.query_block_size - 1):
            corrected = self.model.correct_hidden_states(hidden[:, step + 1], previous_ids)
            logits = self.model.compute_logits(corrected)
            ids[:, step] = logits.argmax(dim=-1)
            # This is a diagnostic prediction, not p/q, not a path selector.
            confidence[:, step] = self.model.predict_confidence(corrected).float()
            previous_ids = ids[:, step]
        self.confidence_logits = confidence
        return self.commit_draft_probs(ids)

    def commit_draft_probs(self, draft_ids: torch.Tensor) -> torch.Tensor:
        """The current proposer is greedy; its actual q is a delta distribution."""
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
        # No live target read/write and no context projection. Allocate worst-case
        # gathered context for memory profiling, even though all entries are masked.
        self._contexts = self.model.dummy_contexts(num_reqs, self.context_topk + self.context_local_window)
        self._target_final_hidden = torch.zeros((num_reqs, self.hidden_size), dtype=self.dtype, device=self.device)
        count = num_reqs * self.query_block_size
        self.input_ids[:count].fill_(self.parallel_drafting_token_id)
        self.input_ids[: count : self.query_block_size].zero_()
        self.mrope_positions[:, :count].copy_(
            self.arange_dflash[: self.query_block_size].repeat(num_reqs)[None].expand(3, -1)
        )
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
