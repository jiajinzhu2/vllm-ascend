# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""HY4 dense KV-share draft orchestration.

Target context KV is read-only. Only current-block KV is computed by the draft.
The target/draft integration bridge is documented in hy_dflash_prototype.md.
"""

from copy import copy
from typing import Any

import torch
from vllm.compilation.monitor import set_cudagraph_capturing_enabled
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import BatchDescriptor
from vllm.logger import logger
from vllm.v1.attention.backends.utils import CommonAttentionMetadata

from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.attention.hy_dflash import (
    HYContext,
    HYTargetLayerState,
    HYTargetState,
    gather_context,
    select_context_indices,
)
from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
from vllm_ascend.patch.hy4_config import get_hy_dflash_config
from vllm_ascend.patch.worker.hy_dflash import HYDflashAdapter
from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer
from vllm_ascend.utils import lmhead_tp_enable


class AscendHYDflashProposer(AscendDflashProposer):
    """KV-share backbone with independent v1 draft graph capture/replay."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device, runner=None):
        hf_config = vllm_config.speculative_config.draft_model_config.hf_config
        hf_config.dflash_config = get_hy_dflash_config(hf_config)
        self._validate_config(vllm_config)
        super().__init__(vllm_config, device, runner=runner)
        config = self.speculative_config.draft_model_config.hf_config.dflash_config
        self.target_layer_ids = tuple(config["target_kv_layers"])
        self.target_index_layers = tuple(config["target_index_layers"])
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
        self._target_row_indices = None
        # The target's sink fallback remains eager. Its dispatcher consequently
        # has no capture descriptors; draft graph dispatch must be independent.
        self.use_cuda_graph = self.speculative_config.enforce_eager is False
        self.graph_batch_sizes = tuple(
            sorted({self.max_batch_size, *(1 << i for i in range(self.max_batch_size.bit_length()))})
        )
        self._graph_contexts = ()
        self._graphs_captured = False
        self.graph_replay_count = 0
        rope = getattr(hf_config, "rope_parameters", None) or getattr(hf_config, "rope_scaling", None) or {}
        self._graph_text_positions = rope.get("mrope_section") is None

    @staticmethod
    def _validate_config(vllm_config: VllmConfig) -> None:
        # Matched TP uses the native sharded projections; other distributed
        # layouts need additional cache/row-mapping integration.
        # Check capabilities once; model dimensions belong to the checkpoint.
        spec = vllm_config.speculative_config
        unsupported = [
            name
            for name in (
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
                (
                    "draft TP must equal target TP",
                    spec.draft_tensor_parallel_size != getattr(vllm_config.parallel_config, "tensor_parallel_size", 1),
                ),
                ("LoRA", vllm_config.lora_config is not None),
                ("LM-head TP", lmhead_tp_enable()),
            )
            if enabled
        ]
        if unsupported:
            raise NotImplementedError("HY KV-share does not implement: " + ", ".join(unsupported))
        model_config = getattr(vllm_config, "model_config", None)
        if model_config is not None and not model_config.enforce_eager:
            raise NotImplementedError("HY target sink fallback and indexer capture require target enforce_eager")
        if model_config is not None and model_config.dtype != torch.bfloat16:
            raise ValueError("HY BF16 draft sharing target embeddings requires --dtype bfloat16")
        cache_config = getattr(vllm_config, "cache_config", None)
        if cache_config is not None and cache_config.cache_dtype not in ("auto", "bfloat16"):
            raise ValueError("HY KV sharing requires floating-point target KV; use BF16 dtype and cache dtype auto")

    def _raise_if_mrope(self):
        # This proposer carries all three axes through the draft attention.
        pass

    def _get_model(self):
        # vLLM 0.20.2's common loader can otherwise retain the target quant_config
        # when a different model_config is supplied. Never quantize this BF16
        # checkpoint with the target's MXFP8 scheme.
        from vllm.compilation.backends import set_model_tag
        from vllm.model_executor.model_loader import get_model

        config = copy(self.vllm_config)
        config.model_config = copy(self.speculative_config.draft_model_config)
        if config.model_config.quantization is not None:
            raise ValueError("HY4DflyModels requires an unquantized BF16 draft checkpoint")
        config.model_config.dtype = torch.bfloat16
        config.quant_config = None
        config.parallel_config = self.speculative_config.draft_parallel_config
        with set_model_tag("eagle_head"):
            return get_model(vllm_config=config)

    def prepare_inputs(self, *args, **kwargs):
        metadata, rows = super().prepare_inputs(*args, **kwargs)
        self._target_row_indices = rows
        return metadata, rows

    def prepare_inputs_padded(self, *args, **kwargs):
        result = super().prepare_inputs_padded(*args, **kwargs)
        self._target_row_indices = result[1]
        return result

    def _get_eagle3_use_aux_hidden_state_from_config(self) -> bool:
        # Fusion consumes target final-norm hidden at anchor-1, not EAGLE aux
        # features. Attention consumes each target layer's existing latent KV.
        return False

    def load_model(self, model: torch.nn.Module) -> None:
        """Load the BF16 checkpoint, then compose its loaded modules.

        No draft persistent context cache is discovered/allocated by this path.
        HY4DflyModels implements to_hy_dflash_adapter(target_model). External
        wrappers can supply the same bridge without changing this proposer.
        """
        self._target_model = model
        with self.maybe_eager_context:
            draft = self._get_model()
        if not isinstance(draft, HYDflashAdapter):
            draft = draft.to_hy_dflash_adapter(model)
        self.model = draft
        logger.info(
            "HY DFlash draft attention backends: %s", [layer.attention.attention_backend for layer in draft.layers]
        )
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
        if self.use_cuda_graph:
            self._initialize_graph_buffers()

    def _initialize_graph_buffers(self) -> None:
        if self.model.correction is not None or self.model.confidence_head is not None:
            raise NotImplementedError("HY draft graphs require the checkpoint without auxiliary heads")
        # Allocate before memory profiling. Every bucket uses views of these
        # buffers, including profiling, capture and subsequent request batches.
        self._graph_contexts = self.model.dummy_contexts(
            self.max_batch_size, self.context_topk + self.context_local_window
        )
        self._graph_target_hidden = torch.zeros(
            (self.max_batch_size, self.hidden_size), dtype=self.dtype, device=self.device
        )
        self._runnable = ACLGraphWrapper(
            self._run_merged_draft, self.vllm_config, runtime_mode=CUDAGraphMode.FULL, use_eagle=True
        )

    def _graph_batch_size(self, batch: int) -> int:
        if not 0 < batch <= self.max_batch_size:
            raise ValueError(f"HY draft batch {batch} exceeds capacity {self.max_batch_size}")
        return next(size for size in self.graph_batch_sizes if size >= batch)

    def _select_graph_buffers(self, batch: int) -> None:
        self._contexts = tuple(
            HYContext(context.layer_id, context.indices[:batch], context.kv[:batch], context.valid[:batch])
            for context in self._graph_contexts
        )
        self._target_final_hidden = self._graph_target_hidden[:batch]

    def _graph_descriptor(self, batch: int) -> BatchDescriptor:
        return BatchDescriptor(num_tokens=batch * self.query_block_size, num_reqs=batch, uniform=True)

    @torch.inference_mode()
    def capture_model(self) -> int:
        """Called by model_runner_v1 on its capture stream, even for eager target."""
        if not self.use_cuda_graph or self._graphs_captured:
            return 0
        torch.npu.synchronize()
        torch.npu.empty_cache()
        free_before = torch.npu.mem_get_info()[0]
        warmups = max(1, self.vllm_config.compilation_config.cudagraph_num_of_warmups)
        set_cudagraph_capturing_enabled(True)
        try:
            # Largest first: all buckets share the normal ACLGraphWrapper pool.
            for batch in reversed(self.graph_batch_sizes):
                count = batch * self.query_block_size
                for _ in range(warmups):
                    self.dummy_run(count, num_reqs=batch)
                self.dummy_run(count, num_reqs=batch, aclgraph_runtime_mode=CUDAGraphMode.FULL)
                torch.npu.synchronize()
        finally:
            set_cudagraph_capturing_enabled(False)
        self._graphs_captured = True
        used = free_before - torch.npu.mem_get_info()[0]
        logger.info("HY DFlash captured FULL draft graphs for request batches %s", self.graph_batch_sizes)
        return used

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

    def _target_state_from_main_model(
        self,
        anchor_positions: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor,
    ) -> HYTargetState:
        """Expose the public main model's live SFA pages to the draft."""
        batch = anchor_positions.shape[0]
        rows = token_indices_to_sample[:batch].long()
        layers = []
        source_rows = rows if self._target_row_indices is None else self._target_row_indices[rows].long()
        for layer_id, index_layer_id, group_id in zip(
            self.target_layer_ids, self.target_index_layers, self.target_kv_cache_group_ids, strict=True
        ):
            attention = self._target_model.get_hy_dflash_attention(layer_id)
            pages = attention.mla_attn.mla_attn.kv_cache
            layers.append(
                HYTargetLayerState(
                    layer_id=layer_id,
                    cache=pages[0],
                    rope_cache=pages[1],
                    block_table=self.runner.input_batch.block_table[group_id].get_device_tensor()[:batch],
                    topk_indices=self._target_model.get_hy_dflash_topk_buffer(index_layer_id)[source_rows],
                )
            )

        anchor_query_positions = target_positions[..., rows].long() + 1
        offsets = torch.arange(self.query_block_size, dtype=torch.int64, device=self.device)
        if anchor_query_positions.ndim == 1:
            query_positions = anchor_query_positions[:, None] + offsets
            query_positions = query_positions.unsqueeze(0).expand(3, -1, -1)
        else:
            query_positions = anchor_query_positions[..., None] + offsets

        document_starts = getattr(self.runner.input_batch, "document_starts", None)
        if document_starts is None:
            document_starts = torch.zeros_like(anchor_positions)
        else:
            document_starts = document_starts[:batch].to(device=self.device, dtype=anchor_positions.dtype)
        return HYTargetState(
            layers=tuple(layers),
            document_starts=document_starts,
            final_hidden=target_hidden_states[rows],
            query_positions=query_positions,
        )

    def prepare_context(
        self,
        anchor_positions: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor,
    ) -> None:
        """Gather each target layer's anchor-1 selection without KV writes."""
        batch = anchor_positions.shape[0]
        request_ids = tuple(self.runner.input_batch.req_ids[:batch])
        if hasattr(self._target_model, "get_hy_dflash_attention"):
            state = self._target_state_from_main_model(
                anchor_positions,
                target_positions,
                target_hidden_states,
                token_indices_to_sample,
            )
        else:
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
        if self.use_cuda_graph:
            # Validate outside capture: torch.equal synchronizes with the host.
            # Text-only RoPE then receives a 1-D view, avoiding that check in
            # each captured layer. Check every live batch, including replays.
            if self._graph_text_positions and not torch.equal(
                state.query_positions, state.query_positions[:1].expand_as(state.query_positions)
            ):
                raise ValueError("Distinct HY M-RoPE axes require checkpoint mrope_section")
            padded = self._graph_batch_size(batch)
            self._select_graph_buffers(padded)
            for destination, source in zip(self._contexts, contexts, strict=True):
                destination.indices[:batch].copy_(source.indices)
                destination.kv[:batch].copy_(source.kv)
                destination.valid[:batch].copy_(source.valid)
                destination.indices[batch:].fill_(-1)
                destination.kv[batch:].zero_()
                destination.valid[batch:].zero_()
            self._target_final_hidden[:batch].copy_(state.final_hidden)
            self._target_final_hidden[batch:].zero_()
            self.input_ids[batch * self.query_block_size : padded * self.query_block_size].fill_(
                self.parallel_drafting_token_id
            )
            self.mrope_positions[:, batch * self.query_block_size : padded * self.query_block_size].zero_()
        else:
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
        accepted_lengths = cad.seq_lens[:batch]
        if num_rejected_tokens_gpu is not None:
            accepted_lengths = accepted_lengths - num_rejected_tokens_gpu.reshape(batch)
        num_tokens = batch * self.query_block_size
        self.input_ids[:num_tokens].fill_(self.parallel_drafting_token_id)
        self.input_ids[: num_tokens : self.query_block_size].copy_(next_token_ids.reshape(batch))
        if token_indices_to_sample is None:
            token_indices_to_sample = cad.query_start_loc[1 : batch + 1] - 1
            if num_rejected_tokens_gpu is not None:
                token_indices_to_sample = token_indices_to_sample - num_rejected_tokens_gpu.reshape(batch)
        self.prepare_context(
            accepted_lengths,
            target_positions,
            target_hidden_states,
            token_indices_to_sample,
        )
        self._target_row_indices = None
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
        positions = self.mrope_positions[:, :num_input_tokens].view(3, batch, self.query_block_size)
        if self.use_cuda_graph and self._graph_text_positions:
            positions = positions[0].reshape(-1)
        return {
            "input_ids": self.input_ids[:num_input_tokens].view(batch, self.query_block_size),
            "positions": positions,
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
        batch = metadata.num_reqs
        padded = self._graph_batch_size(batch) if self.use_cuda_graph else batch
        count = padded * self.query_block_size
        mode = CUDAGraphMode.FULL if self.use_cuda_graph else CUDAGraphMode.NONE
        descriptor = self._graph_descriptor(padded) if self.use_cuda_graph else None
        if self.use_cuda_graph and not self._graphs_captured:
            raise RuntimeError("HY draft graphs were not captured by model_runner_v1 during warmup")
        with set_ascend_forward_context(
            None,
            self.vllm_config,
            num_tokens=count,
            num_actual_tokens=num_tokens,
            aclgraph_runtime_mode=mode,
            batch_descriptor=descriptor,
            is_draft_model=True,
        ):
            output = self._runnable(
                num_input_tokens=count,
                batch_size=padded,
                token_indices_to_sample=indices,
                target_positions=target_positions,
                inputs_embeds=None,
                multi_steps_attn_metadata=[],
                num_tokens=count,
            )
        if self.use_cuda_graph:
            self.graph_replay_count += 1
            # Replaying the same graph overwrites its output allocation. The
            # scheduler/rejection sampler owns the returned proposals instead.
            return output[:batch].clone()
        return output

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
        if self.model.correction is None:
            # This checkpoint has no Markov/correction or confidence head.
            # One backbone pass and one batched head projection produce all
            # seven proposals. Never manufacture untrained output heads.
            sample_hidden = hidden[:, 1:].reshape(-1, self.hidden_size)
            logits = self.model.compute_logits(sample_hidden)
            self.confidence_logits = None
            return self.commit_draft_probs(logits.argmax(dim=-1).view(batch_size, -1))
        ids = torch.empty((batch_size, self.query_block_size - 1), dtype=torch.int64, device=self.device)
        confidence = torch.empty_like(ids, dtype=torch.float32) if self.model.confidence_head is not None else None
        previous_ids = self.input_ids[: num_tokens : self.query_block_size].long()
        for step in range(self.query_block_size - 1):
            corrected = self.model.correct_hidden_states(hidden[:, step + 1], previous_ids)
            logits = self.model.compute_logits(corrected)
            ids[:, step] = logits.argmax(dim=-1)
            # This is a diagnostic prediction, not p/q, not a path selector.
            if confidence is not None:
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
            num_reqs = min(self.max_batch_size, max(1, num_tokens // self.query_block_size))
        # No live target read/write and no context projection. Allocate worst-case
        # gathered context for memory profiling, even though all entries are masked.
        mode = CUDAGraphMode.NONE
        descriptor = None
        if self.use_cuda_graph:
            num_reqs = self._graph_batch_size(num_reqs)
            self._select_graph_buffers(num_reqs)
            for context in self._contexts:
                context.indices.fill_(-1)
                context.kv.zero_()
                context.valid.zero_()
            self._target_final_hidden.zero_()
            if not is_profile and aclgraph_runtime_mode == CUDAGraphMode.FULL:
                mode = CUDAGraphMode.FULL
                descriptor = self._graph_descriptor(num_reqs)
        else:
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
            aclgraph_runtime_mode=mode,
            batch_descriptor=descriptor,
            is_draft_model=True,
            in_profile_run=is_profile,
        ):
            self._runnable(count, num_reqs, None, None, None, [], count)
        # Profiling does not publish confidence for any live request.
        self.confidence_logits = None
        self.confidence_request_ids = ()
