# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Eager DSpark block proposer for the vLLM 0.20 model-runner interface.

Adapted from the DSpark proposer in upstream vllm-ascend #16925. Model-specific
draft networks and attention implementations are supplied by a model adapter.
"""

from copy import copy

import torch
from vllm.config import CompilationMode, CUDAGraphMode, VllmConfig, get_layers_from_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer
from vllm_ascend.spec_decode.dspark_utils import (
    get_dspark_aux_layers,
    get_dspark_config,
    prepare_dspark_inputs,
    sample_dspark_greedy,
)


class AscendDSparkProposer(AscendDflashProposer):
    """One draft-backbone forward, followed by greedy Markov token sampling."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device, runner=None):
        spec = vllm_config.speculative_config
        if spec is None or spec.method not in ("mtp", "dflash"):
            raise ValueError("vLLM 0.20 DSpark uses an existing mtp or dflash config method")
        if spec.disable_padded_drafter_batch:
            raise ValueError("DSpark currently requires padded drafter batches")
        parallel = vllm_config.parallel_config
        if (
            parallel.prefill_context_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
            or parallel.pipeline_parallel_size != 1
        ):
            raise ValueError("This DSpark backport requires PCP=DCP=PP=1")
        if getattr(spec, "draft_sample_method", "greedy") != "greedy":
            raise ValueError("This DSpark backport supports greedy draft sampling")
        if spec.draft_tensor_parallel_size != parallel.tensor_parallel_size:
            raise ValueError("This DSpark backport requires matching target/draft TP sizes")
        config = get_dspark_config(vllm_config)
        noise_token_id = getattr(config, "dspark_noise_token_id", None)
        if noise_token_id is None:
            noise_token_id = getattr(config, "parallel_drafting_token_id", None)
        if noise_token_id is None or not 0 <= noise_token_id < config.vocab_size:
            raise ValueError("DSpark draft must declare its trained noise token id")
        get_dspark_aux_layers(vllm_config)

        # The release's base initializer knows MTP/DFlash, not DSpark. Avoid
        # initializing its parallel mask-hidden-state or graph path. Copies
        # leave the target's speculative/compilation configuration untouched.
        draft_config = copy(vllm_config)
        draft_config.speculative_config = copy(spec)
        draft_config.speculative_config.parallel_drafting = False
        draft_config.speculative_config.enforce_eager = True
        draft_config.speculative_config.draft_model_config = copy(spec.draft_model_config)
        draft_config.speculative_config.draft_model_config.enforce_eager = True
        draft_config.compilation_config = copy(vllm_config.compilation_config)
        draft_config.compilation_config.mode = CompilationMode.NONE
        draft_config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
        super().__init__(draft_config, device, runner=runner)
        if self.uses_mrope or self.draft_uses_xdrope_dim:
            raise ValueError("DSpark currently supports one-dimensional RoPE positions")
        self.method = "dspark"
        self.use_cuda_graph = False
        self.parallel_drafting = True
        self.parallel_drafting_token_id = noise_token_id
        self.sample_from_anchor = getattr(config, "sample_from_anchor", True)
        self.num_query_per_req = self.num_speculative_tokens + (not self.sample_from_anchor)
        self.max_query_tokens = self.max_batch_size * self.num_query_per_req
        self.input_ids = torch.zeros(self.max_query_tokens, dtype=torch.int32, device=device)
        self.positions = torch.zeros_like(self.input_ids)
        self.hidden_size = spec.draft_model_config.get_hidden_size()
        self.hidden_states = torch.zeros(
            (self.max_num_tokens, self.hidden_size), dtype=self.dtype, device=device
        )
        self._per_group_block_tables: dict[int, torch.Tensor] = {}
        self._per_group_slot_mappings: dict[int, torch.Tensor] = {}
        self._per_group_kernel_block_sizes: dict[int, int] = {}
        self._query_slots: dict[int, torch.Tensor] = {}

    def _maybe_share_embeddings(self, target_language_model):
        # Missing flags must not overwrite checkpoint-trained draft embeddings.
        if getattr(self.model, "has_own_embed_tokens", True) is False:
            super()._maybe_share_embeddings(target_language_model)

    def _maybe_share_lm_head(self, target_model):
        if getattr(self.model, "has_own_lm_head", True) is False:
            target = target_model.get_language_model() if hasattr(target_model, "get_language_model") else target_model
            self.model.lm_head = target.lm_head

    def _maybe_share_topk_indices(self, target_language_model):
        # A model adapter owns any trained target/draft index-sharing contract.
        pass

    def load_model(self, model):
        super().load_model(model)
        required = (
            "combine_hidden_states",
            "precompute_and_store_context_kv",
            "get_draft_kv_cache_layer_names",
            "compute_draft_logits",
            "markov_embed",
            "markov_bias",
            "map_draft_to_target",
        )
        missing = [name for name in required if not callable(getattr(self.model, name, None))]
        if missing:
            raise TypeError(f"DSpark draft adapter is missing methods: {missing}")
        names = list(self.model.get_draft_kv_cache_layer_names())
        if not names or len(names) != len(set(names)) or set(names) != self._draft_attn_layer_names:
            raise ValueError("DSpark draft must declare each of its cache layers once, in model order")
        self.attn_layer_names = names

    def initialize_attn_backend(self, kv_cache_config, kernel_block_sizes=None):
        layers = get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase)
        self.draft_attn_groups = []
        self._layer_group_idx = {}
        for gid, group in enumerate(kv_cache_config.kv_cache_groups):
            groups = {}
            for name in self.attn_layer_names:
                if name not in group.layer_names:
                    continue
                backend = layers[name].get_attn_backend()
                if not backend.supports_non_causal():
                    raise ValueError("DSpark draft cache backend must declare non-causal attention support")
                spec = group.kv_cache_spec
                if isinstance(spec, UniformTypeKVCacheSpecs):
                    spec = spec.kv_cache_specs[name]
                kernel_size = spec.block_size
                if kernel_block_sizes is not None:
                    kernel_size = (
                        kernel_block_sizes[gid]
                        if isinstance(kernel_block_sizes, (list, tuple))
                        else kernel_block_sizes
                    )
                if kernel_size <= 0 or spec.block_size % kernel_size:
                    raise ValueError("DSpark kernel block size must divide the scheduler block size")
                key = (backend.full_cls_name(), spec)
                if key not in groups:
                    attn_group = AttentionGroup(backend, [name], spec, gid)
                    attn_group.create_metadata_builders(
                        self.vllm_config, self.device, kernel_block_size=kernel_size
                    )
                    groups[key] = attn_group
                else:
                    groups[key].layer_names.append(name)
                self._layer_group_idx[name] = gid
                self._per_group_kernel_block_sizes[gid] = kernel_size
            self.draft_attn_groups.extend(groups.values())
        if set(self._layer_group_idx) != set(self.attn_layer_names):
            raise ValueError("Every DSpark draft cache layer must belong to a KV cache group")
        self.kv_cache_gid = self._layer_group_idx[self.attn_layer_names[0]]
        self.kernel_block_size = self._per_group_kernel_block_sizes[self.kv_cache_gid]
        self._query_slots = {
            gid: torch.full((self.max_query_tokens,), -1, dtype=torch.int32, device=self.device)
            for gid in set(self._layer_group_idx.values())
        }

    def set_per_group_attn_metadata(self, gid, block_table, slot_mapping):
        if gid in self._per_group_kernel_block_sizes:
            self._per_group_block_tables[gid] = block_table
            self._per_group_slot_mappings[gid] = slot_mapping

    def model_returns_tuple(self):
        return False

    @torch.inference_mode()
    def _propose(
        self,
        target_token_ids,
        target_positions,
        target_hidden_states,
        next_token_ids,
        token_indices_to_sample,
        common_attn_metadata,
        target_model_batch_desc=None,
        sampling_metadata=None,
        num_rejected_tokens_gpu=None,
        **kwargs,
    ):
        del target_token_ids, token_indices_to_sample, target_model_batch_desc, sampling_metadata, kwargs
        cad = copy(common_attn_metadata)
        if next_token_ids.ndim != 1 or next_token_ids.numel() > cad.num_reqs:
            raise ValueError("DSpark needs one seed token per actual request")
        # Target graph metadata can contain extra padded request rows.
        batch_size = next_token_ids.numel()
        cad.num_reqs = batch_size
        num_context = target_positions.numel()
        num_queries = batch_size * self.num_query_per_req
        if batch_size > self.max_batch_size or num_context > self.max_num_tokens:
            raise ValueError("DSpark batch exceeds configured buffer capacity")
        missing = set(self._query_slots) - self._per_group_block_tables.keys()
        if missing:
            raise RuntimeError(f"Runner did not supply DSpark KV cache groups: {sorted(missing)}")
        cad.query_start_loc = cad.query_start_loc[:batch_size + 1]
        cad.seq_lens = cad.seq_lens[:batch_size]
        context_slots = {}
        prepared = None
        for gid in self._query_slots:
            prepared = prepare_dspark_inputs(
                cad.query_start_loc,
                target_positions,
                cad.seq_lens,
                next_token_ids,
                self._per_group_block_tables[gid],
                self._per_group_slot_mappings[gid],
                block_size=self._per_group_kernel_block_sizes[gid],
                num_speculative_tokens=self.num_speculative_tokens,
                noise_token_id=self.parallel_drafting_token_id,
                max_model_len=self.max_model_len,
                num_rejected_tokens=num_rejected_tokens_gpu,
                sample_from_anchor=self.sample_from_anchor,
            )
            self._query_slots[gid].fill_(-1)
            self._query_slots[gid][:num_queries].copy_(prepared.query_slots)
            context_slots[gid] = prepared.context_slots
        if prepared is None:
            raise RuntimeError("DSpark attention groups have not been initialized")
        self.input_ids.fill_(self.parallel_drafting_token_id)
        self.positions.zero_()
        self.input_ids[:num_queries].copy_(prepared.input_ids)
        self.positions[:num_queries].copy_(prepared.positions)
        num_input_tokens, across_dp, _ = self.runner._sync_metadata_across_dp(num_queries, is_draft_model=True)
        if num_input_tokens > self.max_query_tokens:
            raise ValueError("DSpark DP padding exceeds the query buffer capacity")

        # The 0.20 metadata builders still consume host lengths/positions.
        # Transfer one packed batch, including rejection corrections, so neither
        # CPU mirror can retain optimistic (pre-rejection) target lengths.
        host = torch.cat((prepared.seq_lens, prepared.positions)).to("cpu")
        seq_lens_cpu = host[:batch_size].clone()
        positions_cpu = torch.zeros(num_input_tokens, dtype=torch.int32)
        positions_cpu[:num_queries] = host[batch_size:]
        query_start_cpu = torch.arange(batch_size + 1, dtype=torch.int32) * self.num_query_per_req
        query_start = query_start_cpu.to(self.device)
        per_layer = {}
        for group in self.draft_attn_groups:
            gid = group.kv_cache_group_id
            group_cad = copy(cad)
            group_cad.query_start_loc = query_start
            group_cad.query_start_loc_cpu = query_start_cpu
            group_cad.seq_lens = prepared.seq_lens
            group_cad._seq_lens_cpu = seq_lens_cpu.clone()
            # Ascend adds a writable field; upstream 0.20 exposes a getter
            # backed by _seq_lens_cpu. Keep both forms valid.
            if "seq_lens_cpu" in vars(group_cad):
                group_cad.seq_lens_cpu = seq_lens_cpu.clone()
            group_cad.seq_lens_cpu_upper_bound = seq_lens_cpu.clone()
            group_cad._num_computed_tokens_cpu = seq_lens_cpu - self.num_query_per_req
            group_cad._num_computed_tokens_cache = None
            if "num_computed_tokens_cpu" in vars(group_cad):
                group_cad.num_computed_tokens_cpu = group_cad._num_computed_tokens_cpu.clone()
            group_cad.block_table_tensor = self._per_group_block_tables[gid][:batch_size]
            group_cad.slot_mapping = self._query_slots[gid][:num_input_tokens]
            group_cad.positions = self.positions[:num_input_tokens]
            group_cad.positions_cpu = positions_cpu
            group_cad.num_actual_tokens = num_queries
            group_cad.num_input_tokens = num_input_tokens
            group_cad.max_query_len = self.num_query_per_req
            group_cad.max_seq_len = int(seq_lens_cpu.max()) if batch_size else 0
            group_cad.actual_seq_lengths_q = [self.num_query_per_req] * batch_size
            group_cad.decode_token_per_req = self.num_query_per_req
            if group_cad.is_prefilling is not None:
                group_cad.is_prefilling = group_cad.is_prefilling[:batch_size]
            group_cad.attn_state = AscendAttentionState.ChunkedPrefill
            group_cad.causal = False
            group_cad.attn_mask = None
            # Ascend's 0.20 builders accept the target as the third argument.
            metadata = group.get_metadata_builder().build(0, group_cad, self.runner.get_model())
            for name in group.layer_names:
                per_layer[name] = metadata
        with self.tp_group_context, set_ascend_forward_context(
            per_layer,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=across_dp,
            num_actual_tokens=num_queries,
            aclgraph_runtime_mode=CUDAGraphMode.NONE,
            is_draft_model=True,
            draft_attn_metadatas=[per_layer],
        ):
            get_forward_context().moe_layer_index = 0
            context_states = self.model.combine_hidden_states(target_hidden_states)
            if context_states.shape != (num_context, self.hidden_size):
                raise ValueError("DSpark target projection returned an incompatible hidden-state shape")
            self.model.precompute_and_store_context_kv(
                context_states,
                target_positions,
                [context_slots[self._layer_group_idx[name]] for name in self.attn_layer_names],
            )
            get_forward_context().moe_layer_index = 0
            hidden = self.model(
                input_ids=self.input_ids[:num_input_tokens],
                positions=self.positions[:num_input_tokens],
                inputs_embeds=None,
            )
            seeds = torch.where(next_token_ids >= 0, next_token_ids, self.parallel_drafting_token_id)
            tokens = sample_dspark_greedy(
                self.model, hidden[prepared.sample_indices.long()], seeds, self.num_speculative_tokens
            )
        self._per_group_block_tables.clear()
        self._per_group_slot_mappings.clear()
        return tokens

    @torch.inference_mode()
    def dummy_run(self, num_tokens, num_reqs=0, is_profile=False, **kwargs):
        del kwargs
        num_reqs = min(num_reqs or self.max_batch_size, self.max_batch_size)
        num_queries = num_reqs * self.num_query_per_req
        num_input_tokens, across_dp, _ = self.runner._sync_metadata_across_dp(
            num_queries, is_draft_model=True
        )
        self.input_ids.fill_(self.parallel_drafting_token_id)
        self.positions.zero_()
        with self.tp_group_context, set_ascend_forward_context(
            None,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=across_dp,
            num_actual_tokens=num_queries,
            in_profile_run=is_profile,
            aclgraph_runtime_mode=CUDAGraphMode.NONE,
            is_draft_model=True,
            draft_attn_metadatas=[],
        ):
            get_forward_context().moe_layer_index = 0
            hidden = self.model(
                input_ids=self.input_ids[:num_input_tokens],
                positions=self.positions[:num_input_tokens],
                inputs_embeds=None,
            )
            self.model.compute_draft_logits(hidden[:num_queries])
