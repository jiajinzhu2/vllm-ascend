# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""A3 paged GQA attention for MRV2 DSpark, using flash-attn-npu 0.4.2.post1."""

from dataclasses import dataclass, field
from importlib.metadata import version

import torch
from vllm.config import VllmConfig
from vllm.v1.attention.backend import AttentionCGSupport, AttentionMetadataBuilder, AttentionType
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
from vllm.v1.kv_cache_interface import AttentionSpec

from vllm_ascend.attention.attention_v1 import AscendAttentionBackend, AscendAttentionBackendImpl, AscendMetadata
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata


@register_backend(AttentionBackendEnum.FLASH_ATTN)
class AscendDSparkFABackend(AscendAttentionBackend):
    # The external kernel addresses packed K and V, without a page stride.
    requires_contiguous_kv_cache = True

    @staticmethod
    def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str=""):
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN"

    @staticmethod
    def get_impl_cls() -> type["AscendDSparkFAImpl"]:
        return AscendDSparkFAImpl

    @staticmethod
    def get_builder_cls() -> type["AscendDSparkFAMetadataBuilder"]:
        return AscendDSparkFAMetadataBuilder

    @classmethod
    def supports_sliding_window(cls) -> bool:
        return True

    @classmethod
    def supports_non_causal(cls) -> bool:
        return True

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: str | None) -> bool:
        return kv_cache_dtype in (None, "auto", "float16", "bfloat16")

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return 0 < head_size <= 256

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[str, ...]:
        return ("NHD",)

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER


@dataclass
class AscendDSparkFAMetadata(AscendMetadata):
    # A step-local cache shared only by layers with identical scheduling inputs.
    scheduler_metadata: dict[tuple, torch.Tensor] = field(default_factory=dict)


class AscendDSparkFAMetadataBuilder(AttentionMetadataBuilder[AscendDSparkFAMetadata]):
    def __init__(
        self, kv_cache_spec: AttentionSpec, layer_names: list[str], vllm_config: VllmConfig, device: torch.device
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    @classmethod
    def get_cudagraph_support(cls, vllm_config: VllmConfig, kv_cache_spec: AttentionSpec) -> AttentionCGSupport:
        # AICPU metadata runs on a separate stream in 0.4.2.post1. Enable FULL
        # only after device-length changes have been validated across replay.
        return AttentionCGSupport.NEVER

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        fast_build: bool = False,
    ) -> AscendDSparkFAMetadata:
        common = common_attn_metadata
        num_reqs = common.num_reqs
        # CPU query boundaries are already known to MRV2. Never read device KV
        # lengths here: the CPU mirror may include rejected/lookahead tokens.
        return AscendDSparkFAMetadata(
            num_actual_tokens=common.num_actual_tokens,
            query_start_loc=common.query_start_loc[: num_reqs + 1],
            query_start_loc_gpu=common.query_start_loc[: num_reqs + 1],
            actual_seq_lengths_q=common.query_start_loc_cpu[1 : num_reqs + 1].tolist(),
            seq_lens=common.seq_lens[:num_reqs],
            seq_lens_gpu=common.seq_lens[:num_reqs],
            block_tables=common.block_table_tensor[:num_reqs],
            slot_mapping=common.slot_mapping[: common.num_actual_tokens],
            max_query_len=common.max_query_len,
            causal=common.causal,
            model_runner_type=self.vllm_config.model_config.runner_type,
        )


class AscendDSparkFAImpl(AscendAttentionBackendImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
        **kwargs,
    ):
        if alibi_slopes is not None or sinks is not None:
            raise ValueError("DSpark FA3 does not support ALiBi or attention sinks.")
        if attn_type != AttentionType.DECODER:
            raise ValueError("DSpark FA3 only supports decoder GQA attention.")
        if not AscendDSparkFABackend.supports_kv_cache_dtype(kv_cache_dtype):
            raise ValueError("DSpark FA3 requires an FP16/BF16 KV cache; set speculative_config.kv_cache_dtype=auto.")
        if sliding_window is not None and sliding_window <= 0:
            raise ValueError("DSpark FA3 sliding_window must be positive.")
        super().__init__(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            alibi_slopes=alibi_slopes,
            sliding_window=sliding_window,
            kv_cache_dtype=kv_cache_dtype,
            logits_soft_cap=logits_soft_cap,
            attn_type=attn_type,
            kv_sharing_target_layer_name=kv_sharing_target_layer_name,
            sinks=sinks,
            **kwargs,
        )
        if self.use_bnsd_kv_cache:
            raise ValueError("DSpark FA3 requires the NHD KV cache layout.")
        self.softcap = float(logits_soft_cap or 0.0)
        # Worker-only import: the package probes the runtime NPU on import.
        try:
            if version("flash-attn-npu") != "0.4.2.post1":
                raise ValueError("DSpark FA3 currently requires flash-attn-npu==0.4.2.post1 for its metadata ABI.")
            from flash_attn_npu_3 import flash_attn_with_kvcache, get_scheduler_metadata
        except ImportError as exc:
            raise ImportError("DSpark FA3 requires flash-attn-npu==0.4.2.post1 with the Ascend 910 backend.") from exc
        self._fa3_fn = flash_attn_with_kvcache
        self._get_scheduler_metadata = get_scheduler_metadata

    def forward_impl(self, query, key, value, kv_cache, attn_metadata: AscendDSparkFAMetadata, output):
        num_tokens = attn_metadata.num_actual_tokens
        if num_tokens == 0:
            return output
        if not self.key_cache.is_contiguous() or not self.value_cache.is_contiguous():
            raise ValueError("DSpark FA3 requires contiguous NHD K/V caches from the MRV2 cache allocator.")
        query = query[:num_tokens]
        block_size = self.key_cache.shape[1]
        num_reqs = attn_metadata.seq_lens_gpu.shape[0]
        causal = attn_metadata.causal
        if self.sliding_window is None:
            window_size = (-1, -1)
        else:
            # vLLM's window includes the query itself. Non-causal SWA uses
            # the same symmetric window convention as its GPU FA backend.
            window_size = (self.sliding_window - 1, 0 if causal else self.sliding_window - 1)
        params = dict(
            cache_seqlens=attn_metadata.seq_lens_gpu,
            cu_seqlens_q=attn_metadata.query_start_loc_gpu,
            max_seqlen_q=attn_metadata.max_query_len,
            causal=causal,
            window_size=window_size,
            softmax_scale=self.scale,
            softcap=self.softcap,
            num_splits=1,
        )
        # This is the paged capacity, not the maximum dynamic KV length. The
        # 0.4.2.post1 metadata fingerprint requires it to match the forward.
        max_seqlen_k = block_size * attn_metadata.block_tables.shape[1]
        schedule_key = (
            num_reqs,
            attn_metadata.max_query_len,
            max_seqlen_k,
            self.num_heads,
            self.num_kv_heads,
            self.head_size,
            query.dtype,
            block_size,
            causal,
            window_size,
            self.scale,
            self.softcap,
        )
        scheduler_metadata = attn_metadata.scheduler_metadata.get(schedule_key)
        if scheduler_metadata is None:
            scheduler_metadata = self._get_scheduler_metadata(
                batch_size=num_reqs,
                max_seqlen_k=max_seqlen_k,
                num_heads_q=self.num_heads,
                num_heads_kv=self.num_kv_heads,
                headdim=self.head_size,
                qkv_dtype=query.dtype,
                page_size=block_size,
                **params,
            )
            attn_metadata.scheduler_metadata[schedule_key] = scheduler_metadata
        # Ascend's existing cache-update hook writes K/V before attention.
        # Do not pass append-KV arguments: that would write the query twice.
        attn_output = self._fa3_fn(
            query,
            self.key_cache,
            self.value_cache,
            page_table=attn_metadata.block_tables,
            scheduler_metadata=scheduler_metadata,
            **params,
        )
        output[:num_tokens].copy_(attn_output)
        return output
