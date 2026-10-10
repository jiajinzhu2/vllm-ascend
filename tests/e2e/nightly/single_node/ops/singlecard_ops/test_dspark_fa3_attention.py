# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Actual A3 operator vs dense SDPA, including post-rejection length changes."""

import pytest
import torch
import torch.nn.functional as F

from vllm_ascend.attention.dspark_fa3 import AscendDSparkFAImpl, AscendDSparkFAMetadata


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("sliding_window", [None, 1024])
def test_dspark_fa3_against_sdpa(dtype, causal, sliding_window):
    if not torch.npu.is_available():
        pytest.skip("Requires an Ascend A3 NPU")
    fa3 = pytest.importorskip("flash_attn_npu_3")
    torch.manual_seed(123)
    query_lengths = [7, 3]
    query_starts = torch.tensor([0, 7, 10], dtype=torch.int32, device="npu")
    cache_lengths = torch.tensor([1025, 2053], dtype=torch.int32, device="npu")
    num_heads, num_kv_heads, head_size = 4, 2, 64
    block_size, blocks_per_request = 128, 24
    num_blocks = blocks_per_request * len(query_lengths)
    page_table_cpu = torch.randperm(num_blocks).reshape(len(query_lengths), blocks_per_request).to(torch.int32)
    page_table = page_table_cpu.npu()
    key_cpu = torch.randn(num_blocks, block_size, num_kv_heads, head_size, dtype=dtype)
    value_cpu = torch.randn_like(key_cpu)
    query_cpu = torch.randn(sum(query_lengths), num_heads, head_size, dtype=dtype)
    query = query_cpu.npu()

    impl = object.__new__(AscendDSparkFAImpl)
    impl.num_heads, impl.num_kv_heads, impl.head_size = num_heads, num_kv_heads, head_size
    impl.scale, impl.softcap = 0.17, 0.0
    impl.sliding_window = sliding_window
    impl.key_cache, impl.value_cache = key_cpu.npu(), value_cpu.npu()
    impl._get_scheduler_metadata = fa3.get_scheduler_metadata
    impl._fa3_fn = fa3.flash_attn_with_kvcache

    # Reuse the device length buffer, but rebuild the step-local schedule.
    # A later step can shrink after rejection, rather than always growing.
    for lengths in ([1025, 2053], [1021, 2049]):
        cache_lengths.copy_(torch.tensor(lengths, dtype=torch.int32))
        metadata = AscendDSparkFAMetadata(
            num_actual_tokens=sum(query_lengths),
            max_query_len=max(query_lengths),
            query_start_loc_gpu=query_starts,
            seq_lens_gpu=cache_lengths,
            block_tables=page_table,
            causal=causal,
        )
        output = torch.empty_like(query)
        impl.forward_impl(query, None, None, None, metadata, output)
        expected = []
        query_offset = 0
        for request_idx, (q_len, kv_len) in enumerate(zip(query_lengths, lengths)):
            pages = page_table_cpu[request_idx].long()
            k = key_cpu[pages].flatten(0, 1)[:kv_len].float()
            v = value_cpu[pages].flatten(0, 1)[:kv_len].float()
            q = query_cpu[query_offset : query_offset + q_len].float()
            query_offset += q_len
            q_positions = torch.arange(kv_len - q_len, kv_len)[:, None]
            k_positions = torch.arange(kv_len)[None, :]
            visible = torch.ones(q_len, kv_len, dtype=torch.bool)
            if causal:
                visible &= k_positions <= q_positions
            if sliding_window is not None:
                visible &= k_positions >= q_positions - sliding_window + 1
                visible &= k_positions <= q_positions + sliding_window - 1
            grouped_k = k.repeat_interleave(num_heads // num_kv_heads, dim=1)
            grouped_v = v.repeat_interleave(num_heads // num_kv_heads, dim=1)
            reference = F.scaled_dot_product_attention(
                q.transpose(0, 1),
                grouped_k.transpose(0, 1),
                grouped_v.transpose(0, 1),
                attn_mask=visible,
                scale=impl.scale,
            ).transpose(0, 1)
            expected.append(reference)
        torch.testing.assert_close(output.cpu().float(), torch.cat(expected), rtol=2e-2, atol=2e-2)
        # Perturb every token outside the valid KV range. None may affect the
        # draft, even though the scheduler's CPU upper bound could include it.
        poisoned_k, poisoned_v = key_cpu.clone(), value_cpu.clone()
        for request_idx, kv_len in enumerate(lengths):
            for position in range(kv_len, blocks_per_request * block_size):
                page = page_table_cpu[request_idx, position // block_size]
                poisoned_k[page, position % block_size].fill_(50)
                poisoned_v[page, position % block_size].fill_(100)
        impl.key_cache.copy_(poisoned_k)
        impl.value_cache.copy_(poisoned_v)
        poisoned_output = torch.empty_like(query)
        impl.forward_impl(query, None, None, None, metadata, poisoned_output)
        torch.testing.assert_close(poisoned_output, output, rtol=0, atol=0)
        impl.key_cache.copy_(key_cpu)
        impl.value_cache.copy_(value_cpu)
