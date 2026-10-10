# A3 DSpark FlashAttention with device metadata

This experimental backend connects the paged GQA layers of a MiniMax M3 DSpark
drafter to `flash-attn-npu==0.4.2.post1`. The target keeps its existing MSA
backend. ModelRunner V2 is required.

The backend is selected only while constructing a DSpark draft model, through
`speculative_config.attention_backend=FLASH_ATTN`. Other attention selections
keep their existing behavior, including the training-consistency FA3 backend.

## Operator installation

Install the external operator into the same Python environment as vLLM Ascend.
Follow the package's CANN and PyTorch requirements in the
[operator repository](https://github.com/MinghuasLab/flash-attention-npu/tree/v0.4.2.post1).
For a local build of the 910 implementation:

```bash
FLASH_ATTENTION_FORCE_BUILD=TRUE \
FLASH_ATTN_BUILD_VERSION=v3 \
FLASH_ATTN_BUILD_NPU=910 \
python -m pip install 'flash-attn-npu==0.4.2.post1' --no-build-isolation
```

The public Python module for this package is `flash_attn_npu_3`.
It must provide both `flash_attn_with_kvcache` and `get_scheduler_metadata`.

## Enabling the draft backend

Keep the target model's existing MSA startup options. Add the following draft
options to its speculative configuration:

```json
{
  "method": "dspark",
  "model": "/path/to/minimax-m3-dspark-draft",
  "num_speculative_tokens": 7,
  "attention_backend": "FLASH_ATTN",
  "kv_cache_dtype": "auto",
  "enforce_eager": true
}
```

Use the checkpoint's actual speculative block width rather than assuming seven
tokens. Set `VLLM_USE_V2_MODEL_RUNNER=1` and `VLLM_KV_CACHE_LAYOUT=NHD`.
The draft KV cache must be FP16/BF16 even if the target uses a quantized cache.

Attention visibility comes from the draft checkpoint. For the Qwen3-style
DSpark model already registered by vLLM Ascend, the following draft HF config
selects non-causal SWA with a window parameter of 1024:

```json
{
  "dflash_config": {
    "use_swa": true,
    "swa_window_size": 1024,
    "causal": false
  }
}
```

These are checkpoint semantics, not speculative-config fields. Verify them
against the trained draft before changing them. Other DSpark model classes
must supply their own layer window and `get_draft_attn_causal()` declarations.

## Mask and metadata behavior

- Full attention uses `window_size=(-1, -1)`.
- Causal SWA 1024 uses `causal=True, window_size=(1023, 0)`.
- Non-causal SWA 1024 uses `causal=False, window_size=(1023, 1023)`, matching
  vLLM's symmetric GPU FlashAttention convention.
- Query positions are aligned to the end of the valid KV sequence:
  `absolute_q = valid_kv_length - query_length + local_q_index`.
- The exact device KV lengths are used after rejection sampling. CPU upper
  bounds are not used to expose rejected or unwritten cache tokens.
- AICPU scheduling metadata and the attention call receive matching window,
  causality, scale, softcap, query-length bound, and split parameters.
- Metadata is shared by matching layers within one forward step and rebuilt
  for the next step.
- MRV2 creates contiguous K/V views over the existing allocation. It does not
  copy or transpose the whole KV cache each step.

The public operator's standard sliding window is distinct from a specialized
Anchor block mask. A checkpoint requiring different anchor visibility needs
an explicit operator contract and additional integration work.

## Current validation boundary

This is a draft integration. Only A3, decoder GQA, FP16/BF16, and the NHD cache
layout are enabled. Draft PCP/DCP, attention sinks, ALiBi, RSWA, multimodal
prefix masks, and quantized draft KV caches are outside this implementation.

The metadata builder advertises no FULL graph support. The package schedules
its metadata work on a separate AICPU stream, so a future graph path must first
prove that updated device lengths change the schedule on every replay. The
draft runs eagerly; the target's graph configuration is not changed by this
feature. This can affect throughput relative to an existing graph-based draft.
An end-to-end performance comparison is required before deployment.

The new NPU operator test compares paged GQA against dense SDPA for full/SWA,
causal/non-causal, FP16/BF16, and shrinking post-rejection sequence lengths:

```bash
python -m pytest -q \
  tests/e2e/nightly/single_node/ops/singlecard_ops/test_dspark_fa3_attention.py
```

See the [detailed MiniMax M3 adaptation report](../../developer_guide/Design_Documents/minimax_m3_dspark_flash_attention_npu.md)
for implementation scope, verification evidence, and the remaining deployment gates.
