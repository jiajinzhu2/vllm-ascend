# HY4 BF16 DFlash on vLLM-Ascend 0.20.2

This **experimental source adapter** targets `releases/v0.20.2rc`, with a
78-layer HY4 MXFP8 target and a separate five-layer BF16 `hy4dflymodels` draft.
It implements the November 4 checkpoint description. Private checkpoint keys,
real NPU execution, distributed capture, acceptance rates and performance have
**not** been validated. Keep the PR in Draft until hardware validation passes.

## Public HY4 compatibility

The target adapter includes gated MLA, learned sinks, full/shared indexer
scheduling, iHC, checkpoint mapping and retained per-layer indexer results.
The public [Hy4-preview configuration](https://huggingface.co/tencent/Hy4-preview/blob/main/config.json)
has 78 layers, hidden size 6144, 64 query heads, latent rank 512, rotary width 64,
nonrotary query width 192 and value width 256. At TP8 there are eight local
query heads. Public native MTP is not the separate five-layer DFlash checkpoint.
Preserve the private checkpoint's rotary parameters and `mrope_section`.

Architecture checks used the
[pinned upstream HY4 implementation](https://github.com/vllm-project/vllm/blob/dd4e40142a3280ef404493e5117bde0c60c6e389/vllm/models/hy_v4/nvidia/model.py).
The [Ascend HY4 port, PR #15377](https://github.com/vllm-project/vllm-ascend/pull/15377),
was still unmerged when checked on November 4. This 0.20.2 adapter is not a
backport of that PR or of the unavailable `hy4-a3` image source.

The public `enable_lm_head_fp32` setting is honored. Because 0.20.2 lacks the
newer mixed-dtype head API, this adapter stores the sharded, unquantized LM head
in FP32 and casts its input to FP32. That costs two additional bytes per local
head parameter compared with BF16; it is a correctness baseline, not a claim
of optimal A5 head performance. Both target and draft call the same target
logits processor, including capping. The existing Ascend MXFP8 MoE path already
passes `swiglu_limit` to its gated activation; no second clamp is added.

## Checkpoint contract

The November 4 document supersedes the earlier no-head description:
**hidden correction and a confidence head are enabled by default**.
`mtp_fusion_init=random` describes training initialization. Inference requires
trained fusion/correction/confidence tensors; missing tensors fail loading.
An older no-head checkpoint must explicitly set both `use_hidden_correction`
and `use_confidence_head` to `false`. Do not disable them to bypass missing
weights in the updated checkpoint. `use_markov_head` is a legacy alias for
hidden correction; conflicting flags are rejected.

Keep the complete config. These fields do not replace its attention dimensions,
rotary parameters or other metadata:

```json
{
  "architectures": ["hy4dflymodels"],
  "model_type": "dflash",
  "torch_dtype": "bfloat16",
  "num_hidden_layers": 5,
  "hidden_size": 6144,
  "intermediate_size": 18432,
  "target_layer_ids": [60, 64, 68, 72, 76],
  "target_kv_layers": [61, 65, 69, 73, 77],
  "target_index_layers": [61, 65, 69, 73, 77],
  "target_init_attn_layers": [61, 65, 69, 73, 77],
  "context_local_window": 8,
  "dflash_causal_attn": true,
  "mtp_fusion_slot0": true,
  "use_hidden_correction": true,
  "use_confidence_head": true,
  "dflash_config": {
    "mask_token_id": 120009,
    "context_topk": 2048,
    "ascend_attention_backend": "sfa",
    "ascend_kv_layout": "paged"
  }
}
```

`target_kv_layers` and `target_index_layers` are literal zero-based runtime layer
IDs. `target_layer_ids` and `target_init_attn_layers` retain training metadata;
there is no implicit index shift or inference-time attention-weight copying.
`HY4DflyModels` and vLLM's DFlash-prefixed architecture aliases are registered.
The model plugin also registers the missing HF config and target MLA converter.

## Execution and DSpark reference

1. Read the target final-normalized hidden at `anchor - 1`, and the anchor token
   just produced by the target. No target hidden for the anchor exists yet.
2. Each draft layer reads its own target layer's latent KV and already-rotated
   RoPE keys. Select top-2048 at `anchor - 1`, union the preceding eight
   same-document positions, deduplicate and keep only positions below the
   anchor. Prefixes at most 2048 tokens use all visible context.
3. Form `[anchor, mask, mask, mask, mask, mask, mask, mask]`. Slot 0 alone receives
   trained MTP fusion: `E(anchor) + gate * eh_proj([norm(E(anchor)); norm(h)])`.
4. Run five dense gated-MLA/SwiGLU layers once, then final RMSNorm. Each query
   attends jointly to selected target context and causal current-block KV.
   The draft computes its own current-block KV, without a context projection,
   draft indexer, historical draft cache or cross-block draft context.
5. For slots 1 through 7, serially compute
   `h_corrected = h + down(SiLU(gate(z)) * up(z))`, where
   `z = [norm(h); norm(E(previous_token))]`. The first previous token is the
   anchor; subsequent ones are the actual preceding draft proposals. The
   correction MLP is dense, with widths 12288 → 6144 → 6144.
6. Apply the **shared target LM head** and argmax at each step. In parallel with
   that step's logits path, the confidence linear (6144 → 1, with bias) reads
   the corrected hidden and produces an observational acceptance logit. Its
   sigmoid is a trained prediction, not measured acceptance or sampler `q`.
   Confidence never changes hidden states, logits, proposals or verification.

The backbone runs once, but correction and the head have seven dependent steps.
Greedy proposals have a delta distribution, represented by `draft_probs=None`
in the existing rejection sampler. All three M-RoPE axes are preserved;
distinct axes require `mrope_section` with three interleaved rotary-pair counts.
Cache addresses and gathered slots never replace original rotary coordinates.

The proposer retains the compacted-to-original row map for target indexer
reads; target hidden states use compacted rows. Padded/compacted paths both
need hardware validation. [Main's DSpark implementation](https://github.com/vllm-project/vllm-ascend/blob/b87b0d881533f390747b2ee885615333174c506e/vllm_ascend/models/deepseek_v41/dspark.py)
is the reference for separate loading, one block forward and serial auxiliary
heads. HY's direct KV sharing does not use DSpark context projection or SWA KV.

## Strict weight loading

The private checkpoint's exact names/shapes are unavailable. Supported canonical
keys are below; vendor spellings require explicit `dflash_config.weight_name_mapping`.

| Component | Keys |
| --- | --- |
| Layer norms | `model.layers.{0..4}.input_layernorm.weight`, `post_attention_layernorm.weight` |
| MLA under each layer | `self_attn.q_a_proj.weight`, `q_a_layernorm.weight`, `q_b_proj.weight`, `kv_a_proj_with_mqa.weight`, `kv_a_layernorm.weight`, `kv_b_proj.weight`, `o_proj.weight`, `linear_gate.weight` |
| Optional learned sink under each layer | `self_attn.learnable_sink_param` when configured |
| SwiGLU under each layer | `mlp.gate_proj.weight`, `up_proj.weight`, `down_proj.weight` |
| Slot-0 fusion | `model.fusion.embedding_norm.weight`, `hidden_norm.weight`, `eh_proj.weight`, `gate` |
| Hidden correction | `model.correction.hidden_norm.weight`, `embedding_norm.weight`, `gate_proj.weight`, `up_proj.weight`, `down_proj.weight` |
| Confidence | `model.confidence_head.weight`, `model.confidence_head.bias` |
| Final norm | `model.norm.weight` |
| Optional embedding / redundant head copy | `model.embed_tokens.weight`, `lm_head.weight` |

The correction gate/up are column-parallel; down is row-parallel. Confidence
and fusion are replicated. All draft-trained parameters remain unquantized
BF16. Packed `fused_qkv_a_proj` and `gate_up_proj` are accepted. Unprefixed keys
receive `model.`. Unused draft indexer weights/rotary buffers are skipped; other
unknown names, missing parameters and partial packed shards fail loading.
Embedding falls back to the target when absent. The target head is always
shared, even if a redundant `lm_head.weight` is included in the draft export.

Inspect checkpoint headers on the serving host before mapping names:

```bash
python tools/inspect_hy_dflash_checkpoint.py /models/HY4-draft-bf16 --output /tmp/hy4-manifest.json
```

This reads JSON and safetensors headers without loading tensor data or an NPU.
Do not manufacture random missing weights or infer aliases from similar names.

## Paged A5 Sparse Flash Attention

`dflash_config.ascend_attention_backend` accepts `auto`, `sfa` or `reference`.
`auto` selects native SFA on A5 and the numerical reference elsewhere; `sfa`
requires A5. Native SFA requires BF16, latent rank 512, rotary width 64, and
1–128 local query heads. Learned sinks require the `sinks` operator argument
and the corresponding `aclnnSparseFlashAttentionV2` kernel. A5 / CANN 9.2.0 is
the intended environment; a torch_npu version string alone is insufficient:

```bash
python -c 'import torch, torch_npu; print(torch.ops.npu.npu_sparse_flash_attention.default._schema)'
```

See the [official SFA constraints](https://github.com/Ascend/op-plugin/blob/35742aa0e083ec06cd9ef317d17f2eca95fa4e7d/docs/zh/custom_APIs/torch_npu/torch_npu-npu_sparse_flash_attention.md).
Missing schema features or kernel failures are fatal; they never silently drop
sinks or switch to Python attention.

`dflash_config.ascend_kv_layout` accepts `auto`, `paged` or `gather`. `auto`
chooses paged KV when all five layers use native SFA. `paged` requires native
SFA; `gather` retains the contiguous comparison path.

For paged KV, the worker extends the selected target layer allocations with
private scratch pages. Their bytes are charged before memory profiling and
reserved outside scheduler `num_blocks`; original scheduler config and prefix
cache allocation remain unchanged. Independently allocated layer pools are
required. The scratch suffix contains all target cache planes to preserve the
allocator's layout, although draft attention uses only latent/RoPE planes.
Each padded request row has unique scratch slots; the native scatter writer
replaces all eight current-block keys on every execution. Historical target
pages, including shared prefix pages, are read-only. No draft historical cache
is registered with the scheduler, and scratch never becomes accepted history.

SFA reads the live target allocation as `PA_BSND`, with a fixed block table
whose appended columns address scratch. Sparse indices address original
logical context plus an artificial suffix for the causal block. That suffix
is only attention addressing: rotary positions remain the original positions.
The block table, int32 device sequence lengths and indices keep fixed shapes.
No gather or concatenation of historical KV occurs in this attention path.
The `gather` comparison materializes selected KV and concatenates current-block
KV in a fixed BSND buffer.

Both layouts use one joint softmax, `attention_mode=2`, `sparse_block_size=1`,
`sparse_mode=0`, causal per-query indices and trailing `-1` padding. K and V alias
the same latent storage; sinks are FP32 and LSE output is disabled. BF16
absorption/output projections may round differently from the FP32 reference.

## Model runner v1 graphs and first hardware run

Keep the target eager; its learned-sink implementation still includes a
PyTorch reference fallback. Draft `enforce_eager=false` enables an independent
v1 FULL-graph path. Runner v2 is unsupported. A starting command for TP8/EP8:

```bash
vllm serve /models/HY4-mxfp8 \
  --quantization ascend \
  --dtype bfloat16 --kv-cache-dtype auto \
  --tensor-parallel-size 8 --enable-expert-parallel \
  --enforce-eager --no-async-scheduling \
  --max-model-len 4096 --max-num-seqs 1 --max-num-batched-tokens 256 \
  --speculative-config '{"method":"dflash","model":"/models/HY4-draft-bf16","num_speculative_tokens":7,"draft_tensor_parallel_size":8,"enforce_eager":false,"quantization":null}'
```

The target needs MXFP8 metadata compatible with the Ascend loader. BF16 dtype
controls activations/default KV, not quantized target weight storage. The
separate draft config clears quantization and uses BF16. Target/draft TP must
match; PP/DP/PCP/DCP must be one, without LoRA or separate LM-head TP.

In disaggregated serving, start a prefill-only producer **without** speculative
config. Only decode runs DFlash. The decode side must execute a target forward
that supplies current anchor-minus-one hidden/indexer snapshots; KV transfer
alone does not supply these. End-to-end PD transport is unvalidated.

Draft capture covers fusion, five backbone layers, scratch writes, SFA,
seven dependent correction/head/argmax steps and confidence. Context selection,
block-table updates, validation and copying live inputs remain outside capture.
Confidence uses a persistent output buffer; replay publishes copies for real
requests, just like proposals. It does not depend on Python assignments running
again during replay. Padding is cleared; request reordering updates metadata.

Buckets count requests, not target padded tokens: powers of two up to
`max_num_seqs`, plus that maximum, captured largest first. A maximum of four
captures 4, 2 and 1. Start small and measure memory before increasing concurrency.
Startup logs selected layouts/backends and captured batches. CPU stand-ins
cannot establish CANN capture or TP collective compatibility.

## Validation

Portable tests execute actual selection, correction, loading and proposer code
with CPU infrastructure replacements. They check paged/gather numerical
semantics, learned sinks, shared prefixes, no historical writes, reserved-page
bounds, request reordering/shrinking, graph buffer ownership, serial token
conditioning, confidence isolation and missing trained weights:

```bash
pytest --noconftest tests/ut/spec_decode/test_hy_dflash_proposer.py -q
```

Run the standalone native A5 test first. It needs no private weights and checks
both KV layouts, sink/no-sink, full/holed/empty contexts, causal isolation,
fixed-address NPUGraph replay, historical KV preservation and warmed timing:

```bash
python tests/e2e/singlecard/test_hy_dflash_sfa.py --heads 8 --nope-dim 192 --value-dim 256
```

Then load the real checkpoints and require SFA on every rank:

```bash
python tests/e2e/multicard/test_hy_dflash.py --target /models/HY4-mxfp8 --draft /models/HY4-draft-bf16 --tp 8 --enable-expert-parallel --require-sfa --kv-layout both
```

Separate processes run target baseline plus both KV layouts in eager/graph,
padded/compacted modes with long/short prompts, changing batches and repeated
prefixes. They compare final tokens, per-step draft proposals and confidence;
final tokens alone could conceal a broken draft with zero acceptance. Tracing
synchronizes the device and is for correctness. `--benchmark-repeats 5` measures
additional generations after disabling tracing. Check target-alone fidelity
against a trusted HY4 implementation before interpreting draft acceptance.
For a divergence, compare logical indices, physical slots, selected latent/RoPE
KV, fusion and the first divergent attention/correction/head output.
