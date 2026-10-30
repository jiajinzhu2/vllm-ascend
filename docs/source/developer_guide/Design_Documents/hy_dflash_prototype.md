# HY4 DFlash KV-share draft prototype

## Status and scope

Experimental, opt-in adaptation for vLLM 0.20.2 and Ascend
`releases/v0.20.2rc`. This revision replaces the earlier three-layer,
projected-context/MoE/path-selector sketch with the supplied **five-layer dense
KV-share** architecture.

The patch implements a PyTorch eager attention correctness path, draft module
composition, and proposer orchestration. It does **not** register a speculative
checkpoint architecture or guess checkpoint tensor names. An existing HY4 target
and vendor draft loader must implement the integration bridge below. Having
ordinary HY4 target inference working is not, by itself, sufficient to provide
that bridge.

No checkpoint loading, NPU execution, acceptance-rate benchmark, ACL graph, or
distributed execution has been validated. Keep the PR in draft until those
integration checks pass.

## Architecture

| Component | Required behavior |
| --- | --- |
| Backbone | Five dense layers; hidden 6144; SwiGLU 6144 → 18432 → 6144 |
| Target KV sources | Draft layers 0–4 read target layers 61, 65, 69, 73, 77 |
| Context | Existing target latent KV, read-only; no context projection |
| DSA selection | Each target layer's indexer at anchor−1, top-2048 ∪ previous 8 |
| Query block | Anchor + seven mask embeddings; mask token ID 120009 |
| Attention | HY4 gated MLA; 3-D M-RoPE; causal within each isolated block |
| Fusion | Per-channel gated MTP fusion, slot 0 only |
| Correction | Residual SwiGLU on hidden + previous-token embedding |
| Output | Target LM head, seven sequential correction/argmax steps |
| Confidence | Linear(6144, 1), including bias, on corrected hidden |

There is no MoE, IHC merge, path selector, or causal convolution in this variant.
The draft does not run its own context indexer. It consumes the five target
indexers' selections, independently, at the correct accepted boundary.

### Execution sequence

1. Target forward and verification establish the accepted prefix and anchor.
2. For each source layer, retrieve its anchor−1 indexer row and read selected KV.
3. Construct eight embeddings and add MTP fusion only to slot 0.
4. Run the five-layer backbone once, using each layer's selected target KV plus
   its own current-block KV.
5. Apply final RMSNorm.
6. For slots 1 through 7, run correction, target LM head, argmax and confidence.
   Feed the actual sampled token as the next correction's previous token.
7. Target verification remains authoritative.

For the first predicted token, the previous token is the anchor. Slot 0 does not
produce a draft output. The backbone's mask inputs do not change during the
seven correction steps.

## Read-only context and dual-source MLA

`vllm_ascend/attention/hy_dflash.py` contains the actual selection, paged-cache
gather, absorbed MLA math and attention composition.

Logical token offsets are separate from M-RoPE coordinates. Context candidates
are filtered to the same document and strictly before the anchor, sorted and
deduplicated. Documents with at most 2048 visible context tokens explicitly use
their entire prefix. The fixed gather capacity is 2056, with invalid entries
masked; the current block's eight keys are separate and never sparsified.

For each layer:

- The eight queries share one context selection.
- Context keys already contain the target's rotated RoPE component. Do not
  rotate them again.
- Draft Q and current-block K receive the supplied 3-D M-RoPE positions.
- Current-block latent KV comes from that draft layer's own normalized
  `kv_a` output.
- Selected context and causal block keys participate in **one joint softmax**.
  Adding two independently normalized attention outputs is not equivalent.
- Request batches do not attend to one another, and draft block KV is transient:
  it is not committed as future target context.

The correctness kernel absorbs bias-free `kv_b` key expansion into Q and value
expansion after softmax accumulation. It avoids expanding all context into
per-head K/V. Scores and accumulation use FP32; output returns to query dtype.

The bridge must provide expansion weights compatible with the checkpoint's
shared latent basis for both sources. Equal dimensions alone do not prove
compatibility. Source-specific expansion, biased expansion, quantized packed
projections, or a different target cache layout require an explicit adaptation;
they must not be silently reinterpreted.

`HYDflashAttention` composes three already-loaded vendor modules:

- `project(hidden[B,8,H])` returns Q-nope, Q-rope, normalized block latent,
  and block K-rope.
- `rotary(positions[3,B,8], q_rope, k_rope)` returns rotated Q/K components.
  This wrapper preserves HY4's actual rotary section layout and scaling.
- `finish(hidden, output[B,8,heads,value_dim])` applies HY4's actual output
  gate and output projection in the trained order.

The gate activation, projection layout and rotary implementation are not
invented here. Tiny test modules demonstrate this contract, not the vendor's
checkpoint format. No fused Ascend MLA/DSA operator is claimed by this path.

## Loaded draft composition

`vllm_ascend/patch/worker/hy_dflash.py` adds composition modules without global
monkey-patching or a new model registration:

- `HYMTPFusion`: `E(anchor) + gate * eh_proj(norm(E(anchor)) || norm(h))`,
  where `h` is already target-final-normalized hidden at anchor−1.
- `HYDenseDraftLayer`: attention residual followed by dense MLP residual.
  Norm/MLP modules receive flattened token rows.
- `HYHiddenCorrection`: `h + down(silu(gate(z)) * up(z))`, with
  `z = norm(h) || norm(E(previous_token))`.
- `HYDflashAdapter`: exactly five layers, final norm, correction, LM-head
  callback and independent confidence head.

Pass loaded checkpoint modules and weights to these constructors. Do not
randomly initialize fusion/correction/confidence for inference: “random
initialization” describes their training initialization, not serving behavior.
The expected trained widths are 12288 → 6144 for fusion and
12288 → 6144 → 6144 for correction. The target final norm must not be applied a
second time by the provider; fusion's own hidden norm is an additional,
intentional normalization.

The vendor wrapper's `to_hy_dflash_adapter(target_model)` returns the composed
adapter, or the vendor loader can return an adapter directly. It is responsible
for embedding-sharing policy and for a `compute_logits` callback backed by the
target LM head with full-vocabulary logits. Projection wrappers and expansion
weight views must be constructed after weights have been loaded and transformed.

Loading this wrapper must not register new persistent draft attention/indexer
cache layers. Current-block KV stays inside the attention forward.

## Target integration bridge

The target exposes two read-only methods:

```python
get_hy_dflash_layer_names(layer_ids=(61, 65, 69, 73, 77))

get_hy_dflash_context(
    *,
    request_ids,
    anchor_positions,
    layer_ids=(61, 65, 69, 73, 77),
    block_size=8,
) -> HYTargetState
```

These are required new integration contracts, not existing upstream HY4 APIs.
No target architecture is changed by this patch. The existing target adaptation
must export its state through these methods.

`HYTargetState` contains:

- Five `HYTargetLayerState` objects in the requested layer order. Each includes
  `layer_id`, floating-point paged `cache[physical_blocks, block_size, R+Dr]`,
  request-ordered `block_table[B,max_blocks]`, and logical
  `topk_indices[B,<=2048]`.
- `document_starts[B]`, in sequence-token coordinates.
- `final_hidden[B,6144]`: target final-norm output at anchor−1.
- `query_positions[3,B,8]`: correct M-RoPE positions including multimodal deltas.

The cache's first R channels are normalized latent KV; its final Dr channels
are already-rotated key positions. A vendor-specific view/dequantization is
needed if the actual NPU cache uses another layout. The gather fails on a
selected token pointing to an unallocated block, rather than dropping context.

The provider must retain each layer's indexer results for the target forward
rows until verification determines anchor−1. A single shared scratch top-k
buffer overwritten by later layers is insufficient. Likewise, selecting the
last scheduled row instead of the last accepted row leaks rejected future
tokens. Resolve rows using request identity and logical position, including
batch reordering, chunked prefill, rejection and accepted-prefix compaction.
Read references only when target work producing them has completed on the
relevant stream.

The supplied path is intentionally not required to develop/test this boundary.
Actual checkpoint names, physical cache layout and retained indexer-row access
remain vendor integration work.

## Proposer responsibilities

`AscendHYDflashProposer` still inherits `AscendDflashProposer`, but replaces
ordinary DFlash's projected-context execution.

| Method | Responsibility in this variant |
| --- | --- |
| `load_model` | Load vendor draft, compose adapter, identify five target source layers |
| `initialize_attn_backend` | Bind target KV groups; no new draft context cache |
| `set_inputs_first_pass` | Build anchor/mask blocks and accepted logical boundaries |
| `prepare_context` | Read target state; select, deduplicate and gather each layer's KV |
| `build_model_inputs_first_pass` | Pass blocks, 3-D positions, final hidden and five contexts |
| `_propose` | Enter eager draft context, bypass shared single-layer metadata builder |
| `_run_merged_draft` | Run backbone once, then seven sequential correction/head steps |
| `commit_draft_probs` | Publish deterministic proposals; reject unsupported random q |
| `verification_probs` | Return actual draft q only; never confidence |
| `dummy_run` | Profile draft math with masked contexts and no live target cache access |

`initialize_attn_backend` does not replace `set_inputs_first_pass`: one binds
cache groups at initialization; the other prepares the current batch every
round.

There is no `prefill_context` projection/cache-write stage. Ordinary DFlash's
`_context_slot_mapping_buffer` is removed. The only slot mapping here is the
query metadata's all-minus-one placeholder; actual block KV is transient.
Logical context indices are translated through each target layer's block table
solely to **read** existing cache slots.

The unmodified shared proposer still handles its existing input preparation.
HY's first pass uses rejection-adjusted `seq_lens`, not a M-RoPE axis, to derive
the anchor. Its private `_propose` avoids routing HY through Qwen DFlash's
`combine_hidden_states` or generic per-layer attention metadata assumptions.

### Confidence and sampling

The confidence output is an acceptance logit; sigmoid is a learned estimate
of distribution overlap, not a measured overlap and not the actual rejection
ratio. `confidence_logits[B,7]` is diagnostic, associated with
`confidence_request_ids`. It never modifies hidden states, logits, draft
length, or target verification.

This implementation uses deterministic argmax proposals, as the existing base
path does. Therefore draft q is a delta distribution and
`verification_probs()` returns `None` to the existing rejection sampler.
It must not return a softmax distribution after generating tokens with argmax.
The runner continues passing its original target logits to verification.
Random proposal sampling and request-aligned storage of its actual q are not
implemented.

## Opt-in and limitations

The loaded draft HF configuration opts in explicitly:

```json
{
  "dflash_config": {
    "ascend_proposer": "hy",
    "mask_token_id": 120009
  }
}
```

Use speculative method `dflash`, parallel drafting, exactly seven speculative
tokens, and speculative `enforce_eager=true`. This is a configuration fragment,
not a complete model configuration or a standalone serving command.

TP, DP, PP, PCP and DCP must all be 1; draft TP must be 1. LoRA and separate
LM-head TP are rejected. Other DFlash models remain on their original path.

The reference gather has one allocation-validity host synchronization per source
layer; it is documented and deliberately outside attention forward. It still
affects latency. Gather allocations, FP32 reference attention, seven LM-head
calls and confidence computation need NPU profiling before performance claims.
Distributed collective semantics, graph-safe metadata and optimized Ascend
sparse kernels are separate follow-up work.

## Validation

Portable tests:

```bash
python -m pytest --noconftest -q tests/ut/spec_decode/test_hy_dflash_proposer.py
```

The tests execute real selection, gather, absorbed attention, fusion,
correction and five-layer composition with tiny loaded modules. They include
an independent expanded-K/V attention reference in FP32/BF16, causal and batch
isolation, context union/bounds, physical-page lookup, read-only cache, 3-D
positions, serial predecessor tokens, confidence isolation, factory routing,
runner verification and unsupported-mode guards.

The vLLM/Ascend infrastructure is mocked; the runner sampling method is loaded
from its source AST. These are not full runner, Triton, HY4 checkpoint or NPU
integration tests. Before marking ready, validate vendor bridge/weight mapping,
prefill and rejection/reordering with real target indexers, NPU numerical
parity, end-to-end generation, memory use and acceptance/throughput.
