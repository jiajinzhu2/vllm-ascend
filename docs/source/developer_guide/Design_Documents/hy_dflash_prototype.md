# HY DFlash proposer prototype (v0.20.2)

## Status and scope

This is **WIP orchestration**, not support for a released HY/GLM DFlash2 checkpoint.
No draft source, checkpoint configuration, or trained projection/correction weights
were available. The interfaces below are proposed adapter contracts, not reverse
engineered checkpoint APIs. A public HY target or its native MTP head is not a
substitute for this draft.

`AscendHYDflashProposer` inherits `AscendDflashProposer`. Ordinary DFlash keeps its
existing noncausal behavior. HY opts into causal metadata and a separate sampling
loop. The prototype accepts a single-device eager configuration only: TP, DP, PP,
PCP and DCP must all be 1; text-only, no LoRA or IHC streams. "No IHC" means an
ordinary `[tokens, hidden_size]` residual representation, **not** replacing the
draft's MoE FFN with a dense MLP.

The intended model has three causal gated MLA/DSA + MoE layers. This PR does not
implement or register that model, change its target MoE class, guess expert weight
names, or claim NPU/ACL graph correctness. It provides the proposer boundary to
which an actual model implementation can be attached.

## Execution and ownership

```text
Target forward / verification
    -> accepted target hidden rows (aux layer features if configured)
    -> combine_hidden_states
    -> set_inputs_first_pass: context + [anchor, mask ... mask]
    -> prepare_context / prefill_context                     [phase A, eager]
         model context normalization / projection
         layer-major packing (MLA latent + RoPE + DSA index keys)
         per-layer cache writes, ignoring slot -1
    -> build_model_inputs_first_pass                         [query only]
    -> _run_merged_draft                                     [phase B]
         model: three causal MLA/DSA + gate + MoE layers
         for i = 0 .. K-1:
             correction(hidden block, sampled prefix, i)
             path selection -> LM head -> deterministic token
    -> commit_draft_probs                                   [publish q]
    -> next target verification: target logits p + draft q
         runner/rejection sampler decides acceptance, not the proposer
```

The query region is the **future graph-capture boundary**. Graph execution is
currently rejected at configuration time. `dummy_run` is complete eager
profiling/warmup, not a graph implementation.

| Proposer method | Responsibility | Cache/probability effect |
| --- | --- | --- |
| `set_inputs_first_pass` | Reuse DFlash packing of one accepted/bonus anchor and K masks per request; set query positions and `causal=True` | Keep target metadata unchanged; invalidate stale CPU mirrors |
| `context_slot_mapping` | Mark rejected rows in padded target batches as `-1` | Prevent rejected hidden states from becoming persistent context |
| `prepare_context` | Finalize accepted-context staging and invoke phase A | No query forward or sampling |
| `prefill_context` | Call the model's context projection/packing/cache writer | Write every draft layer's own MLA and indexer cache; profile with no writes |
| `build_model_inputs_first_pass` | Return query token IDs/positions only | Must not re-run context projection inside query replay |
| `_run_merged_draft` | One query forward, then correction/selection/LM head for each position | Query KV is provisional; only sampled prefixes are passed to correction |
| `commit_draft_probs` | Publish the proposal distribution contract | Does **not** accept tokens or commit query KV |
| `verification_probs` | Supply draft q to the existing rejection sampler | Does **not** compute the target's p |
| `dummy_run` | Exercise context projection, query forward and every sampling hook | Context slots are `None`, so no live context cache is modified |

## Context projection is not the query projection

The target supplies hidden features, not target KV tensors. Optional auxiliary
hidden states come from configured target layer outputs; the model's
`combine_hidden_states` maps those features to the draft hidden width. The actual
target hook locations and feature ordering must match training and are not guessed
here.

After that feature projection, each draft layer has its own context projection.
Packing all layers into one GEMM is an implementation optimization, not a change
to that per-layer meaning. In MLA, the cached representation is model-specific
(for example compressed latent KV plus a rotary key component), not Qwen DFlash's
ordinary multi-head K/V layout. DSA also needs context indexer keys. Therefore the
Qwen dense K/V patch must **not** be used as the HY MLA cache writer.

For a token at position `p`, the inherited packer produces:

```text
slot = block_table[request, p // block_size] * block_size + p % block_size
```

The HY layer marks rejected context rows invalid. On a rejection, the next anchor
starts immediately after the last accepted context position. Previous speculative
slots can remain physically allocated, but sequence lengths and sparse selection
must hide them. At the next target pass, accepted positions are overwritten with
target-derived context KV. A masked query token's KV is never promoted merely
because `commit_draft_probs` was called.

`prefill_context` handles the **new target rows in this invocation**, including
prefill chunks and accepted verification rows. It does not rebuild the complete
history every time; older accepted context remains in the draft cache. The model
must not share the target's query-specific DSA top-k indices. It may share immutable
weights as appropriate, but owns its own context/query cache and sparse selections.

## External model contract, version 1

The externally registered vLLM model wrapper must advertise
`hy_dflash_contract_version = 1` and supply the following methods. These are
interface requirements, not implementations included in this patch.

```python
combine_hidden_states(target_features) -> Tensor[T, H]

precompute_and_store_context_kv(
    context_states: Tensor[T, H],
    context_positions: Tensor[T],
    context_slot_mapping: Tensor[T] | None,
) -> None

forward(input_ids, positions, inputs_embeds=None) -> Tensor[B * (K + 1), H]

correct_hidden_states(
    available_hidden_states: Tensor[B, i + 2, H],
    previous_token_ids: Tensor[B, i],
    step: int,
) -> Tensor  # model-defined corrected path representation

select_hidden_path(corrected_paths, step: int) -> Tensor[B, H]

compute_logits(selected_hidden_states: Tensor[B, H]) -> Tensor[B, V]
```

Here `T` is the current context row count, `B` is the request count, `K` is the
speculation length and `H` is the draft hidden width. Column 0 is the anchor;
columns 1 through K are mask positions. Correction at step i predicts the token
for column i+1. Only the hidden prefix through that column and already sampled
tokens are exposed to the correction hook. Future hidden positions/tokens are not
passed. The model must also ensure that its base layers and any persistent
correction state do not leak future positions.

The contract assumes correction -> path selection -> LM head. If a real checkpoint
uses a different ordering or maintains additional path state, revise the contract
explicitly rather than silently falling back to ordinary DFlash sampling.

The model implementation must:

- Use the target's compatible MoE implementation unchanged, while loading the
  **draft's** expert/router/shared-expert weights and topology. Same class does not
  mean same weights or same expert count.
- Override the draft MLA/DSA projection/forward path where required. The future
  TP/CP adaptation must keep draft gate/o-projection tokens unsplit and avoid the
  target fused MLA path that imposes DP lockstep. This prototype rejects those
  distributed configurations; it does not pretend the overrides are implemented.
- Load the shared indexer `wk` in BF16 at model root and keep it out of fused
  quantized projections, as required by the intended checkpoint contract. No
  invented tensor name or shape is hardcoded in the proposer.
- Write both compressed MLA KV and DSA indexer context keys to the correct layer
  caches, ignoring `-1` slots. `None` slots mean projection-only profiling without
  storage. Query attention must use both context and query sources with correct
  causal visibility.
- Follow ordinary vLLM wrapper requirements for model registration, weight loading,
  embeddings, LM head and attention/cache registration. The prototype inherits
  the single draft attention-group assumption; custom indexer caches need a real
  backend integration before serving.
- Return ordinary 2-D hidden states, not IHC streams or a tuple of stream tensors.

Missing contract methods fail explicitly. There is no identity correction/path
selector fallback in production code.

## Probabilities and verification

This proposer samples each draft position by argmax after correction and path
selection. The actual proposal distribution q is therefore a point mass at the
drafted token, **not** `softmax(draft_logits)`. vLLM 0.20.2 represents this
deterministic proposal by `draft_probs=None`.

The runner still supplies target logits and target sampling metadata to the
rejection sampler, which owns p and acceptance/recovery. A deterministic draft
does not require the target sampler to be greedy. `None` also avoids retaining a
batch-ordered q tensor across request reordering, preemption or partial verification.
Non-None draft probabilities currently raise an error; stochastic drafting needs
both a real sampler and request/position-aligned q storage before it can be enabled.

## Opt-in and validation

Only an external model implementing the contract can use this opt-in. This is
**not a runnable model-download example**. In its draft `config.json`, add:

```json
{
  "dflash_config": {
    "ascend_proposer": "hy",
    "mask_token_id": 123
  }
}
```

The token ID above is illustrative; use the trained mask token ID. Configure
`method="dflash"`, `enforce_eager=True` in the speculative configuration, and
`draft_tensor_parallel_size=1`. vLLM enables parallel drafting for DFlash: here it
means a single block forward, **not** independent sampling or noncausal attention.
The adapter's actual architecture must be registered in vLLM; its DFlash/EAGLE
configuration wrapping must resolve to that architecture. This patch intentionally
does not register a fictitious checkpoint class.

Portable contract tests:

```bash
python -m pytest --noconftest -q tests/ut/spec_decode/test_hy_dflash_proposer.py
```

These run real proposer Python/tensor logic with CPU reference packing, toy model
hooks and isolated hardware imports. Shared `_propose` and runner `_sample` methods
are loaded from source for dispatch tests. They do not run Triton, real model
loading, the vLLM scheduler, the NPU runtime, MLA/DSA kernels, or ACL graph replay.

Before enabling this for serving, add a real model/weight adapter; NPU tests for
context/query cache equivalence and causal sparse visibility; greedy and random
target rejection tests; prefill/chunking, request reordering and preemption tests;
and graph/distributed regression tests with realistic batches. No acceptance-rate
or performance claim is made by this prototype.
