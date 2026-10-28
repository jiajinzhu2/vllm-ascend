# DSpark backport for vLLM Ascend 0.20

This draft ports the **eager DSpark proposer** to the
`releases/v0.20.2rc` runner interface. It also preserves the
DeepSeek V4.1 draft network as a model-specific reference. Complete serving
still needs a compatible target/draft model adapter and NPU validation.

## Implemented

- `vllm_ascend/spec_decode/dspark_proposer.py`: active runtime implementation.
  Project target auxiliary states, populate context KV, prepare an anchor/noise
  query block, run the draft backbone once, then sample sequentially through
  the lightweight Markov head.
- `dspark_utils.py`: device tensor input preparation, rejection rollback,
  cache-slot masking and greedy Markov sampling with draft-to-target id mapping.
- Proposer dispatch and MRV1 hooks: select a resolved draft architecture whose
  name contains DSpark, capture configured target auxiliary layers, supply each
  cache group's block table/slots and pass per-group kernel block sizes.
- Preserve model-declared cache-layer order when passing context slots.
  Refresh both 0.20 CPU length mirrors after rejection; keep target metadata
  unchanged. Mask padding and slots past the model's position limit.
- Ascend-local Markov/confidence heads compatible with vLLM 0.20 linear layers,
  plus the DeepSeek checkpoint-name conversion helper.

The proposer uses its own eager first-pass path rather than the release's
autoregressive MTP loop. The target's graph/parallel configuration is not
mutated. Ordinary MTP architectures retain their existing proposer dispatch.

## Integration contract

vLLM 0.20 has no DSpark config method. The model adapter must resolve to a
registered DSpark draft architecture while using an existing `mtp` or
`dflash` config method. This PR does not register a new target/draft network
or claim that a new checkpoint can be launched without its model adaptation.

The resolved draft config must provide:

- Its trained `dspark_noise_token_id` (or `parallel_drafting_token_id`).
- `dspark_target_layer_ids` (or `target_layer_ids`), interpreted as
  backbone layer ids. The target capture interface uses each id plus one.
  An explicit `eagle_aux_hidden_state_layer_ids` uses capture ids directly.
- `sample_from_anchor`, default true: K query tokens when true, K+1 when false.

The draft adapter supplies these methods:

```text
combine_hidden_states
precompute_and_store_context_kv
get_draft_kv_cache_layer_names
compute_draft_logits
markov_embed
markov_bias
map_draft_to_target
```

Logits must cover the complete draft vocabulary on each TP rank. The declared
cache backend must support non-causal attention and honor the block's metadata.
The target must implement vLLM's auxiliary-hidden-state capture interface.

Current execution constraints: greedy draft sampling, padded drafter batches,
one-dimensional positions, PCP=DCP=PP=1 and equal target/draft TP sizes.
Graph, CP, probabilistic draft sampling and dynamic confidence scheduling
are outside this implementation. Multi-rank execution remains unvalidated.

The eager metadata path performs one packed batch transfer to the host for
exact lengths/positions needed by the old builders. Its performance needs
measurement before treating this as a performance-ready implementation.

## Model-specific reuse boundary

Only the DeepSeek draft network remains under `reference/dspark.py`;
the proposer is now in the runtime package. The reference network is not
registered and has not been adapted to the release's model/attention APIs.

Head computations can be reused only with matching trained parameters.
`dspark_weights.py` is DeepSeek-specific; another checkpoint needs its own
loader, hidden-state projection and attention/cache adaptation. Native MTP
support alone does not establish DSpark compatibility.

The complete V4.1 target, vision, Engram, CP, compressor/indexer and native
operator source imports are excluded from this PR.

## Sources

- Base: `jiajinzhu2/vllm-ascend:releases/v0.20.2rc`,
  `91e6b539d2b4de64eefa0745ec7853e0e5e8a317`.
- DSpark reference and proposer: upstream
  [#16925](https://github.com/vllm-project/vllm-ascend/pull/16925),
  `0deca318102a07f9d8e027047652a2a0baab4fbc`.
- Head mathematics: upstream vLLM
  `84030bbe3d74d99bad477a3d2e37a973ccd8865c`,
  `vllm/model_executor/models/qwen3_dspark.py`.

`sources.json` pins the source revisions and distinguishes the adapted
runtime proposer from the retained model reference.

## Validation

In an isolated vLLM `0.20.2+cpu` / PyTorch `2.11.0+cpu` container:

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 VLLM_PLUGINS=__none__ \
GLOO_SOCKET_IFNAME=lo python3 -m pytest --noconftest -q -o addopts='' \
  tests/ut/spec_decode/test_dspark_backport.py \
  tests/ut/spec_decode/test_dspark_backport_npu.py \
  tests/ut/models/test_deepseek_v41_backport_primitives.py \
  tests/ut/models/test_deepseek_v41_backport_runtime.py
```

The CPU suite contains 61 cases: 25 algorithm/proposer/dispatch checks, 30
head/checkpoint primitives and six real vLLM linear-layer/Gloo checks.
The proposer tests use real vLLM 0.20 metadata and AttentionGroup classes;
NPU context and base initialization are substituted and the draft is a toy
adapter. They verify one backbone call, Markov conditioning, cache-group
order, rejection, padded target rows and host length mirrors. This is not
full model-runner or real-weight inference validation.

The five tests in `test_dspark_backport_npu.py` also passed on an Ascend
950DT with PyTorch/torch_npu 2.10. They compare input preparation to CPU for
both anchor layouts and rejection cases, and check a mapped Markov chain.
These are small tensor tests, not native-attention/model inference tests.

Python syntax and whitespace checks passed. The required
`bash format.sh ci` cannot run because `pre-commit` is unavailable.
Complete model loading, target/draft attention correctness, multi-rank
execution, acceptance rates and performance remain unvalidated.
