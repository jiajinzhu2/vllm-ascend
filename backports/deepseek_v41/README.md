# DeepSeek V4.1 DSpark backport components

This draft is limited to **DSpark** on `releases/v0.20.2rc`.
It does not enable complete DeepSeek V4.1 inference.

## Scope

Runtime components adapted to vLLM 0.20:

- `dspark_heads.py`: replicated Markov projection and confidence head.
  It preserves checkpoint parameter names, logits soft capping and scaling
  without depending on the newer upstream `qwen3_dspark` module or
  `ParallelLMHead(disable_tp=True)` API.
- `dspark_weights.py`: checkpoint namespace conversion for the MTP stages,
  first/last-layer projections, Markov/confidence heads and MoE parameters.
- Two unit-test files covering these components.

The DSpark draft model and proposer are preserved under `reference/` for
the remaining integration work. They are **reference source, not registered
runtime implementations**. They still require a V4.1 base implementation and
adaptation to the release runner and attention APIs.

The earlier full-source staging has been removed: the V4.1 target model,
vision, Engram, context-parallel implementation, compressor/indexer, general
KV-cache extensions, four complete native operator directories, their
standalone extensions, examples and unrelated reference tests.

## Reuse boundary for another target model

These files are a DSpark reference for a later model-specific port. They do
not establish compatibility with another model or checkpoint.

- Head computations can be reused only when the new draft has the same trained
  Markov/confidence architecture and corresponding checkpoint parameters.
- `dspark_weights.py` is **DeepSeek-specific**. A different checkpoint needs
  its own namespace conversion and loading checks.
- The draft network, auxiliary-hidden-state selection/projection, attention
  and cache layout must follow the actual draft implementation.
- The proposer provides a reference for block preparation and cache-group
  bookkeeping. Its newer runner/kernel dependencies still require a 0.20 port.
- Native MTP support alone does not establish DSpark compatibility. The
  draft forward path and trained parameters are needed to determine that.

## Sources

- Base: `jiajinzhu2/vllm-ascend:releases/v0.20.2rc`,
  `91e6b539d2b4de64eefa0745ec7853e0e5e8a317`.
- DSpark model/proposer: upstream
  [#16925](https://github.com/vllm-project/vllm-ascend/pull/16925),
  `0deca318102a07f9d8e027047652a2a0baab4fbc`.
- Head mathematics: upstream vLLM
  `84030bbe3d74d99bad477a3d2e37a973ccd8865c`,
  `vllm/model_executor/models/qwen3_dspark.py`.

`sources.json` pins the two retained integration references. Their existing
Ascend-local head imports and checkpoint remapping are the changes made in
the previous staging commits. Source attribution headers are preserved.

## Validation

In an isolated vLLM `0.20.2+cpu` / PyTorch `2.11.0+cpu` container:

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 VLLM_PLUGINS=__none__ \
GLOO_SOCKET_IFNAME=lo python3 -m pytest --noconftest -q -o addopts='' \
  tests/ut/models/test_deepseek_v41_backport_primitives.py \
  tests/ut/models/test_deepseek_v41_backport_runtime.py
```

**Result: 36 passed.** The cases include 30 source-isolated primitive tests and six real vLLM
`ReplicatedLinear` tests with weight-loading postprocessing and a single-rank
CPU/Gloo process group. These establish head-level CPU compatibility only.

Full draft-model loading, proposer/runner integration, native operators and
NPU inference have not passed validation. The required format check was
attempted with `bash format.sh ci`; it cannot run because `pre-commit`
is not installed. This remains a draft. A compatible
V4.1 target-model/attention implementation on the release branch is a
prerequisite for completing the DSpark integration; it is outside this PR's
reduced scope.
