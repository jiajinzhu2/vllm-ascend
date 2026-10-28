# DeepSeek V4.1 DSpark backport — work in progress

This draft stages the DeepSeek V4.1 DSpark sources on the
`releases/v0.20.2rc` code line. **It is not a runnable or merge-ready
DeepSeek V4.1 enablement.** Runner, cache and native-build integration are
still required. Do not deploy this branch as a completed DSpark backport.

## Baseline and source revisions

The target fork branch is `jiajinzhu2/vllm-ascend:releases/v0.20.2rc`,
at `91e6b539d2b4de64eefa0745ec7853e0e5e8a317`.
The official branch was inspected at
`792ea0c6de317f364d542aff0e758af383ce9aec`; the ten additional official
commits change documentation and dependency pins. The backport PR uses the
fork branch directly so those unrelated changes are excluded.

Sources are pinned to:

- Framework and DSpark: upstream PR #16925,
  `0deca318102a07f9d8e027047652a2a0baab4fbc`.
- SparseFlashMla and QLI V2 operators: upstream PR #16422,
  `bd69bad88fc19e1aeeea585416d408df8bda8fef`.
- Head mathematics: upstream vLLM
  `84030bbe3d74d99bad477a3d2e37a973ccd8865c`,
  `vllm/model_executor/models/qwen3_dspark.py`.

The imported files retain their upstream license and attribution headers.
`sources.json` records the original file paths and revisions.

## Implemented release adaptations

- Move the Markov and confidence heads into Ascend-owned code. vLLM 0.20
  has no `qwen3_dspark` module and its `ParallelLMHead` does not accept
  `disable_tp=True`. The backported Markov projection uses
  `ReplicatedLinear`, preserving the replicated weight layout and checkpoint
  parameter names. It preserves optional logits soft capping and scaling.
- Extract checkpoint namespace conversion into a runtime-independent helper.
  It remaps the MTP stages, first/last-layer projections, Markov parameters,
  confidence parameters and MoE names. Invalid stage ranges fail explicitly.
- Add 30 CPU primitive tests and a source-compatibility diagnostic.
- Stage the target/draft, attention, cache and operator source dependencies.

The model registry and existing release runner are not switched to these
staged implementations. The new native operator directories are not yet
wired into the release build. This keeps the unfinished integration out of
the existing serving path.

## Remaining integration gates

1. Backport the V4.1 configuration and draft-selection contract. vLLM 0.20
   lacks the newer V4.1 configuration/model modules and DSpark method APIs.
2. Adapt circular compressor state, MLA compression and cache grouping to
   the release KV specifications/managers. The staged modern cache registry
   cannot be imported on 0.20 as-is.
3. Adapt MoE construction, sequence parallel helpers, RoPE metadata and
   device metadata execution to the old interfaces.
4. Wire DSpark proposer selection, target auxiliary hidden states,
   per-group block tables, rejection lengths and draft sampling into MRV1.
   The staged proposer still depends on newer DFlash and input-kernel APIs.
5. Integrate the four native operators' build lists, shared headers, Torch
   bindings and Meta implementations without replacing existing release
   operator contracts.
6. Establish eager correctness before enabling graph replay or CP. Run
   real-weight loading, multi-request rejection/turnover, accuracy and
   acceptance regressions. Probabilistic and graph support have not been
   established for this backport.

The original framework tests are preserved under `reference_tests/`
because they depend on newer framework interfaces. They were not run or
counted as passing release tests.

## Validation performed

A dedicated CPU container was created on the user-specified development
host. Its vLLM source is the unmodified `v0.20.2` tag
(`bc150f5029`), with PyTorch 2.10.

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 -m pytest \
  --noconftest -q -o addopts='' \
  tests/ut/models/test_deepseek_v41_backport_primitives.py
```

Result: **30 passed**. These are source-isolated primitive tests. A linear
stand-in exercises head mathematics; this does not validate the real vLLM
linear layer, the model import chain, CANN operators or inference.

Source review:

```bash
python3 tools/check_deepseek_v41_backport.py --vllm-root /path/to/vllm
```

The recorded report has 34 review items in 17 files. Some are dynamic
exports requiring inspection; others are confirmed missing old-version
interfaces. The diagnostic intentionally exits nonzero while review items
remain. `source-compatibility.json` contains the exact list.

No native build, complete framework test pass, NPU inference or performance
result is claimed. All 16 NPUs on the supplied host were occupied by existing
VLLM workers during inspection, so no model was launched. Full format/lint
checks have not run; Ruff is unavailable in the isolated container and the
configured package mirrors timed out.
