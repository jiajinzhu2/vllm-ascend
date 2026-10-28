# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of DSpark algorithms and the production eager proposer.

NPU imports, forward contexts and base initialization are substituted.
The shared proposer methods, metadata and AttentionGroup are real code.
"""

import importlib.util
import sys
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from vllm.config import CUDAGraphMode
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def modules(monkeypatch):
    def install(name, **attrs):
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    package = install("vllm_ascend.spec_decode")
    package.__path__ = [str(ROOT / "vllm_ascend/spec_decode")]

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, ROOT / path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    utils = load(
        "vllm_ascend.spec_decode.dspark_utils",
        "vllm_ascend/spec_decode/dspark_utils.py",
    )
    contexts = []

    @contextmanager
    def forward_context(metadata, config, **kwargs):
        contexts.append((metadata, kwargs))
        yield

    install(
        "vllm_ascend.ascend_forward_context",
        set_ascend_forward_context=forward_context,
        _EXTRA_CTX=SimpleNamespace(capturing=False),
    )
    install("vllm_ascend.ascend_config", get_ascend_config=lambda: SimpleNamespace(enable_reduce_sample=False))
    install("vllm_ascend.attention.attention_mask", AttentionMaskBuilder=object)
    install(
        "vllm_ascend.attention.attention_v1",
        AscendAttentionState=SimpleNamespace(ChunkedPrefill="chunked"),
    )
    install("vllm_ascend.attention.utils", AscendCommonAttentionMetadata=CommonAttentionMetadata)
    install(
        "vllm_ascend.compilation.acl_graph",
        ACLGraphWrapper=type("ACLGraphWrapper", (), {}),
        update_full_graph_params=None,
    )
    install("vllm_ascend.distributed.parallel_state", get_lmhead_tp_group=None)
    install("vllm_ascend.ops.triton.spec_decode.utils", prepare_inputs_padded_kernel=None)
    install("vllm_ascend.ops.triton.triton_utils", get_vectorcore_num=None)
    install(
        "vllm_ascend.utils",
        enable_sp=lambda *args: False,
        lmhead_tp_enable=lambda: False,
        shared_expert_dp_enabled=lambda: False,
    )
    base = load(
        "vllm_ascend.spec_decode.llm_base_proposer",
        "vllm_ascend/spec_decode/llm_base_proposer.py",
    )
    monkeypatch.setattr(base, "get_forward_context", lambda: SimpleNamespace())
    monkeypatch.setattr(base, "get_pp_group", lambda: SimpleNamespace(world_size=1))

    class Base(base.AscendSpecDecodeBaseProposer):
        def __init__(self, config, device, runner=None):
            self.vllm_config = config
            self.speculative_config = config.speculative_config
            self.draft_model_config = config.speculative_config.draft_model_config
            self.device = device
            self.runner = runner
            self.dtype = torch.float32
            self.num_speculative_tokens = config.speculative_config.num_speculative_tokens
            self.max_batch_size = config.scheduler_config.max_num_seqs
            self.max_num_tokens = config.scheduler_config.max_num_batched_tokens
            self.max_model_len = config.model_config.max_model_len
            self.uses_mrope = False
            self.uses_xdrope_dim = 0
            self.draft_uses_xdrope_dim = 0
            self.tp_group_context = nullcontext()
            self._runnable = self._run_merged_draft

        def load_model(self, target):
            self.model = target.draft
            self._draft_attn_layer_names = set(self.model.get_draft_kv_cache_layer_names())

    install("vllm_ascend.spec_decode.dflash_proposer", AscendDflashProposer=Base)
    proposer = load(
        "vllm_ascend.spec_decode.dspark_proposer",
        "vllm_ascend/spec_decode/dspark_proposer.py",
    )
    monkeypatch.setattr(proposer, "get_forward_context", lambda: SimpleNamespace())
    return SimpleNamespace(utils=utils, proposer=proposer, base=base, contexts=contexts, install=install, load=load)


def config(sample_from_anchor=True):
    hf = SimpleNamespace(
        architectures=["TestDSparkForCausalLM"],
        dspark_noise_token_id=7,
        dspark_target_layer_ids=[1, 3],
        sample_from_anchor=sample_from_anchor,
        vocab_size=32,
    )
    draft = SimpleNamespace(hf_config=hf, hf_text_config=hf, get_hidden_size=lambda: 2, enforce_eager=False)
    return SimpleNamespace(
        speculative_config=SimpleNamespace(
            method="mtp",
            draft_model_config=draft,
            num_speculative_tokens=3,
            disable_padded_drafter_batch=False,
            parallel_drafting=True,
            enforce_eager=False,
            draft_tensor_parallel_size=1,
        ),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            pipeline_parallel_size=1,
            tensor_parallel_size=1,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=2, max_num_batched_tokens=16),
        model_config=SimpleNamespace(max_model_len=128),
        compilation_config=SimpleNamespace(mode="target_mode", cudagraph_mode="target_graph"),
    )


@pytest.mark.parametrize("anchor", [True, False])
@pytest.mark.parametrize("rejected", [None, [1, 0]])
def test_block_inputs_rejection_and_paging(modules, anchor, rejected):
    reject = None if rejected is None else torch.tensor(rejected)
    output = modules.utils.prepare_dspark_inputs(
        torch.tensor([0, 3, 5]),
        torch.tensor([14, 15, 16, 30, 31]),
        torch.tensor([17, 32]),
        torch.tensor([4, 5]),
        torch.tensor([[9, 2, 8, 6], [7, 1, 4, 5]])[:, ::1],
        torch.tensor([100, 101, 102, 200, 201]),
        block_size=16,
        num_speculative_tokens=3,
        noise_token_id=7,
        max_model_len=128,
        num_rejected_tokens=reject,
        sample_from_anchor=anchor,
    )
    q = 3 if anchor else 4
    first = 17 if rejected is None else 16
    expected_positions = list(range(first, first + q)) + list(range(32, 32 + q))
    assert output.positions.tolist() == expected_positions
    assert output.input_ids.reshape(2, q).tolist() == [[4] + [7] * (q - 1), [5] + [7] * (q - 1)]
    expected_slots = [
        [9, 2, 8, 6][p // 16] * 16 + p % 16 for p in expected_positions[:q]
    ] + [[7, 1, 4, 5][p // 16] * 16 + p % 16 for p in expected_positions[q:]]
    assert output.query_slots.tolist() == expected_slots
    assert output.context_slots.tolist() == [100, 101, 102 if rejected is None else -1, 200, 201]
    offset = 0 if anchor else 1
    assert output.sample_indices.tolist() == [
        row * q + step + offset for row in range(2) for step in range(3)
    ]
    assert output.seq_lens.tolist() == [first + q, 32 + q]


def test_noncontiguous_table_and_end_of_context(modules):
    table = torch.tensor([[3, 99, 8, 99, -1, 99]])[:, ::2]
    output = modules.utils.prepare_dspark_inputs(
        torch.tensor([0, 1]), torch.tensor([30]), torch.tensor([31]),
        torch.tensor([4]), table, torch.tensor([142]),
        block_size=16, num_speculative_tokens=3, noise_token_id=7, max_model_len=33,
    )
    assert output.query_slots.tolist() == [143, -1, -1]
    # A missing physical page must not change another group's logical position.
    assert output.positions.tolist() == [31, 32, 0]
    assert output.seq_lens.tolist() == [33]


def test_fully_rejected_context_and_discarded_seed(modules):
    output = modules.utils.prepare_dspark_inputs(
        torch.tensor([0, 3]), torch.tensor([8, 9, 10]), torch.tensor([11]),
        torch.tensor([-1]), torch.tensor([[4, 5]]), torch.tensor([72, 73, 74]),
        block_size=16, num_speculative_tokens=2, noise_token_id=7, max_model_len=32,
        num_rejected_tokens=torch.tensor([3]),
    )
    assert output.context_slots.tolist() == [-1, -1, -1]
    assert output.input_ids.tolist() == [7, 7]
    assert output.query_slots.tolist() == [-1, -1]


class ToyDraft:
    def __init__(self, reduced=False):
        self.reduced = reduced
        self.forward_calls = 0
        self.logit_calls = 0
        self.previous_ids = []
        self.context = None

    def get_draft_kv_cache_layer_names(self):
        # Intentionally not lexical order.
        return ["draft.2", "draft.10"]

    def combine_hidden_states(self, states):
        return states[:, :2] + 1

    def precompute_and_store_context_kv(self, states, positions, slots):
        self.context = (states.clone(), positions.clone(), [x.clone() for x in slots])

    def __call__(self, input_ids, positions, inputs_embeds=None):
        self.forward_calls += 1
        self.query = (input_ids.clone(), positions.clone())
        return positions[:, None].expand(-1, 2).float()

    def compute_draft_logits(self, hidden):
        self.logit_calls += 1
        return torch.zeros(hidden.shape[0], 3)

    def markov_embed(self, ids):
        self.previous_ids.append(ids.clone())
        return ids // 10 if self.reduced else ids

    def markov_bias(self, embed):
        return torch.nn.functional.one_hot((embed + 1) % 3, 3).float() * 10

    def map_draft_to_target(self, ids):
        return ids * 10 if self.reduced else ids


@pytest.mark.parametrize("reduced", [False, True])
def test_markov_sampling_conditions_on_mapped_previous_token(modules, reduced):
    model = ToyDraft(reduced=reduced)
    seeds = torch.tensor([0, 1]) * (10 if reduced else 1)
    result = modules.utils.sample_dspark_greedy(model, torch.zeros(6, 2), seeds, 3)
    expected = torch.tensor([[1, 2, 0], [2, 0, 1]]) * (10 if reduced else 1)
    assert torch.equal(result, expected)
    assert model.logit_calls == 1
    assert torch.equal(model.previous_ids[0], seeds)
    assert torch.equal(model.previous_ids[1], result[:, 0])
    assert torch.equal(model.previous_ids[2], result[:, 1])


def test_sampling_rejects_incompatible_head(modules):
    model = ToyDraft()
    model.markov_bias = lambda _: torch.zeros(1, 4)
    with pytest.raises(ValueError, match="same vocabulary"):
        modules.utils.sample_dspark_greedy(model, torch.zeros(3, 2), torch.tensor([0]), 3)
    with pytest.raises(ValueError, match="hidden-state count"):
        modules.utils.sample_dspark_greedy(model, torch.zeros(2, 2), torch.tensor([0]), 3)


def test_empty_sampling(modules):
    model = ToyDraft()
    result = modules.utils.sample_dspark_greedy(model, torch.empty(0, 2), torch.empty(0, dtype=torch.long), 3)
    assert result.shape == (0, 3)
    assert model.logit_calls == 0


def test_draft_detection_and_auxiliary_layer_conventions(modules):
    cfg = config()
    assert modules.utils.is_dspark_draft(cfg)
    assert modules.utils.get_dspark_aux_layers(cfg) == (2, 4)
    cfg.speculative_config.draft_model_config.hf_config.eagle_aux_hidden_state_layer_ids = [5, 8]
    assert modules.utils.get_dspark_aux_layers(cfg) == (5, 8)
    cfg.speculative_config.draft_model_config.hf_config.architectures = ["OrdinaryMTP"]
    assert not modules.utils.is_dspark_draft(cfg)


def make_proposer(modules, monkeypatch, anchor=True, pad=0):
    cfg = config(anchor)
    builders = []

    class Builder:
        def __init__(self, spec, names, config, device):
            self.kv_cache_spec = spec
            self.names = names
            self.calls = []
            builders.append(self)

        def build(self, common_prefix_len, metadata, target_model):
            self.calls.append(metadata)
            return SimpleNamespace(common=metadata)

    class Backend:
        @staticmethod
        def full_cls_name():
            return "test.DSparkBackend"

        @staticmethod
        def get_builder_cls():
            return Builder

        @staticmethod
        def supports_non_causal():
            return True

    layers = {name: SimpleNamespace(get_attn_backend=lambda: Backend) for name in ("draft.2", "draft.10")}
    monkeypatch.setattr(modules.proposer, "get_layers_from_vllm_config", lambda *_: layers)
    runner = SimpleNamespace(
        _sync_metadata_across_dp=lambda n, **kw: (n + pad, None, None),
        get_model=lambda: "target",
    )
    proposer = modules.proposer.AscendDSparkProposer(cfg, torch.device("cpu"), runner=runner)
    proposer.model = ToyDraft()
    proposer.attn_layer_names = proposer.model.get_draft_kv_cache_layer_names()
    proposer._draft_attn_layer_names = set(proposer.attn_layer_names)
    spec = FullAttentionSpec(block_size=64, num_kv_heads=1, head_size=8, dtype=torch.float16)
    caches = KVCacheConfig(
        num_blocks=16,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["draft.10"], kv_cache_spec=spec),
            KVCacheGroupSpec(layer_names=["draft.2"], kv_cache_spec=spec),
        ],
    )
    proposer.initialize_attn_backend(caches, [16, 32])
    return cfg, proposer, builders


@pytest.mark.parametrize("anchor", [True, False])
@pytest.mark.parametrize("writable_mirrors", [False, True])
def test_actual_proposer_one_backbone_pass_and_per_group_metadata(modules, monkeypatch, anchor, writable_mirrors):
    cfg, proposer, builders = make_proposer(modules, monkeypatch, anchor=anchor)
    assert type(proposer)._propose is modules.base.AscendSpecDecodeBaseProposer._propose
    assert type(proposer)._run_merged_draft is modules.base.AscendSpecDecodeBaseProposer._run_merged_draft
    merged_calls = []
    run_merged = proposer._runnable

    def record_merged(**kwargs):
        merged_calls.append(kwargs)
        return run_merged(**kwargs)

    proposer._runnable = record_merged
    # Constructor must not mutate the target's configuration.
    assert cfg.speculative_config.parallel_drafting
    assert not cfg.speculative_config.enforce_eager
    assert not proposer.use_cuda_graph
    assert not cfg.speculative_config.draft_model_config.enforce_eager
    assert cfg.compilation_config.mode == "target_mode"
    assert cfg.compilation_config.cudagraph_mode == "target_graph"
    assert [b.kv_cache_spec.block_size for b in builders] == [16, 32]
    table0 = torch.tensor([[9, 2, 8, 6], [7, 1, 4, 5]], dtype=torch.int32)
    table1 = torch.tensor([[5, 3, 9, 1], [6, 8, 2, 1]], dtype=torch.int32)
    proposer.set_per_group_attn_metadata(0, table0, torch.tensor([100, 101, 102, 200, 201]))
    proposer.set_per_group_attn_metadata(1, table1, torch.tensor([300, 301, 302, 400, 401]))
    @dataclass
    class WritableMirrors(CommonAttentionMetadata):
        seq_lens_cpu: torch.Tensor | None = None
        num_computed_tokens_cpu: torch.Tensor | None = None

    metadata_type = WritableMirrors if writable_mirrors else CommonAttentionMetadata
    common = metadata_type(
        query_start_loc=torch.tensor([0, 3, 5], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 3, 5], dtype=torch.int32),
        seq_lens=torch.tensor([17, 32], dtype=torch.int32),
        num_reqs=2,
        num_actual_tokens=5,
        max_query_len=3,
        max_seq_len=32,
        block_table_tensor=table0,
        slot_mapping=torch.tensor([100, 101, 102, 200, 201]),
        causal=True,
    )
    result = proposer._propose(
        target_token_ids=torch.zeros(5, dtype=torch.long),
        target_positions=torch.tensor([14, 15, 16, 30, 31]),
        target_hidden_states=torch.zeros(5, 4),
        next_token_ids=torch.tensor([0, 1]),
        token_indices_to_sample=None,
        common_attn_metadata=common,
        num_rejected_tokens_gpu=torch.tensor([1, 0]),
    )
    assert result.tolist() == [[1, 2, 0], [2, 0, 1]]
    assert proposer.model.forward_calls == 1
    assert proposer.model.logit_calls == 1
    assert len(merged_calls) == 1
    assert merged_calls[0]["batch_size"] == 2
    assert merged_calls[0]["num_tokens"] == 2 * proposer.num_query_per_req
    states, positions, slots = proposer.model.context
    assert torch.equal(states, torch.ones(5, 2))
    # Context slots follow model order, not sorted names or cache-group order.
    assert slots[0].tolist() == [300, 301, -1, 400, 401]
    assert slots[1].tolist() == [100, 101, -1, 200, 201]
    queries = 3 if anchor else 4
    for builder in builders:
        metadata = builder.calls[-1]
        assert metadata.seq_lens.tolist() == [16 + queries, 32 + queries]
        assert torch.equal(metadata.seq_lens_cpu, metadata.seq_lens)
        assert torch.equal(metadata._seq_lens_cpu, metadata.seq_lens)
        assert not metadata.causal
        assert metadata._num_computed_tokens_cpu.tolist() == [16, 32]
        if writable_mirrors:
            assert metadata.num_computed_tokens_cpu.tolist() == [16, 32]
    assert builders[0].calls[-1].slot_mapping[0].item() == 2 * 16
    assert builders[1].calls[-1].slot_mapping[0].item() == 5 * 32 + 16
    assert common.causal
    assert common.seq_lens.tolist() == [17, 32]
    assert proposer._per_group_block_tables == {}


def test_proposer_missing_group_fails_before_forward(modules, monkeypatch):
    _, proposer, _ = make_proposer(modules, monkeypatch)
    with pytest.raises(RuntimeError, match="did not supply"):
        proposer._propose(
            torch.zeros(1), torch.zeros(1), torch.zeros(1, 4), torch.tensor([0]),
            None, SimpleNamespace(num_reqs=1),
        )
    assert proposer.model.forward_calls == 0


@pytest.mark.parametrize("field,value", [
    ("disable_padded_drafter_batch", True),
    ("draft_sample_method", "probabilistic"),
    ("draft_tensor_parallel_size", 2),
])
def test_unsupported_execution_contracts_fail_early(modules, field, value):
    cfg = config()
    setattr(cfg.speculative_config, field, value)
    with pytest.raises(ValueError):
        modules.proposer.AscendDSparkProposer(cfg, torch.device("cpu"))


def test_dispatch_does_not_route_ordinary_mtp_to_dspark(modules):
    classes = {
        "draft_proposer": "AscendDraftModelProposer",
        "eagle_proposer": "AscendEagleProposer",
        "extract_hidden_states_proposer": "AscendExtractHiddenStatesProposer",
        "medusa_proposer": "AscendMedusaProposer",
        "ngram_proposer": "AscendNgramProposer",
        "ngram_proposer_npu": "AscendNgramProposerNPU",
        "suffix_proposer": "AscendSuffixDecodingProposer",
    }
    for module, name in classes.items():
        modules.install(f"vllm_ascend.spec_decode.{module}", **{name: lambda *args: "ordinary"})
    factory = modules.load("dspark_factory_under_test", "vllm_ascend/spec_decode/__init__.py")
    cfg = config()
    factory.AscendDSparkProposer = lambda *args: "dspark"
    assert factory.get_spec_decode_method("mtp", cfg, None, None) == "dspark"
    cfg.speculative_config.draft_model_config.hf_config.architectures = ["OrdinaryMTP"]
    assert factory.get_spec_decode_method("mtp", cfg, None, None) == "ordinary"


def test_dp_padding_does_not_create_live_cache_slots(modules, monkeypatch):
    _, proposer, builders = make_proposer(modules, monkeypatch, pad=3)
    proposer.set_per_group_attn_metadata(0, torch.tensor([[9, 2, 8, 6]]), torch.tensor([99]))
    proposer.set_per_group_attn_metadata(1, torch.tensor([[5, 3, 9, 1]]), torch.tensor([199]))
    common = CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 1, 1]),
        query_start_loc_cpu=torch.tensor([0, 1, 1]),
        seq_lens=torch.tensor([8, 0]),
        num_reqs=2,
        is_prefilling=torch.tensor([False, True]),
        num_actual_tokens=1,
        max_query_len=1,
        max_seq_len=8,
        block_table_tensor=torch.tensor([[9, 2, 8, 6]]),
        slot_mapping=torch.tensor([99]),
    )
    result = proposer._propose(
        torch.tensor([0]), torch.tensor([7]), torch.zeros(1, 4),
        torch.tensor([0]), None, common,
    )
    assert result.tolist() == [[1, 2, 0]]
    assert proposer.model.query[0].tolist() == [0, 7, 7, 7, 7, 7]
    assert proposer.model.query[1].tolist() == [8, 9, 10, 0, 0, 0]
    for builder in builders:
        assert builder.calls[-1].slot_mapping.tolist()[3:] == [-1, -1, -1]
        assert builder.calls[-1].num_reqs == 1
        assert builder.calls[-1].is_prefilling.tolist() == [False]
        assert builder.calls[-1].num_input_tokens == 6
        assert builder.calls[-1].num_actual_tokens == 3


def test_causal_only_backend_is_rejected(modules, monkeypatch):
    _, proposer, _ = make_proposer(modules, monkeypatch)
    backend = SimpleNamespace(supports_non_causal=lambda: False)
    layers = {
        name: SimpleNamespace(get_attn_backend=lambda: backend)
        for name in proposer.attn_layer_names
    }
    monkeypatch.setattr(modules.proposer, "get_layers_from_vllm_config", lambda *_: layers)
    spec = FullAttentionSpec(block_size=64, num_kv_heads=1, head_size=8, dtype=torch.float16)
    caches = KVCacheConfig(
        num_blocks=16, kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=proposer.attn_layer_names, kv_cache_spec=spec)],
    )
    with pytest.raises(ValueError, match="non-causal"):
        proposer.initialize_attn_backend(caches, [16])


@pytest.mark.parametrize("missing", ["dspark_noise_token_id", "dspark_target_layer_ids"])
def test_required_trained_draft_contract_is_not_guessed(modules, missing):
    cfg = config()
    delattr(cfg.speculative_config.draft_model_config.hf_config, missing)
    with pytest.raises(ValueError):
        modules.proposer.AscendDSparkProposer(cfg, torch.device("cpu"))


def test_int32_seed_ids_return_int64_target_ids(modules):
    result = modules.utils.sample_dspark_greedy(
        ToyDraft(), torch.zeros(3, 2), torch.tensor([0], dtype=torch.int32), 3
    )
    assert result.dtype == torch.int64
    assert result.tolist() == [[1, 2, 0]]


@pytest.mark.parametrize("ownership", [None, True, False])
@pytest.mark.parametrize("multimodal_target", [False, True])
def test_base_dspark_weight_sharing_requires_explicit_opt_in(modules, ownership, multimodal_target):
    base = modules.base.AscendSpecDecodeBaseProposer
    proposer = base.__new__(base)
    proposer.method = "dspark"
    proposer.use_compress = False
    proposer.use_cuda_graph = False
    proposer.vllm_config = SimpleNamespace(compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE))
    own_embed, own_head, own_indices = object(), object(), object()
    proposer.model = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=own_embed, topk_indices_buffer=own_indices),
        lm_head=own_head,
    )
    if ownership is not None:
        proposer.model.has_own_embed_tokens = ownership
        proposer.model.has_own_lm_head = ownership
    target = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=object(), topk_indices_buffer=object()),
        lm_head=object(),
    )
    outer = SimpleNamespace(get_language_model=lambda: target) if multimodal_target else target
    proposer._maybe_share_embeddings(target)
    proposer._maybe_share_lm_head(outer)
    proposer._maybe_share_topk_indices(target)
    assert proposer.model.model.embed_tokens is (target.model.embed_tokens if ownership is False else own_embed)
    assert proposer.model.lm_head is (target.lm_head if ownership is False else own_head)
    assert proposer.model.model.topk_indices_buffer is own_indices


def test_base_dspark_missing_shared_head_fails(modules):
    base = modules.base.AscendSpecDecodeBaseProposer
    proposer = base.__new__(base)
    proposer.method = "dspark"
    proposer.model = SimpleNamespace(has_own_lm_head=False)
    with pytest.raises(AttributeError, match="shared LM head"):
        proposer._maybe_share_lm_head(SimpleNamespace())


@pytest.mark.parametrize("method", ["mtp", "dflash", "eagle3"])
def test_shared_merged_draft_keeps_existing_sampling_paths(modules, method):
    base = modules.base.AscendSpecDecodeBaseProposer
    proposer = base.__new__(base)
    proposer.method = method
    proposer.num_speculative_tokens = 1
    proposer.parallel_drafting = False
    proposer.pass_hidden_states_to_model = False
    proposer.uses_mrope = False
    proposer.uses_xdrope_dim = 0
    proposer.draft_uses_xdrope_dim = 0
    proposer.pcp_size = 1
    proposer.input_ids = torch.tensor([4, 5])
    proposer.positions = torch.tensor([10, 11])
    proposer.maybe_all_gather_and_unpad = lambda hidden, positions, last: (hidden, positions, last)
    proposer.build_model_inputs_first_pass = lambda n: dict(
        input_ids=proposer.input_ids[:n], positions=proposer.positions[:n], inputs_embeds=None
    )

    class OrdinaryDraft:
        forward_calls = 0

        def __call__(self, **kwargs):
            self.forward_calls += 1
            assert kwargs["input_ids"].tolist() == [4, 5]
            hidden = torch.zeros(2, 2)
            return (hidden, hidden) if method == "eagle3" else hidden

        def compute_logits(self, hidden):
            return torch.tensor([[0.0, 5.0, 0.0], [0.0, 0.0, 5.0]])

        def compute_draft_logits(self, hidden):
            raise AssertionError("ordinary drafts must not enter DSpark sampling")

    proposer.model = OrdinaryDraft()
    result = proposer._run_merged_draft(
        num_input_tokens=2,
        batch_size=2,
        token_indices_to_sample=torch.tensor([0, 1]),
        target_positions=proposer.positions,
        inputs_embeds=None,
        multi_steps_attn_metadata=[],
        num_tokens=2,
    )
    assert result.tolist() == [[1], [2]]
    assert proposer.model.forward_calls == 1


@pytest.mark.parametrize("failure", ["projection", "sampling", "dp_capacity"])
def test_base_dspark_clears_step_state_on_failure(modules, monkeypatch, failure):
    _, proposer, _ = make_proposer(modules, monkeypatch, pad=8 if failure == "dp_capacity" else 0)
    table = torch.tensor([[9, 2, 8, 6]])
    for gid in (0, 1):
        proposer.set_per_group_attn_metadata(gid, table, torch.tensor([100]))
    common = CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 1]),
        query_start_loc_cpu=torch.tensor([0, 1]),
        seq_lens=torch.tensor([8]),
        num_reqs=1,
        num_actual_tokens=1,
        max_query_len=1,
        max_seq_len=8,
        block_table_tensor=table,
        slot_mapping=torch.tensor([100]),
    )
    if failure == "projection":
        proposer.model.combine_hidden_states = lambda states: states
        message = "projection"
    elif failure == "sampling":
        proposer.model.compute_draft_logits = lambda hidden: torch.zeros(hidden.shape[0], 4)
        message = "same vocabulary"
    else:
        message = "capacity"
    with pytest.raises(ValueError, match=message):
        proposer._propose(
            torch.tensor([0]), torch.tensor([7]), torch.zeros(1, 4),
            torch.tensor([0]), None, common,
        )
    assert proposer._per_group_block_tables == {}
    assert proposer._per_group_slot_mappings == {}
    assert proposer._dspark_seed_token_ids is None
    assert proposer._dspark_context_hidden_states is None
    assert proposer._dspark_context_slots == {}
    # Retrying requires fresh runner metadata, even after a partial forward.
    with pytest.raises(RuntimeError, match="did not supply"):
        proposer._propose(
            torch.tensor([0]), torch.tensor([7]), torch.zeros(1, 4),
            torch.tensor([0]), None, common,
        )
