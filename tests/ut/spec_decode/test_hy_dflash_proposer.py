# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""CPU orchestration contracts, NOT HY checkpoint/NPU attention validation.

Run with pytest --noconftest on a machine without vLLM/torch_npu. The complete
DFlash and HY proposer modules are loaded. The Triton packer is replaced with
a CPU reference; infrastructure imports are isolated and restored per test.
The shared _propose and runner _sample methods are compiled from their actual
source AST, without importing the hardware-dependent runner module.
"""

import __future__

import ast
import importlib.util
import sys
from contextlib import contextmanager
from enum import Enum
from functools import partial
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch


def source_methods(path, class_name, names, namespace):
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in methods} == set(names)
    module = ast.Module(body=methods, type_ignores=[])
    exec(compile(module, str(path), "exec", flags=__future__.annotations.compiler_flag), namespace)
    return {name: namespace[name] for name in names}


class GraphMode(Enum):
    NONE = 0
    FULL = 1


class CpuPacker:
    """Reference layout only; this does not execute or test the Triton kernel."""

    def __getitem__(self, grid):
        return self.run

    @staticmethod
    def run(**args):
        table = args["block_table_ptr"]
        starts = args["query_start_loc_ptr"].tolist()
        positions = args["target_positions_ptr"]
        block_size = args["block_size"]
        query_length = args["num_query_per_req"]
        for req in range(args["batch_size"]):
            start, end = starts[req : req + 2]
            for token in range(start, end):
                pos = int(positions[token])
                args["out_context_positions_ptr"][token] = pos
                args["out_context_slot_mapping_ptr"][token] = (
                    table[req, pos // block_size] * block_size + pos % block_size
                )
            rejected = int(args["num_rejected_tokens_ptr"][req]) if args["HAS_NUM_REJECTED"] else 0
            last_position = int(positions[end - rejected - 1])
            for step in range(query_length):
                token = req * query_length + step
                pos = last_position + step + 1
                args["out_query_positions_ptr"][token] = pos
                args["out_query_slot_mapping_ptr"][token] = (
                    table[req, pos // block_size] * block_size + pos % block_size
                )
                args["out_input_ids_ptr"][token] = (
                    args["next_token_ids_ptr"][req] if step == 0 else args["parallel_drafting_token_id"]
                )
                if step:
                    args["out_token_indices_ptr"][req * (query_length - 1) + step - 1] = token


@pytest.fixture
def harness(monkeypatch):
    root = Path(__file__).resolve().parents[3]
    events = []
    current = SimpleNamespace(value=None)

    @contextmanager
    def forward_context(metadata, config, **kwargs):
        previous = current.value
        current.value = SimpleNamespace(attn_metadata=metadata, cudagraph_runtime_mode=kwargs["aclgraph_runtime_mode"])
        events.append(("context", kwargs))
        try:
            yield
        finally:
            current.value = previous

    def register(name, **attrs):
        module = ModuleType(name)
        module.__path__ = []
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    for name in ("vllm", "vllm_ascend", "vllm_ascend.spec_decode"):
        register(name)
    register("vllm.config", CUDAGraphMode=GraphMode, VllmConfig=SimpleNamespace)
    register("vllm.forward_context", get_forward_context=lambda: current.value)
    register("vllm.v1.attention.backends.utils", CommonAttentionMetadata=SimpleNamespace)
    register(
        "vllm_ascend.ascend_forward_context", set_ascend_forward_context=forward_context, _EXTRA_CTX=SimpleNamespace()
    )
    register("vllm_ascend.attention.attention_v1", AscendAttentionState=SimpleNamespace(ChunkedPrefill="chunked"))
    register("vllm_ascend.attention.utils", AscendCommonAttentionMetadata=SimpleNamespace)
    register("vllm_ascend.utils", lmhead_tp_enable=lambda: False)
    register(
        "vllm_ascend.ops.triton.spec_decode.utils",
        copy_and_expand_dflash_inputs_kernel_single_grid=CpuPacker(),
    )

    class EagleStub:
        def __init__(self, config, device, runner=None):
            self.vllm_config = config
            self.speculative_config = config.speculative_config
            self.device = device
            self.runner = runner
            self.hidden_size = 4
            self.max_num_tokens = 16
            self.max_batch_size = 4
            self.num_speculative_tokens = config.speculative_config.num_speculative_tokens
            self.dtype = torch.float32
            self.uses_mrope = self.uses_xdrope_dim = self.supports_mm_inputs = False
            self.use_cuda_graph = self.use_compress = self.enable_enpu = False
            self.pcp_size = self.dcp_size = 1
            self.method = "dflash"
            self.parallel_drafting = True
            self.parallel_drafting_token_id = 7
            self.kernel_block_size = 4
            self.token_arange_np = np.arange(17, dtype=np.int32)
            self.input_ids = torch.zeros(16, dtype=torch.int32)
            self.slot_mapping_group = [torch.zeros(16, dtype=torch.int32)]
            self.seq_lens_group = [torch.zeros(16, dtype=torch.int32)]
            self.query_start_loc_group = [torch.zeros(16, dtype=torch.int32)]
            self.token_indices_to_sample = torch.zeros(16, dtype=torch.int32)
            self.attn_layer_names = ["draft.layer.0", "draft.layer.1", "draft.layer.2"]
            self._runnable = self._run_merged_draft

        def get_model(self):
            return self.model

        def _update_full_graph_params_if_needed(self, *args):
            pass

    globals_ = dict(
        torch=torch,
        partial=partial,
        CUDAGraphMode=GraphMode,
        set_ascend_forward_context=forward_context,
        get_forward_context=lambda: current.value,
        enable_sp=lambda *args: False,
        Eagle3LlamaForCausalLM=type("EagleModel", (), {}),
        DFlashQwen3ForCausalLM=type("QwenModel", (), {}),
    )
    methods = source_methods(
        root / "vllm_ascend/spec_decode/llm_base_proposer.py",
        "AscendSpecDecodeBaseProposer",
        ["_propose", "_combine_target_hidden_states"],
        globals_,
    )
    for name, method in methods.items():
        setattr(EagleStub, name, method)
    register("vllm_ascend.spec_decode.eagle_proposer", AscendEagleProposer=EagleStub)

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, root / path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    parent = load("vllm_ascend.spec_decode.dflash_proposer", "vllm_ascend/spec_decode/dflash_proposer.py")
    hy = load("vllm_ascend.spec_decode.hy_dflash_proposer", "vllm_ascend/spec_decode/hy_dflash_proposer.py")
    return SimpleNamespace(
        cls=hy.AscendHYDflashProposer,
        parent=parent.AscendDflashProposer,
        events=events,
        context=forward_context,
        config_globals=globals_,
        register=register,
        load=load,
        root=root,
    )


def config(k=2):
    return SimpleNamespace(
        speculative_config=SimpleNamespace(
            method="dflash",
            parallel_drafting=True,
            enforce_eager=True,
            draft_tensor_parallel_size=1,
            num_speculative_tokens=k,
            draft_model_config=SimpleNamespace(hf_config=SimpleNamespace(dflash_config={})),
        ),
        parallel_config=SimpleNamespace(),
        lora_config=None,
    )


class ToyModel:
    """Three cache writers and deterministic prefix hooks, not MLA/DSA math."""

    hy_dflash_contract_version = 1

    def __init__(self, events):
        self.events = events
        self.caches = [torch.full((128, 4), -99.0) for _ in range(3)]
        self.indexer_caches = [torch.full((128, 4), -99.0) for _ in range(3)]
        self.prefixes = []
        self.hidden_prefix_lengths = []

    def combine_hidden_states(self, hidden):
        return hidden

    def precompute_and_store_context_kv(self, hidden, positions, slots):
        self.events.append(("prefill", None if slots is None else slots.clone()))
        if slots is not None:
            valid = slots >= 0
            for layer, (cache, indexer) in enumerate(zip(self.caches, self.indexer_caches)):
                cache[slots[valid].long()] = hidden[valid] + layer
                indexer[slots[valid].long()] = hidden[valid] + layer

    def __call__(self, input_ids, positions, inputs_embeds):
        self.events.append(("query", input_ids.clone()))
        return input_ids.float()[:, None].expand(-1, 4).contiguous()

    def correct_hidden_states(self, hidden, previous_ids, step):
        self.events.append(("correct", step))
        self.prefixes.append(previous_ids.clone())
        self.hidden_prefix_lengths.append(hidden.shape[1])
        return hidden[:, step + 1, :] + previous_ids.sum(dim=-1, keepdim=True)

    def select_hidden_path(self, corrected, step):
        self.events.append(("select", step))
        return corrected

    def compute_logits(self, hidden):
        self.events.append(("logits", len(hidden)))
        ids = hidden[:, 0].long() % 5
        return torch.nn.functional.one_hot(ids, num_classes=5).float()


def make_proposer(harness, k=2):
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(lora_id_to_lora_request={}),
        _sync_metadata_across_dp=lambda tokens, **kwargs: (tokens, None, None),
        get_model=lambda: None,
    )
    proposer = harness.cls(config(k), torch.device("cpu"), runner)
    proposer.model = ToyModel(harness.events)

    def build(_, metadata, target, **kwargs):
        harness.events.append(("build_metadata", metadata))
        return SimpleNamespace(causal=metadata.causal, num_prefills=1)

    proposer.draft_attn_groups = [SimpleNamespace(get_metadata_builder=lambda: SimpleNamespace(build=build))]
    return proposer


def batch():
    positions = torch.tensor([2, 3, 4, 5, 10, 11, 12], dtype=torch.int32)
    seq_lens = torch.tensor([6, 13], dtype=torch.int32)
    starts = torch.tensor([0, 4, 7], dtype=torch.int32)
    metadata = SimpleNamespace(
        num_reqs=2,
        query_start_loc=starts,
        query_start_loc_cpu=starts.clone(),
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens.clone(),
        _seq_lens_cpu=seq_lens.clone(),
        seq_lens_cpu_upper_bound=seq_lens.clone(),
        num_computed_tokens_cpu=torch.tensor([2, 10]),
        block_table_tensor=torch.tensor([[2, 8, 9, 10, 11], [20, 21, 22, 23, 24]], dtype=torch.int32),
        max_seq_len=13,
        slot_mapping=torch.zeros(7, dtype=torch.int32),
        causal=True,
        positions=positions,
        positions_cpu=positions.clone(),
        batch_size=lambda: 2,
    )
    return dict(
        target_token_ids=torch.arange(7),
        target_positions=positions,
        target_hidden_states=torch.arange(28, dtype=torch.float32).view(7, 4),
        next_token_ids=torch.tensor([1, 2]),
        token_indices_to_sample=None,
        cad=metadata,
        num_rejected_tokens_gpu=torch.tensor([2, 1], dtype=torch.int32),
    )


def test_rejected_context_query_positions_and_cache_writes(harness):
    proposer = make_proposer(harness)
    inputs = batch()
    count, indices, metadata, _ = proposer.set_inputs_first_pass(**inputs)
    assert count == 6
    assert indices.tolist() == [1, 2, 4, 5]
    assert proposer.input_ids[:6].tolist() == [1, 7, 7, 2, 7, 7]
    assert metadata.positions.tolist() == [4, 5, 6, 12, 13, 14]
    assert metadata.seq_lens.tolist() == [7, 15]
    assert metadata.slot_mapping.tolist() == [32, 33, 34, 92, 93, 94]
    assert metadata.causal
    assert metadata.seq_lens_cpu is None and metadata._seq_lens_cpu is None
    assert metadata.positions_cpu is None
    assert metadata.seq_lens_cpu_upper_bound.tolist() == [9, 16]
    assert inputs["cad"].seq_lens.tolist() == [6, 13]
    assert inputs["cad"].query_start_loc.tolist() == [0, 4, 7]
    assert proposer._context_slot_mapping_buffer[:7].tolist() == [10, 11, -1, -1, 90, 91, -1]
    for layer, cache in enumerate(proposer.model.caches):
        torch.testing.assert_close(cache[10], inputs["target_hidden_states"][0] + layer)
        assert (cache[32:35] == -99).all()
        assert (proposer.model.indexer_caches[layer][92] == -99).all()
    assert len([event for event in harness.events if event[0] == "prefill"]) == 1


@pytest.mark.parametrize("rejected", [None, torch.zeros(2, dtype=torch.int32)])
def test_unpadded_and_all_accepted_context(harness, rejected):
    proposer = make_proposer(harness)
    inputs = batch()
    inputs["num_rejected_tokens_gpu"] = rejected
    _, _, metadata, _ = proposer.set_inputs_first_pass(**inputs)
    assert (proposer._context_slot_mapping_buffer[:7] >= 0).all()
    assert metadata.positions.tolist() == [6, 7, 8, 13, 14, 15]


def test_real_base_propose_dispatches_hy_hooks_and_samples_in_order(harness):
    proposer = make_proposer(harness)
    inputs = batch()
    inputs["common_attn_metadata"] = inputs.pop("cad")
    ids = proposer._propose(**inputs, target_model_batch_desc=SimpleNamespace(uniform=False), sampling_metadata=None)
    assert ids.tolist() == [[2, 4], [2, 4]]
    events = [event[0] for event in harness.events if event[0] != "context"]
    assert events == [
        "prefill",
        "build_metadata",
        "query",
        "correct",
        "select",
        "logits",
        "correct",
        "select",
        "logits",
    ]
    assert proposer.model.prefixes[0].shape == (2, 0)
    assert proposer.model.prefixes[1].tolist() == [[2], [2]]
    assert proposer.model.hidden_prefix_lengths == [2, 3]
    assert proposer.verification_probs() is None


def test_query_inputs_do_not_repeat_context_prefill(harness):
    proposer = make_proposer(harness)
    proposer.set_inputs_first_pass(**batch())
    harness.events.clear()
    proposer.build_model_inputs_first_pass(6)
    assert harness.events == []


@pytest.mark.parametrize("profile", [False, True])
def test_dummy_runs_all_stages_without_writing_live_context(harness, profile):
    proposer = make_proposer(harness, k=5)
    proposer.dummy_run(num_tokens=7, num_reqs=4, is_profile=profile, aclgraph_runtime_mode=GraphMode.FULL)
    events = [event for event in harness.events if event[0] != "context"]
    assert events[0] == ("prefill", None)
    assert events[1][0] == "query" and len(events[1][1]) == 24
    assert [event[0] for event in events].count("logits") == 5
    assert all((cache == -99).all() for cache in proposer.model.caches)
    assert all(event[1]["aclgraph_runtime_mode"] == GraphMode.NONE for event in harness.events if event[0] == "context")


def test_dummy_empty_batch_is_noop(harness):
    proposer = make_proposer(harness)
    proposer.dummy_run(0, num_reqs=0)
    assert harness.events == []


@pytest.mark.parametrize(
    "name",
    [
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "data_parallel_size",
        "prefill_context_parallel_size",
        "decode_context_parallel_size",
    ],
)
def test_distributed_configuration_fails_closed(harness, name):
    cfg = config()
    setattr(cfg.parallel_config, name, 2)
    with pytest.raises(NotImplementedError, match=name):
        harness.cls._validate_config(cfg)


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("enforce_eager", False, NotImplementedError),
        ("parallel_drafting", False, ValueError),
        ("method", "mtp", ValueError),
        ("draft_tensor_parallel_size", 2, NotImplementedError),
    ],
)
def test_unsupported_draft_configuration(harness, field, value, error):
    cfg = config()
    setattr(cfg.speculative_config, field, value)
    with pytest.raises(error):
        harness.cls._validate_config(cfg)


def test_contract_and_ihc_errors(harness):
    proposer = make_proposer(harness)
    proposer._validate_model_contract()
    proposer.model.hy_dflash_contract_version = 0
    with pytest.raises(ValueError, match="external model"):
        proposer._validate_model_contract()
    proposer.model.hy_dflash_contract_version = 1
    proposer.model.select_hidden_path = None
    with pytest.raises(TypeError, match="select_hidden_path"):
        proposer._validate_model_contract()
    with pytest.raises(ValueError, match="IHC"):
        proposer._combine_target_hidden_states(torch.zeros(2, 4, 4))


def test_stochastic_probabilities_are_not_silently_used(harness):
    proposer = make_proposer(harness)
    ids = torch.zeros(2, 2, dtype=torch.long)
    assert proposer.commit_draft_probs(ids, None) is ids
    with pytest.raises(NotImplementedError, match="Stochastic"):
        proposer.commit_draft_probs(ids, torch.ones(2, 2, 5) / 5)


def test_sparse_topk_is_not_shared_with_target(harness):
    proposer = make_proposer(harness)
    sentinel = object()
    proposer.model.model = SimpleNamespace(topk_indices_buffer=sentinel)
    proposer._maybe_share_topk_indices(SimpleNamespace(model=SimpleNamespace(topk_indices_buffer=object())))
    assert proposer.model.model.topk_indices_buffer is sentinel


def test_factory_is_explicit_opt_in(harness, monkeypatch):
    for module_name, class_name in (
        ("draft_proposer", "AscendDraftModelProposer"),
        ("extract_hidden_states_proposer", "AscendExtractHiddenStatesProposer"),
        ("medusa_proposer", "AscendMedusaProposer"),
        ("ngram_proposer", "AscendNgramProposer"),
        ("ngram_proposer_npu", "AscendNgramProposerNPU"),
        ("suffix_proposer", "AscendSuffixDecodingProposer"),
    ):
        harness.register(f"vllm_ascend.spec_decode.{module_name}", **{class_name: Mock()})
    factory = harness.load("vllm_ascend.spec_decode", "vllm_ascend/spec_decode/__init__.py")
    regular, hy = Mock(return_value="regular"), Mock(return_value="hy")
    monkeypatch.setattr(factory, "AscendDflashProposer", regular)
    monkeypatch.setattr(factory, "AscendHYDflashProposer", hy)
    cfg = config()
    assert factory.get_spec_decode_method("dflash", cfg, "cpu", None) == "regular"
    cfg.speculative_config.draft_model_config.hf_config.dflash_config["ascend_proposer"] = "hy"
    assert factory.get_spec_decode_method("dflash", cfg, "cpu", None) == "hy"


def test_runner_passes_draft_q_but_keeps_target_logits(harness):
    proposer = make_proposer(harness)
    logits, metadata = torch.randn(6, 5), object()
    sampling = SimpleNamespace(top_k=None)
    runner = SimpleNamespace(
        drafter=proposer,
        input_batch=SimpleNamespace(sampling_metadata=sampling, update_async_output_token_ids=Mock()),
        rejection_sampler=Mock(return_value="verified"),
    )
    method = source_methods(
        harness.root / "vllm_ascend/worker/model_runner_v1.py",
        "NPUModelRunner",
        ["_sample"],
        dict(AscendHYDflashProposer=harness.cls, lmhead_tp_enable=lambda: False),
    )["_sample"]
    assert method(runner, logits, metadata) == "verified"
    runner.rejection_sampler.assert_called_once_with(metadata, None, logits, sampling)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("target_positions", torch.zeros(3, 7), "one position"),
        ("target_hidden_states", torch.zeros(7, 4, 4), "context must"),
        ("target_token_ids", torch.zeros(17), "context buffer"),
        ("target_token_ids", torch.zeros(0), "at least one"),
        ("next_token_ids", torch.zeros(1), "one anchor"),
        ("num_rejected_tokens_gpu", torch.zeros(1), "rejected-token count"),
    ],
)
def test_input_shape_errors_precede_any_cache_write(harness, field, value, message):
    proposer = make_proposer(harness)
    inputs = batch()
    inputs[field] = value
    with pytest.raises(ValueError, match=message):
        proposer.set_inputs_first_pass(**inputs)
    assert harness.events == []


@pytest.mark.parametrize("num_reqs", [0, 5])
def test_batch_capacity_errors(harness, num_reqs):
    proposer = make_proposer(harness)
    inputs = batch()
    inputs["cad"].num_reqs = num_reqs
    with pytest.raises(ValueError, match="max_batch_size"):
        proposer.set_inputs_first_pass(**inputs)


def test_query_start_shape_error(harness):
    proposer = make_proposer(harness)
    inputs = batch()
    inputs["cad"].query_start_loc = torch.tensor([0, 4, 7, 7])
    with pytest.raises(ValueError, match="unpadded"):
        proposer.set_inputs_first_pass(**inputs)


@pytest.mark.parametrize("num_tokens", [-1, 13])
def test_query_input_capacity_errors(harness, num_tokens):
    proposer = make_proposer(harness)
    with pytest.raises(ValueError, match="query buffer"):
        proposer.build_model_inputs_first_pass(num_tokens)


def run_query(harness, proposer):
    with harness.context(None, proposer.vllm_config, aclgraph_runtime_mode=GraphMode.NONE):
        return proposer._run_merged_draft(6, 2, None, None, None, [], 6)


@pytest.mark.parametrize("output", [torch.zeros(6, 2, 4), torch.zeros(5, 4), (torch.zeros(6, 4),)])
def test_model_output_contract_failure(harness, monkeypatch, output):
    proposer = make_proposer(harness)
    monkeypatch.setattr(ToyModel, "__call__", lambda *args, **kwargs: output)
    with pytest.raises(ValueError, match="without IHC"):
        run_query(harness, proposer)


def test_path_selection_shape_failure(harness):
    proposer = make_proposer(harness)
    proposer.model.select_hidden_path = lambda *args: torch.zeros(2, 2, 4)
    with pytest.raises(ValueError, match="path selector"):
        run_query(harness, proposer)


@pytest.mark.parametrize("output", [None, torch.zeros(2, 3, 4), torch.zeros(3, 5)])
def test_lm_head_shape_failure(harness, output):
    proposer = make_proposer(harness)
    proposer.model.compute_logits = lambda *args: output
    with pytest.raises(ValueError, match="LM head"):
        run_query(harness, proposer)


@pytest.mark.parametrize(
    "method,model_type", [("eagle3", "Eagle3LlamaForCausalLM"), ("dflash", "DFlashQwen3ForCausalLM")]
)
def test_default_hidden_combination_preserves_model_guard(harness, method, model_type):
    proposer = harness.parent.__new__(harness.parent)
    proposer.method, proposer.hidden_size = method, 4
    proposer.model = harness.config_globals[model_type]()
    hidden = torch.zeros(3, 4)
    proposer.model.combine_hidden_states = Mock(return_value=hidden + 1)
    torch.testing.assert_close(proposer._combine_target_hidden_states(hidden), hidden + 1)
    proposer.model = ToyModel([])
    with pytest.raises(AssertionError):
        proposer._combine_target_hidden_states(hidden)


def test_mtp_hidden_combination_is_unchanged(harness):
    proposer = harness.parent.__new__(harness.parent)
    proposer.method = "mtp"
    hidden = torch.zeros(3, 4)
    assert proposer._combine_target_hidden_states(hidden) is hidden


def test_lora_is_rejected(harness):
    cfg = config()
    cfg.lora_config = object()
    with pytest.raises(NotImplementedError, match="LoRA"):
        harness.cls._validate_config(cfg)


def test_next_round_replaces_rejected_slot_mask(harness):
    proposer = make_proposer(harness)
    proposer.set_inputs_first_pass(**batch())
    inputs = batch()
    inputs["num_rejected_tokens_gpu"] = None
    proposer.set_inputs_first_pass(**inputs)
    assert (proposer._context_slot_mapping_buffer[:7] >= 0).all()
    torch.testing.assert_close(proposer.model.caches[0][32], inputs["target_hidden_states"][2])
