# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Portable numerical/contract tests; NOT checkpoint or Ascend kernel validation.

Run pytest --noconftest on CPU without vLLM/torch_npu. Infrastructure is isolated
and restored per test. Attention, adapter, DFlash parent and HY proposer modules
are loaded from real source. Tiny projections/3-D rotary exercise composition;
they do not assert that a vendor checkpoint uses these toy weight layouts.
"""

import __future__

import ast
import importlib.util
import json
import struct
import sys
import typing
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn


class GraphMode(Enum):
    NONE = 0
    FULL = 1


@dataclass(frozen=True)
class BatchDescriptor:
    num_tokens: int
    num_reqs: int
    uniform: bool


@pytest.fixture
def harness(monkeypatch):
    root = Path(__file__).resolve().parents[3]
    events = []
    capture_events = []

    class AddressGraph:
        """CPU stand-in that reuses capture-time tensor views on every replay.

        It deliberately never calls the proposer's Python runnable on replay.
        This tests buffer lifetimes/data updates, not NPU graph operator support.
        """

        def __init__(self, runnable, config, **kwargs):
            self.proposer = runnable.__self__
            self.runnable = runnable
            self.concrete_aclgraph_entries = {}
            self.captured_batches = []

        def __call__(self, *args, **kwargs):
            context = events[-1]
            if context["aclgraph_runtime_mode"] == GraphMode.NONE:
                return self.runnable(*args, **kwargs)
            descriptor = context["batch_descriptor"]
            entry = self.concrete_aclgraph_entries.get(descriptor)
            if entry is None:
                assert capture_events[-1] is True
                inputs = self.proposer.build_model_inputs_first_pass(descriptor.num_tokens)
                entry = SimpleNamespace(inputs=inputs, output=None)
                self.concrete_aclgraph_entries[descriptor] = entry
                self.captured_batches.append(descriptor.num_reqs)
            hidden = self.proposer.model(**entry.inputs)
            output = self.proposer.model.compute_logits(hidden[:, 1:].reshape(-1, self.proposer.hidden_size))
            output = output.argmax(-1).view(descriptor.num_reqs, -1)
            if entry.output is None:
                entry.output = output
            else:
                entry.output.copy_(output)
            return entry.output

    @contextmanager
    def forward_context(metadata, config, **kwargs):
        events.append(kwargs)
        yield

    def register(name, **attrs):
        module = ModuleType(name)
        module.__path__ = []
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    for name in (
        "vllm",
        "vllm_ascend",
        "vllm_ascend.spec_decode",
        "vllm_ascend.attention",
        "vllm_ascend.patch",
        "vllm_ascend.patch.worker",
    ):
        register(name)
    register("vllm.config", CUDAGraphMode=GraphMode, VllmConfig=SimpleNamespace)
    register("vllm.forward_context", get_forward_context=lambda: None, BatchDescriptor=BatchDescriptor)
    register("vllm.compilation.monitor", set_cudagraph_capturing_enabled=capture_events.append)
    register("vllm.logger", logger=Mock())
    register("vllm_ascend.compilation.acl_graph", ACLGraphWrapper=AddressGraph)
    register("vllm.v1.attention.backends.utils", CommonAttentionMetadata=SimpleNamespace)
    register(
        "vllm_ascend.ascend_forward_context", set_ascend_forward_context=forward_context, _EXTRA_CTX=SimpleNamespace()
    )
    register("vllm_ascend.attention.attention_v1", AscendAttentionState=SimpleNamespace())
    register("vllm_ascend.attention.utils", AscendCommonAttentionMetadata=SimpleNamespace)
    register("vllm_ascend.utils", lmhead_tp_enable=lambda: False)
    register("vllm_ascend.ops.triton.spec_decode.utils", copy_and_expand_dflash_inputs_kernel_single_grid=None)

    class EagleStub:
        def __init__(self, config, device, runner=None):
            self.vllm_config = config
            self.speculative_config = config.speculative_config
            self.device, self.runner = device, runner
            self.hidden_size, self.max_num_tokens, self.max_batch_size = 4, 16, 4
            self.num_speculative_tokens = config.speculative_config.num_speculative_tokens
            self.dtype = torch.float32
            self.parallel_drafting_token_id = config.speculative_config.draft_model_config.hf_config.dflash_config[
                "mask_token_id"
            ]
            self.eagle3_use_aux_hidden_state = self._get_eagle3_use_aux_hidden_state_from_config()
            self.maybe_eager_context = nullcontext()
            self._runnable = self._run_merged_draft

        def get_model(self):
            return self.model

    register("vllm_ascend.spec_decode.eagle_proposer", AscendEagleProposer=EagleStub)

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, root / path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    config_module = load("vllm_ascend.patch.hy4_config", "vllm_ascend/patch/hy4_config.py")
    attention = load("vllm_ascend.attention.hy_dflash", "vllm_ascend/attention/hy_dflash.py")
    adapter = load("vllm_ascend.patch.worker.hy_dflash", "vllm_ascend/patch/worker/hy_dflash.py")
    parent = load("vllm_ascend.spec_decode.dflash_proposer", "vllm_ascend/spec_decode/dflash_proposer.py")
    hy = load("vllm_ascend.spec_decode.hy_dflash_proposer", "vllm_ascend/spec_decode/hy_dflash_proposer.py")
    return SimpleNamespace(
        cls=hy.AscendHYDflashProposer,
        parent=parent.AscendDflashProposer,
        attention=attention,
        adapter=adapter,
        events=events,
        capture_events=capture_events,
        root=root,
        register=register,
        load=load,
        config_module=config_module,
    )


def config(k=7):
    return SimpleNamespace(
        speculative_config=SimpleNamespace(
            method="dflash",
            parallel_drafting=True,
            enforce_eager=True,
            draft_tensor_parallel_size=1,
            num_speculative_tokens=k,
            draft_model_config=SimpleNamespace(
                hf_config=SimpleNamespace(
                    dflash_config={
                        "ascend_proposer": "hy",
                        "mask_token_id": 120009,
                        "target_layer_ids": [61, 65, 69, 73, 77],
                        "context_topk": 2048,
                        "context_local_window": 8,
                    }
                )
            ),
        ),
        parallel_config=SimpleNamespace(),
        lora_config=None,
    )


@pytest.mark.parametrize("query_len", [1, 8])
@pytest.mark.parametrize("capacity", [0, 1, 13, 2056])
def test_sfa_indices_exact_context_and_causal_block(harness, query_len, capacity):
    valid = torch.arange(capacity).expand(3, -1) % 3 == 0
    valid = valid.clone()
    valid[1] = False
    valid[2] = True
    indices = harness.attention.hy_sfa_indices(valid, query_len)
    assert indices.shape == (3, query_len, 1, capacity + query_len)
    assert indices.dtype == torch.int32 and indices.is_contiguous()
    for batch in range(3):
        for query in range(query_len):
            wanted = torch.cat((torch.where(valid[batch])[0], torch.arange(capacity, capacity + query + 1)))
            actual = indices[batch, query, 0]
            torch.testing.assert_close(actual[: len(wanted)].long(), wanted)
            assert (actual[len(wanted) :] == -1).all()


def emulate_native_sfa(query, key, value, indices, scale, **kwargs):
    """CPU semantic oracle for the operator contract, not an NPU emulator."""
    assert key.data_ptr() == value.data_ptr()
    assert kwargs["attention_mode"] == 2 and kwargs["sparse_mode"] == 0
    assert kwargs["sparse_block_size"] == 1 and not kwargs["return_softmax_lse"]
    assert kwargs["layout_query"] == kwargs["layout_kv"] == "BSND"
    assert query.is_contiguous() and key.is_contiguous() and indices.is_contiguous()
    assert kwargs["key_rope"].is_contiguous() and kwargs["query_rope"].is_contiguous()
    output = torch.empty_like(query)
    for b in range(query.shape[0]):
        for q in range(query.shape[1]):
            row = indices[b, q, 0]
            slots = row[row >= 0].long()
            assert slots.unique().numel() == slots.numel()
            scores = query[b, q].float() @ key[b, slots, 0].float().T
            scores += kwargs["query_rope"][b, q].float() @ kwargs["key_rope"][b, slots, 0].float().T
            scores *= scale
            values = value[b, slots, 0].float()
            if "sinks" in kwargs:
                assert kwargs["sinks"].dtype == torch.float32
                scores = torch.cat((scores, kwargs["sinks"][:, None]), dim=-1)
                values = torch.cat((values, torch.zeros_like(values[:1])))
            output[b, q] = scores.softmax(-1) @ values
    return output, torch.empty(0), torch.empty(0)


@pytest.mark.parametrize("sink", [False, True])
def test_sfa_attention_matches_reference_and_refreshes_indices(harness, sink):
    torch.manual_seed(27)
    a = harness.attention
    batch, queries, heads, nope, rank, rope = 3, 8, 3, 16, 512, 64
    capacity = 11

    def bf16(*shape):
        return torch.randn(*shape).bfloat16()

    q = bf16(batch, queries, heads, nope)
    qr = bf16(batch, queries, heads, rope)
    latent, kr = bf16(batch, queries, rank), bf16(batch, queries, rope)
    key_up = (torch.randn(heads, nope, rank) / rank**0.5).bfloat16()
    value_up = (torch.randn(heads, 12, rank) / rank**0.5).bfloat16()
    valid = torch.arange(capacity).expand(batch, -1).clone() % 3 != 1
    valid[1] = False
    kv = bf16(batch, capacity, rank + rope)
    context = a.HYContext(61, torch.arange(capacity).expand(batch, -1), kv, valid)
    sinks = torch.tensor([-3, 0, 4], dtype=torch.bfloat16) if sink else None
    scale = (nope + rope) ** -0.5

    def run():
        return a.sfa_absorbed_mla(q, qr, latent, kr, context, key_up, value_up, scale, sinks, sfa_op=emulate_native_sfa)

    for _ in range(2):
        expected = a.absorbed_mla(q, qr, latent, kr, context, key_up, value_up, scale, sinks)
        actual = run()
        torch.testing.assert_close(actual.float(), expected.float(), atol=0.02, rtol=0.03)
        before = actual[:, 0].clone()
        saved_latent, saved_rope, saved_kv = latent.clone(), kr.clone(), kv.clone()
        # Future draft K/V and invalid context must not affect query zero.
        latent[:, 1:] += 10
        kr[:, 1:] -= 10
        kv[~valid] += 20
        torch.testing.assert_close(run()[:, 0], before, atol=0, rtol=0)
        latent.copy_(saved_latent)
        kr.copy_(saved_rope)
        kv.copy_(saved_kv)
        # Update the captured inputs in place: holes, empty/full context, order.
        valid.copy_(~valid.roll(1, dims=0))
        kv.copy_(kv.roll(1, dims=0))


@pytest.mark.parametrize("has_sink", [False, True])
def test_sfa_operator_requires_sink_capability(harness, monkeypatch, has_sink):
    op = SimpleNamespace(
        _schema=SimpleNamespace(arguments=[SimpleNamespace(name=n) for n in ("attention_mode", "return_softmax_lse")])
    )
    monkeypatch.setattr(torch.ops.npu, "npu_sparse_flash_attention", SimpleNamespace(default=op), raising=False)
    if has_sink:
        with pytest.raises(RuntimeError, match="sinks"):
            harness.attention.get_hy_sfa_op(has_sink=True)
    else:
        assert harness.attention.get_hy_sfa_op(has_sink=False) is op
    op._schema.arguments.append(SimpleNamespace(name="sinks"))
    assert harness.attention.get_hy_sfa_op(has_sink=True) is op


def test_sfa_backend_config_is_explicit_and_validated(harness):
    cfg = config().speculative_config.draft_model_config.hf_config
    assert harness.config_module.get_hy_dflash_config(cfg)["ascend_attention_backend"] == "auto"
    for backend in ("sfa", "reference"):
        cfg.dflash_config["ascend_attention_backend"] = backend
        assert harness.config_module.get_hy_dflash_config(cfg)["ascend_attention_backend"] == backend
    cfg.dflash_config["ascend_attention_backend"] = "unknown"
    with pytest.raises(ValueError, match="ascend_attention_backend"):
        harness.config_module.get_hy_dflash_config(cfg)


def test_sfa_rejects_incompatible_dtype_before_kernel(harness):
    q = torch.zeros(1, 8, 1, 16)
    context = harness.attention.HYContext(
        61, torch.zeros(1, 4), torch.zeros(1, 4, 576), torch.ones(1, 4, dtype=torch.bool)
    )
    op = Mock()
    with pytest.raises(ValueError, match="BF16"):
        harness.attention.sfa_absorbed_mla(
            q,
            torch.zeros(1, 8, 1, 64),
            torch.zeros(1, 8, 512),
            torch.zeros(1, 8, 64),
            context,
            torch.zeros(1, 16, 512),
            torch.zeros(1, 12, 512),
            0.2,
            None,
            sfa_op=op,
        )
    op.assert_not_called()


class TinyProject(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(4, 16, bias=False)
        self.kv_a = nn.Linear(4, 9, bias=False)
        self.norm = nn.RMSNorm(3)

    def forward(self, hidden):
        query = self.q(hidden).view(*hidden.shape[:2], 2, 8)
        kv = self.kv_a(hidden)
        return query[..., :2], query[..., 2:], self.norm(kv[..., :3]), kv[..., 3:]


class TinyRotary(nn.Module):
    def forward(self, positions, query, key):
        self.seen_positions = positions.clone()
        angles = positions.permute(1, 2, 0).float() * 0.03

        def rotate(value, angle):
            pairs = value.reshape(*value.shape[:-1], 3, 2)
            first = pairs[..., 0] * angle.cos() - pairs[..., 1] * angle.sin()
            second = pairs[..., 0] * angle.sin() + pairs[..., 1] * angle.cos()
            return torch.stack((first, second), dim=-1).flatten(-2)

        return rotate(query, angles[:, :, None]), rotate(key, angles)


class TinyFinish(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(4, 2)
        self.o_proj = nn.Linear(4, 4, bias=False)

    def forward(self, hidden, output):
        return self.o_proj((output * self.gate(hidden).sigmoid()[..., None]).flatten(-2))


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate, self.up, self.down = nn.Linear(4, 12), nn.Linear(4, 12), nn.Linear(12, 4)

    def forward(self, hidden):
        assert hidden.ndim == 2
        return self.down(torch.nn.functional.silu(self.gate(hidden)) * self.up(hidden))


def make_model(harness, cfg=None):
    torch.manual_seed(34)
    a, m = harness.attention, harness.adapter
    layer_ids = (cfg or config()).speculative_config.draft_model_config.hf_config.dflash_config["target_layer_ids"]
    layers = []
    for _ in layer_ids:
        attention = a.HYDflashAttention(
            TinyProject(),
            TinyRotary(),
            TinyFinish(),
            torch.randn(2, 2, 3) * 0.1,
            torch.randn(2, 2, 3) * 0.1,
            rope_dim=6,
            scale=8**-0.5,
        )
        layers.append(m.HYDenseDraftLayer(nn.RMSNorm(4), attention, nn.RMSNorm(4), TinyMLP()))
    fusion = m.HYMTPFusion(nn.RMSNorm(4), nn.RMSNorm(4), nn.Linear(8, 4), torch.randn(4))
    correction = m.HYHiddenCorrection(nn.RMSNorm(4), nn.RMSNorm(4), nn.Linear(8, 4), nn.Linear(8, 4), nn.Linear(4, 4))
    return m.HYDflashAdapter(
        layers,
        nn.Embedding(120010, 4),
        nn.RMSNorm(4),
        fusion,
        correction,
        nn.Linear(4, 1, bias=True),
        nn.Linear(4, 9),
        hidden_size=4,
        target_layer_ids=layer_ids,
    )


class TargetProvider:
    def __init__(self, attention, cfg):
        self.attention = attention
        draft = cfg.speculative_config.draft_model_config.hf_config.dflash_config
        self.context_topk = draft["context_topk"]
        self.caches = [torch.randn(8, 4, 9) for _ in draft["target_layer_ids"]]
        self.calls = []

    def get_hy_dflash_layer_names(self, layer_ids):
        return [f"target.layers.{layer_id}.attn" for layer_id in layer_ids]

    def get_hy_dflash_context(self, *, request_ids, anchor_positions, layer_ids, block_size):
        self.calls.append((request_ids, anchor_positions.clone()))
        batch = len(request_ids)
        page_base = torch.tensor([0 if req == "a" else 4 for req in request_ids])
        table = page_base[:, None] + torch.arange(4)[None]
        layers = tuple(
            self.attention.HYTargetLayerState(layer_id, cache, table, torch.arange(self.context_topk).expand(batch, -1))
            for layer_id, cache in zip(layer_ids, self.caches)
        )
        positions = anchor_positions[:, None] + torch.arange(block_size)[None]
        positions = positions[None] + torch.tensor([10, 20, 30])[:, None, None]
        final_hidden = (anchor_positions - 1)[:, None].float().expand(-1, 4)
        final_hidden = torch.nn.functional.rms_norm(final_hidden, (4,))
        return self.attention.HYTargetState(layers, torch.zeros(batch, dtype=torch.int64), final_hidden, positions)


def make_proposer(harness, cfg=None):
    cfg = cfg or config()
    runner = SimpleNamespace(input_batch=SimpleNamespace(req_ids=["a", "b"]))
    proposer = harness.cls(cfg, torch.device("cpu"), runner)
    proposer.model = make_model(harness, cfg)
    proposer._target_model = TargetProvider(harness.attention, cfg)
    return proposer


def metadata(lengths=(5, 9)):
    return SimpleNamespace(
        num_reqs=len(lengths),
        seq_lens=torch.tensor(lengths),
        max_seq_len=max(lengths),
        slot_mapping=torch.tensor([888]),
        query_start_loc=torch.tensor([0, 1, 2]),
    )


def propose(proposer, lengths=(5, 9), rejected=None):
    return proposer._propose(
        None,
        torch.full((3, 2), 999),
        None,
        torch.tensor([3, 4]),
        None,
        metadata(lengths),
        None,
        None,
        num_rejected_tokens_gpu=rejected,
    )


@pytest.mark.parametrize("anchor,doc", [(0, 0), (1, 0), (3, 0), (4, 0), (9, 6)])
def test_short_context_dense(harness, anchor, doc):
    actual = harness.attention.select_context_indices(
        torch.tensor([[777, -1]]), torch.tensor([anchor]), torch.tensor([doc]), topk=4, local_window=2
    )
    assert actual[actual >= 0].tolist() == list(range(doc, anchor))
    assert actual.shape == (1, 6)


def test_context_union_layer_specific_and_document_bounds(harness):
    result = harness.attention.select_context_indices(
        torch.tensor([[1, 3, 7, 9], [4, 4, 8, 12]]),
        torch.tensor([10, 10]),
        torch.tensor([2, 2]),
        topk=4,
        local_window=3,
    )
    assert result[0].tolist() == [3, 7, 8, 9, -1, -1, -1]
    assert result[1].tolist() == [4, 7, 8, 9, -1, -1, -1]


def test_real_context_capacity(harness):
    result = harness.attention.select_context_indices(
        torch.arange(2048)[None], torch.tensor([3000]), torch.tensor([0]), topk=2048, local_window=8
    )
    assert result.shape == (1, 2056)
    assert result[0, -8:].tolist() == list(range(2992, 3000))


def test_gather_pages_read_only(harness):
    a = harness.attention
    cache = torch.arange(3 * 4 * 9).view(3, 4, 9).float()
    original = cache.clone()
    layer = a.HYTargetLayerState(61, cache, torch.tensor([[2, 0], [1, 2]]), torch.empty(2, 0))
    context = a.gather_context(layer, torch.tensor([[0, 5, -1], [3, 4, -1]]))
    torch.testing.assert_close(context.kv[0, :2], torch.stack([cache[2, 0], cache[0, 1]]))
    torch.testing.assert_close(context.kv[1, :2], torch.stack([cache[1, 3], cache[2, 0]]))
    assert torch.count_nonzero(context.kv[:, -1]) == 0
    context.kv.zero_()
    assert torch.equal(cache, original)


@pytest.mark.parametrize("table,indices", [([[0]], [[4]]), ([[-1]], [[0]]), ([[2]], [[0]])])
def test_gather_unallocated_fails(harness, table, indices):
    a = harness.attention
    layer = a.HYTargetLayerState(61, torch.zeros(2, 4, 9), torch.tensor(table), torch.empty(1, 0))
    with pytest.raises(ValueError, match="unallocated"):
        a.gather_context(layer, torch.tensor(indices))


def test_quantized_cache_requires_explicit_bridge(harness):
    a = harness.attention
    layer = a.HYTargetLayerState(61, torch.zeros(1, 4, 9, dtype=torch.int8), torch.tensor([[0]]), torch.empty(1, 0))
    with pytest.raises(ValueError, match="dequantized"):
        a.gather_context(layer, torch.tensor([[0]]))


def attention_inputs(harness, dtype=torch.float32):
    torch.manual_seed(52)
    a = harness.attention
    qn, qr = torch.randn(2, 8, 2, 2).to(dtype), torch.randn(2, 8, 2, 6).to(dtype)
    latent, rope = torch.randn(2, 8, 3).to(dtype), torch.randn(2, 8, 6).to(dtype)
    indices = torch.tensor([[0, 1, -1], [2, -1, -1]])
    context = a.HYContext(61, indices, torch.randn(2, 3, 9).to(dtype), indices >= 0)
    return qn, qr, latent, rope, context, torch.randn(2, 2, 3).to(dtype), torch.randn(2, 2, 3).to(dtype), 0.3


def expanded_attention(args):
    qn, qr, latent, rope, context, key_up, value_up, scale = args
    all_latent = torch.cat((context.kv[..., :3], latent), 1).float()
    all_rope = torch.cat((context.kv[..., 3:], rope), 1).float()
    keys = torch.einsum("bsr,hdr->bshd", all_latent, key_up.float())
    values = torch.einsum("bsr,hvr->bshv", all_latent, value_up.float())
    outputs = []
    for request in range(2):
        rows = []
        for slot in range(8):
            allowed = torch.cat((context.valid[request], torch.arange(8) <= slot))
            score = (keys[request, allowed] * qn[request, slot].float()).sum(-1)
            score += (all_rope[request, allowed, None] * qr[request, slot].float()).sum(-1)
            probs = (score * scale).softmax(dim=0)
            rows.append((values[request, allowed] * probs[..., None]).sum(dim=0))
        outputs.append(torch.stack(rows))
    return torch.stack(outputs).to(qn.dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_absorbed_matches_explicit_joint_softmax(harness, dtype):
    args = attention_inputs(harness, dtype)
    actual = harness.attention.absorbed_mla(*args)
    torch.testing.assert_close(
        actual,
        expanded_attention(args),
        rtol=0.015 if dtype == torch.bfloat16 else 1e-5,
        atol=0.015 if dtype == torch.bfloat16 else 2e-6,
    )


def test_block_causal_and_request_isolation(harness):
    args = list(attention_inputs(harness))
    before = harness.attention.absorbed_mla(*args)
    args[2] = args[2].clone()
    args[3] = args[3].clone()
    args[2][0, 4:] += 100
    args[3][0, 4:] += 100
    after = harness.attention.absorbed_mla(*args)
    torch.testing.assert_close(after[0, :4], before[0, :4])
    torch.testing.assert_close(after[1], before[1])
    assert not torch.allclose(after[0, 4:], before[0, 4:])


def test_masked_context_has_no_effect(harness):
    args = list(attention_inputs(harness))
    before = harness.attention.absorbed_mla(*args)
    context = args[4]
    kv = context.kv.clone()
    kv[~context.valid] = 100
    args[4] = replace(context, kv=kv)
    torch.testing.assert_close(harness.attention.absorbed_mla(*args), before)


@pytest.mark.parametrize("component", ["layout", "key_up", "query_rope", "block"])
def test_attention_shape_guards(harness, component):
    args = list(attention_inputs(harness))
    if component == "layout":
        args[4] = replace(args[4], kv=args[4].kv[..., :-1])
    elif component == "key_up":
        args[5] = args[5][..., :-1]
    elif component == "query_rope":
        args[1] = args[1][..., :-1]
    else:
        args[2] = args[2][:, :-1]
    with pytest.raises(RuntimeError):
        harness.attention.absorbed_mla(*args)


def test_fusion_and_correction_formulas(harness):
    model = make_model(harness)
    anchor, target = torch.randn(2, 4), torch.randn(2, 4)
    f = model.fusion
    joined = torch.cat((f.embedding_norm(anchor), f.hidden_norm(target)), -1)
    torch.testing.assert_close(f(anchor, target), anchor + f.gate * f.eh_proj(joined))
    c = model.correction
    z = torch.cat((c.hidden_norm(target), c.embedding_norm(anchor)), -1)
    expected = target + c.down_proj(torch.nn.functional.silu(c.gate_proj(z)) * c.up_proj(z))
    torch.testing.assert_close(c(target, anchor), expected)


def test_deferred_linear_bias(harness):
    class Deferred(nn.Module):
        def forward(self, value):
            return value * 2, torch.ones_like(value)

    value = torch.randn(2, 4)
    torch.testing.assert_close(harness.adapter.linear_output(Deferred(), value), value * 2 + 1)


def test_public_decoder_layer_reuse_keeps_loaded_modules(harness):
    class PublicAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.project = TinyProject()
            self.rotary = TinyRotary()
            self.finish = TinyFinish()
            self.key_up = nn.Parameter(torch.randn(2, 2, 3))
            self.value_up = nn.Parameter(torch.randn(2, 2, 3))
            self.qk_rope_head_dim = 6
            self.scaling = 8**-0.5

        def draft_project(self, hidden):
            return self.project(hidden)

        def draft_rotary(self, positions, query, key):
            return self.rotary(positions, query, key)

        def draft_finish(self, hidden, output):
            return self.finish(hidden, output)

        def draft_kv_up_weights(self):
            return self.key_up, self.value_up

    public = SimpleNamespace(
        input_layernorm=nn.RMSNorm(4),
        self_attn=PublicAttention(),
        post_attention_layernorm=nn.RMSNorm(4),
        mlp=TinyMLP(),
    )
    adapted = harness.adapter.HYDenseDraftLayer.from_hyv4_layer(public)
    assert adapted.input_norm is public.input_layernorm
    assert adapted.post_attention_norm is public.post_attention_layernorm
    assert adapted.mlp is public.mlp
    assert adapted.attention.source_attention is public.self_attn
    assert adapted.attention.key_up.data_ptr() == public.self_attn.key_up.data_ptr()
    assert adapted.attention.value_up.data_ptr() == public.self_attn.value_up.data_ptr()


def test_public_indexer_schedule_keeps_draft_source_layers_full(harness):
    path = harness.root / "vllm_ascend/models/hy_v4.py"
    tree = ast.parse(path.read_text())
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "compute_skip_topk_layers"
    )
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function],
        type_ignores=[],
    )
    namespace = {}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    indexer_types = ["full", "full"] + ["shared", "shared", "shared", "full"] * 19
    config = SimpleNamespace(num_hidden_layers=78, indexer_types=indexer_types)
    skipped = namespace["compute_skip_topk_layers"](config)
    assert {2, 3, 4, 6, 7, 8}.issubset(skipped)
    assert not skipped.intersection({61, 65, 69, 73, 77})


def test_fusion_only_slot_zero_and_five_distinct_contexts(harness):
    p = make_proposer(harness)
    seen = []
    for layer in p.model.layers:
        layer.register_forward_pre_hook(lambda module, args: seen.append((args[0].clone(), args[2].layer_id)))
    propose(p)
    inputs = p.model.embed_tokens(p.input_ids[:16].long()).view(2, 8, 4)
    torch.testing.assert_close(seen[0][0][:, 1:], inputs[:, 1:])
    torch.testing.assert_close(seen[0][0][:, 0], p.model.fusion(inputs[:, 0], p._target_final_hidden))
    assert [layer for _, layer in seen] == [61, 65, 69, 73, 77]
    for layer in p.model.layers:
        torch.testing.assert_close(layer.attention.rotary.seen_positions, p.mrope_positions[:, :16].view(3, 2, 8))


@torch.inference_mode()
def test_serial_correction_uses_actual_predecessor(harness):
    p = make_proposer(harness)
    previous_embeddings, backbone_calls, corrected_seen = [], [], []
    p.model.register_forward_hook(lambda *args: backbone_calls.append(True))
    p.model.correction.register_forward_pre_hook(lambda module, args: previous_embeddings.append(args[1].clone()))
    hook = p.model.confidence_head.register_forward_pre_hook(
        lambda module, args: corrected_seen.append(args[0].clone())
    )
    ids = propose(p)
    hook.remove()
    assert ids.shape == (2, 7) and len(backbone_calls) == 1 and len(corrected_seen) == 7
    torch.testing.assert_close(previous_embeddings[0], p.model.embed_tokens(torch.tensor([3, 4])))
    for step in range(1, 7):
        torch.testing.assert_close(previous_embeddings[step], p.model.embed_tokens(ids[:, step - 1]))
    for step, hidden in enumerate(corrected_seen):
        torch.testing.assert_close(p.confidence_logits[:, step], p.model.confidence_head(hidden).squeeze(-1))
    assert p.verification_probs() is None


def test_confidence_does_not_change_tokens_or_q(harness):
    p = make_proposer(harness)
    before = propose(p)
    confidence = p.confidence_logits.clone()
    with torch.no_grad():
        p.model.confidence_head.bias.add_(100)
    after = propose(p)
    assert torch.equal(before, after)
    torch.testing.assert_close(p.confidence_logits, confidence + 100)
    assert p.verification_probs() is None


def test_first_pass_rejections_logical_positions_and_read_only_cache(harness):
    p = make_proposer(harness)
    original = [cache.clone() for cache in p._target_model.caches]
    cad = metadata()
    count, indices, query, extra = p.set_inputs_first_pass(
        None, torch.tensor([3, 4]), torch.full((3, 2), 999), None, None, cad, torch.tensor([1, 2])
    )
    assert count == 16 and extra is None and query.causal
    assert indices.tolist() == list(range(1, 8)) + list(range(9, 16))
    assert query.seq_lens.tolist() == [12, 15]
    assert cad.seq_lens.tolist() == [5, 9] and cad.slot_mapping.tolist() == [888]
    assert query.slot_mapping.tolist() == [-1] * 16
    assert p._target_model.calls[0][1].tolist() == [4, 7]
    assert p.mrope_positions[:, 0].tolist() == [14, 24, 34]
    assert p.input_ids[:16].view(2, 8)[:, 1:].eq(120009).all()
    p._run_merged_draft(16, 2, indices, None, None, [], 16)
    for before, after in zip(original, p._target_model.caches):
        torch.testing.assert_close(before, after)
    assert not hasattr(p, "_context_slot_mapping_buffer")
    assert not p.eagle3_use_aux_hidden_state
    assert isinstance(p, harness.parent)


def test_main_model_live_kv_bridge_uses_sample_rows_and_layer_groups(harness):
    p = make_proposer(harness)
    layer_ids = p.target_layer_ids

    class BlockTable:
        def __init__(self, value):
            self.value = value

        def get_device_tensor(self):
            return self.value

    class MainTarget:
        def __init__(self):
            self.attentions = {}
            self.topk = {}
            for layer_id in layer_ids:
                pages = [torch.randn(8, 4, 1, 3), torch.randn(8, 4, 1, 6)]
                self.attentions[layer_id] = SimpleNamespace(
                    mla_attn=SimpleNamespace(mla_attn=SimpleNamespace(kv_cache=pages))
                )
                self.topk[layer_id] = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])

        def get_hy_dflash_attention(self, layer_id):
            return self.attentions[layer_id]

        def get_hy_dflash_topk_buffer(self, layer_id):
            return self.topk[layer_id]

    p._target_model = MainTarget()
    p.target_kv_cache_group_ids = tuple(range(len(layer_ids)))
    page_table = torch.stack((torch.arange(4), torch.arange(4, 8)))
    p.runner.input_batch.block_table = {idx: BlockTable(page_table) for idx in range(len(layer_ids))}
    hidden = torch.arange(8).view(2, 4).float()
    positions = torch.tensor([[4, 8], [14, 18], [24, 28]])
    p.set_inputs_first_pass(None, torch.tensor([3, 4]), positions, hidden, None, metadata(), None)
    torch.testing.assert_close(p._target_final_hidden, hidden)
    assert p._contexts[0].kv.shape[-1] == 9
    assert p.mrope_positions[:, 0].tolist() == [5, 15, 25]
    assert p.mrope_positions[:, 8].tolist() == [9, 19, 29]


def test_request_reordering_changes_context_binding(harness):
    p = make_proposer(harness)
    propose(p, (5, 5))
    before = p._contexts[0].kv.clone()
    p.runner.input_batch.req_ids = ["b", "a"]
    propose(p, (5, 5))
    torch.testing.assert_close(p._contexts[0].kv, before.flip(0))
    assert p.confidence_request_ids == ("b", "a")


def test_dummy_uses_masked_context_and_no_live_target(harness):
    p = make_proposer(harness)
    p.dummy_run(16, num_reqs=2, is_profile=True)
    assert not p._target_model.calls
    assert p.confidence_logits is None and not p.confidence_request_ids
    assert len(p._contexts) == 5 and p._contexts[0].kv.shape == (2, 2056, 9)
    assert not p._contexts[0].valid.any()
    assert harness.events[-1]["in_profile_run"]
    assert harness.events[-1]["aclgraph_runtime_mode"] is GraphMode.NONE


def test_model_load_bridge_and_target_cache_groups(harness):
    p = make_proposer(harness)
    model, target = p.model, p._target_model
    p._get_model = lambda: SimpleNamespace(to_hy_dflash_adapter=lambda supplied: model if supplied is target else None)
    p.load_model(target)
    assert p.model is model and p.draft_attn_groups == [] and p._draft_attn_layer_names == set()
    groups = [SimpleNamespace(layer_names=[name]) for name in p.attn_layer_names]
    p.initialize_attn_backend(SimpleNamespace(kv_cache_groups=groups), 4)
    assert p.target_kv_cache_group_ids == (0, 1, 2, 3, 4) and p.kv_cache_gid == 0
    with pytest.raises(KeyError):
        p.initialize_attn_backend(SimpleNamespace(kv_cache_groups=groups[:-1]), 4)


@pytest.mark.parametrize("missing", ["target", "draft"])
def test_missing_vendor_bridge_fails_explicitly(harness, missing):
    p = make_proposer(harness)
    p._get_model = lambda: nn.Identity()
    with pytest.raises(AttributeError):
        p.load_model(nn.Identity() if missing == "target" else p._target_model)


@pytest.mark.parametrize("field,value", [("draft_tensor_parallel_size", 2)])
def test_unsupported_spec_config(harness, field, value):
    cfg = config()
    setattr(cfg.speculative_config, field, value)
    with pytest.raises((ValueError, NotImplementedError)):
        harness.cls(cfg, torch.device("cpu"))


def make_graph_proposer(harness, monkeypatch):
    cfg = config()
    cfg.speculative_config.enforce_eager = False
    cfg.speculative_config.draft_model_config.hf_config.rope_parameters = {"mrope_section": [1, 1, 1]}
    cfg.speculative_config.draft_model_config.hf_config.dflash_config.update(context_topk=4, context_local_window=2)
    cfg.compilation_config = SimpleNamespace(cudagraph_num_of_warmups=0)
    cfg.model_config = SimpleNamespace(enforce_eager=True, dtype=torch.bfloat16)
    p = make_proposer(harness, cfg)
    p.model.correction = p.model.confidence_head = None
    p._initialize_graph_buffers()
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(synchronize=Mock(), empty_cache=Mock(), mem_get_info=lambda: (1000, 2000)),
        raising=False,
    )
    return p


def graph_propose(p, requests, lengths, anchors, rejected=None):
    p.runner.input_batch.req_ids = requests
    common = metadata(lengths)
    common.query_start_loc = torch.arange(len(requests) + 1)
    return p._propose(
        None, None, None, torch.tensor(anchors), None, common, None, None, num_rejected_tokens_gpu=rejected
    )


def test_graph_capture_is_independent_of_target_and_profile_is_eager(harness, monkeypatch):
    p = make_graph_proposer(harness, monkeypatch)
    addresses = [context.kv.data_ptr() for context in p._graph_contexts]
    p.dummy_run(32, num_reqs=4, is_profile=True, aclgraph_runtime_mode=GraphMode.FULL)
    assert harness.events[-1]["aclgraph_runtime_mode"] == GraphMode.NONE
    assert p.use_cuda_graph and p.vllm_config.model_config.enforce_eager
    assert p.capture_model() == 0
    assert p._runnable.captured_batches == [4, 2, 1]
    assert harness.capture_events == [True, False]
    assert p.capture_model() == 0  # No accidental recapture.
    assert harness.capture_events == [True, False]
    assert addresses == [context.kv.data_ptr() for context in p._graph_contexts]


def test_graph_replay_updates_context_and_pads_requests_without_rebinding(harness, monkeypatch):
    p = make_graph_proposer(harness, monkeypatch)
    p.capture_model()
    eager_cfg = config()
    eager_cfg.speculative_config.draft_model_config.hf_config.dflash_config.update(
        context_topk=4, context_local_window=2
    )
    eager = make_proposer(harness, eager_cfg)
    eager.model.correction = eager.model.confidence_head = None
    addresses = [context.kv.data_ptr() for context in p._graph_contexts]
    saved_output = None
    # Exercise the same bucket with different request order, rejected tokens,
    # and a dirty fourth row before shrinking to a padded three-request batch.
    for requests, lengths, anchors, rejected in (
        (["a", "b", "a", "b"], [5, 9, 6, 10], [3, 4, 8, 9], None),
        (["b", "a", "b"], [9, 5, 10], [4, 3, 9], torch.tensor([2, 1, 3])),
        (["a"], [6], [5], None),
        (["b", "a"], [11, 7], [8, 9], None),
    ):
        actual = graph_propose(p, requests, lengths, anchors, rejected)
        expected = graph_propose(eager, requests, lengths, anchors, rejected)
        torch.testing.assert_close(actual, expected)
        batch = len(requests)
        assert actual.shape == (batch, 7)
        assert addresses == [context.kv.data_ptr() for context in p._graph_contexts]
        for context, reference in zip(p._contexts, eager._contexts):
            torch.testing.assert_close(context.kv[:batch], reference.kv)
            assert not context.kv[batch:].any()
            assert not context.valid[batch:].any()
            assert context.indices[batch:].eq(-1).all()
        assert not p._target_final_hidden[batch:].any()
        if saved_output is None:
            saved_output, saved_value = actual, actual.clone()
    torch.testing.assert_close(saved_output, saved_value)
    assert p._runnable.captured_batches == [4, 2, 1]
    assert p.graph_replay_count == 4


def test_graph_refuses_lazy_capture_and_resets_capture_guard_on_failure(harness, monkeypatch):
    p = make_graph_proposer(harness, monkeypatch)
    with pytest.raises(RuntimeError, match="not captured"):
        graph_propose(p, ["a"], [5], [3])
    p.dummy_run = Mock(side_effect=RuntimeError("capture failed"))
    with pytest.raises(RuntimeError, match="capture failed"):
        p.capture_model()
    assert harness.capture_events == [True, False]
    assert not p._graphs_captured


def test_graph_text_axes_checked_before_capture_and_every_replay(harness, monkeypatch):
    p = make_graph_proposer(harness, monkeypatch)
    p._graph_text_positions = True
    with pytest.raises(ValueError, match="mrope_section"):
        graph_propose(p, ["a"], [5], [3])
    provider = p._target_model.get_hy_dflash_context

    def text_context(**kwargs):
        state = provider(**kwargs)
        return replace(state, query_positions=state.query_positions[:1].expand(3, -1, -1))

    p._target_model.get_hy_dflash_context = text_context
    p.runner.input_batch.req_ids = ["a"]
    p.prepare_context(torch.tensor([5]), None, None, None)
    positions = p.build_model_inputs_first_pass(8)["positions"]
    assert positions.ndim == 1
    torch.testing.assert_close(positions, torch.arange(15, 23))
    p._target_model.get_hy_dflash_context = provider
    with pytest.raises(ValueError, match="mrope_section"):
        p.prepare_context(torch.tensor([6]), None, None, None)


def test_graph_auxiliary_heads_and_oversize_batches_rejected(harness, monkeypatch):
    p = make_graph_proposer(harness, monkeypatch)
    with pytest.raises(ValueError, match="capacity"):
        p._graph_batch_size(5)
    p.model.confidence_head = nn.Linear(4, 1)
    with pytest.raises(NotImplementedError, match="auxiliary heads"):
        p._initialize_graph_buffers()


@pytest.mark.parametrize("draft_graph", [False, True])
def test_v1_worker_calls_draft_capture_with_eager_target(harness, monkeypatch, draft_graph):
    p = make_graph_proposer(harness, monkeypatch)
    p.use_cuda_graph = draft_graph
    p.capture_model = Mock(return_value=123)
    runner_path = harness.root / "vllm_ascend/worker/model_runner_v1.py"
    cls = next(
        node
        for node in ast.parse(runner_path.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner"
    )
    methods = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name in ("has_draft_only_graphs", "capture_model")
    ]
    parent_capture = Mock(return_value=456)
    namespace = dict(
        AscendHYDflashProposer=harness.cls,
        _get_gpu_model_runner_module_name=lambda runner: "runner",
        _torch_cuda_wrapper=nullcontext,
        _replace_gpu_model_runner_function_wrapper=lambda name: nullcontext(),
        graph_capture=lambda **kwargs: nullcontext(),
        GPUModelRunner=SimpleNamespace(capture_model=parent_capture),
    )
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(runner_path), "exec"), namespace)
    runner = SimpleNamespace(drafter=p, device="cpu", _freeze_gc=nullcontext)
    runner.has_draft_only_graphs = lambda: namespace["has_draft_only_graphs"](runner)
    runner.capture_model = lambda: namespace["capture_model"](runner)
    worker = SimpleNamespace(model_config=SimpleNamespace(enforce_eager=True), model_runner=runner)
    worker_path = harness.root / "vllm_ascend/worker/worker.py"
    # Execute the actual worker's capture branch without importing NPU drivers.
    condition = next(
        node
        for node in ast.walk(ast.parse(worker_path.read_text()))
        if isinstance(node, ast.If)
        and any(
            isinstance(child, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "npugraph_memory_bytes" for target in child.targets)
            for child in node.body
        )
    )
    local = dict(self=worker, npugraph_memory_bytes=0)
    exec(compile(ast.Module(body=[condition], type_ignores=[]), str(worker_path), "exec"), local)
    assert local["npugraph_memory_bytes"] == (123 if draft_graph else 0)
    assert p.capture_model.call_count == int(draft_graph)
    if not draft_graph:
        assert runner.capture_model() == 456  # Other proposers retain the upstream capture path.


@pytest.mark.parametrize(
    "field",
    [
        "pipeline_parallel_size",
        "data_parallel_size",
        "prefill_context_parallel_size",
        "decode_context_parallel_size",
    ],
)
def test_distributed_not_silently_supported(harness, field):
    cfg = config()
    setattr(cfg.parallel_config, field, 2)
    with pytest.raises(NotImplementedError, match=field):
        harness.cls(cfg, torch.device("cpu"))


def test_lora_rejected_at_initialization(harness):
    cfg = config()
    cfg.lora_config = object()
    with pytest.raises(NotImplementedError, match="LoRA"):
        harness.cls(cfg, torch.device("cpu"))


@pytest.mark.parametrize("bad", ["layers", "positions", "hidden"])
def test_target_contract_shapes(harness, bad):
    p = make_proposer(harness)
    provider = p._target_model.get_hy_dflash_context

    def broken(**kwargs):
        state = provider(**kwargs)
        if bad == "layers":
            return replace(state, layers=state.layers[::-1])
        if bad == "positions":
            return replace(state, query_positions=state.query_positions[0])
        return replace(state, final_hidden=state.final_hidden[:, :3])

    p._target_model.get_hy_dflash_context = broken
    with pytest.raises((ValueError, RuntimeError)):
        propose(p)


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
    monkeypatch.setattr(factory, "AscendDflashProposer", Mock(return_value="regular"))
    monkeypatch.setattr(factory, "AscendHYDflashProposer", Mock(return_value="hy"))
    cfg = config()
    assert factory.get_spec_decode_method("dflash", cfg, "cpu", None) == "hy"
    cfg.speculative_config.draft_model_config.hf_config.dflash_config.pop("ascend_proposer")
    assert factory.get_spec_decode_method("dflash", cfg, "cpu", None) == "regular"


@pytest.mark.parametrize("use_hy", [False, True])
def test_runner_preserves_target_logits_and_never_passes_confidence_as_q(harness, use_hy):
    proposer = make_proposer(harness) if use_hy else object()
    if use_hy:
        proposer.confidence_logits = torch.ones(2, 7)
    logits, spec_metadata = torch.randn(16, 9), object()
    sampling = SimpleNamespace(top_k=None)
    runner = SimpleNamespace(
        drafter=proposer,
        input_batch=SimpleNamespace(sampling_metadata=sampling, update_async_output_token_ids=Mock()),
        rejection_sampler=Mock(return_value="verified"),
    )
    path = harness.root / "vllm_ascend/worker/model_runner_v1.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_sample")
    namespace = dict(AscendHYDflashProposer=harness.cls, lmhead_tp_enable=lambda: False)
    exec(
        compile(
            ast.Module(body=[method], type_ignores=[]), str(path), "exec", flags=__future__.annotations.compiler_flag
        ),
        namespace,
    )
    assert namespace["_sample"](runner, logits, spec_metadata) == "verified"
    runner.rejection_sampler.assert_called_once_with(spec_metadata, None, logits, sampling)


@pytest.mark.parametrize("output", [None, torch.zeros(2, 3, 4), torch.zeros(3, 5)])
def test_lm_head_shape_failure(harness, output):
    p = make_proposer(harness)
    p.model.compute_logits = lambda *args: output
    with pytest.raises((AttributeError, RuntimeError)):
        propose(p)


def test_confidence_head_shape_failure(harness):
    p = make_proposer(harness)
    p.model.confidence_head = nn.Linear(4, 2)
    with pytest.raises(RuntimeError):
        propose(p)


@pytest.mark.parametrize(
    "tokens,layer_ids,topk,window,mask",
    [(3, [2, 7, 11], 4, 2, 15), (1, [9], 2, 0, 17)],
)
def test_configuration_drives_full_forward_and_dummy(harness, tokens, layer_ids, topk, window, mask):
    cfg = config(tokens)
    cfg.speculative_config.draft_model_config.hf_config.dflash_config.update(
        target_layer_ids=layer_ids,
        context_topk=topk,
        context_local_window=window,
        mask_token_id=mask,
    )
    p = make_proposer(harness, cfg)
    calls = []
    p.model.correction.register_forward_pre_hook(lambda *args: calls.append(True))
    ids = propose(p)
    block_size = tokens + 1
    assert ids.shape == (2, tokens) and len(calls) == tokens
    assert p.input_ids[: 2 * block_size].view(2, block_size)[:, 1:].eq(mask).all()
    assert [context.layer_id for context in p._contexts] == layer_ids
    assert all(context.kv.shape == (2, topk + window, 9) for context in p._contexts)
    assert p.mrope_positions[:, : 2 * block_size].view(3, 2, block_size)[0, 0, 0] == 15
    p.dummy_run(2 * block_size, num_reqs=2)
    assert len(calls) == 2 * tokens
    assert len(p._contexts) == len(layer_ids)
    assert all(context.kv.shape[1] == topk + window for context in p._contexts)


def test_configured_layer_mapping_checked_once_on_load(harness):
    p = make_proposer(harness)
    p.model.target_layer_ids = p.target_layer_ids[::-1]
    p._get_model = lambda: p.model
    with pytest.raises(ValueError, match="layer mapping"):
        p.load_model(p._target_model)


@pytest.mark.parametrize("field,value", [("context_topk", 0), ("context_local_window", -1), ("target_layer_ids", [])])
def test_invalid_context_configuration(harness, field, value):
    cfg = config()
    cfg.speculative_config.draft_model_config.hf_config.dflash_config[field] = value
    with pytest.raises(ValueError, match="HY requires"):
        harness.cls(cfg, torch.device("cpu"))


def test_hy_checkpoint_keeps_hidden_kv_and_indexer_maps_distinct(harness):
    c = harness.config_module.HYDflashConfig(
        architectures=["hy4dflymodels"],
        num_hidden_layers=5,
        target_layer_ids=[60, 64, 68, 72, 76],
        target_kv_layers=[61, 65, 69, 73, 77],
        target_index_layers=[61, 65, 69, 73, 77],
        target_init_attn_layers=[61, 65, 69, 73, 77],
    )
    assert c.dflash_config["target_layer_ids"] == [60, 64, 68, 72, 76]
    assert c.dflash_config["target_kv_layers"] == [61, 65, 69, 73, 77]
    cfg = config()
    cfg.speculative_config.draft_model_config.hf_config = c
    p = harness.cls(cfg, torch.device("cpu"))
    assert p.target_layer_ids == p.target_index_layers == (61, 65, 69, 73, 77)


def test_checkpoint_conflicting_layer_maps_fail(harness):
    with pytest.raises(ValueError, match="Conflicting"):
        harness.config_module.HYDflashConfig(
            architectures=["hy4dflymodels"],
            num_hidden_layers=1,
            target_layer_ids=[60],
            dflash_config={"target_layer_ids": [61]},
        )


def test_matched_target_and_draft_tp(harness):
    cfg = config()
    cfg.parallel_config.tensor_parallel_size = 8
    cfg.speculative_config.draft_tensor_parallel_size = 8
    harness.cls(cfg, torch.device("cpu"))


def test_no_heads_batches_all_seven_logits_once(harness):
    p = make_proposer(harness)
    p.model.correction = p.model.confidence_head = None
    calls = []
    del p.model.logits_callback
    p.model.logits_callback = lambda h: calls.append(h.clone()) or torch.arange(9).expand(h.shape[0], -1)
    result = propose(p)
    assert result.shape == (2, 7) and (result == 8).all()
    assert len(calls) == 1 and calls[0].shape == (14, 4)
    assert p.confidence_logits is None and p.verification_probs() is None


def test_bf16_loader_does_not_inherit_or_mutate_target_quantization(harness):
    p = make_proposer(harness)
    target_quant = object()
    target_config = SimpleNamespace(dtype=torch.float16, quantization="ascend")
    p.vllm_config.model_config = target_config
    p.vllm_config.quant_config = target_quant
    p.speculative_config.draft_parallel_config = SimpleNamespace(tensor_parallel_size=1)
    draft_config = p.speculative_config.draft_model_config
    draft_config.quantization = None
    draft_config.dtype = torch.float16
    captured = []
    harness.register("vllm.compilation.backends", set_model_tag=lambda _: nullcontext())
    harness.register("vllm.model_executor.model_loader", get_model=lambda **kw: captured.append(kw) or "model")
    assert p._get_model() == "model"
    loaded = captured[0]["vllm_config"]
    assert loaded.quant_config is None and loaded.model_config.dtype is torch.bfloat16
    assert p.vllm_config.quant_config is target_quant
    assert target_config.dtype is torch.float16 and target_config.quantization == "ascend"
    assert draft_config.dtype is torch.float16


def test_compacted_rows_index_original_target_buffer(harness):
    p = make_proposer(harness)
    p.target_kv_cache_group_ids = (0,) * 5
    p._target_row_indices = torch.tensor([0, 1, 4, 5, 6])
    p.target_index_layers = (1, 5, 9, 13, 17)
    table = torch.tensor([[0, 1], [2, 3]])
    p.runner.input_batch.block_table = {0: SimpleNamespace(get_device_tensor=lambda: table)}
    pages = [torch.zeros(4, 4, 1, 3), torch.zeros(4, 4, 1, 6)]
    seen_layers = []

    def index_buffer(layer):
        seen_layers.append(layer)
        return torch.arange(7)[:, None]

    p._target_model = SimpleNamespace(
        get_hy_dflash_attention=lambda _: SimpleNamespace(
            mla_attn=SimpleNamespace(mla_attn=SimpleNamespace(kv_cache=pages))
        ),
        get_hy_dflash_topk_buffer=index_buffer,
    )
    rows = torch.tensor([1, 4])
    hidden = torch.randn(5, 4)
    state = p._target_state_from_main_model(torch.tensor([2, 7]), torch.arange(5), hidden, rows)
    assert state.layers[0].topk_indices[:, 0].tolist() == [1, 6]
    torch.testing.assert_close(state.final_hidden, hidden[rows])
    assert seen_layers == [1, 5, 9, 13, 17]


def test_rejected_rows_without_explicit_sample_indices(harness):
    p = make_proposer(harness)
    seen = []
    p.prepare_context = lambda *args: seen.append(args[-1].clone())
    meta = metadata((9, 11))
    meta.query_start_loc = torch.tensor([0, 8, 16])
    p.set_inputs_first_pass(None, torch.tensor([3, 4]), None, None, None, meta, torch.tensor([6, 2]))
    assert seen[0].tolist() == [1, 13]


def test_three_axis_rotary_selects_pairs_without_collapsing(harness):
    positions = torch.tensor([[[1, 2]], [[11, 12]], [[21, 22]]])
    q, k = torch.zeros(2, 2, 6), torch.zeros(2, 1, 6)

    def rotary(pos, query, key):
        return query + pos[:, None, None], key + pos[:, None, None]

    rotated_q, rotated_k = harness.attention.apply_hy_rotary(rotary, positions, q, k, sections=[1, 1, 1])
    assert rotated_q[0, 0].tolist() == [1, 1, 11, 11, 21, 21]
    assert rotated_k[1, 0].tolist() == [2, 2, 12, 12, 22, 22]
    with pytest.raises(ValueError, match="mrope_section"):
        harness.attention.apply_hy_rotary(rotary, positions, q, k)


@pytest.mark.parametrize("padded", [False, True])
def test_sparse_target_sink_matches_explicit_attention(harness, padded):
    torch.manual_seed(72)
    q, qr = torch.randn(3, 2, 3), torch.randn(3, 2, 2)
    cache, rope = torch.randn(4, 2, 1, 3), torch.randn(4, 2, 1, 2)
    table = torch.tensor([[1, 0], [3, 2]])
    indices = torch.tensor([[0, 1, 2, -1], [0, 1, 2, 3], [0, 1, 2, -1]])
    if padded:
        q, qr = torch.cat((q, torch.randn(1, 2, 3))), torch.cat((qr, torch.randn(1, 2, 2)))
        indices = torch.cat((indices, torch.full((1, 4), 9999)))
    sinks, scale = torch.tensor([2.0, -1.0]), 0.4
    actual = harness.attention.sparse_mla_with_sink(q, qr, cache, rope, indices, table, [2, 3], [3, 2], scale, sinks)
    expected = []
    for row, (req, position) in enumerate(((0, 1), (0, 2), (1, 1))):
        keep = indices[row][(indices[row] >= 0) & (indices[row] <= position)]
        keys = torch.stack([cache[table[req, token // 2], token % 2, 0] for token in keep])
        rope_keys = torch.stack([rope[table[req, token // 2], token % 2, 0] for token in keep])
        scores = (q[row] @ keys.T + qr[row] @ rope_keys.T) * scale
        probs = torch.cat((scores, sinks[:, None]), -1).softmax(-1)
        expected.append(probs[:, :-1] @ keys)
    torch.testing.assert_close(actual[:3], torch.stack(expected))
    if padded:
        assert torch.count_nonzero(actual[3]) == 0


def test_draft_sink_changes_denominator(harness):
    args = attention_inputs(harness)
    normal = harness.attention.absorbed_mla(*args)
    with_sink = harness.attention.absorbed_mla(*args, sinks=torch.full((2,), 80.0))
    assert normal.abs().max() > 0.01
    assert with_sink.abs().max() < 1e-20


def test_profile_without_request_count_still_runs_draft(harness):
    p = make_proposer(harness)
    p.model.correction = p.model.confidence_head = None
    p.dummy_run(16, is_profile=True)
    assert len(p._contexts) == 5 and p._contexts[0].kv.shape[0] == 2


@pytest.fixture
def checkpoint_model(harness):
    """Run real HY4 attention construction/loading with CPU linear primitives.

    This does not mock the checkpoint wrapper or its load_weights method.
    Distributed collectives and NPU kernels still require the system test.
    """

    def default_loader(param, weight):
        assert param.shape == weight.shape
        with torch.no_grad():
            param.copy_(weight)

    class Linear(nn.Linear):
        def __init__(self, input_size, output_size, bias=False, **kwargs):
            super().__init__(input_size, output_size, bias=bias)
            self.weight.weight_loader = default_loader

        def forward(self, hidden):
            return super().forward(hidden), None

    class PackedLinear(Linear):
        def __init__(self, input_size, sizes, **kwargs):
            super().__init__(input_size, sum(sizes), **kwargs)
            self.sizes = sizes
            self.weight.weight_loader = self.load_shard

        def load_shard(self, param, weight, shard=None):
            if shard is None:
                default_loader(param, weight)
            else:
                default_loader(param.narrow(0, sum(self.sizes[:shard]), self.sizes[shard]), weight)

    class Embedding(nn.Embedding):
        def __init__(self, vocab, hidden, **kwargs):
            super().__init__(vocab, hidden)
            self.weight.weight_loader = default_loader

    class MLP(nn.Module):
        def __init__(self, hidden, intermediate, *args, **kwargs):
            super().__init__()
            self.gate_up_proj = PackedLinear(hidden, [intermediate, intermediate])
            self.down_proj = Linear(intermediate, hidden)

        def forward(self, hidden):
            gate, up = self.gate_up_proj(hidden)[0].chunk(2, dim=-1)
            return self.down_proj(torch.nn.functional.silu(gate) * up)[0]

    def get_rope(*args, **kwargs):
        return lambda positions, query, key: (query, key)

    # Execute the actual HY attention class. Its draft-only constructor must
    # never touch Indexer, MLAModules or AscendMultiHeadLatentAttention.
    source = harness.root / "vllm_ascend/models/hy_v4.py"
    tree = ast.parse(source.read_text())
    nodes = [
        node for node in tree.body if getattr(node, "name", "") in ("HYV4MLAAttention", "compute_skip_topk_layers")
    ]
    namespace = dict(
        nn=nn,
        torch=torch,
        get_tensor_model_parallel_world_size=lambda: 1,
        DeepSeekV2FusedQkvAProjLinear=PackedLinear,
        RMSNorm=nn.RMSNorm,
        ColumnParallelLinear=Linear,
        RowParallelLinear=Linear,
        get_rope=get_rope,
        _SPARSE_LAYER_TYPES={"sparse_attention"},
    )
    exec(
        compile(
            ast.Module(body=nodes, type_ignores=[]), str(source), "exec", flags=__future__.annotations.compiler_flag
        ),
        namespace,
    )
    harness.register("vllm_ascend.models.hy_v4", HYV4MLAAttention=namespace["HYV4MLAAttention"])
    harness.register("vllm_ascend.models.deepseek_v4", DeepseekV2MLP=MLP)
    harness.register(
        "vllm.distributed", get_tensor_model_parallel_rank=lambda: 0, get_tensor_model_parallel_world_size=lambda: 1
    )
    harness.register("vllm.model_executor.layers.layernorm", RMSNorm=nn.RMSNorm)
    harness.register("vllm.model_executor.layers.linear", ReplicatedLinear=Linear)
    harness.register(
        "vllm.model_executor.layers.vocab_parallel_embedding",
        VocabParallelEmbedding=Embedding,
        ParallelLMHead=Embedding,
    )
    harness.register(
        "vllm.model_executor.layers.logits_processor",
        LogitsProcessor=lambda _: lambda head, hidden: hidden @ head.weight.T,
    )
    harness.register("vllm.model_executor.model_loader.weight_utils", default_weight_loader=default_loader)
    module = harness.load("vllm_ascend.patch.worker.hy_dflash_model", "vllm_ascend/patch/worker/hy_dflash_model.py")
    cfg = harness.config_module.HYDflashConfig(
        architectures=["hy4dflymodels"],
        num_hidden_layers=5,
        target_layer_ids=[60, 64, 68, 72, 76],
        target_kv_layers=[61, 65, 69, 73, 77],
        target_index_layers=[61, 65, 69, 73, 77],
        hidden_size=4,
        intermediate_size=12,
        num_attention_heads=2,
        q_lora_rank=3,
        kv_lora_rank=3,
        qk_nope_head_dim=2,
        qk_rope_head_dim=6,
        v_head_dim=2,
        rms_norm_eps=1e-5,
        vocab_size=19,
    )
    vconfig = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=cfg, dtype=torch.bfloat16), quant_config=None, cache_config=None
    )
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        model = module.HY4DflyModels(vllm_config=vconfig)
    finally:
        torch.set_default_dtype(original_dtype)
    weights = [
        (name, torch.randn_like(param))
        for name, param in model.named_parameters()
        if name not in ("model.embed_tokens.weight", "lm_head.weight")
    ]
    return model, weights


def test_checkpoint_loads_bf16_backbone_and_shares_optional_modules(harness, checkpoint_model):
    model, weights = checkpoint_model
    model.load_weights(iter(weights))
    embed, head = nn.Embedding(19, 4).to(torch.bfloat16), nn.Linear(4, 19).to(torch.bfloat16)
    target = SimpleNamespace(
        config=SimpleNamespace(hidden_size=4, vocab_size=19, num_hidden_layers=78),
        model=SimpleNamespace(embed_tokens=embed),
        lm_head=head,
        compute_logits=head,
        get_hy_dflash_attention=lambda _: SimpleNamespace(kv_lora_rank=3, qk_rope_head_dim=6),
    )
    adapter = model.to_hy_dflash_adapter(target)
    assert adapter.embed_tokens is embed
    assert adapter.correction is None and adapter.confidence_head is None
    assert adapter.target_layer_ids == (61, 65, 69, 73, 77)
    assert all(p.dtype == torch.bfloat16 for p in model.parameters())
    hidden = adapter(
        torch.zeros(1, 8, dtype=torch.long),
        torch.arange(8).expand(3, 1, -1),
        torch.zeros(1, 4, dtype=torch.bfloat16),
        adapter.dummy_contexts(1, 4),
    )
    assert hidden.shape == (1, 8, 4) and torch.isfinite(hidden).all()
    assert adapter.compute_logits(hidden[:, 1:].reshape(-1, 4)).shape == (7, 19)


def test_missing_trained_fusion_gate_is_not_randomly_initialized(checkpoint_model):
    model, weights = checkpoint_model
    with pytest.raises(ValueError, match="model.fusion.gate"):
        model.load_weights((name, weight) for name, weight in weights if name != "model.fusion.gate")


def test_draft_does_not_cast_quantized_weight_bytes_to_bf16(checkpoint_model):
    model, weights = checkpoint_model
    name, value = weights[0]
    with pytest.raises(ValueError, match="unquantized checkpoint"):
        model.load_weights([(name, value.to(torch.float8_e4m3fn))])


def test_missing_half_of_fused_weight_is_detected(checkpoint_model):
    model, weights = checkpoint_model
    name = "model.layers.0.self_attn.fused_qkv_a_proj.weight"
    q_half = dict(weights)[name][:3]
    weights = [(key, value) for key, value in weights if key != name]
    weights.append(("model.layers.0.self_attn.q_a_proj.weight", q_half))
    with pytest.raises(ValueError, match="fused_qkv_a_proj"):
        model.load_weights(iter(weights))


def test_vendor_weight_mapping_must_be_explicit(checkpoint_model):
    model, weights = checkpoint_model
    weights = [("mtp_fusion.channel_gate" if name == "model.fusion.gate" else name, value) for name, value in weights]
    with pytest.raises(ValueError, match="Unknown HY draft weight"):
        model.load_weights(iter(weights))
    model.config.dflash_config["weight_name_mapping"] = {"mtp_fusion.channel_gate": "model.fusion.gate"}
    model.load_weights(iter(weights))
    assert model._weights_loaded


def test_public_hy4_full_indexer_schedule(harness):
    config = harness.config_module.HYV4Config(num_hidden_layers=78)
    assert all(config.indexer_types[i] == "full" for i in (61, 65, 69, 73, 77))
    assert all(config.indexer_types[i] == "shared" for i in (60, 64, 68, 72, 76))


def test_fused_mxfp8_expert_scales_do_not_load_as_weights(harness):
    source = harness.root / "vllm_ascend/models/hy_v4.py"
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendHYV4ForCausalLM")
    method = next(node for node in cls.body if getattr(node, "name", "") == "_load_fused_experts")
    method.decorator_list = []

    namespace = {"typing": typing, "Callable": typing.Callable, "torch": torch}
    exec(
        compile(
            ast.Module(body=[method], type_ignores=[]), str(source), "exec", flags=__future__.annotations.compiler_flag
        ),
        namespace,
    )
    calls = []
    param = SimpleNamespace(
        weight_loader=lambda p, w, name, **kw: calls.append((w.clone(), name, kw["shard_id"])) or True
    )
    params = {"model.layers.1.mlp.experts.w13_weight_scale": param}
    mappings = [("experts.w13_", "experts.0.gate_proj.", 0, "w1"), ("experts.w13_", "experts.0.up_proj.", 0, "w3")]
    scales = torch.arange(12, dtype=torch.uint8).view(1, 4, 3)
    loaded = set()
    assert namespace["_load_fused_experts"](
        "model.layers.1.mlp.experts.gate_up_proj.weight_scale", scales, params, mappings, loaded
    )
    assert loaded == set(params)
    assert len(calls) == 2 and calls[0][2] == "w1" and calls[1][2] == "w3"
    torch.testing.assert_close(calls[0][0], scales[0, :2])
    torch.testing.assert_close(calls[1][0], scales[0, 2:])


def test_checkpoint_inspector_reads_only_header(harness, tmp_path):
    module = harness.load("inspect_hy_checkpoint", "tools/inspect_hy_dflash_checkpoint.py")
    (tmp_path / "config.json").write_text('{"architectures": ["hy4dflymodels"]}')
    header = json.dumps({"model.fusion.gate": {"dtype": "BF16", "shape": [6144], "data_offsets": [0, 12288]}}).encode()
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header)
    report = module.inspect_checkpoint(tmp_path)
    assert report["tensors"]["model.fusion.gate"]["shape"] == [6144]
    assert report["config"]["architectures"] == ["hy4dflymodels"]
