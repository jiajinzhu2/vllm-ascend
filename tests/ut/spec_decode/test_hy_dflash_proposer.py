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
import sys
from contextlib import contextmanager, nullcontext
from dataclasses import replace
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


@pytest.fixture
def harness(monkeypatch):
    root = Path(__file__).resolve().parents[3]
    events = []

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
    register("vllm.forward_context", get_forward_context=lambda: None)
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
        root=root,
        register=register,
        load=load,
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


@pytest.mark.parametrize("field,value", [("enforce_eager", False), ("draft_tensor_parallel_size", 2)])
def test_unsupported_spec_config(harness, field, value):
    cfg = config()
    setattr(cfg.speculative_config, field, value)
    with pytest.raises((ValueError, NotImplementedError)):
        harness.cls(cfg, torch.device("cpu"))


@pytest.mark.parametrize(
    "field",
    [
        "tensor_parallel_size",
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
