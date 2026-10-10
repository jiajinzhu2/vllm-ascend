# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import sys
from contextlib import nullcontext
from dataclasses import replace
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from vllm.compilation.backends import set_model_tag
from vllm.model_executor.models.qwen3_dflash import _resolve_layer_attention
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.selector import AttentionSelectorConfig, _cached_get_attn_backend
from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

from vllm_ascend.attention import dspark_fa3
from vllm_ascend.attention.attention_v1 import AscendAttentionBackend, AscendAttentionBackendImpl
from vllm_ascend.attention.dspark_fa3 import (
    AscendDSparkFABackend,
    AscendDSparkFAImpl,
    AscendDSparkFAMetadata,
    AscendDSparkFAMetadataBuilder,
)
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.device.hardware_profile import get_hardware_profile
from vllm_ascend.platform import _validate_dspark_fa3_backend
from vllm_ascend.utils import AscendDeviceType
from vllm_ascend.worker.v2 import attn_utils
from vllm_ascend.worker.v2.spec_decode.dspark import speculator as dspark_speculator


class _DeviceOnlyLengths(torch.Tensor):
    def tolist(self):
        raise AssertionError("KV lengths must stay on the device")

    def cpu(self, *args, **kwargs):
        raise AssertionError("KV lengths must stay on the device")

    def item(self):
        raise AssertionError("KV lengths must stay on the device")


def _metadata(*, causal=False):
    # The host upper bound intentionally includes rejected/lookahead tokens.
    valid_lengths = torch.tensor([1027, 2050], dtype=torch.int32).as_subclass(_DeviceOnlyLengths)
    common = AscendCommonAttentionMetadata(
        num_reqs=2,
        num_actual_tokens=4,
        max_query_len=3,
        max_seq_len=3072,
        query_start_loc=torch.tensor([0, 3, 4], dtype=torch.int32),
        query_start_loc_cpu=None,
        seq_lens=valid_lengths,
        seq_lens_cpu=torch.tensor([1034, 2057], dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([1034, 2057], dtype=torch.int32),
        block_table_tensor=torch.zeros(2, 24, dtype=torch.int32),
        slot_mapping=torch.arange(4, dtype=torch.int32),
        causal=causal,
    )
    builder = AscendDSparkFAMetadataBuilder(
        SimpleNamespace(),
        [],
        SimpleNamespace(model_config=SimpleNamespace(runner_type="generate")),
        torch.device("cpu"),
    )
    return builder.build(0, common), common


def _impl(*, sliding_window=None, scale=0.17):
    impl = object.__new__(AscendDSparkFAImpl)
    impl.num_heads = 4
    impl.num_kv_heads = 2
    impl.head_size = 128
    impl.scale = scale
    impl.softcap = 0.0
    impl.sliding_window = sliding_window
    impl.key_cache = torch.zeros(8, 128, 2, 128, dtype=torch.bfloat16)
    impl.value_cache = torch.zeros_like(impl.key_cache)
    impl._get_scheduler_metadata = Mock(return_value=torch.empty(1, dtype=torch.uint8))
    impl._fa3_fn = Mock(side_effect=lambda q, *args, **kwargs: torch.ones_like(q))
    return impl


def test_builder_uses_valid_device_lengths_and_query_maximum():
    metadata, common = _metadata()
    assert metadata.seq_lens_gpu.data_ptr() == common.seq_lens.data_ptr()
    assert metadata.query_start_loc_gpu.data_ptr() == common.query_start_loc.data_ptr()
    assert metadata.seq_lens_cpu is None
    assert metadata.seq_lens_list is None
    assert metadata.max_query_len == 3
    assert metadata.actual_seq_lengths_q is None
    assert metadata.seq_lens is metadata.seq_lens_gpu
    assert metadata.query_start_loc is metadata.query_start_loc_gpu
    assert metadata.attn_mask is None
    assert metadata.causal is False


@pytest.mark.parametrize("writes_output", [False, True])
def test_forward_copies_output_once_and_preserves_padding(writes_output):
    class CountCopies(TorchDispatchMode):
        copies = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func == torch.ops.aten.copy_.default:
                self.copies += 1
            return func(*args, **(kwargs or {}))

    metadata, _ = _metadata()
    impl = _impl(sliding_window=1024)
    layer = SimpleNamespace(layer_name="draft.attn", _k_scale_float=1.0, _v_scale_float=1.0)
    query = torch.zeros(7, 4, 128, dtype=torch.bfloat16)
    output = torch.full_like(query, -9)
    if not writes_output:
        # Backends returning a separate result still need the parent's copy.
        result = torch.full_like(query, -9)
        result[:4] = 1
        impl.forward_impl = Mock(return_value=result)
    with CountCopies() as counter:
        assert impl.forward(layer, query, None, None, None, metadata, output=output) is output
    assert counter.copies == 1
    torch.testing.assert_close(output[:4], torch.ones_like(output[:4]))
    torch.testing.assert_close(output[4:], torch.full_like(output[4:], -9))


@pytest.mark.parametrize("context_length", [0, 1, 1015, 2045])
def test_public_m3_dspark_window_matches_single_anchor_mask(context_length):
    # nvidia/MiniMax-M3-DSpark config at e82db0e1895bc4e0c339ce670b2b553899a57f59.
    # Model Optimizer's generation mask windows context per query, while keeping
    # the current 8-token anchor block bidirectional. Multiple training anchors
    # are deliberately outside this single-block inference contract.
    config = SimpleNamespace(
        layer_types=["sliding_attention"] * 6,
        sliding_window=1024,
        dflash_config={"use_swa": True, "swa_window_size": 1024, "causal": False},
    )
    assert [_resolve_layer_attention(config, idx) for idx in range(6)] == [(1024, False)] * 6
    sliding_window, causal = _resolve_layer_attention(config, 0)
    block_length = 8
    page_size = 128
    kv_length = context_length + block_length
    metadata = AscendDSparkFAMetadata(
        num_actual_tokens=block_length,
        max_query_len=block_length,
        query_start_loc_gpu=torch.tensor([0, block_length], dtype=torch.int32),
        seq_lens_gpu=torch.tensor([kv_length], dtype=torch.int32),
        block_tables=torch.zeros(1, (kv_length + page_size - 1) // page_size, dtype=torch.int32),
        causal=causal,
    )
    impl = _impl(sliding_window=sliding_window)
    query = torch.zeros(block_length, impl.num_heads, impl.head_size, dtype=torch.bfloat16)
    impl.forward_impl(query, None, None, None, metadata, torch.empty_like(query))
    params = impl._fa3_fn.call_args.kwargs
    assert params["causal"] is False
    left, right = params["window_size"]
    q_positions = context_length + torch.arange(block_length)[:, None]
    kv_positions = torch.arange(kv_length)[None, :]
    operator_mask = (kv_positions >= q_positions - left) & (kv_positions <= q_positions + right)
    context_mask = (kv_positions < context_length) & (kv_positions > q_positions - sliding_window)
    anchor_mask = kv_positions >= context_length
    torch.testing.assert_close(operator_mask, context_mask | anchor_mask)
    assert operator_mask[:, context_length:].all()


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("sliding_window", [None, 1024])
def test_forward_preserves_group_mask_and_matches_metadata_fingerprint(causal, sliding_window):
    metadata, _ = _metadata(causal=causal)
    impl = _impl(sliding_window=sliding_window)
    query = torch.zeros(7, 4, 128, dtype=torch.bfloat16)
    output = torch.full_like(query, -9)
    assert impl.forward_impl(query, None, None, None, metadata, output) is output
    schedule = impl._get_scheduler_metadata.call_args.kwargs
    forward = impl._fa3_fn.call_args.kwargs
    expected_window = (-1, -1) if sliding_window is None else (1023, 0 if causal else 1023)
    for params in (schedule, forward):
        assert params["causal"] is causal
        assert params["window_size"] == expected_window
        assert params["max_seqlen_q"] == 3
        assert params["softmax_scale"] == 0.17
        assert params["num_splits"] == 1
        assert params["cache_seqlens"] is metadata.seq_lens_gpu
        assert params["cu_seqlens_q"] is metadata.query_start_loc_gpu
    assert schedule["max_seqlen_k"] == 24 * 128
    assert schedule["page_size"] == 128
    assert forward["scheduler_metadata"] is impl._get_scheduler_metadata.return_value
    assert forward["page_table"] is metadata.block_tables
    assert "k" not in forward and "v" not in forward
    assert impl._fa3_fn.call_args.args[0].shape[0] == 4
    torch.testing.assert_close(output[:4], torch.ones_like(output[:4]))
    torch.testing.assert_close(output[4:], torch.full_like(output[4:], -9))


def test_schedule_reuse_is_limited_to_matching_layers_and_current_step():
    metadata, _ = _metadata()
    query = torch.zeros(4, 4, 128, dtype=torch.bfloat16)
    first = _impl(sliding_window=1024)
    first.forward_impl(query, None, None, None, metadata, torch.empty_like(query))
    same = _impl(sliding_window=1024)
    same.forward_impl(query, None, None, None, metadata, torch.empty_like(query))
    same._get_scheduler_metadata.assert_not_called()
    assert same._fa3_fn.call_args.kwargs["scheduler_metadata"] is first._get_scheduler_metadata.return_value
    different_scale = _impl(sliding_window=1024, scale=0.11)
    different_scale.forward_impl(query, None, None, None, metadata, torch.empty_like(query))
    different_scale._get_scheduler_metadata.assert_called_once()
    full = _impl()
    full.forward_impl(query, None, None, None, metadata, torch.empty_like(query))
    full._get_scheduler_metadata.assert_called_once()
    next_step, _ = _metadata()
    first.forward_impl(query, None, None, None, next_step, torch.empty_like(query))
    assert first._get_scheduler_metadata.call_count == 2


def test_zero_query_batch_does_not_launch_attention():
    metadata = AscendDSparkFAMetadata(num_actual_tokens=0)
    impl = _impl()
    output = torch.empty(0, 4, 128)
    assert impl.forward_impl(output, None, None, None, metadata, output) is output
    impl._get_scheduler_metadata.assert_not_called()
    impl._fa3_fn.assert_not_called()


def test_backend_is_gqa_and_does_not_claim_unverified_full_graph_replay():
    assert issubclass(AscendDSparkFABackend, AscendAttentionBackend)
    assert AscendDSparkFABackend.supports_non_causal()
    assert AscendDSparkFABackend.supports_sliding_window()
    assert not AscendDSparkFABackend.supports_pcp()
    assert not AscendDSparkFABackend.supports_dcp()
    assert not AscendDSparkFABackend.supports_kv_cache_dtype("fp8")
    assert AscendDSparkFABackend.supported_kv_cache_layouts() == ("NHD",)
    assert AscendDSparkFAMetadataBuilder.get_cudagraph_support(None, None) == AttentionCGSupport.NEVER


@pytest.fixture
def selection(monkeypatch):
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            method="dspark", enforce_eager=True, attention_backend=AttentionBackendEnum.FLASH_ATTN
        ),
        use_v2_model_runner=True,
    )
    monkeypatch.setattr("vllm.config.get_current_vllm_config", lambda: config)
    monkeypatch.setattr(
        "vllm_ascend.platform.get_current_hardware_profile", lambda: get_hardware_profile(AscendDeviceType.A3)
    )
    monkeypatch.setattr("vllm_ascend.platform.util.find_spec", lambda name: object())
    selector = AttentionSelectorConfig(head_size=128, dtype=torch.bfloat16, kv_cache_dtype="auto", block_size=128)
    return config, selector


def test_selection_is_explicit_and_draft_only(selection):
    config, selector = selection
    with set_model_tag("model"):
        assert not _validate_dspark_fa3_backend(AttentionBackendEnum.FLASH_ATTN, selector)
    with set_model_tag("dspark_head"):
        assert not _validate_dspark_fa3_backend(None, selector)
        assert _validate_dspark_fa3_backend(AttentionBackendEnum.FLASH_ATTN, selector)
        config.speculative_config.attention_backend = None
        assert not _validate_dspark_fa3_backend(AttentionBackendEnum.FLASH_ATTN, selector)


def test_draft_loading_isolates_the_upstream_selector_cache(selection, monkeypatch):
    config, selector = selection
    config.speculative_config.attention_backend = AttentionBackendEnum.FLASH_ATTN

    def select_backend(backend, attn_selector_config, **kwargs):
        if _validate_dspark_fa3_backend(backend, attn_selector_config):
            return "vllm_ascend.attention.dspark_fa3.AscendDSparkFABackend"
        return "vllm_ascend.attention.attention_v1.AscendAttentionBackend"

    monkeypatch.setattr("vllm.platforms.current_platform", SimpleNamespace(get_attn_backend_cls=select_backend))
    monkeypatch.setattr(dspark_speculator, "disable_profiling_chunk_for_draft", lambda _: nullcontext())

    def load_draft(*_args):
        with set_model_tag("dspark_head"):
            return _cached_get_attn_backend(AttentionBackendEnum.FLASH_ATTN, selector)

    monkeypatch.setattr(dspark_speculator.DSparkSpeculator, "load_draft_model", load_draft)
    speculator = object.__new__(dspark_speculator.AscendDSparkSpeculator)
    speculator.vllm_config = config
    speculator.speculative_config = config.speculative_config
    monkeypatch.setattr(speculator, "_lmhead_tp_wrap_draft_logits", lambda _: None)
    _cached_get_attn_backend.cache_clear()
    try:
        with set_model_tag("model"):
            # Identical attention keys, but different model scopes.
            assert _cached_get_attn_backend(AttentionBackendEnum.FLASH_ATTN, selector) is AscendAttentionBackend
            assert speculator.load_draft_model(None, set()) is AscendDSparkFABackend
            assert _cached_get_attn_backend(AttentionBackendEnum.FLASH_ATTN, selector) is AscendAttentionBackend
    finally:
        _cached_get_attn_backend.cache_clear()


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"use_mla": True}, "draft GQA"),
        ({"use_sparse": True}, "draft GQA"),
        ({"use_dcp": True}, "context-parallel"),
        ({"use_pcp": True}, "context-parallel"),
        ({"has_sink": True}, "sinks"),
        ({"kv_cache_dtype": "fp8"}, "draft KV cache"),
        ({"dtype": torch.float32}, "queries"),
        ({"head_size": 512}, "head dimensions"),
        ({"use_mm_prefix": True}, "optional sliding window"),
        ({"use_rswa": True}, "optional sliding window"),
    ],
)
def test_selection_rejects_unsupported_inputs(selection, overrides, message):
    _, selector = selection
    with set_model_tag("dspark_head"), pytest.raises(ValueError, match=message):
        _validate_dspark_fa3_backend(AttentionBackendEnum.FLASH_ATTN, selector._replace(**overrides))


def test_selection_requires_v2_a3_and_operator_package(selection, monkeypatch):
    config, selector = selection
    with set_model_tag("dspark_head"):
        config.use_v2_model_runner = False
        with pytest.raises(ValueError, match="ModelRunner V2"):
            _validate_dspark_fa3_backend(AttentionBackendEnum.FLASH_ATTN, selector)
        config.use_v2_model_runner = True
        config.speculative_config.enforce_eager = False
        with pytest.raises(ValueError, match="enforce_eager=true"):
            _validate_dspark_fa3_backend(AttentionBackendEnum.FLASH_ATTN, selector)
        config.speculative_config.enforce_eager = True
        monkeypatch.setattr(
            "vllm_ascend.platform.get_current_hardware_profile", lambda: get_hardware_profile(AscendDeviceType.A5)
        )
        with pytest.raises(ValueError, match="A3 only"):
            _validate_dspark_fa3_backend(AttentionBackendEnum.FLASH_ATTN, selector)
        monkeypatch.setattr(
            "vllm_ascend.platform.get_current_hardware_profile", lambda: get_hardware_profile(AscendDeviceType.A3)
        )
        monkeypatch.setattr("vllm_ascend.platform.util.find_spec", lambda name: None)
        with pytest.raises(ValueError, match="0.4.2.post1"):
            _validate_dspark_fa3_backend(AttentionBackendEnum.FLASH_ATTN, selector)


@pytest.mark.parametrize("windowed", [False, True])
@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("split_allocation", [False, True])
def test_mrv2_cache_views_are_packed_without_copying(windowed, padded, split_allocation, monkeypatch):
    spec_cls = SlidingWindowSpec if windowed else FullAttentionSpec
    spec = spec_cls(
        block_size=128,
        num_kv_heads=2,
        head_size=64,
        dtype=torch.bfloat16,
        **({"sliding_window": 1024} if windowed else {}),
    )
    if padded:
        spec = replace(spec, page_size_padded=spec.page_size_bytes + 128)
    num_blocks = 3
    group = SimpleNamespace(
        layer_names=["draft.attn"], kv_cache_spec=spec, backend=AscendDSparkFABackend, kv_cache_group_id=0
    )
    cache_config = SimpleNamespace(
        num_blocks=num_blocks,
        kv_cache_groups=[SimpleNamespace(layer_names=group.layer_names, kv_cache_spec=spec)],
        kv_cache_tensors=[],
    )
    raw_size = num_blocks * spec.page_size_bytes
    if split_allocation:
        raw = (torch.zeros(raw_size // 2, dtype=torch.int8), torch.zeros(raw_size // 2, dtype=torch.int8))
    else:
        raw = torch.zeros(raw_size, dtype=torch.int8)
    monkeypatch.setattr(attn_utils, "get_current_vllm_config", lambda: SimpleNamespace())
    monkeypatch.setattr(attn_utils, "_is_dsv4_model", lambda _: False)
    monkeypatch.setattr(attn_utils, "enable_sfa", lambda _: False)
    caches = attn_utils._reshape_kv_cache_v2([group], {"draft.attn": raw}, "auto", [128], {}, cache_config)
    key, value = caches["draft.attn"]
    assert key.shape == value.shape == (num_blocks, 128, 2, 64)
    assert key.is_contiguous() and value.is_contiguous()
    backing = raw if split_allocation else (raw, raw)
    assert key.untyped_storage().data_ptr() == backing[0].untyped_storage().data_ptr()
    assert value.untyped_storage().data_ptr() == backing[1].untyped_storage().data_ptr()
    key.fill_(1)
    value.fill_(2)
    torch.testing.assert_close(key, torch.ones_like(key))
    torch.testing.assert_close(value, torch.full_like(value, 2))


def test_forward_rejects_interleaved_page_storage():
    impl = _impl()
    impl.key_cache = torch.empty(8, 2, 128, 2, 128, dtype=torch.bfloat16)[:, 0]
    metadata, _ = _metadata()
    with pytest.raises(ValueError, match="contiguous NHD"):
        impl.forward_impl(torch.zeros(4, 4, 128, dtype=torch.bfloat16), None, None, None, metadata, None)
    impl._get_scheduler_metadata.assert_not_called()


@pytest.mark.parametrize("package_version", ["0.4.2.post1", "0.4.2", "0.4.3"])
def test_operator_loading_requires_the_validated_metadata_abi(package_version, monkeypatch):
    monkeypatch.setattr(
        AscendAttentionBackendImpl, "__init__", lambda self, **kwargs: setattr(self, "use_bnsd_kv_cache", False)
    )
    monkeypatch.setattr(dspark_fa3, "version", lambda name: package_version)
    operator = ModuleType("flash_attn_npu_3")
    operator.flash_attn_with_kvcache = Mock()
    operator.get_scheduler_metadata = Mock()
    monkeypatch.setitem(sys.modules, operator.__name__, operator)
    params = dict(
        num_heads=4,
        head_size=128,
        scale=0.17,
        num_kv_heads=2,
        alibi_slopes=None,
        sliding_window=1024,
        kv_cache_dtype="auto",
        logits_soft_cap=None,
        attn_type="decoder",
    )
    if package_version == "0.4.2.post1":
        impl = AscendDSparkFAImpl(**params)
        assert impl._fa3_fn is operator.flash_attn_with_kvcache
        assert impl._get_scheduler_metadata is operator.get_scheduler_metadata
    else:
        with pytest.raises(ValueError, match="metadata ABI"):
            AscendDSparkFAImpl(**params)
