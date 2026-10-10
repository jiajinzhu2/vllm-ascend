# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu import model_runner as upstream
from vllm.v1.worker.gpu.attn_utils import get_attn_cg_support

from vllm_ascend.attention.dspark_fa3 import AscendDSparkFABackend, AscendDSparkFAMetadataBuilder
from vllm_ascend.patch.worker.patch_v2 import patch_model_runner  # noqa: F401


@pytest.mark.parametrize("is_profiling", [False, True])
def test_initialize_preserves_connector_containers_and_flattens_runner_cache(is_profiling):
    k, v, conv, ssm, single = [torch.empty(3, 4) for _ in range(5)]
    other = torch.empty(3, 4, device="meta")
    caches = {"attention": (k, None, v), "mamba": [conv, ssm], "single": single, "other": other}
    runner = MagicMock()
    runner.device = torch.device("cpu")
    runner.is_encoder_decoder = False
    runner.speculator = None
    runner.vocab_size = 16
    runner.max_model_len = 128
    runner.cache_config.kv_sharing_fast_prefill = False
    runner.jit_warmup_registry.activate.side_effect = nullcontext
    config = KVCacheConfig(num_blocks=3, kv_cache_tensors=[], kv_cache_groups=[])
    registration_order = []
    runner._register_sparse_kv_caches.side_effect = lambda _: registration_order.append("offload")

    def register_connector(*_args):
        registration_order.append("connector")

    with (
        patch.object(upstream, "init_attn_backend", return_value=([], MagicMock(), [])),
        patch.object(upstream, "maybe_create_adaptive_verification_manager", return_value=None),
        patch.object(upstream, "BlockTables"),
        patch.object(upstream.pcp, "maybe_build_pcp_manager", return_value=None),
        patch.object(upstream, "maybe_build_ubatch_runner", return_value=None),
        patch.object(upstream, "initialize_mamba_ssu_backend"),
        patch.object(upstream, "has_compiled_submodule", return_value=False),
        patch.object(upstream, "ModelCudaGraphManager"),
        patch.object(upstream, "check_attention_cp_compatibility"),
        patch.object(upstream, "init_kv_cache", return_value=caches),
        patch.object(upstream, "get_kv_connector", side_effect=register_connector) as connector,
    ):
        upstream.GPUModelRunner.initialize_kv_cache(runner, config, is_profiling=is_profiling)

    assert [id(tensor) for tensor in runner.kv_caches] == [id(tensor) for tensor in (k, v, conv, ssm, single)]
    runner._register_sparse_kv_caches.assert_called_once_with(caches)
    if is_profiling:
        assert registration_order == ["offload"]
        connector.assert_not_called()
        assert runner.kv_connector is upstream.NO_OP_KV_CONNECTOR
    else:
        assert registration_order == ["offload", "connector"]
        assert connector.call_args.args[1] is caches
        assert type(caches["attention"]) is tuple
        assert type(caches["mamba"]) is list
        assert caches["attention"][2] is v
        assert caches["mamba"][1] is ssm
        assert caches["other"] is other


def test_mrv2_block_copy_preserves_segmented_mamba_storage():
    storage = torch.arange(36, dtype=torch.float32)
    conv = storage[:12].view(3, 4)
    ssm = storage[12:].view(3, 8)
    before_conv, before_ssm = conv.clone(), ssm.clone()
    upstream.copy_kv_cache_blocks_inplace([conv, ssm, conv, ssm], 3, [KVCacheBlockCopy(src_block_id=0, dst_block_id=2)])
    torch.testing.assert_close(conv[2], before_conv[0])
    torch.testing.assert_close(ssm[2], before_ssm[0])
    torch.testing.assert_close(conv[:2], before_conv[:2])
    torch.testing.assert_close(ssm[:2], before_ssm[:2])


@pytest.mark.parametrize("external_fa3", [False, True])
def test_eager_external_dspark_does_not_disable_target_graph(external_fa3):
    target_builder = SimpleNamespace(get_cudagraph_support=lambda *_: AttentionCGSupport.ALWAYS)
    groups = [
        [
            SimpleNamespace(
                layer_names=["target"],
                kv_cache_spec=None,
                backend=type("TargetMSA", (), {}),
                get_metadata_builder=lambda _: target_builder,
            )
        ],
        [
            SimpleNamespace(
                layer_names=["draft"],
                kv_cache_spec=None,
                backend=AscendDSparkFABackend,
                get_metadata_builder=lambda _: AscendDSparkFAMetadataBuilder,
            )
        ],
    ]
    runner = MagicMock()
    runner.device = torch.device("cpu")
    runner.is_encoder_decoder = False
    runner.speculator = MagicMock(spec=upstream.DraftModelSpeculator)
    runner.speculator.draft_attn_layer_names = {"draft"}
    runner.vocab_size = 16
    runner.vllm_config.speculative_config = SimpleNamespace(
        method="dspark",
        enforce_eager=True,
        attention_backend=AttentionBackendEnum.FLASH_ATTN if external_fa3 else None,
    )
    runner.cache_config.kv_sharing_fast_prefill = False
    runner.jit_warmup_registry.activate.side_effect = nullcontext
    cache_group = SimpleNamespace(layer_names=["target", "draft"], kv_cache_spec=MagicMock())
    runner.model_state.get_additional_cg_support.return_value = (AttentionCGSupport.ALWAYS, None)
    cache_group.kv_cache_spec.max_num_blocks_per_req.return_value = 1
    config = KVCacheConfig(num_blocks=3, kv_cache_tensors=[], kv_cache_groups=[cache_group])
    combined_support = get_attn_cg_support(groups, runner.vllm_config)
    assert combined_support.min_cg_support == AttentionCGSupport.NEVER
    with (
        patch.object(upstream, "get_block_table_width", return_value=1),
        patch.object(upstream, "init_attn_backend", return_value=(groups, combined_support, [128])),
        patch.object(upstream, "maybe_create_adaptive_verification_manager", return_value=None),
        patch.object(upstream, "BlockTables"),
        patch.object(upstream.pcp, "maybe_build_pcp_manager", return_value=None),
        patch.object(upstream, "maybe_build_ubatch_runner", return_value=None),
        patch.object(upstream, "initialize_mamba_ssu_backend"),
        patch.object(upstream, "has_compiled_submodule", return_value=False),
        patch.object(upstream, "ModelCudaGraphManager"),
        patch.object(upstream, "check_attention_cp_compatibility"),
        patch.object(upstream, "init_kv_cache", return_value={}),
        patch.object(upstream, "get_kv_connector"),
    ):
        upstream.GPUModelRunner.initialize_kv_cache(runner, config)
    resolved_support = runner.compilation_config.resolve_cudagraph_mode_and_sizes.call_args.args[0]
    assert resolved_support == (AttentionCGSupport.ALWAYS if external_fa3 else AttentionCGSupport.NEVER)
