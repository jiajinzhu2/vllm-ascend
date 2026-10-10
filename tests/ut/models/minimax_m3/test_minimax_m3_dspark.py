# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import sys
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.models.minimax_m3.minimax_m3 import MiniMaxM3Model, _uses_aux_hidden_states
from vllm_ascend.models.minimax_m3.minimax_m3_vl import _install_fused_allreduce_norm_fallback


@pytest.mark.parametrize("method,enabled", [(None, False), ("eagle3", True), ("dspark", True), ("mtp", False)])
def test_m3_aux_hidden_state_capture_matches_speculation_method(method, enabled, monkeypatch):
    config = SimpleNamespace(speculative_config=None if method is None else SimpleNamespace(method=method))
    model = MiniMaxM3Model.__new__(MiniMaxM3Model)
    torch.nn.Module.__init__(model)
    model._enable_aux_hidden_states = _uses_aux_hidden_states(config)
    monkeypatch.setattr(model, "_cache_aux_pp_layout", lambda: None)
    model.set_aux_hidden_state_layers((1, 3))
    hidden = torch.randn(2, 8)
    residual = torch.randn_like(hidden)
    captured = []
    model._maybe_add_hidden_state(captured, 1, hidden, residual)
    if enabled:
        assert model.aux_hidden_state_layers == (1, 3)
        assert len(captured) == 1
        torch.testing.assert_close(captured[0], hidden + residual)
        # Later in-place residual changes must not corrupt the draft input.
        expected = captured[0].clone()
        residual.add_(10)
        torch.testing.assert_close(captured[0], expected)
    else:
        assert model.aux_hidden_state_layers == ()
        assert captured == []


def test_m3_fallback_exports_the_dspark_common_import_contract(monkeypatch):
    module_name = "vllm.model_executor.layers.fused_allreduce_gemma_rms_norm"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    _install_fused_allreduce_norm_fallback()
    fallback = sys.modules[module_name]
    assert callable(fallback.fused_allreduce_gemma_rms_norm)
    assert fallback.flashinfer_trtllm_fused_allreduce_norm is None
    assert fallback._can_use_flashinfer(None, 8) == (False, 0)
    assert fallback._AR_RESIDUAL_RMS_NORM is None
