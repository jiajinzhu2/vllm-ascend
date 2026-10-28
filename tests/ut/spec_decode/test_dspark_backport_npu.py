# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small NPU tensor checks; no model weights or distributed process group."""

import importlib.util
import sys
from dataclasses import fields
from pathlib import Path

import pytest
import torch


@pytest.fixture
def dspark_utils(monkeypatch):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("An NPU is required")
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/spec_decode/dspark_utils.py"
    spec = importlib.util.spec_from_file_location("dspark_npu_check", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("anchor", [True, False])
@pytest.mark.parametrize("rejected", [None, [1, 0]])
def test_dspark_input_tensors_match_cpu(dspark_utils, anchor, rejected):
    args = dict(
        query_start_loc=torch.tensor([0, 3, 5]),
        target_positions=torch.tensor([14, 15, 16, 30, 31]),
        seq_lens=torch.tensor([17, 32]),
        next_token_ids=torch.tensor([0, 1]),
        block_table=torch.tensor([[9, 2, 8, 6], [7, 1, 4, 5]]),
        context_slots=torch.tensor([100, 101, 102, 200, 201]),
        block_size=16,
        num_speculative_tokens=3,
        noise_token_id=7,
        max_model_len=128,
        num_rejected_tokens=None if rejected is None else torch.tensor(rejected),
        sample_from_anchor=anchor,
    )
    expected = dspark_utils.prepare_dspark_inputs(**args)
    device_args = {k: v.npu() if isinstance(v, torch.Tensor) else v for k, v in args.items()}
    actual = dspark_utils.prepare_dspark_inputs(**device_args)
    for field in fields(expected):
        torch.testing.assert_close(getattr(actual, field.name).cpu(), getattr(expected, field.name))


def test_dspark_mapped_markov_chain_on_npu(dspark_utils):
    class Draft:
        def compute_draft_logits(self, hidden):
            return torch.zeros(hidden.shape[0], 3, device=hidden.device)

        def markov_embed(self, ids):
            return ids // 10

        def markov_bias(self, embed):
            return torch.nn.functional.one_hot((embed + 1) % 3, 3).float() * 10

        def map_draft_to_target(self, ids):
            return ids * 10

    actual = dspark_utils.sample_dspark_greedy(
        Draft(), torch.zeros(6, 2).npu(), torch.tensor([0, 10]).npu(), 3
    )
    torch.testing.assert_close(actual.cpu(), torch.tensor([[10, 20, 0], [20, 0, 10]]))
