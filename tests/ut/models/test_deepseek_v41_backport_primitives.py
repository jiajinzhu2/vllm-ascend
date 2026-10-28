# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for the backported checkpoint and head primitives.

Load the owned modules independently of the unfinished V4.1 runner/cache port.
The linear stand-in tests head mathematics, not vLLM/NPU integration.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

_ROOT = Path(__file__).resolve().parents[3]
_MODEL = _ROOT / "vllm_ascend/models/deepseek_v41"


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, _MODEL / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


weights = _load("dspark_backport_weights", "dspark_weights.py")


@pytest.mark.parametrize(
    "name, expected",
    [
        ("mtp.0.embed.weight", "model.embed_tokens.weight"),
        ("mtp.2.head.weight", "lm_head.weight"),
        ("mtp.0.main_proj.weight", "model.layers.40.main_proj.weight"),
        ("mtp.1.main_norm.weight", "model.layers.40.main_norm.weight"),
        ("mtp.0.norm.weight", "model.layers.42.norm.weight"),
        ("mtp.2.markov_head.embed.weight", "model.layers.42.markov_head.markov_w1.weight"),
        ("mtp.2.markov_head.head.weight", "model.layers.42.markov_head.markov_w2.weight"),
        ("mtp.2.confidence_head.weight", "model.confidence_head.proj.weight"),
        ("mtp.1.attn.wq_a.weight", "model.layers.41.self_attn.wq_a.weight"),
        ("mtp.1.ffn.w1.weight", "model.layers.41.mlp.gate_proj.weight"),
        ("mtp.1.ffn.w2.weight", "model.layers.41.mlp.down_proj.weight"),
        ("mtp.1.ffn.w3.weight", "model.layers.41.mlp.up_proj.weight"),
        ("mtp.1.ffn.gate.bias", "model.layers.41.mlp.gate.e_score_correction_bias"),
        ("mtp.1.attn_norm.weight", "model.layers.41.input_layernorm.weight"),
        ("mtp.1.ffn_norm.weight", "model.layers.41.post_attention_layernorm.weight"),
        ("mtp.1.hc_head_fn", "model.hc_head_fn"),
        ("model.layers.0.attn.wq_a.weight", None),
        ("mtp.invalid.head.weight", None),
    ],
)
def test_checkpoint_names(name, expected):
    assert weights.remap_dspark_weight(name, 40, 3) == expected


@pytest.mark.parametrize("first, count, name", [(-1, 3, "mtp.0.x"), (40, 0, "mtp.0.x"), (40, 3, "mtp.3.x")])
def test_invalid_layer_ranges_fail_explicitly(first, count, name):
    with pytest.raises(ValueError):
        weights.remap_dspark_weight(name, first, count)


class _ReplicatedLinear(nn.Linear):
    def __init__(self, input_size, output_size, *, bias, return_bias, params_dtype=None, quant_config=None, prefix=""):
        assert return_bias is False
        assert quant_config is None
        super().__init__(input_size, output_size, bias=bias, dtype=params_dtype)


@pytest.fixture
def heads():
    linear = ModuleType("vllm.model_executor.layers.linear")
    linear.ReplicatedLinear = _ReplicatedLinear
    with patch.dict(sys.modules, {"vllm.model_executor.layers.linear": linear}):
        return _load("dspark_backport_heads", "dspark_heads.py")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("soft_cap", [None, 2.0])
def test_markov_projection_matches_dense_transition(heads, dtype, soft_cap):
    torch.manual_seed(9)
    head = heads.DSparkMarkovHead(13, 11, 3, "draft").to(dtype)
    ids = torch.tensor([0, 12, 3])
    processor = SimpleNamespace(soft_cap=soft_cap, scale=0.75)
    expected = head.markov_w1.weight[ids] @ head.markov_w2.weight.T
    if soft_cap is not None:
        expected = torch.tanh(expected / soft_cap) * soft_cap
    expected *= processor.scale
    torch.testing.assert_close(head.bias(head.embed(ids), processor), expected)
    assert head.bias(head.embed(ids)).shape == (3, 11)
    assert set(head.state_dict()) == {"markov_w1.weight", "markov_w2.weight"}


@pytest.mark.parametrize("with_markov", [False, True])
def test_confidence_projection_uses_fp32(heads, with_markov):
    hidden = torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16)
    markov = torch.tensor([[5], [6]], dtype=torch.bfloat16)
    width = 3 if with_markov else 2
    head = heads.DSparkConfidenceHead(width, "draft", with_markov=with_markov)
    x = torch.cat((hidden, markov), dim=-1) if with_markov else hidden
    expected = torch.nn.functional.linear(x.float(), head.proj.weight).squeeze(-1)
    result = head(hidden, markov)
    assert result.dtype == torch.float32
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize("sizes", [(0, 8, 2), (8, 0, 2), (8, 8, 0)])
def test_invalid_head_dimensions(heads, sizes):
    with pytest.raises(ValueError):
        heads.DSparkMarkovHead(*sizes, prefix="draft")
