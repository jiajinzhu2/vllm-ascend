# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the backported heads with real vLLM linear layers on CPU/Gloo."""

import gc
from types import SimpleNamespace

import pytest
import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.platforms import current_platform

from vllm_ascend.models.deepseek_v41.dspark_heads import (
    DSparkConfidenceHead,
    DSparkMarkovHead,
)


@pytest.fixture(scope="module", autouse=True)
def cpu_parallel_context(tmp_path_factory):
    if not current_platform.is_cpu():
        pytest.skip("Real linear-layer integration uses the CPU/Gloo runtime")
    if torch.distributed.is_initialized():
        pytest.skip("This test owns a standalone single-rank process group")
    rendezvous = tmp_path_factory.mktemp("dspark-head-gloo") / "rendezvous"
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=rendezvous.as_uri(),
            local_rank=0,
            backend="gloo",
        )
        initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            backend="gloo",
        )
        try:
            yield
        finally:
            gc.collect()
            destroy_model_parallel()
            destroy_distributed_environment()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("soft_cap", [None, 2.0])
def test_markov_real_linear(dtype, soft_cap):
    torch.manual_seed(17)
    head = DSparkMarkovHead(13, 11, 3, "draft").to(dtype)
    w1 = torch.randn(13, 3, dtype=dtype)
    w2 = torch.randn(11, 3, dtype=dtype)
    with torch.no_grad():
        head.markov_w1.weight.copy_(w1)
        head.markov_w2.weight.copy_(w2)
    head.markov_w2.quant_method.process_weights_after_loading(head.markov_w2)
    ids = torch.tensor([0, 12, 3])
    processor = SimpleNamespace(soft_cap=soft_cap, scale=0.75)
    expected = w1[ids] @ w2.T
    if soft_cap is not None:
        expected = torch.tanh(expected / soft_cap) * soft_cap
    expected *= processor.scale
    actual = head.bias(head.embed(ids), processor)
    torch.testing.assert_close(actual, expected)
    assert actual.shape == (3, 11)


@pytest.mark.parametrize("with_markov", [False, True])
def test_confidence_real_linear(with_markov):
    hidden = torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16)
    markov = torch.tensor([[5], [6]], dtype=torch.bfloat16)
    width = 3 if with_markov else 2
    head = DSparkConfidenceHead(width, "draft", with_markov=with_markov)
    weight = torch.arange(width, dtype=torch.float32).reshape(1, width)
    with torch.no_grad():
        head.proj.weight.copy_(weight)
    head.proj.quant_method.process_weights_after_loading(head.proj)
    inputs = torch.cat((hidden, markov), dim=-1) if with_markov else hidden
    torch.testing.assert_close(head(hidden, markov), (inputs.float() @ weight.T).squeeze(-1))
