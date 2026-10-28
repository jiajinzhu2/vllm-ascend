# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark heads for the vLLM 0.20 linear-layer API.

The upstream implementation uses ParallelLMHead(disable_tp=True), which is
not available in 0.20. ReplicatedLinear preserves replicated Markov weights
without a per-position tensor-parallel gather.
"""

import torch
from torch import nn
from vllm.model_executor.layers.linear import ReplicatedLinear


class DSparkMarkovHead(nn.Module):
    def __init__(
        self, vocab_size: int, draft_vocab_size: int, markov_rank: int, prefix: str
    ) -> None:
        super().__init__()
        if min(vocab_size, draft_vocab_size, markov_rank) <= 0:
            raise ValueError("DSpark vocabulary sizes and Markov rank must be positive")
        self.markov_w1 = nn.Embedding(vocab_size, markov_rank)
        self.markov_w2 = ReplicatedLinear(
            markov_rank,
            draft_vocab_size,
            bias=False,
            return_bias=False,
            quant_config=None,
            prefix=f"{prefix}.markov_w2" if prefix else "markov_w2",
        )

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w1(token_ids)

    def bias(self, markov_embed: torch.Tensor, logits_processor=None) -> torch.Tensor:
        # This projection has an exact vocabulary size, with no TP padding.
        # Preserve LogitsProcessor's optional scaling and soft cap.
        logits = self.markov_w2(markov_embed)
        if logits_processor is not None:
            soft_cap = logits_processor.soft_cap
            if soft_cap is not None:
                logits = torch.tanh(logits / soft_cap) * soft_cap
            logits = logits * logits_processor.scale
        return logits


class DSparkConfidenceHead(nn.Module):
    def __init__(
        self, input_dim: int, prefix: str, bias: bool = False, with_markov: bool = True
    ) -> None:
        super().__init__()
        self.with_markov = with_markov
        self.proj = ReplicatedLinear(
            input_dim,
            1,
            bias=bias,
            return_bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.proj" if prefix else "proj",
        )

    def forward(self, hidden: torch.Tensor, markov_embed: torch.Tensor) -> torch.Tensor:
        x = torch.cat((hidden, markov_embed), dim=-1) if self.with_markov else hidden
        return self.proj(x.float()).squeeze(-1)
