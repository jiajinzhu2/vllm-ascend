# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Worker-owned draft scratch suffixes inside the selected target KV tensors."""

from dataclasses import replace

import torch


class HYDraftCacheReservation:
    """Charge scratch bytes during profiling, then move them into target pools.

    The scheduler's config/num_blocks and prefix-cache manager are untouched.
    Only the runner's private copy of each selected allocation grows. Scratch
    therefore cannot be hashed, transferred as accepted KV, or recycled by the
    scheduler. Each padded request row owns disjoint slots on every layer.
    """

    def __init__(self, specs, names, batch, query_len, device):
        self.plans = {}
        for name in names:
            spec = specs[name]
            extra = batch * ((query_len + spec.block_size - 1) // spec.block_size)
            self.plans[name] = (spec.block_size, spec.page_size_bytes, extra)
        self.prefix_tokens = {}
        self.profile_bytes = sum(page_bytes * extra for _, page_bytes, extra in self.plans.values())
        self.profile_buffer = torch.empty(self.profile_bytes, dtype=torch.uint8, device=device)

    def extend(self, config):
        if self.prefix_tokens:
            raise RuntimeError("HY scratch reservation cannot be applied twice")
        tensors, prefixes = [], {}
        for tensor in config.kv_cache_tensors:
            selected = set(tensor.shared_by) & self.plans.keys()
            if not selected:
                tensors.append(tensor)
                continue
            if len(tensor.shared_by) != 1:
                raise ValueError("HY scratch requires independently allocated source-layer KV tensors")
            name = next(iter(selected))
            block_size, page_bytes, extra = self.plans[name]
            if tensor.size % page_bytes:
                raise ValueError("HY target KV size is not an integral number of pages")
            prefixes[name] = tensor.size // page_bytes * block_size
            tensors.append(replace(tensor, size=tensor.size + extra * page_bytes))
        if prefixes.keys() != self.plans.keys():
            raise ValueError("HY target KV allocation is missing a selected source layer")
        self.profile_buffer = None
        self.prefix_tokens = prefixes
        config.kv_cache_tensors = tensors
