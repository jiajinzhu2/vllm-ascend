# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Explicit A5 SFA precision/replay smoke test, without private model weights.

Run as a script on the serving host; failure is fatal, never a fallback/skip:
python tests/e2e/singlecard/test_hy_dflash_sfa.py --heads 16
"""

import argparse
import json
import time

import torch


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heads", type=int, default=16, help="Local query heads after TP sharding")
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--context-capacity", type=int, default=2056)
    parser.add_argument("--queries", type=int, default=8)
    parser.add_argument("--nope-dim", type=int, default=128)
    parser.add_argument("--value-dim", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()
    if (
        not 1 <= args.heads <= 128
        or min(args.batch_size, args.queries, args.nope_dim, args.value_dim, args.repeats) <= 0
        or args.context_capacity < 0
    ):
        parser.error("Invalid dimensions/repeats")
    # Worker isolation: load NPU/vLLM only for this explicit hardware check.
    import torch_npu

    from vllm_ascend.attention.hy_dflash import HYContext, absorbed_mla, get_hy_sfa_op, sfa_absorbed_mla
    from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

    assert get_ascend_device_type() == AscendDeviceType.A5, "This test requires an A5 build/device"
    torch.npu.set_device(0)
    torch.manual_seed(42)
    op = get_hy_sfa_op(has_sink=True)
    b, q, h, c = args.batch_size, args.queries, args.heads, args.context_capacity

    def random(*shape, scale=1):
        return (torch.randn(*shape) * scale).bfloat16().npu()

    query = random(b, q, h, args.nope_dim)
    query_rope, latent, rope = random(b, q, h, 64), random(b, q, 512), random(b, q, 64)
    ku = random(h, args.nope_dim, 512, scale=512**-0.5)
    vu = random(h, args.value_dim, 512, scale=512**-0.5)
    context = HYContext(
        61,
        torch.arange(c, device="npu").expand(b, -1),
        random(b, c, 576),
        torch.ones((b, c), dtype=torch.bool, device="npu"),
    )
    scale = (args.nope_dim + 64) ** -0.5
    results = []

    for with_sink in (False, True):
        sinks = torch.linspace(-2, 3, h, device="npu") if with_sink else None

        def run(sinks=sinks):
            return sfa_absorbed_mla(query, query_rope, latent, rope, context, ku, vu, scale, sinks, sfa_op=op)

        def reference(sinks=sinks):
            ctx = HYContext(61, context.indices.cpu(), context.kv.cpu(), context.valid.cpu())
            return absorbed_mla(
                query.cpu(),
                query_rope.cpu(),
                latent.cpu(),
                rope.cpu(),
                ctx,
                ku.cpu(),
                vu.cpu(),
                scale,
                None if sinks is None else sinks.cpu(),
            )

        for _ in range(3):
            run()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            captured = run()
        torch.npu.synchronize()
        max_error = 0.0
        for case in range(3):
            # Same addresses/shapes; full, holed, empty and padded contexts.
            if case == 0:
                context.valid.fill_(True)
            elif case == 1:
                context.valid[:, 1::3] = False
                context.valid[-1].zero_()
            else:
                context.valid.zero_()
            context.kv.copy_(context.kv.roll(1, dims=0))
            query.copy_(query.roll(1, dims=0))
            latent.copy_(latent.roll(1, dims=1))
            graph.replay()
            actual = captured.cpu().float()
            expected = reference().float()
            torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.02)
            torch.testing.assert_close(actual, run().cpu().float(), rtol=0.005, atol=0.005)
            max_error = max(max_error, (actual - expected).abs().max().item())
        context.valid.fill_(True)
        graph.replay()
        first_query = captured[:, 0].clone()
        latent[:, 1:] += 10
        rope[:, 1:] -= 10
        graph.replay()
        torch.testing.assert_close(captured[:, 0], first_query, rtol=0, atol=0)
        # Time a normal full-context batch; do not include CPU references/copies.
        latent[:, 1:] -= 10
        rope[:, 1:] += 10
        torch.npu.synchronize()
        started = time.perf_counter()
        for _ in range(args.repeats):
            graph.replay()
        torch.npu.synchronize()
        results.append(
            dict(
                sink=with_sink, max_abs_error=max_error, graph_ms=(time.perf_counter() - started) * 1000 / args.repeats
            )
        )
    print(json.dumps(dict(torch_npu=torch_npu.__version__, dimensions=vars(args), results=results), indent=2))


if __name__ == "__main__":
    main()
