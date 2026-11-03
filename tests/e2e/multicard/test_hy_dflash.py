# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Explicit NPU checkpoint check, run as a script (not an automatic CI model).

python tests/e2e/multicard/test_hy_dflash.py --target /models/HY4-mxfp8 \
    --draft /models/HY4-draft-bf16 --tp 8

Each engine runs in its own process so the target weights are fully released
between cases. The baseline is this target implementation, not a trusted
external HY4 oracle. Validate target fidelity separately.
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def start_draft_trace(worker):
    """Diagnostic synchronization is intentional here; never use in a benchmark."""
    from types import MethodType

    drafter = worker.model_runner.drafter
    if drafter is None:
        return
    original = drafter._propose
    drafter._hy_original_propose = original
    drafter._hy_validation_trace = []

    def traced(self, *args, **kwargs):
        result = original(*args, **kwargs)
        self._hy_validation_trace.append(result.cpu().tolist())
        return result

    drafter._propose = MethodType(traced, drafter)


def stop_draft_trace(worker):
    drafter = worker.model_runner.drafter
    if drafter is not None:
        drafter._propose = drafter._hy_original_propose


def draft_graph_stats(worker):
    drafter = worker.model_runner.drafter
    if drafter is None:
        return dict(captured=0, replays=0, trace=[])
    entries = getattr(drafter._runnable, "concrete_aclgraph_entries", {})
    return dict(
        captured=sum(entry.aclgraph is not None for entry in entries.values()),
        replays=drafter.graph_replay_count,
        trace=getattr(drafter, "_hy_validation_trace", []),
    )


def run_engine(args):
    from vllm import LLM, SamplingParams

    speculative = None
    if args.worker != "baseline":
        speculative = dict(
            method="dflash",
            model=args.draft,
            num_speculative_tokens=7,
            draft_tensor_parallel_size=args.tp,
            enforce_eager=not args.worker.startswith("graph-"),
            quantization=None,
            disable_padded_drafter_batch=args.worker.endswith("compacted"),
        )
    llm = LLM(
        model=args.target,
        quantization="ascend",
        dtype="bfloat16",
        kv_cache_dtype="auto",
        tensor_parallel_size=args.tp,
        enforce_eager=True,
        async_scheduling=False,
        max_model_len=4096,
        max_num_seqs=4,  # Three live requests pad to 4, then shrink as requests finish.
        max_num_batched_tokens=256,
        speculative_config=speculative,
        seed=42,
    )
    prompts = [
        "请解释为什么月球有阴晴圆缺。",
        "Continue the sequence and explain the rule: 1, 1, 2, 3, 5, 8,",
        "The code for the blue door is 7139. "
        + "A green door is locked. " * 350
        + " What is the code for the blue door?",
    ]
    if args.worker != "baseline":
        llm.collective_rpc(start_draft_trace)
    params = [SamplingParams(temperature=0, max_tokens=n) for n in (16, 31, 64)]
    token_ids = []
    started = time.perf_counter()
    for inputs in (prompts, list(reversed(prompts))):
        outputs = llm.generate(inputs, params)
        token_ids.extend(list(result.outputs[0].token_ids) for result in outputs)
    stats = llm.collective_rpc(draft_graph_stats)
    if args.worker.startswith("graph-"):
        assert all(item["captured"] == 3 and item["replays"] > 0 for item in stats), stats
    else:
        assert all(item["captured"] == 0 and item["replays"] == 0 for item in stats), stats
    report = dict(tokens=token_ids, stats=stats, diagnostic_seconds=time.perf_counter() - started)
    if args.benchmark_repeats:
        llm.collective_rpc(stop_draft_trace)
        started = time.perf_counter()
        generated = 0
        for _ in range(args.benchmark_repeats):
            outputs = llm.generate(prompts, params)
            generated += sum(len(result.outputs[0].token_ids) for result in outputs)
        elapsed = time.perf_counter() - started
        report["benchmark"] = dict(seconds=elapsed, output_tokens=generated, tokens_per_second=generated / elapsed)
        print(f"{args.worker}: {report['benchmark']}")
    args.output.write_text(json.dumps(report))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument(
        "--benchmark-repeats", type=int, default=0, help="Time additional generations with tracing disabled"
    )
    modes = ("baseline", "padded", "compacted", "graph-padded", "graph-compacted")
    parser.add_argument("--worker", choices=modes)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        run_engine(args)
        return
    with tempfile.TemporaryDirectory(prefix="hy4-dflash-") as directory:
        results = {}
        for mode in modes:
            output = Path(directory) / f"{mode}.json"
            subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "--target",
                    args.target,
                    "--draft",
                    args.draft,
                    "--tp",
                    str(args.tp),
                    "--benchmark-repeats",
                    str(args.benchmark_repeats),
                    "--worker",
                    mode,
                    "--output",
                    str(output),
                ],
                check=True,
            )
            results[mode] = json.loads(output.read_text())
        for mode in modes[1:]:
            if results[mode]["tokens"] != results["baseline"]["tokens"]:
                raise AssertionError(f"HY4 greedy token mismatch for {mode}: {results}")
        for mode in ("padded", "compacted"):
            eager = [item["trace"] for item in results[mode]["stats"]]
            graph = [item["trace"] for item in results[f"graph-{mode}"]["stats"]]
            if eager != graph:
                raise AssertionError(f"HY4 draft proposals differ between eager and graph {mode}")
    print("HY4 target tokens and draft proposals match; FULL graph capture/replay verified on every TP rank.")


if __name__ == "__main__":
    main()
