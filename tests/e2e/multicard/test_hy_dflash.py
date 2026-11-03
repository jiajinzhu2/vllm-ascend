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
from pathlib import Path


def run_engine(args):
    from vllm import LLM, SamplingParams

    speculative = None
    if args.worker != "baseline":
        speculative = dict(
            method="dflash",
            model=args.draft,
            num_speculative_tokens=7,
            draft_tensor_parallel_size=args.tp,
            enforce_eager=True,
            quantization=None,
            disable_padded_drafter_batch=args.worker == "compacted",
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
        max_num_seqs=2,
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
    outputs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=64))
    args.output.write_text(json.dumps([list(result.outputs[0].token_ids) for result in outputs]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--worker", choices=("baseline", "padded", "compacted"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        run_engine(args)
        return
    with tempfile.TemporaryDirectory(prefix="hy4-dflash-") as directory:
        results = {}
        for mode in ("baseline", "padded", "compacted"):
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
                    "--worker",
                    mode,
                    "--output",
                    str(output),
                ],
                check=True,
            )
            results[mode] = json.loads(output.read_text())
        for mode in ("padded", "compacted"):
            if results[mode] != results["baseline"]:
                raise AssertionError(f"HY4 greedy token mismatch for {mode}: {results}")
    print("HY4 greedy tokens match for baseline, padded draft and compacted draft.")


if __name__ == "__main__":
    main()
