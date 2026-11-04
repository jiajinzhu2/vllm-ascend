# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Print checkpoint config and tensor metadata without loading tensor data."""

import argparse
import json
import struct
from pathlib import Path


def inspect_checkpoint(directory: Path) -> dict:
    config = json.loads((directory / "config.json").read_text())
    tensors = {}
    files = sorted(directory.glob("*.safetensors"))
    if not files:
        raise ValueError(f"No safetensors files in {directory}")
    for path in files:
        with path.open("rb") as stream:
            size_bytes = stream.read(8)
            if len(size_bytes) != 8:
                raise ValueError(f"Truncated safetensors header: {path.name}")
            size = struct.unpack("<Q", size_bytes)[0]
            if size > min(path.stat().st_size - 8, 64 * 1024 * 1024):
                raise ValueError(f"Invalid or oversized safetensors header: {path.name}")
            header = json.loads(stream.read(size))
        for name, info in header.items():
            if name == "__metadata__":
                continue
            if name in tensors:
                raise ValueError(f"Duplicate checkpoint tensor: {name}")
            tensors[name] = {"shape": info["shape"], "dtype": info["dtype"], "file": path.name}
    return {"config": config, "tensors": tensors}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.dumps(inspect_checkpoint(args.directory), ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(report)
    else:
        print(report, end="")
