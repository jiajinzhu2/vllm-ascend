#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Report unresolved source imports in the staged DeepSeek V4.1 backport.

This is a migration diagnostic. A zero exit status checks source availability
only; it does not validate cache semantics, native binaries, or inference.
"""

import argparse
import ast
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-root", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    roots = {"vllm": args.vllm_root, "vllm_ascend": root}
    paths = list((root / "vllm_ascend/models/deepseek_v41").rglob("*.py"))
    paths.extend(root / name for name in (
        "vllm_ascend/attention/dsa_v41.py",
        "vllm_ascend/attention/context_parallel/dsa_v41_cp.py",
        "vllm_ascend/core/kv_cache_interface.py",
        "vllm_ascend/spec_decode/dspark_proposer.py",
    ))
    issues = []
    for path in paths:
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.level or not node.module:
                continue
            package = node.module.split(".")[0]
            if package not in roots:
                continue
            target = roots[package].joinpath(*node.module.split("."))
            module_file = target.with_suffix(".py")
            if not module_file.is_file():
                module_file = target / "__init__.py"
            if not module_file.is_file():
                issues.append({
                    "path": str(path.relative_to(root)),
                    "line": node.lineno,
                    "module": node.module,
                    "reason": "missing source module",
                })
                continue
            # Attribute exports can also be dynamic; report missing literal
            # declarations as review items rather than claiming import failure.
            target_tree = ast.parse(module_file.read_text())
            declared = set()
            for item in ast.walk(target_tree):
                if isinstance(item, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    declared.add(item.name)
                elif isinstance(item, (ast.Import, ast.ImportFrom)):
                    declared.update(alias.asname or alias.name.split(".")[0] for alias in item.names)
                elif isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store):
                    declared.add(item.id)
            for alias in node.names:
                if alias.name not in declared and alias.name != "*":
                    issues.append({
                        "path": str(path.relative_to(root)),
                        "line": node.lineno,
                        "module": node.module,
                        "symbol": alias.name,
                        "reason": "export requires compatibility review",
                    })
    print(json.dumps({
        "checked_files": len(paths),
        "source_review_items": issues,
        "integration_ready": False,
        "note": "Source inspection only; runner/cache/native/inference gates remain separate.",
    }, indent=2))
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
