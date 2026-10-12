#!/usr/bin/env python3
"""One-time mechanical relocation of completed MaxFuse benchmark manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
RUNNER = HERE / "_maxfuse_benchmark.py"
BENCHMARKS = ("bone_marrow", "tnbc", "melanoma", "lung_cancer")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replace(value: Any, old: str, new: str) -> Any:
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, list):
        return [replace(item, old, new) for item in value]
    if isinstance(value, dict):
        return {key: replace(item, old, new) for key, item in value.items()}
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def verify_outputs(root: Path, outputs: dict[str, str]) -> None:
    for relative, expected in outputs.items():
        path = root / relative
        if not path.is_file() or sha256(path) != expected:
            raise AssertionError(f"Output hash mismatch: {path}")


def main() -> None:
    runner_hash = sha256(RUNNER)
    repaired = 0
    for benchmark in BENCHMARKS:
        root = HERE / benchmark / "maxfuse"
        config = json.loads((root / "config.json").read_text())
        for spec in config["datasets"]:
            hub = spec["slug"]
            alignment = root / f"hub_{hub}" / "alignment"
            formal_path = alignment / "formal_fit" / "manifest.json"
            alignment_path = alignment / "alignment_manifest.json"
            umap_root = root / f"hub_{hub}" / "umap"
            umap_path = umap_root / "umap_manifest.json"
            for path in (formal_path, alignment_path, umap_path):
                if not path.is_file():
                    raise FileNotFoundError(path)

            formal = json.loads(formal_path.read_text())
            old_input = str(formal["input"])
            marker = "/.alignment.tmp-"
            if marker in old_input:
                prefix, remainder = old_input.split(marker, 1)
                temp_leaf = remainder.split("/", 1)[0]
                old_root = prefix + marker + temp_leaf
                formal = replace(formal, old_root, str(alignment.resolve()))
                write_json(formal_path, formal)
            verify_outputs(alignment / "formal_fit", formal["output_sha256"])

            aligned = json.loads(alignment_path.read_text())
            if marker in json.dumps(aligned):
                command_input = aligned["command"][aligned["command"].index("--input") + 1]
                prefix, remainder = command_input.split(marker, 1)
                temp_leaf = remainder.split("/", 1)[0]
                old_root = prefix + marker + temp_leaf
                aligned = replace(aligned, old_root, str(alignment.resolve()))
            aligned["formal_fit_manifest"] = str(formal_path.resolve())
            aligned["formal_fit_manifest_sha256"] = sha256(formal_path)
            aligned["benchmark_runner"] = str(RUNNER.resolve())
            aligned["benchmark_runner_sha256"] = runner_hash
            verify_outputs(alignment, aligned["outputs"])
            write_json(alignment_path, aligned)

            umap = json.loads(umap_path.read_text())
            umap["alignment_manifest"] = str(alignment_path.resolve())
            umap["alignment_manifest_sha256"] = sha256(alignment_path)
            umap["benchmark_runner"] = str(RUNNER.resolve())
            umap["benchmark_runner_sha256"] = runner_hash
            verify_outputs(umap_root, umap["outputs"])
            write_json(umap_path, umap)
            repaired += 1
    print(f"Repaired and verified {repaired} MaxFuse hub manifest chains")


if __name__ == "__main__":
    main()
