"""Explicit create-only snapshot of the reviewed specs; never run scientific code."""

import argparse
import json
from pathlib import Path

from inputs import HERE, child, input_paths, inspect_cohort, record, sha256, verify_source, write_new_json


def check_historical_raw(manifest, kind, expected_hash):
    """Check the producing manifest's exact raw.npy field, never free text."""
    fields = {"technical": "inputs", "application_alignment": "output_sha256"}
    if kind not in fields:
        raise ValueError(f"Unknown historical manifest kind: {kind}")
    hashes = manifest.get(fields[kind]) if isinstance(manifest, dict) else None
    if not isinstance(hashes, dict) or hashes.get("raw.npy") != expected_hash:
        raise ValueError(f"Historical {fields[kind]}['raw.npy'] does not match frozen raw bytes")


def freeze(specs_path, source_root, output):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Snapshot already exists: {output}")
    root = Path(source_root).resolve(strict=True)
    resolved_output = output.resolve()
    if resolved_output == root or root in resolved_output.parents:
        raise ValueError("Never write a registry into the source tree")
    specs = json.loads(Path(specs_path).read_text(encoding="utf-8"))
    registry = {
        "schema_version": 1, "scope": specs["scope"],
        "specs_sha256": sha256(specs_path),
        "global_references": [record(root, p) for p in specs["global_references"]],
        "cases": [],
    }
    for spec in specs["cases"]:
        paths = input_paths(spec)
        item = dict(spec)
        item["files"] = {role: record(root, p) for role, p in paths.items()}
        item["observed"] = inspect_cohort({k: child(root, p) for k, p in paths.items()}, spec)
        item["references"] = [record(root, p) for p in spec["references"]]
        # The existing run manifest must already attest to the raw matrix hash.
        historical = json.loads(child(root, paths["source_manifest"]).read_text(encoding="utf-8"))
        check_historical_raw(historical, spec["kind"], item["files"]["raw"]["sha256"])
        registry["cases"].append(item)
    verify_source(registry, root)
    if sha256(specs_path) != registry["specs_sha256"]:
        raise ValueError("Specs changed while snapshotting")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_new_json(output, registry)
    return registry


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specs", type=Path, default=HERE / "specs.json")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    data = freeze(args.specs, args.source_root, args.output)
    print(json.dumps({"output": str(args.output), "cases": len(data["cases"]), "sha256": sha256(args.output)}))


if __name__ == "__main__":
    main()
