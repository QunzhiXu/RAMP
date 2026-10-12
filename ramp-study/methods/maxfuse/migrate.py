"""Create-only source/config migration; fingerprint existing fits without fitting.

Copies only the local driver and selected text sources. Never copies upstream
MaxFuse, environments, raw data, or numerical benchmark arrays.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
COMMON = HERE.parents[1] / "benchmarks/integration_inputs/registry_20261010.json"
ADAPTER_FILES = [
    "README.md", "PARAMETER_PROVENANCE.md", "LICENSE_NOTICE.md", "pyproject.toml",
    "requirements.txt", "bootstrap_venv.sh", "upstream_lock.json",
    "run_maxfuse_pairwise.py", "run_maxfuse_threeway.py", "run_maxfuse_three_view_extension.py",
    *["maxfuse_repro/" + n for n in [
        "__init__.py", "gcca.py", "pairwise.py", "projection.py", "provenance.py",
        "threeway.py", "types.py", "requirements-lock.txt", "environment-lock.txt", "upstream_lock.json"]],
    *["tests/" + n for n in [
        "smoke_test_core.py", "smoke_test_pairwise_adapter.py", "smoke_test_archive_gcca.py",
        "smoke_test_threeway_logic.py", "validate_archive_gcca_parity.py"]],
]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def relative(root, path):
    value = Path(path)
    if not value.is_absolute():
        value = root / value
    return value.resolve(strict=True).relative_to(root.resolve(strict=True)).as_posix()


def field(obj, keys):
    for key in keys:
        obj = obj[key]
    return obj


def create_json(path, payload):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


class Collector:
    def __init__(self, root):
        self.root = root.resolve(strict=True)
        self.records = {}
        self.links = []

    def add(self, path, role, copy=False):
        rel = relative(self.root, path)
        source = self.root / rel
        if rel not in self.records:
            self.records[rel] = {"source": rel, "bytes": source.stat().st_size,
                                 "sha256": sha(source), "roles": []}
        item = self.records[rel]
        if role not in item["roles"]:
            item["roles"].append(role)
        if copy:
            if source.suffix not in {".py", ".R", ".sh", ".json", ".md", ".txt", ".toml"}:
                raise ValueError(f"Refusing to copy non-text source: {rel}")
            if source.stat().st_size > 5_000_000:
                raise ValueError(f"Oversized source/config: {rel}")
            prefix = "algorithm/maxfuse/"
            item["copy"] = "adapter/" + rel[len(prefix):] if rel.startswith(prefix) else "study_sources/" + rel
        return rel

    def attest(self, manifest, keys, target, role, copy=False):
        """Check exact historical hash field, not a substring or new self-report."""
        mrel = self.add(manifest, "historical_manifest", copy=True)
        rel = self.add(target, role, copy=copy)
        expected = field(json.loads((self.root / mrel).read_text()), keys)
        if expected != self.records[rel]["sha256"]:
            raise ValueError(f"Historical hash mismatch: {mrel} {keys} -> {rel}")
        self.links.append({"manifest": mrel, "keys": keys, "target": rel})
        return rel

    def hash_map(self, manifest, key, base, role):
        mapping = json.loads(Path(manifest).read_text())[key]
        for name in sorted(mapping):
            target = Path(base) / name
            self.attest(manifest, [key, name], target, role,
                        copy=target.suffix in {".json", ".txt"})


def check_cell_order(embedding, ids_path, common_metadata):
    values = np.load(embedding, mmap_mode="r", allow_pickle=False)
    ids = np.load(ids_path, allow_pickle=False)
    with gzip.open(common_metadata, "rt", newline="") as f:
        expected = [r["cell_id"] for r in csv.DictReader(f)]
    if values.ndim != 2 or len(values) != len(expected) or ids.ndim != 1:
        raise ValueError(f"Unexpected embedding/ID shape: {embedding}")
    if len(set(ids.tolist())) != len(ids) or not np.array_equal(ids.astype(str), expected):
        raise ValueError(f"Cell identity/order mismatch: {ids_path}")
    if not np.isfinite(values).all():
        raise ValueError(f"Nonfinite embedding: {embedding}")
    return {"shape": list(values.shape), "dtype": str(values.dtype), "cell_order_matches_common_cohort": True}


def build_catalog(root):
    root = root.resolve(strict=True)
    common = json.loads(COMMON.read_text())
    collector = Collector(root)
    for name in ADAPTER_FILES:
        collector.add(root / "algorithm/maxfuse" / name, "local_adapter", copy=True)
    for name in ["_maxfuse_benchmark.py", "_maxfuse_archive_replay.py", "MAXFUSE_ARCHIVE_CURRENT_AUDIT.md"]:
        collector.add(root / "benchmark_results" / name, "historical_study_runner", copy=True)
    cases = []
    for task in common["cases"]:
        case = task["case"]
        base = root / task["current_maxfuse_directory"]
        metadata = root / task["files"]["evaluation_metadata"]["path"]
        # Verify the trusted cohort sidecar before comparing saved result IDs.
        if sha(metadata) != task["files"]["evaluation_metadata"]["sha256"]:
            raise ValueError(f"Common cohort changed: {case}")
        info = {"case": case, "source_root": task["current_maxfuse_directory"], "runs": [],
                "common_input_raw_sha256": task["files"]["raw"]["sha256"],
                "common_metadata": collector.add(metadata, "common_cohort_identity")}
        if task["kind"] == "technical":
            config_path = base / "config.json"
            config = json.loads(config_path.read_text())
            info["config"] = collector.add(config_path, "case_config", copy=True)
            info["pairwise_profile"] = collector.add(config["pairwise_config"], "pairwise_profile", copy=True)
            info["preprocessing"] = [{"dataset": d["name"], "active_width": len(d["active_markers"]),
                                      "input_protocol": d["input_protocol"]} for d in config["datasets"]]
            info["fit_sampling"] = config["fit_sampling"]
            info["mode"] = "fresh_pairwise_then_archive" if case == "tnbc" else "frozen_pairwise_triplets_then_archive"
            expected_hubs = ["hub_" + d["slug"] for d in config["datasets"]]
            actual_hubs = sorted(p.parent.name for p in base.glob("hub_*/alignment"))
            if actual_hubs != sorted(expected_hubs):
                raise ValueError(f"Missing/extra hubs: {case}")
            if case == "tnbc":
                collector.add(base / "build_inputs.py", "historical_input_builder", copy=True)
            for hub in expected_hubs:
                folder = base / hub / "alignment"
                mp = folder / "alignment_manifest.json"
                m = json.loads(mp.read_text())
                if m["status"] != "complete" or m["labels_used_for_fitting"] is not False:
                    raise ValueError(f"Invalid alignment status/provenance: {mp}")
                collector.attest(mp, ["config_sha256"], config_path, "case_config", copy=True)
                collector.hash_map(mp, "outputs", folder, "current_alignment_output")
                formal = folder / "formal_fit/manifest.json"
                collector.attest(mp, ["formal_fit_manifest_sha256"], formal, "formal_manifest", copy=True)
                collector.hash_map(formal, "output_sha256", formal.parent, "formal_output")
                if case != "tnbc":
                    if m["pairwise_reused_without_refitting"] is not True or m["pairwise_subprocess_invoked"] is not False:
                        raise ValueError("Replay history changed")
                    legacy = m["legacy_pairwise_source"]
                    old = root / relative(root, legacy["alignment"])
                    for key, suffix in [("alignment_manifest_sha256", "alignment_manifest.json"),
                                        ("formal_manifest_sha256", "formal_fit/manifest.json"),
                                        ("fit_input_sha256", "input/fit_input.npz"),
                                        ("triplet_matching_sha256", "formal_fit/triplet_matching.npz")]:
                        collector.attest(mp, ["legacy_pairwise_source", key], old / suffix, "legacy_pairwise_dependency", copy=suffix.endswith(".json"))
                    prepared = root / relative(root, m["prepared_source"]["directory"])
                    collector.attest(mp, ["prepared_source", "manifest_sha256"], prepared / "manifest.json", "prepared_manifest", copy=True)
                    info.setdefault("legacy_sources", []).append(relative(root, old))
                    oldformal = json.loads((old / "formal_fit/manifest.json").read_text())
                    for key in ("pair_config_map",):
                        if key in oldformal:
                            collector.attest(old / "formal_fit/manifest.json", [key + "_sha256"], oldformal[key], "legacy_pair_config", copy=True)
                else:
                    fm = json.loads(formal.read_text())
                    collector.attest(formal, ["input_sha256"], fm["input"], "prepared_fit_input")
                    collector.attest(formal, ["pair_config_map_sha256"], fm["pair_config_map"], "resolved_pair_config", copy=True)
                    prepared = root / relative(root, m["preprocessing_manifest"]).rsplit("/", 1)[0]
                    collector.attest(mp, ["preprocessing_manifest_sha256"], prepared / "manifest.json", "prepared_manifest", copy=True)
                # Prepared full-panel arrays are fingerprinted, not copied.
                collector.hash_map(prepared / "manifest.json", "outputs", prepared, "prepared_full_panel_dependency")
                embedding = folder / "data/maxfuse_embedding_float64.npy"
                ids = folder / "data/cell_ids.npy"
                info["runs"].append({"name": hub, "manifest": relative(root, mp),
                                     "embedding": relative(root, embedding), "cell_ids": relative(root, ids),
                                     **check_cell_order(embedding, ids, metadata)})
        else:
            info["mode"] = "fresh_pairwise" if case != "lung_story" else "fresh_pairwise_then_archive"
            config = json.loads((base / "run_config.json").read_text())
            info["config"] = collector.add(base / "run_config.json", "case_config", copy=True)
            for file in ("prepare.py", "run_maxfuse.py"):
                collector.add(base / "scripts" / file, "historical_story_runner", copy=True)
            manifest_path = base / "data/manifest.json"
            manifest = json.loads(manifest_path.read_text())
            info["preprocessing"] = manifest["views"]
            collector.attest(manifest_path, ["original_input_manifest_sha256"], base.parent / "input/manifest.json", "common_input_manifest", copy=True)
            collector.hash_map(manifest_path, "output_sha256", base / "data", "prepared_full_panel_dependency")
            for name in (["maxfuse_hub0", "maxfuse_hub1", "maxfuse_hub2"] if case == "lung_story" else ["maxfuse"]):
                folder = base / "methods" / name
                mp = folder / "manifest.json"
                m = json.loads(mp.read_text())
                if m["status"] != "complete" or m["labels_supplied_to_maxfuse"] is not False:
                    raise ValueError(f"Invalid formal provenance: {mp}")
                collector.hash_map(mp, "output_sha256", folder, "formal_output")
                collector.attest(mp, ["input_sha256"], m["input"], "prepared_fit_input")
                key = "config_file" if case != "lung_story" else "pair_config_map"
                collector.attest(mp, [key + "_sha256"], m[key], "resolved_pair_config", copy=True)
                complete = folder / "COMPLETE.json"
                done = json.loads(complete.read_text())
                collector.attest(complete, ["values_sha256"], folder / "values.npy", "current_alignment_output")
                collector.attest(complete, ["cell_ids_sha256"], base.parent / "input/cell_ids.npy", "result_row_identity")
                collector.attest(complete, ["input_manifest_sha256"], manifest_path, "prepared_manifest", copy=True)
                collector.attest(complete, ["source_profile_sha256"], done["source_profile"], "pairwise_profile", copy=True)
                info["runs"].append({"name": name, "manifest": relative(root, mp),
                                     "embedding": relative(root, folder / "values.npy"),
                                     "cell_ids": relative(root, base.parent / "input/cell_ids.npy"),
                                     **check_cell_order(folder / "values.npy", base.parent / "input/cell_ids.npy", metadata)})
        cases.append(info)
    return {"schema_version": 1, "common_registry_sha256": sha(COMMON), "cases": cases,
            "files": list(collector.records.values()), "historical_hash_links": collector.links,
            "scope": "Local adapter byte-copy; selected study source/config snapshots; frozen prepared/fit/output dependencies hashed but not copied. Not a complete from-provider pipeline."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.source_root.resolve(strict=True)
    target = args.output.resolve()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    if target == root or root in target.parents or target in root.parents:
        raise ValueError("Snapshot output overlaps source")
    catalog = build_catalog(root)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(exist_ok=False)
    for row in catalog["files"]:
        source = root / row["source"]
        if "copy" in row:
            dest = target / row["copy"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            with source.open("rb") as src, dest.open("xb") as dst:
                for block in iter(lambda: src.read(1024 * 1024), b""):
                    dst.write(block)
            if sha(dest) != row["sha256"]:
                raise RuntimeError(f"Copy mismatch: {source}")
        if sha(source) != row["sha256"]:
            raise RuntimeError(f"Source changed: {source}")
    create_json(target / "catalog.json", catalog)
    print(json.dumps({"output": str(target), "cases": len(catalog['cases']),
                      "runs": sum(len(c['runs']) for c in catalog['cases']),
                      "files_hashed": len(catalog['files']),
                      "text_files_copied": sum('copy' in f for f in catalog['files'])}))


if __name__ == "__main__":
    main()
