"""Snapshot/verify nine explicitly scoped upstream files without loading arrays.

Python standard library only. Inputs are opened read-only and streamed into
SHA-256; no research modules are imported, no source files are copied. Writes
are restricted to this inventory's evidence directory. The original census
and curated snapshots are never regenerated or changed by this script.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
SNAPSHOT = HERE / "evidence/audit2_dependency_inputs.json"
VERIFICATION = HERE / "evidence/audit2_dependency_inputs_verify.json"

HNSCC_E = "final_xu/figure_HNSCC/big_figure/panel_E/group_fraction_scan_20260910/scan_group_fractions.py"
HNSCC_F = "final_xu/figure_HNSCC/spatial_score_review_20260913/check_codex_phenotype.py"
LUNG = "final_xu/story_lung_HLA_ABC/releases/senosain140k_20260921/scripts/align.py"
MELANOMA = "final_xu/figure_melanoma/figure_layout_v8/build_figure.py"

INPUTS = [
    ("critical_upstream", "final_xu/test/hnscc_tigit_cd39_molecular_spatial_refinement_20260904/citeseq_state_validation/results/tumour_cd8_molecular_scores.csv.gz", HNSCC_E),
    ("critical_upstream", "final_xu/story_test_hnscc/ending_exploration_20260910/individual_marker_donor_contrasts.csv", HNSCC_E),
    ("critical_upstream", "final_xu/story_test_hnscc/final_return_exploration_20260910/target_all_metadata.csv.gz", HNSCC_F),
    ("critical_upstream", "final_xu/story_test_hnscc/final_return_exploration_20260910/target_all_aligned.npy", HNSCC_F),
    ("critical_upstream", "final_xu/cross_platform_trainability_v2_20260919/reports/senosain_new_annotation_review_20260921/CANDIDATE_REFERENCE_IDS_REVIEW_ONLY.csv.gz", LUNG),
    ("melanoma_display", "final_xu/figure_melanoma/cache/ramp230_clean15_display_umap.npy", MELANOMA),
    ("melanoma_display", "final_xu/figure_melanoma/cache/ramp230_clean15_display_metadata.csv.gz", MELANOMA),
    ("melanoma_display", "final_xu/figure_melanoma/cache/ramp230_clean15_display_manifest.json", MELANOMA),
    ("melanoma_display", "final_xu/figure_melanoma/figure_layout_v7/assets/ramp_sign_cross_platform_alignment.png", MELANOMA),
]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def signature(stat):
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def fingerprint(path):
    before = path.stat()
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    after = path.stat()
    if signature(before) != signature(after):
        raise RuntimeError(f"File changed while hashing: {path}")
    return {"bytes": before.st_size, "mtime_ns": before.st_mtime_ns,
            "sha256": checksum.hexdigest(), "hashed_utc": utc_now()}


def consumer_evidence(relative, input_name):
    path = ROOT / relative
    result = {"path": relative, **fingerprint(path)}
    result["matching_source_lines"] = [
        {"line": number, "text": line}
        for number, line in enumerate(path.read_text().splitlines(), 1)
        if input_name in line
    ]
    if not result["matching_source_lines"]:
        raise RuntimeError(f"Input filename not found in consumer: {relative}: {input_name}")
    return result


def snapshot():
    if SNAPSHOT.exists():
        raise FileExistsError("Snapshot already exists; use verify, do not overwrite the baseline.")
    started = utc_now()
    records = []
    for group, relative, consumer in INPUTS:
        path = ROOT / relative
        records.append({"group": group, "path": relative, **fingerprint(path),
                        "consumer": consumer_evidence(consumer, path.name)})
    counts = {group: sum(r["group"] == group for r in records)
              for group in sorted({r["group"] for r in records})}
    result = {"schema": "ramp-study.audit2-input-snapshot.v1", "started_utc": started,
              "completed_utc": utc_now(), "workspace": str(ROOT), "counts": counts,
              "scope": "5 critical upstream inputs + 4 melanoma display dependencies. Byte identity only; not producer reconstruction, numerical validation, or proof of unchanged files before this snapshot.",
              "files": records}
    # Exclusive creation protects the first snapshot even under concurrent runs.
    with SNAPSHOT.open("x") as output:
        output.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"snapshot": str(SNAPSHOT), "counts": counts,
                      "total_input_bytes": sum(r["bytes"] for r in records)}))


def verify():
    saved = json.loads(SNAPSHOT.read_text())
    allowed = {relative for _, relative, _ in INPUTS}
    if len(saved["files"]) != len(INPUTS) or {r["path"] for r in saved["files"]} != allowed:
        raise ValueError("Snapshot input list differs from the scoped allowlist")
    mismatches = []
    consumers = {}
    for record in saved["files"]:
        path = ROOT / record["path"]
        current = fingerprint(path) if path.is_file() else None
        if current is None or any(current[k] != record[k] for k in ["bytes", "sha256"]):
            mismatches.append({"kind": "input", "path": record["path"]})
        consumers[record["consumer"]["path"]] = record["consumer"]
    for relative, reference in consumers.items():
        path = ROOT / relative
        if not path.is_file() or fingerprint(path)["sha256"] != reference["sha256"]:
            mismatches.append({"kind": "consumer", "path": relative})
    result = {"checked_utc": utc_now(), "snapshot_sha256": fingerprint(SNAPSHOT)["sha256"],
              "checked_inputs": len(saved["files"]), "checked_consumers": len(consumers),
              "mismatches": mismatches, "status": "FAIL" if mismatches else "PASS"}
    VERIFICATION.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result))
    if mismatches:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["snapshot", "verify"])
    {"snapshot": snapshot, "verify": verify}[parser.parse_args().action]()


if __name__ == "__main__":
    main()
