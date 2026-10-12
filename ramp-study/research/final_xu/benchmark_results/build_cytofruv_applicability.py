#!/usr/bin/env python3
"""Publish and audit CytofRUV applicability for the four frozen benchmarks.

CytofRUV requires a real biological reference specimen remeasured across every
target batch.  This script records the design audit; it never invents reference
samples and never writes an alignment or UMAP for an inapplicable benchmark.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile


HERE = Path(__file__).resolve().parent
FINAL_XU = HERE.parent
ALGORITHM = FINAL_XU / "algorithm" / "cytofruv"

BENCHMARKS = {
    "bone_marrow": {
        "display_name": "Bone marrow",
        "datasets": ["BM CODEX", "BM CyTOF", "Triana AbSeq"],
        "units": [12, 2, 3],
        "evidence": [
            "benchmark_data/bone_marrow/README.md",
            "benchmark_data/bone_marrow/dataset_manifest.csv",
            "benchmark_data/bone_marrow/manifest.json",
            "raw_data/bone_marrow/BM_CyTOF/CANDIDATE_BM_CYTOF_AUDIT.md",
        ],
        "finding": (
            "The three providers contain independent biological units. BM CyTOF H1/H2 "
            "files are provider-pre-gated cell-population files, not a reference specimen "
            "remeasured on CODEX, CyTOF, and AbSeq."
        ),
    },
    "tnbc": {
        "display_name": "TNBC",
        "datasets": ["Tietscher IMC", "Meyer IMC", "Shiao CODEX"],
        "units": [1, 5, 5],
        "evidence": [
            "benchmark_data/tnbc/README.md",
            "benchmark_data/tnbc/dataset_manifest.csv",
            "benchmark_data/tnbc/manifest.json",
        ],
        "finding": (
            "ROIs and images are spatial acquisitions from independent cohorts; no patient "
            "or biological reference material was measured across all three provider studies."
        ),
    },
    "melanoma": {
        "display_name": "Melanoma",
        "datasets": ["Hickey CODEX", "Hoch IMC", "Ptacek MIBI"],
        "units": [6, 17, 23],
        "evidence": [
            "benchmark_data/melanoma/README.md",
            "benchmark_data/melanoma/dataset_manifest.csv",
            "benchmark_data/melanoma/manifest.json",
            "raw_data/melanoma/hickey_codex/HICKEY_PAPER_METADATA_AND_ALIGNMENT_REPRODUCTION_PLAN.md",
        ],
        "finding": (
            "The three studies are independent patient cohorts. Multiple acquisitions are "
            "regions/images, not a common reference specimen spanning CODEX, IMC, and MIBI."
        ),
    },
    "lung_cancer": {
        "display_name": "Lung cancer",
        "datasets": ["Senosain CyTOF", "Cords IMC", "Monkman CODEX"],
        "units": [10, 10, 10],
        "evidence": [
            "benchmark_data/lung_cancer/README.md",
            "benchmark_data/lung_cancer/dataset_manifest.csv",
            "benchmark_data/lung_cancer/manifest.json",
            "raw_data/lung_cancer/Senosain_2025_CyTOF_and_omics/annotation/README.md",
        ],
        "finding": (
            "The three studies contain independent patients/cores. One Senosain patient has "
            "two within-study CyTOF acquisitions, but that does not bridge Cords IMC and "
            "Monkman CODEX."
        ),
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def algorithm_identity() -> dict:
    files = [
        "bootstrap_library.R",
        "cytofruv_repro.R",
        "run_cytofruv.R",
        "upstream_lock.json",
        "runtime_lock.csv",
        "runtime_function_lock.csv",
        "METHOD_SCOPE.md",
    ]
    identity = {}
    for name in files:
        path = ALGORITHM / name
        if not path.is_file():
            raise FileNotFoundError(path)
        identity[name] = sha256(path)
    return identity


def evidence_records(relative_paths: list[str]) -> list[dict]:
    records = []
    for relative in relative_paths:
        path = FINAL_XU / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append({"path": relative, "sha256": sha256(path)})
    return records


def status_payload(name: str, spec: dict, identity: dict) -> dict:
    dataset_rows = [
        {
            "dataset": dataset,
            "biological_units": units,
            "cross_platform_reference_replicate": False,
        }
        for dataset, units in zip(spec["datasets"], spec["units"], strict=True)
    ]
    return {
        "schema_version": "final_xu.cytofruv.applicability.v1",
        "benchmark": name,
        "display_name": spec["display_name"],
        "method": "CytofRUV",
        "status": "NOT_APPLICABLE",
        "reason_code": "missing_cross_batch_replicated_reference_sample",
        "reason": (
            "Official CytofRUV requires the same biological reference material to be "
            "physically measured in every target batch. This frozen three-provider "
            "benchmark has no such cross-platform reference replicate."
        ),
        "benchmark_specific_finding": spec["finding"],
        "dataset_design": dataset_rows,
        "fit_decision": {
            "alignment_generated": False,
            "umap_generated": False,
            "fabricated_reference_samples": False,
            "celltype_labels_used_as_replicates": False,
            "similar_donors_used_as_replicates": False,
            "adtnorm_all_samples_adaptation_used": False,
        },
        "algorithm": {
            "status": "implemented_and_positive_control_validated",
            "mode": "official_replicated_reference",
            "upstream_commit": "7088af8794ede0fc7e3c6ef545638e279e8101eb",
            "identity_sha256": identity,
        },
        "evidence": evidence_records(spec["evidence"]),
    }


def readme_text(payload: dict) -> str:
    design_lines = "\n".join(
        f"- {row['dataset']}: {row['biological_units']} frozen biological "
        f"{'unit' if row['biological_units'] == 1 else 'units'}; "
        "no cross-platform reference replicate."
        for row in payload["dataset_design"]
    )
    evidence_lines = "\n".join(f"- `{item['path']}`" for item in payload["evidence"])
    return f"""# CytofRUV - NOT APPLICABLE

Official CytofRUV was not run for the {payload['display_name']} benchmark.

CytofRUV requires the same biological reference specimen to have been physically
measured in every batch being normalized. This benchmark combines three
independent provider studies and has no such cross-platform reference. Treating
different patients, shared cell types, ROIs, images, or matched populations as
technical reference replicates would change the method and risk treating
biological variation as unwanted variation.

## Frozen design

{design_lines}

## Decision

- Alignment generated: no.
- UMAP generated: no.
- Cell-type or outcome labels used: no.
- ADTnorm's all-samples-as-replicates adaptation used: no.
- Machine-readable reason: `{payload['reason_code']}`.

Benchmark-specific finding: {payload['benchmark_specific_finding']}

The locked official core is implemented and positive-control tested in
`final_xu/algorithm/cytofruv`; this N/A decision concerns experimental design,
not software availability.

## Evidence

{evidence_lines}
"""


def publish_one(name: str, spec: dict, identity: dict) -> dict:
    target = HERE / name / "cytofruv"
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing method status: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = status_payload(name, spec, identity)
    staging = Path(tempfile.mkdtemp(prefix="cytofruv-status-", dir=target.parent))
    try:
        status_path = staging / "METHOD_STATUS.json"
        readme_path = staging / "README.md"
        evidence_path = staging / "applicability_evidence.csv"
        status_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        readme_path.write_text(readme_text(payload))
        with evidence_path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["dataset", "biological_units", "cross_platform_reference_replicate"],
            )
            writer.writeheader()
            writer.writerows(payload["dataset_design"])
        complete = {
            "schema_version": "final_xu.cytofruv.not_applicable.complete.v1",
            "status": "NOT_APPLICABLE",
            "method_status_sha256": sha256(status_path),
            "readme_sha256": sha256(readme_path),
            "applicability_evidence_sha256": sha256(evidence_path),
        }
        (staging / "COMPLETE.json").write_text(json.dumps(complete, indent=2, sort_keys=True) + "\n")
        os.rename(staging, target)
    except Exception:
        for path in sorted(staging.rglob("*"), reverse=True):
            if path.is_file() or path.is_symlink():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        if staging.exists():
            staging.rmdir()
        raise
    return payload


def audit_existing() -> list[dict]:
    identity = algorithm_identity()
    rows = []
    for name, spec in BENCHMARKS.items():
        target = HERE / name / "cytofruv"
        status_path = target / "METHOD_STATUS.json"
        complete_path = target / "COMPLETE.json"
        if not status_path.is_file() or not complete_path.is_file():
            raise FileNotFoundError(f"Incomplete CytofRUV status folder: {target}")
        payload = json.loads(status_path.read_text())
        complete = json.loads(complete_path.read_text())
        expected_payload = status_payload(name, spec, identity)
        if payload != expected_payload:
            raise RuntimeError(f"CytofRUV status payload differs from frozen evidence: {name}")
        if complete["method_status_sha256"] != sha256(status_path):
            raise RuntimeError(f"Status hash mismatch: {name}")
        if complete.get("readme_sha256") != sha256(target / "README.md"):
            raise RuntimeError(f"README hash mismatch: {name}")
        if complete.get("applicability_evidence_sha256") != sha256(
            target / "applicability_evidence.csv"
        ):
            raise RuntimeError(f"Evidence CSV hash mismatch: {name}")
        if (target / "README.md").read_text() != readme_text(payload):
            raise RuntimeError(f"README content differs from status payload: {name}")
        if payload["status"] != "NOT_APPLICABLE":
            raise RuntimeError(f"Unexpected CytofRUV status: {name}")
        if payload["algorithm"]["identity_sha256"] != identity:
            raise RuntimeError(f"Algorithm identity changed after status publication: {name}")
        expected_evidence = evidence_records(spec["evidence"])
        if payload["evidence"] != expected_evidence:
            raise RuntimeError(f"Benchmark evidence changed after status publication: {name}")
        if (target / "alignment").exists() or (target / "umap").exists():
            raise RuntimeError(f"Inapplicable benchmark must not contain alignment/UMAP: {name}")
        rows.append(
            {
                "benchmark": name,
                "display_name": payload["display_name"],
                "status": payload["status"],
                "reason_code": payload["reason_code"],
                "alignment_generated": False,
                "umap_generated": False,
            }
        )
    return rows


def write_global_summary(rows: list[dict]) -> None:
    csv_path = HERE / "cytofruv_applicability_summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    audit = {
        "schema_version": "final_xu.cytofruv.global_applicability_audit.v1",
        "status": "PASS",
        "method": "CytofRUV",
        "benchmarks": rows,
        "summary_sha256": sha256(csv_path),
        "interpretation": (
            "The official algorithm is implemented, but all four frozen cross-platform "
            "benchmarks lack the required replicated reference specimen."
        ),
    }
    (HERE / "cytofruv_applicability_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )


def audit_global_summary(rows: list[dict]) -> None:
    """Verify the already-published global files without changing their bytes."""
    csv_path = HERE / "cytofruv_applicability_summary.csv"
    audit_path = HERE / "cytofruv_applicability_audit.json"
    if not csv_path.is_file() or not audit_path.is_file():
        raise FileNotFoundError("Missing global CytofRUV applicability summary/audit")
    with csv_path.open(newline="") as handle:
        observed_rows = list(csv.DictReader(handle))
    expected_rows = [
        {key: str(value) for key, value in row.items()}
        for row in rows
    ]
    if observed_rows != expected_rows:
        raise RuntimeError("Global CytofRUV applicability CSV differs from benchmark statuses")
    audit = json.loads(audit_path.read_text())
    if audit.get("status") != "PASS" or audit.get("method") != "CytofRUV":
        raise RuntimeError("Invalid global CytofRUV audit status")
    if audit.get("benchmarks") != rows:
        raise RuntimeError("Global CytofRUV audit rows differ from benchmark statuses")
    if audit.get("summary_sha256") != sha256(csv_path):
        raise RuntimeError("Global CytofRUV summary hash mismatch")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if not args.audit_only:
        identity = algorithm_identity()
        for name, spec in BENCHMARKS.items():
            publish_one(name, spec, identity)
    rows = audit_existing()
    if args.audit_only:
        audit_global_summary(rows)
    else:
        write_global_summary(rows)
    print("PASS: CytofRUV applicability audit; 4/4 benchmarks NOT_APPLICABLE")


if __name__ == "__main__":
    main()
