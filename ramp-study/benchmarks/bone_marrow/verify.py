#!/usr/bin/env python3
"""Compare a regenerated run to frozen historical fingerprints (read-only by default)."""

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def digest(path, decompress=False):
    with (gzip.open(path, "rb") if decompress else path.open("rb")) as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(run_dir):
    reference_path = HERE / "provenance/reference_outputs.json"
    reference = json.loads(reference_path.read_text())
    checks = []
    for item in reference["files"]:
        path = run_dir / item["run_path"]
        actual = digest(path, item["comparison"] == "gzip_payload_sha256") if path.is_file() else None
        checks.append({"path": item["run_path"], "comparison": item["comparison"],
                       "expected": item["sha256"], "actual": actual,
                       "match": actual == item["sha256"],
                       "file_bytes_match": path.is_file() and digest(path) == item["file_sha256"]})
    manifest_path = run_dir / "data/manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    checks.append({"path": "data/manifest.json:provider_source_hashes", "match":
                   manifest.get("source_files_sha256") == reference["provider_files_sha256"],
                   "note": "Hashes recorded by the builder; this verifier does not reread provider files."})
    return {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "reference_sha256": digest(reference_path),
        "checks": checks,
        "all_reference_checks_passed": all(item["match"] for item in checks),
        "data_and_numerical_checks_passed": all(item["match"] for item in checks if not item["path"].endswith(".png")),
        "scope": "Historical BM14 benchmark inputs, relabel, RAMP outputs/audits and UMAP; not all paper analyses or final publication layouts.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, help="Optional new report; refuses overwrite")
    args = parser.parse_args()
    if args.report and (args.report.exists() or args.report.is_symlink()):
        parser.error("Report exists; choose a new path")
    result = verify(args.run_dir.expanduser().resolve(strict=True))
    text = json.dumps(result, indent=2) + "\n"
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x") as stream:
            stream.write(text)
    raise SystemExit(0 if result["all_reference_checks_passed"] else 1)


if __name__ == "__main__":
    main()
