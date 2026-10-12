"""Read-only source census for the RAMP manuscript code inventory.

Writes derived indexes only below this inventory directory. Does not import or
execute study scripts, traverse provider raw-data folders, or run analyses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parents[1]
WORKSPACE = HERE.parents[1]
PAPER = WORKSPACE / "ramp_paper/RAMP_bioRxiv_2026.09.25.754510v1.pdf"
ROOTS = [
    "ramp-proteomics", "bioarxiv_Ver2", "Biorxiv",
    "cross_platform_trainability", "lung_caf_covariate_fix_20260917",
    "melanoma_story_alignment_audit_20260919",
    "final_xu/algorithm", "final_xu/benchmark_data", "final_xu/benchmark_results",
    "final_xu/Main Figure", "final_xu/Supplementary Figures",
    "final_xu/figure_HNSCC", "final_xu/figure_melanoma",
    "final_xu/story_cd19_prediction", "final_xu/story_lung_HLA_ABC",
    "final_xu/shared_utils", "final_xu/cross_platform_trainability_audit_20260919",
    "final_xu/cross_platform_trainability_v2_20260919",
    "final_xu/manuscript_trainability_revision_20260921",
    "final_xu/publication_cleanup_20260922", "final_xu/figure_raw_labels_20260924",
    "final_xu/figure3_readability_preview_20260924",
    "final_xu/manuscript_final_check_overleaf_20260924",
]
SUFFIXES = {".py", ".r", ".sh", ".md", ".toml", ".yaml", ".yml", ".tex", ".cff"}
PATTERNS = {
    "absolute_local_path": r"/Users/[^\s\"'<>]+|/private/(?:tmp|var)/[^\s\"'<>]+",
    "test_path_reference": r"(?:\btest0827\b|story_test_|[\"'/]test[\"'/])",
    "historical_reference": r"algorithm_old|algorithm_ver1|ramp_ver1|previous_|backup_",
    "sampling_or_seed": r"\b(?:seed|random_state|default_rng|random\.seed|np\.random|sample\()",
    "data_io": r"read_(?:csv|parquet|h5ad|table)|np\.load|numpy\.load|readRDS|read\.csv|fread\(|load_dataset|load\(",
    "execution_or_import": r"^\s*(?:from |import |source\(|library\(|require\()|subprocess\.|runpy\.|spec_from_file_location|sys\.path",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def census() -> None:
    roots = [WORKSPACE / p for p in ROOTS if (WORKSPACE / p).is_dir()]
    command = ["rg", "--files", "--hidden", "--no-ignore"]
    for glob in [
        "!**/.git/**", "!**/.venv*/**", "!**/venv*/**", "!**/env/**", "!**/envs/**",
        "!**/node_modules/**", "!**/__pycache__/**", "!**/site-packages/**",
        "!**/raw_data/**", "!**/.pytest_cache/**", "!**/.ruff_cache/**",
        "!**/build/**", "!**/dist/**", "!**/.env", "!**/.env.*",
        "!**/R_libs/**", "!**/r-library/**", "!**/r_libs/**", "!**/renv/library/**",
    ]:
        command.extend(["-g", glob])
    command.extend(str(p) for p in roots)
    listing = subprocess.run(command, check=True, capture_output=True, text=True).stdout.splitlines()
    records = []
    for name in sorted(set(listing)):
        p = Path(name)
        if p.suffix.lower() not in SUFFIXES and not p.name.startswith(("requirements", "environment")):
            continue
        if p.is_symlink():
            records.append({"path": str(p.relative_to(WORKSPACE)), "symlink": True, "target": str(p.resolve())})
            continue
        st = p.stat()
        rec = {"path": str(p.relative_to(WORKSPACE)), "bytes": st.st_size, "mtime_ns": st.st_mtime_ns,
               "sha256": digest(p)}
        if p.suffix.lower() in {".py", ".r", ".sh"} and st.st_size <= 2_000_000:
            lines = p.read_text(errors="replace").splitlines()
            flags = {}
            for key, pattern in PATTERNS.items():
                hits = [{"line": i, "text": line[:500]} for i, line in enumerate(lines, 1)
                        if re.search(pattern, line, flags=re.IGNORECASE if key == "sampling_or_seed" else 0)]
                if hits:
                    flags[key] = hits
            rec["static_reference_hits"] = flags
        records.append(rec)
        if len(records) % 1000 == 0:
            print(f"Indexed {len(records)} code/document/environment files", flush=True)
    write_json(HERE / "evidence/source_census.json", {
        "snapshot_utc": datetime.now(timezone.utc).isoformat(),
        "workspace": str(WORKSPACE), "scope": ROOTS,
        "mode": "static only; source scripts were not executed; regex hits are candidates, not resolved dependencies; JSON result files are excluded from this census and selected manifests are indexed separately",
        "files": records,
    })
    print(json.dumps({"files": len(records), "roots": len(roots), "source_bytes": sum(r.get("bytes", 0) for r in records)}))


def paper() -> None:
    text = subprocess.run(["pdftotext", "-layout", str(PAPER), "-"], check=True, capture_output=True, text=True).stdout
    (HERE / "evidence").mkdir(parents=True, exist_ok=True)
    (HERE / "evidence/paper_layout.txt").write_text(text)
    pages = text.split("\f")
    if not pages[-1].strip():
        pages.pop()
    summaries = []
    for number, content in enumerate(pages, 1):
        headings = [line.strip() for line in content.splitlines() if re.search(
            r"^\s*(?:Figure \d+:|Supplementary (?:Figure|Fig\.|Table)|(?:\d+\s+)?(?:Results|Methods|Discussion|Code Availability|Data Availability|Benchmark cohorts)\s*$)", line)]
        summaries.append({"pdf_page": number, "headings": headings})
    write_json(HERE / "evidence/paper_identity.json", {
        "path": str(PAPER), "bytes": PAPER.stat().st_size, "sha256": digest(PAPER),
        "pages": len(pages), "page_index": summaries,
    })
    print(json.dumps(summaries, ensure_ascii=False))


def verify() -> None:
    saved = json.loads((HERE / "evidence/source_census.json").read_text())
    changed = []
    for rec in saved["files"]:
        p = WORKSPACE / rec["path"]
        if rec.get("symlink"):
            if not p.is_symlink() or str(p.resolve()) != rec["target"]:
                changed.append(rec["path"])
        elif not p.is_file() or digest(p) != rec["sha256"]:
            changed.append(rec["path"])
    result = {"checked_utc": datetime.now(timezone.utc).isoformat(), "checked_files": len(saved["files"]),
              "changed_or_missing": changed, "scope": "Only files present in the initial source census; not provider data or all workspace files."}
    write_json(HERE / "evidence/source_unchanged_check.json", result)
    print(json.dumps(result))
    if changed:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["census", "paper", "verify"])
    args = parser.parse_args()
    {"census": census, "paper": paper, "verify": verify}[args.action]()
