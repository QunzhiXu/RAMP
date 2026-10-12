"""Locate byte-identical local figure sources; never rebuild research figures.

Requires Python >=3.11, rg, and pdftotext on PATH. Importing this module has no
I/O side effects. The explicit 'rebuild' action overwrites two derived evidence
JSON files; do not use it to replace the original snapshot during an audit.
"""

import argparse
from pathlib import Path
import hashlib
import json
import re
import subprocess

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
MANUSCRIPT = ROOT / "bioarxiv_Ver2"
SOURCE_ROOTS = [
    ROOT / "final_xu/Main Figure", ROOT / "final_xu/Supplementary Figures",
    ROOT / "final_xu/story_cd19_prediction", ROOT / "final_xu/story_lung_HLA_ABC",
    ROOT / "final_xu/figure_melanoma", ROOT / "final_xu/figure_HNSCC",
    ROOT / "final_xu/figure_raw_labels_20260924", ROOT / "final_xu/publication_cleanup_20260922",
    ROOT / "final_xu/figure3_readability_preview_20260924",
    ROOT / "final_xu/cross_platform_trainability_v2_20260919/Figure",
]


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def body(path):
    raw = subprocess.run(["pdftotext", "-layout", str(path), "-"], check=True, text=True, capture_output=True).stdout
    pages = raw.split("\f")
    result = []
    for page in pages:
        lines = page.splitlines()
        if lines and "bioRxiv preprint doi:" in lines[0]:
            lines = lines[3:]
        lines = [line for line in lines if not re.fullmatch(r"\s*\d+\s*", line)]
        result.extend(" ".join(lines).split())
    return result


def rebuild():
    tex = (MANUSCRIPT / "main.tex").read_text()
    main_figures = re.findall(r"\\paperfig\{([^}]+)\}", tex)
    supp = re.findall(r"\\suppfigurepage\{([^}]+)\}\{([^}]+)\}", tex)
    targets = []
    for i, stem in enumerate(main_figures, 1):
        found = [MANUSCRIPT / "figure" / (stem + ext) for ext in [".pdf", ".png", ".jpg"]
                 if (MANUSCRIPT / "figure" / (stem + ext)).exists()]
        targets.append((f"Figure {i}", found[0], "main"))
    for path, label in supp:
        targets.append((label.replace("fig:s", "S"), MANUSCRIPT / path, "supplementary"))
    cmd = ["rg", "--files", "-g", "*.pdf", "-g", "*.png", "-g", "!**/.venv*/**", "-g", "!**/venv*/**"]
    cmd.extend(str(p) for p in SOURCE_ROOTS)
    candidates = subprocess.run(cmd, check=True, text=True, capture_output=True).stdout.splitlines()
    by_size = {}
    for name in candidates:
        p = Path(name)
        if p.is_file():
            by_size.setdefault(p.stat().st_size, []).append(p)
    records = []
    for label, target, kind in targets:
        checksum = sha(target)
        same = [str(p.relative_to(ROOT)) for p in by_size.get(target.stat().st_size, []) if sha(p) == checksum]
        records.append({"label": label, "kind": kind, "local_manuscript_asset": str(target.relative_to(ROOT)),
                        "bytes": target.stat().st_size, "sha256": checksum, "byte_identical_local_candidates": same,
                        "verification_scope": "Local manuscript asset equality only; not extraction of embedded assets from the bioRxiv PDF."})
        print(label, target.name, len(same), "exact local source matches")
    pub = ROOT / "ramp_paper/RAMP_bioRxiv_2026.09.25.754510v1.pdf"
    local = MANUSCRIPT / "main.pdf"
    a, b = body(pub), body(local)
    comparison = {"published_pdf": str(pub), "published_sha256": sha(pub), "local_pdf": str(local),
                  "local_sha256": sha(local), "published_body_tokens": len(a), "local_body_tokens": len(b),
                  "normalized_body_exact_match": a == b,
                  "normalization": "Remove per-page bioRxiv three-line header and all-number lines; collapse whitespace. No semantic equivalence assertion.",
                  "first_mismatching_tokens": next(({"token_index": i, "published": a[max(0,i-10):i+25], "local": b[max(0,i-10):i+25]}
                                                    for i in range(min(len(a), len(b))) if a[i] != b[i]), None)}
    (HERE / "evidence/publication_assets.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
    (HERE / "evidence/manuscript_identity_comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(comparison, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["rebuild"], help="Overwrite the two derived figure/manuscript evidence JSON files")
    parser.parse_args()
    rebuild()


if __name__ == "__main__":
    main()
