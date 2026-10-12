"""Hash selected study entry points/configurations; never execute study code.

The allowlist complements the broad census with JSON pointers, input manifests,
selected upstream test scripts, and code living in raw_data/annotation. Large
data matrices are not read or hashed. All writes stay in this inventory folder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
OUT = HERE / "evidence"
MAX_HASH_BYTES = 8_000_000

# A pattern that returns no files is exposed, not silently considered verified.
GROUPS = {
    "manuscript": ["bioarxiv_Ver2/main.tex", "bioarxiv_Ver2/PROVENANCE.json", "bioarxiv_Ver2/supplementary_data/README.md"],
    "software": ["ramp-proteomics/README.md", "ramp-proteomics/PROVENANCE.md", "ramp-proteomics/pyproject.toml", "ramp-proteomics/reproducibility/requirements-paper.txt", "ramp-proteomics/docs/reproducibility.md"],
    "method_sources": [
        "final_xu/algorithm/ramp/ramp_ver2.py", "final_xu/algorithm/ramp/ramp.py",
        "final_xu/algorithm/*/README*", "final_xu/algorithm/*/run_*.py", "final_xu/algorithm/*/run_*.R",
        "final_xu/algorithm/*/*_repro.R", "final_xu/algorithm/*/bootstrap*",
        "final_xu/algorithm/*/upstream_lock.json", "final_xu/algorithm/*/runtime*lock.csv",
        "final_xu/algorithm/*/requirements.txt", "final_xu/algorithm/maxfuse/maxfuse_repro/*.py",
        "final_xu/algorithm/maxfuse/maxfuse_repro/*lock*", "final_xu/algorithm/cycombine/vendor/publication-renv.lock",
        "final_xu/algorithm/CytoNorm/benchmark.py",
    ],
    "four_benchmark_data": [
        "final_xu/benchmark_data/*/build_benchmark.py", "final_xu/benchmark_data/*/README.md",
        "final_xu/benchmark_data/*/manifest.json", "final_xu/benchmark_data/*/relabel/build_relabel.py",
        "final_xu/benchmark_data/*/relabel/README.md", "final_xu/benchmark_data/bone_marrow/export_codex_30k.R",
    ],
    "upstream_sampling": [
        "final_xu/test/tnbc_meyer5_resample_seed20260902/scripts/build_test_input.py",
        "final_xu/test/tnbc_meyer5_resample_seed20260902/scripts/screen_coverage_seeds.py",
        "final_xu/test/tnbc_meyer5_resample_seed20260902/scripts/export_meyer.R",
        "final_xu/test/melanoma_metadata_matched_20k_20260827/01_build_inputs.py",
        "final_xu/test0827/**/prepare_lung_inputs.py",
        "final_xu/raw_data/lung_cancer/Senosain_2025_CyTOF_and_omics/annotation/*.py",
        "final_xu/raw_data/lung_cancer/Senosain_2025_CyTOF_and_omics/annotation/rough_annotation/*.py",
        "final_xu/raw_data/lung_cancer/Senosain_2025_CyTOF_and_omics/annotation/rough_annotation/README.md",
    ],
    "benchmark_runners": [
        "final_xu/benchmark_results/*/ramp/alignment/run_alignment.py",
        "final_xu/benchmark_results/*/ramp/umap/plot_umap.py",
        "final_xu/benchmark_results/*/ramp/alignment/*manifest*.json",
        "final_xu/benchmark_results/*/maxfuse/run_alignment.py",
        "final_xu/benchmark_results/*/maxfuse/plot_umap.py",
        "final_xu/benchmark_results/*/harmony/run_all.py", "final_xu/benchmark_results/*/fastmnn/run_all.py",
        "final_xu/benchmark_results/*/cycombine/preprocessing/build_inputs.*",
        "final_xu/benchmark_results/*/cycombine/alignment/run_alignment.R",
        "final_xu/benchmark_results/*_benchmark_runner.py", "final_xu/benchmark_results/_maxfuse*.py",
        "final_xu/benchmark_results/cytofruv_applicability_summary.csv",
        "final_xu/benchmark_results/*/maxfuse_archive/config.json",
        "final_xu/benchmark_results/*/maxfuse_archive/README.md",
        "final_xu/benchmark_results/*/maxfuse_archive/*.py",
        "final_xu/benchmark_results/*/maxfuse_archive/*.R",
        "final_xu/benchmark_results/*/maxfuse_v2/build_inputs.py",
        "final_xu/benchmark_results/*/maxfuse_v2/config.json",
        "final_xu/benchmark_results/MAXFUSE_ARCHIVE_CURRENT_AUDIT.md",
    ],
    "metrics_values": [
        "final_xu/benchmark_results/comparison_evaluation/scripts/*.py",
        "final_xu/benchmark_results/exact_value_artifacts/scripts/*.py",
        "final_xu/Main Figure/celltype_percentile_shift_20260914/*.py",
        "final_xu/Main Figure/pair_inversion_boxplot_20260914/compute_pair_inversions.py",
        "final_xu/Main Figure/fastmnn_reconstructed_audit_20260914/*.py",
        "final_xu/Main Figure/Figure 2/value_evaluation_20260914/*.py",
        "final_xu/Main Figure/Figure 2/lung_reference_update_20260921/*.py",
        "final_xu/Main Figure/Figure 2/lung_reference_update_20260921/selection/README.md",
        "final_xu/Main Figure/Figure 2/lung_reference_update_20260921/Panel */build_panel.py",
    ],
    "application_comparisons": [
        "final_xu/figure_melanoma/Comparison/scripts/*", "final_xu/figure_HNSCC/Comparison/scripts/*",
        "final_xu/story_lung_HLA_ABC/Comparison_senosain140k_20260921/scripts/*",
        "final_xu/figure_melanoma/Comparison/maxfuse_full_panel/scripts/*",
        "final_xu/figure_HNSCC/Comparison/maxfuse_full_panel/scripts/*",
        "final_xu/story_lung_HLA_ABC/Comparison_senosain140k_20260921/maxfuse_full_panel/scripts/*",
    ],
    "trainability": [
        "final_xu/cross_platform_trainability_v2_20260919/CURRENT_FIGURE3_RESULTS.json",
        "final_xu/cross_platform_trainability_v2_20260919/src/*.py",
        "final_xu/cross_platform_trainability_v2_20260919/src/trainability/*.py",
        "final_xu/cross_platform_trainability_v2_20260919/config/formal_v0_5/*.json",
        "final_xu/cross_platform_trainability_v2_20260919/releases/figure3_bc_update_20260922/*",
        "final_xu/cross_platform_trainability_v2_20260919/releases/j1_scope_update_20260921_r3/*",
        "final_xu/cross_platform_trainability_v2_20260919/exploration_tnbc_five_classes_20260921/*.py",
        "final_xu/cross_platform_trainability_v2_20260919/exploration_melanoma_three_classes_20260921/*.py",
        "final_xu/cross_platform_trainability_v2_20260919/exploration_lung_original_four_classes_20260921/*.py",
        "final_xu/cross_platform_trainability_v2_20260919/exploration_lung_new140k_comparison_20260921/*.py",
        "final_xu/cross_platform_trainability_v2_20260919/exploration_melanoma_story_lymphocytes_20260921/*.py",
        "final_xu/cross_platform_trainability_v2_20260919/exploration_hnscc_cd4_cd8_b_20260921/*.py",
        "final_xu/cross_platform_trainability_v2_20260919/exploration_lung_new140k_j2_20260922/*.py",
        "final_xu/cross_platform_trainability_v2_20260919/exploration_lung_new140k_j2_20260922/MANIFEST.json",
        "final_xu/cross_platform_trainability_v2_20260919/Figure/build_panels_bc.py",
    ],
    "cd19": [
        "final_xu/story_cd19_prediction/prediction/*.py",
        "final_xu/story_cd19_prediction/alignment/run_alignment.py",
        "final_xu/story_cd19_prediction/umap/plot_umap.py",
        "final_xu/story_cd19_prediction/Exploration0909/scripts/cd3_nn_joint_validation.py",
        "final_xu/story_cd19_prediction/Exploration0909/scripts/all_marker_nn_validation.py",
        "final_xu/story_cd19_prediction/Exploration0909/scripts/assemble_cd19_layout_v30_polish.py",
        "final_xu/story_cd19_prediction/Exploration0909/figure_layout_v31_DAPI_pair/build_figure.py",
        "final_xu/story_cd19_prediction/Exploration0909/real_spatial0913/scripts/*.py",
    ],
    "melanoma_story": [
        "final_xu/figure_melanoma/inputs/cohort_package/build_story_sample.py",
        "final_xu/figure_melanoma/inputs/*manifest*.json", "final_xu/figure_melanoma/inputs/observed_survival_results.csv",
        "final_xu/figure_melanoma/analysis/alignment/run_alignment.py",
        "final_xu/figure_melanoma/analysis/run_all.sh",
        "final_xu/figure_melanoma/analysis/all_cd8_prediction/*.py",
        "final_xu/figure_melanoma/figure_layout_v8/prepare_visual_data.py",
        "final_xu/figure_melanoma/figure_layout_v8/build_figure.py",
    ],
    "lung_story": [
        "final_xu/story_lung_HLA_ABC/CURRENT_ANALYSIS.json",
        "final_xu/story_lung_HLA_ABC/releases/senosain140k_20260921/scripts/*.py",
        "final_xu/story_lung_HLA_ABC/releases/senosain140k_20260921/scripts/*.R",
        "final_xu/story_lung_HLA_ABC/releases/senosain140k_20260921/run.sh",
        "final_xu/story_lung_HLA_ABC/releases/senosain140k_20260921/vendor/ramp/*.py",
        "final_xu/story_lung_HLA_ABC/data_umap/scripts/*.py",
        "final_xu/story_lung_HLA_ABC/data_cords/scripts/*.py",
        "final_xu/story_lung_HLA_ABC/validation/scripts/*.py",
        "final_xu/story_lung_HLA_ABC/Figure/Panel */scripts/draw_panel_*.py",
        "final_xu/story_lung_HLA_ABC/Figure/Combined ABCDEF/scripts/assemble_ABCDEF.py",
        "final_xu/story_lung_HLA_ABC/update_20260921/UPDATE_RECORD.md",
    ],
    "hnscc_story": [
        "final_xu/figure_HNSCC/survival_motivation/0*.py",
        "final_xu/figure_HNSCC/survival_motivation/0*.R",
        "final_xu/figure_HNSCC/alignment/0*.py",
        "final_xu/figure_HNSCC/biomarker_discovery/0*.py",
        "final_xu/figure_HNSCC/spatial_return_survival/01_return_fixed_program.py",
        "final_xu/figure_HNSCC/big_figure/panel_E/group_fraction_scan_20260910/scan_group_fractions.py",
        "final_xu/figure_HNSCC/spatial_score_review_20260913/*.py",
        "final_xu/figure_HNSCC/spatial_score_review_20260913/roi_selection_20260913/*.py",
        "final_xu/figure_HNSCC/spatial_score_review_20260913/roi_selection_20260913/selected_spatial_panel/*.py",
        "final_xu/figure_HNSCC/big_figure/polish_20260913/CEF_D_colorbar_aligned_20260914/build_figure.py",
    ],
    "final_display_and_supplements": [
        "final_xu/figure_raw_labels_20260924/update_labels.py",
        "final_xu/figure3_readability_preview_20260924/draw_preview.py",
        "final_xu/publication_cleanup_20260922/build_display_assets.py",
        "final_xu/Supplementary Figures/ridge_compact_preview_20260924/S*/build_figure.py",
        "final_xu/Supplementary Figures/ridge_compact_preview_20260924/ridge_common.py",
        "final_xu/Supplementary Figures/ridge_compact_preview_20260924/TARGETS.json",
        "final_xu/Supplementary Figures/Figure 2 complete benchmark/preview_v1/build_figures.py",
        "final_xu/Supplementary Figures/Figure 3 model benchmark breakdown/final_20260922/*.py",
    ],
}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(name, content):
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text(json.dumps(content, indent=2, ensure_ascii=False) + "\n")


def build():
    files = {}
    unresolved = []
    patterns = []
    for group, globs in GROUPS.items():
        for pattern in globs:
            paths = sorted(p for p in ROOT.glob(pattern) if p.is_file() and p.suffix != ".pyc")
            patterns.append({"group": group, "pattern": pattern, "matches": len(paths)})
            if not paths:
                unresolved.append({"group": group, "pattern": pattern})
            for path in paths:
                relative = str(path.relative_to(ROOT))
                if relative not in files:
                    size = path.stat().st_size
                    files[relative] = {"path": relative, "bytes": size, "groups": [], "sha256": digest(path) if size <= MAX_HASH_BYTES else None}
                if group not in files[relative]["groups"]:
                    files[relative]["groups"].append(group)
    save("curated_sources.json", {
        "snapshot_utc": datetime.now(timezone.utc).isoformat(), "workspace": str(ROOT),
        "scope": "Selected entry points/configs/pointers; not a complete runtime/import dependency closure or a shipping allowlist.",
        "hash_limit_bytes": MAX_HASH_BYTES, "patterns": patterns,
        "unmatched_patterns": unresolved, "files": list(files.values()),
    })
    data_root = ROOT / "bioarxiv_Ver2/supplementary_data"
    listing = [{"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size}
               for p in sorted(data_root.rglob("*")) if p.is_file()]
    save("supplementary_data_file_list.json", {"scope": "Names/sizes only; no table contents copied, no claim that each table producer is resolved.", "files": listing})
    print(json.dumps({"curated_files": len(files), "unmatched_patterns": unresolved, "supplementary_files": len(listing)}, ensure_ascii=False))


def verify():
    saved = json.loads((OUT / "curated_sources.json").read_text())
    changed = []
    checked = 0
    for entry in saved["files"]:
        if not entry["sha256"]:
            continue
        checked += 1
        path = ROOT / entry["path"]
        if not path.is_file() or digest(path) != entry["sha256"]:
            changed.append(entry["path"])
    result = {"checked_utc": datetime.now(timezone.utc).isoformat(), "checked_files": checked,
              "changed_or_missing": changed, "scope": "Selected files hashed at curated snapshot time; supplementary table contents and large matrices not included."}
    save("curated_unchanged_check.json", result)
    print(json.dumps(result))
    if changed:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["build", "verify"])
    {"build": build, "verify": verify}[parser.parse_args().action]()
