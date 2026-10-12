from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ramp_proteomics.cli import main
from ramp_proteomics.io import OUTPUT_FILES, read_metadata, write_result_directory

from _synthetic import synthetic_input


def test_cli_roundtrip_preserves_donor_metadata(tmp_path: Path) -> None:
    raw, metadata, markers = synthetic_input(datasets=2, seed=401)
    raw_path = tmp_path / "raw.npy"
    metadata_path = tmp_path / "metadata.csv.gz"
    markers_path = tmp_path / "markers.txt"
    output = tmp_path / "output"
    np.save(raw_path, raw)
    metadata.to_csv(metadata_path, index=False)
    markers_path.write_text("\n".join(markers) + "\n", encoding="utf-8")

    status = main(
        [
            "align",
            "--raw",
            str(raw_path),
            "--metadata",
            str(metadata_path),
            "--markers",
            str(markers_path),
            "--datasets",
            "D1",
            "D2",
            "--family",
            "cli-roundtrip",
            "--output",
            str(output),
        ]
    )
    assert status == 0
    for name in OUTPUT_FILES.values():
        assert (output / name).exists(), name
    aligned = np.load(output / OUTPUT_FILES["aligned"], allow_pickle=False)
    assert aligned.shape == raw.shape
    written_metadata = read_metadata(output / OUTPUT_FILES["metadata"])
    assert written_metadata["cell_id"].astype(str).tolist() == metadata["cell_id"].tolist()
    assert written_metadata["donor_id"].astype(str).tolist() == metadata["donor_id"].tolist()
    manifest = json.loads((output / OUTPUT_FILES["manifest"]).read_text())
    assert manifest["algorithm_version"] == "2.3.0"
    assert manifest["strict_rank_inversions"] == 0
    assert manifest["profile_audit_rows"] > 0
    assert manifest["fitted_profile_rows"] > 0
    assert manifest["corrected_cell_marker_values"] > 0
    assert manifest["identity_only_alignment"] is False
    assert manifest["donor_id_present"] is True
    assert manifest["donor_count"] == 2
    assert manifest["dataset_donor_pair_count"] == 4
    assert manifest["donor_counts_by_dataset"] == {"D1": 2, "D2": 2}
    assert manifest["metadata_columns_read_for_fitting"] == ["dataset", "unit_key"]
    assert manifest["cell_id_generated"] is False
    assert set(manifest["inputs"]["raw"]) == {"sha256"}
    assert "/" not in manifest["source"]["algorithm"]["file"]
    assert set(manifest["output_sha256"]) == set(OUTPUT_FILES) - {"manifest"}


def test_input_cannot_collide_with_planned_output(tmp_path: Path) -> None:
    raw, metadata, markers = synthetic_input(datasets=2, seed=402)
    datasets = ["D1", "D2"]
    from ramp_proteomics import fit_transform

    result = fit_transform(
        raw, metadata, markers, datasets=datasets, family="collision-test"
    )
    output = tmp_path / "output"
    output.mkdir()
    colliding_input = output / OUTPUT_FILES["aligned"]
    np.save(colliding_input, raw)

    with pytest.raises(ValueError, match="planned RAMP output path"):
        write_result_directory(
            result,
            metadata,
            markers,
            output,
            family="collision-test",
            datasets=datasets,
            overwrite=True,
            input_paths={"raw": colliding_input},
        )


def test_identity_only_result_has_readable_audits_and_manifest(tmp_path: Path) -> None:
    from ramp_proteomics import fit_transform

    raw = np.arange(44, dtype=float).reshape(22, 2)
    metadata = pd.DataFrame(
        {
            "cell_id": [f"c{i:02d}" for i in range(22)],
            "dataset": ["A"] * 11 + ["B"] * 11,
            "donor_id": ["P01"] * 11 + ["P02"] * 11,
            "unit_key": ["A::P01"] * 11 + ["B::P02"] * 11,
        }
    )
    with pytest.warns(RuntimeWarning, match="no correction profiles"):
        result = fit_transform(
            raw,
            metadata,
            ["M1", "M2"],
            datasets=["A", "B"],
            family="identity-only-output",
        )
    manifest = write_result_directory(
        result,
        metadata,
        ["M1", "M2"],
        tmp_path / "output",
        family="identity-only-output",
        datasets=["A", "B"],
    )
    assert manifest["fitted_profile_rows"] == 0
    assert manifest["corrected_cell_marker_values"] == 0
    assert manifest["identity_only_alignment"] is True
    for name in (
        "strength_audit",
        "evidence_audit",
        "reference_weight_audit",
        "profile_audit",
    ):
        table = pd.read_csv(tmp_path / "output" / OUTPUT_FILES[name])
        assert len(table) == 0
        assert len(table.columns) > 0
