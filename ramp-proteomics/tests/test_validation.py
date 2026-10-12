from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ramp_proteomics import fit_transform
from ramp_proteomics import _algorithm
from ramp_proteomics.io import (
    capture_input_signatures,
    read_metadata,
    validate_cell_metadata,
    validate_input_signatures,
)

from _synthetic import synthetic_input


def test_unit_key_cannot_be_reused_across_datasets() -> None:
    metadata = pd.DataFrame(
        {
            "cell_id": ["a", "b"],
            "dataset": ["CODEX", "CyTOF"],
            "donor_id": ["P01", "P01"],
            "unit_key": ["P01", "P01"],
        }
    )
    with pytest.raises(ValueError, match="globally unique"):
        validate_cell_metadata(metadata, 2)


def test_missing_cell_ids_are_generated_deterministically() -> None:
    metadata = pd.DataFrame(
        {
            "dataset": ["CODEX", "CODEX"],
            "unit_key": ["CODEX::P01", "CODEX::P01"],
        }
    )
    answer = validate_cell_metadata(metadata, 2)
    assert answer["cell_id"].tolist() == ["cell_0000000", "cell_0000001"]


def test_core_rejects_invalid_shapes_and_missing_units() -> None:
    raw, metadata, markers = synthetic_input(datasets=2, seed=301)
    with pytest.raises(ValueError, match="finite two-dimensional"):
        fit_transform(raw[:, :, None], metadata, markers)
    broken = metadata.copy()
    broken.loc[0, "unit_key"] = np.nan
    with pytest.raises(ValueError, match="cannot contain missing"):
        fit_transform(raw, broken, markers)


def test_signed_integer_extremes_match_canonical_result() -> None:
    raw = np.tile(
        np.array([[-128, 127], [127, -128]], dtype=np.int8), (150, 1)
    )
    metadata = pd.DataFrame(
        {
            "dataset": ["A"] * 150 + ["B"] * 150,
            "unit_key": ["A::P01"] * 150 + ["B::P01"] * 150,
        }
    )
    arguments = {"datasets": ["A", "B"], "family": "signed-integers"}
    expected = _algorithm.fit_transform(raw, metadata, ["M1", "M2"], **arguments)
    observed = fit_transform(raw, metadata, ["M1", "M2"], **arguments)
    assert np.isfinite(observed.aligned).all()
    np.testing.assert_array_equal(observed.baseline, expected.baseline)
    np.testing.assert_array_equal(observed.aligned, expected.aligned)


@pytest.mark.parametrize("column", ["dataset", "unit_key"])
@pytest.mark.parametrize("value", ["", "   ", " P01", "P01 "])
def test_python_api_rejects_blank_or_padded_fitting_identity(column, value) -> None:
    raw, metadata, markers = synthetic_input(datasets=2, seed=303)
    metadata.loc[0, column] = value
    with pytest.raises(ValueError, match="blank|whitespace"):
        fit_transform(raw, metadata, markers)


def test_noninformative_shared_panel_has_clear_error() -> None:
    raw = np.ones((22, 2), dtype=float)
    metadata = pd.DataFrame(
        {
            "dataset": ["A"] * 11 + ["B"] * 11,
            "unit_key": ["A::P01"] * 11 + ["B::P01"] * 11,
        }
    )
    with pytest.raises(ValueError, match="non-zero variation"):
        fit_transform(raw, metadata, ["M1", "M2"], datasets=["A", "B"])


def test_dataset_audit_slug_collisions_are_rejected() -> None:
    raw = np.arange(66, dtype=float).reshape(33, 2)
    metadata = pd.DataFrame(
        {
            "dataset": ["A-B"] * 11 + ["A B"] * 11 + ["C"] * 11,
            "unit_key": ["A-B::P01"] * 11
            + ["A B::P01"] * 11
            + ["C::P01"] * 11,
        }
    )
    with pytest.raises(ValueError, match="audit-column normalization"):
        fit_transform(raw, metadata, ["M1", "M2"], datasets=["A-B", "A B", "C"])


def test_small_input_warns_when_no_profiles_are_learned() -> None:
    raw = np.arange(44, dtype=float).reshape(22, 2)
    metadata = pd.DataFrame(
        {
            "dataset": ["A"] * 11 + ["B"] * 11,
            "unit_key": ["A::P01"] * 11 + ["B::P01"] * 11,
        }
    )
    with pytest.warns(RuntimeWarning, match="no correction profiles"):
        result = fit_transform(
            raw, metadata, ["M1", "M2"], datasets=["A", "B"], family="small"
        )
    assert result.profile_audit.empty


def test_nonempty_audit_without_fitted_profiles_still_warns() -> None:
    raw, metadata, markers = synthetic_input(
        datasets=2, cells_per_unit=100, seed=808
    )
    with pytest.warns(RuntimeWarning, match="no correction profiles"):
        result = fit_transform(
            raw,
            metadata,
            markers,
            datasets=["D1", "D2"],
            family="screened-but-unfitted",
        )
    assert not result.profile_audit.empty
    assert not bool(result.profile_audit["curve_profile_fitted"].any())


def test_duplicate_cell_id_is_rejected_by_file_interface() -> None:
    raw, metadata, _ = synthetic_input(datasets=2, seed=302)
    metadata.loc[1, "cell_id"] = metadata.loc[0, "cell_id"]
    with pytest.raises(ValueError, match="unique"):
        validate_cell_metadata(metadata, len(raw))


def test_csv_identity_columns_preserve_leading_zeroes(tmp_path) -> None:
    path = tmp_path / "metadata.csv"
    path.write_text(
        "cell_id,dataset,donor_id,unit_key,age\n"
        "000123,CODEX,001,CODEX::001,61\n"
        "000124,CODEX,001,CODEX::001,61\n",
        encoding="utf-8",
    )
    metadata = validate_cell_metadata(read_metadata(path), 2)
    assert metadata["cell_id"].tolist() == ["000123", "000124"]
    assert metadata["donor_id"].tolist() == ["001", "001"]
    assert metadata["unit_key"].tolist() == ["CODEX::001", "CODEX::001"]
    assert metadata["age"].tolist() == ["61", "61"]


def test_optional_identifiers_preserve_leading_zeroes(tmp_path) -> None:
    path = tmp_path / "metadata.tsv"
    path.write_text(
        "cell_id\tdataset\tdonor_id\tunit_key\troi_id\tsample_id\n"
        "c1\tCODEX\t001\tCODEX::001\t0007\t01\n"
        "c2\tCODEX\t001\tCODEX::001\t0007\t01\n",
        encoding="utf-8",
    )
    metadata = validate_cell_metadata(read_metadata(path), 2)
    assert metadata["roi_id"].tolist() == ["0007", "0007"]
    assert metadata["sample_id"].tolist() == ["01", "01"]


def test_literal_na_identifier_is_not_converted_to_missing(tmp_path) -> None:
    path = tmp_path / "metadata.csv"
    path.write_text(
        "cell_id,dataset,donor_id,unit_key\n"
        "NA,CODEX,NA,CODEX::NA\n"
        "cell2,CODEX,NA,CODEX::NA\n",
        encoding="utf-8",
    )
    metadata = validate_cell_metadata(read_metadata(path), 2)
    assert metadata.loc[0, "cell_id"] == "NA"
    assert metadata["donor_id"].tolist() == ["NA", "NA"]


def test_input_mutation_during_loading_is_rejected(tmp_path) -> None:
    path = tmp_path / "raw.npy"
    np.save(path, np.arange(12, dtype=float))
    inputs = {"raw": path}
    signatures = capture_input_signatures(inputs)
    np.save(path, np.arange(13, dtype=float))
    with pytest.raises(RuntimeError, match="changed while it was being read: raw"):
        validate_input_signatures(inputs, signatures)


@pytest.mark.parametrize("column", ["cell_id", "dataset", "donor_id", "unit_key"])
def test_blank_identity_values_are_rejected(column: str) -> None:
    metadata = pd.DataFrame(
        {
            "cell_id": ["c1", "c2"],
            "dataset": ["CODEX", "CODEX"],
            "donor_id": ["P01", "P01"],
            "unit_key": ["CODEX::P01", "CODEX::P01"],
        }
    )
    metadata.loc[0, column] = "   "
    with pytest.raises(ValueError, match="blank"):
        validate_cell_metadata(metadata, 2)
