from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from ramp_proteomics import fit_transform
from ramp_proteomics import cli, io

from _synthetic import synthetic_input


@pytest.fixture(scope="module")
def aligned_input():
    raw, metadata, markers = synthetic_input(datasets=2, seed=611)
    result = fit_transform(
        raw, metadata, markers, datasets=["D1", "D2"], family="output-safety"
    )
    return raw, metadata, markers, result


def write_inputs(directory: Path, raw=None):
    paths = {
        "raw": directory / "raw.npy",
        "metadata": directory / "input_metadata.csv",
        "markers": directory / "input_markers.txt",
    }
    np.save(paths["raw"], np.arange(44).reshape(22, 2) if raw is None else raw)
    paths["metadata"].write_text(
        "dataset,unit_key\n" + "A,A::1\n" * 11 + "B,B::1\n" * 11,
        encoding="utf-8",
    )
    paths["markers"].write_text("M1\nM2\n", encoding="utf-8")
    return paths


def cli_arguments(paths, output, overwrite=True):
    arguments = [
        "align", "--raw", str(paths["raw"]),
        "--metadata", str(paths["metadata"]),
        "--markers", str(paths["markers"]),
        "--datasets", "A", "B", "--family", "output-safety",
        "--output", str(output),
    ]
    return arguments + (["--overwrite"] if overwrite else [])


@pytest.mark.parametrize("input_name", ["raw", "metadata", "markers"])
@pytest.mark.parametrize("output_name", list(io.OUTPUT_FILES.values()))
@pytest.mark.parametrize("link_kind", ["hard", "symbolic"])
def test_cli_rejects_linked_inputs_before_fitting(
    tmp_path, monkeypatch, input_name, output_name, link_kind
):
    inputs = write_inputs(tmp_path)
    original = {name: path.read_bytes() for name, path in inputs.items()}
    output = tmp_path / "output"
    output.mkdir()
    destination = output / output_name
    if link_kind == "hard":
        os.link(inputs[input_name], destination)
    else:
        destination.symlink_to(inputs[input_name])

    def unexpected_fit(*args, **kwargs):
        pytest.fail("input collision must be rejected before fitting")

    monkeypatch.setattr(cli, "fit_transform", unexpected_fit)
    with pytest.raises(ValueError, match="RAMP output path"):
        cli.main(cli_arguments(inputs, output))
    assert {name: path.read_bytes() for name, path in inputs.items()} == original
    assert set(output.iterdir()) == {destination}


@pytest.mark.parametrize("raw", [np.array(1.0), np.arange(22), np.zeros((2, 3, 4))])
def test_cli_rejects_nonmatrix_before_metadata_validation(tmp_path, raw):
    inputs = write_inputs(tmp_path, raw=raw)
    with pytest.raises(ValueError, match="finite two-dimensional matrix"):
        cli.main(cli_arguments(inputs, tmp_path / "output"))
    assert not (tmp_path / "output").exists()


def test_result_writer_rejects_hard_link_to_input(tmp_path, aligned_input):
    raw, metadata, markers, result = aligned_input
    raw_path = tmp_path / "raw.npy"
    np.save(raw_path, raw)
    original = raw_path.read_bytes()
    output = tmp_path / "output"
    output.mkdir()
    os.link(raw_path, output / io.OUTPUT_FILES["aligned"])
    with pytest.raises(ValueError, match="shares file identity"):
        io.write_result_directory(
            result, metadata, markers, output,
            family="output-safety", datasets=["D1", "D2"], overwrite=True,
            input_paths={"raw": raw_path},
        )
    assert raw_path.read_bytes() == original
    assert not (output / io.OUTPUT_FILES["baseline"]).exists()


@pytest.mark.parametrize("link_kind", ["hard", "symbolic"])
def test_overwrite_preserves_external_link_targets(tmp_path, aligned_input, link_kind):
    _, metadata, markers, result = aligned_input
    output, external = tmp_path / "output", tmp_path / "external"
    output.mkdir()
    external.mkdir()
    for filename in io.OUTPUT_FILES.values():
        target = external / filename
        target.write_bytes(b"external original: " + filename.encode())
        if link_kind == "hard":
            os.link(target, output / filename)
        else:
            (output / filename).symlink_to(target)
    original = {p.name: p.read_bytes() for p in external.iterdir()}
    manifest = io.write_result_directory(
        result, metadata, markers, output, family="output-safety",
        datasets=["D1", "D2"], overwrite=True,
    )
    for filename, contents in original.items():
        assert (external / filename).read_bytes() == contents
        assert not (output / filename).samefile(external / filename)
        assert not (output / filename).is_symlink()
    for name, digest in manifest["output_sha256"].items():
        assert io.sha256_file(output / io.OUTPUT_FILES[name]) == digest


def test_failed_serialization_leaves_existing_outputs_intact(
    tmp_path, aligned_input, monkeypatch
):
    _, metadata, markers, result = aligned_input
    arguments = {"family": "output-safety", "datasets": ["D1", "D2"]}
    io.write_result_directory(result, metadata, markers, tmp_path, **arguments)
    original = {p.name: p.read_bytes() for p in tmp_path.iterdir()}

    def fail_write(*args, **kwargs):
        raise OSError("simulated serialization failure")

    monkeypatch.setattr(io, "_write_audit_table", fail_write)
    with pytest.raises(OSError, match="simulated serialization failure"):
        io.write_result_directory(
            result, metadata, markers, tmp_path, overwrite=True, **arguments
        )
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == original


def test_collision_is_rechecked_after_serialization(tmp_path, aligned_input, monkeypatch):
    raw, metadata, markers, result = aligned_input
    raw_path, output = tmp_path / "raw.npy", tmp_path / "output"
    np.save(raw_path, raw)
    original = raw_path.read_bytes()
    serialize = io._write_result_files

    def create_link_after_serialization(*args, **kwargs):
        manifest = serialize(*args, **kwargs)
        os.link(raw_path, output / io.OUTPUT_FILES["aligned"])
        return manifest

    monkeypatch.setattr(io, "_write_result_files", create_link_after_serialization)
    with pytest.raises(ValueError, match="shares file identity"):
        io.write_result_directory(
            result, metadata, markers, output,
            family="output-safety", datasets=["D1", "D2"], overwrite=True,
            input_paths={"raw": raw_path},
        )
    assert raw_path.read_bytes() == original
    assert set(p.name for p in output.iterdir()) == {io.OUTPUT_FILES["aligned"]}


def test_link_created_after_validation_cannot_truncate_input(
    tmp_path, aligned_input, monkeypatch
):
    raw, metadata, markers, result = aligned_input
    raw_path, output = tmp_path / "raw.npy", tmp_path / "output"
    np.save(raw_path, raw)
    original = raw_path.read_bytes()
    replace = os.replace

    def create_link_before_replace(source, destination):
        if Path(destination).name == io.OUTPUT_FILES["aligned"]:
            os.link(raw_path, destination)
        return replace(source, destination)

    monkeypatch.setattr(io.os, "replace", create_link_before_replace)
    io.write_result_directory(
        result, metadata, markers, output,
        family="output-safety", datasets=["D1", "D2"], overwrite=True,
        input_paths={"raw": raw_path},
    )
    assert raw_path.read_bytes() == original
    np.testing.assert_array_equal(np.load(output / io.OUTPUT_FILES["aligned"]), result.aligned)


def test_overwrite_is_byte_reproducible(tmp_path, aligned_input):
    _, metadata, markers, result = aligned_input
    arguments = {"family": "output-safety", "datasets": ["D1", "D2"]}
    io.write_result_directory(result, metadata, markers, tmp_path, **arguments)
    original = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    io.write_result_directory(result, metadata, markers, tmp_path, overwrite=True, **arguments)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == original
    manifest = json.loads((tmp_path / io.OUTPUT_FILES["manifest"]).read_text())
    for name, digest in manifest["output_sha256"].items():
        assert io.sha256_file(tmp_path / io.OUTPUT_FILES[name]) == digest
