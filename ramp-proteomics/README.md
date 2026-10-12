# RAMP Proteomics

RAMP (**Rank-Preserving Alignment of Multi-Platform Protein Data**) aligns
shared protein markers across two or three single-cell or spatial proteomics
datasets. It returns a corrected marker matrix rather than only a latent
embedding and introduces no strict within-unit marker-rank inversions.

This package provides the RAMP 2.3.0 implementation with a Python API,
command-line interface, file readers, tests and synthetic examples.
Paper-specific settings, including the lung benchmark's pair-fit sizes,
are described in [docs/reproducibility.md](docs/reproducibility.md).

## Associated preprint

RAMP is described in:

> Xu Q, Le T, Luo Z, Ly C, Yan Y, Zheng Y. **Rank-Preserving Alignment
> Enables Cross-platform Learning and Phenotyping for Single-Cell and Spatial
> Proteomics.** bioRxiv (2026).
> [https://doi.org/10.64898/2026.09.25.754510](https://doi.org/10.64898/2026.09.25.754510)

Installation commands below run inside `ramp-proteomics/`.

## Installation

From a local checkout:

```bash
python -m pip install .
```

For development and tests:

```bash
python -m pip install -e '.[test,dev]'
pytest
```

The distribution name is `ramp-proteomics`; the Python import is
`ramp_proteomics`. Install from this repository using the commands above.

## Python quick start

```python
import numpy as np
from ramp_proteomics import fit_transform
from ramp_proteomics.io import read_metadata

raw = np.load("raw.npy")                 # cells x shared markers
metadata = read_metadata("cell_metadata.csv.gz")  # same row order
markers = [line.strip() for line in open("markers.txt") if line.strip()]

result = fit_transform(
    raw,
    metadata,
    markers,
    datasets=["CODEX", "CyTOF", "CITE-seq"],
    family="frozen-study-v1",
)

aligned = result.aligned
assert result.strict_rank_inversions == 0
```

RAMP performs its KMEANSLOW preprocessing internally. Supply the
highest-available source-scale measurements for the shared marker panel; do
not pre-transform the matrix using another comparison method's preprocessing.

## Where donor information goes

Donor membership and cell identities belong in a separate row-aligned cell
metadata table (for example, `cell_metadata.csv.gz`):

```csv
cell_id,dataset,donor_id,unit_key
cell_0001,CODEX,P01,CODEX::P01
cell_0002,CODEX,P01,CODEX::P01
cell_0003,CyTOF,P01,CyTOF::P01
```

- `dataset` and `unit_key` are required and are the only metadata columns read
  during fitting.
- `donor_id` is strongly recommended for provenance and downstream analysis,
  but is not read by the alignment algorithm. It must identify biological
  donors globally; reuse the same value across platforms only for the same
  biological donor.
- `unit_key` must identify one dataset-specific fitting unit and must not be
  reused across datasets. Prefixing it with the dataset is recommended.
- By default, use one donor-level `unit_key` per dataset, such as
  `CODEX::P01`. Keep ROI or sample IDs in separate annotation columns.

Cell-level annotations may be stored in the same table. RAMP discards them
before fitting, and the CLI preserves the complete table beside the aligned
matrix so cell and donor identities remain traceable. Donor-level clinical
attributes such as age, treatment, response, or survival should remain in a
separate `donor_metadata` table keyed by `donor_id`; that table is not an
input to RAMP and therefore cannot influence alignment.

If `cell_id` is omitted, the CLI creates row-based IDs. Those generated IDs
are deterministic for the current file but cannot reconstruct an external
cell mapping, so publication analyses should always supply stable `cell_id`
values.

See [docs/input-format.md](docs/input-format.md) for the full contract.

## Command line

```bash
ramp-proteomics align \
  --raw raw.npy \
  --metadata cell_metadata.csv.gz \
  --markers markers.txt \
  --datasets CODEX CyTOF CITE-seq \
  --family frozen-study-v1 \
  --output ramp_output/
```

The CLI writes:

- `kmeanslow_global.npy` and `ramp_aligned.npy`;
- the validated `cell_metadata.csv.gz` and `markers.txt` in exact row/column
  order;
- coupling, strength, evidence, reference-weight, profile, and algorithm audit
  tables;
- `ramp_manifest.json` containing versions, hashes, parameters, dataset order,
  metadata policy, and output schema.

Existing RAMP files are never replaced unless `--overwrite` is supplied. Even
then, unrelated files are not deleted. Input/output collisions, including
hard links and symbolic links, are rejected. Outputs are staged before
publishing and replaced one file at a time, without truncating existing
files. The manifest is published last; the directory-wide update is not
transactional, so a failed publication should be rerun before using its
results.

## Core guarantees and scope

- Exactly two or three datasets are supported.
- Only shared marker values plus `dataset` and `unit_key` are used for fitting.
- Within each `unit_key` and marker, RAMP introduces no strict rank
  inversions; tied values may remain tied or arise during preprocessing.
- Every final cell-marker displacement is at most 0.75 on the internal
  KMEANSLOW scale.
- RAMP uses a fixed seed and deterministic `family`-dependent role assignment.
  Freeze `family`, row order, marker order, and dataset order for a
  reproducible study.
- Cell labels, spatial coordinates, clinical outcomes, and donor outcomes are
  never required or read by the algorithm.

RAMP alignment operates on shared markers. Prediction of non-overlapping
markers, spatial projection, and survival analyses are downstream workflows,
not hidden steps in the core aligner.

The guarantee and displacement audit are computed in float64. The CLI uses
float64 output by default. Optional float32 storage can collapse extremely
close values into ties and can display a sub-micro-unit rounding excess near
the 0.75 boundary; use the default for publication audits.

## Reproducibility

The public dependency ranges support ordinary installation. The audited
numerical package versions and runtime environment are recorded in
[`reproducibility/requirements-paper.txt`](reproducibility/requirements-paper.txt).
Numerical changes to the algorithm require a new algorithm version and golden
regression against the frozen 2.3.0 implementation.

## License

Original package code and documentation are licensed under **CC BY-NC-ND 4.0**.
See [LICENSE](LICENSE) and [AUTHORS.md](AUTHORS.md). Third-party dependencies
retain their own terms.
