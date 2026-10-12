# RAMP

**Rank-preserving alignment for single-cell and spatial proteomics.**

RAMP aligns shared protein markers across two or three datasets while
preserving their within-unit rank order. It produces an aligned marker matrix
for downstream analysis and cross-platform learning.

This repository accompanies the paper **Rank-Preserving Alignment Enables
Cross-platform Learning and Phenotyping for Single-Cell and Spatial
Proteomics** ([bioRxiv](https://doi.org/10.64898/2026.09.25.754510)).

## Two parts of this repository

- **[ramp-proteomics](ramp-proteomics/README.md)** — the RAMP Python package,
  with a Python API, command-line interface, examples and tests. Start here
  to use RAMP on your own data.
- **[ramp-study](ramp-study/README.md)** — the code and configurations for the
  paper's benchmarks, cross-platform prediction experiments, biological
  applications and figures.

## Install and try RAMP

Requires Python 3.10 or later. From a terminal:

```sh
git clone https://github.com/QunzhiXu/RAMP.git
cd RAMP
python -m pip install ./ramp-proteomics
python ramp-proteomics/examples/quickstart.py
```

The example creates synthetic data and runs an alignment; no research data
are needed. For the command-line options:

```sh
python -m ramp_proteomics --help
```

For your own data, provide a cells-by-markers matrix, a row-matched metadata
table with `dataset` and `unit_key`, and an ordered list of shared markers:

```python
from ramp_proteomics import fit_transform

result = fit_transform(
    raw,                      # cells × shared markers
    metadata,                 # one row per cell
    markers,                  # names in matrix-column order
    datasets=["CODEX", "CyTOF"],
    family="my-study",
)
aligned = result.aligned
```

See the [input guide](ramp-proteomics/docs/input-format.md) for data preparation
and donor/sample metadata, and the [package documentation](ramp-proteomics/README.md)
for complete Python and command-line examples.

## Explore the study

The study code covers integration and marker-value benchmarks, prediction-head
training, and the CD19, melanoma, lung and HNSCC applications.

- [Study guide](ramp-study/README.md) — where to find each workflow.
- [Analysis and figure entrypoints](ramp-study/ENTRYPOINTS.json) — scripts for
  the main analyses and all 20 main and supplementary figures.
- [Figure code map](ramp-study/docs/historical_maps/FIGURE_CODE_MAP.md) — how
  figure scripts connect to their inputs and analyses.
- [Data and environment requirements](ramp-study/docs/EXTERNAL_REQUIREMENTS.md)
  — the resources needed to run the study workflows.
- [Reproduction notes](ramp-study/docs/REPRODUCTION_STATUS.md) — verified
  results, known numerical differences and analysis limitations.

Research datasets and large generated outputs are stored separately. Study
scripts require the relevant datasets and method-specific environments; some
retain the original research paths, which need to be configured for your setup.

## Tests

From the repository root:

```sh
python -m pip install -e './ramp-proteomics[test]'
python -m pytest ramp-proteomics/tests
```

## Citation

Xu Q, Le T, Luo Z, Ly C, Yan Y, Zheng Y. **Rank-Preserving Alignment Enables
Cross-platform Learning and Phenotyping for Single-Cell and Spatial
Proteomics.** bioRxiv (2026).
[doi:10.64898/2026.09.25.754510](https://doi.org/10.64898/2026.09.25.754510).

Machine-readable citation details are in [CITATION.cff](ramp-proteomics/CITATION.cff).

## License

Original RAMP code and documentation are licensed under **CC BY-NC-ND 4.0**:
see the [algorithm license](ramp-proteomics/LICENSE) and
[study license](ramp-study/LICENSE_SCOPE.md). Third-party software retains its
own terms; see [third-party notices](ramp-study/THIRD_PARTY_NOTICES.md).
