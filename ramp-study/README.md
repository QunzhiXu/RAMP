# RAMP study

Code and configurations for the benchmarks, prediction experiments, biological
applications and figures in the RAMP paper.

To run RAMP on your own data, start with the sibling
[ramp-proteomics package](../ramp-proteomics/README.md). This directory is for
exploring and running the study-specific workflows.

## Where to start

- **Benchmarks:** [benchmarks/](benchmarks/) contains cohort preparation,
  integration evaluation, marker-value analysis and cross-platform prediction
  experiments. [methods/](methods/) contains the comparison-method workflows.
- **Applications:** [applications/](applications/) contains the CD19,
  melanoma, lung and HNSCC analyses.
- **Figures:** [ENTRYPOINTS.json](ENTRYPOINTS.json) lists scripts for the main
  analyses and all 20 main and supplementary figures. The
  [figure code map](docs/historical_maps/FIGURE_CODE_MAP.md) traces their inputs
  and supporting scripts.
- **Results and reproducibility:** [results/](results/) contains aggregate
  reference tables. [Reproduction notes](docs/REPRODUCTION_STATUS.md) describe
  the checks performed, known numerical differences and analysis limitations.

## Directory layout

| Directory | Contents |
| --- | --- |
| `research/final_xu/` | Original analysis scripts, configurations and figure producers |
| `research/cross_platform_trainability/` and other research roots | Supporting research code and prediction experiments |
| `benchmarks/` | Benchmark execution, evaluation and tests |
| `methods/` | Method-specific preparation, alignment and comparison scripts |
| `applications/` | CD19, melanoma, lung and HNSCC workflows |
| `source_archive/` | Older source versions, indexed by content hash |
| `docs/historical_maps/` | Analysis and figure code maps |
| `results/` | Nine aggregate reference tables |

The `research/` hierarchy follows the original workspace so that similarly
named scripts remain with their corresponding analyses. Use `ENTRYPOINTS.json`
to identify the paper's figure versions rather than choosing a historical
variant from `source_archive/`.

## Data and environments

Research datasets, fitted models and large generated outputs are stored
separately. Each workflow needs its input data and method-specific dependencies;
see [external requirements](docs/EXTERNAL_REQUIREMENTS.md) and
[environment notes](environments/README.md).

Some scripts retain the original research paths. Configure these for your
machine before running an analysis. Historical verification and replay commands
also require the result files and manifests they check.

## Source index

[SOURCE_COVERAGE.json](SOURCE_COVERAGE.json) maps the indexed research code and
configurations to their locations here. Byte-identical runtime copies share a
retained source file; distinct older versions are kept in `source_archive/`.

To check the included source files, run from the repository root:

```sh
python -B ramp-study/check_source_coverage.py
python -B ramp-study/catalog.py
```

These commands check source records; they do not run the scientific workflows.

## License

Original study code and documentation are licensed under **CC BY-NC-ND 4.0**.
See [LICENSE](LICENSE) and [LICENSE_SCOPE.md](LICENSE_SCOPE.md). Third-party
software retains its own terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
