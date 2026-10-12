# Study source and configuration

This directory preserves the existing analysis, training, evaluation, figure
generation and verification source covered by the study indexes. It also
includes parameter, suite, environment and figure configuration files. Source
coverage is checked independently of the presence of numerical results.

## License

Original RAMP-author source code, tests, configurations and documentation use
**CC BY-NC-ND 4.0**, matching the paper, at the author's explicit instruction.
See [LICENSE](LICENSE) and [LICENSE_SCOPE.md](LICENSE_SCOPE.md). Third-party
material retains its original terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
The specific unresolved MaxFuse license-scope question is recorded there.

## Directory layout

| Directory | Contents |
| --- | --- |
| `research/final_xu/` | Original scientific scripts and configurations, including Main Figure and Supplementary Figures |
| `research/cross_platform_trainability/` and other indexed roots | Original upstream/helper research sources where referenced |
| `benchmarks/` | Migrated cohort, alignment, evaluation, value and trainability controllers/tests |
| `methods/` | Comparator wrappers, provider preparation, original scientific sources and method tests |
| `applications/` | CD19, melanoma, lung and HNSCC controllers and scientific scripts |
| `source_archive/` | Unique older runtime source versions retained by content hash, not current entrypoints |
| `docs/historical_maps/` | Detailed analysis/figure code maps and existing dependency qualifications |
| `results/` | Nine aggregate reference tables; no full predictions or patient-level outcome tables |

The hierarchy under `research/` matches the original workspace, rather than
flattening similarly named `run.py`, `common.py` or `build_figure.py` files.
Current figure producers are listed in `ENTRYPOINTS.json`; historical variants
are preserved but must not be substituted for current routes.

## What complete source means here

`SOURCE_COVERAGE.json` accounts for the code in the original source/curated
indexes and current study source census, current source/config enumeration in
those research scopes, plus discovered literal local code/config dependencies.
Each included record is checked against an actual retained file's hash.
Identical runtime copies are represented by a mapping to retained identical
bytes; unique runtime source is preserved. Installed third-party environments
are listed separately, not mistaken for author source.

This is a finite, explicit source-completeness claim, not a proof of every
possible dynamic import or a claim that a missing historical producer has been
invented. Previously documented missing producers remain documented. Original
source hashes and historical review labels are preserved; the current license
declaration and third-party notice describe the updated licensing scope.

## Running code

Use the sibling [algorithm package](../ramp-proteomics/README.md) for new RAMP
analyses. For the paper, follow `ENTRYPOINTS.json`, then the corresponding
scientific script, its configurations and the code maps. Each workflow needs
its own data and dependencies; see [external requirements](docs/EXTERNAL_REQUIREMENTS.md)
and [environments](environments/README.md).

Many original scripts still contain original absolute paths. Configure paths
in a separate working copy when using another machine. This source delivery
does not rewrite scientific code or bypass fixed hashes to pretend historical
validators work without their missing payloads. Original migrated validators
still need their complete snapshot/bundle/evidence contracts.

```sh
python -B ramp-study/check_source_coverage.py
python -B ramp-study/catalog.py
```

Run these from the repository root. Neither command fits models or accesses
patient data. Numerical reproduction scope and existing scientific limitations
are recorded in [reproduction status](docs/REPRODUCTION_STATUS.md).
