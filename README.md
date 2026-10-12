# RAMP algorithm and study source release

This local candidate contains the installable RAMP algorithm and the existing
research source code and configuration used across the study. It does not
contain the raw datasets, fitted models, full numerical outputs or manuscript
PDFs. The author has requested GitHub publication preparation. The specific
remaining MaxFuse license-scope question is recorded in
[RELEASE_GATES.md](RELEASE_GATES.md); no upload has been performed.

## Algorithm

```sh
python -m pip install ./ramp-proteomics
python ramp-proteomics/examples/quickstart.py
python -m ramp_proteomics --help
```

The [algorithm package](ramp-proteomics/README.md) includes its full source,
API, CLI, tests and synthetic examples. Numerical source is unchanged.

The original algorithm package code and documentation use
[CC BY-NC-ND 4.0](ramp-proteomics/LICENSE), explicitly selected by the package
owner to match the bioRxiv paper. This is a noncommercial, no-derivatives
license, not a permissive open-source license. The paper author list is in
[AUTHORS.md](ramp-proteomics/AUTHORS.md). Original study source now uses the
same license under its own [scope declaration](ramp-study/LICENSE_SCOPE.md).
Third-party material retains its original terms.

## Study

- [Study guide](ramp-study/README.md): analysis, training, evaluation and figures.
- [Research entrypoints](ramp-study/ENTRYPOINTS.json): main and supplementary
  figure producers and principal analysis routes.
- [Code maps](ramp-study/docs/historical_maps/FIGURE_CODE_MAP.md): detailed
  producer relationships, including upstream helpers and old path conventions.
- [Coverage](ramp-study/SOURCE_COVERAGE.json): every indexed source/configuration
  and its retained destination. Identical runtime copies point to retained
  byte-identical source; unique historical versions remain in `source_archive`.
- [External resources](ramp-study/docs/EXTERNAL_REQUIREMENTS.md): input data,
  runtime dependencies and historical evidence contracts are separate resources.

`ramp-study/research/` preserves the original workspace hierarchy for the
research scripts, including `final_xu/`. The migration controllers and their
tests remain under `ramp-study/benchmarks`, `methods` and `applications`.

MaxFuse-related local source is present for completeness. Its specific license
question is explained in [third-party notices](ramp-study/THIRD_PARTY_NOTICES.md).
Installed third-party environments and upstream source archives are external
dependencies, not copied software installations.

## Check the source distribution

```sh
python -B release_check.py
python -B ramp-study/check_source_coverage.py
python -B -m unittest discover -s release_tests -v
python -m pip install -e './ramp-proteomics[test]'
python -m pytest ramp-proteomics/tests -p no:cacheprovider
```

These are source/package checks, not new full scientific runs. The algorithm
can run independently on synthetic or new data. Historical study scripts still
require their datasets, appropriate environments and path configuration;
complete source is not a claim of one-command, data-free reproduction.

Scientific qualifications, post-hoc choices, UMAP differences and figure-QA
boundaries remain in [reproduction status](ramp-study/docs/REPRODUCTION_STATUS.md).
Existing results and PDFs were not changed while preparing this candidate.
