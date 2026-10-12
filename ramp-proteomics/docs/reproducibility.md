# Reproducibility contract

For a reproducible RAMP run, freeze all of the following:

1. expression matrix values and row order;
2. marker names and column order;
3. metadata row order;
4. `dataset` and `unit_key` values;
5. explicit dataset order;
6. the `family` run identifier;
7. RAMP and dependency versions.

The `family` string participates in deterministic role assignment. Changing
it can change the fitted mappings even when all numerical inputs are
unchanged. The CLI therefore requires an explicit value.

RAMP 2.3.0 uses these frozen core settings:

- seed: 42;
- 15 non-self graph neighbors;
- pair-fit minimum/per-unit/maximum: 4,000 / 100 / 8,000;
- compatibility maximum: 3,500 cells per dataset;
- 257-point mapping grid;
- maximum displacement: 0.75 on the KMEANSLOW scale;
- adaptive equal-count curve bins with at least 15 cells per bin.

The algorithm asserts zero strict within-unit marker-rank inversions and
refuses to return an output when the displacement ceiling is violated.

The internal deterministic role allocation is direction-specific in RAMP
2.3.0: a query dataset can use different cell partitions against different
reference datasets. The four allocations are curve fitting (48%), strength
calibration (16%), reference-weight calibration (16%), and a reserved 20%
holdout named `test`. The reserved cells are excluded from downstream curve
fitting, strength calibration, and reference-weight calibration. They can
participate in the upstream correspondence construction; this is not an
independent end-to-end test set. The core package does not report them as an
accuracy estimate. This behavior is retained for exact compatibility with
the audited manuscript algorithm.

Some audit-table columns retain historical names such as `donor_alpha`,
`donor_z`, and `donor_queries`. In RAMP 2.3.0 these fields are calculated for
`unit_key`; “donor” in those column names should be read as “fitting unit.”

The audited numerical dependency lock is in
`reproducibility/requirements-paper.txt`. General installation uses compatible
version ranges, so a publication-critical rerun should use the exact lock and
archive the generated manifest.

The CI `frozen-regression` job uses Python 3.14.1 and this exact dependency
lock on a macOS arm64 runner. It requires the numerical fingerprints to run;
a version mismatch fails collection rather than silently skipping those
tests. The broader Python-version matrix also tests the supported dependency
ranges. Source-equivalence tests additionally use the canonical manuscript
tree when it is available locally.

## Frozen lung benchmark exception

The canonical package defaults are pair-fit minimum/per-unit/maximum
`4000 / 100 / 8000`. The formal three-platform lung benchmark retained an
older frozen runtime depth of `7000 / 0 / 7000` to reproduce its accepted
analysis exactly. That benchmark-specific override does not redefine RAMP
2.3.0 and is not silently applied by this package. Exact reproduction of that
one result must use its archived benchmark runner and manifest; ordinary new
analyses should use the canonical defaults above.

The lock records all direct numerical dependencies plus joblib and
threadpoolctl. The audited run used Python 3.14.1 on macOS arm64 with Apple
Accelerate BLAS/LAPACK. Archive each generated manifest because low-order
floating-point differences can occur across platforms.
