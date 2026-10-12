# Preprocessing boundary

`algorithm/harmony` accepts an already selected, already harmonized
cells-by-shared-markers matrix. It performs no expression preprocessing.

For every benchmark, the benchmark-owned record must state:

1. the exact source matrix or assay layer for each dataset;
2. donor/cell inclusion and sampling;
3. shared-marker naming and ordered marker list;
4. the exact input-scale formula, scope, and implementation;
5. whether the source had already been transformed, to prevent double
   transformation;
6. the hashes of the analysis-ready combined matrix and marker file.

The scale is intentionally not universal. ADTnorm compared Harmony on raw,
arcsinh, CLR, and logCPM inputs, while its main Harmony artifact illustration
used CLR input. Accordingly, raw/shared-only/4k/CLR decisions are made and
justified per benchmark, never hidden in this generic wrapper.

The runner makes only a numeric matrix copy and assigns cell/marker names. It
does not center, standardize, clip, invert, impute, transform, subsample, or run
PCA. The output is an adjusted coordinate on the declared input-marker axis.
Figures must retain the input-scale qualifier.
