# fastMNN preprocessing protocol for the protein benchmarks

## Method-layer rule

`algorithm/fastmnn` accepts a benchmark-owned, analysis-ready matrix. It does
not infer an assay type from the numbers and does not apply arcsinh, CLR, log,
feature scaling, clipping, or inverse transformation.

Inside the official algorithm, `cos.norm=TRUE` performs cell-wise L2
normalization before a single `multiBatchPCA()` across all batches. This is an
algorithm step and must not be confused with upstream assay preprocessing.
In batchelor 1.18.1, an all-zero cell is retained: the internal divisor is
`pmax(1e-8, L2)`, so its zero marker vector remains zero. The wrapper follows
that official behavior and records zero-L2 counts instead of deleting cells or
adding an upstream pseudocount.

## Required benchmark-owned decisions

For each dataset, the preprocessing record must state:

1. exact source file and assay/layer;
2. source technology;
3. frozen cell-selection manifest and cell order;
4. marker-name harmonization and ordered shared-marker intersection;
5. whether the source values are raw counts, technically normalized continuous
   intensities, transformed values, or a provider analysis layer;
6. every transformation formula, cofactor, and scope;
7. source-paper or provider-documentation support;
8. exact preprocessing code and its SHA-256;
9. exact analysis-ready matrix and combined-input SHA-256;
10. explicit confirmation that no unintended second nonlinear transform or
    global marker-wise z-score was applied;
11. explicit confirmation that shared-marker selection, merge order, and
    canonical parameters were frozen without cell-type or outcome labels.

The correct transformation may differ across datasets because the stored
provider layers differ. This is not permission to choose preprocessing by
which UMAP looks best. Each route must be justified from the source data and
paper before fastMNN is run.

The algorithm runner authenticates the exact preprocessing-record bytes and
analysis-ready input by SHA-256, and verifies that they remain unchanged while
the fit is running. It does not infer from numeric values whether upstream
marker selection was label-free. That audit belongs to the benchmark-owned
sampling, harmonization, and preprocessing scripts; their hashes and explicit
declarations must be included in the preprocessing record.

## What not to use

- Do not apply `asinh(x / median(all positive values))` as a universal rule.
- Do not perform pooled or per-dataset marker-wise z-scoring merely to improve
  mixing; standardization changes marker weights and cosine geometry.
- Do not run `multiBatchNorm()` on protein intensities. Its count-depth model
  and log-count semantics are specific to sequencing counts.
- Do not use cell-type labels to select features, merge order, or parameters.
- Do not pass non-shared markers; all input matrices must contain the same
  features in the same order.

## Small-panel PCA policy

Official fastMNN requests `d=50`. For the current setting where the number of
cells is much larger than `p` shared markers, batchelor 1.18.1 with its default
`IrlbaParam()` implementation realizes the request as

```text
d = min(50, p - 1, n - 1)
```

The wrapper still passes `50` unchanged and records both the requested and
realized values, where `n` is the total number of cells. This avoids the old
non-default `d=NA` marker-space adaptation while retaining every component
returned by the official default implementation for the current small panels.

## Downstream use

Use only the corrected latent embedding for UMAP, graph construction,
clustering, LISI, silhouette, neighbour preservation, and transfer metrics.
Use the original benchmark-owned marker matrix for marker visualization or
biology. Do not interpret latent axes or the low-rank reconstructed assay as
corrected biomarker abundance.
