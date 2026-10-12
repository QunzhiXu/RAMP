# cyCombine input-scale protocol

## Core rule

cyCombine takes an **analysis-ready protein expression matrix**. “Normalized”
does not by itself mean “nonlinearly transformed.” Technical normalization,
denoising, background subtraction, and arcsinh/CLR transformation are distinct
operations and must be traced separately for every source dataset.

The runner intentionally performs no assay transform. Its `norm_method="rank"`
option only creates a temporary within-batch rank representation for the
correction SOM. ComBat then corrects the original values supplied to cyCombine.
The output is therefore neither a rank matrix nor a common physical unit when
the three inputs use different analysis scales.

## Required decision process for each dataset

1. Read the source paper and provider metadata; identify the exact assay/layer.
2. Determine whether the stored values are raw, technically normalized but
   untransformed, nonlinearly transformed, standardized, or already on a
   documented provider analysis scale.
3. Reproduce the assay- and study-specific preprocessing. A matrix is passed
   through only when its nonlinear transform/standardization is explicitly
   documented; the word “normalized” is not sufficient evidence.
4. Freeze the same cells and ordered shared markers used for every comparison
   method. No method-specific favorable donor or cell subsample is allowed.
5. Record the formula, cofactor, clipping, scope, assay/layer, source reference,
   preprocessing-code hash, and final numerical scale in the v2 manifest.

The v2 runner hashes the frozen cell and marker manifests, verifies that the
cell IDs and order match, and checks that the ordered row-bind of the three
per-dataset matrices exactly reconstructs the combined cyCombine input. The
biological appropriateness of a declared transform remains an auditable study
decision; software cannot infer it from values alone.

## Official cross-platform example

The official v0.2.5 CITE-seq/spectral-flow/CyTOF vignette uses:

| Source | Values actually supplied to cyCombine |
| --- | --- |
| CITE-seq ADT | Seurat CLR with `margin=2`, extracted from the normalized ADT `data` layer. `ScaleData()` was used for clustering, but `scale.data` was **not** supplied to cyCombine. |
| Spectral-flow FCS | `asinh(x / 6000)` in that example. |
| HIMC CyTOF FCS | Despite prior MATLAB normalization, the vignette still derandomizes and applies `asinh(x / 5)`. |

For conventional flow, cofactor 150 is a documented starting point in the
cyCombine workflow. These values are precedents, not automatic defaults for an
unrelated provider matrix. Derandomization is used only if the stored CyTOF
counts still contain randomization; continuous CCT-normalized values must not be
blindly rounded or derandomized.

IMC, CODEX, and MIBI were not part of the paper's three-platform demonstration.
The data-frame API can be applied to them as an exploratory comparator, but
their clipping, segmentation-derived intensity, tissue/ROI normalization, and
transform must follow the exact source study rather than a technology-wide
default.

## Explicitly prohibited in the canonical comparison

- the historical global transform
  `asinh(x / median(all positive values))`;
- choosing a cofactor or assay layer from whichever UMAP looks best;
- applying a second arcsinh or CLR to already transformed values;
- treating `norm_method="rank"` as if the corrected output preserved ranks;
- feeding cell type, survival, outcome, or spatial labels into fitting;
- using donor as a ComBat covariate when donor is nested within dataset;
- correcting non-shared markers through cyCombine panel imputation;
- silently deleting markers or changing the 8×8 SOM after a failure.

Official v0.2.5 rejects a marker whose values have an exact zero sum within a
batch. This is an upstream implementation constraint, not a general definition
of marker degeneracy. Any marker exclusion must be predeclared, justified, and
applied consistently to all shared-marker comparison methods.
