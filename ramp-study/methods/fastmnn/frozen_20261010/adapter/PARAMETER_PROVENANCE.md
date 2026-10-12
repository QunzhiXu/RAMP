# fastMNN parameter provenance

The canonical profile follows the batchelor 1.18.1 defaults. The only additions
are a fixed seed and `get.variance=TRUE` for diagnostics. The wrapper passes
the official `d=50` unchanged and records the dimensionality that batchelor
actually realizes for each small panel.

| Parameter | Canonical value | Provenance / meaning |
|---|---:|---|
| `k` | `20` | batchelor 1.18.1 default; approximate minimum shared-population size; official small-batch handling is left unchanged |
| `prop.k` | `NULL` | official default; no cell-count-dependent inflation |
| `cos.norm` | `TRUE` | official default; cell-wise cosine/L2 normalization |
| `ndist` | `3` | official default tricube bandwidth multiplier |
| requested `d` | `50` | official default |
| realized `d` | `min(50, p-1, n-1)` | batchelor 1.18.1 + default `IrlbaParam()` realization; the call still receives `50`; current benchmarks have `n≫p` |
| `deferred` | `TRUE` | official default computational setting |
| `weights` | `NULL` | official default batch-balanced `multiBatchPCA()` |
| `get.variance` | `TRUE` | diagnostic-only addition; does not alter corrected coordinates |
| `merge.order` | `NULL` | official default |
| `auto.merge` | `FALSE` | official default progressive merge in configured input order |
| `min.batch.skip` | `0` | official default; no merge is skipped by threshold |
| `subset.row` | `NULL` | all shared markers are used |
| `correct.all` | `FALSE` | official default; reconstructed assay is not exported |
| `BSPARAM` | `IrlbaParam()` | official default PCA solver |
| `BNPARAM` | `KmknnParam()` | official default neighbour solver |
| `BPPARAM` | `SerialParam()` | official default serial execution |
| RNG seed | `42` | reproducibility addition; not a biological parameter |

The batch order is data provenance rather than an algorithm parameter. It is
declared as `batch_levels` in each benchmark config and becomes the progressive
merge order. It must be fixed before post-hoc cell-type evaluation.

Any non-default sensitivity run - e.g. `auto.merge=TRUE`, a larger `k`,
`prop.k`, `min.batch.skip>0`, a different order, or `d=NA` - must use a
separate benchmark-owned configuration and result directory. It must not
overwrite or be called the canonical fastMNN result.
