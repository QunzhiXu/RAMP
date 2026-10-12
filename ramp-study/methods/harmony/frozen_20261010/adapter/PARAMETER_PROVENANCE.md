# Parameter provenance

## Locked Harmony 1.2.3 defaults

The ADTnorm wrapper supplied only the marker matrix, batch vector, and
`do_pca=FALSE`; all other values were Harmony defaults. This wrapper passes the
resolved Harmony 1.2.3 defaults explicitly so a future package release cannot
silently change the comparator:

- `theta = 2`
- `sigma = 0.1`
- `lambda = 1`
- `nclust = min(round(n_cells / 30), 100)`
- `max_iter = 10`
- `early_stop = TRUE`
- `ncores = 1`
- `plot_convergence = FALSE`
- `return_object = FALSE`
- `alpha = 0.2`
- `tau = 0`
- `block.size = 0.05`
- `max.iter.cluster = 20`
- `epsilon.cluster = 0.001`
- `epsilon.harmony = 0.01`

The actual source default for `epsilon.harmony` is `0.01`; it should not be
replaced by the separate `1e-4` prose description of early stopping.

## Reproducibility addition

ADTnorm's published wrapper did not set a local seed. This project adds
`seed = 42`, fixes R's RNG kinds to `Mersenne-Twister`, `Inversion`, and
`Rejection`, and uses one core. The seed is an explicit reproducibility envelope,
not a parameter attributed to the ADTnorm authors. Every manifest records that
distinction and the resolved `nclust`.

The batch vector is passed as character, matching the ADTnorm wrapper exactly;
Harmony constructs its internal factor. Configured `batch_levels` validate the
dataset set and order diagnostics, but do not introduce a different factor-level
ordering into the numerical fit.

The canonical runner rejects parameter tuning. A scientifically predeclared
sensitivity analysis may use a separate benchmark-owned configuration and must
not overwrite the canonical comparator.
