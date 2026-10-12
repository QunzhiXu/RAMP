# Parameter provenance

## Version

The canonical implementation is pinned to `cyCombine v0.2.5`, commit
`5b5903c1accc887b4a11c2b25c550a0b0e36876d`. The official release page labels
this version “cyCombine as used in publication.” The locally available 0.3.0
installation is not used by this runner because its public API and internal
implementation have expanded since publication.

The exact official source archive and the publication's `renv.lock` are stored
under `vendor/`. `upstream_lock.json` records their hashes and the hashes of the
two source files that implement preprocessing and correction.

The implementation has been smoke-tested on the current host with R 4.3.2,
`kohonen 3.0.12`, and `sva 3.50.0`; its full `sessionInfo()` is recorded under
`.Rlib/`. The
publication lock instead records R 4.1.1, `kohonen 3.0.10`, and `sva 3.40.0`.
Accordingly, results should be described as official cyCombine v0.2.5 run with
recorded runtime dependencies, not as a byte-identical reconstruction of the
paper's complete software environment. The runner rehashes the vendored archive
and the installed bodies of the four core functions on every run, but the
current-host dependency stack is recorded rather than fully containerized.

## Locked cross-platform profile

The official cross-platform vignette combined CyTOF, CITE-seq, and spectral
flow in one native three-batch run. This wrapper locks the vignette-explicit
settings plus the unchanged v0.2.5 defaults:

- exactly three datasets/platforms are supplied, each as one `batch`;
- only ordered shared protein markers are corrected;
- a single 8×8 Kohonen SOM is used;
- `seed=840` is explicitly set in the official cross-platform vignette;
- `rlen=10`, Euclidean SOM distance, and `parametric=TRUE` are unchanged v0.2.5 defaults;
- SOM construction uses within-batch ranks and average handling of ties;
- the original method-input expression values are corrected within SOM nodes;
- `covar=NULL` and `anchor=NULL`;
- the R 4.1.1 default RNG kinds are fixed as Mersenne-Twister/Inversion/Rejection;
- donor/sample IDs are retained for row provenance but are not modeled;
- no provider labels or clinical outcomes enter fitting.

The formal runner accepts only these values. There is no 8→6→4 fallback. If a
benchmark requires a different predeclared sensitivity analysis, it must live
beside that benchmark and must not be presented as the canonical publication
profile.

## Cell counts

The paper downsampled each of its three demonstration datasets to 6,776 cells,
the size of the smallest cleaned dataset. That number is a property of the
demonstration, not a cyCombine algorithm parameter. Our comparisons should use
the already frozen benchmark cells shared across methods. Any balanced sampling
decision belongs to the benchmark-data configuration rather than this folder.

## Output semantics

The output is a corrected shared-marker expression matrix. cyCombine uses ranks
only for the pre-correction SOM; it then applies ComBat to the supplied
expression scale and caps corrected values to the input range within each SOM
node. Therefore cyCombine is a deliberately non-rank-preserving comparison
method.

`correction_som_assignments.csv.gz` contains that pre-correction rank SOM and
must not be mistaken for the post-correction SOM used by the paper's EMD
evaluation. The runner also reports every node as empty, single-batch
passthrough, multi-batch mean-only ComBat, or full ComBat. `sva::ComBat`
automatically switches the whole node to mean-only mode when any participating
batch contributes only one cell.

## Interpretation boundary

The publication example used balanced healthy-PBMC samples. In the cancer
benchmarks, platform/dataset is also cohort, and donors are nested within
dataset. With `covar=NULL`, technical and true biological cohort effects are not
identifiable. This implementation is therefore a faithful label-free comparison
method, not evidence that cyCombine has isolated a purely technical effect.
Applications to IMC, CODEX, or MIBI should be described as exploratory
cross-assay use outside the paper's demonstrated three technologies.
