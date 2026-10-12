# Environment references

The algorithm package has general dependency ranges plus an exact historical
numerical requirements file. The study used multiple environments; one UMAP
version must not be advertised as exact for every application.

| Component | Local reference condition |
| --- | --- |
| Canonical Python numerical work | Python 3.14.1, NumPy 2.4.0, pandas 2.3.3, SciPy 1.16.3, scikit-learn 1.8.0 |
| Integration UMAP | 0.5.11; 11/63 fresh layouts still differed |
| CD19 and melanoma application UMAP | 0.5.11 |
| HNSCC exact UMAP diagnostic | 0.5.11; 0.5.12 differed |
| Lung application UMAP | 0.5.12 |
| R correction work | R 4.3.2 and method-specific recorded package locks |

Selected method `upstream_lock.json` files contain exact package/version/source
references and hashes. The upstream archives are not shipped or downloaded by
this candidate. Their URLs are historical provenance, not a newly tested
availability guarantee. Retain the applicable upstream notices when obtaining
and using those packages.

`cycombine-direct.json` and `cytonorm-runtime.json` contain package/version
pairs derived from runtime tables, with host library paths removed. They are
not byte-identical replacements for the original locked files and cannot be
substituted into a historical validator. Harmony/fastMNN reference version
tables remain in their method source directories.

Do not package an existing `.venv` or `.Rlib` as a portable installation.
Clean installation of all research environments has not been completed in
this candidate. Local package installation testing reuses existing numerical
dependencies and is explicitly not a clean-room study installation.
