# Changelog

## 2.3.0 — initial publication package

- Package the audited RAMP 2.3.0 numerical implementation.
- Add the `ramp_proteomics` Python API and `ramp-proteomics align` CLI.
- Preserve row-aligned cell and donor metadata beside aligned matrices.
- Add input validation, run manifests, source hashes, tests, and examples.
- Record the exact manuscript dependency environment.
- Add the associated bioRxiv citation and public repository metadata.
- Fix hard-link input/output collision detection and stage output files before
  replacing them without truncating shared inodes.
- Convert matrix inputs to float64 before wrapper validation; reject blank
  fitting identities consistently in the Python and CLI interfaces.
- Clarify reserved-cell participation in upstream correspondence construction.
- Add a required numerical regression job using the frozen dependency lock.
