# Input format

## Expression matrix

The command-line interface accepts a NumPy `.npy` matrix with cells in rows
and shared protein markers in columns. It must be finite and two-dimensional.
The marker order must exactly match `markers.txt`.

RAMP expects the highest-available source-scale values distributed by each
study. “Source scale” does not necessarily mean untouched instrument output;
the provenance of provider processing should be reported with each dataset.

## Marker file

`markers.txt` contains one unique marker name per line and at least two
markers. Empty lines are ignored.

## Cell metadata

The metadata file must have exactly one row per matrix row, in the same order.
CSV, compressed CSV, TSV, compressed TSV, and (with the optional `parquet`
dependency) Parquet are accepted.

For CSV and TSV input, all columns are read as strings so identifiers such as
`001` or ROI `0007` survive round trips exactly. Cast non-fitting annotation
columns to numeric types later in the downstream analysis if needed. Parquet
retains its stored column types.

Recommended schema:

| Column | Required | Used for fitting | Meaning |
|---|---|---|---|
| `cell_id` | recommended | no | unique stable cell identifier |
| `dataset` | yes | yes | platform/cohort identifier |
| `donor_id` | recommended | no | globally unique biological donor identity |
| `unit_key` | yes | yes | dataset-specific fitting/preprocessing unit |
| other annotations | optional | no | preserved for downstream use |

If `cell_id` is absent, the CLI generates deterministic row IDs. Generated
IDs describe only the current row order and cannot recover an external cell
mapping, so publication analyses should provide stable IDs. When provided,
`cell_id` must be unique and non-missing. `donor_id`, when provided, must be
non-missing. Identity values cannot be blank or padded with whitespace.

`unit_key` must belong to exactly one dataset. This is intentionally stricter
than donor identity. For a donor observed on multiple platforms, use:

```text
dataset       donor_id       unit_key
CODEX         P01            CODEX::P01
CyTOF         P01            CyTOF::P01
```

The shared `donor_id` records biological correspondence; the distinct
`unit_key` values prevent platform-specific cells from being combined during
within-unit preprocessing and rank calculations.

Use the same `donor_id` across datasets only when the rows represent the same
biological donor. If local source files reuse identifiers such as `P01` for
unrelated cohorts, namespace them before input (for example, `cohortA::P01`).

## Choosing the fitting unit

`unit_key` is the unit within which RAMP preprocesses marker distributions and
enforces zero strict rank inversions. It is also the independent unit used
when RAMP aggregates provider evidence.

- Prefer one unit per donor/patient within each dataset; the public default is
  `unit_key = dataset::donor_id`.
- Multiple ROIs from the same donor should normally share that donor-level
  unit, with ROI identity retained in a separate annotation column.
- `unit_key` is also the replicate used when RAMP aggregates evidence. Use an
  ROI- or sample-level unit only when that unit is genuinely the independent
  biological replicate for the intended analysis, not merely because the
  tissue was split into several fields of view.
- Do not create one unit per cell or use cell-type labels as units.

The CLI reports both unit and donor counts in the run manifest when
`donor_id` is available.

## Donor-level attributes

The row-aligned cell metadata records donor membership through `donor_id`.
Do not duplicate donor-level clinical attributes across every cell. Keep age,
sex, treatment, response, survival, and similar variables in a separate table
with one row per donor, keyed by `donor_id`. This donor table is joined only in
downstream analyses and is never passed to RAMP.

A recommended study layout is:

```text
study/
  input/
    shared_markers.npy
    markers.txt
    cell_metadata.csv.gz
  clinical/
    donor_metadata.csv
  results/
    ramp/
```

This separation makes the fitting boundary explicit: the algorithm receives
the expression matrix plus row-aligned `dataset` and `unit_key`; `donor_id` is
retained for provenance, while clinical variables remain outside the fit.

## Dataset requirements

- Exactly two or three unique datasets.
- At least 11 cells per dataset.
- At least two cells per unit.
- At least one shared marker with non-zero within-unit information after
  KMEANSLOW preprocessing.
- Dataset names must remain distinct after punctuation and spaces are
  normalized for audit-column names (for example, do not combine `A-B` and
  `A B` in one run).
- Dataset names supplied through `--datasets` must exactly match the metadata
  values; their order is frozen in the manifest.

The 11-cell dataset minimum is a computational lower bound, not a recommended
study size. Although profile screening begins at 40 curve-role cells, the
adaptive-bin rule needs at least four bins of 15 cells, so an actually fitted
curve requires at least 60 curve-role cells. Because that role receives
approximately 48% of cells, roughly 125 cells per fitting unit is the expected
bare practical minimum, and more is recommended. The Python API warns when no
profiles are learned;
the CLI writes the auditable identity-only result and records that status in
the manifest rather than silently implying that a correction was learned.

## Privacy

Use deidentified donor identifiers. RAMP runs locally and does not transmit
metadata, but output metadata intentionally preserves the input identifiers to
maintain a reliable cell-to-result mapping.
