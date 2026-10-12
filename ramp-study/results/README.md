# Aggregate reference tables

These nine CSV files are exact copies of selected previously checked aggregate
tables. They contain benchmark, method, metric, family and aggregate-score/rank
fields, not cell identities, patient identities, predictions or clinical rows.
Their source hashes are recorded in `../SOURCE_MANIFEST.json`.

## Integration

`all31_ranks.csv` has 1,302 rows and `main23_ranks.csv` has 966 rows. They come
from the frozen-coordinate metric replay, not the fresh-UMAP variant. The
main23 subset and category-balanced weighting were post-hoc reporting choices;
the broader sensitivity context must not be discarded.

## Trainability

Seven tables come from the full-head combined aggregation: `primary_means`,
`PER_BENCHMARK`, `OVERALL`, and J1/J2 per-benchmark and overall summaries.
**Do not average the `ALL` rows in `primary_means.csv` as though they were an
additional benchmark.** They are inherited upstream pooled summaries, not the
published benchmark-equal means. Use the explicit overall/per-benchmark
tables and the historical nested aggregation logic for published scores.

These tables retain the published scope, including exploratory J1 revisions
and lung keep_c02 selection. Numerical agreement does not remove those
qualifications. Including MaxFuse summary rows does not include MaxFuse
implementation code or newly fit embeddings.

Distribution of summaries remains subject to the author/data decisions in the
root release gates. The full local audit reports and per-job tables are not
bundled.
