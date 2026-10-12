# Code configuration and external resources

The source and parameter configuration belong in the code distribution.
Measurements, identities, predictions, images and full execution evidence are
separate resources. This separation avoids copying gigabytes of results while
preserving the code needed to produce and inspect them.

## Included

All accounted-for existing author code, analysis/figure producers and helpers,
scientific parameter and task-suite configuration, method version/source
records, tests, necessary directory relationships and aggregate reference
tables. Source is byte-preserved. Task configurations include the original
grids, candidate parameters and pseudonymous sample/unit identifiers. Original
source licensing is stated in `../LICENSE_SCOPE.md`. The specific MaxFuse
license-scope question is recorded in `../THIRD_PARTY_NOTICES.md`; it does not
automatically apply to every configuration or aggregate result mentioning the
method.

## External

| Resource | Examples | Required for |
| --- | --- | --- |
| Provider measurements and permissions | FCS, H5AD, RDS, raw CSV, expression matrices | Provider preparation and scientific fitting |
| Cohort and identity data | Cell IDs, patient/donor metadata, frozen selections and input hashes | Matching the paper's exact cohorts/order |
| Clinical and spatial resources | Outcomes, clinical covariates, microscopy images and geometry | Authorized application/figure workflows |
| Numerical method artifacts | Aligned arrays, models, saved predictions and intermediate tables | Replay or downstream stages when not regenerated |
| Full historical evidence contracts | Large catalogs, result manifests, complete bundles and old report anchors | Original strict audit/replay commands |
| Dependency installations | Python environments, R libraries and upstream source archives | Executing the relevant method |
| Non-code figure assets | Acquired microscopy and authored schematic artwork | Exact final figure assembly |

Obtain these through the original providers and authorized custodians. No
download URL or permission is invented here. `research/` preserves the original
code hierarchy, but source availability does not supply data access rights.

## Use the correct scientific route

The lung technical benchmark uses Monkman CODEX; the lung story uses Zheng
mIF. Both names and input scales matter. Harmony story consumes raw inputs;
cyCombine and fastMNN story use their declared analysis-ready inputs. The
Figure 3 formal suite retains historical grids and must be combined with its
documented J1 revisions and current lung route, not treated as a replacement
for the published release selection. See the code maps and ENTRYPOINTS.json.

Do not use an old figure layout, an old cohort or a historical fit result as a
substitute merely because a filename matches. Do not replace fixed hashes to
force an incomplete audit tree to pass. Work in a separate data/output tree;
keep the preserved sources unchanged, and record any necessary path adaptation.

The source checks establish that code/configuration was included and preserved.
They do not rerun clinical analyses, prove a clean installation of every method
or prove the completeness of an undocumented upstream data producer.
