# Third-party dependencies and retained notices

The CC-BY-NC-ND-4.0 declaration in [LICENSE_SCOPE.md](LICENSE_SCOPE.md) covers
original RAMP-author material only. It does not replace the following terms.
No installed Python/R environment, upstream source archive or MaxFuse library
source is included in this source candidate. Users obtain dependencies
separately and comply with their original terms.

| Component | Recorded terms | Retained source of the record |
| --- | --- | --- |
| MaxFuse 0.0.2 | Stanford/Penn Academic Software License Agreement | `third_party/maxfuse/LICENSE`; `research/final_xu/algorithm/maxfuse/LICENSE_NOTICE.md` and `upstream_lock.json` |
| cyCombine | MIT | `research/final_xu/algorithm/cycombine/vendor/LICENSE-cyCombine.md` |
| Harmony 1.2.3 | GPL-3 | `research/final_xu/algorithm/harmony/vendor/LICENSE_NOTICE.md` |
| batchelor 1.18.1 (fastMNN) | GPL-3 | `research/final_xu/algorithm/fastmnn/vendor/LICENSE_NOTICE.md` |
| CytofRUV | GPL-3 | `research/final_xu/algorithm/cytofruv/vendor/LICENSE_NOTICE.md` |
| Other dependencies | Their own package terms | Existing `upstream_lock.json`, runtime locks and dependency package metadata |

These are the terms recorded for the frozen research dependencies, not an
assertion about newer versions. Their package metadata and upstream licenses
are authoritative. Notices in frozen source copies are also retained without
changes. Some historical vendor notices describe archives that existed in the
research environment; those archives are external to this source release.

## Specific MaxFuse question before publishing the complete tree

The reviewed upstream commit is
`2900b260b697deec678060ea653015291c4dc5cb`. Its [LICENSE](https://github.com/shuxiaoc/maxfuse/blob/2900b260b697deec678060ea653015291c4dc5cb/LICENSE)
has SHA-256
`7fb73faa0652b79b4e340d2b6314af6715a7122cd267ede7817017112965a9ac`.
The retained license is byte-identical to that pinned upstream file.

Sections 6 and 9 restrict redistribution of the Software. Section 7 also
claims programs based on it as derivatives, and section 8 prohibits
integration of the Software into the recipient's own software. The applicability
of these broad terms to the independently written RAMP research drivers has
not been resolved by the source audit.

The concrete connection is in
`research/final_xu/algorithm/maxfuse/maxfuse_repro/pairwise.py`, whose
`_load_upstream()` imports `maxfuse.model.Fusor`, and
`research/final_xu/algorithm/maxfuse/maxfuse_repro/gcca.py`, which follows the
locked Archive notebook procedure and optionally calls MaxFuse's SVD helper.
Corresponding frozen and historical source copies are indexed in
`SOURCE_COVERAGE.json`. Configurations or files that merely mention MaxFuse
have not individually been declared prohibited material.

No MaxFuse upstream library file or copied notebook is present. That rules out
one redistribution issue, but does not by itself settle sections 7 and 8.
Keeping a copy of the license does not grant permission beyond its terms.
This is a specific unresolved license-scope question, not a finding that
every MaxFuse-related file infringes copyright or is forbidden to publish.
