# Source provenance

## Associated preprint

This package accompanies the bioRxiv preprint:

> Qunzhi Xu, Tian Le, Zhicheng Luo, Christopher Ly, Yinqiao Yan, and Ye Zheng.
> “Rank-Preserving Alignment Enables Cross-platform Learning and Phenotyping
> for Single-Cell and Spatial Proteomics.” bioRxiv, version 1, posted October
> 1, 2026. DOI: [10.64898/2026.09.25.754510](https://doi.org/10.64898/2026.09.25.754510).

The paper identifies <https://github.com/QunzhiXu/RAMP> as the public code
repository, and its Methods explicitly identify the described implementation
as RAMP version 2.3.0. The source hashes below identify the audited local RAMP
2.3.0 implementation packaged here; they do not imply that the current
contents of the public repository have the same hashes.

## Numerical source

The packaged numerical implementation was derived from the audited local
RAMP 2.3.0 sources without changing its mathematical call path.

| Role | Manuscript source | SHA-256 |
|---|---|---|
| frozen low-level helpers | `final_xu/algorithm/ramp/ramp.py` | `96845657ec9dbe652073cf7d2e5666c96141168c03604d6ba7a13075d5b4969d` |
| canonical algorithm | `final_xu/algorithm/ramp/ramp_ver2.py` | `2501055cf269e383179b602bd99ff39171c8d283bc6332cd4143eed30c4a68c1` |
| historical file runner | `final_xu/algorithm/ramp/run_ramp.py` | `3829aae72f7df09b0bc829be73e362da6ac0786af5b1ee442b192e7f25ed7ffb` |

Packaging changes are limited to module names/import wiring, user-facing I/O,
manifest generation, documentation, and tests. `tests/test_source_equivalence.py`
compares the packaged and manuscript implementations exactly when the original
source tree is available.

The canonical implementation defines pair-fit minimum/per-unit/maximum as
`4000 / 100 / 8000`. One archived lung benchmark applied an explicit
`7000 / 0 / 7000` runtime override to preserve a historical accepted result.
That exception belongs to the benchmark protocol, not the canonical package;
it is disclosed in `docs/reproducibility.md`.
