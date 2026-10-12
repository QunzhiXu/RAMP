# GitHub publication preparation

The author has explicitly selected **CC-BY-NC-ND-4.0** for original code in
both `ramp-proteomics` and `ramp-study`, with third-party terms preserved, and
requested checks followed by GitHub preparation if no concrete restriction
is found. The original-code license decision is complete. No additional
institutional approval requirement has been established by this review.

## Content review

The source candidate was screened for private keys, credential/token literals,
credential-bearing URLs, direct identity fields, confidential-data notices,
non-text payloads and large files. The initial 3,251-file screen found no such
matches. Manual review covered all 17 CSV schemas, four environment logs and
22 JSON files with sample/unit identifier fields. They contain package locks,
aggregate benchmark statistics and pseudonymous cohort selections, rather than
patient outcome tables or direct patient identifiers. Historical local paths
remain in scientific sources and are documented portability limitations.

Pattern screening cannot prove absence of all sensitive information. No
specific forbidden data file or credential has been identified in this review.
Original measurements, images, models, predictions, installed environments and
manuscript PDFs are absent. Scientific code and original third-party notices
have not been changed.

## One specific unresolved question

The frozen MaxFuse license has additional sections 7 and 8 concerning programs
based on MaxFuse and integration into other software. The release includes
research drivers that import its API and follow a locked notebook procedure.
Whether those provisions restrict public distribution of these independently
written drivers has not been established. See exact source paths, upstream
commit, license hash and clauses in
[THIRD_PARTY_NOTICES.md](ramp-study/THIRD_PARTY_NOTICES.md).

The full tree is therefore not marked cleared for publication. No finding of
infringement is made, and ordinary configuration/reference files are not deemed
prohibited solely because they mention MaxFuse. Existing written permission or
a determination of the agreement's scope could resolve this question; a public
subset omitting affected drivers would need an explicit, accurate scope record.
The complete local source has been preserved.

## Upload mechanics

The intended repository is <https://github.com/QunzhiXu/RAMP>. Its public API
reported an empty repository with default branch `main` during preparation.
No commit, push, GitHub release or PyPI upload was performed. The original
source-completeness and numerical-reproduction boundaries remain unchanged;
having a CI workflow is not evidence that GitHub CI has run.
