# Public release checklist

- [x] Verify the paper author list and correspondence contact against the preprint.
- [x] Record the owner's choice of CC BY-NC-ND 4.0 for the algorithm package.
- [x] Add the official `LICENSE` and update `LEGAL_STATUS.md` and package metadata.
- [ ] Confirm any required institutional release authority and distribution permissions;
  the owner's reported supervisor decision is recorded, not independently certified.
- [x] Record the bioRxiv preprint and public repository URLs in package metadata.
- [ ] Publish this audited package to <https://github.com/QunzhiXu/RAMP>.
- [ ] Confirm the GitHub CI jobs pass after the first push.
- [ ] Confirm the distribution name on PyPI immediately before release.
- [x] Build and inspect both wheel and source distribution.
- [x] Test installation of the built wheel outside the source checkout.
- [x] Run the complete test suite under the exact manuscript dependency lock
  (113 tests passed against both source and installed wheel).
- [ ] Archive benchmark manifests and package version with the manuscript.
- [ ] Create a signed/tagged release matching the published algorithm version.
