# Scope of the completed local reproduction

These statements summarize the accepted local work, not new execution of all
analyses from this source-only checkout.

- **RAMP and comparator outputs:** accepted local re-executions cover the
  declared frozen cohorts for RAMP, cyCombine, Harmony, fastMNN and CytoNorm.
  MaxFuse provider/input preparation was checked, but 17 historical MaxFuse
  representations used by evaluation were not newly fitted.
- **Integration evaluation:** replaying frozen coordinates reproduced the
  metric values. Fresh UMAP reproduced 52 of 63 coordinate arrays exactly;
  11 branches differed. The differences affected 660 visual-metric values and
  four all31 ranks, not non-visual metrics, diagnostics, main23 ranks or its
  overall scores. The remaining integration UMAP causes were not established.
- **Marker-value analysis:** the four technical benchmark value-analysis
  chains were checked, including fastMNN reconstruction. Small serialization
  and floating-point differences retain their original scope.
- **Figure 3:** all 32,370 saved predictions were independently rescored and
  all 32,370 heads were retrained on 860 frozen representations. Prediction
  files matched their historical counterparts. The representations themselves
  were not all re-fitted; head weights were not compared as serialized models.
- **Applications:** CD19, melanoma, lung and HNSCC numerical chains were
  checked within their documented input/analysis boundaries. CD19 retains
  small QC differences and a historical label-file provenance gap. HNSCC
  coordinates reproduced exactly under UMAP 0.5.11, not 0.5.12; this positive
  result does not prove the undocumented original environment version.
- **Final figures:** source/data correspondences and published-page geometry
  were inspected, not every producer rebuilt. S1–S9 header collisions and S8
  caption clipping remain in the existing PDF, which this release does not
  modify or include.

## Scientific qualifications that remain

The source cohorts and some provider-derived inputs remain frozen. The RAMP
melanoma-story alignment was thread-sensitive: historical exact reproduction
used the recorded 14-thread OpenMP condition. The lung technical RAMP
benchmark used its historical 7000/0/7000 pair-fit setting; package defaults
remain 4000/100/8000. Neither was tuned for this publication candidate.

The main23 metric subset and category weighting were post-hoc choices. Six
Figure 3 J1 class-scope revisions were exploratory; the lung keep_c02 arm
followed an observed pilot and was not preregistered. Alternative choices can
change rankings. HNSCC cap20 was not a balanced all-method sensitivity set.
Global/transductive representations and selected ROIs/marker rules do not
become independent prospective validation by being reproducible.

Clinical analyses reuse global RAMP representations and fixed selections;
they are not demonstrations of per-fold representation fitting or newly
validated clinical tools. Some original extraction/width-table producers and
microscopy preparation steps remain outside the reconstructed scope.

No universal clean-install, cross-OS, cross-thread or cross-machine exact
reproduction claim is made. A structural hash check is not an independent
proof of arithmetic correctness or a tamper-proof signature.
