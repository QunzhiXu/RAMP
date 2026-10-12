# External MaxFuse license notice

This directory does **not** contain a copy or modification of the MaxFuse
source code. It contains an independently written, internal research driver
that calls the separately installed `maxfuse==0.0.2` API and independently
implements the numerical procedure documented in two provenance-locked
Archive notebooks. Direct parity tests extract upstream function definitions
only at test time from a separately obtained, hash-verified repository clone.

The upstream repository currently uses the Stanford/Penn Academic Software
License Agreement. It restricts use to internal academic non-commercial work,
restricts redistribution, modification, and integration, and requires
appropriate acknowledgement in publications. Anyone running, sharing, or
redistributing this driver must first review the upstream license and decide
whether their intended use is permitted. This notice is not legal advice and
does not replace the upstream agreement.

Reviewed upstream:

- repository: <https://github.com/shuxiaoc/maxfuse>
- commit: `2900b260b697deec678060ea653015291c4dc5cb`
- package: `maxfuse==0.0.2`
- upstream `LICENSE` SHA-256:
  `7fb73faa0652b79b4e340d2b6314af6715a7122cd267ede7817017112965a9ac`

No license is granted here for MaxFuse itself.
