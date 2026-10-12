# Contributing

Please report issues and suggestions. The package's CC BY-NC-ND 4.0 license
does not grant permission to distribute adapted versions. Before submitting
a patch or publishing a modified fork, obtain appropriate permission from
the rights holder; this guide does not grant additional rights. See
`LEGAL_STATUS.md` and `LICENSE`.

For authorized development, changes to file I/O, documentation, diagnostics,
and tests follow the usual checks.
Changes to the numerical algorithm require:

1. a new algorithm version;
2. exact regression against RAMP 2.3.0 under the frozen dependency lock;
3. verification of zero strict rank inversions;
4. verification of the 0.75 displacement ceiling;
5. benchmark evaluation using the manuscript protocol.

Do not commit provider raw data, protected health information, generated
benchmark matrices, or manuscript-only analysis outputs to this package.

The frozen `_algorithm.py` and `_helpers.py` sources are excluded from
automatic formatting so style-only changes cannot disturb provenance. Lint
and format new wrapper, I/O, CLI, test, and example code normally.
