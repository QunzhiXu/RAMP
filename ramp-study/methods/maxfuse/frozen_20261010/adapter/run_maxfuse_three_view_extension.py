#!/usr/bin/env python3
"""Canonical CLI for the official-Archive three-view MaxFuse workflow.

The implementation is shared with the legacy filename
``run_maxfuse_threeway.py``; both now execute pairwise hub-pivot chaining plus
the released HuBMAP Archive iterative GCCA procedure.
"""

from run_maxfuse_threeway import main


if __name__ == "__main__":
    main()
