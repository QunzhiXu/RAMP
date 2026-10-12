#!/usr/bin/env python3
"""Install locked source packages offline without changing the user's R library."""
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
here = Path(__file__).resolve().parent
temporary = here.parents[1] / "algorithm_test/cytonorm_validation/tmp"
temporary.mkdir(parents=True, exist_ok=True)
link = Path("/private/tmp" if Path("/private/tmp").exists() else "/tmp") / f"cytonorm-bootstrap-{os.getpid()}"
link.symlink_to(temporary, target_is_directory=True)
try:
    subprocess.run(["Rscript", "--vanilla", str(here / "bootstrap_library.R")],
                   env=dict(os.environ, TMPDIR=str(link)), check=True)
finally:
    link.unlink()
