"""Test-only filesystem isolation; production output_path is not made configurable."""
from contextlib import contextmanager
from pathlib import Path
import os
import tempfile
from unittest.mock import patch

from trainability import core


@contextmanager
def temporary_artifacts():
    # Direct unittest runs default to the OS temp directory. The check runner
    # may point test scratch space under its own report/sandbox directory.
    with tempfile.TemporaryDirectory(dir=os.environ.get('TRAINABILITY_TEST_TMPDIR')) as td:
        root = Path(td).resolve()
        with patch.object(core, 'ROOT', root):
            yield root
