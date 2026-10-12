"""Create a new validation report; runs only synthetic/structural checks."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

from migrate import HERE, create_json, sha
from run import reserve
from verify import verify_sources


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--python", type=Path, required=True, help="Compatible existing MaxFuse Python, no installation")
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    if not args.output.resolve().is_relative_to(HERE / "validation"):
        p.error("Use a new directory under this module's validation/")
    snapshot = HERE / "frozen_20261010"
    output = reserve(args.output, (args.source_root, snapshot))
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1"}
    env.pop("PYTHONPATH", None)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[key] = "1"
    for key, dirname in [("NUMBA_CACHE_DIR", "cache/numba"), ("MPLCONFIGDIR", "cache/matplotlib"),
                         ("XDG_CACHE_HOME", "cache/xdg"), ("TMPDIR", "tmp")]:
        folder = output / dirname
        folder.mkdir(parents=True, exist_ok=True)
        env[key] = str(folder)
    python = str(args.python.absolute())
    adapter = snapshot / "adapter"
    commands = [
        [python, "-B", "-m", "unittest", "discover", "-s", str(HERE / "tests"), "-v"],
        [python, "-B", str(HERE / "probe_runtime.py"), "--adapter", str(adapter)],
    ]
    commands += [[python, "-B", str(adapter / "tests" / name)] for name in [
        "smoke_test_core.py", "smoke_test_archive_gcca.py", "smoke_test_pairwise_adapter.py", "smoke_test_threeway_logic.py"]]
    commands += [[python, "-B", str(adapter / name), "--help"] for name in [
        "run_maxfuse_pairwise.py", "run_maxfuse_three_view_extension.py"]]
    report = {"captured_utc": datetime.now(timezone.utc).isoformat(), "commands": [],
              "scope": "Existing source/result hashes and IDs; relocated-core synthetic tests. No benchmark refit, no raw-data copy, no clean-environment install. Fake Fusor call-sequence test is not a real pairwise parity test."}
    for cmd in commands:
        result = subprocess.run(cmd, cwd=output, env=env, capture_output=True, text=True)
        report["commands"].append({"command": cmd, "returncode": result.returncode,
                                  "stdout": result.stdout, "stderr": result.stderr})
        if result.returncode:
            create_json(output / "FAILED.json", report)
            raise RuntimeError(f"Validation failed: {cmd}")
        print("Passed", Path(cmd[2]).name if len(cmd)>2 else cmd[0], flush=True)
    report["source_and_result_verification"] = verify_sources(snapshot, args.source_root)
    report["module_sha256"] = {str(x.relative_to(HERE)): sha(x) for x in sorted(HERE.glob("*")) if x.is_file()}
    report["module_sha256"].update({str(x.relative_to(HERE)): sha(x) for x in sorted((HERE / "tests").glob("*.py"))})
    report["status"] = "passed"
    create_json(output / "report.json", report)
    print(output / "report.json", flush=True)


if __name__ == "__main__":
    main()
