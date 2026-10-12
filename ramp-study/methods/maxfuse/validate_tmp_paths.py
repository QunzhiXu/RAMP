"""Create-only evidence for a fixture-only macOS temporary-path correction.

Runs synthetic tests under the inherited default TMPDIR and an explicit alias.
Does not alter the production launcher, prepare inputs, or fit any cohort.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

from migrate import HERE, create_json, sha
from run import reserve
from study_inputs import verify_bundle
from verify import verify_local


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.output.resolve().is_relative_to(HERE / "validation"):
        parser.error("Use a new directory under validation")
    snapshot = HERE / "frozen_20261010"
    baseline = HERE / "validation/study_handoff_20261010_r2/report.json"
    old = json.loads(baseline.read_text())
    output = reserve(args.output, (snapshot, baseline, HERE / "runs"))
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1"}
    env.pop("PYTHONPATH", None)
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[key] = "1"
    # Deliberately DO NOT redirect TMPDIR for the default-mode check.
    for key, leaf in [("MPLCONFIGDIR", "matplotlib"), ("NUMBA_CACHE_DIR", "numba"), ("XDG_CACHE_HOME", "xdg")]:
        path = output / "cache" / leaf
        path.mkdir(parents=True)
        env[key] = str(path)
    python = str(args.python.expanduser().absolute())
    report = {"scope": "Fixture-path correction only; synthetic tests, no cohort computation",
              "baseline_report_sha256": sha(baseline), "commands": []}
    try:
        with tempfile.TemporaryDirectory(prefix="maxfuse-path-audit-") as tmp:
            root = Path(tmp).resolve()
            real = root / "physical"
            real.mkdir()
            alias = root / "alias"
            alias.symlink_to(real, target_is_directory=True)
            for mode, environment in (("inherited_default", env), ("explicit_symlink", {**env, "TMPDIR": str(alias)})):
                probe = [python, "-B", "-c", "import json,tempfile; from pathlib import Path; p=tempfile.gettempdir(); print(json.dumps({'tempdir':p,'resolved':str(Path(p).resolve())}))"]
                probe_result = subprocess.run(probe, env=environment, text=True, capture_output=True, check=True)
                paths = json.loads(probe_result.stdout)
                if mode == "explicit_symlink" and paths["tempdir"] == paths["resolved"]:
                    raise RuntimeError("Symlink path was not exercised")
                command = [python, "-B", "-m", "unittest", "discover", "-s", str(HERE / "tests"), "-v"]
                result = subprocess.run(command, cwd=output, env=environment, text=True, capture_output=True)
                report["commands"].append({"mode": mode, "paths": paths, "command": command,
                                            "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
                if result.returncode:
                    raise RuntimeError(f"Tests failed under {mode}")
                print("PASS", mode, paths, flush=True)
        unchanged = []
        for name, expected in old["module_sha256"].items():
            if name == "tests/test_study.py":
                continue
            if sha(HERE / name) != expected:
                raise RuntimeError(f"Unexpected existing-module change: {name}")
            unchanged.append(name)
        report["unchanged_prior_module_files"] = unchanged
        report["copied_sources"] = verify_local(snapshot)
        report["unchanged_bundles"] = []
        for case in old["real_bundles"]:
            bundle = HERE / "runs/study_inputs_20261010_r2" / case["case"]
            if sha(bundle / "MANIFEST.json") != case["bundle_manifest_sha256"]:
                raise RuntimeError("Existing bundle manifest changed")
            manifest = verify_bundle(bundle)
            report["unchanged_bundles"].append({"case": case["case"], "files": len(manifest["outputs"]),
                                                "manifest_sha256": case["bundle_manifest_sha256"]})
        report["changed_test_before_sha256"] = old["module_sha256"]["tests/test_study.py"]
        report["current_files"] = {name: sha(HERE / name) for name in (
            "tests/test_study.py", "validate_tmp_paths.py", "TMP_PATH_FIX_20261010.md")}
        report["status"] = "passed"
        create_json(output / "report.json", report)
    except BaseException as error:
        create_json(output / "FAILED.json", {**report, "error": repr(error)})
        raise
    print(output / "report.json")


if __name__ == "__main__":
    main()
