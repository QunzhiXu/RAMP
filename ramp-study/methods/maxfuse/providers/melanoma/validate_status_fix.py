"""Create-only P3 status-check evidence; no provider extraction or cohort fitting."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from unittest.mock import patch

import build_inputs as method
from melanoma_support import HERE, MAXFUSE, create_json, reserve, sha
from study_inputs import verify_bundle
from verify import matches, verify_local, within


BASELINE_SHA = "98e33b4e1f4b2325602a7f8ba86a574b583ba9bef48fa4295ae4417832af875a"


def require_hash(path, expected):
    if sha(path) != expected:
        raise ValueError(f"Changed file: {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider-python", type=Path, required=True)
    parser.add_argument("--maxfuse-python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    baseline = HERE / "validation/provider_handoff_20261010/report.json"
    require_hash(baseline, BASELINE_SHA)
    previous = json.loads(baseline.read_text())
    study = MAXFUSE.parents[1]
    if not args.output.resolve().is_relative_to(HERE / "validation"):
        parser.error("Use a new directory under melanoma/validation")
    output = reserve(args.output, (baseline.parent, HERE / "runs", MAXFUSE / "runs", HERE / "source"))
    report = {"scope": "Source completion status guard only; no extraction or real MaxFuse fit",
              "utc": datetime.now(timezone.utc).isoformat(),
              "baseline_report_sha256": BASELINE_SHA, "commands": []}
    try:
        prefix = "methods/maxfuse/providers/melanoma/"
        allowed = {prefix + "build_inputs.py", prefix + "tests/test_provider.py"}
        changed = {}
        for name, digest in previous["current_code_sha256"].items():
            current = sha(study / name)
            if name in allowed:
                changed[name] = {"before": digest, "after": current}
            elif current != digest:
                raise ValueError("Unexpected module change: " + name)
        # Removing just the added predicate must reconstruct the exact audited file.
        addition = b'complete.get("status") != "complete" or '
        current = (HERE / "build_inputs.py").read_bytes()
        if current.count(addition) != 1:
            raise ValueError("Unexpected status predicate diff")
        restored = current.replace(addition, b"", 1)
        if hashlib.sha256(restored).hexdigest() != previous["current_code_sha256"][prefix + "build_inputs.py"]:
            raise ValueError("Production changes extend beyond the single status predicate")
        report["code_delta"] = {"changed_files": changed, "single_predicate_only": True,
                                "other_prior_files_unchanged": len(previous["current_code_sha256"]) - 2}

        report["provider_stages_unchanged"] = []
        for stage, record in previous["provider_stages"].items():
            root = Path(record["root"])
            for name, digest in record["outputs"].items():
                require_hash(within(root, name), digest)
            for name, key in (("manifest.json", "manifest_sha256"), ("REQUEST.json", "request_sha256"),
                              ("COMPLETE.json", "complete_sha256")):
                require_hash(root / name, record[key])
            report["provider_stages_unchanged"].append({"stage": stage, "files": len(record["outputs"]),
                                                       "provenance_unchanged": True})
        source = Path(previous["provider_stages"]["source"]["root"])
        with patch.object(method, "SOURCE_ACTIVE", source):
            method.ensure_source_active()  # Real existing inputs, no mocked verifier, read-only.
        report["real_completed_source_accepted"] = True

        bundles = [(MAXFUSE / "runs/study_inputs_20261010_r2" / row["case"], row["manifest_sha256"])
                   for row in previous["unchanged_bundles"]]
        bundles.append((Path(previous["bridge"]["root"]), previous["bridge"]["manifest_sha256"]))
        report["bundles_unchanged"] = []
        for root, digest in bundles:
            require_hash(root / "MANIFEST.json", digest)
            manifest = verify_bundle(root)
            report["bundles_unchanged"].append({"root": str(root), "files": len(manifest["outputs"]),
                                                "manifest_sha256": digest})
        report["copied_sources"] = verify_local(MAXFUSE / "frozen_20261010")
        report["baselines_unchanged"] = []
        for item in previous["baselines"]:
            path = Path(item["snapshot"])
            require_hash(path, item["sha256"])
            base = study if path.name == "code_snapshot_output_safety_20261010.json" else study.parent
            count = 0
            for row in json.loads(path.read_text())["files"]:
                if row.get("sha256"):
                    matches(within(base, row["path"]), row)
                    count += 1
            if count != item["checked"]:
                raise ValueError("Baseline count changed")
            report["baselines_unchanged"].append(item)

        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        env.pop("PYTHONPATH", None)
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            env[key] = "1"
        for key, name in (("MPLCONFIGDIR", "matplotlib"), ("NUMBA_CACHE_DIR", "numba"), ("XDG_CACHE_HOME", "xdg")):
            path = output / "cache" / name
            path.mkdir(parents=True)
            env[key] = str(path)
        # Inherit TMPDIR, including macOS /var -> /private/var aliases.
        for python, directory in ((args.provider_python, HERE / "tests"),
                                  (args.provider_python, MAXFUSE / "providers/bone_marrow/tests"),
                                  (args.provider_python, MAXFUSE / "providers/tnbc/tests"),
                                  (args.maxfuse_python, MAXFUSE / "tests")):
            command = [str(python.absolute()), "-B", "-m", "unittest", "discover", "-s", str(directory), "-v"]
            result = subprocess.run(command, cwd=output, env=env, capture_output=True, text=True)
            counts = re.findall(r"Ran (\d+) tests? in", result.stderr)
            record = {"command": command, "returncode": result.returncode,
                      "stdout": result.stdout, "stderr": result.stderr}
            report["commands"].append(record)
            if result.returncode or len(counts) != 1 or "skipped=" in result.stderr:
                raise RuntimeError("Test suite failed or skipped tests")
            record["tests_passed"] = int(counts[0])
            print("PASS", directory, counts[0], flush=True)
        report["tests_passed"] = sum(c["tests_passed"] for c in report["commands"])
        require_hash(baseline, BASELINE_SHA)
        report["current_files"] = {name: sha(HERE / name) for name in (
            "build_inputs.py", "tests/test_provider.py", "validate_status_fix.py", "STATUS_FIX_20261010.md")}
        report["status"] = "passed"
        create_json(output / "report.json", report)
    except BaseException as error:
        create_json(output / "FAILED.json", {**report, "error": repr(error)})
        raise
    print(output / "report.json", flush=True)


if __name__ == "__main__":
    main()
