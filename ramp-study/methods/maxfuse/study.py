"""Seven-task entry: prepare frozen full-native inputs, verify, or explicitly fit.

Preparation verifies historical NPZ arrays/configs but does not fit MaxFuse.
Only the fit subcommand performs numerical alignment. No overwrite/resume.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess

from migrate import HERE, create_json, sha
from run import reserve
from study_inputs import SNAPSHOT, prepare, verify_bundle
from verify import verify_local


def fit(python, bundle, output, route, branch="all"):
    python = Path(python).expanduser().absolute()  # Preserve virtualenv symlink.
    if not python.is_file():
        raise FileNotFoundError(python)
    bundle = Path(bundle).resolve(strict=True)
    bundle_hash = sha(bundle / "MANIFEST.json")
    manifest = verify_bundle(bundle)
    if sha(bundle / "MANIFEST.json") != bundle_hash:
        raise RuntimeError("Bundle manifest changed during initial verification")
    verify_local(SNAPSHOT)
    branches = manifest["branches"] if branch == "all" else [b for b in manifest["branches"] if b["name"] == branch]
    if not branches:
        raise ValueError(f"Unknown branch: {branch}")
    if route not in ("fresh", "replay"):
        raise ValueError("Choose fresh or replay explicitly")
    if route == "replay" and any(b["cache"] is None for b in branches):
        raise ValueError("Selected case lacks a frozen triplet cache; no silent fallback to fresh fitting")
    root = reserve(output, (bundle, SNAPSHOT, python))
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1"}
    env.pop("PYTHONPATH", None)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[key] = "1"
    for key, leaf in [("TMPDIR", "tmp"), ("NUMBA_CACHE_DIR", "cache/numba"),
                      ("MPLCONFIGDIR", "cache/matplotlib"), ("XDG_CACHE_HOME", "cache/xdg")]:
        directory = root / leaf
        directory.mkdir(parents=True)
        env[key] = str(directory)
    try:
        # Actual upstream source and all 56 environment versions, before any fit.
        with (root / "runtime.json").open("x") as stdout, (root / "runtime.stderr").open("x") as stderr:
            check = subprocess.run([str(python), "-B", str(HERE / "probe_runtime.py"), "--adapter", str(SNAPSHOT / "adapter")],
                                   cwd=root, env=env, stdout=stdout, stderr=stderr)
        if check.returncode:
            raise RuntimeError("Locked MaxFuse runtime verification failed")
        for b in branches:
            if sha(bundle / "MANIFEST.json") != bundle_hash:
                raise RuntimeError("Bundle manifest changed before branch execution")
            directory = root / b["name"]
            directory.mkdir(exist_ok=False)
            request = {"bundle": str(bundle), "branch": b["name"], "route": route,
                       "bundle_manifest_sha256": bundle_hash}
            create_json(directory / "REQUEST.json", request)
            command = [str(python), "-B", str(HERE / "study_worker.py"), "--bundle", str(bundle),
                       "--branch", b["name"], "--route", route, "--output", str(directory)]
            with (directory / "stdout.log").open("x") as stdout, (directory / "stderr.log").open("x") as stderr:
                result = subprocess.run(command, cwd=directory, env=env, stdout=stdout, stderr=stderr)
            if result.returncode:
                raise RuntimeError(f"{b['name']} failed; see retained logs")
            done = json.loads((directory / "COMPLETE.json").read_text())
            if done["bundle_manifest_sha256"] != request["bundle_manifest_sha256"] or done["route"] != route:
                raise RuntimeError("Worker completion provenance mismatch")
            if done["values_sha256"] != sha(directory / "values.npy") or done["cell_ids_sha256"] != sha(directory / "cell_ids.npy"):
                raise RuntimeError("Worker completion output hash mismatch")
        verify_bundle(bundle)
        if sha(bundle / "MANIFEST.json") != bundle_hash:
            raise RuntimeError("Bundle manifest changed during execution")
        create_json(root / "COMPLETE.json", {"case": manifest["case"], "route": route,
                    "branches": [b["name"] for b in branches],
                    "branch_manifests": {b["name"]: sha(root / b["name"] / "COMPLETE.json") for b in branches}})
    except BaseException as error:
        create_json(root / "FAILED.json", {"error": repr(error), "partial_output_retained": True})
        raise
    return root


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="List seven frozen tasks; read-only")
    prep = sub.add_parser("prepare", help="Assemble fit inputs without fitting; requires new output")
    prep.add_argument("--case", required=True)
    prep.add_argument("--source-root", type=Path, required=True, help="Relocatable tree of the locked native inputs, not necessarily the original final_xu")
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--with-replay-cache", action="store_true")
    prep.add_argument("--native-root", type=Path, help="Verified provider rebuild for any of the seven benchmark cases; native files must match historical hashes")
    verify = sub.add_parser("verify", help="Verify a moved fit bundle; no source tree needed")
    verify.add_argument("--bundle", type=Path, required=True)
    run = sub.add_parser("fit", help="Explicit expensive numerical computation; never called by prepare/verify")
    run.add_argument("--python", type=Path, required=True)
    run.add_argument("--bundle", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--route", choices=["fresh", "replay"], required=True)
    run.add_argument("--branch", default="all")
    args = p.parse_args(argv)
    if args.command == "list":
        verify_local(SNAPSHOT)
        j = json.loads((SNAPSHOT / "catalog.json").read_text())
        print(json.dumps([{k: c[k] for k in ("case", "mode")} for c in j["cases"]], indent=2))
    elif args.command == "prepare":
        print(prepare(args.case, args.source_root, args.output, args.with_replay_cache, args.native_root))
    elif args.command == "verify":
        m = verify_bundle(args.bundle)
        print(json.dumps({"status": "passed", "case": m["case"], "files": len(m["outputs"]),
                          "branches": len(m["branches"]), "original_source_tree_required": False}))
    else:
        print(fit(args.python, args.bundle, args.output, args.route, args.branch))


if __name__ == "__main__":
    main()
