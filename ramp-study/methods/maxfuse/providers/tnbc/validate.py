"""Create-only TNBC provider-port evidence; no extraction, sampling or fitting."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess

import tnbc_support as support
from tnbc_support import HERE, MAXFUSE, create_json, sha
from run import reserve
from study_inputs import npz_equal, verify_bundle
from verify import matches, verify_sources, within


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("source-root", "rebuilt", "bundle", "provider-python", "maxfuse-python", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    args = p.parse_args(argv)
    if not args.output.resolve().is_relative_to(HERE / "validation"):
        p.error("Choose a new directory under this module's validation/")
    out = reserve(args.output, (args.source_root, args.rebuilt, args.bundle, HERE / "source"))
    study, project = MAXFUSE.parents[1], args.source_root.resolve().parent
    report = {"scope": "TNBC provider/row-lock to native panels and three-hub input assembly only",
              "utc": datetime.now(timezone.utc).isoformat(), "commands": []}
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    env.pop("PYTHONPATH", None)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[key] = "1"
    for key, leaf in (("MPLCONFIGDIR", "matplotlib"), ("NUMBA_CACHE_DIR", "numba"), ("XDG_CACHE_HOME", "xdg")):
        d = out / "cache" / leaf
        d.mkdir(parents=True)
        env[key] = str(d)
    # Inherit the default TMPDIR, including macOS symlink aliases.
    def command(cmd):
        result = subprocess.run(list(map(str, cmd)), cwd=out, env=env, capture_output=True, text=True)
        report["commands"].append({"command": list(map(str,cmd)), "returncode": result.returncode,
                                   "stdout": result.stdout, "stderr": result.stderr})
        if result.returncode:
            raise RuntimeError("Validation command failed")
        print("PASS", cmd[-1], flush=True)
    try:
        for interpreter, tests in ((args.provider_python, HERE / "tests"),
                                   (args.provider_python, MAXFUSE / "providers/bone_marrow/tests"),
                                   (args.maxfuse_python, MAXFUSE / "tests")):
            command([interpreter.absolute(), "-B", "-m", "unittest", "discover", "-s", tests, "-v"])
        command([args.provider_python.absolute(), "-B", study / "benchmarks/integration_inputs/inputs.py",
                 "verify-staged", "--input-root", study / "benchmarks/integration_inputs/runs/frozen_inputs_20261010"])
        command([shutil.which("Rscript"), "-e", 'cat(R.version.string, "\\n"); for(p in c("SingleCellExperiment","rhdf5")) cat(p, as.character(packageVersion(p)), "\\n")'])
        old, new = support.check_code(), support.verify_reference(args.rebuilt)
        reference = args.source_root / "benchmark_results/tnbc/maxfuse_archive"
        for local, original in support.SNAPSHOTS.items():
            if sha(HERE / local) != sha(reference / original):
                raise ValueError("Historical source changed: " + original)
        for name, digest in new["output_sha256"].items():
            if sha(reference / "input" / name) != digest:
                raise ValueError("Original TNBC input changed: " + name)
        request = json.loads((args.rebuilt / "REQUEST.json").read_text())
        complete = json.loads((args.rebuilt / "COMPLETE.json").read_text())
        for name, key in (("build_inputs.py", "builder_sha256"), ("tnbc_support.py", "support_sha256")):
            if sha(HERE / name) != request[key]:
                raise ValueError("Provider run code changed")
        for name, digest in request["source_sha256"].items():
            if sha(HERE / name) != digest: raise ValueError("Provider run source snapshot changed")
        if sha(Path(request["cell_lock"])) != request["cell_lock_sha256"]:
            raise ValueError("Row lock changed")
        if complete["status"] != "complete" or complete["manifest_sha256"] != sha(args.rebuilt / "input_manifest.json"):
            raise ValueError("Incomplete provider run")
        changed = sorted(k for k in old.keys() | new.keys() if old.get(k) != new.get(k))
        if changed != ["builder", "builder_sha256", "input_protocol", "source_files", "writes_confined_to"]:
            raise ValueError("Unexpected TNBC manifest changes")
        report["provider_run"] = {"root":str(args.rebuilt.resolve()), "cells":62701,
            "source_files":new["source_files"], "byte_identical_outputs":new["output_sha256"],
            "manifest_differences":changed, "dataset_records_equal":old["datasets"]==new["datasets"],
            "software":new["software"], "request_sha256":sha(args.rebuilt / "REQUEST.json"),
            "completion_sha256":sha(args.rebuilt / "COMPLETE.json")}
        print("PASS provider sources, lock and 11 outputs", flush=True)

        previous = json.loads((MAXFUSE / "validation/study_handoff_20261010_r2/report.json").read_text())
        report["unchanged_bundles"] = []
        for row in previous["real_bundles"]:
            bundle = MAXFUSE / "runs/study_inputs_20261010_r2" / row["case"]
            if sha(bundle / "MANIFEST.json") != row["bundle_manifest_sha256"]:
                raise ValueError("Old bundle manifest changed")
            m = verify_bundle(bundle)
            report["unchanged_bundles"].append({"case":row["case"], "files":len(m["outputs"]),
                                                "manifest_sha256":row["bundle_manifest_sha256"]})
        bundle = verify_bundle(args.bundle)
        previous_tnbc = MAXFUSE / "runs/study_inputs_20261010_r2/tnbc"
        original = verify_bundle(previous_tnbc)
        if set(bundle["outputs"]) != set(original["outputs"]): raise ValueError("Bundle inventory differs")
        for name, record in original["outputs"].items(): matches(args.bundle / name,record)
        for b in bundle["branches"]:
            f = Path("branches") / b["name"] / "fit_input.npz"
            npz_equal(args.bundle / f, previous_tnbc / f)
            if b["cache"] is not None: raise ValueError("TNBC should not request the historical replay route")
        origins = {k:v["read_from_provider_rebuild"] for k,v in bundle["source_files"].items() if "read_from_provider_rebuild" in v}
        routing = bundle["native_input_override"]
        if len(origins)!=7 or set(origins)!=set(routing["file_mapping"]): raise ValueError("Wrong routing inventory")
        for name,path in origins.items():
            expected = within(args.rebuilt,routing["file_mapping"][name])
            if Path(path).resolve(strict=True)!=expected: raise ValueError("Wrong native source path")
            matches(expected,bundle["source_files"][name])
        if routing["manifest_sha256"]!=sha(args.rebuilt/"input_manifest.json") or routing["completion_sha256"]!=sha(args.rebuilt/"COMPLETE.json"):
            raise ValueError("Provider manifest differs from consumed input")
        report["bridge"] = {"cells":bundle["cells"], "root":str(args.bundle.resolve()),
            "manifest_sha256":sha(args.bundle/"MANIFEST.json"), "branches":[b["name"] for b in bundle["branches"]],
            "fit_arrays_equal":45, "payload_files_byte_identical":len(bundle["outputs"]),
            "native_origins":origins, "fit_performed":False}
        print("PASS three hubs and 7 native sources",flush=True)

        bm_report_path = MAXFUSE / "providers/bone_marrow/validation/provider_handoff_20261010/report.json"
        bm = json.loads(bm_report_path.read_text())
        bm_codes = {n:h for n,h in bm["current_code_sha256"].items() if n.startswith("methods/maxfuse/providers/bone_marrow/")}
        for name,digest in bm_codes.items():
            if sha(study/name)!=digest: raise ValueError("Accepted BM provider module changed")
        bm_root = Path(bm["provider_run"]["root"])
        for name,digest in bm["provider_run"]["byte_identical_files"].items():
            if sha(bm_root/name)!=digest: raise ValueError("Accepted BM provider result changed")
        for name,key in (("REQUEST.json","request_sha256"),("COMPLETE.json","completion_sha256")):
            if sha(bm_root/name)!=bm["provider_run"][key]: raise ValueError("BM provider provenance changed")
        if sha(bm_root/"manifest.json")!=json.loads((bm_root/"COMPLETE.json").read_text())["manifest_sha256"]:
            raise ValueError("BM provider manifest changed")
        bm_bundle=Path(bm["bridge"]["bundle"])
        if sha(bm_bundle/"MANIFEST.json")!=bm["bridge"]["manifest_sha256"]: raise ValueError("BM bridge changed")
        verify_bundle(bm_bundle)
        report["bm_unchanged"]={"provider_code_files":len(bm_codes),"provider_outputs":10,"bridge_payload_files":25,
                                "reference_report_sha256":sha(bm_report_path)}
        report["historical_checks"] = verify_sources(MAXFUSE/"frozen_20261010",args.source_root)
        report["baselines"]=[]
        for snapshot,base in ((study/"inventory_20261010/evidence/source_census.json",project),
                              (study/"inventory_20261010/evidence/curated_sources.json",project),
                              (study/"benchmarks/bone_marrow/validation/code_snapshot_output_safety_20261010.json",study)):
            count=0
            for row in json.loads(snapshot.read_text())["files"]:
                if row.get("sha256"):
                    matches(within(base,row["path"]),row); count+=1
            report["baselines"].append({"snapshot":str(snapshot),"sha256":sha(snapshot),"checked":count})
        allowed={"README.md","STUDY_WORKFLOW.md","study.py","study_inputs.py","tests/test_study.py"}
        for name,digest in previous["module_sha256"].items():
            if name not in allowed and sha(MAXFUSE/name)!=digest: raise ValueError("Unexpected existing MaxFuse code change")
        fixture = MAXFUSE/"validation/tmp_path_fix_20261010/report.json"
        for name,digest in json.loads(fixture.read_text())["current_files"].items():
            if sha(MAXFUSE/name)!=digest: raise ValueError("Prior fixture fix changed")
        files=[f for f in HERE.rglob("*") if f.is_file() and not ({"runs","validation","__pycache__"}&set(f.relative_to(HERE).parts))]
        files += [MAXFUSE/n for n in ("study.py","study_inputs.py","README.md","STUDY_WORKFLOW.md","tests/test_provider_bridge.py","tests/test_tnbc_provider_bridge.py")]
        files += [study/"README.md"]
        report["current_code_sha256"]={str(f.relative_to(study)):sha(f) for f in sorted(files)}
        report["status"]="passed"
        create_json(out/"report.json",report)
    except BaseException as error:
        create_json(out/"FAILED.json",{**report,"error":repr(error)})
        raise
    print(out/"report.json",flush=True)


if __name__=="__main__":
    main()
