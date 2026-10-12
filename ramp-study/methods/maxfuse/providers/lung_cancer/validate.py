"""Create-only lung_cancer technical upstream audit; no provider extraction or fit."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess

import lung_support as support
from lung_support import HERE, MAXFUSE, create_json, sha, reserve
from study_inputs import npz_equal, verify_bundle
from verify import matches, verify_sources, within


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("source-root","rebuilt","bundle","provider-python","maxfuse-python","output"):
        p.add_argument("--"+key,type=Path,required=True)
    args=p.parse_args(argv)
    if not args.output.resolve().is_relative_to(HERE/"validation"):
        p.error("Choose a new provider-module validation directory")
    out=reserve(args.output,(args.source_root,args.rebuilt,args.bundle,HERE/"source"))
    study,project=MAXFUSE.parents[1],args.source_root.resolve().parent
    report={"scope":"Lung technical provider/derived inputs to native panels and 3-hub fit inputs only",
            "utc":datetime.now(timezone.utc).isoformat(),"commands":[]}
    env={**os.environ,"PYTHONDONTWRITEBYTECODE":"1"};env.pop("PYTHONPATH",None)
    for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMBA_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
        env[key]="1"
    for key,leaf in (("MPLCONFIGDIR","matplotlib"),("NUMBA_CACHE_DIR","numba"),("XDG_CACHE_HOME","xdg")):
        d=out/"cache"/leaf;d.mkdir(parents=True);env[key]=str(d)
    # Do not redirect TMPDIR: retain the macOS alias regression coverage.
    def command(cmd):
        result=subprocess.run(list(map(str,cmd)),cwd=out,env=env,capture_output=True,text=True)
        report["commands"].append({"command":list(map(str,cmd)),"returncode":result.returncode,
                                    "stdout":result.stdout,"stderr":result.stderr})
        if result.returncode:raise RuntimeError("Validation command failed")
        print("PASS",cmd[-1],flush=True)
    try:
        for interpreter,tests in ((args.provider_python,HERE/"tests"),
                                  (args.provider_python,MAXFUSE/"providers/bone_marrow/tests"),
                                  (args.provider_python,MAXFUSE/"providers/tnbc/tests"),
                                  (args.provider_python,MAXFUSE/"providers/melanoma/tests"),
                                  (args.maxfuse_python,MAXFUSE/"tests")):
            command([interpreter.absolute(),"-B","-m","unittest","discover","-s",tests,"-v"])
        command([args.provider_python.absolute(),"-B",study/"benchmarks/integration_inputs/inputs.py",
                 "verify-staged","--input-root",study/"benchmarks/integration_inputs/runs/frozen_inputs_20261010"])
        command([args.provider_python.absolute(),"-B","-c","import json,platform,numpy,pandas; print(json.dumps({'python':platform.python_version(),'numpy':numpy.__version__,'pandas':pandas.__version__}))"])
        reference=args.source_root/"benchmark_results/lung_cancer/maxfuse_v2"
        copies={"source/rebuild_source_active.py":"rebuild_source_active.py","source/build_inputs.py":"build_inputs.py",
                "source/historical_source_manifest.json":"source_active/manifest.json",
                "source/historical_method_manifest.json":"method_inputs/manifest.json",
                "config.json":"config.json","pairwise_config.json":"pairwise_config.json"}
        support.check_code()
        for name,original in copies.items():
            if sha(HERE/name)!=sha(reference/original):raise ValueError("Original source changed: "+original)
        request=json.loads((args.rebuilt/"REQUEST.json").read_text())
        complete=json.loads((args.rebuilt/"COMPLETE.json").read_text())
        if request["launcher_sha256"]!=sha(HERE/"rebuild.py") or complete["status"]!="complete":
            raise ValueError("Incomplete or different two-stage launcher")
        report["provider_stages"]={}
        differences={"source":["recipe","recipe_sha256"],"method":["recipe","recipe_sha256","source_active_manifest","source_active_manifest_sha256","source_active_recipe","source_active_recipe_sha256"]}
        for stage,folder in (("source","source_active"),("method","method_inputs")):
            root=args.rebuilt/folder
            actual,old=support.verify_reference(root,stage),support.history(stage)
            req=json.loads((root/"REQUEST.json").read_text());done=json.loads((root/"COMPLETE.json").read_text())
            if done.get("stage")!=stage or done.get("status")!="complete" or done["manifest_sha256"]!=sha(root/"manifest.json"):
                raise ValueError("Stage incomplete")
            if complete["stage_manifests"][folder]!=sha(root/"manifest.json"):raise ValueError("Driver/stage differs")
            for name,digest in req["code_sha256"].items():
                if sha(HERE/name)!=digest:raise ValueError("Executed provider code differs")
            if req["source_sha256"]!=support.SOURCE_HASHES:raise ValueError("Source snapshot changed")
            support.check_inputs(Path(req["raw_root"]) if req["raw_root"] else None,Path(req["benchmark_root"]))
            for name,digest in actual["outputs"].items():
                if sha(reference/folder/name)!=digest:raise ValueError("Historical output changed")
            changed=sorted(k for k in old.keys()|actual.keys() if old.get(k)!=actual.get(k))
            if changed!=differences[stage]:raise ValueError("Unexpected stage manifest changes")
            report["provider_stages"][stage]={"root":str(root.resolve()),"outputs":actual["outputs"],
                "manifest_sha256":sha(root/"manifest.json"),"manifest_differences":changed,
                "request_sha256":sha(root/"REQUEST.json"),"complete_sha256":sha(root/"COMPLETE.json"),
                "dataset_records_equal":old["datasets"]==actual["datasets"]}
        report["source_inputs"]={"provider_or_derived_files":support.input_records()[0],"canonical_files":support.input_records()[1]}
        print("PASS 6 provider/derived files, 6 canonical inputs, 9+6 outputs",flush=True)

        previous=json.loads((MAXFUSE/"validation/study_handoff_20261010_r2/report.json").read_text())
        report["unchanged_bundles"]=[]
        for row in previous["real_bundles"]:
            root=MAXFUSE/"runs/study_inputs_20261010_r2"/row["case"]
            if sha(root/"MANIFEST.json")!=row["bundle_manifest_sha256"]:raise ValueError("Old bundle changed")
            m=verify_bundle(root)
            report["unchanged_bundles"].append({"case":row["case"],"files":len(m["outputs"]),"manifest_sha256":row["bundle_manifest_sha256"]})
        oldroot=MAXFUSE/"runs/study_inputs_20261010_r2/lung_cancer"
        old,new=verify_bundle(oldroot),verify_bundle(args.bundle)
        if set(old["outputs"])!=set(new["outputs"]):raise ValueError("Bundle inventory changed")
        for name,record in old["outputs"].items():matches(args.bundle/name,record)
        for branch in new["branches"]:
            relative=Path("branches")/branch["name"]/"fit_input.npz"
            npz_equal(args.bundle/relative,oldroot/relative)
            if branch["cache"] is None:raise ValueError("Historical lung_cancer replay cache was omitted")
        origins={k:v["read_from_provider_rebuild"] for k,v in new["source_files"].items() if "read_from_provider_rebuild" in v}
        routing=new["native_input_override"];native=args.rebuilt/"method_inputs"
        if len(origins)!=6 or set(origins)!=set(routing["file_mapping"]):raise ValueError("Wrong native routing")
        for name,path in origins.items():
            actual=within(native,routing["file_mapping"][name])
            if Path(path).resolve(strict=True)!=actual:raise ValueError("Did not consume new native input")
            matches(actual,new["source_files"][name])
        if routing["manifest_sha256"]!=sha(native/"manifest.json") or routing["completion_sha256"]!=sha(native/"COMPLETE.json"):
            raise ValueError("Consumed provider provenance changed")
        report["bridge"]={"root":str(args.bundle.resolve()),"manifest_sha256":sha(args.bundle/"MANIFEST.json"),
            "cells":new["cells"],"fit_arrays_equal":45,"byte_identical_payload_files":len(new["outputs"]),
            "native_origins":origins,"branches":[b["name"] for b in new["branches"]],"fit_performed":False}

        report["prior_providers_unchanged"]=[]
        for case in ("bone_marrow","tnbc"):
            path=MAXFUSE/"providers"/case/"validation/provider_handoff_20261010/report.json"
            prior=json.loads(path.read_text());prefix=f"methods/maxfuse/providers/{case}/"
            codes={n:h for n,h in prior["current_code_sha256"].items() if n.startswith(prefix)}
            for name,digest in codes.items():
                if sha(study/name)!=digest:raise ValueError("Accepted provider module changed")
            p=prior["provider_run"];root=Path(p["root"])
            outputs=p["byte_identical_files"] if case=="bone_marrow" else p["byte_identical_outputs"]
            for name,digest in outputs.items():
                if sha(root/name)!=digest:raise ValueError("Prior provider outputs changed")
            for name,key in (("REQUEST.json","request_sha256"),("COMPLETE.json","completion_sha256")):
                if sha(root/name)!=p[key]:raise ValueError("Prior provider provenance changed")
            manifest_name="manifest.json" if case=="bone_marrow" else "input_manifest.json"
            if sha(root/manifest_name)!=json.loads((root/"COMPLETE.json").read_text())["manifest_sha256"]:raise ValueError("Prior provider manifest changed")
            bridge=prior["bridge"];root=Path(bridge["bundle"] if case=="bone_marrow" else bridge["root"])
            if sha(root/"MANIFEST.json")!=bridge["manifest_sha256"]:raise ValueError("Prior bridge changed")
            verify_bundle(root)
            report["prior_providers_unchanged"].append({"case":case,"code_files":len(codes),"outputs":len(outputs),"report_sha256":sha(path)})
        
        # Melanoma was independently accepted with the later one-predicate status fix.
        prior_path=MAXFUSE/"providers/melanoma/validation/provider_handoff_20261010/report.json"
        fixed_path=MAXFUSE/"providers/melanoma/validation/status_fix_20261010_r2/report.json"
        prior=json.loads(prior_path.read_text());fixed=json.loads(fixed_path.read_text())
        if sha(prior_path)!=fixed["baseline_report_sha256"]:
            raise ValueError("Melanoma historical report changed")
        codes={n:h for n,h in prior["current_code_sha256"].items() if n.startswith("methods/maxfuse/providers/melanoma/")}
        for name,digest in fixed["current_files"].items():
            codes["methods/maxfuse/providers/melanoma/"+name]=digest
        for name,digest in codes.items():
            if sha(study/name)!=digest:raise ValueError("Accepted melanoma provider changed")
        for record in prior["provider_stages"].values():
            root=Path(record["root"])
            for name,digest in record["outputs"].items():
                if sha(root/name)!=digest:raise ValueError("Melanoma output changed")
            for name,key in (("manifest.json","manifest_sha256"),("REQUEST.json","request_sha256"),("COMPLETE.json","complete_sha256")):
                if sha(root/name)!=record[key]:raise ValueError("Melanoma provenance changed")
        bridge=prior["bridge"];root=Path(bridge["root"])
        if sha(root/"MANIFEST.json")!=bridge["manifest_sha256"]:raise ValueError("Melanoma bridge changed")
        verify_bundle(root)
        report["prior_providers_unchanged"].append({"case":"melanoma","code_files":len(codes),"outputs":15,
            "report_sha256":sha(prior_path),"status_fix_report_sha256":sha(fixed_path)})
        report["historical_checks"]=verify_sources(MAXFUSE/"frozen_20261010",args.source_root)
        report["baselines"]=[]
        for path,base in ((study/"inventory_20261010/evidence/source_census.json",project),
                          (study/"inventory_20261010/evidence/curated_sources.json",project),
                          (study/"benchmarks/bone_marrow/validation/code_snapshot_output_safety_20261010.json",study)):
            count=0
            for row in json.loads(path.read_text())["files"]:
                if row.get("sha256"):matches(within(base,row["path"]),row);count+=1
            report["baselines"].append({"snapshot":str(path),"sha256":sha(path),"checked":count})
        allowed={"README.md","STUDY_WORKFLOW.md","study.py","study_inputs.py","tests/test_study.py"}
        for name,digest in previous["module_sha256"].items():
            if name not in allowed and sha(MAXFUSE/name)!=digest:raise ValueError("Unexpected existing MaxFuse code change")
        fixture=MAXFUSE/"validation/tmp_path_fix_20261010/report.json"
        for name,digest in json.loads(fixture.read_text())["current_files"].items():
            if sha(MAXFUSE/name)!=digest:raise ValueError("Previously accepted fixture fix changed")
        files=[f for f in HERE.rglob("*") if f.is_file() and not ({"runs","validation","__pycache__"}&set(f.relative_to(HERE).parts))]
        files += [MAXFUSE/n for n in ("study.py","study_inputs.py","README.md","STUDY_WORKFLOW.md","tests/test_provider_bridge.py","tests/test_lung_provider_bridge.py")]
        files += [study/"README.md"]
        report["current_code_sha256"]={str(f.relative_to(study)):sha(f) for f in sorted(files)}
        report["tests_passed"]=sum(int(n) for c in report["commands"] for n in re.findall(r"Ran (\d+) tests? in",c["stderr"]))
        report["status"]="passed";create_json(out/"report.json",report)
    except BaseException as error:
        create_json(out/"FAILED.json",{**report,"error":repr(error)});raise
    print(out/"report.json",flush=True)


if __name__=="__main__":main()
