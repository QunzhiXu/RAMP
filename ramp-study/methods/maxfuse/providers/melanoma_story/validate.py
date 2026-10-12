"""Create-only Melanoma story migration evidence; tests/hash checks, no extraction or fit."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess

import melanoma_story_support as support
from melanoma_story_support import HERE, MAXFUSE, create_json, reserve, sha
from study_inputs import npz_equal, verify_bundle
from verify import matches, verify_sources, within


def same(path,digest):
    if sha(path)!=digest:raise ValueError("File changed: "+str(path))


def prior_providers(study):
    records=[]
    report_hashes={
        "bone_marrow":"edf2929900ca7d593af8790300fe331d580845888ea1ff92f086aa9c973cff77",
        "tnbc":"ff31b214b386901f2efb094b24ce5412e4156f428fd4dd2975428b963e9d862e",
        "melanoma":"98e33b4e1f4b2325602a7f8ba86a574b583ba9bef48fa4295ae4417832af875a",
        "lung_cancer":"8426fb8b19afee973aef23e449921c99c1892325dc18feff6184f1ac72e37a33",
        "hnscc_story":"5592e58107d8c65d5763a27abf149cfa8f9d761065ef135a7f64cf66b4f2041d",
    }
    for case,digest in report_hashes.items():
        path=MAXFUSE/"providers"/case/"validation/provider_handoff_20261010/report.json"
        same(path,digest);prior=json.loads(path.read_text())
        prefix=f"methods/maxfuse/providers/{case}/"
        codes={n:h for n,h in prior["current_code_sha256"].items() if n.startswith(prefix)}
        if case=="melanoma":
            fixed_path=path.parents[1]/"status_fix_20261010_r2/report.json"
            same(fixed_path,"fbf7b36c42849c221e94ecc7facc5df0cca1a1ee10d85a608960b5eb2a3372f5")
            fixed=json.loads(fixed_path.read_text())
            codes.update({prefix+n:h for n,h in fixed["current_files"].items()})
        for name,h in codes.items():same(study/name,h)
        count=0
        if "provider_stages" in prior:
            for record in prior["provider_stages"].values():
                root=Path(record["root"])
                for name,h in record["outputs"].items():same(root/name,h);count+=1
                for name,key in (("manifest.json","manifest_sha256"),("REQUEST.json","request_sha256"),("COMPLETE.json","complete_sha256")):
                    same(root/name,record[key])
        elif case=="hnscc_story":
            record=prior["provider_run"];root=Path(record["root"])
            for name,h in record["byte_identical_outputs"].items():same(root/"data"/name,h);count+=1
            for name,key in (("data/manifest.json","manifest_sha256"),("REQUEST.json","request_sha256"),
                             ("COMPLETE.json","completion_sha256"),("data/COMPLETE.json","data_completion_sha256"),
                             ("run_config.json","config_sha256")):same(root/name,record[key])
        else:
            record=prior["provider_run"];root=Path(record["root"])
            key="byte_identical_files" if case=="bone_marrow" else "byte_identical_outputs"
            for name,h in record[key].items():same(root/name,h);count+=1
            same(root/"REQUEST.json",record["request_sha256"])
            same(root/"COMPLETE.json",record["completion_sha256"])
            manifest="manifest.json" if case=="bone_marrow" else "input_manifest.json"
            same(root/manifest,json.loads((root/"COMPLETE.json").read_text())["manifest_sha256"])
        bridge=prior["bridge"];root=Path(bridge["bundle"] if case=="bone_marrow" else bridge["root"])
        same(root/"MANIFEST.json",bridge["manifest_sha256"]);verify_bundle(root)
        records.append({"case":case,"existing_code_files_unchanged":len(codes),"outputs_unchanged":count,"report_sha256":digest})
    return records


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("source-root","rebuilt","bundle","provider-python","maxfuse-python","output"):
        parser.add_argument("--"+key,type=Path,required=True)
    args=parser.parse_args(argv)
    if not args.output.resolve().is_relative_to(HERE/"validation"):
        parser.error("Use a new Melanoma story validation directory")
    output=reserve(args.output,(args.source_root,args.rebuilt,args.bundle,HERE/"source"))
    study=MAXFUSE.parents[1]
    report={"utc":datetime.now(timezone.utc).isoformat(),"scope":"Melanoma story provider/native-input migration and pairwise fit-input assembly only","commands":[]}
    env={**os.environ,"PYTHONDONTWRITEBYTECODE":"1"};env.pop("PYTHONPATH",None)
    for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMBA_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):env[key]="1"
    for key,leaf in (("MPLCONFIGDIR","matplotlib"),("NUMBA_CACHE_DIR","numba"),("XDG_CACHE_HOME","xdg")):
        path=output/"cache"/leaf;path.mkdir(parents=True);env[key]=str(path)
    # Keep the inherited macOS TMPDIR and the provider environment's user-site.
    def command(cmd):
        result=subprocess.run(list(map(str,cmd)),cwd=output,env=env,capture_output=True,text=True)
        report["commands"].append({"command":list(map(str,cmd)),"returncode":result.returncode,"stdout":result.stdout,"stderr":result.stderr})
        if result.returncode:raise RuntimeError("Validation command failed")
        print("PASS",cmd[-1],flush=True)
    try:
        for case in ("melanoma_story","hnscc_story","bone_marrow","tnbc","melanoma","lung_cancer"):
            command([args.provider_python.absolute(),"-B","-m","unittest","discover","-s",MAXFUSE/"providers"/case/"tests","-v"])
        command([args.maxfuse_python.absolute(),"-B","-m","unittest","discover","-s",MAXFUSE/"tests","-v"])
        command([args.provider_python.absolute(),"-B",study/"benchmarks/integration_inputs/inputs.py","verify-staged","--input-root",study/"benchmarks/integration_inputs/runs/frozen_inputs_20261010"])
        command([args.provider_python.absolute(),"-B","-c","import json,platform,numpy,pandas,h5py,scipy; print(json.dumps({'python':platform.python_version(),'numpy':numpy.__version__,'pandas':pandas.__version__,'h5py':h5py.__version__,'scipy':scipy.__version__}))"])
        support.check_code()
        reference=args.source_root/"figure_melanoma/Comparison/maxfuse_full_panel"
        copies={"source/prepare.py":reference/"scripts/prepare.py",
                "source/inputs/cohort_package/build_story_sample.py":reference.parents[1]/"inputs/cohort_package/build_story_sample.py",
                "source/historical_manifest.json":reference/"data/manifest.json",
                "source/comparison_config.json":reference.parent/"run_config.json",
                "source/comparison_input_manifest.json":reference.parent/"input/manifest.json",
                "source/full_run_config.json":reference/"run_config.json"}
        for name,path in copies.items():same(path,sha(HERE/name))
        request=json.loads((args.rebuilt/"REQUEST.json").read_text())
        for name,digest in request["code_sha256"].items():same(HERE/name,digest)
        if request["source_sha256"]!=support.SOURCE_HASHES:raise ValueError("Source snapshot changed")
        for record in request["inputs"].values():matches(Path(record["path"]),record)
        manifest=support.verify_reference(args.rebuilt/"data")
        complete=json.loads((args.rebuilt/"data/COMPLETE.json").read_text())
        if (complete.get("status"),complete.get("case"),complete.get("stage"),complete.get("cells"))!=("complete","melanoma_story","native",165686):
            raise ValueError("Incomplete native stage")
        same(args.rebuilt/"data/manifest.json",complete["manifest_sha256"])
        done=json.loads((args.rebuilt/"COMPLETE.json").read_text())
        if {k:v for k,v in done.items() if k!="config_sha256"}!=complete:raise ValueError("Root/data completion differs")
        same(args.rebuilt/"run_config.json",done["config_sha256"])
        expected_sources={r["path"]:r["sha256"] for k,r in request["inputs"].items() if k!="comparison_manifest"}
        expected_sources[str(HERE/"source/inputs/cohort_package/build_story_sample.py")]=support.SOURCE_HASHES["source/inputs/cohort_package/build_story_sample.py"]
        if manifest["source_sha256"]!=expected_sources:raise ValueError("Unexpected read provenance")
        for name,digest in manifest["output_sha256"].items():same(reference/"data"/name,digest)
        report["provider_run"]={"root":str(args.rebuilt.resolve()),"byte_identical_outputs":manifest["output_sha256"],
            "inputs":request["inputs"],"manifest_sha256":sha(args.rebuilt/"data/manifest.json"),
            "request_sha256":sha(args.rebuilt/"REQUEST.json"),"completion_sha256":sha(args.rebuilt/"COMPLETE.json"),
            "data_completion_sha256":sha(args.rebuilt/"data/COMPLETE.json"),"config_sha256":sha(args.rebuilt/"run_config.json"),
            "views_equal":manifest["views"]==support.history()["views"],"manifest_difference_keys":["source_sha256"]}

        previous=json.loads((MAXFUSE/"validation/study_handoff_20261010_r2/report.json").read_text())
        report["unchanged_bundles"]=[]
        for row in previous["real_bundles"]:
            root=MAXFUSE/"runs/study_inputs_20261010_r2"/row["case"]
            same(root/"MANIFEST.json",row["bundle_manifest_sha256"])
            m=verify_bundle(root);report["unchanged_bundles"].append({"case":row["case"],"files":len(m["outputs"]),"manifest_sha256":sha(root/"MANIFEST.json")})
        oldroot=MAXFUSE/"runs/study_inputs_20261010_r2/melanoma_story"
        old,new=verify_bundle(oldroot),verify_bundle(args.bundle)
        if set(new["outputs"])!=set(old["outputs"]):raise ValueError("Bundle inventory changed")
        for name,record in old["outputs"].items():matches(args.bundle/name,record)
        if len(new["branches"])!=1:raise ValueError("Melanoma story must be one pairwise branch")
        branch=new["branches"][0];relative=Path("branches")/branch["name"]/"fit_input.npz"
        npz_equal(args.bundle/relative,oldroot/relative)
        if branch["cache"] is not None:raise ValueError("Unexpected triplet cache")
        origins={k:v["read_from_provider_rebuild"] for k,v in new["source_files"].items() if "read_from_provider_rebuild" in v}
        routing=new["native_input_override"];native=args.rebuilt/"data"
        if len(origins)!=12 or set(origins)!=set(routing["file_mapping"]):raise ValueError("Wrong native routing")
        for name,path in origins.items():
            actual=within(native,routing["file_mapping"][name])
            if Path(path).resolve(strict=True)!=actual:raise ValueError("Not read from rebuilt input")
            matches(actual,new["source_files"][name])
        same(native/"manifest.json",routing["manifest_sha256"]);same(native/"COMPLETE.json",routing["completion_sha256"])
        report["bridge"]={"root":str(args.bundle.resolve()),"manifest_sha256":sha(args.bundle/"MANIFEST.json"),
            "fit_arrays_equal":10,"payload_files":len(new["outputs"]),"native_origins":origins,"fit_performed":False}

        report["prior_providers_unchanged"]=prior_providers(study)
        report["historical_checks"]=verify_sources(MAXFUSE/"frozen_20261010",args.source_root)
        report["baselines"]=[]
        for path,base in ((study/"inventory_20261010/evidence/source_census.json",args.source_root.resolve().parent),
                          (study/"inventory_20261010/evidence/curated_sources.json",args.source_root.resolve().parent),
                          (study/"benchmarks/bone_marrow/validation/code_snapshot_output_safety_20261010.json",study)):
            count=0
            for row in json.loads(path.read_text())["files"]:
                if row.get("sha256"):matches(within(base,row["path"]),row);count+=1
            report["baselines"].append({"path":str(path),"sha256":sha(path),"checked":count})
        recent=json.loads((MAXFUSE/"providers/hnscc_story/validation/provider_handoff_20261010/report.json").read_text())
        allowed={"README.md","methods/maxfuse/README.md","methods/maxfuse/STUDY_WORKFLOW.md","methods/maxfuse/study.py","methods/maxfuse/study_inputs.py"}
        for name,digest in recent["current_code_sha256"].items():
            if name not in allowed:same(study/name,digest)
        for name,digest in previous["module_sha256"].items():
            if name not in ("README.md","STUDY_WORKFLOW.md","study.py","study_inputs.py","tests/test_study.py"):same(MAXFUSE/name,digest)
        fixture=json.loads((MAXFUSE/"validation/tmp_path_fix_20261010/report.json").read_text())
        for name,digest in fixture["current_files"].items():same(MAXFUSE/name,digest)
        files=[p for p in HERE.rglob("*") if p.is_file() and not ({"runs","validation","__pycache__"}&set(p.relative_to(HERE).parts))]
        files += [MAXFUSE/name for name in ("study.py","study_inputs.py","README.md","STUDY_WORKFLOW.md","tests/test_melanoma_story_provider_bridge.py","tests/test_melanoma_provider_bridge.py")]
        files += [study/"README.md"]
        report["current_code_sha256"]={str(p.relative_to(study)):sha(p) for p in sorted(files)}
        report["tests_passed"]=sum(int(n) for c in report["commands"] for n in re.findall(r"Ran (\d+) tests? in",c["stderr"]))
        report["status"]="passed";create_json(output/"report.json",report)
    except BaseException as error:
        create_json(output/"FAILED.json",{**report,"error":repr(error)});raise
    print(output/"report.json",flush=True)


if __name__=="__main__":main()
