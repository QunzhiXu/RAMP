"""Create-only Lung story migration evidence; tests/hash checks, no extraction or fit."""
import argparse
from datetime import datetime, timezone
import json
import hashlib
import os
from pathlib import Path
import re
import subprocess

import lung_story_support as support
from lung_story_support import HERE, MAXFUSE, create_json, reserve, sha
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
        "melanoma_story":"b99aa72b0a22b8592b7dd601084507f329c8f9905d225d99b13d8bbfde91cefb",
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
        elif case in ("hnscc_story","melanoma_story"):
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
        parser.error("Use a new Lung story validation directory")
    output=reserve(args.output,(args.source_root,args.rebuilt,args.bundle,HERE/"source"))
    study=MAXFUSE.parents[1]
    report={"utc":datetime.now(timezone.utc).isoformat(),"scope":"Lung story provider/native-input migration and three-hub fit-input assembly only","commands":[]}
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
        for case in ("lung_story","melanoma_story","hnscc_story","bone_marrow","tnbc","melanoma","lung_cancer"):
            command([args.provider_python.absolute(),"-B","-m","unittest","discover","-s",MAXFUSE/"providers"/case/"tests","-v"])
        command([args.maxfuse_python.absolute(),"-B","-m","unittest","discover","-s",MAXFUSE/"tests","-v"])
        command([args.provider_python.absolute(),"-B",study/"benchmarks/integration_inputs/inputs.py","verify-staged","--input-root",study/"benchmarks/integration_inputs/runs/frozen_inputs_20261010"])
        command([args.provider_python.absolute(),"-B","-c","import json,platform,numpy,pandas,h5py,scipy; print(json.dumps({'python':platform.python_version(),'numpy':numpy.__version__,'pandas':pandas.__version__,'h5py':h5py.__version__,'scipy':scipy.__version__}))"])
        support.check_code()
        reference=args.source_root/"story_lung_HLA_ABC/Comparison_senosain140k_20260921/maxfuse_full_panel"
        old_comparison=reference.parents[1]/"Comparison/maxfuse_full_panel"
        copies={"source/prepare.py":reference/"scripts/prepare.py",
                "source/export_cords.R":old_comparison/"scripts/export_cords.R",
                "source/historical_manifest.json":reference/"data/manifest.json",
                "source/comparison_config.json":reference.parent/"run_config.json",
                "source/comparison_input_manifest.json":reference.parent/"input/manifest.json",
                "source/full_run_config.json":reference/"run_config.json",
                "source/historical_export_complete.json":reference/"data/cords_imc/EXPORT_COMPLETE.json",
                "source/historical_export_command.json":old_comparison/"CORDS_EXPORT_COMMAND.json"}
        for name,path in copies.items():same(path,sha(HERE/name))
        request=json.loads((args.rebuilt/"REQUEST.json").read_text())
        for name,digest in request["code_sha256"].items():same(HERE/name,digest)
        if request["source_sha256"]!=support.SOURCE_HASHES:raise ValueError("Source snapshot changed")
        for record in request["inputs"].values():matches(Path(record["path"]),record)
        manifest=support.verify_reference(args.rebuilt/"data")
        complete=json.loads((args.rebuilt/"data/COMPLETE.json").read_text())
        if (complete.get("status"),complete.get("case"),complete.get("stage"),complete.get("cells"))!=("complete","lung_story","native",84000):
            raise ValueError("Incomplete native stage")
        same(args.rebuilt/"data/manifest.json",complete["manifest_sha256"])
        done=json.loads((args.rebuilt/"COMPLETE.json").read_text())
        if {k:v for k,v in done.items() if k!="config_sha256"}!=complete:raise ValueError("Root/data completion differs")
        same(args.rebuilt/"run_config.json",done["config_sha256"])
        expected_sources={r["path"]:r["sha256"] for k,r in request["inputs"].items() if k!="comparison_manifest"}
        if manifest["source_sha256"]!=expected_sources:raise ValueError("Unexpected read provenance")
        old_manifest=support.history()
        byte_identical={n:h for n,h in manifest["output_sha256"].items() if n!="cords_imc/EXPORT_COMPLETE.json"}
        for name,digest in byte_identical.items():same(reference/"data"/name,digest)
        export=support.check_export(args.rebuilt/"data/cords_imc",Path(request["inputs"]["cords_rds"]["path"]))
        same(Path(request["rscript"]),request["rscript_sha256"])
        report["provider_run"]={"root":str(args.rebuilt.resolve()),"byte_identical_outputs":byte_identical,
            "reconstructed_output_count":len(manifest["output_sha256"]),"inputs":request["inputs"],
            "manifest_sha256":sha(args.rebuilt/"data/manifest.json"),
            "request_sha256":sha(args.rebuilt/"REQUEST.json"),"completion_sha256":sha(args.rebuilt/"COMPLETE.json"),
            "data_completion_sha256":sha(args.rebuilt/"data/COMPLETE.json"),"config_sha256":sha(args.rebuilt/"run_config.json"),
            "views_equal":manifest["views"]==old_manifest["views"],
            "export_provenance_only":{"file":"cords_imc/EXPORT_COMPLETE.json","sha256":sha(args.rebuilt/"data/cords_imc/EXPORT_COMPLETE.json"),
                "old_sha256":old_manifest["output_sha256"]["cords_imc/EXPORT_COMPLETE.json"],"allowed_difference_keys":["source"]},
            "manifest_difference_keys":[k for k in manifest if manifest[k]!=old_manifest[k]]}
        r_command=json.loads((args.rebuilt/"_work/CORDS_EXPORT_COMMAND.json").read_text())
        if r_command["script_sha256"]!=support.SOURCE_HASHES["source/export_cords.R"]:
            raise ValueError("R script provenance changed")
        expected_command=[request["rscript"],"--vanilla",str(HERE/"source/export_cords.R"),
                          request["inputs"]["cords_rds"]["path"],str(args.rebuilt/"_work/data/cords_imc")]
        if r_command["command"]!=expected_command:raise ValueError("R command changed")
        report["r_export"]={"command":r_command,"record":export,"log_sha256":{n:sha(args.rebuilt/"_work"/n) for n in ("export_cords.stdout.log","export_cords.stderr.log","CORDS_EXPORT_COMMAND.json")}}
        command([Path(request["rscript"]).absolute(),"--vanilla","-e",
                 "p<-c('SingleCellExperiment','data.table','jsonlite','digest','matrixStats','GenomeInfoDb'); cat(jsonlite::toJSON(list(R=R.version.string,packages=setNames(lapply(p,function(x) list(version=as.character(packageVersion(x)),built=packageDescription(x)$Built)),p)),auto_unbox=TRUE))"])

        previous=json.loads((MAXFUSE/"validation/study_handoff_20261010_r2/report.json").read_text())
        report["unchanged_bundles"]=[]
        for row in previous["real_bundles"]:
            root=MAXFUSE/"runs/study_inputs_20261010_r2"/row["case"]
            same(root/"MANIFEST.json",row["bundle_manifest_sha256"])
            m=verify_bundle(root);report["unchanged_bundles"].append({"case":row["case"],"files":len(m["outputs"]),"manifest_sha256":sha(root/"MANIFEST.json")})
        oldroot=MAXFUSE/"runs/study_inputs_20261010_r2/lung_story"
        old,new=verify_bundle(oldroot),verify_bundle(args.bundle)
        if set(new["outputs"])!=set(old["outputs"]):raise ValueError("Bundle inventory changed")
        for name,record in old["outputs"].items():matches(args.bundle/name,record)
        if len(new["branches"])!=3:raise ValueError("Lung story must have three hub branches")
        for branch in new["branches"]:
            relative=Path("branches")/branch["name"]/"fit_input.npz"
            npz_equal(args.bundle/relative,oldroot/relative)
            if branch["cache"] is not None or branch["mode"]!="three-view":raise ValueError("Unexpected cache/mode")
        origins={k:v["read_from_provider_rebuild"] for k,v in new["source_files"].items() if "read_from_provider_rebuild" in v}
        routing=new["native_input_override"];native=args.rebuilt/"data"
        if len(origins)!=18 or set(origins)!=set(routing["file_mapping"]):raise ValueError("Wrong native routing")
        for name,path in origins.items():
            actual=within(native,routing["file_mapping"][name])
            if Path(path).resolve(strict=True)!=actual:raise ValueError("Not read from rebuilt input")
            matches(actual,new["source_files"][name])
        same(native/"manifest.json",routing["manifest_sha256"]);same(native/"COMPLETE.json",routing["completion_sha256"])
        report["bridge"]={"root":str(args.bundle.resolve()),"manifest_sha256":sha(args.bundle/"MANIFEST.json"),
            "fit_arrays_equal":45,"payload_files":len(new["outputs"]),"native_origins":origins,"fit_performed":False}

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
        recent=json.loads((MAXFUSE/"providers/melanoma_story/validation/provider_handoff_20261010/report.json").read_text())
        allowed={"README.md","methods/maxfuse/README.md","methods/maxfuse/STUDY_WORKFLOW.md","methods/maxfuse/study.py","methods/maxfuse/study_inputs.py"}
        allowed.update(["methods/maxfuse/tests/test_provider_bridge.py","methods/maxfuse/tests/test_melanoma_provider_bridge.py","methods/maxfuse/tests/test_lung_provider_bridge.py","methods/maxfuse/tests/test_hnscc_provider_bridge.py","methods/maxfuse/tests/test_melanoma_story_provider_bridge.py"])
        for name,digest in recent["current_code_sha256"].items():
            if name not in allowed:same(study/name,digest)
        for name,digest in previous["module_sha256"].items():
            if name not in ("README.md","STUDY_WORKFLOW.md","study.py","study_inputs.py","tests/test_study.py"):same(MAXFUSE/name,digest)
        fixture=json.loads((MAXFUSE/"validation/tmp_path_fix_20261010/report.json").read_text())
        for name,digest in fixture["current_files"].items():same(MAXFUSE/name,digest)
        old_tests={
        "tests/test_provider_bridge.py": "f60b1e51f020620027c226f26e4a5486c134c6c6582ca1774693d6be2c0731e3",
        "tests/test_melanoma_provider_bridge.py": "59d786e7b8350ffc66aa69ce06527d1752709adf56fcceef9d6e37e45bce09e1",
        "tests/test_lung_provider_bridge.py": "ce89556cd555e2bd02fc4c5c3f9b4e77eb58b52c12d209d5886fe966ddba7cd3",
        "tests/test_hnscc_provider_bridge.py": "db4f4d30c0a2b00345d24ab0f3a45ddb1d94360cf16841bf0cd59431d19272a3",
        "tests/test_melanoma_story_provider_bridge.py": "15e67860a37cfe76f1dcae6b7b4d917362cf56883cef6276d7bb2ca01d886447"
}
        report["upper_negative_tests_only_case_literal_changed"]={}
        for name,digest in old_tests.items():
            text=(MAXFUSE/name).read_text()
            if text.count('prepare("unregistered_case",')!=1:raise ValueError("Unexpected negative test edits")
            previous_text=text.replace('prepare("unregistered_case",','prepare("lung_story",')
            if hashlib.sha256(previous_text.encode()).hexdigest()!=digest:raise ValueError("Other changes to old test: "+name)
            report["upper_negative_tests_only_case_literal_changed"][name]={"previous_sha256":digest,"current_sha256":sha(MAXFUSE/name)}
        failed=HERE/"runs/rebuild_20261010"
        if not (failed/"FAILED.json").is_file() or (failed/"COMPLETE.json").exists():raise ValueError("First failed run missing/misclassified")
        report["retained_failed_run"]={"root":str(failed),"reason":"R_TempDir contained space, before SCE read; new r2 run uses disposable space-free R scratch",
            "files":{str(p.relative_to(failed)):sha(p) for p in sorted(failed.rglob("*")) if p.is_file()}}
        files=[p for p in HERE.rglob("*") if p.is_file() and not ({"runs","validation","__pycache__"}&set(p.relative_to(HERE).parts))]
        files += [MAXFUSE/name for name in ("study.py","study_inputs.py","README.md","STUDY_WORKFLOW.md","tests/test_lung_story_provider_bridge.py",*old_tests)]
        files += [study/"README.md"]
        report["current_code_sha256"]={str(p.relative_to(study)):sha(p) for p in sorted(files)}
        report["tests_passed"]=sum(int(n) for c in report["commands"] for n in re.findall(r"Ran (\d+) tests? in",c["stderr"]))
        report["status"]="passed";create_json(output/"report.json",report)
    except BaseException as error:
        create_json(output/"FAILED.json",{**report,"error":repr(error)});raise
    print(output/"report.json",flush=True)


if __name__=="__main__":main()
