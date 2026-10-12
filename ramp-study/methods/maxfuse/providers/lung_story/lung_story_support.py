"""Pinned Lung story source identities and create-only publication."""
import json
from pathlib import Path
import shutil
import sys

HERE=Path(__file__).resolve().parent
MAXFUSE=HERE.parents[1]
sys.path.insert(0,str(MAXFUSE))
from migrate import create_json, sha
from run import reserve
from verify import within

SOURCE_HASHES = {
    "source/prepare.py": "269afed484ec944470f151f024341b1e8b2af844d84d8dfe7c803bc414790ac9",
    "source/export_cords.R": "f39947226127dabe1c3dccdd81958a9bc689ba4263adc70e4b0e74acb504279c",
    "source/historical_manifest.json": "2b31cbc78407d605d92d0382e7d95dbf0518bbe4f6401c7c0d5e3231d33f443e",
    "source/comparison_config.json": "0e2969c9153cac16d9e92c81e6e6df2d6a05ad533fb4cf6039facad1f5726a1c",
    "source/comparison_input_manifest.json": "e49bcd9ea0a318b83a219d60daa6ae8643bbaa6a979672a6c407a9fd7f457604",
    "source/full_run_config.json": "b6298d88b7b6537af8e4992e30f572bcf577eede0e45739d99a817d7e338bd6d",
    "source/historical_export_complete.json": "822aad4af21648f4135c4fbd1259e605b13b6a1d14038592af4d7adbf1f098e4",
    "source/historical_export_command.json": "279be85be40979f150294aa266d6b3eb343e4f56d982e60257f43e3d2d2e3f1d"
}


def history():
    return json.loads((HERE/"source/historical_manifest.json").read_text())


def check_code():
    for name,digest in SOURCE_HASHES.items():
        if sha(HERE/name)!=digest:raise ValueError("Frozen Lung story source changed: "+name)


def input_contract(comparison,metadata,senosain,cords,zheng):
    old=history()
    sources=old["source_sha256"]
    def original(suffix):
        matches=[v for k,v in sources.items() if k.endswith(suffix)]
        if len(matches)!=1:raise ValueError("Ambiguous source hash")
        return matches[0]
    records={
        "comparison_config":(comparison/"run_config.json",SOURCE_HASHES["source/comparison_config.json"]),
        "fit_metadata":(comparison/"input/fit_metadata.csv.gz",original("/input/fit_metadata.csv.gz")),
        "shared_reference":(comparison/"input/raw.npy",original("/input/raw.npy")),
        "comparison_manifest":(comparison/"input/manifest.json",old["original_input_manifest_sha256"]),
        "cohort_metadata":(metadata,original("/inputs/CELL_METADATA.csv.gz")),
        "senosain_values":(senosain/"senosain_140k_raw35.npy",original("/annotation/senosain_140k_raw35.npy")),
        "senosain_markers":(senosain/"senosain_140k_markers35.csv",original("/annotation/senosain_140k_markers35.csv")),
        "senosain_metadata":(senosain/"senosain_140k_cell_metadata.csv.gz",original("/annotation/senosain_140k_cell_metadata.csv.gz")),
        "cords_rds":(cords,original("/SingleCellExperiment Objects/sce_all_annotated.rds")),
        "zheng_h5ad":(zheng,original("/Zheng2025/mIF_data.h5ad")),
    }
    return records


def check_inputs(comparison,metadata,senosain,cords,zheng):
    check_code()
    records=input_contract(comparison,metadata,senosain,cords,zheng)
    for role,(path,digest) in records.items():
        if sha(path)!=digest:raise ValueError("Pinned input changed: "+role)
    return {role:{"path":str(path.resolve()),"sha256":digest,"bytes":path.stat().st_size}
            for role,(path,digest) in records.items()}


def claim(output,comparison,metadata,senosain,cords,zheng,inputs,rscript):
    root=reserve(output,(comparison,metadata,senosain,cords,zheng,rscript,HERE/"source"))
    create_json(root/"REQUEST.json",{"case":"lung_story","inputs":inputs,
        "code_sha256":{name:sha(HERE/name) for name in ("prepare.py","lung_story_support.py")},
        "source_sha256":SOURCE_HASHES,"rscript":str(rscript),"rscript_sha256":sha(rscript),"cell_resampling":False,"new_maxfuse_fit_performed":False})
    return root


def check_export(folder,source):
    old=json.loads((HERE/"source/historical_export_complete.json").read_text())
    actual=json.loads((folder/"EXPORT_COMPLETE.json").read_text())
    if {k:v for k,v in actual.items() if k!="source"}!={k:v for k,v in old.items() if k!="source"}:
        raise ValueError("Cords export scientific record differs")
    if actual["source"]!=str(source.resolve()):raise ValueError("Wrong Cords source path")
    for leaf in ("requested_cells.csv","raw_feature_names.txt","counts.float64.bin","c_counts_asinh.float64.bin","source_identity.csv"):
        if sha(folder/leaf)!=history()["output_sha256"]["cords_imc/"+leaf]:raise ValueError("Cords export differs: "+leaf)
    return actual


def verify_reference(data):
    old=history()
    actual=json.loads(within(data,"manifest.json").read_text())
    if set(actual.get("output_sha256",{}))!=set(old["output_sha256"]):
        raise ValueError("Lung story output inventory changed")
    for name,digest in old["output_sha256"].items():
        observed=sha(within(data,name))
        if name!="cords_imc/EXPORT_COMPLETE.json" and observed!=digest:raise ValueError("Lung story output differs: "+name)
        if observed!=actual["output_sha256"][name]:raise ValueError("Output not attested: "+name)
    export=json.loads((data/"cords_imc/EXPORT_COMPLETE.json").read_text())
    old_export=json.loads((HERE/"source/historical_export_complete.json").read_text())
    if {k:v for k,v in export.items() if k!="source"}!={k:v for k,v in old_export.items() if k!="source"}:
        raise ValueError("Cords export scientific record differs")
    ignored={"source_sha256","output_sha256"}
    if {k:v for k,v in actual.items() if k not in ignored}!={k:v for k,v in old.items() if k not in ignored}:
        raise ValueError("Lung story scientific manifest differs")
    return actual


def publish_new(stage,output):
    stage,output=stage.resolve(strict=True),output.resolve(strict=True)
    if stage!=output/"_work":raise ValueError("Only this run's staging may be published")
    request=json.loads((output/"REQUEST.json").read_text())
    if request.get("case")!="lung_story":raise ValueError("Wrong request")
    check_code()
    for name,digest in request["code_sha256"].items():
        if sha(HERE/name)!=digest:raise ValueError("Builder changed during run")
    for record in request["inputs"].values():
        if sha(record["path"])!=record["sha256"]:raise ValueError("Input changed during run")
    if sha(request["rscript"])!=request["rscript_sha256"]:raise ValueError("R executable changed during run")
    manifest=verify_reference(stage/"data")
    check_export(stage/"data/cords_imc",Path(request["inputs"]["cords_rds"]["path"]))
    expected_sources={r["path"]:r["sha256"] for k,r in request["inputs"].items() if k!="comparison_manifest"}
    if manifest["source_sha256"]!=expected_sources:raise ValueError("Unexpected actual read provenance")
    config=json.loads((stage/"run_config.json").read_text())
    old_config=json.loads((HERE/"source/full_run_config.json").read_text())
    if {k:v for k,v in config.items() if k not in ("comparison","full_input_manifest_sha256")}!={k:v for k,v in old_config.items() if k not in ("comparison","full_input_manifest_sha256")}:
        raise ValueError("Scientific configuration changed")
    if config["comparison"]!=str(output) or config["full_input_manifest_sha256"]!=sha(stage/"data/manifest.json"):
        raise ValueError("Configuration provenance mismatch")
    create_json(output/"PUBLISH_STARTED.json",{"data_manifest_sha256":sha(stage/"data/manifest.json")})
    # Claim the whole child directory before any payload: a competing publisher
    # fails here, so it cannot interleave writes in the same stage.
    (output/"data").mkdir(exist_ok=False)
    names=[*("data/"+n for n in sorted(manifest["output_sha256"])),"data/manifest.json","run_config.json"]
    for name in names:
        src,dst=within(stage,name),output/name
        dst.parent.mkdir(parents=True,exist_ok=True)
        with src.open("rb") as reader,dst.open("xb") as writer:shutil.copyfileobj(reader,writer)
    verify_reference(output/"data")
    complete={"status":"complete","case":"lung_story","stage":"native","cells":84000,
              "manifest_sha256":sha(output/"data/manifest.json"),"new_maxfuse_fit_performed":False}
    create_json(output/"data/COMPLETE.json",complete)
    create_json(output/"COMPLETE.json",{**complete,"config_sha256":sha(output/"run_config.json")})


def fail(output,error):
    create_json(output/"FAILED.json",{"error":repr(error),"partial_output_retained":True})
