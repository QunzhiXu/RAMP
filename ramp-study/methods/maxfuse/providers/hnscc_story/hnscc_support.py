"""Pinned HNSCC source identities and create-only publication."""
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
    "source/prepare.py": "011f54fcf57d485a6446ed3bcce47b89b3fb8be4fa492a009c78dc25599085a4",
    "source/alignment/01_prepare_alignment_input.py": "d5ff1784c8c56122c948fcb294daed6f402494189333b98884cd27778cf492c6",
    "source/historical_manifest.json": "1d3de26d8275769369d293a6d49c577c31ca1153a244d2509ec790336b710ef6",
    "source/comparison_config.json": "08b77deae4ef148f55b9765585f35ccb882cce40eaa32b76469f0fdf5dd3dfe7",
    "source/comparison_input_manifest.json": "6704b0b5b573f4e966cbcb97fa059c886f66a2e322c46458bccfb9a77b0dcd7d",
    "source/full_run_config.json": "dde8fafb4910f6dd49c0164b64c0a4a86407b8ec47583575ae2343281a464393"
}


def history():
    return json.loads((HERE/"source/historical_manifest.json").read_text())


def check_code():
    for name,digest in SOURCE_HASHES.items():
        if sha(HERE/name)!=digest:raise ValueError("Frozen HNSCC source changed: "+name)


def input_contract(comparison,rahim,haist):
    old=history()
    sources=old["source_sha256"]
    def original(suffix):
        matches=[v for k,v in sources.items() if k.endswith(suffix)]
        if len(matches)!=1:raise ValueError("Ambiguous source hash")
        return matches[0]
    records={
        "comparison_config":(comparison/"run_config.json",SOURCE_HASHES["source/comparison_config.json"]),
        "fit_metadata":(comparison/"input/fit_metadata.csv.gz",original("/Comparison/input/fit_metadata.csv.gz")),
        "shared_reference":(comparison/"input/raw.npy",original("/Comparison/input/raw.npy")),
        "comparison_manifest":(comparison/"input/manifest.json",old["original_input_manifest_sha256"]),
        "rahim_adt":(rahim,original("/complete/GSE212797_adata_ADT.h5ad")),
        "haist_cells":(haist,original("/complete/HNSCCdiscovery_CODEX.csv")),
    }
    return records


def check_inputs(comparison,rahim,haist):
    check_code()
    records=input_contract(comparison,rahim,haist)
    for role,(path,digest) in records.items():
        if sha(path)!=digest:raise ValueError("Pinned input changed: "+role)
    return {role:{"path":str(path.resolve()),"sha256":digest,"bytes":path.stat().st_size}
            for role,(path,digest) in records.items()}


def claim(output,comparison,rahim,haist,inputs):
    root=reserve(output,(comparison,rahim,haist,HERE/"source"))
    create_json(root/"REQUEST.json",{"case":"hnscc_story","inputs":inputs,
        "code_sha256":{name:sha(HERE/name) for name in ("prepare.py","hnscc_support.py")},
        "source_sha256":SOURCE_HASHES,"cell_resampling":False,"new_maxfuse_fit_performed":False})
    return root


def verify_reference(data):
    old=history()
    actual=json.loads(within(data,"manifest.json").read_text())
    if actual.get("output_sha256")!=old["output_sha256"]:
        raise ValueError("HNSCC output inventory/hashes changed")
    for name,digest in old["output_sha256"].items():
        if sha(within(data,name))!=digest:raise ValueError("HNSCC output differs: "+name)
    if {k:v for k,v in actual.items() if k!="source_sha256"}!={k:v for k,v in old.items() if k!="source_sha256"}:
        raise ValueError("HNSCC scientific manifest differs")
    return actual


def publish_new(stage,output):
    stage,output=stage.resolve(strict=True),output.resolve(strict=True)
    if stage!=output/"_work":raise ValueError("Only this run's staging may be published")
    request=json.loads((output/"REQUEST.json").read_text())
    if request.get("case")!="hnscc_story":raise ValueError("Wrong request")
    check_code()
    for name,digest in request["code_sha256"].items():
        if sha(HERE/name)!=digest:raise ValueError("Builder changed during run")
    for record in request["inputs"].values():
        if sha(record["path"])!=record["sha256"]:raise ValueError("Input changed during run")
    manifest=verify_reference(stage/"data")
    expected_sources={r["path"]:r["sha256"] for k,r in request["inputs"].items() if k!="comparison_manifest"}
    expected_sources[str(HERE/"source/alignment/01_prepare_alignment_input.py")]=SOURCE_HASHES["source/alignment/01_prepare_alignment_input.py"]
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
    complete={"status":"complete","case":"hnscc_story","stage":"native","cells":94488,
              "manifest_sha256":sha(output/"data/manifest.json"),"new_maxfuse_fit_performed":False}
    create_json(output/"data/COMPLETE.json",complete)
    create_json(output/"COMPLETE.json",{**complete,"config_sha256":sha(output/"run_config.json")})


def fail(output,error):
    create_json(output/"FAILED.json",{"error":repr(error),"partial_output_retained":True})
