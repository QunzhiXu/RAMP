"""List study sources and external requirements; never launches analysis."""
import argparse
import json
from pathlib import Path

MODULES = {
 'cohorts': ('benchmarks/integration_inputs','Frozen provider-derived measurements, identity/order contracts and registry'),
 'bone_marrow': ('benchmarks/bone_marrow','Provider data and accepted BM input contract'),
 'ramp_alignment': ('benchmarks/ramp_alignment','Complete input bundles; melanoma-story thread condition; historical lung override'),
 'cycombine': ('methods/cycombine','Complete frozen catalog/bundle, R library and upstream archive'),
 'harmony': ('methods/harmony','Complete frozen catalog/bundle, R library and version locks'),
 'fastmnn': ('methods/fastmnn','Complete frozen catalog/bundle, R library and version locks'),
 'cytonorm': ('methods/cytonorm','Complete frozen catalog/bundle, R library, upstream archives and version locks'),
 'maxfuse': ('methods/maxfuse','Local source included; redistribution terms and complete runtime/input resources still required'),
 'integration': ('benchmarks/integration_evaluation','Method outputs, evaluation metadata, frozen coordinates and evidence contracts'),
 'umap': ('benchmarks/umap_reproduction','Method outputs, labels for evaluation only, full replay bundle and recorded environment'),
 'marker_values': ('benchmarks/value_evaluation','Accepted method outputs, raw shared inputs, metadata and historical manifests'),
 'trainability': ('benchmarks/trainability','Native pools, frozen representations, task contracts and permitted labels'),
 'prediction_replay': ('benchmarks/trainability/prediction_replay','Full prediction catalog, saved predictions, ledgers and raw truth'),
 'head_retraining': ('benchmarks/trainability/head_retraining_full','Full group locks, representations and supervision; merge needs run artifacts'),
 'cd19': ('applications/cd19','Locked molecular/spatial inputs and label provenance'),
 'melanoma': ('applications/melanoma','Accepted RAMP outputs, spatial/provider tables and authorized clinical table; original shared helper included locally'),
 'lung': ('applications/lung','Accepted RAMP outputs, full-cell molecular inputs and authorized clinical/immune tables'),
 'hnscc': ('applications/hnscc','Accepted RAMP outputs, clinical/spatial inputs and comparison/reference locks'),
}


def describe(name):
    directory,resources=MODULES[name]
    return {'module':name,'directory':directory,'directory_present':(Path(__file__).resolve().parent/directory).is_dir(),
            'external_requirements':resources,'full_historical_execution_ready':False,
            'note':'Source index only; no data/environment/permission verification performed'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--module',choices=sorted(MODULES))
    a=p.parse_args();print(json.dumps([describe(n) for n in ([a.module] if a.module else MODULES)],indent=2))
