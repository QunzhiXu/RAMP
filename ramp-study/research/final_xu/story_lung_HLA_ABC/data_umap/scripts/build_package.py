"""Import the established reference without resampling; optional provenance operation."""
import package_runtime as rt
from package_runtime import BASE
from pathlib import Path
import argparse
import shutil
import sys
import platform
import importlib.metadata as im


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, default=BASE.parents[1])
    args = parser.parse_args()
    root = args.source_root.resolve()
    old = root/'overnight0909'
    alignment = old/'results/zheng_senosain_cords_alignment'
    if (BASE/'MANIFEST.json').exists():
        raise SystemExit('Package already imported; use run.sh verify or reproduce.')
    plan = []
    for name in ['RAW_SHARED9.npy','SELECTED_DONORS.csv','MARKER_MAPPING.csv','DESIGN.json']:
        plan.append((alignment/name, BASE/'data'/name))
    plan.append((alignment/'DISPLAY_METADATA.csv.gz', BASE/'data/CELL_METADATA.csv.gz'))
    for method in ['baseline','RAMP']:
        for suffix in ['SHARED9.npy','STANDARDIZED.npy','UMAP.npy','UMAP.json']:
            name=f'{method}_{suffix}'
            plan.append((alignment/name, BASE/'reference'/name))
    for name in ['ALIGNMENT.json','UMAP_COORDINATES.csv.gz','DONOR_COLORS.csv',
                 'ALGORITHM_AUDIT.csv','PROFILE_AUDIT.csv','COUPLING_AUDIT.csv','STRENGTH_AUDIT.csv',
                 'EVIDENCE_AUDIT.csv','REFERENCE_WEIGHT_AUDIT.csv']:
        plan.append((alignment/name, BASE/'reference'/name))
    for name in ['PREPARATION.json','VERIFICATION.json','DONOR_ELIGIBILITY.csv','PATIENT_CELL_AUDIT.csv',
                 'CELL_METADATA.csv.gz','DISPLAY_METADATA_NOTE.json','DISPLAY_LABEL_MAPPING.csv','DISPLAY_CELL_TYPE_COUNTS.csv']:
        plan.append((alignment/name, BASE/'provenance'/name))
    for name in ['REFERENCE_PROFILES.npz','REFERENCE_REPRODUCTION.json']:
        plan.append((old/'results/vcaf_projected_dfs'/name, BASE/'reference'/name))
    for name in ['alignment_overview.png','aligned_dataset_and_fibroblast_facets.png','aligned_donor_facets.png']:
        plan.append((alignment/'figures'/name, BASE/'reference/figures'/name))
    for name in ['__init__.py','ramp.py','ramp_ver2.py']:
        plan.append((old/'vendor/ramp'/name, BASE/'vendor/ramp'/name))
    for name in ['prepare_zheng_alignment.py','fit_zheng_alignment.py','inspect_zheng_alignment.py']:
        # Text archives document upstream preparation without exposing unsafe legacy entry points.
        plan.append((old/'scripts'/name, BASE/'provenance/source_workflow_scripts'/(name+'.txt')))
    records=[]
    for source,dest in plan:
        digest=rt.sha(source)
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,dest)
        assert rt.sha(source)==digest and rt.sha(dest)==digest
        records.append({'file':str(dest.relative_to(BASE)),'source':str(source),
                        'sha256':digest,'bytes':dest.stat().st_size})
    versions={name:im.version(name) for name in ['numpy','pandas','scipy','scikit-learn','umap-learn',
                                              'numba','llvmlite','pynndescent','matplotlib','Pillow']}
    (BASE/'requirements.txt').write_text('\n'.join(f'{k}=={v}' for k,v in versions.items())+'\n')
    rt.save_json(BASE/'ENVIRONMENT.json',{'python':sys.version,'platform':platform.platform(),'packages':versions,
        'python_executable_used':sys.executable,'threads':1,'recorded_utc':rt.now()})
    rt.save_json(BASE/'MANIFEST.json',{'created_utc':rt.now(),'scope':'Frozen balanced reference and UMAP; not full Cords projection or HLA prediction',
        'rows':84000,'markers':9,'unit_count':84,'datasets':3,'copied_files':records,
        'metadata':'data/CELL_METADATA.csv.gz is the corrected DISPLAY_METADATA, preserving all row IDs/order and provider labels. Only dataset/unit_key are passed to RAMP.',
        'original_sources_unchanged':True,'preparation':'Import existing fixed row selection; do not resample.',
        'reference_profiles':'Captured from the identical 84k RAMP replay; included for the next projection stage, with no extra donors.'})
    print(f'Imported {len(records)} files; all source/destination SHA256 values match.',flush=True)


if __name__=='__main__':main()
