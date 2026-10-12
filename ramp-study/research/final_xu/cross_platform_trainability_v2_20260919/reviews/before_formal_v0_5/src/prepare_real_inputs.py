"""Register seven frozen cohorts without touching their source folders.

Do not infer patient identities from cell/ROI name strings. Unresolved upstream
normalization remains explicitly ineligible for strict missing-marker analysis.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json
import numpy as np
import pandas as pd
from trainability.core import ROOT, require, file_sha256, write_json, output_path
from trainability.inputs import FittingPool, assert_target_invariance
from trainability.real_data import load_task

FX=ROOT.parent
STORY={'lung_story':'story_lung_HLA_ABC','melanoma_story':'figure_melanoma','hnscc_story':'figure_HNSCC'}


def prepare(out):
    out=output_path(out)
    require(not out.exists(),'Registration already exists; choose another version')
    config=json.loads((ROOT/'config/protocol.draft.json').read_text())
    cyto=json.loads((FX/'algorithm/CytoNorm/config.json').read_text())
    summary=[]
    for spec in config['tasks']:
        name=spec['name']; refs=spec['historical_references']; sources={}
        def track(p):
            p=Path(p); sources[str(p)]=file_sha256(p); return p
        meta=pd.read_csv(track(FX/refs['metadata']['path']),dtype=str,keep_default_na=False)
        markers=track(FX/refs['markers']['path']).read_text().splitlines()
        require(meta.cell_id.is_unique,'Repeated cell IDs')
        if name=='bone_marrow':
            frames=[pd.read_csv(track(FX/f'benchmark_data/bone_marrow/{d}/alignment_input_14.csv.gz'),dtype={'cell_id':str})
                    for d in ['bm_codex','bm_cytof','triana_abseq']]
            table=pd.concat(frames).set_index('cell_id')
            require(table.index.is_unique and set(meta.cell_id)==set(table.index),'Frozen BM identity mismatch')
            native=table.loc[meta.cell_id,markers].to_numpy(float)
        elif spec['layout']=='benchmark':
            p=track(FX/f'benchmark_data/{name}/input/raw.npy')
            require(file_sha256(p)==refs['native_values']['expected_sha256'],'Frozen native hash mismatch')
            native=np.load(p,allow_pickle=False)
        else:
            p=track(FX/refs['native_values']['path'])
            require(file_sha256(p)==refs['native_values']['expected_sha256'],'Frozen story native hash mismatch')
            native=np.load(p,allow_pickle=False)
        historical=np.load(FX/refs['native_values']['path'],allow_pickle=False)
        require(np.array_equal(native,historical),'Native row/marker values differ from frozen reference')
        require(native.shape==(len(meta),len(markers)) and np.isfinite(native).all(),'Invalid native matrix')
        if spec['layout']=='benchmark':
            profile=cyto['cases'][name]
            recipes=dict(zip(profile['datasets'],profile['transforms']))
            track(FX/f'benchmark_data/{name}/build_benchmark.py')
            track(FX/f'benchmark_data/{name}/manifest.json')
            track(FX/'algorithm/CytoNorm/config.json')
            if name=='melanoma':
                track(FX/'benchmark_results/melanoma/cycombine/preprocessing/preprocessing_manifest.json')
        else:
            manifest=json.loads(track(FX/f'{STORY[name]}/Comparison/input/manifest.json').read_text())
            recipes={p['dataset']:p for p in manifest['preprocessing']}
            track(FX/f'{STORY[name]}/Comparison/scripts/prepare.py')
        # Retrieve identities by an exact cell-ID join, never by row-position guess.
        if name=='lung_story':
            p=track(FX/'story_lung_HLA_ABC/data_umap/data/CELL_METADATA.csv.gz')
            identity=pd.read_csv(p,dtype=str,keep_default_na=False).rename(columns={'cell_key':'cell_id'})
            patient_col='patient_id'
        elif name=='melanoma_story':
            p=track(FX/'figure_melanoma/analysis/alignment/output/cell_metadata.csv.gz')
            identity=pd.read_csv(p,dtype=str,keep_default_na=False).rename(columns={'cell_uid':'cell_id'})
            patient_col='donor'
            track(FX/'figure_melanoma/inputs/cohort_package/DATA_SELECTION.md')
            track(FX/'figure_melanoma/inputs/cohort_package/build_story_sample.py')
        elif name=='hnscc_story':
            p=track(FX/'figure_HNSCC/alignment/input/cell_metadata.csv.gz')
            identity=pd.read_csv(p,dtype=str,keep_default_na=False).rename(columns={'cell_uid':'cell_id'})
            patient_col='donor'
            track(FX/'figure_HNSCC/alignment/01_prepare_alignment_input.py')
            track(FX/'raw_data/hnscc/haist_2026_codex/Haist_2026_Cancer_Cell_HNSCC_LN_niches.pdf')
        else:
            identity=meta.copy();patient_col='patient_id' if name=='melanoma' else 'donor'
        require(identity.cell_id.is_unique,'Nonunique identity lookup')
        aligned=identity.set_index('cell_id').loc[meta.cell_id]
        require(aligned[patient_col].ne('').all(),'Missing patient/unit identity')
        require(np.array_equal(aligned.dataset,meta.dataset),'Identity dataset disagreement')
        meta['patient_id']=meta.dataset+'::'+aligned[patient_col].to_numpy(str)
        meta['identity_level']='patient'
        # The Monkman local package calls its IDs core/patient; no clinical crosswalk.
        unknown=meta.dataset.eq('Monkman CODEX')
        meta.loc[unknown,'identity_level']='unit_descriptive'
        meta.loc[unknown,'patient_id']=meta.loc[unknown,'unit_key']
        require(meta.groupby('unit_key').patient_id.nunique().max()==1,'Unit maps to multiple patients')
        canonical={m:''.join(c for c in m.upper() if c.isalnum()) for m in markers}
        require(len(set(canonical.values()))==len(markers),'Duplicate marker aliases in frozen panel')
        contracts={}
        for ds in dict.fromkeys(meta.dataset):
            recipe=recipes[ds]
            # Unknown *upstream* normalization cannot be certified by perturbing
            # an already-normalized table; keep these two roles visibly blocked.
            unresolved=ds in {'MEL_Ptacek2021_MIBI'}
            contracts[ds]=dict(operator=recipe['operator'],cofactor=recipe.get('cofactor'),
                j2_eligible=not unresolved,
                source_dependency_status='upstream_normalization_unresolved' if unresolved else 'column_local_frozen_source',
                reason=recipe.get('reason','Frozen source layer; recipe traced in original input manifest'),
                caveat='Provider processed values, not instrument raw' if recipe['operator']=='identity' else 'Simulated missing marker on frozen selected cells',
                patient_status='unit_descriptive' if ds=='Monkman CODEX' else 'mapped_from_source_metadata')
            if ds=='Haist_CODEX':
                contracts[ds]['reason']='Haist 2026 STAR Methods, CODEX imaging data analysis: marker expression z-normalized per imaging run after lateral spillover compensation; no per-cell cross-marker denominator reported. Frozen selection and provider labels remain outside missing-assay claims.'
        folder=out/name;folder.mkdir(parents=True)
        np.save(folder/'native.npy',native)
        meta.to_csv(folder/'metadata.csv.gz',index=False,compression={'method':'gzip','mtime':0})
        pool=FittingPool(f'{name}:frozen_U_v0.3',tuple(meta.cell_id),tuple(meta.dataset),tuple(meta.unit_key))
        classes=sorted(set.intersection(*(set(g.celltype) for _,g in meta.groupby('dataset')))-set(config['exclude_labels']))
        write_json(folder/'manifest.json',dict(name=name,markers=markers,canonical_markers=canonical,
            pool_id=pool.pool_id,pool_sha256=pool.sha256,contracts=contracts,classes=classes,
            sources=sources,outputs={f:file_sha256(folder/f) for f in ['native.npy','metadata.csv.gz']},
            sources_read_only=True,eligibility_precedes_model_scores=True))
        task=load_task(name,out)
        # Numerical mask isolation covers every shared marker when strict inputs exist.
        checks=[]
        if all(c['j2_eligible'] for c in contracts.values()):
            for i,m in enumerate(markers):
                indices=np.concatenate([g.index.to_numpy()[:100] for _,g in meta.groupby('dataset',sort=False)])
                small=task.subset(indices,f'mask_{m}')
                original=small.native.copy()
                first=small.input('unaligned',canonical[m]).x
                small.native[:,i]=np.nan
                require(np.array_equal(first,small.input('unaligned',canonical[m]).x),'Target changes remaining features')
                checks.append(m)
                small.native=original
        write_json(folder/'mask_checks.json',dict(markers_checked=checks,scope='post-frozen-input transform; not upstream certification'))
        summary.append(dict(task=name,cells=len(meta),markers=len(markers),roles=len(contracts),
            classes=classes,j2_eligible=all(c['j2_eligible'] for c in contracts.values()),
            unresolved_roles=[d for d,c in contracts.items() if not c['j2_eligible']],
            unit_descriptive_roles=meta.loc[meta.identity_level.eq('unit_descriptive'),'dataset'].unique().tolist()))
        print(name,len(meta),'registered',flush=True)
    write_json(out/'SUMMARY.json',dict(tasks=summary,formal_training_started=False))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(ROOT/'data/real_inputs_v0_3'))
    prepare(Path(p.parse_args().out))
