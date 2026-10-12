"""Freeze exact story inputs; writes only to this Comparison directory."""
import os,sys,json,hashlib,importlib.util,shutil
from pathlib import Path
sys.dont_write_bytecode=True
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parents[1]
STORY=HERE.parent
ROOT=STORY.parent
SOURCES={}

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def track(p):
    p=Path(p).resolve();SOURCES[str(p)]=sha(p);return p
def csv(p,**kwargs):return pd.read_csv(track(p),low_memory=False,**kwargs)
def load(p):return np.load(track(p))
def writej(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def exportcsv(d,p):d.to_csv(p,index=False,float_format='%.17g',compression={'method':'gzip','mtime':0})

def prepare():
    target=HERE/'input'
    if (target/'manifest.json').exists():
        m=json.loads((target/'manifest.json').read_text())
        assert all(sha(p)==v for p,v in m['source_sha256'].items())
        assert all(sha(target/p)==v for p,v in m['output_sha256'].items())
        print('Verified existing inputs',HERE,flush=True);return
    target.mkdir(exist_ok=True)
    for x in ['methods','evaluation','logs','.cache','tmp']: (HERE/x).mkdir(exist_ok=True)
    if STORY.name=='story_lung_HLA_ABC':
        case='lung_story';src=STORY/'data_umap'
        raw=load(src/'data/RAW_SHARED9.npy')
        md=csv(src/'data/CELL_METADATA.csv.gz')
        design=json.loads(track(src/'data/DESIGN.json').read_text())
        markers=design['markers'];datasets=design['datasets']
        ramp_path=track(src/'reference/RAMP_SHARED9.npy')
        md=md.rename(columns={'cell_key':'cell_id'})
        md['celltype']=md['broad_type'].fillna('Other / uncertain')
        transforms={datasets[0]:('asinh',5),datasets[1]:('asinh',1),datasets[2]:('asinh',1)}
        expected=(84000,9)
    elif STORY.name=='figure_HNSCC':
        case='hnscc_story';src=STORY/'alignment'
        raw=load(src/'input/raw_shared_markers.float64.npy')
        md=csv(src/'input/cell_metadata.csv.gz').rename(columns={'cell_uid':'cell_id'})
        markers=track(src/'input/markers.txt').read_text().splitlines()
        manifest=json.loads(track(src/'output/alignment_manifest.json').read_text())
        datasets=manifest['datasets'];ramp_path=track(src/'output/aligned_ramp_ver2.float32.npy')
        md['celltype']=md.display_label.fillna('Other')
        transforms={datasets[0]:('log1p',None),datasets[1]:('identity',None)}
        expected=(94488,22)
    else:
        assert STORY.name=='figure_melanoma';case='melanoma_story'
        src=STORY/'analysis/alignment'
        mp=track(src/'output/manifest.json');manifest=json.loads(mp.read_text())
        script=track(src/'run_alignment.py')
        spec=importlib.util.spec_from_file_location('frozen_melanoma_input',script)
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        for p,k in [(mod.MOLD_INPUT,'moldoveanu_shared13_input'),(mod.ENG_INPUT,'engelbrecht_shared13_input'),
                    (mod.MOLD_RAW,'moldoveanu_raw'),(mod.ENG_ADT,'engelbrecht_adt'),(mod.FROZEN_STORY_METADATA,'frozen_story_metadata')]:
            track(p);assert sha(p)==manifest['sources'][k+'_sha256'],(p,'original run hash differs')
        print('Reassembling original melanoma clean15 input',flush=True)
        mold,_=mod.build_moldoveanu(pd.read_csv(mod.MOLD_INPUT,low_memory=False))
        eng,_=mod.build_engelbrecht(pd.read_csv(mod.ENG_INPUT,low_memory=False))
        joined=pd.concat([mold,eng],ignore_index=True)
        markers=mod.CLEAN15;datasets=mod.DATASET_ORDER;raw=joined[markers].to_numpy(float)
        md=csv(src/'output/cell_metadata.csv.gz').rename(columns={'cell_uid':'cell_id'})
        assert np.array_equal(md.cell_id.astype(str),joined.cell_uid.astype(str))
        assert np.array_equal(md.dataset.astype(str),joined.dataset.astype(str))
        assert np.array_equal(md.donor.astype(str),joined.donor.astype(str))
        # Extend the existing display-label mapping to the complete frozen cohort.
        mapping=csv(STORY/'inputs/harmonized_display_labels.csv.gz')
        rows=[]
        for ds,subdir,key,provider in [(datasets[0],'moldoveanu_imc','cell_uid','Cluster.v2'),
                                      (datasets[1],'engelbrecht_citeseq','source_cell_id','Broad_cell.type.1')]:
            labels=csv(STORY/'inputs/cohort_package'/subdir/'locked_metadata.csv.gz',usecols=[key,provider])
            rules=mapping.loc[mapping.dataset.eq(ds),['provider_broad_label','harmonized_coarse_label']].drop_duplicates()
            assert rules.provider_broad_label.is_unique
            lookup=dict(zip(rules.provider_broad_label,rules.harmonized_coarse_label))
            labels['celltype']=labels[provider].map(lookup).fillna('Non-joint / ambiguous')
            rows.append(labels.rename(columns={key:'cell_id'})[['cell_id','celltype']])
        labels=pd.concat(rows).set_index('cell_id');assert labels.index.is_unique
        md['celltype']=labels.reindex(md.cell_id).celltype.to_numpy();assert md.celltype.notna().all()
        ramp_path=track(src/'output/ramp_aligned.npy')
        transforms={datasets[0]:('asinh',1),datasets[1]:('log1p',None)}
        expected=(165686,15)
    assert raw.shape==expected and np.isfinite(raw).all()
    assert len(md)==len(raw) and md.cell_id.is_unique
    ramp=np.load(ramp_path);assert ramp.shape==raw.shape and np.isfinite(ramp).all()
    md['cell_id']=md.cell_id.astype(str);md['dataset']=md.dataset.astype(str);md['unit_key']=md.unit_key.astype(str)
    fitmd=md[['cell_id','dataset','unit_key']].copy()
    labels=md[['cell_id','dataset','unit_key','celltype']].copy()
    bad=labels.celltype.str.contains('^Other$|uncertain|ambiguous|Non-joint|unclassified|Unknown',case=False,regex=True)
    labels['joint_evaluation']=~bad
    assert list(pd.unique(fitmd.dataset))==datasets
    np.save(target/'raw.npy',raw);shutil.copy2(ramp_path,target/'ramp.npy')
    (target/'markers.txt').write_text('\n'.join(markers)+'\n')
    exportcsv(fitmd,target/'fit_metadata.csv.gz');exportcsv(labels,target/'evaluation_metadata.csv.gz')
    rawtable=pd.concat([fitmd,pd.DataFrame(raw,columns=markers)],axis=1)
    exportcsv(rawtable,target/'raw.csv.gz')
    ready=raw.copy();protocol=[]
    for ds,(op,cofactor) in transforms.items():
        rows=fitmd.dataset.eq(ds).to_numpy();x=raw[rows]
        if op=='log1p':assert (x>=0).all();y=np.log1p(x)
        elif op=='asinh':y=np.arcsinh(x/cofactor)
        else:y=x.copy()
        ready[rows]=y
        protocol.append({'dataset':ds,'cells':int(rows.sum()),'units':int(fitmd.loc[rows,'unit_key'].nunique()),
            'operator':op,'cofactor':cofactor,'raw_min':float(x.min()),'raw_max':float(x.max()),
            'prepared_min':float(y.min()),'prepared_max':float(y.max())})
    np.save(target/'analysis_ready.npy',ready)
    exportcsv(pd.concat([fitmd,pd.DataFrame(ready,columns=markers)],axis=1),target/'analysis_ready.csv.gz')
    np.save(target/'cell_ids.npy',fitmd.cell_id.to_numpy(dtype=str))
    # MaxFuse: identical shared and active panels; no additional story targets.
    mf=ready.copy()
    for ds in datasets:
        rows=fitmd.dataset.eq(ds).to_numpy();x=ready[rows]
        scale=x.std(axis=0,ddof=1);assert (scale>0).all()
        mf[rows]=(x-x.mean(axis=0))/scale
        assert np.all(np.std(mf[rows],axis=1)>1e-12),'MaxFuse zero-variance cell; do not silently exclude'
    np.save(target/'maxfuse_input.npy',mf)
    records={'case':case,'cells':len(raw),'markers':markers,'datasets':datasets,'preprocessing':protocol,
        'fit_metadata_columns':list(fitmd.columns),'labels_passed_to_fit':False,
        'same_cells_and_markers_as_frozen_RAMP':True,'ramp_reused_without_refit':True,
        'method_scope':'Harmony-direct uses source scale; fastMNN/cyCombine/CytoNorm use explicitly fixed shared-only assay adaptations; MaxFuse uses those values followed by within-dataset sample-SD marker z-score. No extra active markers.',
        'preprocessing_note':'Raw ADT uses log1p shared counts (not full-panel CLR); imaging intensity uses asinh cofactor 1; Haist signed provider analysis values are passed through; Senosain uses asinh cofactor 5. Frozen before fitting, not tuned on labels or performance.',
        'source_sha256':SOURCES,'output_sha256':{p.name:sha(p) for p in target.iterdir() if p.is_file()}}
    assert all(sha(p)==h for p,h in SOURCES.items())
    writej(target/'manifest.json',records)
    writej(HERE/'run_config.json',{'case':case,'root':str(ROOT),'comparison':str(HERE),'datasets':datasets,
        'markers':markers,'input_manifest_sha256':sha(target/'manifest.json'),'seed':42,'maxfuse_seed':20260830,
        'scope':'same shared marker set and exact original full story cell cohort for every method'})
    print('INPUT COMPLETE',case,raw.shape,flush=True)

if __name__=='__main__':prepare()
