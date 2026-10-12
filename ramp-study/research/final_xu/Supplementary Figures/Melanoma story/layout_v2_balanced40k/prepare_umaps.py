"""Use the story's fixed 40k cells for every method; full alignments stay frozen."""
from pathlib import Path
import os,sys,json,hashlib,time,importlib.metadata
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[key]='1'
for key,folder in [('MPLCONFIGDIR','mpl'),('NUMBA_CACHE_DIR','numba'),('XDG_CACHE_HOME','xdg')]:
    p=HERE/'.cache'/folder;p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import umap

BASE=ROOT/'figure_melanoma/Comparison'
CACHE=ROOT/'figure_melanoma/cache'
METHODS={'Raw':BASE/'input/raw.npy','Harmony-direct':BASE/'methods/harmony/values.npy',
 'cyCombine':BASE/'methods/cycombine/values.npy','CytoNorm':BASE/'methods/cytonorm/values.npy',
 'fastMNN':BASE/'methods/fastmnn/values.npy','MaxFuse':BASE/'maxfuse_full_panel/methods/maxfuse/values.npy',
 'RAMP':BASE/'input/ramp.npy'}
PROTOCOL=dict(n_neighbors=30,min_dist=.20,n_components=2,metric='euclidean',init='spectral',
 random_state=20260820,transform_seed=20260820,n_jobs=1,low_memory=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def save(p,d):p.write_text(json.dumps(d,indent=2)+'\n')

def main():
    out=HERE/'coordinates';out.mkdir(exist_ok=True)
    meta=pd.read_csv(BASE/'input/evaluation_metadata.csv.gz',low_memory=False)
    dm=pd.read_csv(CACHE/'ramp230_clean15_display_metadata.csv.gz',low_memory=False)
    proof=json.loads((CACHE/'ramp230_clean15_display_manifest.json').read_text())
    row=dm.alignment_row.to_numpy(int)
    assert len(dm)==40000 and len(np.unique(row))==40000
    assert np.array_equal(dm.cell_uid,meta.iloc[row].cell_id)
    assert np.array_equal(dm.dataset,meta.iloc[row].dataset)
    assert np.array_equal(dm.harmonized_coarse_label,meta.iloc[row].celltype)
    assert dm.groupby('dataset').size().eq(20000).all()
    assert sha(METHODS['RAMP'])==proof['alignment_values_sha256']
    assert proof['seed']==PROTOCOL['random_state'] and proof['min_dist']==PROTOCOL['min_dist']
    display=meta.iloc[row].copy().reset_index(drop=True)
    display['alignment_row']=row
    display.to_csv(HERE/'display_metadata.csv.gz',index=False)
    np.save(HERE/'display_rows.npy',row)
    cell_hash=hashlib.sha256('\n'.join(display.cell_id.astype(str)).encode()).hexdigest()
    sources={str(p):sha(p) for p in [BASE/'input/evaluation_metadata.csv.gz',
        CACHE/'ramp230_clean15_display_metadata.csv.gz',CACHE/'ramp230_clean15_display_manifest.json',
        CACHE/'ramp230_clean15_display_umap.npy',*METHODS.values()]}
    software={x:importlib.metadata.version(x) for x in ['umap-learn','scikit-learn','numpy','numba']}
    completed=[]
    for name,p in METHODS.items():
        started=time.time();filename=out/f'{name}.npy';record=out/f'{name}.json'
        if filename.exists() and record.exists():
            old=json.loads(record.read_text())
            assert old['input_sha256']==sources[str(p)] and old['cells_sha256']==cell_hash
            assert old['protocol']==PROTOCOL and old['coordinate_file_sha256']==sha(filename)
            completed.append(name);continue
        save(HERE/'STATUS.json',dict(current=name,completed=completed,total=7,started=time.time()))
        print('START',name,flush=True)
        values=np.load(p,mmap_mode='r')
        assert values.shape[0]==len(meta)
        if name=='RAMP':
            xy=np.load(CACHE/'ramp230_clean15_display_umap.npy')
            mode='Reused exact story coordinates after verifying input matrix, cell IDs/order and UMAP recipe'
        else:
            X=StandardScaler().fit_transform(values[row]).astype(np.float32)
            assert np.isfinite(X).all()
            xy=umap.UMAP(**PROTOCOL).fit_transform(X).astype(np.float32)
            mode='New display-only UMAP fit on identical 40000 story cells'
        assert xy.shape==(40000,2) and np.isfinite(xy).all()
        np.save(filename,xy)
        save(record,dict(method=name,input=str(p),input_sha256=sources[str(p)],
            full_alignment_cells=len(meta),umap_cells=40000,n_dimensions=values.shape[1],
            cells_sha256=cell_hash,protocol=PROTOCOL,
            preprocessing='Pooled dimension-wise StandardScaler fitted on the fixed 40000 display cells; float32 input to UMAP; no additional PCA',
            coordinate_file_sha256=sha(filename),software=software,mode=mode,
            labels_used_for_fit=False,alignment_refit=False,metric_scores_changed=False,
            seconds=time.time()-started))
        completed.append(name);print('DONE',name,round(time.time()-started,1),'seconds',flush=True)
    assert all(sha(p)==h for p,h in sources.items())
    save(HERE/'DISPLAY_CONTRACT.json',dict(status='complete',methods=list(METHODS),
        full_alignment_cells=165686,display_cells=40000,display_cells_per_platform=20000,
        sampling='Exactly the fixed story IDs; donor-balanced within each platform; no reselection',
        cells_sha256=cell_hash,protocol=PROTOCOL,software=software,
        scoring_coordinates='Original full-pool evaluation UMAPs; scores not recomputed from these display coordinates',
        full_alignment_and_metrics_unchanged=True,input_sha256=sources))
    save(HERE/'STATUS.json',dict(status='complete',completed=completed,total=7))

if __name__=='__main__':main()
