"""Use the exact current Lung story cells and UMAP recipe for all methods."""
from pathlib import Path
import os, sys, json, hashlib, time, importlib.metadata, ast
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[key] = '1'
for key, folder in [('MPLCONFIGDIR','mpl'),('NUMBA_CACHE_DIR','numba'),('XDG_CACHE_HOME','xdg')]:
    p = HERE/'.cache'/folder
    p.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(p)
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import umap

STORY = ROOT/'story_lung_HLA_ABC'
RELEASE = STORY/'releases/senosain140k_20260921'
BASE = STORY/'Comparison_senosain140k_20260921'
PANEL = STORY/'Figure/Panel B'
METHODS = {
    'Raw':BASE/'input/raw.npy',
    'Harmony-direct':BASE/'methods/harmony/values.npy',
    'cyCombine':BASE/'methods/cycombine/values.npy',
    'CytoNorm':BASE/'methods/cytonorm/values.npy',
    'fastMNN':BASE/'methods/fastmnn/values.npy',
    'MaxFuse':BASE/'maxfuse_full_panel/methods/maxfuse_hub0/values.npy',
    'RAMP':BASE/'input/ramp.npy',
}

def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def save(p, data):
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n')

def main():
    out = HERE/'coordinates'
    out.mkdir(exist_ok=True)
    config_path = RELEASE/'alignment/UMAP_COMPLETE.json'
    protocol = json.loads(config_path.read_text())['parameters']
    assert protocol['min_dist'] == .12 and protocol['random_state'] == 42
    proof = json.loads((BASE/'input/manifest.json').read_text())
    assert all(sha(BASE/'input'/p) == h for p,h in proof['output_sha256'].items())
    meta = pd.read_csv(BASE/'input/fit_metadata.csv.gz', low_memory=False)
    dm = pd.read_csv(PANEL/'data/PLOT_DATA.csv.gz', low_memory=False)
    release_md = pd.read_csv(RELEASE/'inputs/CELL_METADATA.csv.gz', low_memory=False)
    assert len(dm) == len(meta) == 84000 and meta.cell_id.is_unique
    assert np.array_equal(dm.cell_key, meta.cell_id)
    assert np.array_equal(dm.dataset, meta.dataset)
    assert np.array_equal(release_md.cell_key, meta.cell_id)
    assert np.array_equal(release_md.unit_key, meta.unit_key)
    assert meta.groupby('dataset').size().eq(28000).all()
    assert meta.groupby('dataset').unit_key.nunique().eq(28).all()
    assert meta.groupby('unit_key').size().eq(1000).all()
    assert sha(BASE/'input/raw.npy') == sha(RELEASE/'inputs/RAW_SHARED9.npy')
    assert sha(METHODS['RAMP']) == sha(RELEASE/'alignment/RAMP_SHARED9.npy')
    story_xy = np.load(RELEASE/'alignment/RAMP_UMAP.npy')
    assert np.array_equal(story_xy, np.load(PANEL/'data/RAMP_UMAP.npy'))
    assert np.allclose(story_xy,dm[['RAMP_UMAP1','RAMP_UMAP2']].to_numpy(),rtol=1e-7,atol=1e-7)
    display = dm.rename(columns={'cell_key':'cell_id','display_cell_type':'celltype'}).copy()
    display['unit_key'] = meta.unit_key
    display['alignment_row'] = np.arange(len(meta))
    display.to_csv(HERE/'display_metadata.csv.gz',index=False)
    np.save(HERE/'display_rows.npy',np.arange(len(meta)))
    cell_hash = hashlib.sha256('\n'.join(meta.cell_id).encode()).hexdigest()
    # Read literal style constants without importing a script that stages files.
    script = PANEL/'scripts/draw_panel_B.py'
    constants = {}
    for node in ast.parse(script.read_text()).body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            name=node.targets[0].id
            if name in ['PLATFORMS','PLATFORM_COLORS','TYPE_COLORS']:
                constants[name]=ast.literal_eval(node.value)
    assert set(display.celltype)==set(constants['TYPE_COLORS'])
    save(HERE/'STORY_STYLE.json',dict(
        dataset_palette={ds:constants['PLATFORM_COLORS'][label] for ds,label in constants['PLATFORMS'].items()},
        dataset_display_labels=constants['PLATFORMS'],celltype_palette=constants['TYPE_COLORS'],
        draw_order_bottom_to_top=['Tumor / epithelial','Other identities','Fibroblast / stroma','Myeloid / APC','CD4 T','Endothelial'],
        middle_layer_labels=['CD8 T','Others'],
        display_mapping_reused_from=str(PANEL/'data/PLOT_DATA.csv.gz'),
        style_source=str(script),style_source_sha256=sha(script)))
    paths=[BASE/'input/manifest.json',BASE/'input/fit_metadata.csv.gz',
        RELEASE/'inputs/CELL_METADATA.csv.gz',RELEASE/'inputs/RAW_SHARED9.npy',
        RELEASE/'alignment/RAMP_SHARED9.npy',RELEASE/'alignment/RAMP_UMAP.npy',
        RELEASE/'scripts/umaps.py',config_path,PANEL/'data/PLOT_DATA.csv.gz',
        PANEL/'data/RAMP_UMAP.npy',script,BASE/'evaluation/metric_battery_summary.csv',*METHODS.values()]
    sources={str(p):sha(p) for p in paths}
    software={x:importlib.metadata.version(x) for x in ['umap-learn','scikit-learn','numpy','numba']}
    completed=[]
    for name,p in METHODS.items():
        start=time.time();filename=out/f'{name}.npy';record=out/f'{name}.json'
        if filename.exists() and record.exists():
            old=json.loads(record.read_text())
            assert old['input_sha256']==sha(p) and old['cells_sha256']==cell_hash
            assert old['protocol']==protocol and old['coordinate_file_sha256']==sha(filename)
            completed.append(name);continue
        save(HERE/'STATUS.json',dict(current=name,completed=completed,total=7,started=time.time()))
        print('START',name,flush=True)
        values=np.load(p)
        assert len(values)==84000 and np.isfinite(values).all()
        if name=='RAMP':
            xy=story_xy.copy()
            mode='Exact current story coordinates reused; input values and full cell IDs/order verified'
        else:
            # Match story umaps.py: float64 source, pooled StandardScaler, no explicit pre-cast/PCA.
            x=StandardScaler().fit_transform(values)
            xy=umap.UMAP(**protocol).fit_transform(x)
            mode='New display-only UMAP on all 84000 current story reference cells'
        assert xy.shape==(84000,2) and np.isfinite(xy).all()
        np.save(filename,xy)
        save(record,dict(method=name,input=str(p),input_sha256=sha(p),
            full_alignment_cells=84000,umap_cells=84000,n_dimensions=values.shape[1],
            cells_sha256=cell_hash,protocol=protocol,
            preprocessing='Pooled dimension-wise StandardScaler on all 84000 cells, exactly as story umaps.py; no additional PCA',
            coordinate_file_sha256=sha(filename),software=software,mode=mode,
            labels_used_for_fit=False,alignment_refit=False,metric_scores_changed=False,
            seconds=time.time()-start))
        completed.append(name)
        print('DONE',name,round(time.time()-start,1),'seconds',flush=True)
    assert all(sha(p)==h for p,h in sources.items())
    save(HERE/'DISPLAY_CONTRACT.json',dict(status='complete',methods=list(METHODS),
        full_alignment_cells=84000,display_cells=84000,display_cells_per_platform=28000,
        sampling='Exact complete current story reference pool; all 84 patients and 1000 cells per patient; no new selection',
        cells_sha256=cell_hash,protocol=protocol,software=software,
        scoring_coordinates='Original min_dist 0.15 evaluation UMAPs retained; existing scores are not recomputed from these display coordinates',
        ramp_coordinates_exactly_equal_to_current_story=True,
        full_alignment_and_metrics_unchanged=True,input_sha256=sources))
    save(HERE/'STATUS.json',dict(status='complete',completed=completed,total=7))

if __name__=='__main__':main()
