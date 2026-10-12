"""HNSCC supplement: exact story cells and display protocol; saved fits read-only."""
from pathlib import Path
import os, sys, json, hashlib, time, importlib.metadata, ast
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[k]='1'
for k,f in [('MPLCONFIGDIR','mpl'),('NUMBA_CACHE_DIR','numba'),('XDG_CACHE_HOME','xdg')]:
    p=HERE/'.cache'/f;p.mkdir(parents=True,exist_ok=True);os.environ[k]=str(p)
os.environ['MPLBACKEND']='Agg'
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import umap
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
STORY=ROOT/'figure_HNSCC'
BASE=STORY/'Comparison'
COORD=STORY/'alignment/umap/full_94488_umap_coordinates.csv.gz'
STYLE_SOURCE=STORY/'big_figure/combined_ABCDE_two_rows/build_combined_two_rows.py'
LAYOUT=STORY/'big_figure/polish_20260913/CEF_D_colorbar_aligned_20260914/LAYOUT_AUDIT.json'
METHODS={
 'Raw':BASE/'input/raw.npy','Harmony-direct':BASE/'methods/harmony/values.npy',
 'cyCombine':BASE/'methods/cycombine/values.npy','CytoNorm':BASE/'methods/cytonorm/values.npy',
 'fastMNN':BASE/'methods/fastmnn/values.npy',
 'MaxFuse':BASE/'maxfuse_full_panel/methods/maxfuse/values.npy','RAMP':BASE/'input/ramp.npy'}
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def main():
    out=HERE/'coordinates';out.mkdir(exist_ok=True)
    config=STORY/'alignment/umap/umap_manifest.json'
    protocol=json.loads(config.read_text())['umap']|{'n_jobs':1}
    assert protocol==dict(n_neighbors=30,min_dist=.25,metric='euclidean',random_state=20260824,n_jobs=1)
    proof=json.loads((BASE/'input/manifest.json').read_text())
    assert all(sha(BASE/'input'/p)==h for p,h in proof['output_sha256'].items())
    meta=pd.read_csv(BASE/'input/fit_metadata.csv.gz',low_memory=False,dtype={'unit_key':str})
    dm=pd.read_csv(COORD,low_memory=False,dtype={'unit_key':str})
    original=pd.read_csv(STORY/'alignment/input/cell_metadata.csv.gz',low_memory=False,dtype={'unit_key':str})
    assert len(meta)==len(dm)==94488 and meta.cell_id.is_unique and meta.unit_key.nunique()==77
    for name in ['dataset','unit_key']:assert np.array_equal(dm[name],meta[name])
    assert np.array_equal(dm.cell_uid,meta.cell_id) and np.array_equal(original.cell_uid,meta.cell_id)
    assert np.array_equal(np.load(BASE/'input/ramp.npy'),np.load(STORY/'alignment/output/aligned_ramp_ver2.float32.npy'))
    assert np.array_equal(np.load(BASE/'input/raw.npy'),np.load(STORY/'alignment/input/raw_shared_markers.float64.npy'))
    display=dm.rename(columns={'cell_uid':'cell_id','plot_label':'celltype'}).copy()
    display['alignment_row']=np.arange(len(meta));display.to_csv(HERE/'display_metadata.csv.gz',index=False)
    np.save(HERE/'display_rows.npy',np.arange(len(meta)))
    cell_hash=hashlib.sha256('\n'.join(meta.cell_id).encode()).hexdigest()
    tree=ast.parse(STYLE_SOURCE.read_text());constants={}
    for n in tree.body:
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ['DATASET_COLORS','DATASET_NAMES','DISPLAY_ORDER','TYPE_COLORS']:
            constants[n.targets[0].id]=ast.literal_eval(n.value)
    assert set(display.celltype)==set(constants['TYPE_COLORS'])
    # Execute only the original plotting function, with its save callback replaced;
    # no source imports, cache writes, or original figure outputs.
    captured={}
    def capture(fig,*args):
        fig.canvas.draw();captured['height_mm']=fig.axes[0].get_position().height*fig.get_figheight()*25.4
        captured['width_mm']=fig.axes[0].get_position().width*fig.get_figwidth()*25.4
        captured['area']=float(fig.axes[0].collections[0].get_sizes()[0]);plt.close(fig)
        return 'in-memory'
    env=dict(constants,np=np,pd=pd,plt=plt,Line2D=Line2D,WIDE_WIDTH=7.,ROW_HEIGHT=3.45,INPUTS={'umap':COORD},header=lambda *a,**kw:None,save=capture)
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='build_c')
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(STYLE_SOURCE),'exec'),env);env['build_c']()
    # Verify the latest composite embeds this exact compact Panel C at native size.
    layout=json.loads(LAYOUT.read_text())['placements']['C']
    assert layout['bounds_inches'][2]-layout['bounds_inches'][0]==7.
    assert sha(layout['source'])==layout['sha256']
    save(HERE/'STORY_STYLE.json',dict(dataset_palette=constants['DATASET_COLORS'],
        dataset_display_labels={'GSE212797_ADT':'CITE-seq','Haist_CODEX':'CODEX'},
        celltype_palette=constants['TYPE_COLORS'],draw_order_bottom_to_top=constants['DISPLAY_ORDER'],
        celltype_legend_order=constants['DISPLAY_ORDER'],plot_seed=20260902,
        dataset_alpha=.48,celltype_alpha=.50,story_point_area=captured['area'],
        source_plot_height_mm=captured['height_mm'],source_plot_width_mm=captured['width_mm'],
        margin_fractions=dict(left=.02,right=.02,bottom=.02,top=.25),
        style_source=str(STYLE_SOURCE),style_source_sha256=sha(STYLE_SOURCE),
        current_story_panel=layout,display_mapping_reused_from=str(COORD)))
    paths=[config,COORD,STYLE_SOURCE,LAYOUT,Path(layout['source']),BASE/'input/manifest.json',
        BASE/'input/fit_metadata.csv.gz',STORY/'alignment/03_plot_umap.py',
        STORY/'alignment/input/cell_metadata.csv.gz',STORY/'alignment/output/aligned_ramp_ver2.float32.npy',
        STORY/'alignment/input/raw_shared_markers.float64.npy',
        BASE/'maxfuse_full_panel/evaluation/metric_battery_summary.csv',*METHODS.values()]
    sources={str(p):sha(p) for p in paths}
    software={x:importlib.metadata.version(x) for x in ['umap-learn','scikit-learn','numpy','numba']}
    completed=[]
    # Reuse RAMP first so the figure renderer can be developed while others fit.
    for name in ['RAMP']+[m for m in METHODS if m!='RAMP']:
        p=METHODS[name];start=time.time();filename=out/f'{name}.npy';record=out/f'{name}.json'
        if filename.exists() and record.exists():
            old=json.loads(record.read_text());assert old['input_sha256']==sha(p) and old['cells_sha256']==cell_hash
            assert old['protocol']==protocol and old['coordinate_file_sha256']==sha(filename)
            completed.append(name);continue
        save(HERE/'STATUS.json',dict(current=name,completed=completed,total=7,started=time.time()))
        print('START',name,flush=True);values=np.load(p)
        assert len(values)==94488 and np.isfinite(values).all()
        if name=='RAMP':xy=dm[['UMAP1','UMAP2']].to_numpy();mode='Exact story coordinate table reused'
        else:
            scaled=StandardScaler().fit_transform(np.asarray(values,dtype=np.float64))
            xy=umap.UMAP(**protocol).fit_transform(scaled);mode='Display-only UMAP on complete story pool'
        assert xy.shape==(94488,2) and np.isfinite(xy).all();np.save(filename,xy)
        save(record,dict(method=name,input=str(p),input_sha256=sha(p),full_alignment_cells=94488,
            umap_cells=94488,n_dimensions=values.shape[1],cells_sha256=cell_hash,protocol=protocol,
            preprocessing='Float64 input; pooled StandardScaler over all 94488 cells; no additional PCA',
            coordinate_file_sha256=sha(filename),software=software,mode=mode,labels_used_for_fit=False,
            alignment_refit=False,metric_scores_changed=False,seconds=time.time()-start))
        completed.append(name);print('DONE',name,round(time.time()-start,1),'seconds',flush=True)
    assert all(sha(p)==h for p,h in sources.items())
    save(HERE/'DISPLAY_CONTRACT.json',dict(status='complete',methods=list(METHODS),full_alignment_cells=94488,
        display_cells=94488,sample_units=77,dataset_counts=meta.groupby('dataset').size().to_dict(),
        sampling='Complete frozen story pool: 20488 CITE-seq cells from 3 donors plus 1000 CODEX cells per 74 primary-tumor patients; no new sampling',
        cells_sha256=cell_hash,protocol=protocol,software=software,
        scoring_coordinates='Existing min_dist 0.15 / seed 42 evaluation coordinates and scores unchanged',
        ramp_coordinates_exactly_equal_to_current_story=True,full_alignment_and_metrics_unchanged=True,input_sha256=sources))
    save(HERE/'STATUS.json',dict(status='complete',completed=completed,total=7))
if __name__=='__main__':main()
