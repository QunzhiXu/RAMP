"""One-page benchmark supplement: frozen UMAPs and saved marker outputs."""
from pathlib import Path
import os
import sys
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
os.environ['MPLCONFIGDIR'] = str(HERE / '.cache/matplotlib')
os.environ['MPLBACKEND'] = 'Agg'
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key] = '1'
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator
from scipy.ndimage import gaussian_filter1d
sys.path.insert(0,str(HERE.parents[1]/'ridge_full_range_update_20260922'))
import ridge_common
RIDGE_CONFIG=json.loads((HERE/'RIDGE_CONFIG.json').read_text())

BASE = ROOT / 'figure_melanoma/Comparison'
EVAL = BASE / 'maxfuse_full_panel/evaluation'
COORD = EVAL / 'umap'
METHODS = ['Raw','Harmony-direct','cyCombine','CytoNorm','fastMNN','MaxFuse','RAMP']
RIDGE_METHODS = [m for m in METHODS if m != 'MaxFuse']
STEMS = dict(zip(METHODS, ['Unaligned','Harmony-direct','cyCombine',
                          'CytoNorm','fastMNN','MaxFuse','RAMP']))
KEYS = dict(zip(METHODS,['raw','harmony','cycombine','cytonorm','fastmnn','maxfuse','ramp']))
MARKERS = ['CD3','CD8','CD20']
WIDTH, HEIGHT = 210., 297.
INK, FRAME, GREEN = '#263238', '#9AA1A5', '#00866A'
SOURCES, AUDIT, GEOMETRY = {}, [], []
STYLE = {
    'dataset_palette': {'Moldoveanu IMC':'#3F78B5','Engelbrecht CITE-seq':'#E68A2E'},
    'celltype_palette': {'Tumor':'#D94B4B','CD8 T':'#155C8A','Other T / CD8-':'#6E9ACB',
        'B lineage':'#E5B13C','NK':'#13A89E','Myeloid':'#D9781E','Endothelial':'#4D9851',
        'Stroma':'#9A6DB0','Non-joint / ambiguous':'#9A9A9A'},
    'draw_order_bottom_to_top':['Non-joint / ambiguous','Tumor','Stroma','Endothelial',
        'NK','Myeloid','B lineage','Other T / CD8-','CD8 T'],
}

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def track(path):
    path = Path(path)
    SOURCES[str(path)] = sha(path)
    return path


def array_sha(values):
    contiguous = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode())
    digest.update(str(contiguous.dtype).encode())
    digest.update(memoryview(contiguous).cast('B'))
    return digest.hexdigest()


def load():
    style = STYLE
    meta = pd.read_csv(track(BASE/'input/evaluation_metadata.csv.gz'),dtype={'cell_id':str},low_memory=False)
    markers = track(BASE/'input/markers.txt').read_text().splitlines()
    input_proof = json.loads(track(BASE/'input/manifest.json').read_text())
    assert len(meta)==165686 and meta.unit_key.nunique()==38 and meta.cell_id.is_unique
    assert meta.groupby('dataset').size().to_dict()=={'Moldoveanu IMC':112497,'Engelbrecht CITE-seq':53189}
    assert set(meta.celltype)==set(style['celltype_palette'])
    assert markers==input_proof['markers'] and len(markers)==15 and set(MARKERS)<=set(markers)
    ids=np.load(track(BASE/'input/cell_ids.npy'),allow_pickle=True)
    assert np.array_equal(ids,meta.cell_id)
    fitting=pd.read_csv(track(BASE/'input/fit_metadata.csv.gz'),dtype={'cell_id':str})
    assert fitting[['cell_id','dataset','unit_key']].equals(meta[['cell_id','dataset','unit_key']])
    contract=json.loads(track(HERE/'DISPLAY_CONTRACT.json').read_text())
    assert contract['status']=='complete' and contract['display_cells']==40000
    display_meta=pd.read_csv(track(HERE/'display_metadata.csv.gz'),low_memory=False)
    rows=np.load(track(HERE/'display_rows.npy'))
    assert np.array_equal(display_meta.cell_id,meta.iloc[rows].cell_id)
    assert display_meta.groupby('dataset').size().eq(20000).all()
    assert display_meta.unit_key.nunique()==38
    coords,protocols={},{}
    for method in METHODS:
        coords[method]=np.load(track(HERE/'coordinates'/f'{method}.npy'))
        proof=json.loads(track(HERE/'coordinates'/f'{method}.json').read_text())
        assert coords[method].shape==(40000,2) and np.isfinite(coords[method]).all()
        assert sha(HERE/'coordinates'/f'{method}.npy')==proof['coordinate_file_sha256']
        assert proof['protocol']==contract['protocol'] and not proof['labels_used_for_fit']
        assert proof['cells_sha256']==contract['cells_sha256']
        assert sha(track(proof['input']))==proof['input_sha256']
        protocols[method]=proof
    assert np.array_equal(coords['RAMP'],np.load(track(ROOT/'figure_melanoma/cache/ramp230_clean15_display_umap.npy')))
    # Ridges retain all cells; the balanced 40k sample is solely for UMAP display.
    values={'Raw':np.load(track(BASE/'input/analysis_ready.npy')),
            'RAMP':np.load(track(BASE/'input/ramp.npy'))}
    for k in ['analysis_ready.npy','ramp.npy','raw.npy','evaluation_metadata.csv.gz','markers.txt']:
        assert sha(BASE/'input'/k)==input_proof['output_sha256'][k]
    for method in ['Harmony-direct','cyCombine','CytoNorm','fastMNN']:
        p=HERE.parent/'layout_v1/fastmnn_reconstructed.csv.gz' if method=='fastMNN' else BASE/'methods'/KEYS[method]/'corrected_expression.csv.gz'
        frame=pd.read_csv(track(p),dtype={'cell_id':str})
        assert len(frame)==len(meta) and frame.cell_id.is_unique and set(frame.cell_id)==set(meta.cell_id)
        values[method]=frame.set_index('cell_id').loc[meta.cell_id,markers].to_numpy()
        if method!='fastMNN':
            assert np.allclose(values[method],np.load(BASE/'methods'/KEYS[method]/'values.npy'),rtol=1e-10,atol=1e-10)
    for x in values.values():assert x.shape==(len(meta),15) and np.isfinite(x).all()
    track(BASE/'methods/fastmnn/fit.rds')
    track(HERE.parent/'layout_v1/extract_fastmnn.R')
    track(ROOT/'Main Figure/fastmnn_reconstructed_audit_20260914/official_convertPCsToSCE.R')
    return meta,markers,coords,values,style,protocols,display_meta


def axes(fig,x,y,w,h):
    """Millimetres; y is distance down from top."""
    return fig.add_axes([x/WIDTH,1-(y+h)/HEIGHT,w/WIDTH,h/HEIGHT])


def text(fig,x,y,label,size=8.5,weight='bold',color=INK,ha='left',va='top'):
    label=label.replace('Harmony-direct','Harmony')
    return fig.text(x/WIDTH,1-y/HEIGHT,label,fontsize=size,weight=weight,color=color,ha=ha,va=va)


def scatter(ax,meta,xy,by,style):
    if by == 'Dataset':
        colors = np.array([style['dataset_palette'][v] for v in meta.dataset])
        idx = np.random.default_rng(42).permutation(len(meta))
        ax.scatter(xy[idx,0],xy[idx,1],c=colors[idx],s=.25,alpha=.88,
                   edgecolors='none',rasterized=True)
    else:
        labels = meta.celltype.to_numpy()
        for z,label in enumerate(style['draw_order_bottom_to_top']):
            idx = np.random.default_rng(42).permutation(np.flatnonzero(labels==label))
            background = label == 'Non-joint / ambiguous'
            ax.scatter(xy[idx,0],xy[idx,1],c=style['celltype_palette'][label],
                       s=.12 if background else .29,alpha=.40 if background else .92,
                       edgecolors='none',rasterized=True,zorder=z+1)
    low,high = xy.min(axis=0),xy.max(axis=0)
    pad = (high-low)*.025
    ax.set_xlim(low[0]-pad[0],high[0]+pad[0]);ax.set_ylim(low[1]-pad[1],high[1]+pad[1])
    ax.set_aspect('equal',adjustable='box');ax.set_axis_off()


def ridge(ax,v,meta,method,marker,style):
    return ridge_common.draw_ridge(ax,v,meta,method,marker,RIDGE_CONFIG,AUDIT)


def main():
    track(HERE/'RIDGE_CONFIG.json');track(Path(ridge_common.__file__))
    meta,markers,coords,values,style,protocols,display_meta = load()
    plt.rcParams.update({'font.family':'Arial','font.size':8,'svg.fonttype':'none',
                         'pdf.fonttype':42,'ps.fonttype':42,'text.color':INK})
    fig=plt.figure(figsize=(WIDTH/25.4,HEIGHT/25.4),facecolor='white',dpi=200)
    text(fig,9,9,'Melanoma 2 | IMC and CITE-seq',size=13)
    text(fig,9,17.0,'165,686 cells  |  UMAP: 40,000  |  38 donors  |  15 markers',size=8.2)
    dataset_names = {'Moldoveanu IMC':'IMC','Engelbrecht CITE-seq':'CITE-seq'}
    handles = [Line2D([],[],ls='',marker='o',ms=3.5,mew=0,color=style['dataset_palette'][d],label=dataset_names[d]) for d in meta.dataset.unique()]
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(9/WIDTH,1-34/HEIGHT),
               ncol=3,frameon=False,prop={'size':7.4,'weight':'bold'},columnspacing=1.,handletextpad=.35,borderaxespad=0)
    fig.add_artist(Line2D([9/WIDTH,201/WIDTH],[1-25/HEIGHT]*2,transform=fig.transFigure,lw=.5,color='#CDD3D5'))
    text(fig,9,29,'A',size=12)
    text(fig,15,29,'Cross-platform integration',size=9.4)
    text(fig,100,29,'B',size=12)
    text(fig,106,29,'Marker distributions',size=9.4)
    ridge_common.add_legend(fig,RIDGE_CONFIG)
    text(fig,28.5,40.,'Dataset',size=8.5,ha='center')
    text(fig,70.5,40.,'Cell type',size=8.5,ha='center')
    for j,marker in enumerate(MARKERS):text(fig,115.5+j*34.5,40.,marker,size=9,ha='center')
    # Two independent blocks: 7 UMAP rows and 6 ridge rows occupy equal heights.
    for i,method in enumerate(METHODS):
        top=46.5+i*31.8
        label=method
        text(fig,9,top,label,size=8.2,color=GREEN if method=='RAMP' else INK)
        for j,by in enumerate(['Dataset','Cell type']):
            x,y,w,h=9+j*42,top+3.6,39,27
            ax=axes(fig,x,y,w,h)
            scatter(ax,display_meta,coords[method],by,style)
            fig.add_artist(Rectangle((x/WIDTH,1-(y+h)/HEIGHT),w/WIDTH,h/HEIGHT,
                transform=fig.transFigure,fill=False,ec=GREEN if method=='RAMP' else FRAME,lw=.4,zorder=100))
            GEOMETRY.append(dict(block='UMAP',method=method,colour_by=by,x_mm=x,y_mm=y,width_mm=w,height_mm=h))
    for i,method in enumerate(RIDGE_METHODS):
        top=46.5+i*37.24
        label={'Raw':'Raw (assay-transformed)','fastMNN':'fastMNN (reconstructed)'}.get(method,method)
        text(fig,100,top,label,size=8.2,color=GREEN if method=='RAMP' else INK)
        for j,marker in enumerate(MARKERS):
            x,y,w,h=100+j*34.5,top+3.6,31,30.0
            ax=axes(fig,x,y,w,h)
            v=values[method][:,markers.index(marker)]
            ridge(ax,v,meta,method,marker,style)
            if method=='RAMP':
                for spine in ax.spines.values():spine.set_color(GREEN);spine.set_linewidth(.4)
            GEOMETRY.append(dict(block='ridge',method=method,marker=marker,x_mm=x,y_mm=y,width_mm=w,height_mm=h))
    fig.add_artist(Line2D([95/WIDTH]*2,[1-29/HEIGHT,1-270/HEIGHT],transform=fig.transFigure,lw=.5,color='#E2E6E7'))
    labels=['Tumor','Stroma','Myeloid','B lineage','CD8 T','Other T / CD8-','Endothelial','NK','Non-joint / ambiguous']
    handles=[Line2D([],[],ls='',marker='o',ms=3.2,mew=0,color=style['celltype_palette'][s],label=s.replace('−','-')) for s in labels]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,8/HEIGHT),ncol=5,
               frameon=False,prop={'size':7.4,'weight':'bold'},columnspacing=1.,handletextpad=.35,labelspacing=.8)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    artists=list(fig.texts)+list(fig.legends)
    for ax in fig.axes:
        if ax.axison:artists.extend(t for t in ax.get_xticklabels() if t.get_visible() and t.get_text())
    bounds=[]
    for artist in artists:
        box=artist.get_window_extent(renderer)
        assert box.x0>=-1 and box.y0>=-1 and box.x1<=fig.bbox.width+1 and box.y1<=fig.bbox.height+1,(str(artist),box)
        bounds.append(dict(label=artist.get_text() if hasattr(artist,'get_text') else 'legend',
                           bbox_pixels=list(map(float,box.extents))))
    out=HERE/'output';(out/'pdf').mkdir(parents=True,exist_ok=True)
    stem='melanoma_story_supplement'
    fig.savefig(out/f'{stem}.png',dpi=300,facecolor='white')
    fig.savefig(out/'pdf'/f'{stem}.pdf',dpi=300,facecolor='white')
    fig.savefig(out/f'{stem}.svg',dpi=300,facecolor='white')
    plt.close(fig)
    audit=pd.DataFrame(AUDIT)
    ridge_common.validate_audit(audit,RIDGE_CONFIG)
    audit.to_csv(HERE/'ridge_display_audit.csv',index=False)
    meta.groupby(['dataset','unit_key','celltype']).size().rename('cells').reset_index().to_csv(HERE/'cell_counts.csv',index=False)
    display_meta.groupby(['dataset','unit_key','celltype']).size().rename('cells').reset_index().to_csv(HERE/'display_cell_counts.csv',index=False)
    assert all(sha(p)==h for p,h in SOURCES.items())
    report=dict(version='layout_v2_balanced40k',benchmark='melanoma_story',cells=165686,sample_units=38,shared_markers=15,
        dataset_counts=meta.groupby('dataset').size().to_dict(),umap_methods=METHODS,ridge_methods=RIDGE_METHODS,
        ridge_markers=MARKERS,all_cells_retained_for_alignment_and_ridges=True,all_sample_units_retained=True,labels_merged=False,
        display_cells=40000,display_cells_per_platform=20000,display_source="Exact story 40k IDs; same across all methods",
        alignment_refit=False,umap_refit=True,ramp_umap_reused=True,coordinate_transform=None,equal_aspect=True,
        maxfuse_display='Current Figure 2 full-active-panel pairwise MaxFuse; 32 IMC / 130 CITE-seq active proteins, 15 shared. No refit.',
        fastmnn_display='UMAP: fixed 40k cells from archived corrected PCs, using common display protocol; ridge: saved PCs @ rotation.T per batchelor convertPCsToSCE, not original intensity units.',
        raw_ridge_display='Frozen analysis_ready: IMC asinh(x/1), CITE-seq log1p counts. Display only; raw UMAP is from the same 40k cells in source-scale values.',ridge_range_quantiles=[.005,.995],
        ridge_bins=160,ridge_sigma_bins=6,ridge_height='Each curve normalized to its own density maximum',
        dataset_palette=style['dataset_palette'],celltype_palette=style['celltype_palette'],
        draw_order=style['draw_order_bottom_to_top'],umap_protocols=protocols,
        page_mm=[WIDTH,HEIGHT],layout=GEOMETRY,text_bounds_checked=True,
        text_bounds=bounds,input_sha256=SOURCES,inputs_unchanged=True,script_sha256=sha(__file__),
        output_sha256={str(p.relative_to(HERE)):sha(p) for p in out.rglob('*') if p.is_file()})
    report.update(ridge_common.manifest_extra(RIDGE_CONFIG))
    (HERE/'MANIFEST.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(out/f'{stem}.png',flush=True)


if __name__=='__main__':main()
