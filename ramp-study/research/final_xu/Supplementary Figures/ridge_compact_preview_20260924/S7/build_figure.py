"""One-page benchmark supplement: frozen UMAPs and saved marker outputs."""
from pathlib import Path
import os
import sys
sys.dont_write_bytecode = True
HERE = Path('/Users/qxu6/Documents/New project/final_xu/Supplementary Figures/HNSCC story/ridge_full_range_20260922')
OUT = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
os.environ['MPLCONFIGDIR'] = str(OUT / '.cache/matplotlib')
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
sys.path.insert(0,str(OUT.parent))
import ridge_common
RIDGE_CONFIG=json.loads((HERE/'RIDGE_CONFIG.json').read_text())

STORY = ROOT / 'figure_HNSCC'
BASE = STORY / 'Comparison'
METHODS = ['Raw','Harmony-direct','cyCombine','CytoNorm','fastMNN','MaxFuse','RAMP']
RIDGE_METHODS = [m for m in METHODS if m != 'MaxFuse']
KEYS = dict(zip(METHODS,['raw','harmony','cycombine','cytonorm','fastmnn','maxfuse','ramp']))
MARKERS = ['CD3','CD4','CD8']
WIDTH, HEIGHT = 210., 297.
INK, FRAME, GREEN = '#263238', '#9AA1A5', '#00866A'
SOURCES, AUDIT, GEOMETRY = {}, [], []
SCATTER_AREA = None


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
    style=json.loads(track(HERE/'STORY_STYLE.json').read_text())
    meta=pd.read_csv(track(HERE/'display_metadata.csv.gz'),low_memory=False,dtype={'unit_key':str})
    fitmeta=pd.read_csv(track(BASE/'input/fit_metadata.csv.gz'),low_memory=False,dtype={'unit_key':str})
    assert len(meta)==94488 and meta.unit_key.nunique()==77 and meta.cell_id.is_unique
    assert meta.groupby('dataset').size().to_dict()=={'GSE212797_ADT':20488,'Haist_CODEX':74000}
    assert np.array_equal(meta.cell_id,fitmeta.cell_id)
    assert np.array_equal(meta.unit_key,fitmeta.unit_key)
    assert set(meta.celltype)==set(style['celltype_palette'])
    markers=track(BASE/'input/markers.txt').read_text().splitlines()
    assert len(markers)==22 and set(MARKERS)<=set(markers)
    proof=json.loads(track(BASE/'input/manifest.json').read_text())
    assert markers==proof['markers']
    for name,digest in proof['output_sha256'].items():
        assert sha(track(BASE/'input'/name))==digest
    contract=json.loads(track(HERE/'DISPLAY_CONTRACT.json').read_text())
    assert contract['status']=='complete' and contract['display_cells']==94488
    assert all(sha(track(p))==h for p,h in contract['input_sha256'].items())
    rows=np.load(track(HERE/'display_rows.npy'))
    assert np.array_equal(rows,np.arange(94488))
    coords,protocols={},{}
    for method in METHODS:
        coords[method]=np.load(track(HERE/'coordinates'/f'{method}.npy'))
        pr=json.loads(track(HERE/'coordinates'/f'{method}.json').read_text())
        assert coords[method].shape==(94488,2) and np.isfinite(coords[method]).all()
        assert sha(HERE/'coordinates'/f'{method}.npy')==pr['coordinate_file_sha256']
        assert pr['protocol']==contract['protocol'] and not pr['labels_used_for_fit']
        assert pr['cells_sha256']==contract['cells_sha256']
        assert sha(track(pr['input']))==pr['input_sha256']
        protocols[method]=pr
    original=pd.read_csv(track(STORY/'alignment/umap/full_94488_umap_coordinates.csv.gz'),low_memory=False)
    assert np.array_equal(coords['RAMP'],original[['UMAP1','UMAP2']].to_numpy())
    assert np.array_equal(meta.cell_id,original.cell_uid)
    assert np.array_equal(meta.celltype,original.plot_label)
    values={'Raw':np.load(track(BASE/'input/analysis_ready.npy')),
            'RAMP':np.load(track(BASE/'input/ramp.npy'))}
    for method in ['Harmony-direct','cyCombine','CytoNorm','fastMNN']:
        p=HERE/'fastmnn_reconstructed.csv.gz' if method=='fastMNN' else BASE/'methods'/KEYS[method]/'corrected_expression.csv.gz'
        frame=pd.read_csv(track(p),dtype={'cell_id':str})
        assert len(frame)==94488 and frame.cell_id.is_unique and set(frame.cell_id)==set(meta.cell_id)
        values[method]=frame.set_index('cell_id').loc[meta.cell_id,markers].to_numpy()
        if method!='fastMNN':
            assert np.allclose(values[method],np.load(track(BASE/'methods'/KEYS[method]/'values.npy')),rtol=1e-10,atol=1e-10)
    for v in values.values():assert v.shape==(94488,22) and np.isfinite(v).all()
    track(BASE/'methods/fastmnn/fit.rds')
    track(HERE/'extract_fastmnn.R')
    track(ROOT/'Main Figure/fastmnn_reconstructed_audit_20260914/official_convertPCsToSCE.R')
    return meta,markers,coords,values,style,protocols,meta


def axes(fig,x,y,w,h):
    """Millimetres; y is distance down from top."""
    return fig.add_axes([x/WIDTH,1-(y+h)/HEIGHT,w/WIDTH,h/HEIGHT])


def text(fig,x,y,label,size=8.5,weight='bold',color=INK,ha='left',va='top'):
    label=label.replace('Harmony-direct','Harmony')
    return fig.text(x/WIDTH,1-y/HEIGHT,label,fontsize=size,weight=weight,color=color,ha=ha,va=va)


def point_orders(meta,by,style):
    seed=style['plot_seed']
    if by=='Dataset':return [np.random.default_rng(seed).permutation(len(meta))]
    result=[]
    for i,label in enumerate(style['draw_order_bottom_to_top']):
        ix=np.flatnonzero(meta.celltype.eq(label))
        result.append(ix[np.random.default_rng(seed+i).permutation(len(ix))])
    return result


def scatter(ax,meta,xy,by,style):
    orders=point_orders(meta,by,style)
    artists=[]
    if by=='Dataset':
        order=orders[0]
        artists.append(ax.scatter(xy[order,0],xy[order,1],c=meta.dataset.map(style['dataset_palette']).to_numpy()[order],
             s=SCATTER_AREA,alpha=style['dataset_alpha'],linewidths=0,rasterized=True))
    else:
        for label,order in zip(style['draw_order_bottom_to_top'],orders):
            artists.append(ax.scatter(xy[order,0],xy[order,1],color=style['celltype_palette'][label],
               s=SCATTER_AREA,alpha=style['celltype_alpha'],linewidths=0,rasterized=True))
    low,high=xy.min(axis=0),xy.max(axis=0);span=high-low
    ax.set_xlim(low[0]-.02*span[0],high[0]+.02*span[0])
    ax.set_ylim(low[1]-.02*span[1],high[1]+.25*span[1])
    ax.set_aspect('equal',adjustable='box');ax.set_axis_off()
    return artists


def ridge(ax,v,meta,method,marker,style):
    return ridge_common.draw_ridge(ax,v,meta,method,marker,RIDGE_CONFIG,AUDIT)


def main():
    track(HERE/'RIDGE_CONFIG.json');track(Path(ridge_common.__file__))
    global SCATTER_AREA
    meta,markers,coords,values,style,protocols,display_meta = load()
    span=np.ptp(coords['RAMP'],axis=0)
    ramp_height_mm=min(27.,39.*(1.27*span[1])/(1.04*span[0]))
    SCATTER_AREA=style['story_point_area']*(ramp_height_mm/style['source_plot_height_mm'])**2
    plt.rcParams.update({'font.family':'Arial','font.size':8,'svg.fonttype':'none',
                         'pdf.fonttype':42,'ps.fonttype':42,'text.color':INK})
    fig=plt.figure(figsize=(WIDTH/25.4,HEIGHT/25.4),facecolor='white',dpi=200)
    text(fig,9,9,'HNSCC story | CITE-seq and CODEX',size=13)
    text(fig,9,17.0,'94,488 cells  |  77 donors/patients  |  22 shared proteins',size=8.2)
    dataset_names = style['dataset_display_labels']
    handles = [Line2D([],[],ls='',marker='o',ms=3.5,mew=0,color=style['dataset_palette'][d],label=dataset_names[d]) for d in dataset_names]
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
        label={'Raw':'Raw','fastMNN':'fastMNN (reconstructed)'}.get(method,method)
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
    labels=style['celltype_legend_order']
    legend_labels=[labels[i] for col in range(3) for i in (col,col+3,col+6) if i<len(labels)]
    handles=[Line2D([],[],ls='',marker='o',ms=3.2,mew=0,color=style['celltype_palette'][s],label=s) for s in legend_labels]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,5/HEIGHT),ncol=3,
               frameon=False,prop={'size':7.4,'weight':'bold'},columnspacing=1.6,handletextpad=.35,labelspacing=.6)
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
    out=OUT/'output';(out/'pdf').mkdir(parents=True,exist_ok=True)
    stem='hnscc_story_supplement'
    fig.savefig(out/f'{stem}.png',dpi=300,facecolor='white')
    fig.savefig(out/'pdf'/f'{stem}.pdf',dpi=300,facecolor='white')
    fig.savefig(out/f'{stem}.svg',dpi=300,facecolor='white')
    plt.close(fig)
    audit=pd.DataFrame(AUDIT)
    ridge_common.validate_audit(audit,RIDGE_CONFIG)
    audit.to_csv(OUT/'ridge_display_audit.csv',index=False)
    meta.groupby(['dataset','unit_key','celltype']).size().rename('cells').reset_index().to_csv(OUT/'cell_counts.csv',index=False)
    display_meta.groupby(['dataset','unit_key','celltype']).size().rename('cells').reset_index().to_csv(OUT/'display_cell_counts.csv',index=False)
    assert all(sha(p)==h for p,h in SOURCES.items())
    report=dict(version='layout_v1_story94488',benchmark='hnscc_story',cells=94488,sample_units=77,shared_markers=22,
        dataset_counts=meta.groupby('dataset').size().to_dict(),umap_methods=METHODS,ridge_methods=RIDGE_METHODS,
        ridge_markers=MARKERS,all_cells_retained_for_alignment_and_ridges=True,all_sample_units_retained=True,
        display_grouping_as_story=True,display_cells=94488,display_source='Exact current story 94488 IDs and display labels; same complete pool across methods',
        alignment_refit=False,umap_refit=True,ramp_umap_reused=True,coordinate_transform=None,equal_aspect=True,
        maxfuse_display='Current full-active-panel MaxFuse; 130 CITE-seq / 57 CODEX active proteins; 22 shared. Pairwise 20D saved output.',
        fastmnn_display='UMAP: saved 21 corrected PCs; ridge: PCs @ rotation.T, reconstructed coordinates, not original intensity units.',
        raw_ridge_display='Frozen analysis_ready: CITE-seq ADT log1p; Haist signed provider values identity. Raw UMAP uses source-scale values.',
        ridge_range_quantiles=[.005,.995],ridge_bins=160,ridge_sigma_bins=6,ridge_height='Each curve normalized to its own density maximum',
        dataset_palette=style['dataset_palette'],celltype_palette=style['celltype_palette'],draw_order=style['draw_order_bottom_to_top'],
        umap_protocols=protocols,scatter_alpha={'Dataset':style['dataset_alpha'],'Cell type':style['celltype_alpha']},
        scatter_area_pt2=SCATTER_AREA,source_story_scatter_area_pt2=style['story_point_area'],
        source_story_plot_height_mm=style['source_plot_height_mm'],ramp_plot_height_mm=ramp_height_mm,
        scatter_draw_path='Exact story per-cell order; one dataset collection / nine cell-type collections; no outlines',
        coordinate_margin_fractions=style['margin_fractions'],dataset_legend_order=list(dataset_names.values()),celltype_legend_reading_order=labels,
        page_mm=[WIDTH,HEIGHT],layout=GEOMETRY,text_bounds_checked=True,text_bounds=bounds,input_sha256=SOURCES,
        inputs_unchanged=True,script_sha256=sha(__file__),output_sha256={str(p.relative_to(OUT)):sha(p) for p in out.rglob('*') if p.is_file()})
    report.update(ridge_common.manifest_extra(RIDGE_CONFIG))
    (OUT/'MANIFEST.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(out/f'{stem}.png',flush=True)


if __name__=='__main__':main()
