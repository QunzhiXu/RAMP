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

STORY = ROOT / 'story_lung_HLA_ABC'
RELEASE = STORY / 'releases/senosain140k_20260921'
BASE = STORY / 'Comparison_senosain140k_20260921'
METHODS = ['Raw','Harmony-direct','cyCombine','CytoNorm','fastMNN','MaxFuse','RAMP']
RIDGE_METHODS = [m for m in METHODS if m != 'MaxFuse']
KEYS = dict(zip(METHODS,['raw','harmony','cycombine','cytonorm','fastmnn','maxfuse_hub0','ramp']))
MARKERS = ['CD3','CD4','CD8']
WIDTH, HEIGHT = 210., 297.
INK, FRAME, GREEN = '#263238', '#9AA1A5', '#00866A'
SOURCES, AUDIT, GEOMETRY = {}, [], []
STORY_POINT_AREA = .60
STORY_ALPHA = .65
STORY_PLOT_HEIGHT_MM = 1.8815771449095156 * 25.4
SCATTER_AREA = STORY_POINT_AREA * (27. / STORY_PLOT_HEIGHT_MM) ** 2


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
    story_qa=json.loads(track(STORY/'Figure/Panel B/provenance/VERIFICATION.json').read_text())
    assert np.isclose(story_qa['plot_size_inches'][1]*25.4,STORY_PLOT_HEIGHT_MM)
    meta=pd.read_csv(track(HERE/'display_metadata.csv.gz'),low_memory=False)
    fitmeta=pd.read_csv(track(BASE/'input/fit_metadata.csv.gz'),low_memory=False)
    assert len(meta)==84000 and meta.unit_key.nunique()==84 and meta.cell_id.is_unique
    assert meta.groupby('dataset').size().eq(28000).all()
    assert meta.groupby('unit_key').size().eq(1000).all()
    assert np.array_equal(meta.cell_id,fitmeta.cell_id)
    assert np.array_equal(meta.unit_key,fitmeta.unit_key)
    assert set(meta.celltype)==set(style['celltype_palette'])
    markers=track(BASE/'input/markers.txt').read_text().splitlines()
    assert len(markers)==9 and set(MARKERS)<=set(markers)
    proof=json.loads(track(BASE/'input/manifest.json').read_text())
    assert markers==proof['markers']
    for name,digest in proof['output_sha256'].items():
        assert sha(track(BASE/'input'/name))==digest
    contract=json.loads(track(HERE/'DISPLAY_CONTRACT.json').read_text())
    assert contract['status']=='complete' and contract['display_cells']==84000
    assert all(sha(track(p))==h for p,h in contract['input_sha256'].items())
    rows=np.load(track(HERE/'display_rows.npy'))
    assert np.array_equal(rows,np.arange(84000))
    coords,protocols={},{}
    for method in METHODS:
        coords[method]=np.load(track(HERE/'coordinates'/f'{method}.npy'))
        pr=json.loads(track(HERE/'coordinates'/f'{method}.json').read_text())
        assert coords[method].shape==(84000,2) and np.isfinite(coords[method]).all()
        assert sha(HERE/'coordinates'/f'{method}.npy')==pr['coordinate_file_sha256']
        assert pr['protocol']==contract['protocol'] and not pr['labels_used_for_fit']
        assert pr['cells_sha256']==contract['cells_sha256']
        assert sha(track(pr['input']))==pr['input_sha256']
        protocols[method]=pr
    assert np.array_equal(coords['RAMP'],np.load(track(RELEASE/'alignment/RAMP_UMAP.npy')))
    assert np.array_equal(coords['RAMP'],np.load(track(STORY/'Figure/Panel B/data/RAMP_UMAP.npy')))
    values={'Raw':np.load(track(BASE/'input/analysis_ready.npy')),
            'RAMP':np.load(track(BASE/'input/ramp.npy'))}
    for method in ['Harmony-direct','cyCombine','CytoNorm','fastMNN']:
        p=HERE/'fastmnn_reconstructed.csv.gz' if method=='fastMNN' else BASE/'methods'/KEYS[method]/'corrected_expression.csv.gz'
        frame=pd.read_csv(track(p),dtype={'cell_id':str})
        assert len(frame)==84000 and frame.cell_id.is_unique and set(frame.cell_id)==set(meta.cell_id)
        values[method]=frame.set_index('cell_id').loc[meta.cell_id,markers].to_numpy()
        if method!='fastMNN':
            assert np.allclose(values[method],np.load(track(BASE/'methods'/KEYS[method]/'values.npy')),rtol=1e-10,atol=1e-10)
    for v in values.values():assert v.shape==(84000,9) and np.isfinite(v).all()
    track(BASE/'methods/fastmnn/fit.rds')
    track(HERE/'extract_fastmnn.R')
    track(ROOT/'Main Figure/fastmnn_reconstructed_audit_20260914/official_convertPCsToSCE.R')
    return meta,markers,coords,values,style,protocols,meta


def axes(fig,x,y,w,h):
    """Millimetres; y is distance down from top."""
    return fig.add_axes([x/WIDTH,1-(y+h)/HEIGHT,w/WIDTH,h/HEIGHT])


def text(fig,x,y,label,size=8.5,weight='bold',color=INK,ha='left',va='top'):
    return fig.text(x/WIDTH,1-y/HEIGHT,label,fontsize=size,weight=weight,color=color,ha=ha,va=va)


def point_order(meta,by):
    order=np.random.default_rng(42).permutation(len(meta))
    if by=='Dataset':
        return order
    types=meta.celltype.to_numpy()[order]
    bottom=types=='Tumor / epithelial'
    upper=['Fibroblast / stroma','Myeloid / APC','CD4 T','Endothelial']
    middle=~(bottom|np.isin(types,upper))
    masks=[bottom,middle]+[types==label for label in upper]
    return np.concatenate([order[mask] for mask in masks])


def scatter(ax,meta,xy,by,style):
    order=point_order(meta,by)
    field,palette=('dataset','dataset_palette') if by=='Dataset' else ('celltype','celltype_palette')
    colors=meta[field].map(style[palette]).to_numpy()[order]
    # Use the story's one-collection draw path and alpha. Scale marker area
    # quadratically with the reduction in physical plot height (47.79 -> 27 mm).
    artist=ax.scatter(xy[order,0],xy[order,1],c=colors,s=SCATTER_AREA,alpha=STORY_ALPHA,
                      linewidths=0,rasterized=True)
    low,high=xy.min(axis=0),xy.max(axis=0)
    pad=(high-low)*.035  # Same margin fraction as the current story Panel B.
    ax.set_xlim(low[0]-pad[0],high[0]+pad[0]);ax.set_ylim(low[1]-pad[1],high[1]+pad[1])
    ax.set_aspect('equal',adjustable='box');ax.set_axis_off()
    return artist


def ridge(ax,v,meta,method,marker,style):
    low,high = np.quantile(v,[.005,.995])
    assert high > low
    ds = meta.dataset.astype(str).to_numpy()
    units = meta.unit_key.astype(str).to_numpy()
    groups = [(d,u) for d in meta.dataset.unique() for u in sorted(set(units[ds==d]))]
    level,previous = 0.,None
    for i,(d,u) in enumerate(reversed(groups)):
        if previous is not None and d != previous: level += .36
        vals = v[(ds==d)&(units==u)]
        counts,edges = np.histogram(vals,bins=160,range=(low,high))
        density = gaussian_filter1d(counts.astype(float),sigma=6,mode='constant')
        if density.max()>0: density /= density.max()
        x = (edges[:-1]+edges[1:])/2
        y = level+.70*density
        ax.fill_between(x,level,y,color=style['dataset_palette'][d],alpha=.82,lw=0,zorder=2*i+1)
        ax.plot(x,y,color='#46505A',lw=.28,zorder=2*i+2)
        AUDIT.append(dict(method=method,marker=marker,dataset=d,unit_key=u,n_cells=len(vals),
            display_low=float(low),display_high=float(high),n_below=int((vals<low).sum()),
            n_above=int((vals>high).sum()),bins=160,sigma_bins=6,
            raw_assay_transform=method=='Raw',peak_height_normalized=True))
        level += .57;previous = d
    ax.set_xlim(low,high);ax.set_ylim(-.09,level-.57+.70+.04)
    if low < 0 < high:
        ax.axvline(0,color='#666666',lw=.35,ls=(0,(2,3)),alpha=.7,zorder=100)
    ax.set_yticks([])
    ticks = MaxNLocator(3).tick_values(low,high)
    ax.set_xticks(ticks[(ticks>=low)&(ticks<=high)])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x/1000:g}k' if abs(x)>=1000 else f'{x:g}'))
    ax.tick_params(axis='x',labelsize=6.8,length=1.7,pad=1.1,width=.4,colors=INK)
    for spine in ax.spines.values():spine.set_color(FRAME);spine.set_linewidth(.35)


def main():
    meta,markers,coords,values,style,protocols,display_meta = load()
    plt.rcParams.update({'font.family':'Arial','font.size':8,'svg.fonttype':'none',
                         'pdf.fonttype':42,'ps.fonttype':42,'text.color':INK})
    fig=plt.figure(figsize=(WIDTH/25.4,HEIGHT/25.4),facecolor='white',dpi=200)
    text(fig,9,9,'Lung HLA-ABC story | CyTOF, IMC and mIF',size=13)
    text(fig,9,17.0,'84,000 cells  |  84 patients  |  9 shared features',size=8.2)
    dataset_names = style['dataset_display_labels']
    handles = [Line2D([],[],ls='',marker='o',ms=3.5,mew=0,color=style['dataset_palette'][d],label=dataset_names[d]) for d in dataset_names]
    fig.legend(handles=handles,loc='upper right',bbox_to_anchor=(201/WIDTH,1-17/HEIGHT),
               ncol=3,frameon=False,prop={'size':8,'weight':'bold'},columnspacing=1.,handletextpad=.35,borderaxespad=0)
    fig.add_artist(Line2D([9/WIDTH,201/WIDTH],[1-25/HEIGHT]*2,transform=fig.transFigure,lw=.5,color='#CDD3D5'))
    text(fig,9,29,'A',size=12)
    text(fig,15,29,'Cross-platform integration',size=9.4)
    text(fig,100,29,'B',size=12)
    text(fig,106,29,'Marker distributions',size=9.4)
    text(fig,28.5,37.5,'Dataset',size=8.5,ha='center')
    text(fig,70.5,37.5,'Cell type',size=8.5,ha='center')
    for j,marker in enumerate(MARKERS):text(fig,115.5+j*34.5,37.5,marker,size=9,ha='center')
    # Two independent blocks: 7 UMAP rows and 6 ridge rows occupy equal heights.
    for i,method in enumerate(METHODS):
        top=45+i*31.8
        label='MaxFuse (CyTOF hub)' if method=='MaxFuse' else method
        text(fig,9,top,label,size=8.2,color=GREEN if method=='RAMP' else INK)
        for j,by in enumerate(['Dataset','Cell type']):
            x,y,w,h=9+j*42,top+3.6,39,27
            ax=axes(fig,x,y,w,h)
            scatter(ax,display_meta,coords[method],by,style)
            fig.add_artist(Rectangle((x/WIDTH,1-(y+h)/HEIGHT),w/WIDTH,h/HEIGHT,
                transform=fig.transFigure,fill=False,ec=GREEN if method=='RAMP' else FRAME,lw=.4,zorder=100))
            GEOMETRY.append(dict(block='UMAP',method=method,colour_by=by,x_mm=x,y_mm=y,width_mm=w,height_mm=h))
    for i,method in enumerate(RIDGE_METHODS):
        top=45+i*37.24
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
    fig.add_artist(Line2D([95/WIDTH]*2,[1-29/HEIGHT,1-267/HEIGHT],transform=fig.transFigure,lw=.5,color='#E2E6E7'))
    labels=list(style['celltype_palette'])
    # Matplotlib fills columns; reorder handles so reading across the two rows
    # follows the original story's vertical legend order.
    legend_labels=[labels[i] for col in range(4) for i in (col,col+4) if i<len(labels)]
    handles=[Line2D([],[],ls='',marker='o',ms=3.2,mew=0,color=style['celltype_palette'][s],label=s.replace('−','-')) for s in legend_labels]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,8/HEIGHT),ncol=4,
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
    stem='lung_hla_abc_story_supplement'
    fig.savefig(out/f'{stem}.png',dpi=300,facecolor='white')
    fig.savefig(out/'pdf'/f'{stem}.pdf',dpi=300,facecolor='white')
    fig.savefig(out/f'{stem}.svg',dpi=300,facecolor='white')
    plt.close(fig)
    audit=pd.DataFrame(AUDIT)
    assert len(audit)==6*3*84
    assert audit.groupby(['method','marker']).n_cells.sum().eq(84000).all()
    audit.to_csv(HERE/'ridge_display_audit.csv',index=False)
    meta.groupby(['dataset','unit_key','celltype']).size().rename('cells').reset_index().to_csv(HERE/'cell_counts.csv',index=False)
    display_meta.groupby(['dataset','unit_key','celltype']).size().rename('cells').reset_index().to_csv(HERE/'display_cell_counts.csv',index=False)
    assert all(sha(p)==h for p,h in SOURCES.items())
    report=dict(version='layout_v1_story84k',benchmark='lung_story',cells=84000,sample_units=84,shared_markers=9,
        dataset_counts=meta.groupby('dataset').size().to_dict(),umap_methods=METHODS,ridge_methods=RIDGE_METHODS,
        ridge_markers=MARKERS,all_cells_retained_for_alignment_and_ridges=True,all_sample_units_retained=True,additional_label_merging=False,display_grouping_as_story=True,
        display_cells=84000,display_cells_per_platform=28000,display_source='Exact current story 84k IDs and display labels; all cells, same across all methods',
        alignment_refit=False,umap_refit=True,ramp_umap_reused=True,coordinate_transform=None,equal_aspect=True,story_display_labels_reused=True,coordinate_margin_fraction=.035,
        maxfuse_display='Current Figure 2 full-active-panel MaxFuse; 34 CyTOF / 41 IMC / 33 mIF active proteins, 9 shared. Fixed first registered CyTOF hub. No alignment refit; scores retain three-hub median aggregation.',
        fastmnn_display='UMAP: all 84k cells in archived 8 corrected PCs, story display protocol; ridge: PCs @ rotation.T per batchelor convertPCsToSCE, not original intensity units.',
        raw_ridge_display='Frozen analysis_ready: Senosain asinh(x/5), Cords/Zheng asinh(x/1). Display only; raw UMAP uses source-scale values.',ridge_range_quantiles=[.005,.995],
        ridge_bins=160,ridge_sigma_bins=6,ridge_height='Each curve normalized to its own density maximum',
        dataset_palette=style['dataset_palette'],celltype_palette=style['celltype_palette'],
        draw_order=style['draw_order_bottom_to_top'],umap_protocols=protocols,
        rendering_revision='story_render_match_20260922',
        scatter_alpha=STORY_ALPHA,scatter_area_pt2=SCATTER_AREA,
        source_story_scatter_area_pt2=STORY_POINT_AREA,source_story_plot_height_mm=STORY_PLOT_HEIGHT_MM,
        scatter_draw_path='Single collection, exact per-cell story order, zero-width outlines',
        dataset_legend_order=list(dataset_names.values()),celltype_legend_reading_order=labels,
        page_mm=[WIDTH,HEIGHT],layout=GEOMETRY,text_bounds_checked=True,
        text_bounds=bounds,input_sha256=SOURCES,inputs_unchanged=True,script_sha256=sha(__file__),
        output_sha256={str(p.relative_to(HERE)):sha(p) for p in out.rglob('*') if p.is_file()})
    (HERE/'MANIFEST.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(out/f'{stem}.png',flush=True)


if __name__=='__main__':main()
