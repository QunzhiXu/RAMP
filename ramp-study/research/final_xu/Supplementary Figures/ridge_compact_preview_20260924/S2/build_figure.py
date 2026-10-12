"""One-page benchmark supplement: frozen UMAPs and saved marker outputs."""
from pathlib import Path
import os
import sys
sys.dont_write_bytecode = True
HERE = Path('/Users/qxu6/Documents/New project/final_xu/Supplementary Figures/TNBC/ridge_full_range_20260922')
OUT = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
os.environ['MPLCONFIGDIR'] = str(OUT / '.cache/matplotlib')
os.environ['MPLBACKEND'] = 'Agg'
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key] = '1'
import hashlib
import ast
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

BASE = ROOT / 'benchmark_results/tnbc'
COORD = ROOT / 'benchmark_results/comparison_evaluation/tnbc/umap'
RECON = ROOT / 'Main Figure/fastmnn_reconstructed_audit_20260914/tnbc'
STYLE = ROOT / 'Main Figure/icon_update_20260915/rendering_helpers.py'
MAIN_MANIFEST = ROOT / 'Main Figure/icon_update_20260915/manifest.json'
POINT_AREA_SCALE = (27. / 21.7)**2
METHODS = ['Raw','Harmony-direct','cyCombine','CytoNorm','fastMNN','MaxFuse','RAMP']
RIDGE_METHODS = [m for m in METHODS if m != 'MaxFuse']
STEMS = dict(zip(METHODS, ['Unaligned','Harmony-direct_source_scale','cyCombine',
                          'CytoNorm','fastMNN','MaxFuse_tietscher_imc','RAMP']))
MARKERS = ['CD3','CD8','PANCK']
WIDTH, HEIGHT = 210., 297.
INK, FRAME, GREEN = '#263238', '#9AA1A5', '#00866A'
SOURCES, AUDIT, GEOMETRY = {}, [], []


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
    meta = pd.read_csv(track(BASE/'ramp/alignment/input/cell_metadata.csv.gz'), dtype={'cell_id':str}, low_memory=False)
    markers = track(BASE/'ramp/alignment/input/markers.txt').read_text().splitlines()
    assert len(meta) == 62701 and meta.unit_key.nunique() == 11 and meta.cell_id.is_unique
    assert meta.groupby('dataset').size().to_dict() == {'Tietscher IMC':20000,'Meyer IMC':22701,'Shiao CODEX':20000}
    assert meta.groupby('dataset').unit_key.nunique().to_dict() == {'Tietscher IMC':1,'Meyer IMC':5,'Shiao CODEX':5}
    if 'row_id' in meta: assert np.array_equal(meta.row_id.to_numpy(), np.arange(len(meta)))
    assert len(markers) == 11 and set(MARKERS) <= set(markers)
    constants={}
    for node in ast.parse(track(STYLE).read_text()).body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ['DS_COLORS','PALETTE','ORDER']:
            constants[node.targets[0].id]=ast.literal_eval(node.value)
    style=dict(dataset_palette=dict(zip(meta.dataset.unique(),constants['DS_COLORS'])),
        celltype_palette=constants['PALETTE'],draw_order_bottom_to_top=constants['ORDER'],
        dataset_display_labels={'Tietscher IMC':'IMC1','Meyer IMC':'IMC2','Shiao CODEX':'CODEX'},
        legend_order=['Tumor / epithelial','Stroma','CD4 T','B lineage','Non-joint','Myeloid','CD8 T','Endothelial','NK'],
        source=str(STYLE),source_sha256=sha(STYLE),source_box_mm=[28.6,21.7],target_box_mm=[39.,27.],
        point_area_scale=POINT_AREA_SCALE,dataset_point_area=.058,dataset_alpha=.72,
        default_celltype_point_area=.070,default_celltype_alpha=.85,
        tumor_point_area=.058,tumor_alpha=.5,non_joint_point_area=.058,non_joint_alpha=.65)
    proof=json.loads(track(MAIN_MANIFEST).read_text())
    for name,h in proof['all_source_hashes'].items():
        if '/benchmark_results/' in name:assert sha(track(name))==h
    (OUT/'STORY_STYLE.json').write_text(json.dumps(style,indent=2,ensure_ascii=False)+'\n')
    assert set(meta.celltype) == set(style['celltype_palette'])
    coords, protocols = {}, {}
    for method, stem in STEMS.items():
        p = track(COORD/'coordinates'/f'{stem}.csv.gz')
        frame = pd.read_csv(p).sort_values('row_id')
        assert np.array_equal(frame.row_id.to_numpy(), np.arange(len(meta)))
        if 'cell_id' in frame: assert np.array_equal(frame.cell_id,meta.cell_id)
        xy = frame[['UMAP1','UMAP2']].to_numpy()
        assert xy.shape == (62701,2) and np.isfinite(xy).all()
        coords[method] = xy
        proof = json.loads(track(COORD/'manifests'/f'{stem}.json').read_text())
        assert proof['n_cells'] == len(meta) and not proof['labels_used_for_fit']
        protocols[method] = proof
        # Match the evaluation script's array hash, including shape and dtype.
        npy = track(COORD/'coordinates'/f'{stem}.npy')
        full_precision = np.load(npy)
        assert array_sha(full_precision) == proof['coordinate_sha256']
        assert np.allclose(xy, full_precision, rtol=1e-6, atol=1e-6)
    assert all(p['protocol'] == protocols['RAMP']['protocol'] for p in protocols.values())
    paths = {
        'Raw': BASE/'ramp/alignment/input/raw.npy',
        'Harmony-direct': BASE/'harmony/source_scale/alignment/data/harmony_adjusted_marker_coordinates.csv.gz',
        'cyCombine': BASE/'cycombine/alignment/data/corrected_expression.csv.gz',
        'CytoNorm': BASE/'cytonorm/alignment/data/corrected_expression.csv.gz',
        'fastMNN': RECON/'reconstructed.npy',
        'RAMP': BASE/'ramp/alignment/data/ramp_aligned.npy',
    }
    rm = pd.read_csv(track(RECON/'cell_metadata.csv.gz'),dtype={'cell_id':str},low_memory=False)
    assert np.array_equal(rm.cell_id, meta.cell_id)
    assert track(RECON/'markers.txt').read_text().splitlines() == markers
    assert track(BASE/'cytonorm/alignment/data/markers.txt').read_text().splitlines() == markers
    values = {}
    for method, path in paths.items():
        track(path)
        if path.suffix == '.npy':
            x = np.load(path,allow_pickle=False)
        else:
            frame = pd.read_csv(path,dtype={'cell_id':str})
            assert len(frame) == len(meta) and frame.cell_id.is_unique
            assert set(frame.cell_id) == set(meta.cell_id)
            x = frame.set_index('cell_id').loc[meta.cell_id,markers].to_numpy()
        assert x.shape == (62701,11) and np.isfinite(x).all()
        values[method] = x
    return meta, markers, coords, values, style, protocols


def axes(fig,x,y,w,h):
    """Millimetres; y is distance down from top."""
    return fig.add_axes([x/WIDTH,1-(y+h)/HEIGHT,w/WIDTH,h/HEIGHT])


def text(fig,x,y,label,size=8.5,weight='bold',color=INK,ha='left',va='top'):
    label=label.replace('Harmony-direct','Harmony')
    return fig.text(x/WIDTH,1-y/HEIGHT,label,fontsize=size,weight=weight,color=color,ha=ha,va=va)


def scatter(ax,meta,xy,by,style):
    artists=[]
    if by=='Dataset':
        palette=style['dataset_palette']
        colors=np.array([palette[d] for d in meta.dataset])
        order=np.random.default_rng(42).permutation(len(meta))
        artists.append(ax.scatter(xy[order,0],xy[order,1],c=colors[order],s=.058*POINT_AREA_SCALE,alpha=.72,
                       edgecolors='none',rasterized=True))
    else:
        for z,label in enumerate(style['draw_order_bottom_to_top']):
            idx=np.flatnonzero(meta.celltype.to_numpy()==label)
            idx=np.random.default_rng(42).permutation(idx)
            size,alpha=(.058,.5) if label=='Tumor / epithelial' else (.070,.85)
            if label=='Non-joint':size,alpha=.058,.65
            artists.append(ax.scatter(xy[idx,0],xy[idx,1],c=style['celltype_palette'][label],s=size*POINT_AREA_SCALE,alpha=alpha,
                           edgecolors='none',rasterized=True,zorder=z+1))
    low,high=xy.min(axis=0),xy.max(axis=0);pad=(high-low)*.025
    ax.set_xlim(low[0]-pad[0],high[0]+pad[0]);ax.set_ylim(low[1]-pad[1],high[1]+pad[1])
    ax.set_aspect('equal',adjustable='box');ax.set_axis_off()
    return artists


def ridge(ax,v,meta,method,marker,style):
    return ridge_common.draw_ridge(ax,v,meta,method,marker,RIDGE_CONFIG,AUDIT)


def main():
    track(HERE/'RIDGE_CONFIG.json');track(Path(ridge_common.__file__))
    meta,markers,coords,values,style,protocols = load()
    plt.rcParams.update({'font.family':'Arial','font.size':8,'svg.fonttype':'none',
                         'pdf.fonttype':42,'ps.fonttype':42,'text.color':INK})
    fig=plt.figure(figsize=(WIDTH/25.4,HEIGHT/25.4),facecolor='white',dpi=200)
    text(fig,9,9,'TNBC | IMC1, IMC2 and CODEX',size=13)
    text(fig,9,17.0,'62,701 cells  |  11 sample units  |  11 shared markers',size=8.2)
    dataset_names = style['dataset_display_labels']
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
        label='MaxFuse (IMC1 hub)' if method=='MaxFuse' else method
        text(fig,9,top,label,size=8.2,color=GREEN if method=='RAMP' else INK)
        for j,by in enumerate(['Dataset','Cell type']):
            x,y,w,h=9+j*42,top+3.6,39,27
            ax=axes(fig,x,y,w,h)
            scatter(ax,meta,coords[method],by,style)
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
            if method=='Raw':v=np.arcsinh(v)
            ridge(ax,v,meta,method,marker,style)
            if method=='RAMP':
                for spine in ax.spines.values():spine.set_color(GREEN);spine.set_linewidth(.4)
            GEOMETRY.append(dict(block='ridge',method=method,marker=marker,x_mm=x,y_mm=y,width_mm=w,height_mm=h))
    fig.add_artist(Line2D([95/WIDTH]*2,[1-29/HEIGHT,1-270/HEIGHT],transform=fig.transFigure,lw=.5,color='#E2E6E7'))
    labels=style['legend_order']
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
    stem='tnbc_benchmark_supplement'
    fig.savefig(out/f'{stem}.png',dpi=300,facecolor='white')
    fig.savefig(out/'pdf'/f'{stem}.pdf',dpi=300,facecolor='white')
    fig.savefig(out/f'{stem}.svg',dpi=300,facecolor='white')
    plt.close(fig)
    audit=pd.DataFrame(AUDIT)
    ridge_common.validate_audit(audit,RIDGE_CONFIG)
    audit.to_csv(OUT/'ridge_display_audit.csv',index=False)
    meta.groupby(['dataset','unit_key','celltype']).size().rename('cells').reset_index().to_csv(OUT/'cell_counts.csv',index=False)
    assert all(sha(p)==h for p,h in SOURCES.items())
    report=dict(version='layout_v1',benchmark='tnbc',cells=62701,sample_units=11,shared_markers=11,
        dataset_counts=meta.groupby('dataset').size().to_dict(),umap_methods=METHODS,ridge_methods=RIDGE_METHODS,
        ridge_markers=MARKERS,all_cells_and_sample_units_retained=True,labels_merged=False,
        alignment_refit=False,umap_refit=False,coordinate_transform=None,equal_aspect=True,
        maxfuse_display='First registered dataset hub: Tietscher IMC (IMC1); fixed independently of scores. Figure 2 aggregates metrics across all three hubs.',
        fastmnn_display='UMAP: corrected PCs; ridge: official PCA reconstructed marker coordinates, not original intensity units.',
        raw_ridge_display='arcsinh(x/1) only for visualization',ridge_range_quantiles=[.005,.995],
        ridge_bins=160,ridge_sigma_bins=6,ridge_height='Each curve normalized to its own density maximum',
        dataset_palette=style['dataset_palette'],celltype_palette=style['celltype_palette'],
        draw_order=style['draw_order_bottom_to_top'],umap_protocols=protocols,
        style_reference=str(ROOT/'Supplementary Figures/Melanoma story/layout_v2_balanced40k'),
        point_area_scale=POINT_AREA_SCALE,scatter_dataset_size=.058*POINT_AREA_SCALE,scatter_dataset_alpha=.72,
        scatter_celltype_size=.070*POINT_AREA_SCALE,scatter_celltype_alpha=.85,
        tumor_size=.058*POINT_AREA_SCALE,tumor_alpha=.5,non_joint_size=.058*POINT_AREA_SCALE,non_joint_alpha=.65,
        current_figure1_palette_and_draw_order_retained=True,source_style=style,
        page_mm=[WIDTH,HEIGHT],layout=GEOMETRY,text_bounds_checked=True,
        text_bounds=bounds,input_sha256=SOURCES,inputs_unchanged=True,script_sha256=sha(__file__),
        output_sha256={str(p.relative_to(OUT)):sha(p) for p in out.rglob('*') if p.is_file()})
    report.update(ridge_common.manifest_extra(RIDGE_CONFIG))
    (OUT/'MANIFEST.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(out/f'{stem}.png',flush=True)


if __name__=='__main__':main()
