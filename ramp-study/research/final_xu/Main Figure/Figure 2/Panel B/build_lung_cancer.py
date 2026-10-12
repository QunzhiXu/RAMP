"""Render the frozen lung-cancer benchmark using the accepted Figure 1B style."""
from pathlib import Path
import os
import sys
import json

HERE = Path(__file__).resolve().parent
MAIN = HERE.parents[1]
HERE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(HERE/'.cache/matplotlib'))
os.environ.setdefault('MPLBACKEND', 'Agg')
sys.path.insert(0, str(MAIN/'Panel B/benchmark_review'))
from build_review import load_case, sha, legend, METHODS, DATASET_COLORS
from build_ridge_review import load_matrices, tick
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator
from scipy.ndimage import gaussian_filter1d

PALETTE = {
    'Tumor/Epithelial': '#D56D79', 'Mix': '#BFC3C8', 'Myeloid': '#DDB130',
    'CD4 T': '#2878B5', 'CD8 T': '#E46A2E', 'B-like': '#32906C',
    'Endothelial': '#079EAA',
}
ORDER = ['Myeloid', 'Tumor/Epithelial', 'Mix', 'Endothelial', 'B-like', 'CD4 T', 'CD8 T']
FRAME = '#8C8C8C'
MARKERS = ['CD3', 'VIMENTIN', 'EPI']
DATASET_LABELS = {'Senosain CyTOF': 'CyTOF', 'Cords IMC': 'IMC', 'Monkman CODEX': 'CODEX'}


def draw_ridge(ax, values, meta, method, marker, audit):
    low, high = np.quantile(values, [.005,.995])
    if high <= low: high = low+1
    datasets = meta.dataset.astype(str).to_numpy()
    units = meta.unit_key.astype(str).to_numpy()
    colors = dict(zip(meta.dataset.unique(), DATASET_COLORS))
    groups = [(dataset, unit) for dataset in colors for unit in sorted(set(units[datasets == dataset]))]
    level, previous = 0., None
    for i,(dataset,unit) in enumerate(reversed(groups)):
        if previous is not None and dataset != previous: level += .36
        vals = values[(datasets == dataset)&(units == unit)]
        counts, edges = np.histogram(vals,bins=160,range=(low,high))
        smooth = gaussian_filter1d(counts.astype(float),sigma=6,mode='constant')
        if smooth.max()>0: smooth /= smooth.max()
        x=(edges[1:]+edges[:-1])/2
        y=level+.70*smooth
        ax.fill_between(x,level,y,color=colors[dataset],alpha=.82,linewidth=0,zorder=2*i+1)
        ax.plot(x,y,color='#46505A',linewidth=.55,zorder=2*i+2)
        audit.append({'benchmark':'lung_cancer','method':method,'marker':marker,
                      'dataset':dataset,'unit_key':unit,'n_cells':len(vals),
                      'smoothing_sigma_bins':6,'display_low':float(low),'display_high':float(high),
                      'n_below_display':int((vals<low).sum()),'n_above_display':int((vals>high).sum())})
        level += .57
        previous = dataset
    ax.set_xlim(low,high); ax.set_ylim(-.09,level-.57+.70+.04)
    if low < 0 < high:
        ax.axvline(0,color='#666666',linewidth=.55,linestyle=(0,(2,3)),zorder=100,alpha=.7)
    ax.set_yticks([])
    ax.xaxis.set_major_locator(MaxNLocator(3)); ax.xaxis.set_major_formatter(FuncFormatter(tick))
    ax.tick_params(axis='x',labelsize=8,length=2.1,pad=2,width=.5)
    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_linewidth(.45); spine.set_color(FRAME)


def draw_scatter(ax,meta,xy,color_by):
    if color_by == 'Dataset':
        palette = dict(zip(meta.dataset.unique(),DATASET_COLORS))
        colors = np.asarray([palette[label] for label in meta.dataset])
        order = np.random.default_rng(42).permutation(len(meta))
        ax.scatter(xy[order,0],xy[order,1],c=colors[order],s=.28,alpha=.72,
                   edgecolors='none',rasterized=True)
    else:
        labels = meta.celltype.to_numpy()
        for z,label in enumerate(ORDER):
            idx = np.random.default_rng(42).permutation(np.flatnonzero(labels==label))
            size,alpha = (.28,.65) if label=='Mix' else (.34,.85)
            ax.scatter(xy[idx,0],xy[idx,1],c=PALETTE[label],s=size,alpha=alpha,
                       edgecolors='none',rasterized=True,zorder=z+1)
    low,high=xy.min(axis=0),xy.max(axis=0)
    margin=(high-low)*.025
    ax.set_xlim(low[0]-margin[0],high[0]+margin[0])
    ax.set_ylim(low[1]-margin[1],high[1]+margin[1])
    ax.set_aspect('equal',adjustable='box'); ax.set_axis_off()


def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none'})
    sources,audit={},[]
    meta,coords=load_case('lung_cancer',sources)
    markers,matrices=load_matrices('lung_cancer',meta,sources)
    assert len(meta)==60000 and meta.unit_key.nunique()==30
    assert set(meta.celltype)==set(PALETTE)==set(ORDER) and len(ORDER)==7
    frozen=json.loads((MAIN/'Panel B/benchmark_review/ridge_review_manifest.json').read_text())
    assert all(frozen['input_sha256'][path]==value for path,value in sources.items())
    reference=MAIN/'Panel B/tnbc_bandwidth_review/adtnorm_style_review/manifest.json'
    ref=json.loads(reference.read_text())
    row_height=ref['layout']['umap_axes'][0]['height_in']
    row_gap,header,footer=.16,.90,.60
    height=header+4*row_height+3*row_gap+footer
    width=ref['layout']['figure_width_in']
    ridge_left,ridge_width,ridge_gap=.56,1.313,.10
    pair_left=ridge_left+3*ridge_width+2*ridge_gap+.12
    slot_width=ref['layout']['umap_axes'][0]['frame_width_in']
    umap_gap=.10
    assert np.isclose(pair_left+2*slot_width+umap_gap+.14,width)
    fig=plt.figure(figsize=(width,height),facecolor='white',dpi=210)
    fig.text(.16/width,1-.29/height,'B',fontsize=17,weight='bold')
    fig.text(ridge_left/width,1-.29/height,'Cross-platform alignment in lung cancer',
             fontsize=13.5,weight='bold')
    fig.text(1-.14/width,1-.13/height,'60,000 cells  |  30 sample units',fontsize=9,ha='right',va='top')
    da=fig.add_axes([.04,1-.54/height,.95,.20/height])
    legend(da,[DATASET_LABELS[d] for d in meta.dataset.unique()],DATASET_COLORS,3,9)
    geometry=[]
    for row,method in enumerate(METHODS):
        bottom=height-header-(row+1)*row_height-row*row_gap
        for col,marker in enumerate(MARKERS):
            left=ridge_left+col*(ridge_width+ridge_gap)
            ax=fig.add_axes([left/width,bottom/height,ridge_width/width,row_height/height])
            vals=matrices[method][:,markers.index(marker)]
            audit_method=method
            if method=='Unaligned':
                vals=np.arcsinh(vals); audit_method='Unaligned_arcsinh_display'
            draw_ridge(ax,vals,meta,audit_method,marker,audit)
            if row==0: ax.set_title(marker,fontsize=11.5,weight='bold',pad=8)
            if col==0:
                ax.set_ylabel('Raw\n(arcsinh)' if method=='Unaligned' else METHODS[method],
                              fontsize=11.5,weight='bold',labelpad=8)
        for col,color_by in enumerate(['Dataset','Cell type']):
            slot_left=pair_left+col*(slot_width+umap_gap)
            aspect=np.ptp(coords[method][:,0])/np.ptp(coords[method][:,1])
            actual_height=min(row_height,slot_width/aspect)
            actual_width=actual_height*aspect
            left=slot_left+(slot_width-actual_width)/2
            actual_bottom=bottom+(row_height-actual_height)/2
            ax=fig.add_axes([left/width,actual_bottom/height,actual_width/width,actual_height/height])
            draw_scatter(ax,meta,coords[method],color_by)
            fig.add_artist(Rectangle((slot_left/width,bottom/height),slot_width/width,row_height/height,
                           fill=False,edgecolor=FRAME,linewidth=.45,transform=fig.transFigure,zorder=100))
            if row==0:
                fig.text((slot_left+slot_width/2)/width,(bottom+row_height+.11)/height,
                         color_by,ha='center',va='bottom',fontsize=11.5,weight='bold')
            geometry.append({'method':method,'color_by':color_by,'width_in':actual_width,
                             'height_in':actual_height,'frame_width_in':slot_width,'frame_height_in':row_height})
    labels=meta.celltype.value_counts().index.tolist()
    la=fig.add_axes([.02,.055/height,.97,.34/height])
    legend(la,labels,[PALETTE[label] for label in labels],5,8.5)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    artists=list(fig.texts)
    for ax in fig.axes:
        artists += [t for t in [ax.title,ax.yaxis.label] if t.get_text()]
        if ax.axison:
            artists += [t for t in ax.get_xticklabels() if t.get_text() and t.get_visible()]
        if ax.get_legend() is not None: artists.append(ax.get_legend())
    for artist in artists:
        box=artist.get_window_extent(renderer)
        assert box.x0>=-1 and box.y0>=-1 and box.x1<=fig.bbox.width+1 and box.y1<=fig.bbox.height+1,(str(artist),box)
    path=HERE/'lung_cancer_ridge_umap_figure1_style.png'
    fig.savefig(path,dpi=210)
    fig.savefig(path.with_suffix('.svg'),dpi=210)
    plt.close(fig)
    table=pd.DataFrame(audit)
    assert len(table)==4*3*30
    assert table.groupby(['method','marker']).n_cells.sum().eq(len(meta)).all()
    table.to_csv(HERE/'ridge_display_audit.csv',index=False)
    counts=meta.groupby(['dataset','unit_key','celltype']).size().rename('n_cells').reset_index()
    counts.to_csv(HERE/'display_cell_counts.csv',index=False)
    pd.DataFrame({'celltype':list(PALETTE),'color':list(PALETTE.values())}).to_csv(HERE/'celltype_palette.csv',index=False)
    assert all(sha(Path(path))==value for path,value in sources.items())
    report={'panel_label':'B','title':'Cross-platform alignment in lung cancer','cells':len(meta),
            'sample_units':30,'cell_types':7,'shared_markers':8,'ridge_markers':MARKERS,'methods':METHODS,
            'input_sha256':sources,'inputs_unchanged':True,'matches_previous_frozen_input_hashes':True,
            'style_reference':str(reference),'style_reference_sha256':sha(reference),
            'all_cells_retained':True,'labels_merged':False,'alignment_refit':False,'umap_refit':False,
            'coordinate_transform':None,'equal_aspect_preserved':True,'same_rules_for_all_methods':True,
            'ridge_bins':160,'ridge_sigma_bins':6,'ridge_range_quantiles':[.005,.995],
            'raw_ridge_display':'arcsinh(x/1)','ridge_height_normalization':'Each curve divided by its own peak',
            'ridge_peak_height':.70,'ridge_fill_alpha':.82,'ridge_outline_color':'#46505A',
            'frame_color':FRAME,'frame_linewidth_pt':.45,'dataset_palette':dict(zip(meta.dataset.unique(),DATASET_COLORS)),
            'dataset_display_labels':DATASET_LABELS,
            'celltype_palette':PALETTE,'draw_order_bottom_to_top':ORDER,
            'scatter_dataset_size':.28,'scatter_dataset_alpha':.72,'scatter_celltype_size':.34,'scatter_celltype_alpha':.85,
            'mix_size':.28,'mix_alpha':.65,
            'layout':{'figure_width_in':width,'figure_height_in':height,'row_height_in':row_height,
                      'row_gap_in':row_gap,'ridge_width_in':ridge_width,'ridge_gap_in':ridge_gap,
                      'umap_frame_gap_in':umap_gap,'footer_in':footer,'legend_columns':5,'umap_axes':geometry}}
    (HERE/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(path,flush=True)


if __name__=='__main__': main()
