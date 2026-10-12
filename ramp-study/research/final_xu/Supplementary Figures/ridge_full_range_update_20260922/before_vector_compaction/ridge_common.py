"""Shared ridge rendering. All expression matrices and UMAPs remain frozen."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator,FuncFormatter
from matplotlib.colors import to_rgb

PALETTE={'CODEX':'#2775B6','CyTOF':'#38A169','IMC':'#E58A2B','IMC2':'#F2BD77',
         'CITE-seq':'#9B76B5','AbSeq':'#C35199','MIBI':'#079EAA','mIF':'#D56D79'}
STYLE=dict(fill_alpha=.82,line_width_pt=.35,line_color='80% RGB of platform fill',
           line_alpha=.90,zero_density_outline='continuous over full display range, including zero-density tails',
           bins='Extended grid at the original bin width',sigma_bins=6,range_quantiles=None,
           bandwidth_reference_quantiles=[.005,.995],bandwidth_reference_bins=160,
           gaussian_truncate_sigma=4,padding_bins=26,
           range_reference='Full frozen cohort minimum to maximum, plus at least four Gaussian bandwidths at both ends',
           curve_height=.70,curve_spacing=.57,platform_gap=.36,peak_normalized=True)

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def draw_ridge(ax,v,meta,method,marker,cfg,audit):
    v=np.asarray(v,dtype=float)
    assert np.isfinite(v).all()
    old_low,old_high=np.quantile(v,[.005,.995]);assert old_high>old_low
    dx=(old_high-old_low)/160
    # Retain both the old bin width and its grid origin. This preserves the
    # smoothing bandwidth in expression units; only support is extended.
    left=int(np.floor((v.min()-old_low)/dx))-26
    right=int(np.ceil((v.max()-old_low)/dx))+26
    assert 0 < right-left < 1000000, 'Unexpected range; inspect without truncating'
    edges=old_low+np.arange(left,right+1,dtype=float)*dx
    low,high=edges[0],edges[-1]
    centers=(edges[:-1]+edges[1:])/2
    ds=meta.dataset.astype(str).to_numpy();units=meta.unit_key.astype(str).to_numpy()
    groups=[(p['dataset'],u,p['color']) for p in cfg['platforms'] for u in p['selected_unit_ids']]
    level,previous=0.,None
    for i,(d,u,color) in enumerate(reversed(groups)):
        if previous is not None and d!=previous:level+=.36
        vals=v[(ds==d)&(units==u)];assert len(vals)>0
        counts,_=np.histogram(vals,bins=edges)
        assert counts.sum()==len(vals), 'Every displayed cell must enter the histogram'
        density=gaussian_filter1d(counts.astype(float),sigma=6,mode='constant',truncate=4)
        if density.max()>0:density/=density.max()
        assert density[0]==0 and density[-1]==0, 'Density must return to baseline'
        x=centers;y=level+.70*density
        ax.fill_between(x,level,y,color=color,alpha=.82,lw=0,zorder=2*i+1)
        # Fully evaluated tails reach the baseline before the frame. No artificial
        # dropping of nonzero endpoint densities or quantile clipping is used.
        edge=tuple(.8*c for c in to_rgb(color))
        outline_x=np.r_[low,x,high];outline_y=np.r_[y[0],y,y[-1]]
        assert np.isfinite(outline_y).all()
        ax.plot(outline_x,outline_y,color=edge,lw=.35,alpha=.90,zorder=2*i+2)
        audit.append(dict(method=method,marker=marker,dataset=d,unit_key=u,n_cells=len(vals),
            display_low=float(low),display_high=float(high),n_below=int((vals<low).sum()),n_above=int((vals>high).sum()),
            bins=len(counts),sigma_bins=6,bin_width=float(dx),bandwidth_expression_units=float(6*dx),
            histogram_count=int(counts.sum()),endpoint_density_left=float(density[0]),endpoint_density_right=float(density[-1]),
            frozen_value_min=float(v.min()),frozen_value_max=float(v.max()),
            old_display_low=float(old_low),old_display_high=float(old_high),
            old_n_below=int((vals<old_low).sum()),old_n_above=int((vals>old_high).sum()),
            peak_height_normalized=True,ridge_color=color,fill_alpha=.82,
            density_sha256=hashlib.sha256(np.ascontiguousarray(density).tobytes()).hexdigest(),
            top_to_bottom_index=len(groups)-1-i))
        level+=.57;previous=d
    ax.set_xlim(low,high);ax.set_ylim(-.09,level-.57+.70+.04);ax.set_yticks([])
    if low<0<high:ax.axvline(0,color='#666666',lw=.35,ls=(0,(2,3)),alpha=.7,zorder=100)
    ticks=MaxNLocator(3).tick_values(low,high);ax.set_xticks(ticks[(ticks>=low)&(ticks<=high)])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x/1000:g}k' if abs(x)>=1000 else f'{x:g}'))
    ax.tick_params(axis='x',labelsize=6.8,length=1.7,pad=1.1,width=.4,colors='#263238')
    for spine in ax.spines.values():spine.set_color('#9AA1A5');spine.set_linewidth(.35)

def add_legend(fig,cfg):
    handles=[Line2D([],[],ls='',marker='o',ms=3.3,mew=0,color=p['color'],label=f"{p['name']} ({len(p['selected_unit_ids'])})") for p in cfg['platforms']]
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(100/210,1-34/297),ncol=len(handles),frameon=False,
               prop={'size':7.4,'weight':'bold'},columnspacing=1.,handletextpad=.35,borderaxespad=0)

def validate_audit(a,cfg):
    assert len(a)==18*cfg['ridge_units']
    assert a.n_below.eq(0).all() and a.n_above.eq(0).all()
    assert a.histogram_count.eq(a.n_cells).all()
    assert a.endpoint_density_left.eq(0).all() and a.endpoint_density_right.eq(0).all()
    assert a.groupby(['method','marker']).n_cells.sum().eq(cfg['ridge_cells']).all()
    assert a.groupby(['method','marker']).unit_key.nunique().eq(cfg['ridge_units']).all()
    for _,g in a.groupby(['method','marker']):
        for p in cfg['platforms']:
            rows=g[g.dataset.eq(p['dataset'])]
            assert set(rows.unit_key)==set(p['selected_unit_ids'])
            assert rows.ridge_color.eq(p['color']).all()

def manifest_extra(cfg):
    all_units=cfg['ridge_units']==cfg['full_units']
    return dict(version='ridge_full_range_20260922',revision_scope='Complete ridge density support and axis margins; original bandwidth, selection and palettes retained',
        ridge_range_quantiles=None,ridge_bins='Extended at fixed original bin width',ridge_sigma_bins=6,
        ridge_bandwidth_reference_quantiles=[.005,.995],ridge_bandwidth_reference_bins=160,
        ridge_full_distribution=True,ridge_density_padding_bins=26,
        umap_refit=False,umap_content_unchanged=True,alignment_refit=False,metric_scores_changed=False,
        original_version=cfg['original_folder'],ridge_palette_mode=cfg['palette_mode'],
        ridge_dataset_palette={p['dataset']:p['color'] for p in cfg['platforms']},
        ridge_platform_order=[p['dataset'] for p in cfg['platforms']],ridge_legend_order=[p['name'] for p in cfg['platforms']],
        ridge_display_units=cfg['ridge_units'],ridge_display_cells=cfg['ridge_cells'],ridge_full_units=cfg['full_units'],
        ridge_selected_unit_ids={p['dataset']:p['selected_unit_ids'] for p in cfg['platforms']},
        ridge_selection=cfg['selection'],ridge_style=STYLE,
        all_cells_and_sample_units_retained=all_units,all_cells_retained_for_alignment_and_ridges=all_units,
        all_sample_units_retained=all_units,all_alignment_cells_retained=True,
        ridge_all_cells_of_selected_units_retained=True,ridge_separate_legend=True,
        layout_legend_note='Separate local legend for each block; ridge legend counts are displayed sample units; UMAP palette unchanged')
