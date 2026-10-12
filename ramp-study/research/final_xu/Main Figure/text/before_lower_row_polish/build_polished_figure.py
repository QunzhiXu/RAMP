"""Rebuild Figure 1 at its manuscript size; write only beside this script."""
from pathlib import Path
import os
import sys
import json
import hashlib

HERE = Path(__file__).resolve().parent
MAIN = HERE.parent
ROOT = MAIN.parent
os.environ['MPLCONFIGDIR'] = str(HERE / '.cache' / 'matplotlib')
os.environ['MPLBACKEND'] = 'Agg'
sys.dont_write_bytecode = True
sys.path.insert(0, str(MAIN / 'Panel B' / 'benchmark_review'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FuncFormatter
from scipy.ndimage import gaussian_filter1d
from scipy.stats import rankdata, mannwhitneyu
from build_ridge_review import load_case, load_matrices, tick

W, H = 180., 188.
DPI = 600
INK = '#202529'
LINE = '#586165'
GRID = '#E2E6E8'
GREEN = '#087F5B'
BLUE = '#297AA4'
ORANGE = '#D36C58'
DS_COLORS = ['#2775B6', '#E58A2B', '#38A169']
PALETTE = {
    'Tumor / epithelial': '#D56D79', 'Myeloid': '#DDB130', 'Stroma': '#9B76B5',
    'Endothelial': '#079EAA', 'B lineage': '#32906C', 'NK': '#6154A3',
    'CD4 T': '#2878B5', 'CD8 T': '#E46A2E', 'Non-joint': '#BFC3C8',
}
ORDER = ['Myeloid', 'Tumor / epithelial', 'Non-joint', 'Endothelial',
         'B lineage', 'NK', 'CD4 T', 'CD8 T', 'Stroma']
B_METHODS = ['Unaligned', 'Harmony-direct_source_scale', 'cyCombine', 'RAMP']
METHODS = ['Harmony-direct', 'cyCombine', 'CytoNorm', 'fastMNN', 'MaxFuse', 'RAMP']
CASES = ['bone_marrow', 'tnbc', 'melanoma', 'lung_cancer']
BENCH_COLORS = ['#397CA8', '#E2A13D', '#9772B4', '#429984']
READS = {}
AUDIT = []
TEXT = []


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def track(p):
    p = Path(p).resolve()
    READS[str(p)] = sha(p)
    return p


def text(fig, x, y, label, size=6.2, bold=False, ha='left', va='center', color=INK, **kw):
    artist = fig.text(x/W, 1-y/H, label, fontsize=size, weight='bold' if bold else 'normal',
                      ha=ha, va=va, color=color, **kw)
    TEXT.append(artist)
    return artist


def line(fig, x1, y1, x2, y2, color=LINE, lw=.5, **kw):
    fig.add_artist(Line2D([x1/W, x2/W], [1-y1/H, 1-y2/H], transform=fig.transFigure,
                          color=color, linewidth=lw, **kw))


def arrow(fig, x1, y1, x2, y2, lw=.7, color=INK, **kw):
    fig.add_artist(FancyArrowPatch((x1/W, 1-y1/H), (x2/W, 1-y2/H),
                                  transform=fig.transFigure, arrowstyle='-|>',
                                  mutation_scale=6, linewidth=lw, color=color, **kw))


def axes(fig, x, y, w, h):
    return fig.add_axes([x/W, 1-(y+h)/H, w/W, h/H])


def panel_title(fig, letter, title, x, y):
    text(fig, x, y, letter, size=10.5, bold=True)
    text(fig, x+4.6, y, title, size=8.0, bold=True)


def frame(ax, grid=False):
    for spine in ax.spines.values():
        spine.set_linewidth(.45)
        spine.set_color(LINE)
    ax.tick_params(labelsize=5.8, width=.45, length=1.7, pad=1.6, colors=INK)
    ax.set_axisbelow(True)
    if grid:
        ax.grid(axis='y', linewidth=.35, color=GRID)


def mini_legend(fig, x, y, labels, colors, spacing, size=5.7, marker='o'):
    for i, (label, color) in enumerate(zip(labels, colors)):
        px = x+i*spacing
        fig.add_artist(Line2D([px/W], [1-y/H], marker=marker, linestyle='none',
                              markersize=2.6, markeredgewidth=0, color=color,
                              transform=fig.transFigure))
        text(fig, px+1.5, y, label, size=size)


def build_a(fig):
    """Schematic only: expose matching evidence without changing the panel footprint."""
    panel_title(fig, 'A', 'RAMP overview', 2, 4.2)
    text(fig, 2, 12.0, '1', size=8, bold=True, color=GREEN)
    text(fig, 6, 12.0, 'Sample-wise preprocessing', size=7.2, bold=True)
    text(fig, 18, 17, 'Source', size=6.2, ha='center')
    text(fig, 43.5, 17, 'Low-population anchor', size=6.0, ha='center')
    # Schematic distributions, deliberately separate from benchmark measurements.
    x = np.linspace(0, 1, 240)
    for j, color in enumerate(DS_COLORS):
        y = 20+j*5.1
        for col, left in enumerate([9., 34.5]):
            ax = axes(fig, left, y, 18, 4.8)
            center = [.2, .43, .62][j] if col == 0 else .27
            high = [.62, .78, .9][j] if col == 0 else [.65, .8, .92][j]
            density = np.exp(-.5*((x-center)/.065)**2)+.28*np.exp(-.5*((x-high)/.1)**2)
            ax.fill_between(x, 0, density, color=color, alpha=.85, linewidth=0)
            ax.plot(x, density, color=color, linewidth=.7)
            ax.axvline(center, color=INK, lw=.45, ls=(0, (2, 2)))
            ax.set_xlim(0, 1); ax.set_ylim(0, 1.15); ax.axis('off')
        text(fig, 7.5, y+3.2, f'S{j+1}', size=5.6, ha='right')
    arrow(fig, 28, 28, 33, 28)
    text(fig, 28, 37.8, 'Adaptive arcsinh + 1D k-means', size=6, ha='center')
    text(fig, 28, 41.2, 'Center the low population; scale each marker', size=5.7, ha='center')
    line(fig, 2, 44.2, 54, 44.2, color='#CBD1D4', lw=.4)

    text(fig, 2, 48, '2', size=8, bold=True, color=GREEN)
    text(fig, 6, 48, 'Label-free matching', size=7.2, bold=True)
    text(fig, 28, 52, 'Shared proteins · no cell-type labels', size=5.8, ha='center')
    for px, label in [(3, 'Source scale'), (31, 'Preprocessed')]:
        fig.add_artist(Rectangle((px/W, 1-61/H), 23/W, 6/H,
                                 facecolor='white', edgecolor=LINE, linewidth=.45,
                                 transform=fig.transFigure))
        text(fig, px+11.5, 58, label, size=6.1, ha='center')
    text(fig, 28.5, 58, '+', size=7.5, ha='center')
    arrow(fig, 14.5, 61.5, 21, 64.4)
    arrow(fig, 42.5, 61.5, 36, 64.4)
    text(fig, 28, 66, 'Dual-view signatures', size=6.2, bold=True, ha='center')
    # Two distinct kinds of agreement: reciprocal links and three-dataset cycles.
    text(fig, 14, 70.5, 'Bidirectional support', size=5.8, ha='center')
    text(fig, 42.5, 70.5, 'Cycle consistency', size=5.8, ha='center')
    left = axes(fig, 3.5, 72.3, 22, 10)
    right = axes(fig, 31.5, 72.3, 22, 10)
    for ax in [left, right]:
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    def node(ax, xy, color, size=20):
        ax.scatter(*xy, s=size, c=color, edgecolors='white', linewidths=.45, zorder=5)
    def link(ax, a, b, lw=.65, color=INK, bend=0):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=5,
                                    linewidth=lw,color=color,shrinkA=3,shrinkB=3,
                                    connectionstyle=f'arc3,rad={bend}'))
    for y in [.18,.5,.82]:
        node(left,(.12,y),DS_COLORS[0]); node(left,(.88,y),DS_COLORS[1])
        link(left,(.12,y),(.88,y),lw=.7,bend=.12)
        link(left,(.88,y),(.12,y),lw=.7,bend=.12)
    link(left,(.12,.18),(.88,.5),lw=.35,color='#A7B0B5')
    link(left,(.88,.82),(.12,.5),lw=.35,color='#A7B0B5')
    triangle=[(.50,.88),(.88,.17),(.12,.17)]
    for j,xy in enumerate(triangle):
        node(right,xy,DS_COLORS[j],size=24)
        link(right,xy,triangle[(j+1)%3],lw=.85,bend=.14)
    # Both sources of evidence feed reference weights and marker-specific targets.
    arrow(fig,14,83,23,86)
    arrow(fig,42.5,83,34,86)
    text(fig,28,87.6,'Consensus-weighted references',size=6.0,bold=True,ha='center')
    ax=axes(fig,7,90,44,6)
    ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
    for j,(yy,weight) in enumerate(zip([.18,.5,.82],[.45,.8,1.15])):
        node(ax,(.05,yy),DS_COLORS[j],size=13)
        link(ax,(.05,yy),(.52,.5),lw=weight)
    ax.scatter([.52],[.5],marker='D',s=23,c=GREEN,edgecolors='white',linewidths=.45,zorder=6)
    text(fig,33,93,'Marker targets',size=5.8)
    line(fig, 2, 98, 54, 98, color='#CBD1D4', lw=.4)

    text(fig, 2, 102, '3', size=8, bold=True, color=GREEN)
    text(fig, 6, 102, 'Monotone transformation', size=7.2, bold=True)
    text(fig,16,106.5,'Fit each sample × marker',size=5.6,ha='center')
    text(fig,44,106.5,'Retain cell order',size=5.8,ha='center')
    ax=axes(fig,6,109,21,10)
    xx=np.linspace(0,1,120)
    for color,yy in zip(DS_COLORS,[xx**.65,xx,xx**1.5]):
        assert np.all(np.diff(yy)>=0)
        ax.plot(xx,yy,color=color,lw=.9)
    ax.set_xlim(-.04,1.04);ax.set_ylim(-.04,1.04)
    ax.set_xticks([]);ax.set_yticks([])
    for side in ['top','right']:ax.spines[side].set_visible(False)
    for side in ['bottom','left']:
        ax.spines[side].set_linewidth(.45);ax.spines[side].set_color(LINE)
    text(fig,3.6,114,'Output',size=5.5,rotation=90,ha='center')
    text(fig,16.5,121,'Input',size=5.5,ha='center')
    before=np.array([40,43,49,53]);after=np.array([40,45,48,53])
    assert np.all(np.diff(before)>0) and np.all(np.diff(after)>0)
    text(fig,31,110,'Before',size=5.5)
    text(fig,31,118,'After',size=5.5)
    for a,b in zip(before,after):
        line(fig,a,110,b,118,color='#ADB7BB',lw=.45,ls=(0,(2,2)))
    for xs,y in [(before,110),(after,118)]:
        line(fig,39.5,y,54,y,color=LINE,lw=.4)
        for px,m in zip(xs,['o','^','s','D']):
            fig.add_artist(Line2D([px/W],[1-y/H],marker=m,linestyle='none',
                                  markersize=3,color=DS_COLORS[0],markeredgecolor=INK,
                                  markeredgewidth=.4,transform=fig.transFigure))
    text(fig,43,121,'Within each sample',size=5.5,ha='center')


def ridge(ax, values, meta, method, marker):
    low, high = np.quantile(values, [.005,.995])
    if high<=low: high=low+1
    datasets=meta.dataset.to_numpy(); units=meta.unit_key.to_numpy()
    palette=dict(zip(meta.dataset.unique(), DS_COLORS))
    groups=[(d,u) for d in palette for u in sorted(set(units[datasets==d]))]
    level=0.; prev=None
    for i,(d,u) in enumerate(reversed(groups)):
        if prev is not None and prev!=d: level+=.36
        vals=values[(datasets==d)&(units==u)]
        hist,edges=np.histogram(vals,bins=160,range=(low,high))
        smooth=gaussian_filter1d(hist.astype(float),sigma=6,mode='constant')
        if smooth.max()>0: smooth/=smooth.max()
        xx=(edges[1:]+edges[:-1])/2; yy=level+.70*smooth
        ax.fill_between(xx,level,yy,color=palette[d],alpha=.82,linewidth=0,zorder=2*i+1)
        ax.plot(xx,yy,color='#46505A',lw=.35,zorder=2*i+2)
        AUDIT.append(dict(benchmark='tnbc',method=method,marker=marker,
                          smoothing_sigma_bins=6,dataset=d,unit_key=u,n_cells=len(vals),
                          display_low=float(low),display_high=float(high),
                          n_below_display=int((vals<low).sum()),n_above_display=int((vals>high).sum())))
        level+=.57; prev=d
    ax.set_xlim(low,high); ax.set_ylim(-.09,level-.57+.70+.04)
    if low<0<high: ax.axvline(0,color='#666666',lw=.35,ls=(0,(1.5,2.2)),alpha=.7,zorder=100)
    ax.set_yticks([]); ax.xaxis.set_major_locator(MaxNLocator(3)); ax.xaxis.set_major_formatter(FuncFormatter(tick))
    frame(ax); ax.tick_params(axis='x',labelsize=5.5,length=1.3,pad=1.0)


def scatter(fig, meta, xy, color_by, x, y, w, h):
    ax=axes(fig,x,y,w,h)
    if color_by=='Dataset':
        palette=dict(zip(meta.dataset.unique(),DS_COLORS))
        colors=np.array([palette[d] for d in meta.dataset])
        order=np.random.default_rng(42).permutation(len(meta))
        ax.scatter(xy[order,0],xy[order,1],c=colors[order],s=.058,alpha=.72,
                   edgecolors='none',rasterized=True)
    else:
        for z,label in enumerate(ORDER):
            idx=np.flatnonzero(meta.celltype.to_numpy()==label)
            idx=np.random.default_rng(42).permutation(idx)
            size,alpha=(.058,.5) if label=='Tumor / epithelial' else (.070,.85)
            if label=='Non-joint':size,alpha=.058,.65
            ax.scatter(xy[idx,0],xy[idx,1],c=PALETTE[label],s=size,alpha=alpha,
                       edgecolors='none',rasterized=True,zorder=z+1)
    lo,hi=xy.min(axis=0),xy.max(axis=0); margin=(hi-lo)*.025
    ax.set_xlim(lo[0]-margin[0],hi[0]+margin[0]);ax.set_ylim(lo[1]-margin[1],hi[1]+margin[1])
    ax.set_aspect('equal',adjustable='box');ax.axis('off')
    fig.add_artist(Rectangle((x/W,1-(y+h)/H),w/W,h/H,fill=False,edgecolor=LINE,
                             linewidth=.45,transform=fig.transFigure,zorder=100))


def build_b(fig,meta,coords,markers,matrices):
    panel_title(fig,'B','Cross-platform alignment in TNBC',59,4.2)
    mini_legend(fig,93,9,['IMC1','IMC2','CODEX'],DS_COLORS,16,size=6)
    xs=[65,82.7,100.4,119.1,149.3]; widths=[16,16,16,28.6,28.6]
    for x,w,title in zip(xs,widths,['CD3','CD8','PANCK','Dataset','Cell type']):
        text(fig,x+w/2,13.2,title,size=6.7,bold=True,ha='center')
    for row,method in enumerate(B_METHODS):
        y=16+row*(21.7+3.3)
        label={'Unaligned':'Raw (arcsinh)','Harmony-direct_source_scale':'Harmony-direct',
               'cyCombine':'cyCombine','RAMP':'RAMP'}[method]
        text(fig,61.9,y+10.85,label,size=6.3,bold=method=='RAMP',ha='center',rotation=90,
             color=GREEN if method=='RAMP' else INK)
        for j,marker in enumerate(['CD3','CD8','PANCK']):
            val=matrices[method][:,markers.index(marker)]
            display=method
            if method=='Unaligned':val=np.arcsinh(val);display='Unaligned_arcsinh_display'
            ridge(axes(fig,xs[j],y,widths[j],21.7),val,meta,display,marker)
        for j,kind in enumerate(['Dataset','Cell type']):
            scatter(fig,meta,coords[method],kind,xs[3+j],y,widths[3+j],21.7)
    # Two compact legend rows; all existing labels are retained.
    labels=['Tumor / epithelial','Stroma','CD4 T','B lineage','Non-joint',
            'Myeloid','CD8 T','Endothelial','NK']
    positions=[(66,117),(100,117),(120,117),(139,117),(159,117),
               (66,120.5),(100,120.5),(120,120.5),(149,120.5)]
    for label,(x,y) in zip(labels,positions):
        mini_legend(fig,x,y,[label],[PALETTE[label]],0,size=5.7)


def statistical_axes(fig,x,y,w,h):
    ax=axes(fig,x,y,w,h);frame(ax,grid=True)
    ax.tick_params(axis='x',length=0,pad=2)
    return ax


def build_c(fig):
    panel_title(fig,'C','Cell-type preservation',2,128)
    data=pd.read_csv(track(MAIN/'Panel C/benchmark_metric_values.csv'))
    assert len(data)==48 and data.n_resamples.eq(20).all()
    mini_legend(fig,8,133,['Bone marrow','TNBC','Melanoma','Lung'],BENCH_COLORS,16,size=5.4)
    mini_legend(fig,60,136.5,['Mean'],[INK],0,size=5.4,marker='D')
    for row,(metric,title) in enumerate([('leiden_celltype_ari','Cell-type ARI'),
                                        ('umap_celltype_asw_macro','UMAP cell-type silhouette')]):
        y=[140,166][row]; h=16.5
        text(fig,8,y-2.6,title,size=6.3,bold=True)
        ax=statistical_axes(fig,8,y,62.5,h)
        v=data[data.metric.eq(metric)].pivot(index='method_family',columns='benchmark',values='mean').loc[METHODS,CASES].to_numpy()
        b=ax.boxplot(list(v),positions=np.arange(6),widths=.5,patch_artist=True,showfliers=False,
                     medianprops={'color':INK,'linewidth':.6},whiskerprops={'color':LINE,'linewidth':.45},
                     capprops={'color':LINE,'linewidth':.45},boxprops={'edgecolor':LINE,'linewidth':.45})
        for i,p in enumerate(b['boxes']):p.set_facecolor('#D0E5DC' if i==5 else '#ECF0F2')
        for j,color in enumerate(BENCH_COLORS):
            ax.scatter(np.arange(6)+[-.14,-.045,.045,.14][j],v[:,j],s=7.5,c=color,
                       edgecolors='white',linewidths=.25,zorder=4)
        ax.scatter(np.arange(6),v.mean(axis=1),marker='D',s=8,c=INK,edgecolors='white',linewidths=.25,zorder=5)
        assert np.argmax(v.mean(axis=1))==5
        ax.set_xlim(-.5,5.5);ax.set_xticks(np.arange(6),['Harmony-\ndirect',*METHODS[1:]])
        ax.tick_params(axis='x',labelsize=5.5)
        ax.get_xticklabels()[-1].set_color(GREEN);ax.get_xticklabels()[-1].set_weight('bold')
        if row==0:ax.set_ylim(0,.70);ax.set_yticks([0,.2,.4,.6])
        else:ax.set_ylim(-.105,.255);ax.set_yticks([-.1,0,.1,.2])


def load_artifact(case,unit,marker):
    base=ROOT/'benchmark_results'/case
    meta=pd.read_csv(track(base/'ramp/alignment/input/cell_metadata.csv.gz'),low_memory=False)
    markers=track(base/'ramp/alignment/input/markers.txt').read_text().splitlines()
    idx=meta.unit_key.eq(unit).to_numpy(); ids=meta.cell_id.astype(str)
    outputs={}
    for label,rel in [('Raw','ramp/alignment/input/raw.npy'),
                      ('Harmony-direct','harmony/source_scale/alignment/data/harmony_adjusted_marker_coordinates.csv.gz'),
                      ('cyCombine','cycombine/alignment/data/corrected_expression.csv.gz'),
                      ('CytoNorm','cytonorm/alignment/data/corrected_expression.csv.gz'),
                      ('RAMP','ramp/alignment/data/ramp_aligned.npy')]:
        p=track(base/rel)
        if p.suffix=='.npy':v=np.load(p,mmap_mode='r')[:,markers.index(marker)]
        else:
            table=pd.read_csv(p,usecols=['cell_id',marker],dtype={'cell_id':str})
            assert table.cell_id.is_unique and set(table.cell_id)==set(ids)
            v=table.set_index('cell_id').loc[ids,marker].to_numpy()
        outputs[label]=np.asarray(v[idx]);assert np.isfinite(outputs[label]).all()
    return meta.loc[idx].reset_index(drop=True),outputs


def build_d(fig):
    panel_title(fig,'D','Artifacts in comparison methods',75,128)
    examples=[('tnbc','Meyer IMC|B12.64192','CD8','CD8 T','CD4 T','CD4 T',
               'CD8 · Meyer B12.64192',['Harmony-direct','cyCombine']),
              ('bone_marrow','BM CyTOF|H1','CD4','CD4 T','Mature / CODEX B','B cells',
               'CD4 · CyTOF H1',['CytoNorm'])]
    records=[]; stats=[]
    for row,(case,unit,marker,target,other,other_label,title,drivers) in enumerate(examples):
        y=[140,166][row]
        text(fig,81,y-2.6,title,size=6.2,bold=True)
        mini_legend(fig,116,y-2.6,[target,other_label],[BLUE,ORANGE],14,size=5.5)
        ax=statistical_axes(fig,81,y,62,16.5)
        meta,outputs=load_artifact(case,unit,marker)
        pos=meta.celltype.eq(target).to_numpy(); neg=meta.celltype.eq(other).to_numpy()
        for method in drivers:
            i=list(outputs).index(method);ax.axvspan(i-.45,i+.45,color='#FBECE6',zorder=0)
        for i,(method,v) in enumerate(outputs.items()):
            ranks=rankdata(v,method='average')/(len(v)+1)
            arrays=[ranks[pos],ranks[neg]]
            boxes=ax.boxplot(arrays,positions=[i-.17,i+.17],widths=.26,patch_artist=True,
                             showfliers=False,manage_ticks=False,
                             medianprops={'color':INK,'linewidth':.6},
                             whiskerprops={'color':LINE,'linewidth':.45},
                             capprops={'color':LINE,'linewidth':.45},
                             boxprops={'edgecolor':LINE,'linewidth':.45})
            for p,c in zip(boxes['boxes'],[BLUE,ORANGE]):p.set_facecolor(c)
            auc=float(mannwhitneyu(v[pos],v[neg],alternative='greater').statistic/(pos.sum()*neg.sum()))
            records.append(dict(case=case,scope=unit,marker=marker,method=method,target_vs_other_auc=auc))
            for label,a in zip([target,other],arrays):
                q1,median,q3=np.quantile(a,[.25,.5,.75])
                stats.append(dict(case=case,marker=marker,method=method,celltype=label,n_cells=len(a),
                                  mean_percentile=float(a.mean()),q1=q1,median=median,q3=q3))
        ax.set_xlim(-.5,4.5);ax.set_ylim(0,1);ax.set_yticks([0,.5,1])
        ax.set_ylabel(f'{marker} percentile',fontsize=5.8,labelpad=2)
        ax.set_xticks(range(5),['Raw','Harmony-\ndirect','cyCombine','CytoNorm','RAMP'])
        ax.tick_params(axis='x',labelsize=5.5)
        for label,method in zip(ax.get_xticklabels(),outputs):
            if method in drivers:label.set_color('#A64736')
            if method=='RAMP':label.set_color(GREEN);label.set_weight('bold')
    old=pd.read_csv(track(MAIN/'Panel D/artifact_auc.csv'))
    pd.testing.assert_frame_equal(pd.DataFrame(records),old,check_exact=False,rtol=1e-12,atol=1e-12)
    oldstats=pd.read_csv(track(MAIN/'Panel D/artifact_box_statistics.csv'))
    fields=list(stats[0]);pd.testing.assert_frame_equal(pd.DataFrame(stats),oldstats[fields],check_exact=False,rtol=1e-12,atol=1e-12)
    pd.DataFrame(records).to_csv(HERE/'artifact_auc_verified.csv',index=False)


def build_e(fig):
    panel_title(fig,'E','Rank preservation',147,128)
    text(fig,164,133,'All 4 benchmarks',size=6,ha='center')
    df=pd.read_csv(track(MAIN/'Panel E/rank_integrity_summary.csv'))
    ax=axes(fig,162.5,137.5,15.5,41.5)
    y=np.arange(5); values=df.percent.to_numpy()
    ax.barh(y,100,height=.52,color='#ECF0F2')
    ax.barh(y,values,height=.52,color='#C9765F')
    labels=['Harmony-\ndirect','Harmony-\narcsinh','cyCombine','CytoNorm','RAMP']
    ax.set_yticks(y,labels);ax.tick_params(axis='y',length=0,labelsize=5.6,pad=4)
    ax.get_yticklabels()[-1].set_color(GREEN);ax.get_yticklabels()[-1].set_weight('bold')
    ax.set_ylim(4.8,-.35);ax.set_xlim(0,100);ax.set_xticks([0,50,100])
    ax.tick_params(axis='x',labelsize=5.5,length=1.5,pad=1.5,width=.45)
    for side in ['top','right','left']:ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(LINE);ax.spines['bottom'].set_linewidth(.45)
    for i,r in enumerate(df.itertuples()):
        # Keep both affected count and denominator, with a separate annotation line.
        if r.percent>=90:
            ax.text(96,i,f'{r.percent:.0f}%',ha='right',va='center',fontsize=5.7,color='white',weight='bold')
        elif r.percent>0:
            ax.text(r.percent+3,i,f'{r.percent:.1f}%',ha='left',va='center',fontsize=5.5,color=INK)
        else:
            ax.scatter([0],[i],s=10,color=GREEN,zorder=5,clip_on=False)
            ax.text(6,i,'0.0%',va='center',fontsize=5.7,color=GREEN,weight='bold')
        ax.text(1,i+.42,f'{r.n_with_inversion:,}/{r.n_tested:,}',fontsize=5.4,va='center',color=INK)
    text(fig,164,184,'Sample × marker\norder inversion (%)',size=5.7,ha='center')


def main():
    HERE.mkdir(parents=True,exist_ok=True)
    for filename in ['Arial.ttf','Arial Bold.ttf']:
        font_manager.fontManager.addfont('/System/Library/Fonts/Supplemental/'+filename)
    plt.rcParams.update({'font.family':'Arial','font.size':6.2,'text.color':INK,'axes.labelcolor':INK,
                         'svg.fonttype':'none','pdf.fonttype':42,'ps.fonttype':42,
                         'axes.linewidth':.45,'savefig.facecolor':'white'})
    original=track(MAIN/'Combined ABCDE/main_figure_ABCDE.png')
    track(MAIN/'Panel A/readability_review/panel_a_ramp_method_overview_readable.pdf')
    track(ROOT/'Algorithm/ramp/ramp_ver2.py')
    meta,coords=load_case('tnbc',READS)
    markers,matrices=load_matrices('tnbc',meta,READS)
    assert len(meta)==62701 and meta.unit_key.nunique()==11
    fig=plt.figure(figsize=(W/25.4,H/25.4),facecolor='white',dpi=160)
    build_a(fig);build_b(fig,meta,coords,markers,matrices);build_c(fig);build_d(fig);build_e(fig)
    # Retain the exact density estimates, units and displayed tails from the accepted figure.
    old=pd.read_csv(track(MAIN/'Panel B/tnbc_bandwidth_review/adtnorm_style_review/ridge_display_audit.csv'))
    pd.testing.assert_frame_equal(pd.DataFrame(AUDIT),old,check_exact=False,rtol=1e-12,atol=1e-12)
    pd.DataFrame(AUDIT).to_csv(HERE/'ridge_display_audit.csv',index=False)
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    boxes=[]
    for artist in TEXT:
        bb=artist.get_window_extent(renderer)
        assert bb.x0>=-1 and bb.y0>=-1 and bb.x1<=fig.bbox.width+1 and bb.y1<=fig.bbox.height+1,(artist.get_text(),bb)
        boxes.append({'text':artist.get_text(),'font_pt':artist.get_fontsize(),
                      'bbox_mm':[v/fig.dpi*25.4 for v in [bb.x0,fig.bbox.height-bb.y1,bb.x1,fig.bbox.height-bb.y0]]})
    # Include statistical tick labels in layout QA, where cross-panel collisions are easy to miss.
    all_text=list(TEXT)
    for ax in fig.axes:
        xmin,xmax=sorted(ax.get_xlim());ymin,ymax=sorted(ax.get_ylim())
        drawn_ticks=[t for loc,t in zip(ax.get_xticks(),ax.get_xticklabels()) if xmin<=loc<=xmax]
        drawn_ticks += [t for loc,t in zip(ax.get_yticks(),ax.get_yticklabels()) if ymin<=loc<=ymax]
        all_text.extend(t for t in [*drawn_ticks,ax.xaxis.label,ax.yaxis.label]
                        if t.get_visible() and t.get_text() and ax.axison)
        all_text.extend(t for t in ax.texts if t.get_visible() and t.get_text())
    collisions=[]
    for i,left in enumerate(all_text):
        a=left.get_window_extent(renderer)
        for right in all_text[i+1:]:
            b=right.get_window_extent(renderer)
            dx=min(a.x1,b.x1)-max(a.x0,b.x0);dy=min(a.y1,b.y1)-max(a.y0,b.y0)
            if dx>0 and dy>0 and dx*dy*(25.4/fig.dpi)**2>.08:
                collisions.append({'left':left.get_text(),'right':right.get_text(),
                                   'overlap_mm2':round(dx*dy*(25.4/fig.dpi)**2,3)})
    (HERE/'qa').mkdir(exist_ok=True)
    (HERE/'qa/text_collisions.json').write_text(json.dumps(collisions,indent=2)+'\n')
    assert not collisions,collisions
    fig.savefig(HERE/'main_figure_1_polished.pdf',dpi=DPI)
    fig.savefig(HERE/'main_figure_1_polished.png',dpi=420)
    fig.savefig(HERE/'main_figure_1_polished.svg',dpi=DPI)
    plt.close(fig)
    assert all(sha(p)==s for p,s in READS.items())
    manifest={'canvas_mm':[W,H],'intended_manuscript_width_mm':W,'raster_scatter_dpi_in_pdf':DPI,
              'font':'Arial','axis_tick_font_pt':[5.5,5.8],'panel_title_font_pt':8,
              'panel_letter_font_pt':10.5,'axes_frame_linewidth_pt':.45,
              'layout':'A+B above C+D+E; A narrowed; B widened; lower plots increased in physical height',
              'workflow':'Redrawn schematic; same three algorithm stages; sample-wise anchoring and rank scope stated explicitly',
              'benchmark_cells':62701,'benchmark_units':11,'all_cells_retained':True,
              'umap_refit':False,'coordinate_values_changed':False,'equal_aspect':True,
              'celltype_palette':PALETTE,'celltype_draw_order':ORDER,'celltype_alpha_unchanged':True,
              'marker_values_changed':False,'ridge_density_audit_matches_original':True,
              'artifact_auc_and_quartiles_match_original':True,'rank_counts_unchanged':True,
              'input_sha256':READS,'inputs_unchanged':True,'original_figure':str(original),
              'text_bounding_boxes':boxes,'text_collisions':collisions,'script_sha256':sha(__file__),
              'outputs':['main_figure_1_polished.pdf','main_figure_1_polished.png','main_figure_1_polished.svg']}
    (HERE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(HERE/'main_figure_1_polished.png',flush=True)
    print(f'Verified {len(READS)} read-only inputs, {len(AUDIT)} density curves, 20 artifact box summaries and 10 AUC values.',flush=True)


if __name__=='__main__':
    main()
