"""Shared physical-size layout, data provenance and export for Figure 2 review."""
from pathlib import Path
import os, json, hashlib
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
FIG2 = BASE.parent
PRIOR = FIG2/'redesign_gating_20260914'
os.environ.setdefault('MPLCONFIGDIR',str(BASE/'.cache'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DPI = 360
METHODS = ['Harmony-direct','cyCombine','CytoNorm','fastMNN','MaxFuse','RAMP']
VALUE_METHODS = ['Raw','Harmony-direct','cyCombine','CytoNorm','RAMP']
DISPLAY = {'Harmony-direct':'Harmony', **{m:m for m in METHODS if m!='Harmony-direct'}, 'Raw':'Raw'}
METHOD_COLORS = {'Raw':'#B4B7BA','Harmony-direct':'#858B91','cyCombine':'#9772B4',
    'CytoNorm':'#479DAD','fastMNN':'#397CA8','MaxFuse':'#D09039','RAMP':'#008866'}
CASES = ['bone_marrow','tnbc','melanoma','lung_cancer']
CASE_NAMES = dict(zip(CASES,['Bone marrow','TNBC','Melanoma','Lung cancer']))
CASE_COLORS = dict(zip(CASES,['#397CA8','#D99632','#9873AE','#469886']))
CASE_SHAPES = dict(zip(CASES,['o','s','^','v']))
INK='#202B30'
READS={}
PANELS={'A':(43,58),'B':(43,58),'C':(43,58),'D':(45,58),
        'E':(133,74),'F':(45,74)}

def setup():
    plt.rcParams.update({'font.family':'Arial','font.size':7,'axes.labelsize':7,
        'axes.titlesize':8,'axes.linewidth':.55,'xtick.labelsize':6.3,'ytick.labelsize':6.3,
        'xtick.major.width':.55,'ytick.major.width':.55,'svg.fonttype':'none',
        'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK})

def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def track(p):
    p=Path(p).resolve(); READS[str(p)]=sha(p); return p

def csv(p,**kwargs):
    kwargs.setdefault('float_precision','round_trip')
    return pd.read_csv(track(p),**kwargs)

def out(letter):
    p=BASE/f'Panel {letter}'; p.mkdir(exist_ok=True); return p

def save_data(letter,frame,name='plotted_values.csv'):
    frame.to_csv(out(letter)/name,index=False)

def mm_axes(fig,box,rect):
    x,y,w,h=box; dx,dy,aw,ah=rect
    fw,fh=fig.get_size_inches()*25.4
    return fig.add_axes([(x+dx)/fw,(fh-y-dy-ah)/fh,aw/fw,ah/fh])

def mm_text(fig,box,x,y,text,**kwargs):
    fw,fh=fig.get_size_inches()*25.4
    return fig.text((box[0]+x)/fw,1-(box[1]+y)/fh,text,**kwargs)

def title(fig,box,letter,text):
    mm_text(fig,box,0,3.7,letter,fontsize=10.5,weight='bold',va='baseline')
    mm_text(fig,box,4.8,3.7,text,fontsize=8,weight='bold',va='baseline')

def style_axis(ax,grid=True):
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_color('#748087')
    ax.tick_params(length=2,pad=2)
    if grid:
        ax.grid(axis='y',color='#E5E9EB',lw=.4); ax.set_axisbelow(True)

def boxpoints(ax,data,methods,value,means=None):
    arrays=[data[data.method.eq(m)][value].to_numpy() for m in methods]
    b=ax.boxplot(arrays,positions=np.arange(len(methods)),widths=.50,patch_artist=True,
        showfliers=False,manage_ticks=False,
        medianprops={'color':'#4E5960','lw':.8},whiskerprops={'color':'#7B858A','lw':.55},
        capprops={'color':'#7B858A','lw':.55},boxprops={'edgecolor':'#7B858A','lw':.55})
    for m,p in zip(methods,b['boxes']): p.set_facecolor('#C5E0D7' if m=='RAMP' else '#EDF0F1')
    for i,m in enumerate(methods):
        for j,c in enumerate(CASES):
            g=data[data.method.eq(m)&data.benchmark.eq(c)]
            if not len(g):continue
            center=[-.165,-.055,.055,.165][j]
            jit=np.linspace(-.034,.034,len(g)) if len(g)>1 else np.zeros(1)
            ax.scatter(i+center+jit,g[value],s=9 if len(data)>50 else 13,marker=CASE_SHAPES[c],
                c=CASE_COLORS[c],edgecolors='white',linewidths=.25,zorder=4,alpha=.88)
    if means is not None:
        ax.scatter(np.arange(len(methods)),[means[m] for m in methods],marker='D',s=15,
            color='#182C33',edgecolors='white',linewidths=.35,zorder=6)
    ax.set_xticks(np.arange(len(methods)),[DISPLAY[m] for m in methods],rotation=53,ha='right',rotation_mode='anchor')
    ax.tick_params(axis='x',length=0,pad=2)
    ax.get_xticklabels()[-1].set_color(METHOD_COLORS['RAMP']); ax.get_xticklabels()[-1].set_weight('bold')
    ax.set_xlim(-.55,len(methods)-.45)
    style_axis(ax)

def case_legend(fig,center_y_mm):
    fw,fh=fig.get_size_inches()*25.4
    handles=[Line2D([],[],ls='none',marker=CASE_SHAPES[c],color=CASE_COLORS[c],ms=3.4,label=CASE_NAMES[c]) for c in CASES]
    handles.append(Line2D([],[],ls='none',marker='D',color='#182C33',ms=3.2,label='Mean'))
    fig.legend(handles=handles,loc='center',bbox_to_anchor=(.5,1-center_y_mm/fh),frameon=False,
        ncol=5 if fw>100 else 2,columnspacing=1.1,handletextpad=.35,handlelength=.8,fontsize=6.3)

def export(fig,folder,stem,detail):
    folder=Path(folder); folder.mkdir(exist_ok=True)
    fig.canvas.draw(); renderer=fig.canvas.get_renderer()
    # Validate every visible text remains on the canvas; visual inspection still
    # checks inter-panel collisions and density.
    for t in fig.findobj(match=lambda a: isinstance(a,plt.Text)):
        if not t.get_visible() or not t.get_text(): continue
        b=t.get_window_extent(renderer)
        if b.width==0:continue
        assert b.x0>=-2 and b.y0>=-2 and b.x1<=fig.bbox.width+2 and b.y1<=fig.bbox.height+2,(t.get_text(),str(b))
    for ext in ['png','svg']: fig.savefig(folder/f'{stem}.{ext}',dpi=DPI)
    detail.update(width_mm=float(fig.get_figwidth()*25.4),height_mm=float(fig.get_figheight()*25.4),
        input_sha256=READS.copy(),inputs_unchanged=all(sha(p)==h for p,h in READS.items()),dpi=DPI)
    assert detail['inputs_unchanged']
    (folder/'manifest.json').write_text(json.dumps(detail,indent=2,ensure_ascii=False)+'\n')
    plt.close(fig)
    print(folder/f'{stem}.png',flush=True)

def run_panel(letter,draw):
    setup(); w,h=PANELS[letter]
    legend=letter in 'ABF'
    fig=plt.figure(figsize=(w/25.4,(h+(15 if legend else 0))/25.4),facecolor='white')
    detail=draw(fig,(0,0,w,h))
    if legend:case_legend(fig,h+7)
    export(fig,out(letter),f'panel_{letter.lower()}',{'panel':letter,**detail})

def typical_data(metric):
    f=csv(ROOT/'benchmark_results/comparison_evaluation/summary/primary_method_family_metric_summary.csv')
    d=f[f.metric.eq(metric)].rename(columns={'method_family':'method'})
    assert len(d)==24 and d.n_resamples.eq(20).all()
    return d

def draw_typical(fig,box,letter,metric,heading,ylab,limits,ticks):
    title(fig,box,letter,heading)
    d=typical_data(metric); save_data(letter,d)
    means=d.groupby('method')['mean'].mean()
    ax=mm_axes(fig,box,(9.5,9,box[2]-10.5,34))
    boxpoints(ax,d,METHODS,'mean',means)
    ax.set_ylabel(ylab,labelpad=3); ax.set_ylim(*limits); ax.set_yticks(ticks)
    return {'metric':metric,'n_benchmarks':4,'n_resamples_per_point':20,'box_n':4,
        'mean':'Equal weight per benchmark','selection':'Previously selected Figure 1 cell-type metrics, reused without recomputing.'}
