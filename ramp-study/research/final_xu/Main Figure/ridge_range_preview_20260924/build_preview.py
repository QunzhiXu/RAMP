"""Display-only Figure 1 ridge-range preview. No manuscript writes, fitting, or filtering."""
from pathlib import Path
import os,sys,json,hashlib
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
os.environ['MPLCONFIGDIR']=str(HERE/'tmp/matplotlib');os.environ['MPLBACKEND']='Agg'
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator,FuncFormatter
from scipy.ndimage import gaussian_filter1d
import fitz

BASE=ROOT/'benchmark_results/melanoma';OLD=HERE.parent/'ridge_full_range_20260922'
METHODS=['Unaligned','Harmony-direct_source_scale','cyCombine','RAMP']
DISPLAY=['Raw (arcsinh)','Harmony','cyCombine','RAMP']
MARKERS=['CD3','CD8','SOX10'];COLORS=['#2775B6','#E58A2B','#38A169']
INK='#202529';LINE='#586165';CUT=.005;MARGIN=.04
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plt.rcParams.update({'font.family':'Arial','font.size':8,'text.color':INK,'axes.labelcolor':INK,
 'xtick.color':INK,'ytick.color':INK,'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})

def load():
    p=BASE/'ramp/alignment/input/cell_metadata.csv.gz';meta=pd.read_csv(p,low_memory=False,dtype={'cell_id':str})
    mp=BASE/'ramp/alignment/input/markers.txt';markers=mp.read_text().splitlines()
    paths={'Unaligned':BASE/'ramp/alignment/input/raw.npy',
      'Harmony-direct_source_scale':BASE/'harmony/source_scale/alignment/data/harmony_adjusted_marker_coordinates.csv.gz',
      'cyCombine':BASE/'cycombine/alignment/data/corrected_expression.csv.gz',
      'RAMP':BASE/'ramp/alignment/data/ramp_aligned.npy'}
    sources={str(p):sha(p),str(mp):sha(mp)};matrices={}
    for method,path in paths.items():
        sources[str(path)]=sha(path)
        if path.suffix=='.npy':z=np.load(path)
        else:
            df=pd.read_csv(path,low_memory=False,dtype={'cell_id':str});assert df.cell_id.is_unique
            df=df.set_index('cell_id').loc[meta.cell_id];z=df[markers].to_numpy()
        assert z.shape==(60000,len(markers));matrices[method]=z
    assert meta.unit_key.nunique()==46
    return meta,markers,matrices,sources

def prepare(meta,markers,matrices):
    oldaudit=pd.read_csv(OLD/'ridge_display_audit.csv')
    unit=meta.unit_key.to_numpy();ds=meta.dataset.to_numpy();datasets=list(meta.dataset.unique())
    palette=dict(zip(datasets,COLORS));groups=[(d,u) for d in datasets for u in sorted(set(unit[ds==d]))]
    prepared={};summary=[];audit=[]
    for method in METHODS:
      for marker in MARKERS:
        original=matrices[method][:,markers.index(marker)]
        values=np.asarray(np.arcsinh(original) if method=='Unaligned' else original,dtype=float)
        qlo,qhi=np.quantile(values,[.005,.995]);dx=(qhi-qlo)/160
        a=int(np.floor((values.min()-qlo)/dx))-26;b=int(np.ceil((values.max()-qlo)/dx))+26
        edges=qlo+np.arange(a,b+1,dtype=float)*dx;centers=(edges[:-1]+edges[1:])/2
        curves=[];level=0;prev=None
        for d,u in reversed(groups):
            if prev is not None and prev!=d:level+=.36
            vals=values[(ds==d)&(unit==u)];hist,_=np.histogram(vals,bins=edges)
            assert hist.sum()==len(vals)
            smooth=gaussian_filter1d(hist.astype(float),sigma=6,mode='constant',truncate=4);smooth/=smooth.max()
            oldname='Unaligned_arcsinh_display' if method=='Unaligned' else method
            row=oldaudit[(oldaudit.method==oldname)&(oldaudit.marker==marker)&(oldaudit.unit_key==u)].iloc[0]
            assert hashlib.sha256(smooth.tobytes()).hexdigest()==row.density_sha256
            curves.append(dict(dataset=d,unit=u,values=vals,density=smooth,level=level,color=palette[d]))
            level+=.57;prev=d
        maximum=np.max([c['density'] for c in curves],axis=0)
        ix=np.flatnonzero(maximum>=CUT);visible=centers[ix[[0,-1]]];pad=np.diff(visible)[0]*MARGIN
        lo=max(float(edges[0]),float(visible[0]-pad));hi=min(float(edges[-1]),float(visible[1]+pad))
        # Apply identical visibility and margin criteria to every method/marker.
        outside_grid=(centers<lo)|(centers>hi)
        assert maximum[outside_grid].max(initial=0)<CUT
        for c in curves:
            vals=c['values'];left=int((vals<lo).sum());right=int((vals>hi).sum())
            audit.append(dict(method=method,marker=marker,dataset=c['dataset'],unit_key=c['unit'],n_cells=len(vals),
              display_low=lo,display_high=hi,n_values_left_of_view=left,n_values_right_of_view=right,
              max_density_outside_view=float(c['density'][outside_grid].max(initial=0)),
              left_edge_density=float(np.interp(lo,centers,c['density'])),right_edge_density=float(np.interp(hi,centers,c['density'])),
              density_identical_to_full_range=True))
        outside=int(((values<lo)|(values>hi)).sum())
        summary.append(dict(method=method,marker=marker,old_low=edges[0],old_high=edges[-1],new_low=lo,new_high=hi,
          width_reduction_percent=100*(1-(hi-lo)/(edges[-1]-edges[0])),n_values_outside_view=outside,
          percent_values_inside_view=100*(1-outside/len(values)),density_peak_cutoff=CUT,margin_fraction=MARGIN,
          max_density_outside_view=float(maximum[outside_grid].max(initial=0)),bin_width=dx,smoothing_sigma_bins=6))
        prepared[(method,marker)]=dict(curves=curves,x=centers,old_limits=edges[[0,-1]],new_limits=[lo,hi],level=level)
    return prepared,summary,audit

def draw(ax,obj,mode):
    limits=obj[mode+'_limits'];x=obj['x']
    for i,c in enumerate(obj['curves']):
        s=c['density'];base=c['level'];changed=np.diff(s)!=0
        keep=np.r_[True,changed]|np.r_[changed,True]
        xx=x[keep];yy=base+.70*s[keep]
        ax.fill_between(xx,base,yy,color=c['color'],alpha=.82,lw=0,zorder=2*i+1)
        # Original curve is retained and clipped only by the viewport; no zeros or tails are invented.
        ax.plot(np.r_[obj['old_limits'][0],xx,obj['old_limits'][1]],np.r_[base,yy,base],c='#46505A',lw=.35,zorder=2*i+2)
    ax.set_xlim(*limits);ax.set_ylim(-.09,obj['level']-.57+.70+.04);ax.set_yticks([])
    if limits[0]<0<limits[1]:ax.axvline(0,c='#666666',lw=.35,ls=(0,(1.5,2.2)),alpha=.7,zorder=100)
    ticks=MaxNLocator(3).tick_values(*limits);ax.set_xticks(ticks[(ticks>=limits[0])&(ticks<=limits[1])])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:g}'))
    ax.tick_params(axis='x',labelsize=8,length=2,pad=2,width=.6)
    for sp in ax.spines.values():sp.set_color(LINE);sp.set_linewidth(.5)


def render(prepared):
    handles=[Line2D([],[],marker='o',ls='none',color=c,markersize=3,label=l) for c,l in zip(COLORS,['CODEX','IMC','MIBI'])]
    # Two full blocks show the same samples and exact density estimates, at equal physical sizes.
    fig=plt.figure(figsize=(10.4,7.3),facecolor='white')
    for k,(mode,title) in enumerate([('old','Current: full-range axes'),('new','Preview: tighter axes')]):
        base=.058+k*.5
        fig.text(base+.17,.976,title,fontsize=12,fontweight='bold',ha='center',va='top')
        for r,(method,label) in enumerate(zip(METHODS,DISPLAY)):
            bottom=.055+(3-r)*.215
            for j,marker in enumerate(MARKERS):
                ax=fig.add_axes([base+j*.135,bottom,.123,.186]);draw(ax,prepared[(method,marker)],mode)
                if r==0:ax.set_title(marker,fontsize=10,fontweight='bold',pad=6)
                if j==0:ax.set_ylabel(label,fontsize=9,fontweight='bold' if method=='RAMP' else 'normal',labelpad=5,
                                      color='#087F5B' if method=='RAMP' else INK)
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.956),ncol=3,frameon=False,fontsize=8,
              handletextpad=.1,columnspacing=1.5)
    out=HERE/'output/pdf/figure1_ridge_range_comparison.pdf';fig.savefig(out,dpi=600)
    fig.savefig(HERE/'output/figure1_ridge_range_comparison.svg');plt.close(fig)
    doc=fitz.open(out);doc[0].get_pixmap(matrix=fitz.Matrix(2,2)).save(HERE/'output/figure1_ridge_range_comparison.png')
    fig=plt.figure(figsize=(5.0,6.8),facecolor='white')
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.54,.994),ncol=3,frameon=False,fontsize=9)
    for r,(method,label) in enumerate(zip(METHODS,DISPLAY)):
        bottom=.055+(3-r)*.219
        for j,marker in enumerate(MARKERS):
            ax=fig.add_axes([.115+j*.293,bottom,.27,.190]);draw(ax,prepared[(method,marker)],'new')
            if r==0:ax.set_title(marker,fontsize=11,fontweight='bold',pad=6)
            if j==0:ax.set_ylabel(label,fontsize=10,fontweight='bold' if method=='RAMP' else 'normal',labelpad=8,
                                     color='#087F5B' if method=='RAMP' else INK)
    fig.savefig(HERE/'output/figure1_ridge_tighter_preview.png',dpi=220);plt.close(fig)

def main():
    locked=[ROOT.parent/'bioarxiv_Ver2/main.pdf',ROOT.parent/'bioarxiv_Ver2/main.tex',ROOT.parent/'bioarxiv_Ver2/figure/figure1_ramp_overview.pdf',OLD/'figure_1_melanoma.pdf']
    before={str(p):sha(p) for p in locked}
    meta,markers,matrices,sources=load();prepared,summary,audit=prepare(meta,markers,matrices)
    pd.DataFrame(summary).to_csv(HERE/'range_comparison.csv',index=False)
    pd.DataFrame(audit).to_csv(HERE/'per_ridge_audit.csv',index=False)
    render(prepared)
    assert all(sha(Path(p))==h for p,h in {**before,**sources}.items())
    prov=dict(status='preview_only_not_installed',visibility_threshold_fraction_of_each_curve_peak=CUT,
      display_padding_fraction=MARGIN,all_60000_cells_used_in_density_estimation=True,all_46_units_retained=True,
      curves_bitwise_identical_to_accepted_full_range_estimates=True,changed_only_x_axis_viewport=True,
      values_clipped_or_winsorized=False,rare_values_can_be_outside_view=True,
      source_hashes=sources,unchanged_manuscript_and_old_figure=before)
    (HERE/'PROVENANCE.json').write_text(json.dumps(prov,indent=2)+'\n')
    print(pd.DataFrame(summary).to_string(index=False))
if __name__=='__main__':main()
