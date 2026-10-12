"""Real CD3 density distributions and frozen threshold, inspired by ADTnorm Fig. 1A.

The test is source-calibrated gate transfer, not ADTnorm's automatic valley gating.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
from scipy.stats import gaussian_kde

CASE='bone_marrow'
SOURCE='Triana AbSeq'
TARGET='BM CODEX'
POSITIVE={'CD4 T','CD8 T','CD8− T','Other T / CD8−','Other T / NKT'}
AMBIGUOUS={'Non-joint','Mix','Unclassified / technical','Unannotated','nan'}

def balanced_weights(pos,units):
    w=np.zeros(len(pos))
    for c in [False,True]:
        us=np.unique(units[pos==c])
        for u in us:
            m=(units==u)&(pos==c);w[m]=.5/(len(us)*m.sum())
    assert np.isclose(w.sum(),1)
    return w

def draw(fig,box):
    title(fig,box,'E','Fixed CD3 gate transfer')
    path=ROOT/'benchmark_results'/CASE/'ramp/alignment'
    meta=csv(path/'input/cell_metadata.csv.gz',low_memory=False)
    markers=track(path/'input/markers.txt').read_text().splitlines()
    col=markers.index('CD3')
    arrays={'Raw':np.asarray(np.load(track(path/'input/raw.npy'),mmap_mode='r')[:,col],dtype=float),
            'RAMP':np.asarray(np.load(track(path/'data/ramp_aligned.npy'),mmap_mode='r')[:,col],dtype=float)}
    p=FIG2/'value_evaluation_20260914/results'/CASE
    eligibility=csv(p/'gate_eligibility.csv')
    gates=csv(p/'source_gates.csv')
    directions=csv(FIG2/'value_evaluation_20260914/summary/gate_direction_means.csv')
    labcol='celltype_joint_evaluation' if 'celltype_joint_evaluation' in meta else 'celltype'
    labs=meta[labcol].fillna('Unannotated').astype(str)
    included=~labs.isin(AMBIGUOUS).to_numpy()
    positive=labs.isin(POSITIVE).to_numpy()
    units=meta.unit_key.astype(str).to_numpy()
    masks={}
    for dataset in [SOURCE,TARGET]:
        eligible=eligibility[(eligibility.marker=='CD3')&(eligibility.dataset==dataset)&eligibility.primary_eligible]
        masks[dataset]=np.flatnonzero(included&meta.dataset.eq(dataset).to_numpy()&np.isin(units,eligible.unit_key.astype(str)))
    assert len(np.unique(units[masks[SOURCE]]))==3 and len(np.unique(units[masks[TARGET]]))==12
    # Real source and target cells; both classes are weighted equally, and units
    # equally within each class. Density curves aid viewing only.
    display=[]; density=[]; stats=[]
    class_colors={True:'#C58A34',False:'#788D9C'}
    for r,method in enumerate(['Raw','RAMP']):
        g=gates[(gates.marker=='CD3')&(gates.source_dataset==SOURCE)&(gates.method==method)].iloc[0]
        threshold=float(g.threshold); gate_x=float(np.arcsinh(threshold))
        all_idx=np.concatenate(list(masks.values()))
        all_x=np.arcsinh(arrays[method][all_idx])
        lo,hi=float(all_x.min()),float(all_x.max());pad=.04*(hi-lo)
        xmin=0 if method=='Raw' else lo-pad;xmax=hi+pad
        xx=np.linspace(xmin,xmax,600)
        curves={}; ymax=0
        for dataset in [SOURCE,TARGET]:
            ids=masks[dataset];y=positive[ids];w=balanced_weights(y,units[ids])
            v=arrays[method][ids];x=np.arcsinh(v)
            for c in [False,True]:
                sel=y==c
                kde=gaussian_kde(x[sel],weights=w[sel],bw_method=.22)
                z=.5*kde(xx);curves[dataset,c]=z;ymax=max(ymax,float(z.max()))
                density.extend(dict(method=method,dataset=dataset,reference_class='T cells' if c else 'Other cells',
                    asinh_cd3=float(a),balanced_density=float(b)) for a,b in zip(xx,z))
            ba=float(w[(v>=threshold)==y].sum())
            expected=float(g.source_balanced_accuracy) if dataset==SOURCE else float(directions[
                (directions.benchmark==CASE)&(directions.marker=='CD3')&(directions.method==method)&
                (directions.source_dataset==SOURCE)&(directions.target_dataset==TARGET)].iloc[0].balanced_accuracy)
            assert np.isclose(ba,expected,atol=1e-12),(method,dataset,ba,expected)
            stats.append(dict(benchmark=CASE,marker='CD3',method=method,dataset=dataset,threshold_native=threshold,
                threshold_asinh=gate_x,balanced_accuracy=ba,n_cells=len(ids),n_units=len(np.unique(units[ids])),
                n_positive=int(y.sum()),n_negative=int((~y).sum())))
            display.append(pd.DataFrame({'cell_id':meta.cell_id.iloc[ids].to_numpy(),'method':method,'dataset':dataset,
                'unit_key':units[ids],'reference_class':np.where(y,'T cells','Other cells'),
                'cd3_native':v,'cd3_asinh_display':x,'weight':w,'predicted_positive':v>=threshold}))
        for c,dataset in enumerate([SOURCE,TARGET]):
            ax=mm_axes(fig,box,(8+c*44,15+r*23,35,15.5))
            ax.axvspan(gate_x,xmax,color='#008866',alpha=.045,zorder=0)
            for cls in [False,True]:
                z=curves[dataset,cls]
                ax.fill_between(xx,z,color=class_colors[cls],alpha=.34,linewidth=0)
                ax.plot(xx,z,color=class_colors[cls],lw=.8)
            ax.axvline(gate_x,color='#202B30',ls=(0,(3,2)),lw=.7)
            ax.set_xlim(xmin,xmax);ax.set_ylim(0,ymax*1.28)
            ax.set_yticks([]);ax.tick_params(axis='x',labelsize=6,length=1.8,pad=1)
            ax.set_xticks([0,2,4,6,8] if method=='Raw' else [-1,0,1,2])
            ax.spines[['top','right','left']].set_visible(False);ax.spines['bottom'].set_color('#7A858C')
            ba=next(s['balanced_accuracy'] for s in stats if s['method']==method and s['dataset']==dataset)
            ax.text(.98,.93,f'BA = {ba:.3f}',ha='right',va='top',transform=ax.transAxes,fontsize=6.3)
            ax.text(gate_x,.96*ymax*1.28,'gate',ha='center',va='top',fontsize=5.7,
                bbox=dict(facecolor='white',edgecolor='none',pad=.25))
            if r==1:ax.set_xlabel('asinh(CD3)',fontsize=6.5,labelpad=1.5)
        mm_text(fig,box,3.7,22.7+r*23,method,rotation=90,va='center',ha='center',fontsize=7.2,
            weight='bold',color=METHOD_COLORS[method] if method=='RAMP' else INK)
    mm_text(fig,box,25.5,10.3,'AbSeq · set gate',ha='center',fontsize=7,weight='bold')
    mm_text(fig,box,69.5,10.3,'CODEX · same gate',ha='center',fontsize=7,weight='bold')
    # Compact class legend beside the panel heading.
    for x,label,cls in [(59,'Other cells',False),(79,'T cells',True)]:
        mm_text(fig,box,x-2,3.4,'●',fontsize=6,color=class_colors[cls],va='center',ha='right')
        mm_text(fig,box,x,3.4,label,fontsize=6.3,va='center')
    save_data('E',pd.DataFrame(stats),'gate_statistics.csv')
    save_data('E',pd.DataFrame(density),'density_coordinates.csv')
    save_data('E',pd.concat(display,ignore_index=True),'display_cells.csv.gz')
    return {'benchmark':CASE,'marker':'CD3','source_dataset':SOURCE,'target_dataset':TARGET,
        'selection':'Illustrative CD3 direction chosen after reviewing direction-level results; not a random selection. All eligible units in both platforms included, with no donor/ROI outcome selection.',
        'source_n_units':3,'target_n_units':12,'statistics':stats,
        'definition':'Frozen source-calibrated gates from the completed value evaluation; no target threshold fitting.',
        'visual_inspiration':'ADTnorm Nature Communications 2025 Fig. 1A density/threshold depiction. ADTnorm Fig. 1E is an auto-gating accuracy summary; this is a distinct cross-platform fixed-gate transfer task.',
        'doi':'10.1038/s41467-025-61023-6',
        'display_transform':'arcsinh(value), cofactor=1, for display of both Raw and RAMP; gate transformed identically. Full observed range shown; no quantile clipping or cell sampling.',
        'density':'Weighted Gaussian KDE in display coordinates, bandwidth factor 0.22. Class and unit balancing match source gate BA; curves are display only.',
        'axis_limits':'Same x/y limits for source and target within each method row. Raw and RAMP rows use their own native output ranges after the same display transform.',
        'ba_recomputed_matches_frozen':True,
        'not_absolute_concentration_validation':True}
if __name__=='__main__':run_panel('E',draw)
