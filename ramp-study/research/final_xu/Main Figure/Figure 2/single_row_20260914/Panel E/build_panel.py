"""Four vertically stacked views of the already frozen CD3 gate example."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *

SOURCE='Triana AbSeq'
TARGET='BM CODEX'
COLORS={'Other cells':'#788D9C','T cells':'#C58A34'}

def draw(fig,box):
    title(fig,box,'E','CD3 gating')
    density=csv(PRIOR/'Panel E/density_coordinates.csv')
    stats=csv(PRIOR/'Panel E/gate_statistics.csv')
    assert len(density)==4800 and len(stats)==4
    assert set(stats.n_units)=={3,12}
    save_data('E',density,'density_coordinates.csv')
    save_data('E',stats,'gate_statistics.csv')
    for x,label in [(3.8,'Other cells'),(20.4,'T cells')]:
        mm_text(fig,box,x-1.1,7.2,'●',color=COLORS[label],fontsize=5.4,ha='right',va='center')
        mm_text(fig,box,x,7.2,label,fontsize=5.7,va='center')
    for method,y0 in [('Raw',11),('RAMP',36)]:
        m=density[density.method.eq(method)]
        xmin,xmax=m.asinh_cd3.min(),m.asinh_cd3.max()
        ymax=m.balanced_density.max()*1.40
        mm_text(fig,box,1.4,y0+10.5,method,rotation=90,ha='center',va='center',
            fontsize=6.3,weight='bold',color='#008866' if method=='RAMP' else INK)
        for i,dataset in enumerate([SOURCE,TARGET]):
            s=stats[stats.method.eq(method)&stats.dataset.eq(dataset)].iloc[0]
            ax=mm_axes(fig,box,(4.4,y0+i*12,24.8,8.8))
            ax.axvspan(s.threshold_asinh,xmax,color='#008866',alpha=.045,zorder=0)
            for label,color in COLORS.items():
                d=m[m.dataset.eq(dataset)&m.reference_class.eq(label)].sort_values('asinh_cd3')
                assert len(d)==600
                ax.fill_between(d.asinh_cd3,d.balanced_density,color=color,alpha=.34,lw=0)
                ax.plot(d.asinh_cd3,d.balanced_density,color=color,lw=.7)
            ax.axvline(s.threshold_asinh,color=INK,ls=(0,(3,2)),lw=.6)
            ax.set_xlim(xmin,xmax);ax.set_ylim(0,ymax);ax.set_yticks([])
            ax.spines[['top','left','right']].set_visible(False)
            ax.spines['bottom'].set_color('#7A858C')
            ax.set_xticks([0,4,8] if method=='Raw' else [-1,0,1,2])
            ax.tick_params(axis='x',length=1.5,pad=1,labelsize=5.6,labelbottom=i==1)
            ax.text(.02,.97,'AbSeq' if dataset==SOURCE else 'CODEX',transform=ax.transAxes,
                fontsize=5.8,va='top',ha='left',weight='bold')
            if dataset==TARGET:
                ax.text(.98,.97,f'BA {s.balanced_accuracy:.3f}',transform=ax.transAxes,
                    fontsize=5.7,va='top',ha='right')
            else:
                ax.text(.98,.97,'Set gate',transform=ax.transAxes,fontsize=5.6,va='top',ha='right')
            if method=='RAMP' and i==1:ax.set_xlabel('asinh(CD3)',fontsize=6.1,labelpad=1.5)
    return {'source':'Exact cached curves and statistics from prior two-row Panel E; no recomputation.',
        'layout':'One column: Raw AbSeq, Raw CODEX, RAMP AbSeq, RAMP CODEX.',
        'source_n_units':3,'target_n_units':12,'statistics':stats.to_dict('records'),
        'gate_definition':'Source labels calibrate each method-specific threshold, which is frozen and applied to the target without refitting.',
        'display':'arcsinh(value), cofactor 1. Same full x/y range for source and target within each method. Raw and RAMP retain their own output ranges.',
        'density':'Unchanged class- and donor-balanced KDE; bandwidth factor 0.22.',
        'selection':'Illustrative CD3 transfer direction previously selected after inspecting results; all eligible source/target donors retained.',
        'not_absolute_concentration_validation':True}

if __name__=='__main__':run_panel('E',draw)
