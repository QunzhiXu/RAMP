"""Compare the current and two candidate cyCombine examples, with true cell labels."""
from pathlib import Path
import sys,os,json,hashlib
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
os.environ['MPLCONFIGDIR']=str(HERE/'.cache/matplotlib')
os.environ['MPLBACKEND']='Agg'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import rankdata

def sha(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def auc(a,b):
    r=rankdata(np.r_[a,b]);return (r[:len(a)].sum()-len(a)*(len(a)+1)/2)/(len(a)*len(b))

def main():
    frozen=json.loads((HERE/'scan_manifest.json').read_text())['input_sha256']
    base=ROOT/'benchmark_results/tnbc'
    paths=[base/'ramp/alignment/input/cell_metadata.csv.gz',base/'ramp/alignment/input/markers.txt',
        base/'ramp/alignment/input/raw.npy',base/'cycombine/alignment/data/corrected_expression.csv.gz']
    for p in paths:assert sha(p)==frozen[str(p)]
    meta=pd.read_csv(paths[0],dtype={'cell_id':str},low_memory=False)
    markers=paths[1].read_text().splitlines();j=markers.index('CD8')
    cy=pd.read_csv(paths[3],dtype={'cell_id':str},usecols=['cell_id',*markers]);assert np.array_equal(cy.cell_id,meta.cell_id)
    rp=base/'ramp/alignment/data/ramp_aligned_float64.npy'
    ramp_manifest=json.loads((base/'ramp/alignment/alignment_manifest.json').read_text())
    assert sha(rp)==ramp_manifest['outputs'][rp.name]
    matrices={'Raw':np.load(paths[2]),'cyCombine':cy[markers].to_numpy(),'RAMP':np.load(rp)}
    units=['Meyer IMC|B12.02518','Meyer IMC|B13.50280','Shiao CODEX|h17']
    colors=['#297AA4','#D36C58','#AAB5BB','#8571A8']
    titles=['Current example','Four matching cell types','Different annotated cell types']
    plt.rcParams.update({'font.family':'Arial','font.size':9,'svg.fonttype':'none','axes.linewidth':.65})
    fig,axes=plt.subplots(1,3,figsize=(11,3.95),layout='constrained')
    result=[];scores=[]
    for ax,unit,title in zip(axes,units,titles):
        ix=np.flatnonzero(meta.unit_key.eq(unit));local=meta.iloc[ix]
        if unit.startswith('Meyer'):
            cells=['CD4 T','CD8 T','B cells','Monocytes']
            masks=[local.celltype.eq('CD4 T'),local.celltype.eq('CD8 T'),local.provider_label_l1.eq('B_cell'),local.provider_label_l1.eq('Monocyte')]
            negative=0
        else:
            cells=['CD4 T','CD8 T','Plasma cells','Macrophages']
            masks=[local.celltype.eq('CD4 T'),local.celltype.eq('CD8 T'),local.provider_label_l1.eq('Plasma'),local.provider_label_l1.eq('Mac')]
            negative=2
        masks=[v.to_numpy() for v in masks]
        assert min(m.sum() for m in masks)>=50
        for i,(method,z) in enumerate(matrices.items()):
            v=z[ix,j];r=rankdata(v)/(len(ix)+1)
            if method=='cyCombine':ax.axvspan(i-.47,i+.47,color='#FBECE6',zorder=0)
            b=ax.boxplot([r[m] for m in masks],positions=i+np.array([-.30,-.10,.10,.30]),widths=.16,
                patch_artist=True,showfliers=False,manage_ticks=False,
                medianprops={'color':'#263239','lw':.9},boxprops={'edgecolor':'#566269','lw':.65},
                whiskerprops={'color':'#566269','lw':.65},capprops={'color':'#566269','lw':.65})
            for box,color in zip(b['boxes'],colors):box.set_facecolor(color)
            score=auc(v[masks[1]],v[masks[negative]])
            scores.append(dict(unit_key=unit,marker='CD8',method=method,positive=cells[1],negative=cells[negative],auc=score))
            for c,m in zip(cells,masks):
                result.append(dict(unit_key=unit,marker='CD8',method=method,celltype=c,n_cells=int(m.sum()),
                    q25=float(np.quantile(r[m],.25)),median=float(np.median(r[m])),q75=float(np.quantile(r[m],.75))))
        sc=[x['auc'] for x in scores if x['unit_key']==unit]
        ax.set_title(title+'\nCD8 T vs '+cells[negative]+f': {sc[0]:.3f} → {sc[1]:.3f}',loc='left',fontsize=10,pad=48)
        ax.legend(handles=[Line2D([],[],ls='',marker='s',color=c,label=l,ms=5) for c,l in zip(colors,cells)],
            loc='lower left',bbox_to_anchor=(0,1.015),frameon=False,ncol=2,fontsize=8,handletextpad=.3,columnspacing=.9,borderaxespad=0)
        ax.set_ylim(0,1);ax.set_yticks([0,.5,1]);ax.set_ylabel('CD8 percentile')
        ax.set_xlim(-.5,2.5);ax.set_xticks(range(3),matrices)
        ax.spines[['top','right']].set_visible(False)
        for t,m in zip(ax.get_xticklabels(),matrices):
            if m=='RAMP':t.set_color('#008561');t.set_weight('bold')
            if m=='cyCombine':t.set_color('#A64736')
    for ext in ['png','svg']:fig.savefig(HERE/f'cycombine_candidate_comparison.{ext}',dpi=260,facecolor='white')
    plt.close(fig)
    pd.DataFrame(result).to_csv(HERE/'cycombine_candidate_plot_statistics.csv',index=False)
    pd.DataFrame(scores).to_csv(HERE/'cycombine_candidate_plot_auc.csv',index=False)
    assert all(sha(p)==frozen[str(p)] for p in paths)
    (HERE/'candidate_plot_manifest.json').write_text(json.dumps(dict(units=units,all_y_axes=[0,1],raw_and_output_unchanged=True,
        raw_celltype_note='Shiao h17 B lineage is provider Plasma, not conventional B_cell; Myeloid is provider Mac, not Monocyte. Both are explicitly labelled in this review only.',
        input_sha256={**{str(p):sha(p) for p in paths},str(rp):sha(rp)}),indent=2)+'\n')
    print(HERE/'cycombine_candidate_comparison.png')

if __name__=='__main__':main()
