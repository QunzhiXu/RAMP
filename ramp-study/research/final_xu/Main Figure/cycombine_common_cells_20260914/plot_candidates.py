"""Compare three common-cell-type cyCombine examples without displaying Plasma."""
from pathlib import Path
import os,sys,json,hashlib
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
os.environ['MPLCONFIGDIR']=str(HERE/'.cache/matplotlib');os.environ['MPLBACKEND']='Agg'
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
    prior=HERE.parent/'harmony_cycombine_examples_20260914'
    frozen=json.loads((prior/'scan_manifest.json').read_text())['input_sha256']
    base=ROOT/'benchmark_results/tnbc'
    paths=[base/'ramp/alignment/input/cell_metadata.csv.gz',base/'ramp/alignment/input/markers.txt',
        base/'ramp/alignment/input/raw.npy',base/'cycombine/alignment/data/corrected_expression.csv.gz']
    for p in paths:assert sha(p)==frozen[str(p)]
    meta=pd.read_csv(paths[0],low_memory=False,dtype={'cell_id':str});markers=paths[1].read_text().splitlines()
    cy=pd.read_csv(paths[3],dtype={'cell_id':str},usecols=['cell_id',*markers]);assert np.array_equal(cy.cell_id,meta.cell_id)
    rp=base/'ramp/alignment/data/ramp_aligned_float64.npy'
    ramp_manifest=json.loads((base/'ramp/alignment/alignment_manifest.json').read_text())
    assert sha(rp)==ramp_manifest['outputs'][rp.name]
    matrices={'Raw':np.load(paths[2]),'cyCombine':cy[markers].to_numpy(),'RAMP':np.load(rp)}
    examples=[('Meyer IMC|B12.64192','CD68','CD68: T-cell signal increases'),
        ('Shiao CODEX|h17','CD8','CD8: target signal decreases'),
        ('Meyer IMC|B12.64192','CD8','CD8: CD4 T signal increases')]
    colors=['#297AA4','#D36C58','#8571A8']
    plt.rcParams.update({'font.family':'Arial','font.size':9,'svg.fonttype':'none','axes.linewidth':.65})
    fig,axes=plt.subplots(1,3,figsize=(11,3.75),layout='constrained')
    statistics=[];scores=[]
    for ax,(unit,marker,title) in zip(axes,examples):
        j=markers.index(marker);ix=np.flatnonzero(meta.unit_key.eq(unit));local=meta.iloc[ix]
        third='Monocytes' if unit.startswith('Meyer') else 'Macrophages'
        native='Monocyte' if unit.startswith('Meyer') else 'Mac'
        labels=['CD4 T','CD8 T',third]
        masks=[local.celltype.eq('CD4 T').to_numpy(),local.celltype.eq('CD8 T').to_numpy(),local.provider_label_l1.eq(native).to_numpy()]
        assert min(m.sum() for m in masks)>=50
        positive,negative=(2,0) if marker=='CD68' else ((1,2) if third=='Macrophages' else (1,0))
        local_auc=[]
        for i,(method,z) in enumerate(matrices.items()):
            v=z[ix,j];r=rankdata(v)/(len(ix)+1)
            if method=='cyCombine':ax.axvspan(i-.46,i+.46,color='#FBECE6',zorder=0)
            b=ax.boxplot([r[m] for m in masks],positions=i+np.array([-.25,0,.25]),widths=.2,
                patch_artist=True,showfliers=False,manage_ticks=False,
                medianprops={'color':'#263239','lw':.9},boxprops={'edgecolor':'#566269','lw':.65},
                whiskerprops={'color':'#566269','lw':.65},capprops={'color':'#566269','lw':.65})
            for box,color in zip(b['boxes'],colors):box.set_facecolor(color)
            score=auc(v[masks[positive]],v[masks[negative]]);local_auc.append(score)
            scores.append(dict(unit_key=unit,marker=marker,method=method,positive=labels[positive],negative=labels[negative],
                n_positive=int(masks[positive].sum()),n_negative=int(masks[negative].sum()),auc=score))
            for c,m in zip(labels,masks):
                statistics.append(dict(unit_key=unit,marker=marker,method=method,celltype=c,n_cells=int(m.sum()),
                    q25=float(np.quantile(r[m],.25)),median=float(np.median(r[m])),q75=float(np.quantile(r[m],.75))))
        ax.set_title(title+'\n'+labels[positive]+' vs '+labels[negative]+f': {local_auc[0]:.3f} → {local_auc[1]:.3f}',loc='left',fontsize=10,pad=32)
        ax.legend(handles=[Line2D([],[],ls='',marker='s',color=c,label=l,ms=5) for c,l in zip(colors,labels)],
            loc='lower left',bbox_to_anchor=(0,1.015),frameon=False,ncol=3,fontsize=8,handletextpad=.25,columnspacing=.7,borderaxespad=0)
        ax.set_ylim(0,1);ax.set_yticks([0,.5,1]);ax.set_ylabel(marker+' percentile')
        ax.set_xlim(-.5,2.5);ax.set_xticks(range(3),matrices)
        ax.spines[['top','right']].set_visible(False)
        for t,m in zip(ax.get_xticklabels(),matrices):
            if m=='RAMP':t.set_color('#008561');t.set_weight('bold')
            if m=='cyCombine':t.set_color('#A64736')
    for ext in ['png','svg']:fig.savefig(HERE/f'candidates_without_plasma.{ext}',dpi=260,facecolor='white')
    plt.close(fig)
    pd.DataFrame(statistics).to_csv(HERE/'candidate_box_statistics.csv',index=False)
    pd.DataFrame(scores).to_csv(HERE/'candidate_auc.csv',index=False)
    assert all(sha(p)==frozen[str(p)] for p in paths)
    (HERE/'candidate_manifest.json').write_text(json.dumps(dict(examples=examples,all_y_axes=[0,1],minimum_group_size=50,
        no_plasma_displayed=True,all_cells_retained_in_percentile_reference=True,
        label_resolution='Meyer provider Monocyte; Shiao provider Mac. They remain explicitly distinct.',
        input_sha256={**{str(p):sha(p) for p in paths},str(rp):sha(rp)}),indent=2)+'\n')
    print(HERE/'candidates_without_plasma.png')

if __name__=='__main__':main()
