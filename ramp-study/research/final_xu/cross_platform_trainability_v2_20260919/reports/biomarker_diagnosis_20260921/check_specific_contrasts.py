"""Fixed cell-type contrasts and composition dependence; no fitting or filtering by score."""
import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata,spearmanr

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
os.environ['MPLCONFIGDIR']=str(HERE/'.mplconfig')
os.environ['MPLBACKEND']='Agg'
CONTRASTS={
 'melanoma_story':[
  ('CD20','B lineage','CD8 T'),('CD20','B lineage','Other T / CD8-'),('CD20','B lineage','Myeloid'),
  ('CD8','CD8 T','Other T / CD8-'),('CD14','Myeloid','B lineage'),('CD14','Myeloid','CD8 T'),
  ('CD14','Myeloid','Other T / CD8-'),('CD31','Endothelial','Myeloid')],
 'hnscc_story':[
  ('CD20','B','CD4 T'),('CD20','B','CD8 T'),('CD21','B','CD4 T'),('CD21','B','CD8 T'),('CD21','B','Myeloid'),
  ('CD8','CD8 T','CD4 T'),('CD56','NK / gdT','CD8 T'),('CD56','NK / gdT','CD4 T'),
  ('CD7','NK / gdT','CD8 T'),('CD38','Plasma','B'),('CD11c','Myeloid','CD4 T'),('CD11c','Myeloid','B')]
}
FOCUS=[('melanoma_story','B lineage','CD20'),('melanoma_story','CD8 T','CD8'),
       ('hnscc_story','B','CD20'),('hnscc_story','CD8 T','CD8')]


def main():
    rows=[];comp=[];stats=[]
    profile=pd.read_csv(HERE/'CLASS_MARKER_QUARTILES_PER_UNIT.csv')
    for task,contrasts in CONTRASTS.items():
        folder=ROOT/f'data/real_inputs_v0_3_r2/{task}'
        man=json.loads((folder/'manifest.json').read_text());md=pd.read_csv(folder/'metadata.csv.gz')
        raw=np.load(folder/'native.npy');ramp=np.load(ROOT/f'representations/formal_v0_5/{task}/complete/ramp/values.npy')
        for marker,pos,neg in contrasts:
            j=man['markers'].index(marker)
            for (ds,unit),g in md[md.celltype.isin([pos,neg])].groupby(['dataset','unit_key']):
                y=g.celltype.eq(pos).to_numpy();np_=y.sum();nn=len(y)-np_
                if min(np_,nn)<5:continue
                for rep,x in [('native',raw),('ramp',ramp)]:
                    auc=(rankdata(x[g.index,j])[y].sum()-np_*(np_+1)/2)/(np_*nn)
                    rows.append(dict(task=task,dataset=ds,unit=unit,marker=marker,positive=pos,negative=neg,
                                     representation=rep,n_positive=np_,n_negative=nn,auc=auc))
        totals=md.groupby('unit_key').size()
        for _,cl,m in [x for x in FOCUS if x[0]==task]:
            g=profile[(profile.task==task)&(profile.celltype==cl)&(profile.marker==m)&profile.representation.isin(['ramp','kmeanslow','native'])].copy()
            g['class_fraction_full_alignment_pool']=g.cells/g.unit.map(totals)
            comp.append(g)
            for (ds,rep),q in g.groupby(['dataset','representation']):
                rho=spearmanr(q.class_fraction_full_alignment_pool,q['median']).statistic
                qq=q['median'].quantile([.25,.5,.75]).to_numpy()
                stats.append(dict(task=task,dataset=ds,celltype=cl,marker=m,representation=rep,units=len(q),
                                  fraction_value_spearman=rho,patient_medians_q25=qq[0],median_of_patient_medians=qq[1],patient_medians_q75=qq[2]))
    df=pd.DataFrame(rows);df.to_csv(HERE/'SPECIFIC_CONTRAST_AUC_PER_UNIT.csv',index=False)
    sm=df.groupby(['task','dataset','marker','positive','negative','representation'],as_index=False).agg(mean_patient_auc=('auc','mean'),patients=('unit','nunique'))
    sm.to_csv(HERE/'SPECIFIC_CONTRAST_AUC_SUMMARY.csv',index=False)
    pd.concat(comp).to_csv(HERE/'ABUNDANCE_VALUE_PER_UNIT.csv',index=False)
    st=pd.DataFrame(stats);st.to_csv(HERE/'ABUNDANCE_VALUE_ASSOCIATION.csv',index=False)
    # Color and scale are identical across the two platforms within each subplot.
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,4,figsize=(13,3.5))
    data=pd.concat(comp)
    for ax,(task,cl,marker) in zip(axes,FOCUS):
        part=data[(data.task==task)&(data.celltype==cl)&(data.marker==marker)&(data.representation=='ramp')]
        for i,(ds,g) in enumerate(part.groupby('dataset')):
            label=ds.replace('Engelbrecht ','').replace('Moldoveanu ','').replace('GSE212797_','').replace('Haist_','')
            ax.scatter(100*g.class_fraction_full_alignment_pool,g['median'],label=label,s=22,color=['#2166ac','#e08214'][i],alpha=.8)
        ax.set_title(f'{"Melanoma" if task=="melanoma_story" else "HNSCC"}: {cl}\n{marker}',fontsize=10,fontweight='bold')
        ax.set_xlabel('Cell-type fraction of all cells (%)',fontsize=8,fontweight='bold')
        ax.set_ylabel('Median RAMP marker value' if ax==axes[0] else '',fontsize=9,fontweight='bold')
        ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False,fontsize=8)
    fig.tight_layout();fig.savefig(HERE/'abundance_vs_ramp_value.png',dpi=230);fig.savefig(HERE/'abundance_vs_ramp_value.pdf');plt.close(fig)
    print(sm[sm.representation.eq('native')].to_string(index=False))
    pivot=df.pivot(index=['task','dataset','unit','marker','positive','negative'],columns='representation',values='auc')
    assert float((pivot.native-pivot.ramp).abs().max())<1e-12
    print('Specific-contrast native/RAMP AUROCs identical; no rank discrimination loss.')


if __name__=='__main__':main()
