import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
COL={'Raw':'#9aa6ad','VirTues-style':'#cb506a','Spatium-style':'#5b6ebf','RAMP':'#008866'}
O=['Raw','VirTues-style','Spatium-style','RAMP']
caps=['linear','mlp4','mlp8','mlp16','mlp32','mlp64']
bn=['bone_marrow','tnbc','melanoma','lung_cancer']; lbl=['BM','TNBC','MEL','LUNG']
ct=pd.read_csv('celltype_transfer_results.csv'); sc1=pd.read_csv('scaling_results.csv')
mc=pd.read_csv('mm_capacity.csv'); ms=pd.read_csv('mm_scaling.csv')
sets=['A@N','A@2N','A+B@2N']
fig,ax=plt.subplots(2,3,figsize=(12.6,6.6))

def panel_bench(a,piv,ylab,ylim,title):
    x=np.arange(4); w=.2
    for i,arm in enumerate(O): a.bar(x+(i-1.5)*w,piv.loc[bn,arm],w,color=COL[arm],label=arm,lw=0)
    a.set_xticks(x); a.set_xticklabels(lbl,fontsize=8); a.set_ylabel(ylab); a.set_ylim(*ylim)
    a.set_title(title,fontsize=9.5,loc='left')

def panel_cap(a,df,val,params,ylab,ylim,title):
    for arm in O:
        d=df[df.arm==arm].groupby('capacity')[val].mean().loc[caps]
        a.plot(params.loc[caps],d,color=COL[arm],marker='o',ms=3.5,lw=1.6)
    a.set_xscale('log'); a.set_xlabel('model parameters'); a.set_ylabel(ylab)
    a.set_ylim(*ylim); a.set_title(title,fontsize=9.5,loc='left')

def panel_scale(a,piv,ylab,ylim,title):
    for arm in O: a.plot(range(3),piv.loc[arm,sets],color=COL[arm],marker='o',ms=5,lw=1.8)
    a.set_xticks(range(3)); a.set_xticklabels(['1 cohort\nN','1 cohort\n2N','2 cohorts\n2N'],fontsize=7.5)
    a.set_ylabel(ylab); a.set_ylim(*ylim); a.set_title(title,fontsize=9.5,loc='left')

# ---- row 1: cell typing
b=ct.groupby(['benchmark','arm','capacity']).macro_f1.mean().reset_index()
piv=b.loc[b.groupby(['benchmark','arm']).macro_f1.idxmax()].pivot(index='benchmark',columns='arm',values='macro_f1')
panel_bench(ax[0,0],piv,'macro F1',(0,1),'A  Job 1 · accuracy by benchmark')
ax[0,0].legend(fontsize=6.3,frameon=False,ncol=2,loc='upper left')
panel_cap(ax[0,1],ct,'macro_f1',ct.groupby('capacity').params.mean(),'macro F1',(.3,.75),'B  Job 1 · capacity needed')
s=sc1.groupby(['arm','train_set']).macro_f1.mean().unstack()
panel_scale(ax[0,2],s,'macro F1',(.3,.78),'C  Job 1 · adding a cohort')

# ---- row 2: masked marker
b2=mc.groupby(['benchmark','arm','capacity']).spearman.mean().reset_index()
piv2=b2.loc[b2.groupby(['benchmark','arm']).spearman.idxmax()].pivot(index='benchmark',columns='arm',values='spearman')
panel_bench(ax[1,0],piv2,'within-unit Spearman',(0,.5),'D  Job 2 · accuracy by benchmark')
panel_cap(ax[1,1],mc,'spearman',mc.groupby('capacity').params.mean(),'within-unit Spearman',(.2,.42),'E  Job 2 · capacity needed')
s2=ms[ms.capacity=='linear'].groupby(['arm','train_set']).spearman.mean().unstack()
panel_scale(ax[1,2],s2,'within-unit Spearman',(.25,.45),'F  Job 2 · adding a cohort')

for r in ax:
    for a_ in r:
        for sp in ('top','right'): a_.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig('figure_2jobs_3questions.png',dpi=210); fig.savefig('figure_2jobs_3questions.pdf')
print('ok')
