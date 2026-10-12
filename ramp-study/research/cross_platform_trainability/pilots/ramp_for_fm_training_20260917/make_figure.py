import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
COL={'Raw':'#9aa6ad','VirTues-style':'#cb506a','Spatium-style':'#5b6ebf','RAMP':'#008866'}
ORDER=['Raw','VirTues-style','Spatium-style','RAMP']
ct=pd.read_csv('celltype_transfer_results.csv'); sc=pd.read_csv('scaling_results.csv')
mm=pd.read_csv('masked_marker_results.csv')
fig,ax=plt.subplots(1,4,figsize=(14.2,3.4))

# A: accuracy per benchmark (best capacity)
b=ct.groupby(['benchmark','arm','capacity']).macro_f1.mean().reset_index()
piv=b.loc[b.groupby(['benchmark','arm']).macro_f1.idxmax()].pivot(index='benchmark',columns='arm',values='macro_f1')
bn=['bone_marrow','tnbc','melanoma','lung_cancer']; x=np.arange(4); w=.2
for i,a in enumerate(ORDER):
    ax[0].bar(x+(i-1.5)*w,piv.loc[bn,a],w,color=COL[a],label=a,lw=0)
ax[0].set_xticks(x); ax[0].set_xticklabels(['BM','TNBC','MEL','LUNG'],fontsize=8)
ax[0].set_ylabel('macro F1 on unseen platform'); ax[0].set_ylim(0,1)
ax[0].set_title('A  Cross-platform cell typing',fontsize=9.5,loc='left')
ax[0].legend(fontsize=6.4,frameon=False,ncol=2,loc='upper left')

# B: capacity curve
caps=['linear','mlp4','mlp8','mlp16','mlp32','mlp64']
pr=ct.groupby('capacity').params.mean()
for a in ORDER:
    d=ct[ct.arm==a].groupby('capacity').macro_f1.mean().loc[caps]
    ax[1].plot(pr.loc[caps],d,color=COL[a],marker='o',ms=3.5,lw=1.5)
ax[1].set_xscale('log'); ax[1].set_xlabel('model parameters')
ax[1].set_ylabel('macro F1 on unseen platform'); ax[1].set_ylim(.3,.75)
ax[1].set_title('B  Capacity needed',fontsize=9.5,loc='left')

# C: masked marker
mb=mm.groupby(['benchmark','arm','capacity']).spearman.mean().reset_index()
mp=mb.loc[mb.groupby(['benchmark','arm']).spearman.idxmax()].pivot(index='benchmark',columns='arm',values='spearman')
for i,a in enumerate(ORDER):
    ax[2].bar(x+(i-1.5)*w,mp.loc[bn,a],w,color=COL[a],lw=0)
ax[2].set_xticks(x); ax[2].set_xticklabels(['BM','TNBC','MEL','LUNG'],fontsize=8)
ax[2].set_ylabel('within-unit Spearman'); ax[2].set_ylim(0,.5)
ax[2].set_title('C  Masked-marker reconstruction',fontsize=9.5,loc='left')

# D: scaling
sets=['A@N','A@2N','A+B@2N']
sp=sc.groupby(['arm','train_set']).macro_f1.mean().unstack()[sets]
for a in ORDER:
    ax[3].plot(range(3),sp.loc[a],color=COL[a],marker='o',ms=5,lw=1.8)
ax[3].set_xticks(range(3))
ax[3].set_xticklabels(['1 cohort\nN cells','1 cohort\n2N cells','2 cohorts\n2N cells'],fontsize=7.5)
ax[3].set_ylabel('macro F1 on unseen platform'); ax[3].set_ylim(.3,.78)
ax[3].set_title('D  Does adding a cohort help?',fontsize=9.5,loc='left')
for s in ('top','right'):
    for a_ in ax: a_.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig('figure_ramp_for_fm.png',dpi=220); fig.savefig('figure_ramp_for_fm.pdf')
print('ok')
