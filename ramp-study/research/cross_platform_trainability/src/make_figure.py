import pandas as pd, numpy as np, glob, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
ORDER=['Raw','Harmony-direct','CytoNorm','cyCombine','fastMNN-reconstructed','VirTues-style','Spatium-style','RAMP']
SHORT={'Raw':'Raw','Harmony-direct':'Harmony','CytoNorm':'CytoNorm','cyCombine':'cyCombine',
       'fastMNN-reconstructed':'fastMNN','VirTues-style':'VirTues','Spatium-style':'Spatium','RAMP':'RAMP'}
COL={a:'#9aa6ad' for a in ORDER}; COL.update({'RAMP':'#008866','VirTues-style':'#cb506a',
     'Spatium-style':'#5b6ebf','cyCombine':'#9772B4','CytoNorm':'#479DAD',
     'fastMNN-reconstructed':'#397CA8','Harmony-direct':'#858B91'})
CAPS=['linear','mlp4','mlp8','mlp16','mlp32','mlp64','mlp128','mlp256']
TASKS=['bone_marrow','tnbc','melanoma','lung_cancer','lung_story','melanoma_story','hnscc_story']
df=pd.concat([pd.read_csv(f,low_memory=False) for f in sorted(glob.glob('../results/raw/*.csv.gz'))],ignore_index=True)

fig,ax=plt.subplots(2,3,figsize=(14.4,7.6))
x=np.arange(len(ORDER))

def bars(a,vals,ylab,title,hl=None,chance=None):
    c=[COL[k] for k in ORDER]
    a.bar(x,[vals.get(k,np.nan) for k in ORDER],color=c,lw=0)
    if chance: a.axhline(chance,color='#333',ls=(0,(3,2)),lw=.7)
    a.set_xticks(x); a.set_xticklabels([SHORT[k] for k in ORDER],rotation=40,ha='right',fontsize=7)
    a.set_ylabel(ylab); a.set_title(title,fontsize=9.5,loc='left')
    if hl: a.text(x[ORDER.index(hl)],vals[hl],'★',ha='center',va='bottom',color='#008866')

# A: job1 mean rank, kNN
d=df[(df.job=='job1_celltype')&(df.capacity=='knn')&(df.train_set=='A@N')]
p=d.groupby(['task','arm']).macro_f1.mean().unstack()[ORDER]
bars(ax[0,0],p.rank(axis=1,ascending=False).mean(),'mean rank (1 = best)',
     'A  Job 1 cell typing · kNN · mean rank',hl=None)
ax[0,0].invert_yaxis()

# B: job1 per task, linear
d=df[(df.job=='job1_celltype')&(df.capacity=='linear')&(df.train_set=='A@N')]
p1=d.groupby(['task','arm']).macro_f1.mean().unstack()[ORDER].reindex(TASKS)
for a_ in ORDER:
    ax[0,1].plot(range(7),p1[a_],marker='o',ms=3.5,lw=1.4 if a_=='RAMP' else .9,
                 color=COL[a_],alpha=1 if a_=='RAMP' else .75,label=SHORT[a_])
ax[0,1].set_xticks(range(7)); ax[0,1].set_xticklabels(['BM','TNBC','MEL','LUNG','LUNGs','MELs','HNSCC'],fontsize=7)
ax[0,1].set_ylabel('macro F1'); ax[0,1].set_title('B  Job 1 per task · linear',fontsize=9.5,loc='left')
ax[0,1].legend(fontsize=5.8,frameon=False,ncol=2,loc='upper right')

# C: capacity curve
d=df[(df.job=='job1_celltype')&(df.train_set=='A@N')&(df.capacity!='knn')]
pr=d.groupby('capacity').params.mean()[CAPS]
for a_ in ORDER:
    v=d[d.arm==a_].groupby('capacity').macro_f1.mean()[CAPS]
    ax[0,2].plot(pr,v,marker='o',ms=3,lw=1.6 if a_=='RAMP' else .9,color=COL[a_],
                 alpha=1 if a_=='RAMP' else .7)
ax[0,2].set_xscale('log'); ax[0,2].set_xlabel('parameters'); ax[0,2].set_ylabel('macro F1')
ax[0,2].set_title('C  Job 1 capacity (M2)',fontsize=9.5,loc='left')

# D: job3 control
d=df[(df.job=='job3_platform')&(df.capacity=='knn')]
p3=d.groupby(['task','arm']).balanced_accuracy.mean().unstack()[ORDER]
bars(ax[1,0],p3.rank(axis=1,ascending=True).mean(),'mean rank (1 = hardest to tell)',
     'D  Job 3 platform · CONTROL, lower = better')
ax[1,0].invert_yaxis()

# E: job2
d=df[(df.job=='job2_marker_transfer')&(df.capacity=='linear')&(df.train_set=='A@N')]
p2=d.groupby(['task','arm']).spearman.mean().unstack()[ORDER]
v=p2.mean()
c=[COL[k] for k in ORDER]
ax[1,1].bar(x,[v[k] for k in ORDER],color=c,lw=0)
ax[1,1].bar([ORDER.index('fastMNN-reconstructed')],[v['fastMNN-reconstructed']],
            color='none',edgecolor='#c00',lw=1.4,hatch='///')
ax[1,1].set_xticks(x); ax[1,1].set_xticklabels([SHORT[k] for k in ORDER],rotation=40,ha='right',fontsize=7)
ax[1,1].set_ylabel('within-unit Spearman')
ax[1,1].set_title('E  Job 2 marker transfer · linear',fontsize=9.5,loc='left')
ax[1,1].text(ORDER.index('fastMNN-reconstructed'),v['fastMNN-reconstructed']+.02,
             'rank-\ndeficient',ha='center',fontsize=5.5,color='#c00')

# F: M4
d=df[(df.job=='job1_celltype')&(df.capacity=='linear')]
w=d.pivot_table(index=['task','held_out','train_platform','arm','rep'],columns='train_set',
                values='macro_f1').reset_index().dropna(subset=['A@2N','A+B@2N'])
w['gain']=w['A+B@2N']-w['A@2N']; w['scale']=w['A@2N']-w['A@N']
g=w.groupby('arm')
ax[1,2].bar(x-.2,[g.scale.mean()[k] for k in ORDER],.4,color='#c8ccd0',label='more data, same platform')
ax[1,2].bar(x+.2,[g.gain.mean()[k] for k in ORDER],.4,color=[COL[k] for k in ORDER],label='+ second platform')
ax[1,2].axhline(0,color='#333',lw=.7)
ax[1,2].set_xticks(x); ax[1,2].set_xticklabels([SHORT[k] for k in ORDER],rotation=40,ha='right',fontsize=7)
ax[1,2].set_ylabel('Δ macro F1'); ax[1,2].set_title('F  Job 1 cohort gain (M4)',fontsize=9.5,loc='left')
ax[1,2].legend(fontsize=6,frameon=False,loc='upper left')

for r in ax:
    for a_ in r:
        for s in ('top','right'): a_.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig('../figures/main_grid.png',dpi=200); fig.savefig('../figures/main_grid.pdf')
print('written')
