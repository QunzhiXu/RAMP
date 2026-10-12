"""Summarize the annotation audit and draw diagnostic figures; no relabeling."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/senosain_annotation_review_mpl')
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from audit import ANN,FX,HERE,sha

def save(name,d):d.to_csv(HERE/name,index=False)

md=pd.read_csv(ANN/'rough_annotation/senosain_140k_final_cell_annotations.csv.gz',dtype={'patient_id':str},keep_default_na=False)
mt=pd.read_csv(ANN/'senosain_140k_markers35.csv');mi=dict(zip(mt.marker,mt.raw35_column_zero_based))
x=np.load(ANN/'senosain_140k_raw35.npy',mmap_mode='r')
fcs=np.load(ANN/'senosain_140k_provider_fcs77.npy',mmap_mode='r')
ch=pd.read_csv(ANN/'senosain_140k_channels.csv');ci=dict(zip(ch.channel,ch.matrix_column_zero_based))
cell=pd.read_csv(HERE/'CELL_DIAGNOSTICS.csv.gz',dtype={'patient_id':str})
assert cell.source_cell_id.equals(md.source_cell_id)
original_meta=pd.read_csv(ANN/'senosain_140k_cell_metadata.csv.gz',dtype={'patient_id':str})
assign=pd.read_csv(ANN/'leiden_all34/cluster_assignments.csv.gz')
subassign=pd.read_csv(ANN/'rough_annotation/c06_recluster/c06_subcluster_assignments.csv.gz')
assert original_meta.sample_rank_within_patient.equals(md.sample_rank_within_patient)
assert assign.source_cell_id.equals(md.source_cell_id)
assert assign.cluster.equals(md.parent_cluster)
c06=md[md.parent_cluster.eq('C06')].merge(subassign[['sampled_row_zero_based','subcluster']],
        on='sampled_row_zero_based',validate='one_to_one',suffixes=('','_frozen'))
assert len(c06)==9778 and c06.subcluster.eq(c06.subcluster_frozen).all()

# Myeloid subclusters versus T cells: continuous raw values, not transformed values.
rows=[]
for patient,g in md.groupby('patient_id'):
    for cluster in ['C02','C09']:
        ps=g.parent_cluster.eq(cluster).to_numpy();ns=g.final_cell_type.isin(['CD4 T','CD8 T']).to_numpy();use=ps|ns
        if ps.sum()<20 or ns.sum()<20:continue
        idx=g.index.to_numpy()[use]
        for marker in ['CD11B','CD45','CD3','HLA_DR']:
            rows.append(dict(patient_id=patient,cluster=cluster,marker=marker,n_cluster=int(ps.sum()),n_T=int(ns.sum()),
                             auc_vs_T=roc_auc_score(ps[use],x[idx,mi[marker]])))
myeloid=pd.DataFrame(rows);save('MYELOID_CLUSTER_CONTINUOUS_AUC.csv',myeloid)

qc_channels=[]
for channel in ['Ir191Di','Ir193Di','Event_length','Width','Residual']:
    v=fcs[:,ci[channel]]
    qc_channels.append(dict(channel=channel,nonzero_fraction=float((v!=0).mean()),
         **dict(zip(['q0','q50','q90','q95','q99','maximum'],np.quantile(v,[0,.5,.9,.95,.99,1]).tolist()))))
save('QC_CHANNEL_AVAILABILITY.csv',pd.DataFrame(qc_channels))

partitions=[]
for scope,mask in [('all140k',np.ones(len(cell),bool)),('story28_56k',cell.in_story28),('candidate28k',cell.in_candidate28k)]:
    c=cell[mask & cell.cluster.eq('C02')]
    states={'CD45 low':~c.CD45_high,'T rule':c.T_support,'Myeloid rule':c.Myeloid_support,
            'CD45 high / CD3 low / CD11B low':c.CD45_high & ~c.CD3_high & ~c.CD11B_high}
    assert np.all(np.column_stack(list(states.values())).sum(axis=1)==1)
    for label,s in states.items():partitions.append(dict(scope=scope,diagnostic_group=label,n=int(s.sum()),total=len(c),fraction=s.mean()))
save('C02_DIAGNOSTIC_PARTITIONS.csv',pd.DataFrame(partitions))

# Locate the same issue in the existing original Lung benchmark, without changing it.
orig_path=FX/'cross_platform_trainability_v2_20260919/data/real_inputs_v0_3_r2/lung_cancer/metadata.csv.gz'
orig_hash=sha(orig_path)
orig=pd.read_csv(orig_path,dtype=str);sen=orig[orig.dataset.eq('Senosain CyTOF')]
j=sen.merge(md[['source_cell_id','parent_cluster','subcluster','final_cell_type','qc_recommendation']],
            on='source_cell_id',how='left',validate='one_to_one')
assert len(j)==20000 and j.final_cell_type.notna().all()
assert j.celltype.eq(j.final_cell_type).all()
save('ORIGINAL_LUNG_BENCHMARK_ANNOTATION_COUNTS.csv',j.groupby(['parent_cluster','final_cell_type','qc_recommendation']).size().rename('cells').reset_index())

# A review roster flags low support; it is not an exclusion list.
pg=pd.read_csv(HERE/'PATIENT_CLUSTER_COEXPRESSION.csv',dtype={'patient_id':str})
auc=pd.read_csv(HERE/'PATIENT_CONTINUOUS_MARKER_AUC.csv',dtype={'patient_id':str})
cd4=pg[(pg.alpha==.5)&pg.cluster.eq('C00')&pg.n.ge(20)].copy()
cd4=cd4.merge(auc[auc.test.eq('CD4_CD4_vs_CD8')][['patient_id','auc','in_story28']],on='patient_id',how='left')
cd4['review_flag']=(cd4.CD4_support.lt(.25)|cd4.auc.lt(.8))
save('CD4_PATIENT_REVIEW_ROSTER.csv',cd4[['patient_id','n','in_story28','T_support','CD4_high','CD4_support','CD8_support','auc','review_flag']])
profiles=[]
for patient in cd4.loc[cd4.review_flag,'patient_id']:
    for cl in ['C00','C03','C09']:
        idx=md.index[md.patient_id.eq(patient)&md.parent_cluster.eq(cl)].to_numpy()
        for marker in ['CD45','CD3','CD4','CD8','CD11B']:
            v=x[idx,mi[marker]]
            profiles.append(dict(patient_id=patient,cluster=cl,marker=marker,n=len(idx),zero_fraction=float((v==0).mean()),
                **dict(zip(['q10','q25','q50','q75','q90'],np.quantile(v,[.1,.25,.5,.75,.9]).tolist()))))
save('FLAGGED_PATIENT_RAW_MARKER_PROFILES.csv',pd.DataFrame(profiles))

# Threshold sensitivity and patient variability, deliberately shown together.
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':11,
                     'axes.titleweight':'bold','axes.labelsize':9,'pdf.fonttype':42,'ps.fonttype':42})
fig,axes=plt.subplots(2,2,figsize=(14,10),layout='constrained')
r=pd.read_csv(HERE/'RAW_PATIENT_PERCENTILE_MEDIAN.csv').set_index('cluster')
clusters=['C00','C03','C01','C05','C17','C02','C09','C11','C04','C06-S01','C10','C16']
markers=['CD45','CD3','CD4','CD8','CD11B','CD56','CD31','EPCAM','CYTOKERATIN','CK7','VIMENTIN','CD24']
ax=axes[0,0];im=ax.imshow(r.loc[clusters,markers],aspect='auto',vmin=0,vmax=1,cmap='viridis')
ax.set_yticks(range(len(clusters)),clusters);ax.set_xticks(range(len(markers)),[m.replace('CYTOKERATIN','CK') for m in markers],rotation=60,ha='right')
ax.set_title('A  Raw expression support across clusters',loc='left')
fig.colorbar(im,ax=ax,shrink=.65,label='Median within-patient percentile')
gt=pd.read_csv(HERE/'GROUP_COEXPRESSION.csv')
expected={'C00':'CD4_support','C03':'CD8_support','C01':'Epithelial_support','C05':'Epithelial_support',
          'C17':'Epithelial_support','C02':'Myeloid_support','C09':'Myeloid_support','C11':'Endothelial_support'}
ax=axes[0,1]
for i,(cl,rule) in enumerate(expected.items()):
    sub=gt[(gt.grouping=='cluster')&(gt.scope=='all140k')&(gt.group==cl)]
    vals=sub[rule].to_numpy();mid=sub.loc[sub.alpha.eq(.5),rule].iloc[0]
    ref=gt[(gt.grouping=='cluster')&(gt.scope=='candidate28k')&(gt.group==cl)&gt.alpha.eq(.5)][rule].iloc[0]
    ax.plot([vals.min(),vals.max()],[i,i],color='#50687A',lw=3)
    ax.scatter(mid,i,color='#233F55',s=38,zorder=3)
    ax.scatter(ref,i,facecolors='none',edgecolors='#D05A43',s=68,zorder=4)
ax.set_yticks(range(len(expected)),list(expected));ax.invert_yaxis();ax.set_xlim(0,1)
ax.set_xlabel('Fraction meeting the corresponding diagnostic rule')
ax.set_title('B  Rule support and threshold sensitivity',loc='left')
ax.legend(handles=[Line2D([0],[0],color='#50687A',lw=3,label='All cells: threshold 0.4–0.6'),
                   Line2D([0],[0],marker='o',color='#233F55',lw=0,label='All cells: midpoint threshold'),
                   Line2D([0],[0],marker='o',markerfacecolor='none',markeredgecolor='#D05A43',lw=0,label='Candidate 28k: midpoint')],fontsize=8,loc='upper left')
ax=axes[1,0];tests=auc.test.unique().tolist();rng=np.random.default_rng(42)
labels={'CD3_T_vs_Myeloid':'CD3: T / myeloid','CD4_CD4_vs_CD8':'CD4: CD4 T / CD8 T','CD8_CD8_vs_CD4':'CD8: CD8 T / CD4 T',
        'CD11B_Myeloid_vs_T':'CD11B: myeloid / T','CD45_immune_vs_epithelial':'CD45: immune / epithelial',
        'Epithelial_panel_epithelial_vs_immune':'EPI panel: epithelial / immune','CD31_Endothelial_vs_other_named':'CD31: endothelial / other'}
for i,t in enumerate(tests):
    z=auc[auc.test==t];jit=rng.uniform(-.16,.16,len(z));ax.scatter(z.auc,i+jit,c=np.where(z.in_story28,'#D05A43','#B3BDC6'),s=13,alpha=.8)
    ax.plot([z.auc.median()]*2,[i-.27,i+.27],color='black',lw=2)
ax.axvline(.5,color='gray',ls=':',lw=1);ax.set_xlim(0,1);ax.set_yticks(range(len(tests)),[labels[t] for t in tests]);ax.invert_yaxis()
ax.set_xlabel('Within-patient marker–label AUROC (not prediction accuracy)')
ax.set_title('C  Continuous evidence; one point per patient',loc='left')
ax.legend(handles=[Line2D([0],[0],marker='o',color='#D05A43',lw=0,label='Original 28 patients'),Line2D([0],[0],marker='o',color='#B3BDC6',lw=0,label='Other patients')],loc='lower left',fontsize=8)
q=pd.read_csv(HERE/'CLUSTER_QC.csv');ax=axes[1,1]
qms=['Event_length','Width','DNA191_193','total34','total33_without_MDM2'];qcs=['C06-S03','C14','C16']
mat=q.pivot(index='cluster',columns='metric',values='median_patient_percentile').loc[qcs,qms]
im=ax.imshow(mat,aspect='auto',vmin=0,vmax=1,cmap='viridis')
for y in range(3):
    for xx in range(5):ax.text(xx,y,f'{mat.iloc[y,xx]:.2f}',ha='center',va='center',color='white' if mat.iloc[y,xx]<.55 else 'black',fontweight='bold')
ax.set_yticks(range(3),qcs);ax.set_xticks(range(5),['Event length','Width','DNA*','Total signal','Total excluding\nMDM2'],rotation=25,ha='right')
ax.set_title('D  QC evidence must use more than total signal',loc='left')
ax.set_xlabel('*Each DNA channel is >90% zero in supplied data.')
fig.colorbar(im,ax=ax,shrink=.65,label='Median within-patient percentile')
for ax in axes.flat:
    if ax not in [axes[0,0],axes[1,1]]:
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.15)
fig.savefig(HERE/'ANNOTATION_REVIEW_OVERVIEW.png',dpi=190,bbox_inches='tight');fig.savefig(HERE/'ANNOTATION_REVIEW_OVERVIEW.pdf',bbox_inches='tight');plt.close(fig)

# Focused evidence for C02; show all patients rather than a hand-picked donor.
fig,axes=plt.subplots(1,3,figsize=(13,4.3),layout='constrained')
part=pd.DataFrame(partitions);ax=axes[0];scopes=['all140k','candidate28k'];colors=['#B0B5BC','#487EAC','#D4A43D','#9B82B9']
left=np.zeros(2)
for color,label in zip(colors,part.diagnostic_group.unique()):
    vals=np.array([part[(part.scope==s)&(part.diagnostic_group==label)].fraction.iloc[0] for s in scopes])
    ax.barh([0,1],vals,left=left,color=color,label=label,height=.5)
    for k,v in enumerate(vals):ax.text(left[k]+v/2,k,f'{v:.0%}',ha='center',va='center',fontsize=9)
    left+=vals
ax.set_yticks([0,1],['All C02\n(n=14,963)','Candidate C02\n(n=3,171)']);ax.invert_yaxis();ax.set_xlim(0,1)
ax.set_title('C02 contains multiple diagnostic patterns',fontsize=10)
ax.legend(loc='lower left',bbox_to_anchor=(0,-.55),fontsize=8,frameon=False)
ax.set_xlabel('Mutually exclusive diagnostic fractions')
ax=axes[1];pairs=myeloid[myeloid.marker=='CD11B'].pivot(index='patient_id',columns='cluster',values='auc_vs_T').dropna()
for _,v in pairs.iterrows():ax.plot([0,1],v[['C02','C09']],color='#BEC5CD',lw=.6,alpha=.7)
for i,cl in enumerate(['C02','C09']):
    vals=myeloid[(myeloid.marker=='CD11B')&myeloid.cluster.eq(cl)].auc_vs_T
    ax.scatter(np.full(len(vals),i),vals,color=['#B88D22','#1D7B76'][i],s=15,alpha=.65)
    ax.plot([i-.15,i+.15],[vals.median()]*2,color='black',lw=2)
ax.set_xticks([0,1],['C02','C09']);ax.set_xlim(-.4,1.4);ax.set_ylim(0,1);ax.axhline(.5,color='gray',ls=':')
ax.set_ylabel('CD11B AUROC versus T cells within patient');ax.set_title('Continuous marker support differs',fontsize=10)
ax=axes[2]
for i,cl in enumerate(['C02','C09']):
    z=pg[(pg.alpha==.5)&pg.cluster.eq(cl)&pg.n.ge(20)]
    ax.scatter(z.Myeloid_support,np.full(len(z),i)+rng.uniform(-.12,.12,len(z)),s=18,c=['#B88D22','#1D7B76'][i],alpha=.7)
    ax.plot([z.Myeloid_support.median()]*2,[i-.2,i+.2],color='black',lw=2)
ax.set_yticks([0,1],['C02','C09']);ax.set_xlim(0,1);ax.set_ylim(-.5,1.5);ax.invert_yaxis()
ax.set_xlabel('CD45-high / CD3-low / CD11B-high fraction');ax.set_title('The difference spans many patients',fontsize=10)
for ax in axes:ax.spines[['top','right']].set_visible(False)
fig.savefig(HERE/'C02_MYELOID_REVIEW.png',dpi=190,bbox_inches='tight');fig.savefig(HERE/'C02_MYELOID_REVIEW.pdf',bbox_inches='tight');plt.close(fig)

summary={'original_lung_senosain_n':len(j),'original_lung_senosain_labels_match_new_annotation':True,
         'annotation_matches_frozen_parent_and_subclusters':True,'sample_ranks_match_original_metadata':True,
         'DNA_sum_nonzero_fraction':float(((fcs[:,ci['Ir191Di']]+fcs[:,ci['Ir193Di']])>0).mean()),
         'original_lung_C02_n':int(j.parent_cluster.eq('C02').sum()),
         'original_lung_Myeloid_n':int(j.final_cell_type.eq('Myeloid').sum()),
         'candidate28k_C02_n':int((cell.in_candidate28k&cell.cluster.eq('C02')).sum()),
         'candidate28k_Myeloid_n':int((cell.in_candidate28k&cell.final_cell_type.eq('Myeloid')).sum()),
         'CD4_diagnostic_review_flag_patients_all70':int(cd4.review_flag.sum()),
         'CD4_diagnostic_review_flag_patients_story28':cd4.loc[cd4.review_flag & cd4.in_story28.fillna(False),'patient_id'].tolist(),
         'QC_channel_availability':qc_channels,'marker_auc_medians':auc.groupby('test').auc.median().to_dict(),
         'origin_metadata_sha256':orig_hash,'followup_script_sha256':sha(__file__)}
assert sha(orig_path)==orig_hash
manifest=json.loads((HERE/'INPUT_MANIFEST.json').read_text())
assert all(sha(p)==h for p,h in manifest['inputs_sha256'].items())
summary['all_original_inputs_still_unchanged']=True
(HERE/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
