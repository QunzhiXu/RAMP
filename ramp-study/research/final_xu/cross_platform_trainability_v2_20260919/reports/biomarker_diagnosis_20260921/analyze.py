"""Read-only diagnosis of marker informativeness, calibration and frozen J1 errors.

No new models, label edits, alignment changes or benchmark selections. All
patient/unit AUCs require >=5 positives and >=20 negatives among eligible J1
classes. Within-platform discrimination does not establish cross-platform
calibration or independent biological truth of provider labels.
"""
import os
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
os.environ['MPLCONFIGDIR']=str(HERE/'.mplconfig')
os.environ['MPLBACKEND']='Agg'
TASKS=['melanoma_story','hnscc_story']
METHODS=['ramp','unaligned','clipped_zscore','percentile','fastmnn','maxfuse','cycombine','cytonorm','harmony_direct','minmax']
EXPECTED={
 'melanoma_story': [('B lineage','CD20'),('CD8 T','CD8'),('Other T / CD8-','CD4'),
                     ('Endothelial','CD31'),('Myeloid','CD14'),('Myeloid','CD40')],
 'hnscc_story': [('B','CD20'),('B','CD21'),('CD4 T','CD4'),('CD8 T','CD8'),
                ('NK / gdT','CD56'),('NK / gdT','CD7'),('Plasma','CD38'),
                ('Myeloid','CD11b'),('Myeloid','CD11c'),('Myeloid','HLA-DR'),('Vascular','CD31')]
}


def save(name,rows):
    df=pd.DataFrame(rows);df.to_csv(HERE/name,index=False);return df


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    auc_rows=[];profiles=[];rank_rows=[];class_rows=[];confusion_rows=[];head_rows=[];shift_rows=[]
    sources={};max_score_error=0.;aligned_cache={};change_rows=[]
    inventory=[]
    for path in sorted((ROOT/'data/real_inputs_v0_3_r2').glob('*/manifest.json')):
        d=json.loads(path.read_text())
        for marker in d['markers']:inventory.append(dict(task=d['name'],marker=marker))
    save('MARKER_INVENTORY_SEVEN_BENCHMARKS.csv',inventory)
    for task in TASKS:
        inp=ROOT/'data/real_inputs_v0_3_r2'/task
        man=json.loads((inp/'manifest.json').read_text());markers=man['markers'];classes=man['classes']
        md=pd.read_csv(inp/'metadata.csv.gz',keep_default_na=False)
        X={'native':np.load(inp/'native.npy')}
        for method in ['ramp','unaligned','clipped_zscore','percentile']:
            rep=ROOT/f'representations/formal_v0_5/{task}/complete/{method}'
            ids=np.load(rep/'cell_ids.npy')
            assert np.array_equal(ids.astype(str),md.cell_id.to_numpy(str)),(task,method)
            X[method]=np.load(rep/'values.npy');assert X[method].shape==(len(md),len(markers))
            sources[str(rep/'values.npy')]=sha(rep/'values.npy')
        X['kmeanslow']=np.load(ROOT/f'representations/formal_v0_5/{task}/complete/ramp/kmeanslow.npy')
        for a in X.values():assert np.isfinite(a).all()
        for ds,g in md.groupby('dataset'):
            delta=np.abs(X['ramp'][g.index]-X['kmeanslow'][g.index])
            for j,marker in enumerate(markers):
                change_rows.append(dict(task=task,dataset=ds,marker=marker,
                    fraction_changed_gt1e_8=float(np.mean(delta[:,j]>1e-8)),
                    median_abs_change=float(np.median(delta[:,j])),q95_abs_change=float(np.quantile(delta[:,j],.95))))
        for file in ['metadata.csv.gz','native.npy','manifest.json']:sources[str(inp/file)]=sha(inp/file)
        eligible=md.celltype.isin(classes)
        aligned_cache[task]=(md,markers,X['ramp'])
        for (ds,unit),block in md[eligible].groupby(['dataset','unit_key'],sort=True):
            assert block.patient_id.nunique()==1
            ix=block.index.to_numpy();labels=block.celltype.to_numpy();patient=block.patient_id.iloc[0]
            ranks={rep:rankdata(X[rep][ix],axis=0) for rep in ['native','kmeanslow','ramp']}
            for j,marker in enumerate(markers):
                a=ranks['native'][:,j];b=ranks['ramp'][:,j]
                rho=float(np.corrcoef(a,b)[0,1]) if np.std(a)>0 and np.std(b)>0 else np.nan
                rank_rows.append(dict(task=task,dataset=ds,unit=unit,marker=marker,
                                      rank_spearman=rho,exact_rank_match=bool(np.array_equal(a,b))))
            for cl in classes:
                pos=labels==cl;np_=int(pos.sum());nn=len(pos)-np_
                if np_>=5:
                    for rep,arr in X.items():
                        q=np.quantile(arr[ix[pos]], [.25,.5,.75],axis=0)
                        for j,marker in enumerate(markers):
                            profiles.append(dict(task=task,dataset=ds,unit=unit,patient=patient,celltype=cl,
                                                 representation=rep,marker=marker,cells=np_,q25=q[0,j],median=q[1,j],q75=q[2,j]))
                if np_<5 or nn<20:continue
                for rep,rank in ranks.items():
                    auc=(rank[pos].sum(axis=0)-np_*(np_+1)/2)/(np_*nn)
                    for j,marker in enumerate(markers):
                        auc_rows.append(dict(task=task,dataset=ds,unit=unit,patient=patient,celltype=cl,
                                             representation=rep,marker=marker,positive_cells=np_,negative_cells=nn,auc=auc[j]))
        # Within-patient discrimination between the two poorly anchored nonimmune classes.
        if task=='hnscc_story':
            pairrows=[]
            subset=md[md.celltype.isin(['Stromal','Epithelial / tumor'])]
            for (ds,unit),g in subset.groupby(['dataset','unit_key']):
                y=g.celltype.eq('Stromal').to_numpy();n=int(y.sum());z=len(y)-n
                if min(n,z)<5:continue
                auc=(rankdata(X['native'][g.index],axis=0)[y].sum(0)-n*(n+1)/2)/(n*z)
                for j,m in enumerate(markers):pairrows.append(dict(task=task,dataset=ds,unit=unit,marker=m,auc=auc[j],stromal_cells=n,epithelial_cells=z))
            save('HNSCC_STROMAL_VS_EPITHELIAL_PER_UNIT.csv',pairrows)
        for method in METHODS:
            path=ROOT/f'config/formal_v0_5/{task}__{method}.json';cfg=json.loads(path.read_text())
            jobs=[j for j in cfg['jobs'] if j['task_type']=='J1' and 'primary' in j['analysis_scopes']]
            for job in jobs:
                res=json.loads((Path(cfg['output'])/'jobs'/job['id']/'RESULT.json').read_text())
                s=res['score'];assert s['classes']==classes
                cm=np.array(s['confusion_matrix']);sup=cm.sum(1);pred=cm.sum(0);tp=np.diag(cm)
                recall=np.divide(tp,sup,out=np.zeros_like(tp),where=sup>0)
                f1=np.divide(2*tp,sup+pred,out=np.zeros_like(tp),where=sup+pred>0)
                max_score_error=max(max_score_error,abs(f1.mean()-s['macro_f1']))
                base=dict(task=task,method=method,family=job['family'],source=job['source'],target=job['target'],seed=job['seed'])
                head_rows.append(dict(base,source_validation_f1=res['selection']['source_validation_score'],
                                      target_f1=s['macro_f1'],target_ba=s['balanced_accuracy'],
                                      chosen_scale=res['selection']['candidate']['scale']))
                for i,cl in enumerate(classes):
                    class_rows.append(dict(base,celltype=cl,f1=f1[i],recall=recall[i],truth_fraction=sup[i],predicted_fraction=pred[i]))
                    for k,other in enumerate(classes):
                        confusion_rows.append(dict(base,truth=cl,prediction=other,probability=cm[i,k]/sup[i] if sup[i] else np.nan))
        print(task,'computed',flush=True)
    aucdf=save('MARKER_AUC_PER_UNIT.csv',auc_rows)
    # Average units within patient, then patients; currently each unit is one patient here.
    keys=['task','dataset','celltype','representation','marker']
    pa=aucdf.groupby(keys+['patient'],as_index=False).agg(auc=('auc','mean'))
    summary=pa.groupby(keys,as_index=False).agg(mean_patient_auc=('auc','mean'),patient_auc_sd=('auc','std'),patients=('patient','nunique'))
    summary.to_csv(HERE/'MARKER_AUC_SUMMARY.csv',index=False)
    prof=save('CLASS_MARKER_QUARTILES_PER_UNIT.csv',profiles)
    pp=prof.groupby(keys+['patient'],as_index=False)[['q25','median','q75']].mean()
    pm=pp.groupby(keys,as_index=False).agg(mean_patient_median=('median','mean'),patients=('patient','nunique'))
    pm.to_csv(HERE/'CLASS_MARKER_MEDIANS.csv',index=False)
    for task in TASKS:
        datasets=sorted(pm.loc[pm.task.eq(task),'dataset'].unique())
        for rep in ['kmeanslow','ramp','clipped_zscore','percentile']:
            q=pm[pm.task.eq(task)&pm.representation.eq(rep)].pivot(index=['celltype','marker'],columns='dataset',values='mean_patient_median')
            for (cl,marker),v in q.iterrows():
                shift_rows.append(dict(task=task,representation=rep,celltype=cl,marker=marker,
                     dataset_a=datasets[0],dataset_b=datasets[1],median_a=v[datasets[0]],median_b=v[datasets[1]],
                     signed_b_minus_a=v[datasets[1]]-v[datasets[0]]))
    save('CLASS_MARKER_PLATFORM_SHIFTS.csv',shift_rows)
    save('RAMP_CHANGE_FROM_KMEANSLOW.csv',change_rows)
    rankdf=save('WITHIN_UNIT_RANK_CHECK.csv',rank_rows)
    cd=save('J1_PER_CLASS_PER_JOB.csv',class_rows)
    cd.groupby(['task','method','source','target','celltype'],as_index=False)[['f1','recall','truth_fraction','predicted_fraction']].mean().to_csv(HERE/'J1_CLASS_DIRECTION_MEANS.csv',index=False)
    cf=save('J1_CONFUSION_PER_JOB.csv',confusion_rows)
    cf.groupby(['task','method','source','target','truth','prediction'],as_index=False).probability.mean().to_csv(HERE/'J1_CONFUSION_DIRECTION_MEANS.csv',index=False)
    hd=save('J1_SOURCE_TARGET_SCORES.csv',head_rows)
    hd.groupby(['task','method','source','target'],as_index=False)[['source_validation_f1','target_f1','target_ba']].mean().to_csv(HERE/'J1_SOURCE_TARGET_MEANS.csv',index=False)
    select=pd.DataFrame([dict(task=t,celltype=c,marker=m) for t,pairs in EXPECTED.items() for c,m in pairs])
    expected=summary[summary.representation.eq('native')].merge(select,on=['task','celltype','marker'])
    expected.to_csv(HERE/'EXPECTED_MARKER_AUC.csv',index=False)
    plot(expected,pm,aligned_cache)
    check=dict(primary_f1_reproduction_max_error=max_score_error,finite_rank_pairs=int(rankdf.rank_spearman.notna().sum()),
               minimum_within_unit_spearman=float(rankdf.rank_spearman.min()),
               exact_rank_match_fraction=float(rankdf.exact_rank_match.mean()),minimum_positive_cells=5,minimum_negative_cells=20,
               no_refit=True,no_formal_outputs_changed=True,input_sha256=sources)
    (HERE/'CHECKS.json').write_text(json.dumps(check,indent=2)+'\n')
    print(json.dumps({k:v for k,v in check.items() if k!='input_sha256'},indent=2))


def plot(expected,pm,cache):
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titleweight':'bold','axes.labelweight':'bold'})
    fig,axes=plt.subplots(1,2,figsize=(12,5.9),gridspec_kw={'width_ratios':[1,1.12]})
    names={'melanoma_story':'Melanoma story','hnscc_story':'HNSCC story'}
    for ax,task in zip(axes,TASKS):
        pairs=EXPECTED[task];df=expected[expected.task.eq(task)]
        datasets=sorted(df.dataset.unique());mat=np.array([[df[(df.dataset==d)&(df.celltype==c)&(df.marker==m)].mean_patient_auc.iloc[0] for d in datasets] for c,m in pairs])
        im=ax.imshow(mat,vmin=0,vmax=1,cmap='YlGnBu',aspect='auto')
        ax.set_title(names[task],pad=12)
        ax.set_xticks([0,1],['CITE-seq' if task=='melanoma_story' else 'ADT','IMC' if task=='melanoma_story' else 'CODEX'],fontweight='bold')
        ax.set_yticks(range(len(pairs)),[f'{c} — {m}' for c,m in pairs])
        for i,row in enumerate(mat):
            for j,x in enumerate(row):ax.text(j,i,f'{x:.2f}',ha='center',va='center',fontweight='bold',color='white' if x>.83 else '#202020')
        for spine in ax.spines.values():spine.set_visible(False)
        ax.tick_params(length=0)
    fig.subplots_adjust(left=.2,right=.89,wspace=.95,bottom=.16,top=.9)
    cax=fig.add_axes([.92,.2,.017,.6]);fig.colorbar(im,cax=cax,label='Mean patient AUROC (native marker vs cell label)')
    fig.text(.5,.055,'Same marker, different within-platform discrimination. Each patient/unit is weighted equally.',ha='center',fontsize=10)
    fig.savefig(HERE/'marker_discrimination.png',dpi=220);fig.savefig(HERE/'marker_discrimination.pdf');plt.close(fig)
    examples=[('melanoma_story','B lineage','CD20'),('melanoma_story','CD8 T','CD8'),
              ('hnscc_story','B','CD21'),('hnscc_story','CD8 T','CD8')]
    fig,axes=plt.subplots(1,4,figsize=(12.8,3.5))
    for ax,(task,cl,marker) in zip(axes,examples):
        q=pm[(pm.task==task)&(pm.celltype==cl)&(pm.marker==marker)&(pm.representation=='ramp')]
        # Show per-patient class medians, not pooled-cell quartiles.
        raw=pd.read_csv(HERE/'CLASS_MARKER_QUARTILES_PER_UNIT.csv')
        part=raw[(raw.task==task)&(raw.celltype==cl)&(raw.marker==marker)&(raw.representation=='ramp')]
        for i,ds in enumerate(sorted(q.dataset.unique())):
            y=part[part.dataset==ds].groupby('patient')['median'].mean().to_numpy()
            jitter=np.random.default_rng(1).uniform(-.1,.1,len(y));ax.scatter(np.full(len(y),i)+jitter,y,s=18,alpha=.75,color=['#2166ac','#e08214'][i])
            ax.plot([i-.16,i+.16],[np.median(y)]*2,color='black',lw=1.6)
        ax.set_xticks([0,1],['CITE-seq' if task=='melanoma_story' else 'ADT','IMC' if task=='melanoma_story' else 'CODEX'])
        ax.set_title(f'{names[task]}\n{cl}: {marker}',fontsize=10)
        ax.spines[['top','right']].set_visible(False)
        ax.set_ylabel('Patient median, RAMP value' if ax==axes[0] else '')
    fig.tight_layout();fig.savefig(HERE/'ramp_positive_class_values.png',dpi=220);fig.savefig(HERE/'ramp_positive_class_values.pdf');plt.close(fig)


if __name__=='__main__':main()
