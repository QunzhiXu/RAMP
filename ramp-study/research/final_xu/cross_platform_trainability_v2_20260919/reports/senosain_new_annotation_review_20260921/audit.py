"""Read-only review of the frozen Senosain 140k annotations."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/senosain_annotation_review_mpl')
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
FX = HERE.parents[2]
ANN = FX / 'raw_data/lung_cancer/Senosain_2025_CyTOF_and_omics/annotation'
HASHES = {}
CHECKS = {}
SEED = 20260921
LINEAGE = ['CD45', 'CD3', 'CD4', 'CD8', 'CD11B', 'CD56', 'CD31',
           'EPCAM', 'CYTOKERATIN', 'CK7', 'VIMENTIN', 'CD24', 'HLA_DR', 'KI67']

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def track(p):
    p=Path(p);HASHES[str(p)]=sha(p);return p

def save(name, data):
    data.to_csv(HERE/name,index=False)

def gates(p):
    epi=p['EPCAM'] | p['CYTOKERATIN'] | p['CK7']
    t=p['CD45'] & p['CD3']
    return {
        'CD45_high':p['CD45'], 'CD3_high':p['CD3'],
        'CD4_high':p['CD4'], 'CD8_high':p['CD8'],
        'CD11B_high':p['CD11B'], 'CD56_high':p['CD56'],
        'EPCAM_high':p['EPCAM'],'CK_high':p['CYTOKERATIN'],'CK7_high':p['CK7'],
        'T_support':t,
        'CD4_support':t & p['CD4'], 'CD8_support':t & p['CD8'],
        'T_double_positive':t & p['CD4'] & p['CD8'],
        'CD4_exclusive':t & p['CD4'] & ~p['CD8'],
        'CD8_exclusive':t & p['CD8'] & ~p['CD4'],
        'Myeloid_support':p['CD45'] & ~p['CD3'] & p['CD11B'],
        'CD56_nonT_CD11Blow':p['CD45'] & ~p['CD3'] & p['CD56'] & ~p['CD11B'],
        'Epithelial_any':epi,
        'Epithelial_support':epi & ~p['CD45'],
        'Immune_epithelial_cohigh':p['CD45'] & epi,
        'T_epithelial_cohigh':t & epi,
        'Endothelial_support':p['CD31'] & ~p['CD45'],
        'Endothelial_strict':p['CD31'] & ~p['CD45'] & ~epi,
        'Mesenchymal_strict':p['VIMENTIN'] & ~p['CD45'] & ~p['CD31'] & ~epi,
        'B_like_broad_only':p['CD45'] & ~p['CD3'] & ~p['CD11B'],
        'KI67_high':p['KI67'], 'HLA_DR_high':p['HLA_DR'],
    }

def main():
    md=pd.read_csv(track(ANN/'senosain_140k_cell_metadata.csv.gz'),dtype={'patient_id':str})
    a=pd.read_csv(track(ANN/'rough_annotation/senosain_140k_final_cell_annotations.csv.gz'),
                  dtype={'patient_id':str},keep_default_na=False)
    mt=pd.read_csv(track(ANN/'senosain_140k_markers35.csv'))
    ch=pd.read_csv(track(ANN/'senosain_140k_channels.csv'))
    x=np.load(track(ANN/'senosain_140k_raw35.npy'),mmap_mode='r')
    fcs=np.load(track(ANN/'senosain_140k_provider_fcs77.npy'),mmap_mode='r')
    kl=np.load(track(ANN/'kmeanslow_pca_umap/senosain_140k_kmeanslow35.float32.npy'),mmap_mode='r')
    markers=mt.marker.tolist(); mi={v:i for i,v in enumerate(markers)}
    mapping=pd.read_csv(track(ANN/'rough_annotation/final_cluster_subcluster_annotation_map.csv'),keep_default_na=False)
    for f in ['senosain_140k_manifest.json','kmeanslow_pca_umap/run_manifest.json',
              'rough_annotation/final_annotation_manifest.json','leiden_all34/cluster_assignments.csv.gz',
              'rough_annotation/c06_recluster/c06_subcluster_assignments.csv.gz']:
        track(ANN/f)
    donors=pd.read_csv(track(FX/'story_lung_HLA_ABC/data_umap/data/SELECTED_DONORS.csv'),dtype={'patient_id':str})
    selected=set(donors.loc[donors.dataset.eq('Senosain CyTOF') & donors.selected.astype(str).str.lower().eq('true'),'patient_id'])
    assert len(selected)==28
    assert x.shape==(140000,35) and fcs.shape==(140000,77)
    assert np.isfinite(x).all() and np.isfinite(kl).all()
    assert md.source_cell_id.is_unique and a.source_cell_id.is_unique
    assert md.source_cell_id.equals(a.source_cell_id)
    assert md.patient_id.equals(a.patient_id)
    assert np.array_equal(md.sampled_row_zero_based,np.arange(len(md)))
    assert np.array_equal(a.sampled_row_zero_based,np.arange(len(md)))
    assert md.groupby('patient_id').size().eq(2000).all()
    assert np.array_equal(x,fcs[:,mt.fcs77_column_zero_based.to_numpy()])
    joined=a.merge(mapping,on=['parent_cluster','subcluster'],suffixes=('','_map'),validate='many_to_one',how='left')
    for c in ['final_cell_type','annotation_confidence','qc_recommendation','lineage_reference_use']:
        assert joined[c].equals(joined[c+'_map'])
    CHECKS.update(n_cells=len(md),n_patients=md.patient_id.nunique(),matrix_row_identity=True,
                  raw35_matches_fcs77=True,annotation_map_matches=True)
    md=md.merge(a[['source_cell_id','parent_cluster','subcluster','final_cell_type','qc_recommendation',
                  'annotation_confidence','lineage_reference_use']],on='source_cell_id',validate='one_to_one',sort=False)
    md['cluster']=md.parent_cluster+np.where(md.subcluster.ne(''),'-'+md.subcluster,'')
    md['in_story28']=md.patient_id.isin(selected)
    chosen=(md.loc[md.in_story28 & md.qc_recommendation.eq('keep')]
            .sort_values(['patient_id','sample_rank_within_patient']).groupby('patient_id',sort=False).head(1000))
    assert len(chosen)==28000 and chosen.groupby('patient_id').size().eq(1000).all()
    md['in_candidate28k']=md.source_cell_id.isin(chosen.source_cell_id)
    save('CANDIDATE_REFERENCE_IDS_REVIEW_ONLY.csv.gz',md.loc[md.in_candidate28k,
            ['source_cell_id','patient_id','sample_rank_within_patient','cluster','final_cell_type','qc_recommendation']])
    save('CELL_INVENTORY.csv',md.groupby(['final_cell_type','qc_recommendation'],sort=False).agg(
        all140k=('source_cell_id','size'),story28_all56k=('in_story28','sum'),candidate28k=('in_candidate28k','sum')).reset_index())
    save('PATIENT_CELL_COUNTS.csv',md.groupby(['patient_id','in_story28','cluster','final_cell_type']).agg(
        all_cells=('source_cell_id','size'),candidate_cells=('in_candidate28k','sum')).reset_index())
    ranks=np.zeros(x.shape,dtype=np.float32)
    thresholds=[]
    pos={alpha:{m:np.zeros(len(md),bool) for m in LINEAGE} for alpha in [.4,.5,.6]}
    patient_groups=list(md.groupby('patient_id',sort=False).groups.items())
    for n,(patient,idx) in enumerate(patient_groups,1):
        idx=np.asarray(idx);xp=np.asarray(x[idx],dtype=float)
        ranks[idx]=(rankdata(xp,axis=0,method='average')-.5)/len(idx)
        for m in LINEAGE:
            v=xp[:,mi[m]]; med=float(np.median(v)); cofactor=max(med if med>0 else float(np.mean(v)),1e-6)
            z=np.arcsinh(v/cofactor)
            km=KMeans(n_clusters=2,n_init=20,random_state=SEED,max_iter=500).fit(z[:,None])
            lo,hi=np.sort(km.cluster_centers_.ravel())
            for alpha in pos:
                threshold=float(lo+alpha*(hi-lo));h=z>threshold;pos[alpha][m][idx]=h
                thresholds.append(dict(patient_id=patient,marker=m,alpha=alpha,cofactor=cofactor,
                                       low_center=lo,high_center=hi,threshold_arcsinh=threshold,fraction_high=h.mean()))
        if n%10==0:print(f'Gates and raw ranks: {n}/70 patients',flush=True)
    save('DIAGNOSTIC_THRESHOLDS.csv',pd.DataFrame(thresholds))
    # Raw rank summaries are independent of archived KMEANSLOW coordinates.
    for name,matrix in [('RAW_PATIENT_PERCENTILE_MEDIAN',ranks),('KMEANSLOW_MEDIAN',kl)]:
        frame=pd.DataFrame(matrix,columns=markers);frame['cluster']=md.cluster
        save(name+'.csv',frame.groupby('cluster').median().reset_index())
    all_gate_tables=[];patient_gate_tables=[]
    for alpha,p in pos.items():
        gt=gates(p);frame=md[['source_cell_id','patient_id','cluster','final_cell_type','in_story28','in_candidate28k']].copy()
        for k,v in gt.items():frame[k]=v
        if alpha==.5:save('CELL_DIAGNOSTICS.csv.gz',frame)
        for scope,mask in [('all140k',np.ones(len(md),bool)),('story28_56k',md.in_story28),('candidate28k',md.in_candidate28k)]:
            for key in ['cluster','final_cell_type']:
                g=frame.loc[mask].groupby(key)
                d=g[list(gt)].mean();d.insert(0,'n',g.size());d=d.reset_index().rename(columns={key:'group'})
                d.insert(0,'grouping',key);d.insert(0,'scope',scope);d.insert(0,'alpha',alpha)
                all_gate_tables.append(d)
        g=frame.groupby(['patient_id','cluster','final_cell_type'])
        d=g[list(gt)].mean();d.insert(0,'n',g.size());d=d.reset_index();d.insert(0,'alpha',alpha)
        patient_gate_tables.append(d)
    gate_table=pd.concat(all_gate_tables,ignore_index=True)
    save('GROUP_COEXPRESSION.csv',gate_table)
    patient_gates=pd.concat(patient_gate_tables,ignore_index=True)
    save('PATIENT_CLUSTER_COEXPRESSION.csv',patient_gates)
    # Continuous marker/annotation separation, within each patient, at least 20 cells/side.
    ty=md.final_cell_type
    tests=[('CD3_T_vs_Myeloid','CD3',ty.isin(['CD4 T','CD8 T']),ty.eq('Myeloid')),
           ('CD4_CD4_vs_CD8','CD4',ty.eq('CD4 T'),ty.eq('CD8 T')),
           ('CD8_CD8_vs_CD4','CD8',ty.eq('CD8 T'),ty.eq('CD4 T')),
           ('CD11B_Myeloid_vs_T','CD11B',ty.eq('Myeloid'),ty.isin(['CD4 T','CD8 T'])),
           ('CD45_immune_vs_epithelial','CD45',ty.isin(['CD4 T','CD8 T','Myeloid']),ty.eq('Tumor/Epithelial')),
           ('Epithelial_panel_epithelial_vs_immune','EPI_MAX',ty.eq('Tumor/Epithelial'),ty.isin(['CD4 T','CD8 T','Myeloid'])),
           ('CD31_Endothelial_vs_other_named','CD31',ty.eq('Endothelial'),ty.isin(['CD4 T','CD8 T','Myeloid','Tumor/Epithelial']))]
    aucs=[]
    for test,marker,positive,negative in tests:
        val=ranks[:,[mi[m] for m in ['EPCAM','CYTOKERATIN','CK7']]].max(axis=1) if marker=='EPI_MAX' else x[:,mi[marker]]
        for patient,idx in patient_groups:
            idx=np.asarray(idx);ps=positive.to_numpy()[idx];ns=negative.to_numpy()[idx]
            if min(ps.sum(),ns.sum())<20:continue
            use=ps|ns
            aucs.append(dict(test=test,marker=marker,patient_id=patient,in_story28=patient in selected,
                             n_positive=int(ps.sum()),n_negative=int(ns.sum()),auc=roc_auc_score(ps[use],val[idx][use])))
    save('PATIENT_CONTINUOUS_MARKER_AUC.csv',pd.DataFrame(aucs))
    # Donor/acquisition concentration: descriptive, not automatic exclusions.
    conc=[]
    for cluster,g in md.groupby('cluster'):
        cnt=g.patient_id.value_counts();dates=g.acquisition_date.value_counts()
        conc.append(dict(cluster=cluster,n=len(g),n_patients=len(cnt),max_patient=cnt.index[0],
                         max_patient_fraction=cnt.iloc[0]/len(g),top3_patient_fraction=cnt.iloc[:3].sum()/len(g),
                         max_acquisition_date=dates.index[0],max_acquisition_fraction=dates.iloc[0]/len(g),
                         in_story28_n=int(g.in_story28.sum()),candidate28k_n=int(g.in_candidate28k.sum())))
    save('CLUSTER_DONOR_CONCENTRATION.csv',pd.DataFrame(conc))
    ci=dict(zip(ch.channel,ch.matrix_column_zero_based))
    qvalues={m:np.asarray(fcs[:,ci[m]],dtype=float) for m in ['Event_length','Width','Residual','Center','Offset']}
    qvalues['DNA191_193']=np.asarray(fcs[:,ci['Ir191Di']]+fcs[:,ci['Ir193Di']],dtype=float)
    uniform=[i for m,i in mi.items() if m!='CD90']
    no_mdm2=[i for m,i in mi.items() if m not in ['CD90','MDM2']]
    qvalues['total34']=np.asarray(x[:,uniform].sum(axis=1),dtype=float)
    qvalues['total33_without_MDM2']=np.asarray(x[:,no_mdm2].sum(axis=1),dtype=float)
    qrank={};qhigh={};qraws=[];qaucs=[]
    for metric,v in qvalues.items():
        r=np.zeros(len(md));hi=np.zeros(len(md),bool)
        for patient,idx in patient_groups:
            idx=np.asarray(idx);r[idx]=(rankdata(v[idx],method='average')-.5)/len(idx)
            hi[idx]=v[idx]>np.quantile(v[idx],.95)
        qrank[metric]=r;qhigh[metric]=hi
        for cluster,g in md.groupby('cluster'):
            idx=g.index.to_numpy();qraws.append(dict(cluster=cluster,metric=metric,n=len(idx),
                median_raw=np.median(v[idx]),median_patient_percentile=np.median(r[idx]),fraction_above_patient_q95=hi[idx].mean()))
        for cluster in ['C06-S03','C14','C16']:
            for patient,idx in patient_groups:
                idx=np.asarray(idx);ps=md.cluster.iloc[idx].eq(cluster).to_numpy()
                ns=md.qc_recommendation.iloc[idx].eq('keep').to_numpy()
                if ps.sum()<10 or ns.sum()<50:continue
                use=ps|ns;qaucs.append(dict(cluster=cluster,metric=metric,patient_id=patient,n_cluster=int(ps.sum()),
                                           n_control=int(ns.sum()),auc=roc_auc_score(ps[use],v[idx][use])))
    save('CLUSTER_QC.csv',pd.DataFrame(qraws));save('PATIENT_QC_AUC.csv',pd.DataFrame(qaucs))
    qc=pd.DataFrame({'source_cell_id':md.source_cell_id,'patient_id':md.patient_id,'cluster':md.cluster})
    for metric in qvalues:
        qc[metric+'_patient_percentile']=qrank[metric];qc[metric+'_above_patient_q95']=qhigh[metric]
    save('CELL_QC_DIAGNOSTICS.csv.gz',qc)
    # Cell-level co-elevation of geometry, DNA and signal, avoiding a single-channel QC verdict.
    combos={'geometry_event_or_width':qhigh['Event_length']|qhigh['Width'],
            'geometry_and_DNA':(qhigh['Event_length']|qhigh['Width']) & qhigh['DNA191_193'],
            'geometry_and_total33':(qhigh['Event_length']|qhigh['Width']) & qhigh['total33_without_MDM2'],
            'geometry_DNA_total33':(qhigh['Event_length']|qhigh['Width']) & qhigh['DNA191_193'] & qhigh['total33_without_MDM2']}
    d=md[['cluster']].copy()
    for k,v in combos.items():d[k]=v
    save('CLUSTER_QC_COELEVATION.csv',d.groupby('cluster').mean().reset_index())
    # Support per patient for each named group, not a selective reassignment.
    supports={'CD4 T':'CD4_support','CD8 T':'CD8_support','Myeloid':'Myeloid_support',
              'Tumor/Epithelial':'Epithelial_support','Endothelial':'Endothelial_support'}
    summaries=[]
    mid=patient_gates[patient_gates.alpha.eq(.5)]
    for cluster,g in mid.groupby('cluster'):
        cell_type=g.final_cell_type.iloc[0];expected=supports.get(cell_type)
        if expected is None:continue
        for scope,mask in [('all70',np.ones(len(g),bool)),('story28',g.patient_id.isin(selected))]:
            z=g.loc[mask & g.n.ge(20)]
            if len(z)==0:continue
            summaries.append(dict(cluster=cluster,cell_type=cell_type,expected_rule=expected,scope=scope,
                patients_with_at_least20=len(z),median_support=z[expected].median(),p10_support=z[expected].quantile(.1),
                minimum_support=z[expected].min(),patients_below_half=int(z[expected].lt(.5).sum())))
    save('PATIENT_SUPPORT_SUMMARY.csv',pd.DataFrame(summaries))
    unchanged={p:sha(p)==s for p,s in HASHES.items()};assert all(unchanged.values())
    CHECKS['all_tracked_inputs_unchanged']=True
    CHECKS['source_outcomes_or_method_performance_used']=False
    (HERE/'CHECKS.json').write_text(json.dumps(CHECKS,indent=2)+'\n')
    (HERE/'INPUT_MANIFEST.json').write_text(json.dumps({'inputs_sha256':HASHES,'seed':SEED,
        'threshold_fractions':[.4,.5,.6],'script_sha256':sha(__file__),
        'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__},indent=2)+'\n')
    print(json.dumps(CHECKS,indent=2),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
