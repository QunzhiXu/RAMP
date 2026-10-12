"""Read-only formal inputs; fixed k=30 KNN CD19 test and strict proxy tasks."""
from runtime import *
import argparse, time
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.cluster import KMeans

MARKERS=['CD117','CD11B','CD11C','CD123','CD14','CD3','CD33','CD34','CD38','CD4','CD45','CD45RA','CD61','CD8']
DATASETS=['BM CODEX','BM CyTOF','Triana AbSeq']
SLUGS=['bm_codex','bm_cytof','triana_abseq']
IDENTITY=['cell_id','dataset','donor','unit_key']
RESULTS=BASE/'results'
PRIMARY='RAMP_KNN30_pooled'

def load():
    frames=[]
    for ds,slug in zip(DATASETS,SLUGS):
        f=pd.read_csv(STORY/f'sampled_data/{slug}/source_input_15.csv.gz',float_precision='round_trip')
        assert f.dataset.eq(ds).all()
        frames.append(f[IDENTITY+MARKERS+['CD19']])
    d=pd.concat(frames,ignore_index=True)
    meta=d[IDENTITY].copy()
    assert len(d)==88031 and not meta.duplicated(['dataset','cell_id']).any()
    source=meta.dataset.ne('BM CODEX').to_numpy(); query=~source
    # The failed CODEX CD19 values are never retained in the predictor's raw matrix.
    raw=d[MARKERS].to_numpy(float)
    source_cd19=d.loc[source,'CD19'].to_numpy(float)
    baseline=pd.read_csv(STORY/'alignment/output/without_cd19/kmeanslow_baseline.csv.gz',float_precision='round_trip')
    aligned=pd.read_csv(STORY/'alignment/output/without_cd19/ramp_aligned.csv.gz',float_precision='round_trip')
    for x in [baseline,aligned]: assert x[IDENTITY].astype(str).equals(meta.astype(str))
    y=np.load(STORY/'alignment/output/with_predicted_cd19/kmeanslow_baseline.npy')[source,14].astype(float)
    ranks=np.empty(len(y)); unit=meta.loc[source,'unit_key'].to_numpy()
    for u in np.unique(unit):
        ix=unit==u; ranks[ix]=rankdata(source_cd19[ix])/(ix.sum()+1)
    return meta,raw,baseline[MARKERS].to_numpy(float),aligned[MARKERS].to_numpy(float),np.column_stack([y,ranks]),source,query

def predict(x, ys, meta, source, query, equal=True):
    scaler=StandardScaler().fit(x[source])
    xs=scaler.transform(x[source]); xq=scaler.transform(x[query])
    ds=meta.loc[source,'dataset'].to_numpy()
    def one(keep):
        nn=NearestNeighbors(n_neighbors=30,metric='euclidean',algorithm='brute',n_jobs=1).fit(xs[keep])
        source_y=ys[keep]
        output=[]; contribution=[]
        for start in range(0,len(xq),4096):
            dist,inds=nn.kneighbors(xq[start:start+4096])
            w=np.exp(-(dist/np.maximum(dist[:,-1:],1e-8))**2)
            w/=w.sum(axis=1,keepdims=True)
            output.append(np.einsum('ij,ijk->ik',w,source_y[inds]))
            contribution.append(np.sum(w*(ds[keep][inds]=='BM CyTOF'),axis=1))
        return np.vstack(output),np.concatenate(contribution)
    pooled,contribution=one(np.ones(source.sum(),bool))
    answer={'pooled':pooled,'cytof_weight':contribution}
    if equal:
        cy,_=one(ds=='BM CyTOF'); tr,_=one(ds=='Triana AbSeq')
        answer['equal']=.5*(cy+tr)
    return answer

def freeze():
    paths=[]
    for slug in SLUGS: paths.append(STORY/f'sampled_data/{slug}/source_input_15.csv.gz')
    paths += [STORY/'alignment/output/without_cd19/kmeanslow_baseline.csv.gz',STORY/'alignment/output/without_cd19/ramp_aligned.csv.gz',STORY/'alignment/output/with_predicted_cd19/kmeanslow_baseline.npy',STORY/'prediction/output/codex_cd19_prediction.npz',STORY/'prediction/output/codex_prediction_metadata.csv.gz',ROOT/'algorithm/ramp/ramp_ver2.py',ROOT/'algorithm/ramp/ramp.py']
    design={'primary':PRIMARY,'k':30,'kernel':'normalized exp(-(d/d30)^2)','neighbors':'All 58,031 CyTOF/AbSeq source cells; all target cell types','feature_scaler':'source-only StandardScaler','primary_features':'Fixed 14-marker RAMP 2.3.0 aligned expression, CD19 excluded','primary_target':'Observed source CD19 on existing KMEANSLOW scale, comparable to neural-network output','secondary_targets':'Source-patient CD19 relative ranks, as in the lung story','secondary_variants':['Equal 50/50 platform predictions, k=30 per source','KMEANSLOW feature space, without initial RAMP'], 'selection':'No new k, feature, threshold or donor search','existing_spatial_threshold':1.4,'spatial_donor':'SB67_NBM28_H14_CODEX_Mesmer','ground_truth_CD19_available_in_CODEX':False,'proxy_validation':'Remove held-out marker from all alignment/prediction features; refit RAMP on remaining 13 markers; target observed only in sources; evaluate all 12 CODEX donors','NN_proxy_comparison':'Existing 12-donor reconstruction is not a strict marker holdout; legacy 3-donor outer validation reported separately','write_scope':str(BASE),'input_sha256':{str(p):sha(p) for p in paths}}
    path=BASE/'provenance/DESIGN.json'
    if path.exists(): assert json.loads(path.read_text())==design
    else: save(path,design)

def run_prediction():
    freeze(); meta,raw,baseline,aligned,y,source,query=load()
    out=meta.loc[query].reset_index(drop=True)
    for space,x in [('RAMP',aligned),('KMEANSLOW',baseline)]:
        print(f'Predicting CD19: {space}',flush=True); start=time.time()
        result=predict(x,y,meta,source,query)
        for blend in ['pooled','equal']:
            out[f'{space}_KNN30_{blend}']=result[blend][:,0]
            out[f'{space}_KNN30_{blend}_rank']=result[blend][:,1]
        out[f'{space}_pooled_cytof_weight']=result['cytof_weight']
        print(f'  done in {time.time()-start:.1f}s',flush=True)
    nnmeta=pd.read_csv(STORY/'prediction/output/codex_prediction_metadata.csv.gz')
    assert nnmeta[IDENTITY].astype(str).equals(out[IDENTITY].astype(str))
    out['Neural_network']=np.load(STORY/'prediction/output/codex_cd19_prediction.npz')['predicted_kmeanslow']
    assert np.isfinite(out.select_dtypes('number')).all().all()
    out.to_csv(RESULTS/'CD19_PREDICTIONS.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    np.savez_compressed(RESULTS/'CD19_PREDICTIONS.npz',**{c:out[c].to_numpy() for c in out if c not in IDENTITY})
    save(RESULTS/'PREDICTION_FREEZE.json',{'time_utc':now(),'n_cells':len(out),'primary':PRIMARY,'prediction_sha256':sha(RESULTS/'CD19_PREDICTIONS.csv.gz'),'labels_coordinates_failed_CD19_used':False})
    evaluate()

def evaluate():
    # Load labels only after every prediction is frozen.
    pred=pd.read_csv(RESULTS/'CD19_PREDICTIONS.csv.gz')
    md=pd.read_csv(STORY/'sampled_data/bm_codex/locked_metadata.csv.gz')
    assert pred.cell_id.astype(str).equals(md.cell_id.astype(str))
    labels=pd.read_csv(STORY/'alignment/output/cell_metadata.csv.gz')
    labels=labels[labels.dataset.eq('BM CODEX')].reset_index(drop=True)
    assert pred.cell_id.astype(str).equals(labels.cell_id.astype(str))
    b=md.cluster_anno_l1.eq('B-Cells').to_numpy()
    t=labels.celltype.isin(['CD4 T','CD8 T']).to_numpy()
    technical=labels.celltype.eq('Unclassified / technical').to_numpy()
    methods=['Neural_network','RAMP_KNN30_pooled','RAMP_KNN30_equal','RAMP_KNN30_pooled_rank','RAMP_KNN30_equal_rank','KMEANSLOW_KNN30_pooled','KMEANSLOW_KNN30_equal','KMEANSLOW_KNN30_pooled_rank','KMEANSLOW_KNN30_equal_rank']
    rows=[]; threshold=[]
    for donor,ix in pred.groupby('donor').indices.items():
        ncall=int((pred.iloc[ix].Neural_network>1.4).sum())
        for method in methods:
            score=pred[method].to_numpy()[ix]; truth=b[ix]; bt=truth|t[ix]
            clean=~technical[ix]
            rows.append({'donor':donor,'method':method,'n_cells':len(ix),'n_b':int(truth.sum()),'n_t':int(t[ix].sum()),'B_vs_T_AUROC':roc_auc_score(truth[bt],score[bt]),'B_vs_all_AUROC':roc_auc_score(truth,score),'B_vs_all_AP':average_precision_score(truth,score),'B_vs_clean_AP':average_precision_score(truth[clean],score[clean])})
            calls={'matched_NN_call_count':np.zeros(len(ix),bool)}
            if ncall: calls['matched_NN_call_count'][np.argsort(-score,kind='stable')[:ncall]]=True
            if not method.endswith('_rank'): calls['fixed_1.4']=score>1.4
            for rule,high in calls.items():
                tp=int((high&truth).sum()); n=int(high.sum()); nb=int(truth.sum())
                threshold.append({'donor':donor,'method':method,'rule':rule,'n_high':n,'n_b':nb,'n_overlap':tp,'precision':tp/n if n else 0.,'recall':tp/nb if nb else 0.,'f1':2*tp/(n+nb) if n+nb else 0.})
    scores=pd.DataFrame(rows); scores.to_csv(RESULTS/'CD19_DONOR_METRICS.csv',index=False)
    scores.groupby('method').agg(n_donors=('donor','nunique'),median_B_vs_T_AUROC=('B_vs_T_AUROC','median'),mean_B_vs_T_AUROC=('B_vs_T_AUROC','mean'),median_B_vs_all_AP=('B_vs_all_AP','median'),mean_B_vs_all_AP=('B_vs_all_AP','mean')).to_csv(RESULTS/'CD19_METRIC_SUMMARY.csv')
    pd.DataFrame(threshold).to_csv(RESULTS/'CD19_THRESHOLD_METRICS.csv',index=False)
    print(pd.read_csv(RESULTS/'CD19_METRIC_SUMMARY.csv').to_string(index=False),flush=True)

def ramp_module():
    sys.path.insert(0,str(ROOT/'algorithm'))
    from ramp import ramp_ver2
    assert ramp_ver2.__version__=='2.3.0'
    return ramp_ver2

def run_proxy():
    freeze(); meta,raw,baseline,aligned,y,source,query=load(); ramp=ramp_module()
    units=meta.loc[query,'unit_key'].to_numpy(); source_units=meta.loc[source,'unit_key'].to_numpy()
    records=[]
    for j,marker in enumerate(MARKERS):
        path=RESULTS/f'proxy_{marker}.csv'
        if path.exists(): records.extend(pd.read_csv(path).to_dict('records')); continue
        start=time.time(); keep=[i for i in range(14) if i!=j]
        print(f'Strict proxy {j+1}/14 {marker}: refit RAMP with marker entirely excluded',flush=True)
        fit=ramp.fit_transform(raw[:,keep],meta[['dataset','unit_key']],[MARKERS[i] for i in keep],datasets=DATASETS,family=f'bm_knn_strict_holdout_{marker}')
        assert fit.strict_rank_inversions==0
        ys=baseline[source,j]; ranks=np.empty(len(ys))
        for u in np.unique(source_units):
            ix=source_units==u; ranks[ix]=rankdata(ys[ix])/(ix.sum()+1)
        target=np.column_stack([ys,ranks])
        predictions={}
        for space,x in [('RAMP',fit.aligned),('KMEANSLOW',baseline[:,keep])]:
            v=predict(x,target,meta,source,query)
            for blend in ['pooled','equal']:
                predictions[f'{space}_KNN30_{blend}']=v[blend][:,0]
                predictions[f'{space}_KNN30_{blend}_rank']=v[blend][:,1]
        # CODEX truth enters only after prediction.
        truth=baseline[query,j]; local_records=[]
        for u in np.unique(units):
            ix=units==u
            km=KMeans(n_clusters=2,n_init=20,max_iter=500,random_state=ramp.stable_seed('HIERARCHICAL_PROXY','HIGH',marker,u)).fit(truth[ix,None])
            high=km.labels_==int(np.argmax(km.cluster_centers_.ravel()))
            for method,score in predictions.items():
                rho=spearmanr(truth[ix],score[ix]).statistic
                local_records.append({'marker':marker,'donor':u,'method':method,'n_cells':int(ix.sum()),'spearman':float(rho) if np.isfinite(rho) else 0.,'high_state_auroc':roc_auc_score(high,score[ix]),'mae':float(np.abs(truth[ix]-score[ix]).mean()) if not method.endswith('_rank') else None})
        pd.DataFrame(local_records).to_csv(path,index=False)
        np.savez_compressed(RESULTS/f'proxy_{marker}_predictions.npz',**predictions)
        save(RESULTS/f'proxy_{marker}_audit.json',{'heldout_marker':marker,'fit_markers':[MARKERS[i] for i in keep],'strict_rank_inversions':int(fit.strict_rank_inversions),'labels_coordinates_used':False,'heldout_marker_in_alignment':False,'target_CODEX_marker_used_to_fit_predictor':False,'seconds':time.time()-start})
        records.extend(local_records)
        pd.DataFrame(records).to_csv(RESULTS/'PROXY_DONOR_METRICS.csv',index=False)
        print(f'  {marker} complete in {time.time()-start:.1f}s',flush=True)
    d=pd.DataFrame(records)
    summary=d.groupby(['method','marker']).agg(median_donor_spearman=('spearman','median'),mean_high_state_auroc=('high_state_auroc','mean'),n_donors=('donor','nunique')).reset_index()
    summary.to_csv(RESULTS/'PROXY_MARKER_SUMMARY.csv',index=False)
    summary.groupby('method').agg(median_marker_spearman=('median_donor_spearman','median'),mean_marker_auroc=('mean_high_state_auroc','mean')).to_csv(RESULTS/'PROXY_OVERALL_SUMMARY.csv')
    print(pd.read_csv(RESULTS/'PROXY_OVERALL_SUMMARY.csv').to_string(index=False),flush=True)

def run_alignment():
    meta,raw,baseline,aligned,y,source,query=load(); ramp=ramp_module()
    pred=pd.read_csv(RESULTS/'CD19_PREDICTIONS.csv.gz')
    complete=np.column_stack([baseline,np.zeros(len(meta))]).astype(np.float32).astype(float)
    complete[source,14]=y[:,0]; complete[query,14]=pred[PRIMARY]
    # Same float32 boundary and family as the neural-network completed panel.
    complete=complete.astype(np.float32).astype(float)
    matching=np.column_stack([raw,np.zeros(len(meta))])
    offset=0
    for ds,slug in zip(DATASETS,SLUGS):
        d=pd.read_csv(STORY/f'sampled_data/{slug}/source_input_15.csv.gz',usecols=['CD19'])
        if ds=='BM CODEX': matching[offset:offset+len(d),14]=pred[PRIMARY]
        else: matching[offset:offset+len(d),14]=d.CD19
        offset+=len(d)
    fitmd=meta[['dataset','unit_key']]
    bm=ramp.Benchmark(family='bm_prediction_story_completed15',raw=matching,metadata=fitmd,markers=MARKERS+['CD19'],datasets=DATASETS,reference=DATASETS[0])
    result=ramp.fit_v40_v21(bm,complete)
    inv=ramp.rank_inversion_count(complete,result[0],fitmd)
    assert inv==0 and np.max(np.abs(result[0]-complete))<=.75+1e-9
    np.save(RESULTS/'KNN_COMPLETED15_BASELINE.npy',complete)
    np.save(RESULTS/'KNN_COMPLETED15_RAMP.npy',result[0])
    save(RESULTS/'KNN_ALIGNMENT_AUDIT.json',{'prediction':PRIMARY,'family':bm.family,'n_cells':len(meta),'strict_rank_inversions':int(inv),'max_displacement':float(np.max(np.abs(result[0]-complete))),'float32_interface':True,'time_utc':now()})
    print('Completed KNN 15-marker RAMP alignment',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('mode',choices=['predict','proxy','align','evaluate']); args=parser.parse_args()
    {'predict':run_prediction,'proxy':run_proxy,'align':run_alignment,'evaluate':evaluate}[args.mode]()
