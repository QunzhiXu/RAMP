"""Fixed NN permanent CD3 holdout; joint three-platform UMAP, CODEX display."""
from runtime import BASE,STORY,ROOT,save,sha,now
import sys,time,json,argparse
import numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler

OUT=BASE/'CD3_NN_UMAP_validation'
OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(STORY/'prediction'))
import masked_marker_model as model_def
import torch
ramp=model_def.ramp
MARKER='CD3'
NORMAL=list(model_def.NORMAL_MARKERS)
J=NORMAL.index(MARKER)
KEEP=[i for i in range(14) if i!=J]
FEATURES=[NORMAL[i] for i in KEEP]
ID=['cell_id','dataset','donor','unit_key']
DATASETS=list(model_def.DATASETS)
SLUGS=['bm_codex','triana_abseq','bm_cytof']
T_TYPES=['CD4 T','CD8 T','Other T / NKT']

def load_inputs(targets=False):
    frames=[]
    for dataset,slug in zip(DATASETS,SLUGS):
        cols=ID+FEATURES+([MARKER,'CD19'] if targets and dataset!='BM CODEX' else [])
        f=pd.read_csv(STORY/f'sampled_data/{slug}/source_input_15.csv.gz',usecols=cols)
        assert f.dataset.eq(dataset).all()
        frames.append(f)
    d=pd.concat(frames,ignore_index=True)
    md=d[ID].copy()
    assert len(d)==88031 and not md.duplicated(['dataset','cell_id']).any()
    return d,md

def freeze():
    paths=[STORY/f'sampled_data/{s}/source_input_15.csv.gz' for s in SLUGS]
    paths += [STORY/'prediction/masked_marker_model.py',ROOT/'algorithm/ramp/ramp.py',
              ROOT/'algorithm/ramp/ramp_ver2.py']
    design={'marker':MARKER,'n_codex_cells':30000,'n_codex_donors':12,
        'input_features':FEATURES,'target_platform_marker_permanently_missing':True,
        'architecture':'same 14-input/15-output masked-marker NN; constant missing CD3 feature slot',
        'mask':'CD3 value=0, missing mask=1 on every forward pass in every platform',
        'targets':'CODEX excludes CD3 and CD19; reference platforms retain both as targets only',
        'training':{'fit_cap_per_platform':model_def.FIT_CAP_PER_PLATFORM,'epochs':model_def.EPOCHS_FINAL,
            'batch_size':model_def.BATCH_SIZE,'learning_rate':model_def.LEARNING_RATE,
            'weight_decay':model_def.WEIGHT_DECAY,
            'seed':model_def.stable_seed('DIRECT_MULTITASK_SHARED_SEED')},
        'source_combination':'fixed 0.5 Triana + 0.5 CyTOF output with CODEX input adapter',
        'preprocessing':'donor/marker-independent KMEANSLOW; pooled StandardScaler on 13 inputs',
        'embedding':'joint 3-platform RAMP and UMAP, 88031 cells; display CODEX rows only',
        'umap':{'n_neighbors':30,'min_dist':.15,'metric':'euclidean','seed':42,
                'init':'N(0,1e-4), default_rng(42)','PCA':False},
        'selection':'CD3 requested after CD4 exploration; no additional donor, seed, threshold or hyperparameter search',
        'celltype_display':{'T cells':T_TYPES,'Other cells':'remaining frozen CODEX labels'},
        'input_sha256':{str(p):sha(p) for p in paths}}
    path=OUT/'DESIGN.json'
    if path.exists(): assert json.loads(path.read_text())==design
    else: save(path,design)
    return design

def mask_hook(module,args):
    v=args[0].clone(); v[:,J]=0.; v[:,14+J]=1.
    return (v,)

BaseModel=model_def.MaskedMarkerModel
class HoldoutModel(BaseModel):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.shared.register_forward_pre_hook(mask_hook)

def train():
    design=freeze(); start=time.time()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    d,md=load_inputs(True)
    source=md.dataset.ne('BM CODEX').to_numpy(); qr=np.flatnonzero(~source)
    platform=md.dataset.map(model_def.PLATFORM_TO_ID).to_numpy(int)
    print(f'Preprocess 13 inputs; CODEX {MARKER} and CD19 have not been loaded',flush=True)
    base=ramp.kmeanslow_global(d[FEATURES].to_numpy(float),md.unit_key.to_numpy(),md.dataset.to_numpy(),FEATURES)
    sy=ramp.kmeanslow_global(d.loc[source,[MARKER,'CD19']].to_numpy(float),
        md.loc[source,'unit_key'].to_numpy(),md.loc[source,'dataset'].to_numpy(),[MARKER,'CD19'])
    targets=np.full((len(md),15),np.nan,np.float32)
    targets[:,KEEP]=base; targets[source,J]=sy[:,0]; targets[source,14]=sy[:,1]
    assert np.isnan(targets[~source][:,[J,14]]).all()
    scaler=StandardScaler().fit(base)
    x=np.zeros((len(md),14),np.float32); x[:,KEEP]=scaler.transform(base).astype(np.float32)
    assert np.all(x[:,J]==0)
    weights=model_def.target_balance_weights(targets,md)
    fit=model_def.donor_balanced_fit_indices(md)
    old=model_def.allowed_targets
    def allowed(platform):
        a=old(platform)
        return a[a!=J] if platform==model_def.CODEX_ID else a
    model_def.allowed_targets=allowed; model_def.MaskedMarkerModel=HoldoutModel
    assert J not in allowed(model_def.CODEX_ID) and 14 not in allowed(model_def.CODEX_ID)
    print('Train fixed NN: 60 epochs, 4000 donor-balanced cells per platform',flush=True)
    model=model_def.train_model(x,targets,weights,platform,fit)
    pred,tr,cy=model_def.predict_normal_marker(model,x,qr,J)
    assert np.isfinite(pred).all()
    # Check missing-channel invariance for both its own and another training task.
    probe=x[qr[:256]].copy(); alt=probe.copy(); alt[:,J]=np.linspace(-1e6,1e6,len(probe))
    with torch.no_grad():
        ip=torch.full((len(probe),),model_def.CODEX_ID,dtype=torch.long)
        op=torch.full_like(ip,model_def.CYTOF_ID)
        for target in [J,next(i for i in range(14) if i!=J)]:
            tg=torch.full_like(ip,target)
            assert torch.equal(model(torch.from_numpy(probe),tg,ip,op),
                               model(torch.from_numpy(alt),tg,ip,op))
    torch.save(model.state_dict(),OUT/f'{MARKER}_permanent_holdout_nn.pt')
    np.savez_compressed(OUT/f'{MARKER}_prediction.npz',prediction=pred,triana_component=tr,
        cytof_component=cy,codex_rows=qr,fit_rows=fit,scaler_mean=scaler.mean_,scaler_scale=scaler.scale_)
    md.iloc[qr].to_csv(OUT/'codex_prediction_metadata.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    md.iloc[fit].assign(original_row=fit).to_csv(OUT/'fit_cells.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    save(OUT/'PREDICTION_FROZEN.json',{'time_utc':now(),'seconds':time.time()-start,
        'prediction_sha256':sha(OUT/f'{MARKER}_prediction.npz'),f'codex_{MARKER.lower()}_truth_loaded':False,
        'codex_cd19_loaded':False,'labels_or_coordinates_loaded':False,
        'permanent_mask_perturbation_test':'PASS',f'CODEX_{MARKER}_CD19_targets_all_NaN':True,
        'n_predictions':len(pred),'input_hashes_unchanged':all(sha(p)==h for p,h in design['input_sha256'].items())})
    print(f'Predictions frozen after {time.time()-start:.1f}s',flush=True)

def embed():
    design=freeze(); start=time.time()
    d,md=load_inputs()
    print('RAMP on 13 shared markers, CD3 and CD19 excluded',flush=True)
    fit=ramp.fit_transform(d[FEATURES].to_numpy(float),md[['dataset','unit_key']],
        FEATURES,datasets=DATASETS,family='bm_nn_cd3_permanent_holdout')
    assert fit.strict_rank_inversions==0
    np.save(OUT/'ramp13_all.npy',fit.aligned)
    fit.coupling_audit.to_csv(OUT/'ramp_coupling_audit.csv',index=False)
    fit.strength_audit.to_csv(OUT/'ramp_strength_audit.csv',index=False)
    import umap
    print('Joint three-platform UMAP: 88031 cells, fixed seed 42',flush=True)
    x=StandardScaler().fit_transform(fit.aligned).astype(np.float32)
    init=np.random.default_rng(42).normal(0,1e-4,(len(md),2)).astype(np.float32)
    xy=umap.UMAP(n_neighbors=30,min_dist=.15,metric='euclidean',random_state=42,
        transform_seed=42,n_jobs=1,low_memory=True,init=init).fit_transform(x).astype(np.float32)
    assert xy.shape==(88031,2) and np.isfinite(xy).all()
    q=md.dataset.eq('BM CODEX').to_numpy()
    np.save(OUT/'joint_umap_all.npy',xy); np.save(OUT/'codex_umap.npy',xy[q])
    md.to_csv(OUT/'all_umap_metadata.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    md.loc[q].to_csv(OUT/'codex_umap_metadata.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    save(OUT/'UMAP_FROZEN.json',{'time_utc':now(),'seconds':time.time()-start,
        'coordinate_sha256':sha(OUT/'joint_umap_all.npy'),'n_joint_cells':len(md),'n_codex_cells':int(q.sum()),
        'features':FEATURES,'codex_cd3_or_cd19_loaded':False,'labels_or_predictions_used':False,
        'rank_inversions':int(fit.strict_rank_inversions),
        'input_hashes_unchanged':all(sha(p)==h for p,h in design['input_sha256'].items())})
    print(f'Joint coordinates frozen after {time.time()-start:.1f}s',flush=True)

def evaluate():
    assert (OUT/'PREDICTION_FROZEN.json').exists() and (OUT/'UMAP_FROZEN.json').exists()
    from scipy.stats import spearmanr
    from sklearn.cluster import KMeans
    from sklearn.metrics import roc_auc_score,average_precision_score
    md=pd.read_csv(OUT/'codex_prediction_metadata.csv.gz')
    assert md.equals(pd.read_csv(OUT/'codex_umap_metadata.csv.gz'))
    truthraw=pd.read_csv(STORY/'sampled_data/bm_codex/source_input_15.csv.gz',usecols=ID+[MARKER])
    assert md[ID].astype(str).equals(truthraw[ID].astype(str))
    truth=ramp.kmeanslow_global(truthraw[[MARKER]].to_numpy(float),md.unit_key.to_numpy(),
        md.dataset.to_numpy(),[MARKER])[:,0]
    prediction=np.load(OUT/'CD3_prediction.npz')['prediction']
    labels=pd.read_csv(STORY/'alignment/output/cell_metadata.csv.gz').query("dataset == 'BM CODEX'").reset_index(drop=True)
    assert md[ID].astype(str).equals(labels[ID].astype(str))
    t=labels.celltype.isin(T_TYPES).to_numpy()
    high=np.zeros(len(md),bool); matched=high.copy(); records=[]
    for unit,ix in md.groupby('unit_key',sort=True).indices.items():
        fit=KMeans(n_clusters=2,n_init=20,max_iter=500,
            random_state=model_def.stable_seed('HIGH',MARKER,unit)).fit(truth[ix,None])
        h=fit.labels_==int(np.argmax(fit.cluster_centers_.ravel())); high[ix]=h
        local=np.argsort(-prediction[ix],kind='stable')[:int(h.sum())]; matched[ix[local]]=True
        records.append({'donor':unit,'n_cells':len(ix),'n_T':int(t[ix].sum()),
            'spearman':spearmanr(truth[ix],prediction[ix]).statistic,
            'high_state_auroc':roc_auc_score(h,prediction[ix]),
            'high_state_AP':average_precision_score(h,prediction[ix]),
            'mae':np.abs(truth[ix]-prediction[ix]).mean(),
            'matched_high_overlap_fraction':np.mean(h[local]),
            'measured_T_vs_other_AUROC':roc_auc_score(t[ix],truth[ix]),
            'predicted_T_vs_other_AUROC':roc_auc_score(t[ix],prediction[ix]),
            'measured_T_vs_other_AP':average_precision_score(t[ix],truth[ix]),
            'predicted_T_vs_other_AP':average_precision_score(t[ix],prediction[ix])})
    scores=pd.DataFrame(records); scores.to_csv(OUT/'donor_metrics.csv',index=False)
    cells=md.copy(); cells['celltype']=labels.celltype
    cells['measured_CD3']=truth; cells['predicted_CD3']=prediction
    cells['measured_high_state']=high; cells['predicted_matched_high_state']=matched
    cells[['UMAP1','UMAP2']]=np.load(OUT/'codex_umap.npy')
    cells.to_csv(OUT/'cell_predictions_and_umap.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    groups=cells.groupby('celltype').agg(n=('cell_id','size'),measured_median=('measured_CD3','median'),
        predicted_median=('predicted_CD3','median'),measured_high_fraction=('measured_high_state','mean'),
        predicted_high_fraction=('predicted_matched_high_state','mean'))
    groups.to_csv(OUT/'celltype_summary.csv')
    cells.groupby(['donor','celltype']).agg(n=('cell_id','size'),measured_median=('measured_CD3','median'),
        predicted_median=('predicted_CD3','median')).to_csv(OUT/'celltype_by_donor.csv')
    summary={'marker':MARKER,'n_cells':len(cells),'n_donors':len(scores),'n_T':int(t.sum()),
        'median_donor_spearman':float(scores.spearman.median()),
        'mean_donor_high_state_auroc':float(scores.high_state_auroc.mean()),
        'median_donor_measured_T_vs_other_AUROC':float(scores.measured_T_vs_other_AUROC.median()),
        'median_donor_predicted_T_vs_other_AUROC':float(scores.predicted_T_vs_other_AUROC.median()),
        'median_donor_predicted_T_vs_other_AP':float(scores.predicted_T_vs_other_AP.median()),
        'spearman_range':[float(scores.spearman.min()),float(scores.spearman.max())],
        'pooled_spearman':float(spearmanr(truth,prediction).statistic),
        'model_refit_scope':'permanent CD3 marker holdout, not donor holdout'}
    save(OUT/'METRIC_SUMMARY.json',summary)
    cd4=pd.read_csv(BASE/'CD4_NN_UMAP_validation/donor_metrics.csv')
    comparison=cd4[['donor','spearman','high_state_auroc']].merge(
        scores[['donor','spearman','high_state_auroc']],on='donor',suffixes=('_CD4','_CD3'),validate='one_to_one')
    comparison.to_csv(OUT/'CD3_vs_CD4_marker_recovery.csv',index=False)
    print(json.dumps(summary,indent=2)); print(groups.to_string(),flush=True)
    plot(cells)

def plot(cells):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap,Normalize
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','font.size':9,'text.color':'#27323B'})
    xy=cells[['UMAP1','UMAP2']].to_numpy(); t=cells.celltype.isin(T_TYPES).to_numpy()
    vmin,vmax=np.quantile(cells.measured_CD3,[.01,.99]); norm=Normalize(vmin,vmax,clip=True)
    cmap=LinearSegmentedColormap.from_list('CD3_expression',['#DCE1E6','#F5E7B2','#EDB34E','#CF572D','#861E41'])
    fig=plt.figure(figsize=(12,3.8)); positions=[.022,.352,.663]
    axes=[fig.add_axes([left,.13,.25,.72]) for left in positions]
    palette={'T cells':'#266BAE','Other cells':'#E7C36B'}
    for select,color in [(~t,palette['Other cells']),(t,palette['T cells'])]:
        axes[0].scatter(xy[select,0],xy[select,1],s=1.2,c=color,alpha=1,lw=0,rasterized=True)
    for ax,col in zip(axes[1:],['measured_CD3','predicted_CD3']):
        values=cells[col].to_numpy(); order=np.argsort(values,kind='stable')
        sc=ax.scatter(xy[order,0],xy[order,1],s=1.2,c=values[order],cmap=cmap,norm=norm,alpha=1,lw=0,rasterized=True)
    lo=xy.min(axis=0); hi=xy.max(axis=0); pad=.035*(hi-lo)
    for ax,left,title in zip(axes,positions,['Cell type','Measured CD3','Predicted CD3']):
        ax.set(xlim=(lo[0]-pad[0],hi[0]+pad[0]),ylim=(lo[1]-pad[1],hi[1]+pad[1]))
        ax.set_aspect('equal'); ax.axis('off')
        fig.text(left+.125,.835,title,ha='center',va='bottom',fontsize=11.5,fontweight='bold')
    fig.legend([Line2D([0],[0],marker='o',ls='',color=c,ms=4.3) for c in palette.values()],
        list(palette),loc='center left',bbox_to_anchor=(.269,.47),frameon=False,fontsize=7.8,
        labelspacing=.8,handletextpad=.35,borderaxespad=0)
    cax=fig.add_axes([.943,.28,.009,.39]); cb=fig.colorbar(sc,cax=cax)
    cb.set_label('CD3 (KMEANSLOW)',fontsize=8.2); cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8,width=.6,length=2.5)
    fig.text(.03,.965,'CD3 prediction validation',ha='left',va='top',fontsize=14.5,fontweight='bold')
    fig.text(.03,.044,'Joint three-platform UMAP · 30,000 CODEX cells · Shared coordinates and expression scale',fontsize=8.4)
    fig.text(.03,.011,'CD3 and CD19 excluded from UMAP inputs; T cells drawn above Other cells, with all cells retained.',fontsize=7.8,color='#65717B')
    for ext in ['png','svg']: fig.savefig(OUT/f'CD3_validation_umap.{ext}',dpi=300,facecolor='white')
    plt.close(fig)
    save(OUT/'PLOT_AUDIT.json',{'n_cells':len(cells),'n_T':int(t.sum()),'n_other':int((~t).sum()),
        'T_labels':T_TYPES,'painter_order':['Other cells','T cells'],'uniform_size':1.2,'uniform_alpha':1,
        'all_cells_retained':True,'position_based_filtering':False,'independent_expression_scales':False,
        'vmin':float(vmin),'vmax':float(vmax),'limits':'measured CD3 1%/99% on all CODEX cells',
        'expression_smoothing':False,'selection':'single fixed run'})

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('mode',choices=['freeze','train','embed','evaluate','plot'])
    args=parser.parse_args()
    if args.mode=='plot': plot(pd.read_csv(OUT/'cell_predictions_and_umap.csv.gz'))
    else: {'freeze':freeze,'train':train,'embed':embed,'evaluate':evaluate}[args.mode]()
