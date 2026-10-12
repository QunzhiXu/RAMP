"""Permanent CODEX CD4 holdout with the fixed neural network; UMAP display.

All writes are guarded by runtime.py. CD4 is absent from every platform's
predictor input and from CODEX targets; reference CD4 remains a target.
No CODEX CD19 is loaded. CODEX CD4 truth/labels are loaded only for evaluation.
"""
from runtime import BASE, STORY, ROOT, save, sha, now
import sys, argparse, time, json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

OUT=BASE/'CD4_NN_UMAP_validation'
OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(STORY/'prediction'))
import masked_marker_model as original
import torch
ramp=original.ramp
MARKERS=list(original.NORMAL_MARKERS)
J=MARKERS.index('CD4')
KEEP=[k for k in range(14) if k!=J]
FEATURES=[MARKERS[k] for k in KEEP]
DATASETS=list(original.DATASETS)
SLUGS=['bm_codex','triana_abseq','bm_cytof']
ID=['cell_id','dataset','donor','unit_key']

def load_inputs(targets=False):
    frames=[]; reference_frames=[]
    for ds,slug in zip(DATASETS,SLUGS):
        columns=ID+FEATURES
        if targets and ds!='BM CODEX': columns+=['CD4','CD19']
        f=pd.read_csv(STORY/f'sampled_data/{slug}/source_input_15.csv.gz',usecols=columns)
        assert f.dataset.eq(ds).all()
        frames.append(f)
        if ds!='BM CODEX': reference_frames.append(f)
    d=pd.concat(frames,ignore_index=True)
    md=d[ID].copy()
    assert len(md)==88031 and not md.duplicated(['dataset','cell_id']).any()
    return d,md

def freeze():
    paths=[STORY/f'sampled_data/{s}/source_input_15.csv.gz' for s in SLUGS]
    paths += [STORY/'prediction/masked_marker_model.py',ROOT/'algorithm/ramp/ramp.py',
              ROOT/'algorithm/ramp/ramp_ver2.py']
    design={
        'marker':'CD4','n_codex_cells':30000,'n_codex_donors':12,
        'neural_network':'same 14-input/15-target masked-marker architecture, retrained from scratch',
        'permanent_mask':'CD4 input value forced to 0 and mask to 1 on ALL platforms on EVERY forward pass',
        'training_targets':'CODEX excludes CD4 and CD19; references retain both as supervised targets',
        'preprocessing':'KMEANSLOW independently by donor and marker; pooled scaler on 13 observed inputs only',
        'training_settings':{'fit_cap_per_platform':original.FIT_CAP_PER_PLATFORM,
            'epochs':original.EPOCHS_FINAL,'batch_size':original.BATCH_SIZE,
            'learning_rate':original.LEARNING_RATE,'weight_decay':original.WEIGHT_DECAY,
            'seed':original.stable_seed('DIRECT_MULTITASK_SHARED_SEED')},
        'prediction':'fixed 0.5 Triana output + 0.5 CyTOF output, CODEX input adapter',
        'umap':'refit RAMP on 13 markers across all 3 platforms, fit UMAP on resulting 30000 CODEX rows only',
        'umap_features':FEATURES,'umap_parameters':{'n_neighbors':30,'min_dist':.15,
            'metric':'euclidean','random_state':42,'init':'N(0,1e-4) seed 42','PCA':False},
        'selection':'no marker, donor, threshold, hyperparameter, or UMAP seed search',
        'input_sha256':{str(p):sha(p) for p in paths}}
    path=OUT/'DESIGN.json'
    if path.exists(): assert json.loads(path.read_text())==design
    else: save(path,design)
    return design

def permanent_input_mask(module,args):
    x=args[0].clone()
    x[:,J]=0.
    x[:,14+J]=1.
    return (x,)

BaseModel=original.MaskedMarkerModel
class HoldoutModel(BaseModel):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.shared.register_forward_pre_hook(permanent_input_mask)

def train():
    design=freeze(); start=time.time()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    d,md=load_inputs(targets=True)
    source=md.dataset.ne('BM CODEX').to_numpy(); codex_rows=np.flatnonzero(~source)
    platform_ids=md.dataset.map(original.PLATFORM_TO_ID).to_numpy(int)
    raw=d[FEATURES].to_numpy(float)
    print('Preprocess 13 observed inputs; CODEX CD4 and CD19 are not loaded',flush=True)
    baseline=ramp.kmeanslow_global(raw,md.unit_key.to_numpy(),md.dataset.to_numpy(),FEATURES)
    observed_source=ramp.kmeanslow_global(d.loc[source,['CD4','CD19']].to_numpy(float),
        md.loc[source,'unit_key'].to_numpy(),md.loc[source,'dataset'].to_numpy(),['CD4','CD19'])
    targets=np.full((len(md),15),np.nan,dtype=np.float32)
    targets[:,KEEP]=baseline
    targets[source,J]=observed_source[:,0]
    targets[source,14]=observed_source[:,1]
    assert np.isnan(targets[~source][:,[J,14]]).all()
    scaler=StandardScaler().fit(baseline)
    x=np.zeros((len(md),14),dtype=np.float32)
    x[:,KEEP]=scaler.transform(baseline).astype(np.float32)
    assert np.all(x[:,J]==0)
    weights=original.target_balance_weights(targets,md)
    fit=original.donor_balanced_fit_indices(md)
    old_allowed=original.allowed_targets
    def allowed(platform):
        a=old_allowed(platform)
        return a[a!=J] if platform==original.CODEX_ID else a
    original.allowed_targets=allowed
    original.MaskedMarkerModel=HoldoutModel
    assert J not in allowed(original.CODEX_ID) and 14 not in allowed(original.CODEX_ID)
    assert J in allowed(original.TRIANA_ID) and J in allowed(original.CYTOF_ID)
    print('Train fixed NN from scratch: 60 epochs, 4000 cells/platform',flush=True)
    model=original.train_model(x,targets,weights,platform_ids,fit)
    pred,tr,cy=original.predict_normal_marker(model,x,codex_rows,J)
    assert np.isfinite(pred).all()
    # Meaningful leakage check: arbitrary absent-channel values cannot affect output.
    probe=x[codex_rows[:256]].copy()
    altered=probe.copy(); altered[:,J]=np.linspace(-1e6,1e6,len(probe))
    with torch.no_grad():
        tg=torch.full((len(probe),),J,dtype=torch.long)
        ip=torch.full_like(tg,original.CODEX_ID); op=torch.full_like(tg,original.CYTOF_ID)
        a=model(torch.from_numpy(probe),tg,ip,op)
        b=model(torch.from_numpy(altered),tg,ip,op)
        assert torch.equal(a,b)
        # Also check permanent removal when predicting OTHER targets during training.
        tg[:]=0
        assert torch.equal(model(torch.from_numpy(probe),tg,ip,op),
                           model(torch.from_numpy(altered),tg,ip,op))
    torch.save(model.state_dict(),OUT/'CD4_permanent_holdout_nn.pt')
    np.savez_compressed(OUT/'CD4_prediction.npz',prediction=pred,triana_component=tr,
        cytof_component=cy,codex_rows=codex_rows,fit_rows=fit,
        scaler_mean=scaler.mean_,scaler_scale=scaler.scale_)
    md.iloc[codex_rows].to_csv(OUT/'codex_prediction_metadata.csv.gz',index=False,
                             compression={'method':'gzip','mtime':0})
    pd.DataFrame({'row':fit,'dataset':md.iloc[fit].dataset.to_numpy(),
                  'unit_key':md.iloc[fit].unit_key.to_numpy()}).to_csv(OUT/'fit_cells.csv.gz',index=False)
    save(OUT/'PREDICTION_FROZEN.json',{'time_utc':now(),'seconds':time.time()-start,
        'prediction_sha256':sha(OUT/'CD4_prediction.npz'),'codex_cd4_truth_loaded':False,
        'codex_cd19_loaded':False,'cell_labels_or_spatial_coordinates_loaded':False,
        'permanent_mask_value_perturbation_test':'PASS',
        'codex_heldout_targets_all_nan':True,'n_cells':len(pred),
        'input_hashes_unchanged':all(sha(p)==h for p,h in design['input_sha256'].items())})
    print(f'Prediction frozen after {time.time()-start:.1f} s',flush=True)

def embed():
    design=freeze(); start=time.time()
    d,md=load_inputs(); raw=d[FEATURES].to_numpy(float)
    print('Refit RAMP 2.3.0 using 13 markers, excluding CD4 and CD19',flush=True)
    fit=ramp.fit_transform(raw,md[['dataset','unit_key']],FEATURES,datasets=DATASETS,
                           family='bm_nn_cd4_permanent_holdout')
    assert fit.strict_rank_inversions==0
    q=md.dataset.eq('BM CODEX').to_numpy()
    np.save(OUT/'ramp13_codex.npy',fit.aligned[q])
    fit.coupling_audit.to_csv(OUT/'ramp_coupling_audit.csv',index=False)
    fit.strength_audit.to_csv(OUT/'ramp_strength_audit.csv',index=False)
    print('Fit CODEX-only UMAP on all 30000 cells',flush=True)
    import umap
    init=np.random.default_rng(42).normal(0,1e-4,size=(int(q.sum()),2)).astype(np.float32)
    x=StandardScaler().fit_transform(fit.aligned[q]).astype(np.float32)
    xy=umap.UMAP(n_neighbors=30,min_dist=.15,metric='euclidean',random_state=42,
        transform_seed=42,n_jobs=1,low_memory=True,init=init).fit_transform(x).astype(np.float32)
    assert xy.shape==(30000,2) and np.isfinite(xy).all()
    np.save(OUT/'umap_coordinates.npy',xy)
    md.loc[q].to_csv(OUT/'umap_metadata.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    save(OUT/'UMAP_FROZEN.json',{'time_utc':now(),'seconds':time.time()-start,
        'coordinate_sha256':sha(OUT/'umap_coordinates.npy'),'n_cells':len(xy),'features':FEATURES,
        'codex_cd4_or_cd19_loaded':False,'cell_labels_or_prediction_used':False,
        'rank_inversions':int(fit.strict_rank_inversions),
        'input_hashes_unchanged':all(sha(p)==h for p,h in design['input_sha256'].items())})
    print(f'UMAP frozen after {time.time()-start:.1f} s',flush=True)

def evaluate():
    assert (OUT/'PREDICTION_FROZEN.json').exists() and (OUT/'UMAP_FROZEN.json').exists()
    from scipy.stats import spearmanr
    from sklearn.cluster import KMeans
    from sklearn.metrics import roc_auc_score,average_precision_score
    md=pd.read_csv(OUT/'codex_prediction_metadata.csv.gz')
    umd=pd.read_csv(OUT/'umap_metadata.csv.gz'); assert md.equals(umd)
    raw=pd.read_csv(STORY/'sampled_data/bm_codex/source_input_15.csv.gz',usecols=ID+['CD4'])
    assert md[ID].astype(str).equals(raw[ID].astype(str))
    truth=ramp.kmeanslow_global(raw[['CD4']].to_numpy(float),md.unit_key.to_numpy(),
                               md.dataset.to_numpy(),['CD4'])[:,0]
    p=np.load(OUT/'CD4_prediction.npz')['prediction']
    labels=pd.read_csv(STORY/'alignment/output/cell_metadata.csv.gz')
    labels=labels[labels.dataset.eq('BM CODEX')].reset_index(drop=True)
    assert md[ID].astype(str).equals(labels[ID].astype(str))
    rows=[]; real_high=np.zeros(len(md),bool); matched_high=real_high.copy()
    for unit,ix in md.groupby('unit_key',sort=True).indices.items():
        k=KMeans(n_clusters=2,n_init=20,max_iter=500,
            random_state=original.stable_seed('HIGH','CD4',unit)).fit(truth[ix,None])
        h=k.labels_==int(np.argmax(k.cluster_centers_.ravel()))
        real_high[ix]=h
        local=np.argsort(-p[ix],kind='stable')[:int(h.sum())]
        matched_high[ix[local]]=True
        lt=labels.iloc[ix].celltype.to_numpy(); t=np.isin(lt,['CD4 T','CD8 T'])
        y=lt=='CD4 T'
        rows.append({'donor':unit,'n_cells':len(ix),'spearman':spearmanr(truth[ix],p[ix]).statistic,
            'mae':np.abs(truth[ix]-p[ix]).mean(),'high_state_auroc':roc_auc_score(h,p[ix]),
            'high_state_AP':average_precision_score(h,p[ix]),'n_real_high':int(h.sum()),
            'matched_high_overlap_fraction':np.mean(h[local]),
            'measured_CD4_vs_CD8_AUROC':roc_auc_score(y[t],truth[ix][t]),
            'predicted_CD4_vs_CD8_AUROC':roc_auc_score(y[t],p[ix][t])})
    scores=pd.DataFrame(rows); scores.to_csv(OUT/'donor_metrics.csv',index=False)
    xy=np.load(OUT/'umap_coordinates.npy')
    cells=md.copy(); cells['celltype']=labels.celltype; cells['measured_CD4']=truth
    cells['predicted_CD4']=p; cells['measured_high_state']=real_high
    cells['predicted_matched_high_state']=matched_high
    cells[['UMAP1','UMAP2']]=xy
    cells.to_csv(OUT/'cell_predictions_and_umap.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    cells.groupby(['donor','celltype']).agg(n=('cell_id','size'),measured_mean=('measured_CD4','mean'),
        predicted_mean=('predicted_CD4','mean'),measured_high_fraction=('measured_high_state','mean'),
        predicted_high_fraction=('predicted_matched_high_state','mean')).to_csv(OUT/'celltype_by_donor.csv')
    groups=cells.groupby('celltype').agg(n=('cell_id','size'),measured_median=('measured_CD4','median'),
        predicted_median=('predicted_CD4','median'),measured_high_fraction=('measured_high_state','mean'),
        predicted_high_fraction=('predicted_matched_high_state','mean'))
    groups.to_csv(OUT/'celltype_summary.csv')
    result={'marker':'CD4','n_cells':len(cells),'n_donors':len(scores),
        'median_donor_spearman':float(scores.spearman.median()),
        'mean_donor_high_state_auroc':float(scores.high_state_auroc.mean()),
        'median_donor_predicted_CD4_vs_CD8_AUROC':float(scores.predicted_CD4_vs_CD8_AUROC.median()),
        'median_donor_measured_CD4_vs_CD8_AUROC':float(scores.measured_CD4_vs_CD8_AUROC.median()),
        'pooled_spearman':float(spearmanr(truth,p).statistic),
        'spearman_range':[float(scores.spearman.min()),float(scores.spearman.max())],
        'UMAP_smoothing':False,'donor_or_seed_search':False,
        'evaluation_unit':'within-donor metrics; all frozen CODEX cells; marker holdout, not donor holdout'}
    save(OUT/'METRIC_SUMMARY.json',result)
    print(json.dumps(result,indent=2)); print(groups.to_string())
    plot(cells,result)

def plot(cells,result,output_dir=None,embedding_label='CODEX-only UMAP'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap,Normalize
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none',
                         'pdf.fonttype':42,'text.color':'#26333E'})
    destination=OUT if output_dir is None else output_dir
    xy=cells[['UMAP1','UMAP2']].to_numpy()
    display=np.full(len(cells),'Other cells',dtype=object)
    for key,types in {'CD4 T':['CD4 T'],'CD8 T':['CD8 T'],
        'B / plasma':['Mature / CODEX B','Plasma cell'],
        'Myeloid':['Other myeloid','Monocyte','Macrophage','Dendritic'],
        'Erythroid':['CODEX erythroid']}.items():
        display[cells.celltype.isin(types)]=key
    palette={'Other cells':'#C6CDD3','Erythroid':'#C9A8BE','Myeloid':'#E8B15E',
             'B / plasma':'#C7689B','CD8 T':'#24A5B3','CD4 T':'#235EB0'}
    cmap=LinearSegmentedColormap.from_list('CD4_expression',
        ['#DCE1E6','#F5E7B2','#EDB34E','#CF572D','#861E41'])
    vmin,vmax=np.quantile(cells.measured_CD4,[.01,.99])
    norm=Normalize(vmin,vmax,clip=True)
    fig,axes=plt.subplots(1,3,figsize=(12.0,4.3))
    for key,color in palette.items():
        ix=display==key
        axes[0].scatter(xy[ix,0],xy[ix,1],s=.9,c=color,alpha=.8,linewidths=0,rasterized=True)
    for ax,col in zip(axes[1:],['measured_CD4','predicted_CD4']):
        values=cells[col].to_numpy(); order=np.argsort(values,kind='stable')
        sc=ax.scatter(xy[order,0],xy[order,1],c=values[order],s=.9,
            cmap=cmap,norm=norm,alpha=1,linewidths=0,rasterized=True)
    lo=xy.min(axis=0); hi=xy.max(axis=0); pad=.035*(hi-lo)
    for ax,title in zip(axes,['Cell type','Measured CD4','Predicted CD4']):
        ax.set_title(title,fontsize=12,fontweight='bold',pad=6)
        ax.set(xlim=(lo[0]-pad[0],hi[0]+pad[0]),ylim=(lo[1]-pad[1],hi[1]+pad[1]))
        ax.set_aspect('equal',adjustable='box'); ax.axis('off')
    # Label positions depend only on frozen cell types and coordinates.
    for typ,offset in [('CD4 T',(-8,13)),('CD8 T',(10,-12))]:
        center=np.median(xy[display==typ],axis=0)
        axes[0].annotate(typ,center,xytext=offset,textcoords='offset points',
            fontsize=9,fontweight='bold',color=palette[typ],
            bbox={'boxstyle':'round,pad=.18','fc':'white','ec':'none','alpha':.86},
            arrowprops={'arrowstyle':'-','color':palette[typ],'lw':.8})
    axes[0].legend([Line2D([0],[0],marker='o',ls='',color=c,ms=4) for c in palette.values()],
        list(palette),loc='upper center',bbox_to_anchor=(.5,-.025),ncol=3,fontsize=7.4,
        frameon=False,columnspacing=.8,handletextpad=.25)
    cax=fig.add_axes([.937,.235,.010,.46]); cb=fig.colorbar(sc,cax=cax)
    cb.set_label('CD4 (KMEANSLOW)',fontsize=8.7); cb.outline.set_visible(False)
    fig.suptitle('CD4 recovery on a marker-excluded UMAP',x=.03,ha='left',fontsize=15,fontweight='bold',y=.98)
    fig.text(.03,.052,f'30,000 CODEX cells · 12 donors · {embedding_label} · Shared expression scale',fontsize=8.6)
    fig.text(.03,.016,'CD4 and CD19 excluded from UMAP inputs; CD4 permanently withheld from CODEX training targets.',fontsize=8.2,color='#5E6C76')
    fig.subplots_adjust(left=.02,right=.925,top=.85,bottom=.19,wspace=.06)
    for ext in ('png','svg'): fig.savefig(destination/f'CD4_validation_umap.{ext}',dpi=280,facecolor='white')
    plt.close(fig)
    save(destination/'PLOT_AUDIT.json',{'embedding_label':embedding_label,
        'color_scale':'shared measured-CD4 1st/99th percentile, all CODEX cells',
        'vmin':float(vmin),'vmax':float(vmax),
        'fraction_clipped':{c:{'below':float((cells[c]<vmin).mean()),'above':float((cells[c]>vmax).mean())}
                            for c in ['measured_CD4','predicted_CD4']},
        'display_group_mapping':{'CD4 T':['CD4 T'],'CD8 T':['CD8 T'],
          'B / plasma':['Mature / CODEX B','Plasma cell'],
          'Myeloid':['Other myeloid','Monocyte','Macrophage','Dendritic'],
          'Erythroid':['CODEX erythroid'],'Other cells':'remaining author-derived labels'},
        'all_cells_shown':True,'expression_smoothing':False,'independent_color_scales':False})

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('mode',choices=['freeze','train','embed','evaluate','plot'])
    args=parser.parse_args()
    if args.mode=='plot':
        plot(pd.read_csv(OUT/'cell_predictions_and_umap.csv.gz'),json.loads((OUT/'METRIC_SUMMARY.json').read_text()))
    else: {'freeze':freeze,'train':train,'embed':embed,'evaluate':evaluate}[args.mode]()
