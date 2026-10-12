"""Complete the fixed 12-donor permanent-holdout marker table for the figure."""
from runtime import BASE,STORY,ROOT,save,sha,now
import sys,json,argparse,subprocess
import numpy as np,pandas as pd
import cd3_nn_joint_validation as worker

OUT=BASE/'validation_subpanels'
DATA=OUT/'all_marker_validation'
DATA.mkdir(parents=True,exist_ok=True)
NORMAL=list(worker.NORMAL)
REUSE={'CD3':BASE/'CD3_NN_UMAP_validation','CD4':BASE/'CD4_NN_UMAP_validation'}

def init():
    sources=[STORY/f'sampled_data/{s}/source_input_15.csv.gz' for s in worker.SLUGS]
    sources += [STORY/'prediction/masked_marker_model.py',ROOT/'algorithm/ramp/ramp.py',
                ROOT/'algorithm/ramp/ramp_ver2.py',BASE/'scripts/cd3_nn_joint_validation.py']
    design={'markers':NORMAL,'n_codex_donors':12,'n_codex_cells':30000,
        'protocol':'permanent removal of target marker from all predictor inputs and CODEX training targets',
        'NN':'same fixed architecture, seed, 60 epochs and 4000 donor-balanced fitting cells/platform as CD3/CD4',
        'reference_targets':'held-out marker and CD19 retained only in reference targets; neither is an input',
        'source_average':'fixed 50/50 Triana and CyTOF outputs',
        'preprocessing':'KMEANSLOW donor/marker independent; pooled scaling on remaining 13 features',
        'reused_completed_markers':{m:str(p) for m,p in REUSE.items()},
        'remaining_tasks':'prediction and quantitative evaluation only; no extra UMAP fitting',
        'heatmap':'one row, original marker order, median within-donor Spearman per marker',
        'selection':'all 14 markers included; no donor, hyperparameter or threshold search',
        'input_sha256':{str(p):sha(p) for p in sources}}
    p=DATA/'DESIGN.json'
    if p.exists(): assert json.loads(p.read_text())==design
    else: save(p,design)
    return design

def run_worker(marker):
    assert marker in NORMAL and marker not in REUSE
    design=init(); destination=DATA/marker; destination.mkdir(exist_ok=True)
    if (destination/'donor_metrics.csv').exists():
        print(f'{marker}: existing complete result retained',flush=True); return
    # Each marker runs in a fresh Python process so module target hooks are isolated.
    worker.MARKER=marker; worker.J=NORMAL.index(marker)
    worker.KEEP=[i for i in range(14) if i!=worker.J]
    worker.FEATURES=[NORMAL[i] for i in worker.KEEP]
    worker.OUT=destination
    def freeze_marker():
        cfg={'marker':marker,'input_features':worker.FEATURES,
             'input_sha256':design['input_sha256'],'protocol':design['protocol'],
             'training':design['NN'],'preprocessing':design['preprocessing'],
             'source_average':design['source_average']}
        save(destination/'DESIGN.json',cfg); return cfg
    worker.freeze=freeze_marker
    worker.train()
    # Only now read the CODEX truth for the held-out channel.
    from sklearn.cluster import KMeans
    from sklearn.metrics import roc_auc_score
    from scipy.stats import spearmanr
    md=pd.read_csv(destination/'codex_prediction_metadata.csv.gz')
    raw=pd.read_csv(STORY/'sampled_data/bm_codex/source_input_15.csv.gz',usecols=worker.ID+[marker])
    assert md[worker.ID].astype(str).equals(raw[worker.ID].astype(str))
    truth=worker.ramp.kmeanslow_global(raw[[marker]].to_numpy(float),md.unit_key.to_numpy(),
                                      md.dataset.to_numpy(),[marker])[:,0]
    pred=np.load(destination/f'{marker}_prediction.npz')['prediction']
    records=[]
    for unit,ix in md.groupby('unit_key',sort=True).indices.items():
        k=KMeans(n_clusters=2,n_init=20,max_iter=500,
            random_state=worker.model_def.stable_seed('HIGH',marker,unit)).fit(truth[ix,None])
        high=k.labels_==int(np.argmax(k.cluster_centers_.ravel()))
        rho=spearmanr(truth[ix],pred[ix]).statistic
        assert np.isfinite(rho),'Constant or invalid prediction must not silently enter figure'
        records.append({'marker':marker,'donor':unit,'n_cells':len(ix),'spearman':float(rho),
            'high_state_auroc':roc_auc_score(high,pred[ix]),'mae':np.abs(truth[ix]-pred[ix]).mean()})
    scores=pd.DataFrame(records)
    assert len(scores)==12 and scores.n_cells.eq(2500).all()
    scores.to_csv(destination/'donor_metrics.csv',index=False)
    print(f'{marker} finished: median rho={scores.spearman.median():.3f}, mean AUROC={scores.high_state_auroc.mean():.3f}',flush=True)

def run_many(markers):
    for marker in markers:
        subprocess.run([sys.executable,'-B',str(__file__),'worker','--marker',marker],check=True)

def summarize():
    design=init(); tables=[]; origins={}
    for marker in NORMAL:
        root=REUSE.get(marker,DATA/marker)
        d=pd.read_csv(root/'donor_metrics.csv')
        assert len(d)==12 and d.donor.nunique()==12 and d.n_cells.eq(2500).all()
        d=d[['donor','n_cells','spearman','high_state_auroc','mae']].copy()
        d.insert(0,'marker',marker); tables.append(d)
        origins[marker]={'metrics_path':str(root/'donor_metrics.csv'),
            'metrics_sha256':sha(root/'donor_metrics.csv'),
            'prediction_sha256':sha(root/f'{marker}_prediction.npz'),
            'reused':marker in REUSE}
        # All tasks evaluate the identical CODEX cohort in the identical row order.
        md=pd.read_csv(root/'codex_prediction_metadata.csv.gz')
        if marker==NORMAL[0]: reference_md=md
        else: assert md.equals(reference_md)
    full=pd.concat(tables,ignore_index=True)
    assert len(full)==168 and not full.duplicated(['marker','donor']).any()
    full.to_csv(DATA/'all_14_markers_by_donor.csv',index=False)
    summary=full.groupby('marker').agg(n_donors=('donor','nunique'),
        median_donor_spearman=('spearman','median'),mean_high_state_auroc=('high_state_auroc','mean'),
        min_donor_spearman=('spearman','min'),max_donor_spearman=('spearman','max')).loc[NORMAL]
    summary.to_csv(DATA/'all_14_markers_summary.csv')
    assert all(sha(p)==h for p,h in design['input_sha256'].items())
    save(DATA/'COMPLETION.json',{'time_utc':now(),'markers':14,'donors':12,'combinations':168,
        'identical_codex_cohort':True,'input_hashes_unchanged':True,
        'median_marker_median_donor_spearman':float(summary.median_donor_spearman.median()),
        'mean_marker_mean_donor_auroc':float(summary.mean_high_state_auroc.mean()),'origins':origins})
    print(summary.to_string(),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('mode',choices=['init','worker','run','summarize'])
    parser.add_argument('--marker'); parser.add_argument('--markers',nargs='+')
    args=parser.parse_args()
    if args.mode=='init': init()
    elif args.mode=='worker': run_worker(args.marker)
    elif args.mode=='run': run_many(args.markers)
    else: summarize()
