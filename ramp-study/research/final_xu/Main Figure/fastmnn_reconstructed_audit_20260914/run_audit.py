"""Read-only reconstruction audit of the four frozen standard PCA-fastMNN runs.

All writes are confined to this directory. Reconstructed values are diagnostic
outputs, not calibrated protein abundance. No fitting, tuning, or relabelling.
"""
from pathlib import Path
import os, sys, json, hashlib, time, re, tarfile
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CASES = ['bone_marrow', 'tnbc', 'melanoma', 'lung_cancer']
STAGES = ['Raw', 'Method input', 'Cosine normalized', 'PCA reconstructed', 'fastMNN reconstructed']
KEYS = ['case', 'dataset', 'unit_key', 'celltype', 'marker']
INVALID = re.compile(r'non-joint|unclassified|technical|unannotated|unknown|ambiguous|missing|\bmix\b', re.I)
READS = {}

def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def track(p, expected=None):
    p = Path(p).resolve()
    digest = sha(p)
    if expected is not None:
        assert digest == expected, (p, digest, expected)
    READS[str(p)] = digest
    return p

def read_csv(p, expected=None, **kwargs):
    return pd.read_csv(track(p, expected), float_precision='round_trip', low_memory=False, **kwargs)

def ranks(a):
    return 100 * rankdata(a, method='average', axis=0) / (len(a) + 1.)

def auc(a, b):
    r = rankdata(np.r_[a, b], method='average')
    return float((r[:len(a)].sum() - len(a)*(len(a)+1)/2)/(len(a)*len(b)))

def column_rho(a, b):
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    denominator = np.sqrt((a*a).sum(axis=0)*(b*b).sum(axis=0))
    return np.divide((a*b).sum(axis=0), denominator,
                     out=np.full(a.shape[1], np.nan), where=denominator > 0)

def main():
    started = time.time()
    track(__file__)
    source = track(ROOT/'algorithm/fastmnn/vendor/batchelor_1.18.1.tar.gz',
        'a58590880cf00dd2ff7c9fc75ee611e4d4cc5bae98246a1878f8d7d1da6c8dfc')
    with tarfile.open(source) as t:
        code = t.extractfile('batchelor/R/convertPCsToSCE.R').read().decode()
        assert 'LowRankMatrix(rot, corrected.df$corrected)' in code
        (HERE/'official_convertPCsToSCE.R').write_text(code)
    expectations = read_csv(ROOT/'benchmark_results/exact_value_artifacts/config/canonical_expectations.csv')
    quartile_rows, integrity_rows, contrast_rows, resolution_rows, reconstruction = [], [], [], [], []
    for case in CASES:
        out = HERE/case
        out.mkdir(exist_ok=True)
        base = ROOT/'benchmark_results'/case
        inp = base/'ramp/alignment/input'
        meta = read_csv(inp/'cell_metadata.csv.gz', dtype={'cell_id':str})
        markers = track(inp/'markers.txt').read_text().strip().splitlines()
        raw = np.asarray(np.load(track(inp/'raw.npy')), dtype=np.float64)
        ids = meta.cell_id.to_numpy()
        assert not meta.cell_id.duplicated().any()
        fm = base/'fastmnn/alignment/data'
        manifest = json.loads(track(fm/'manifest.json').read_text())
        hashes = {r['file']:r['sha256'] for r in manifest['output_files']}
        assert manifest['status'] == 'complete'
        assert manifest['input']['markers'] == markers
        def fcsv(name):
            return read_csv(fm/name, hashes[name])
        embedding = fcsv('fastmnn_corrected_embedding.csv.gz')
        rotation = fcsv('fastmnn_pca_rotation.csv.gz')
        centers = fcsv('fastmnn_pca_centers.csv')
        values = read_csv(manifest['input']['path'], manifest['input']['sha256'], dtype={'cell_id':str})
        assert np.array_equal(ids, embedding.cell_id.to_numpy())
        assert np.array_equal(ids, values.cell_id.to_numpy())
        assert rotation.marker.tolist() == markers == centers.marker.tolist()
        x = values[markers].to_numpy(np.float64)
        z = embedding.drop(columns='cell_id').to_numpy(np.float64)
        rotation_values = rotation.drop(columns='marker').to_numpy(np.float64)
        center = centers['center'].to_numpy(np.float64)
        semantics = json.loads(track(fm/'method_semantics.json',hashes['method_semantics.json']).read_text())
        assert semantics['cosine_normalization'] and not semantics['marker_space_d_NA_used']
        assert raw.shape == x.shape == (len(meta),len(markers))
        assert z.shape[1] == rotation_values.shape[1] == len(markers)-1
        orthogonality = float(np.max(np.abs(rotation_values.T @ rotation_values - np.eye(z.shape[1]))))
        assert orthogonality < 1e-10
        cosine = x / np.maximum(1e-8, np.sqrt((x*x).sum(axis=1)))[:,None]
        # multiBatchPCA defaults weight platforms equally when computing centers.
        calculated_center = np.mean([cosine[g].mean(axis=0) for g in meta.groupby('dataset').indices.values()],axis=0)
        center_error = float(np.max(np.abs(calculated_center-center)))
        assert center_error < 1e-10, (case,center_error)
        uncorrected_pc = (cosine-center) @ rotation_values
        pca = uncorrected_pc @ rotation_values.T
        corrected = z @ rotation_values.T
        # Official LowRankMatrix is genes x cells; export its transpose.
        assert np.isfinite(corrected).all()
        manual = np.array([[np.sum(z[i]*rotation_values[j]) for j in range(len(markers))]
                           for i in range(min(20,len(z)))])
        assert np.allclose(manual, corrected[:len(manual)],atol=1e-12,rtol=1e-12)
        arrays = dict(zip(STAGES,[raw,x,cosine,pca,corrected]))
        for key,a in [('reconstructed',corrected),('cosine_normalized',cosine),('pca_reconstructed',pca)]:
            np.save(out/f'{key}.npy',a)
        meta.to_csv(out/'cell_metadata.csv.gz',index=False)
        (out/'markers.txt').write_text('\n'.join(markers)+'\n')
        reconstruction.append(dict(case=case,n_cells=len(meta),n_markers=len(markers),n_pcs=z.shape[1],
            orthogonality_max_error=orthogonality,center_max_error=center_error,
            output_min=float(corrected.min()),output_max=float(corrected.max())))
        labels = meta.celltype.fillna('Missing').astype(str).to_numpy()
        for unit, rows in meta.groupby('unit_key',sort=True).indices.items():
            ds = meta.iloc[rows].dataset.unique()
            assert len(ds)==1
            percentiles = {s:ranks(a[rows]) for s,a in arrays.items()}
            for si,s in enumerate(STAGES[1:],start=1):
                rr, pp = percentiles[s],percentiles[STAGES[si-1]]
                rho_raw, rho_previous = column_rho(percentiles['Raw'],rr),column_rho(pp,rr)
                for j,marker in enumerate(markers):
                    integrity_rows.append(dict(case=case,dataset=ds[0],unit_key=unit,marker=marker,stage=s,
                        n_cells=len(rows),spearman_raw_to_stage=rho_raw[j],spearman_previous_to_stage=rho_previous[j],
                        mean_abs_percentile_change_raw=float(np.abs(rr[:,j]-percentiles['Raw'][:,j]).mean()),
                        mean_abs_percentile_change_previous=float(np.abs(rr[:,j]-pp[:,j]).mean())))
            for label in sorted(set(labels[rows])):
                local = np.flatnonzero(labels[rows]==label)
                eligible = len(local)>=50 and INVALID.search(label) is None
                qq={s:np.quantile(v[local],[.25,.5,.75],axis=0,method='linear') for s,v in percentiles.items()}
                for si,s in enumerate(STAGES[1:],start=1):
                    for j,marker in enumerate(markers):
                        r=dict(case=case,dataset=ds[0],unit_key=unit,celltype=label,marker=marker,stage=s,
                               n_cells=len(local),n_sample_cells=len(rows),included_in_plot=eligible)
                        for k,q in enumerate(['q25','median','q75']):
                            r['raw_'+q]=qq['Raw'][k,j]
                            r['value_'+q]=qq[s][k,j]
                            r['delta_'+q+'_pp']=qq[s][k,j]-qq['Raw'][k,j]
                            r['abs_delta_'+q+'_pp']=abs(r['delta_'+q+'_pp'])
                            r['previous_stage_delta_'+q+'_pp']=qq[s][k,j]-qq[STAGES[si-1]][k,j]
                        quartile_rows.append(r)
            # Canonical rules reused as-is, now evaluated separately per sample.
            for rule in expectations[expectations.cancer.eq(case)].itertuples(index=False):
                if rule.marker not in markers: continue
                if rule.dataset_allowlist!='*' and ds[0] not in str(rule.dataset_allowlist).split(';'): continue
                pos=np.flatnonzero(np.isin(labels[rows],str(rule.positive_labels).split(';')))
                j=markers.index(rule.marker)
                for label in str(rule.negative_labels).split(';'):
                    neg=np.flatnonzero(labels[rows]==label)
                    common=dict(case=case,dataset=ds[0],unit_key=unit,marker=rule.marker,rule_id=rule.rule_id,
                        tier=rule.tier,positive_labels=rule.positive_labels,negative_label=label,n_positive=len(pos),n_negative=len(neg))
                    resolution_rows.append(dict(**common,status='evaluated' if min(len(pos),len(neg))>=50 else 'fewer than 50 cells'))
                    if min(len(pos),len(neg))<50: continue
                    r=dict(**common)
                    for s,a in arrays.items():
                        v=a[rows,j]
                        r[s+'_auc']=auc(v[pos],v[neg])
                        r[s+'_positive_median_percentile']=float(np.median(percentiles[s][pos,j]))
                        r[s+'_negative_median_percentile']=float(np.median(percentiles[s][neg,j]))
                    r['auc_loss_raw_to_corrected']=r['Raw_auc']-r['fastMNN reconstructed_auc']
                    r['auc_loss_pca_to_corrected']=r['PCA reconstructed_auc']-r['fastMNN reconstructed_auc']
                    r['raw_auc_ge_080']=r['Raw_auc']>=.8
                    contrast_rows.append(r)
        print(case, reconstruction[-1],f'{time.time()-started:.1f}s',flush=True)
    q=pd.DataFrame(quartile_rows)
    q.to_csv(HERE/'fastmnn_quartile_shifts_all.csv.gz',index=False)
    primary=q[q.included_in_plot].copy()
    assert len(primary[primary.stage.eq(STAGES[-1])])==6173
    old=read_csv(ROOT/'Main Figure/celltype_percentile_shift_20260914/celltype_quartile_shifts_display.csv.gz')
    common=set(map(tuple,old[old.method.eq('RAMP')][KEYS].to_numpy()))
    for s in STAGES[1:]:
        assert set(map(tuple,primary[primary.stage.eq(s)][KEYS].to_numpy()))==common
    new=primary[primary.stage.eq(STAGES[-1])].rename(columns={'stage':'method'})
    combined=pd.concat([old,new],ignore_index=True)
    combined.to_csv(HERE/'comparison_celltype_shifts.csv.gz',index=False)
    def summarize(d,key,value):
        return d.groupby(key)[value].agg(n='size',mean='mean',median='median',q75=lambda x:x.quantile(.75),
            q90=lambda x:x.quantile(.9),maximum='max',n_gt_10=lambda x:int((x>10).sum()),n_gt_20=lambda x:int((x>20).sum())).reset_index()
    summary=summarize(combined,['method'],'abs_delta_median_pp')
    summary.to_csv(HERE/'comparison_summary.csv',index=False)
    summarize(combined,['case','method'],'abs_delta_median_pp').to_csv(HERE/'comparison_by_benchmark.csv',index=False)
    summarize(primary,['case','stage'],'abs_delta_median_pp').to_csv(HERE/'stage_summary.csv',index=False)
    summarize(primary,['stage'],'abs_delta_median_pp').to_csv(HERE/'stage_summary_overall.csv',index=False)
    integrity=pd.DataFrame(integrity_rows)
    integrity.to_csv(HERE/'sample_marker_rank_integrity.csv.gz',index=False)
    integrity.groupby(['case','stage']).agg(n=('marker','size'),median_rho_raw=('spearman_raw_to_stage','median'),
        median_rho_previous=('spearman_previous_to_stage','median')).to_csv(HERE/'stage_rank_summary.csv')
    contrasts=pd.DataFrame(contrast_rows)
    contrasts.sort_values('auc_loss_raw_to_corrected',ascending=False).to_csv(HERE/'canonical_contrasts_all.csv',index=False)
    qualified=contrasts[contrasts.raw_auc_ge_080].sort_values('auc_loss_raw_to_corrected',ascending=False)
    qualified.to_csv(HERE/'canonical_contrasts_raw_auc_ge_080.csv',index=False)
    pd.DataFrame(resolution_rows).to_csv(HERE/'canonical_rule_resolution.csv',index=False)
    pd.DataFrame(reconstruction).to_csv(HERE/'reconstruction_checks.csv',index=False)
    manifest=dict(cases=CASES,scope='Four original benchmarks; same frozen cells and marker ordering as the current Figure 1',
        reconstruction='Cell x marker output = corrected_embedding @ pca_rotation.T; official batchelor 1.18.1 convertPCsToSCE; no center added back, no inverse cosine scaling.',
        before_mnn='PCA-only diagnostic = ((method_input / max(L2,1e-8) - saved_centers) @ rotation) @ rotation.T',
        interpretation='Diagnostic low-rank reconstructed expression; not calibrated raw protein intensities. Stage effects are not additive on ranks or AUROC.',
        primary_metric='Absolute median percentile shift (pp), ranks within all cells of each sample-marker; eligible cell types >=50 cells; identical 6173 combinations to current Figure 1.',
        aggregation='One observation per sample-celltype-marker; not sample-equal or benchmark-equal. No significance tests.',
        canonical_scan='All existing canonical rules evaluated per sample, both positive and negative groups >=50 cells. All results saved; raw AUROC >=0.80 used only to shortlist presentation examples.',
        alignment_rerun=False,legacy_d_NA_used=False,n_sample_marker_combinations=len(integrity)//4,
        n_sample_celltype_marker_combinations=len(primary)//4,n_samples=primary[['case','unit_key']].drop_duplicates().shape[0],
        all_input_sha256=READS,elapsed_seconds=time.time()-started)
    assert all(sha(p)==digest for p,digest in READS.items())
    manifest['frozen_sources_unchanged']=True
    (HERE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(summary.to_string(index=False),flush=True)
    print(qualified[['case','unit_key','marker','positive_labels','negative_label','Raw_auc','PCA reconstructed_auc',
        'fastMNN reconstructed_auc','auc_loss_raw_to_corrected']].head(16).to_string(index=False),flush=True)

if __name__=='__main__': main()
