"""Read-only comparison. Differences are reported, never tuned or overwritten."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from common import STORY, read, write, sha, reserve, record
from specs import A, R
from prepare import verify
from study import verify_run

def table(left, right, columns=None):
    a = pd.read_csv(left, float_precision='round_trip', low_memory=False)
    b = pd.read_csv(right, float_precision='round_trip', low_memory=False)
    if columns is not None:
        if not set(columns).issubset(a.columns) or not set(columns).issubset(b.columns):
            raise ValueError('Missing consumed column')
        a, b = a[columns], b[columns]
    if a.shape != b.shape or a.columns.tolist() != b.columns.tolist():
        raise ValueError('Table shape/column order differs: ' + str(left))
    differences = {}; numeric_max = 0.
    for name in a.columns:
        x,y=a[name],b[name]
        if not x.isna().equals(y.isna()):
            raise ValueError('Missingness differs: ' + str(left) + ':' + name)
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            keep=x.notna(); xx=x[keep].to_numpy(float); yy=y[keep].to_numpy(float)
            if not np.isfinite(xx).all() or not np.isfinite(yy).all():
                raise ValueError('Nonfinite table')
            changed=int(np.count_nonzero(xx!=yy)); delta=float(np.max(abs(xx-yy),initial=0))
            numeric_max=max(numeric_max,delta)
            if changed: differences[name]={'changed':changed,'max_abs_difference':delta}
        elif not x.fillna('').astype(str).equals(y.fillna('').astype(str)):
            differences[name]={'changed':int((x.fillna('').astype(str)!=y.fillna('').astype(str)).sum()),'nonnumeric':True}
    return {'rows':len(a),'columns':a.columns.tolist(),'exact_values':not differences,
            'max_abs_difference':numeric_max,'differing_columns':differences}

def array(left,right):
    a=np.load(left,allow_pickle=False); b=np.load(right,allow_pickle=False)
    if a.shape!=b.shape or a.dtype!=b.dtype or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Array contract differs')
    return {'shape':list(a.shape),'dtype':str(a.dtype),'exact_values':bool(np.array_equal(a,b)),
            'changed_elements':int(np.count_nonzero(a!=b)),
            'max_abs_difference':float(np.max(abs(a.astype(float)-b.astype(float)),initial=0))}

def tree_differences(a,b,path=''):
    if isinstance(a,dict) and isinstance(b,dict):
        if set(a)!=set(b): return [{'path':path,'key_set_difference':sorted(set(a)^set(b))}]
        return [d for k in a for d in tree_differences(a[k],b[k],path+'/'+str(k))]
    if isinstance(a,list) and isinstance(b,list):
        if len(a)!=len(b): return [{'path':path,'lengths':[len(a),len(b)]}]
        return [d for i,(x,y) in enumerate(zip(a,b)) for d in tree_differences(x,y,path+'/'+str(i))]
    if a==b: return []
    item={'path':path,'historical':a,'new':b}
    if isinstance(a,(int,float)) and isinstance(b,(int,float)):
        if not np.isfinite(a) or not np.isfinite(b): raise ValueError('Nonfinite comparison')
        item['abs_difference']=abs(a-b)
    return [item]

def normalize(obj, root, hashes):
    if isinstance(obj,dict): return {normalize(k,root,hashes):normalize(v,root,hashes) for k,v in obj.items()}
    if isinstance(obj,list): return [normalize(v,root,hashes) for v in obj]
    if isinstance(obj,str): return hashes.get(obj,obj.replace(str(root),'<TREE>'))
    return obj

def compare(lockdir,run,output):
    run=Path(run).resolve(); lockdir=Path(lockdir).resolve()
    structure=verify_run(lockdir,run); lock=verify(lockdir,source=True)
    root=Path(lock['source_root']); tree=run/'tree'
    binding={'run_manifest_sha256':sha(run/'RUN_MANIFEST.json'),'lock_sha256':sha(lockdir/'LOCK.json')}
    hash_map={}; hash_pairs=[]
    for rel,r in lock['files'].items():
        p=tree/rel
        if p.is_file():
            newsha=sha(p)
            if newsha!=r['sha256']:
                hash_map[r['sha256']]=newsha
                hash_pairs.append({'path':rel,'historical_sha256':r['sha256'],'new_sha256':newsha})
    skipped={STORY+'/inputs/observed_survival_results.csv',A+'/spatial_figure/manifest.json',R+'/survival_results_20_vs_30.json'}
    checked=[]; tables=[]; arrays=[]; jsons=[]; texts=[]
    for rel,r in lock['files'].items():
        if r['role']!='reference' or rel in skipped: continue
        p=tree/rel; old=root/rel
        if not p.is_file(): raise ValueError('Missing reproduced reference: '+rel)
        base={'path':rel,'byte_identical':sha(p)==r['sha256']}
        if p.suffix in ('.csv','.gz'):
            columns=None
            if rel.endswith('/inputs/spatial_cell_metadata.csv.gz'):
                columns=pd.read_csv(p,nrows=0).columns.tolist()
                if len(columns)!=10: raise ValueError('Consumed spatial fields changed')
            if rel.endswith('/inputs/observed_patient_level_features.csv'):
                columns=['patient','n_all_cells','n_cd8','n_tcae','cd8_per_all','tcae_per_all',
                         'observed_ki67_tcae_per_all','observed_ki67_contact30_tcae_per_all',
                         'observed_ki67_contact30_fraction_within_tcae','Known_Deceased','Survival_from_Rx_Start',
                         'Response','irRC','Treatment','Age_last_alive','Sex','Tissue_Source_Simplified','Lymphocyte_Infiltration']
            tables.append({**base,**table(old,p,columns)})
        elif p.suffix=='.npy': arrays.append({**base,**array(old,p)})
        elif p.suffix=='.json':
            diffs=tree_differences(normalize(read(old),root,hash_map),normalize(read(p),tree,{}))
            jsons.append({**base,'exact_after_recorded_provenance_relocation':not diffs,'differences':diffs})
        else: texts.append({**base,'text_equal':old.read_text()==p.read_text()})
        checked.append(rel)
    survival=read(tree/STORY/'analysis/observed_survival/survival_30um.json')
    historical=read(root/R/'survival_results_20_vs_30.json')['results']['30um']
    survival_diff=tree_differences(historical,survival)
    if any('abs_difference' not in d or d['abs_difference']>1e-12 for d in survival_diff):
        raise ValueError('Historical 30um survival differs beyond numerical tolerance')
    published=pd.read_csv(root/STORY/'inputs/observed_survival_results.csv',float_precision='round_trip')
    row=published.set_index('feature').loc['observed_ki67_contact30_tcae_per_all']
    fields={'logrank_p':survival['logrank_p'],'cox_hr_per_sd':survival['cox_continuous_per_sd']['hr'],
            'cox_p':survival['cox_continuous_per_sd']['p']}
    published_diff={k:abs(float(row[k])-v) for k,v in fields.items()}
    if max(published_diff.values())>1e-12: raise ValueError('Published survival differs')
    roi=read(tree/STORY/'analysis/fixed_roi.json'); oldroi=read(root/A/'spatial_figure/manifest.json')
    for k,v in roi.items():
        expected=oldroi['fixed_pool_cutoff_comparison'][k] if k in ('top15','top20') else oldroi['recommended_top20'][k]
        if v!=expected: raise ValueError('Fixed ROI differs: '+k)
    # Independently calculate primary discrimination metrics from new predictions.
    from sklearn.metrics import roc_auc_score,average_precision_score
    pred=pd.read_csv(tree/A/'outer_patient_predictions.csv.gz'); metrics=pd.read_csv(tree/A/'patient_metrics.csv')
    auc=[]; ap=[]
    for patient,g in pred.groupby('patient',sort=True):
        if len(g)>=20 and g.target_30um.nunique()==2:
            auc.append(float(roc_auc_score(g.target_30um,g.model_score)))
            ap.append(float(average_precision_score(g.target_30um,g.model_score)/g.target_30um.mean()))
    summary=read(tree/A/'summary.json')
    if len(auc)!=14 or abs(np.median(auc)-summary['median_AUROC'])>1e-14 or abs(np.median(ap)-summary['median_AUPRC_over_prevalence'])>1e-14:
        raise ValueError('Independent heldout metric check differs')
    verify_run(lockdir,run); verify(lockdir,source=True)
    if binding!={'run_manifest_sha256':sha(run/'RUN_MANIFEST.json'),'lock_sha256':sha(lockdir/'LOCK.json')}:
        raise ValueError('Run changed during comparison')
    exact=all(t['exact_values'] for t in tables+arrays) and all(j['exact_after_recorded_provenance_relocation'] for j in jsons) and all(t['text_equal'] for t in texts)
    report={'status':'exact_with_disclosed_reconstruction_scope' if exact else 'completed_with_differences',**binding,
            'structure':structure,'reference_files_compared':len(checked),'tables':tables,'arrays':arrays,'jsons':jsons,'texts':texts,
            'provenance_hash_pairs':hash_pairs,'survival':survival,'survival_tree_differences':survival_diff,
            'published_survival_differences':published_diff,'fixed_roi':roi,
            'independent_prediction_metrics':{'patients':len(auc),'median_AUROC':float(np.median(auc)),
                                              'median_AUPRC_over_prevalence':float(np.median(ap))},
            'partial_reference_contracts':{'spatial_table':'only 10 consumed fields; 3 legacy predicted fields omitted',
               'patient_table':'18 observed/clinical fields compared, not 38-column historical exploration',
               'survival':'historical R 30um tree plus main published row; not all 12 legacy feature screens'},
            'controller':record(Path(__file__))}
    out=reserve(output,[root,lockdir,run]); write(out/'report.json',report)
    write(out/'COMPLETE.json',{'status':'complete','report_sha256':sha(out/'report.json')})
    return {'status':report['status'],'files':len(checked),'survival_logrank_p':survival['logrank_p']}

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--lockdir',required=True); p.add_argument('--run',required=True); p.add_argument('--out',required=True)
    a=p.parse_args(); print(compare(a.lockdir,a.run,a.out))
