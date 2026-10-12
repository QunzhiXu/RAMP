"""Full file and numerical comparison; known run metadata differences are explicit."""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np
import pandas as pd
from common import HERE, EXP, STORY, read, write, sha, reserve, record, codes
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


def canonical(obj, root, hashes):
    if isinstance(obj,dict): return {canonical(k,root,hashes):canonical(v,root,hashes) for k,v in obj.items()}
    if isinstance(obj,list): return [canonical(v,root,hashes) for v in obj]
    if isinstance(obj,str):
        value=hashes.get(obj,obj).replace(str(root),'<TREE>')
        return value.replace(STORY+'/test/new_senosain_20260921',EXP)
    return obj

def scientific_json(obj,rel,root,hashes):
    obj=json.loads(json.dumps(obj)); excluded={}
    field=None
    if rel in [EXP+'/validation/COMPLETE.json',EXP+'/cords/COMPLETE.json']: field='seconds'
    if rel==EXP+'/analysis/MOLECULAR_FREEZE.json': field='time_utc'
    if field:
        excluded[field]=obj.pop(field)
        if field=='seconds' and (isinstance(excluded[field],bool) or not isinstance(excluded[field],(int,float)) or not np.isfinite(excluded[field]) or excluded[field]<0):
            raise ValueError('Invalid duration')
    if '/patients/' in rel:
        parts=obj['signature'].split('|',3)
        if len(parts)!=4: raise ValueError('Invalid patient signature')
        obj['signature']={'code_hashes':parts[:3],'input_hashes':json.loads(parts[3])}
    return canonical(obj,root,hashes),excluded

def arrays(a,b):
    if a.shape!=b.shape or a.dtype!=b.dtype: raise ValueError('Array dtype/shape differs')
    changed=0; delta=0.
    # Limit transient copies for the all-cell arrays.
    for start in range(0,len(a),100000):
        x=a[start:start+100000]; y=b[start:start+100000]
        if not np.isfinite(x).all() or not np.isfinite(y).all(): raise ValueError('Nonfinite array')
        changed+=int(np.count_nonzero(x!=y))
        delta=max(delta,float(np.max(abs(x.astype(float)-y.astype(float)),initial=0)))
    return {'shape':list(a.shape),'dtype':str(a.dtype),'exact_values':changed==0,
            'changed_elements':changed,'max_abs_difference':delta}

def independent_summary(tree):
    base=tree/EXP
    result={}
    for target,count,patients in [('validation',693693,28),('cords',2849385,527)]:
        md=pd.read_csv(base/target/'CELL_METADATA.csv.gz',dtype={'patient_id':str})
        pred=np.load(base/target/'PREDICTIONS.npy',mmap_mode='r')
        if pred.shape!=(count,4) or len(md)!=count or not md.source_row_id.is_unique or md.patient_id.nunique()!=patients:
            raise ValueError('Projection identity differs')
        if not np.isfinite(pred).all() or pred.min()<0 or pred.max()>1: raise ValueError('Prediction range differs')
        audit=pd.read_csv(base/target/'PROJECTION_AUDIT.csv')
        if len(audit)!=patients or audit.n_cells.sum()!=count or audit.rank_inversions.sum()!=0 or audit.max_displacement.max()>.750000001:
            raise ValueError('Projection audit differs')
        result[target]={'cells':count,'patients':patients,'configs':4,'max_displacement':float(audit.max_displacement.max()),
                        'max_anchor_error':float(audit.anchor_error.max()),'prediction_min':float(pred.min()),'prediction_max':float(pred.max())}
    # Independently reduce the newly calculated Cords predictions, without importing
    # the frozen analysis script or looking at historical patient scores.
    sample=pd.read_csv(tree/STORY/'data_cords/reference/THRESHOLD_CAF_CELL_IDS.csv.gz').source_row_id
    selected=md.source_row_id.isin(set(sample)).to_numpy()
    caf=md.cell_category.eq('Fibroblast').to_numpy(); vcaf=caf & md.cell_type.eq('vCAF').to_numpy()
    if selected.sum()!=150024 or caf.sum()!=530184 or vcaf.sum()!=52328 or not caf[selected].all():
        raise ValueError('Molecular aggregation denominator differs')
    configurations=['RAMP_k30','RAMP_k100','baseline_k30','baseline_k100']
    thresholds=np.median(pred[selected],axis=0)
    saved_cuts=pd.read_csv(base/'analysis/NEW_HIGH_STATE_THRESHOLDS.csv',float_precision='round_trip').set_index('config')
    if not np.array_equal(saved_cuts.loc[configurations,'threshold'].to_numpy(),thresholds):
        raise ValueError('Molecular thresholds differ')
    saved_scores=pd.read_csv(base/'analysis/NEW_ALL_PATIENT_SCORES.csv',float_precision='round_trip').set_index(['config','patient_id'])
    max_delta=0.; checked=0
    for patient,ix in md.groupby('patient_id',sort=True).indices.items():
        vi=ix[vcaf[ix]]; si=ix[selected[ix]]; n_caf=int(caf[ix].sum())
        if not n_caf: continue
        for j,config in enumerate(configurations):
            row=saved_scores.loc[(config,patient)]
            if row.n_full_caf!=n_caf or row.n_vcaf!=len(vi): raise ValueError('Patient counts differ')
            values={'vcaf_fraction':len(vi)/n_caf,
                    'all_mean':float(np.mean(pred[vi,j])) if len(vi) else np.nan,
                    'all_high':float(np.mean(pred[vi,j]>=thresholds[j])) if len(vi) else np.nan,
                    'current_global_CAF_median':float(np.median(pred[si,j])) if len(si) else np.nan}
            for key,value in values.items():
                if np.isnan(value)!=pd.isna(row[key]): raise ValueError('Patient score missingness differs')
                if np.isfinite(value): max_delta=max(max_delta,abs(value-float(row[key])))
            checked+=1
    if max_delta>2e-14: raise ValueError('Independent molecular aggregation differs')
    result['independent_molecular_aggregation']={'checked_patient_config_rows':checked,'max_abs_difference':max_delta,
        'threshold_sample_cells':int(selected.sum()),'caf_cells':int(caf.sum()),'vcaf_cells':int(vcaf.sum()),'thresholds':thresholds.tolist()}
    groups=pd.read_csv(base/'analysis/PATIENT_GROUPS.csv',float_precision='round_trip')
    current=groups[groups.version.eq('new')]
    if len(current)!=280 or current.patient_id.nunique()!=280 or current.event.sum()!=96 or set(current.group)!= {'Lower','Higher'}:
        raise ValueError('Clinical cohort differs')
    high=current.all_mean>=current.all_mean.median()
    if not np.array_equal(high.to_numpy(),current.high_HLA.to_numpy().astype(bool)):
        raise ValueError('Patient grouping differs')
    result['clinical']={'patients':280,'events':96,'higher':int(high.sum()),'lower':int((~high).sum()),'median_cut':float(current.all_mean.median())}
    main=pd.read_csv(base/'analysis/COX_TERMS.csv',float_precision='round_trip')
    main=main[main.version.eq('new')&main.config.eq('RAMP_k30')&main.state.eq('mean')&main.cohort.eq('all280')&main.term.eq('state_z')]
    if len(main)!=1: raise ValueError('Primary Cox row differs')
    result['primary_cox']=main.iloc[0].to_dict()
    return result

def compare(lockdir,run,output):
    run=Path(run).resolve(); lockdir=Path(lockdir).resolve()
    structure=verify_run(lockdir,run); lock=verify(lockdir,source=True)
    root=Path(lock['source_root']); tree=run/'tree'
    binding={'run_manifest_sha256':sha(run/'RUN_MANIFEST.json'),'lock_sha256':sha(lockdir/'LOCK.json')}
    # Every hash relocation refers to a separately checked payload; none replaces its comparison.
    hashes={}; pairs=[]; checked=[]; tables=[]; npys=[]; npzs=[]; jsons=[]; texts=[]
    refs={k:v for k,v in lock['files'].items() if v['role']=='reference'}
    for rel,rec in refs.items():
        p=tree/rel
        if not p.is_file(): raise ValueError('Missing reproduced payload: '+rel)
        digest=sha(p)
        if digest!=rec['sha256']:
            hashes[rec['sha256']]=digest
            pairs.append({'path':rel,'historical_sha256':rec['sha256'],'new_sha256':digest})
    for rel,rec in refs.items():
        old=root/rel; p=tree/rel
        base={'path':rel,'byte_identical':sha(p)==rec['sha256']}
        if p.suffix in ('.csv','.gz'):
            item={**base,**table(old,p)}
            if p.suffix=='.gz':
                with gzip.open(old,'rb') as a,gzip.open(p,'rb') as b: item['decompressed_bytes_equal']=a.read()==b.read()
            tables.append(item)
        elif p.suffix=='.npy':
            npys.append({**base,**arrays(np.load(old,mmap_mode='r',allow_pickle=False),np.load(p,mmap_mode='r',allow_pickle=False))})
        elif p.suffix=='.npz':
            with np.load(old,allow_pickle=False) as a,np.load(p,allow_pickle=False) as b:
                if a.files!=b.files: raise ValueError('NPZ member order differs')
                npzs.append({**base,'arrays':{k:arrays(a[k],b[k]) for k in a.files}})
        elif p.suffix=='.json':
            a,ae=scientific_json(read(old),rel,root,hashes); b,be=scientific_json(read(p),rel,tree,{})
            differences=tree_differences(a,b)
            jsons.append({**base,'exact_after_recorded_provenance_relocation':not differences,'differences':differences,
                          'excluded_runtime_fields':{'historical':ae,'new':be}})
        elif p.name=='R_SESSION_INFO.txt':
            a=old.read_text().splitlines(); b=p.read_text().splitlines()
            differences=[{'line':i+1,'historical':x,'new':y} for i,(x,y) in enumerate(zip(a,b)) if x!=y]
            allowed=len(a)==len(b) and all(d['historical']=='Running under: macOS 26.6.1' and d['new']=='Running under: macOS 26.7.1' for d in differences)
            texts.append({**base,'equal_except_disclosed_os_line':allowed,'differences':differences,'line_counts':[len(a),len(b)]})
        else: raise ValueError('Unknown payload kind: '+rel)
        checked.append(rel)
    summary=independent_summary(tree)
    verify_run(lockdir,run); verify(lockdir,source=True)
    if binding!={'run_manifest_sha256':sha(run/'RUN_MANIFEST.json'),'lock_sha256':sha(lockdir/'LOCK.json')}:
        raise ValueError('Run changed during comparison')
    exact=all(t['exact_values'] for t in tables+npys) and all(a['exact_values'] for t in npzs for a in t['arrays'].values()) and all(t['exact_after_recorded_provenance_relocation'] for t in jsons) and all(t['equal_except_disclosed_os_line'] for t in texts)
    report={'status':'exact_with_disclosed_reuse_scope' if exact else 'completed_with_differences',**binding,
        'structure':structure,'reference_files_compared':len(checked),'byte_identical_files':sum(t['byte_identical'] for t in tables+npys+npzs+jsons+texts),
        'tables':tables,'arrays':npys,'npzs':npzs,'jsons':jsons,'texts':texts,'provenance_hash_pairs':pairs,
        'independent_summary':summary,'controller':record(Path(__file__)),'execution_controllers':codes(),
        'scope':'Current Senosain140k release; accepted RAMP reused; frozen full-cell baselines and auxiliary measurement tables; no final figure QA'}
    out=reserve(output,[root,lockdir,run]); write(out/'report.json',report)
    write(out/'COMPLETE.json',{'status':'complete','report_sha256':sha(out/'report.json')})
    return {'status':report['status'],'files':len(checked),'byte_identical_files':report['byte_identical_files']}

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--lockdir',required=True); p.add_argument('--run',required=True); p.add_argument('--out',required=True)
    a=p.parse_args(); print(compare(a.lockdir,a.run,a.out))
