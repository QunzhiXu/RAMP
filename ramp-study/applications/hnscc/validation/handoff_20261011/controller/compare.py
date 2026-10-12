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
from specs import SPECIAL, result_path
COMPARISON_LOCK_SHA='f51da6a2ae9f733c520057c495fc362724096ad2aac2382abaace86c14d5393f'

def comparison_references(folder,source,base_hash):
    folder=Path(folder)
    if sha(folder/'LOCK.json')!=COMPARISON_LOCK_SHA or read(folder/'COMPLETE.json')!={'status':'complete','lock_sha256':COMPARISON_LOCK_SHA} or (folder/'FAILED.json').exists():
        raise ValueError('Comparison identity lock differs')
    value=read(folder/'LOCK.json')
    if value['base_lock_sha256']!=base_hash: raise ValueError('Comparison identity base differs')
    for rel,rec in value['files'].items():
        if record(source/rel)!=rec: raise ValueError('Comparison identity source changed')
    return value

def table(left, right, columns=None):
    a = left.copy() if isinstance(left,pd.DataFrame) else pd.read_csv(left, float_precision='round_trip', low_memory=False)
    b = right.copy() if isinstance(right,pd.DataFrame) else pd.read_csv(right, float_precision='round_trip', low_memory=False)
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

def selected_tables(old,new,rel):
    a=pd.read_csv(old,float_precision='round_trip',low_memory=False)
    b=pd.read_csv(new,float_precision='round_trip',low_memory=False)
    selection='all columns and rows in stored order'
    if 'group_fraction_scan_20260910/' in rel:
        if 'fraction' in a:
            a=a[a.fraction.eq(.3)].reset_index(drop=True)
            if set(b.fraction)!={.3}: raise ValueError('Unapproved fraction')
            selection='published fixed fraction 0.30 only'
        if rel.endswith('equal_donor_contrasts.csv'):
            a=a.drop(columns='effect_relative_to_20pct')
            selection+='; omit relative effect versus unselected 20pct arm'
        if rel.endswith('cell_group_assignments.csv.gz'):
            columns=['cell_uid','donor','compact_top5pct_clr_program','state_rank','group_30pct']
            a=a[columns].sort_values('cell_uid').reset_index(drop=True)
            b=b[columns].sort_values('cell_uid').reset_index(drop=True)
            selection='30pct assignment sorted by cell identity; other fraction arms omitted'
    if rel.endswith('panel_D/plotted_donor_values.csv'):
        columns=['marker_raw','donor','association']
        a=a[columns].sort_values(['marker_raw','donor']).reset_index(drop=True)
        b=b[columns].sort_values(['marker_raw','donor']).reset_index(drop=True)
        selection='18 current figure donor points, sorted by marker and donor'
    return a,b,selection

def normalize(obj, roots, hashes):
    if isinstance(obj,dict): return {normalize(k,roots,hashes):normalize(v,roots,hashes) for k,v in obj.items()}
    if isinstance(obj,list): return [normalize(v,roots,hashes) for v in obj]
    if isinstance(obj,str):
        value=hashes.get(obj,obj)
        for old,new in sorted(roots.items(),key=lambda x:-len(x[0])): value=value.replace(old,new)
        return value
    return obj

def json_result(old,new,rel,roots,hashes):
    a=read(old); b=read(new); excluded={}
    if rel.endswith('/screen/screen_manifest.json'):
        # Preview rendering is separate from final figure QA. Preserve the keys.
        for key in list(a['outputs']):
            if key.startswith('figures/'):
                if key not in b['outputs'] or set(a['outputs'][key])!={'sha256'} or set(b['outputs'][key])!={'sha256'}:
                    raise ValueError('Preview provenance schema differs')
                excluded[key]={'historical':a['outputs'][key],'new':b['outputs'][key]}
                a['outputs'][key]=b['outputs'][key]={'sha256':'DISPLAY_PREVIEW_NOT_VISUALLY_AUDITED'}
    differences=tree_differences(normalize(a,roots,hashes),normalize(b,roots,{}))
    return differences,excluded

def independent_summary(tree):
    base=tree/EXP; out={}
    cells=pd.read_csv(base/'survival_motivation/output/pt_cd8_cell_spatial_niche_scores.csv.gz')
    patient=pd.read_csv(base/'survival_motivation/output/patient_spatial_niche_scores.csv')
    if len(cells)!=39552 or cells.Pat_ID.nunique()!=78 or len(patient)!=78:
        raise ValueError('Observed spatial cohort differs')
    means=cells.groupby('Pat_ID').roi_adjusted_caf_macrophage_score.mean()
    saved=patient.set_index('CaseID').patient_spatial_niche_score
    delta=float(np.max(abs(means-saved)))
    if delta>2e-14: raise ValueError('Patient spatial aggregation differs')
    clinical=pd.read_csv(base/'survival_motivation/output/survival_analysis_input.csv')
    if len(clinical)!=74 or clinical.os_event.sum()!=24 or clinical.CaseID.nunique()!=74:
        raise ValueError('Primary OS cohort differs')
    models=pd.read_csv(base/'survival_motivation/output/survival_models.csv')
    if len(models)!=4: raise ValueError('Primary OS model grid differs')
    out['observed']={'cells':39552,'patients':78,'patient_mean_max_difference':delta}
    out['primary_os']={'patients':74,'deaths':24,'models':4,'groups':clinical.score_group.value_counts().to_dict(),
        'primary_model':models.iloc[0].to_dict()}
    compact=pd.read_csv(base/'biomarker_discovery/screen/final_compact_markers_top5pct.csv')
    if compact.marker_raw.tolist()!=['TIGIT--VSTM3','CD39']: raise ValueError('Compact screen differs')
    out['screen']=read(base/'biomarker_discovery/screen/screen_manifest.json')['stage_counts']
    counts=pd.read_csv(base/'molecular30/group_counts.csv'); totals=counts[['n_low','n_high','n_middle']].sum()
    if totals.tolist()!=[1439,1437,1912]: raise ValueError('Fixed molecular groups differ')
    out['molecular30']=totals.to_dict()
    returned=pd.read_csv(base/'spatial_return_survival/output/haist74_pt_cd8_tigit_cd39_return.csv.gz')
    if len(returned)!=38299 or not returned.cell_uid.is_unique or returned.patient.nunique()!=74:
        raise ValueError('Returned cohort differs')
    if not np.isfinite(returned.predicted_score).all() or not returned.predicted_score.between(0,1).all():
        raise ValueError('Returned score range differs')
    out['return']={'cells':38299,'patients':74}; out['fixed_roi']=read(base/'fixed_roi/SUMMARY.json')
    umap=pd.read_csv(base/'alignment/umap/full_94488_umap_coordinates.csv.gz')
    if len(umap)!=94488 or not umap.cell_uid.is_unique or not np.isfinite(umap[['UMAP1','UMAP2']]).all().all():
        raise ValueError('UMAP cohort differs')
    out['umap']={'cells':94488,'new_fit':True,'labels_used':False}
    return out

def compare(lockdir,run,output,reference_lock=None):
    run=Path(run).resolve(); lockdir=Path(lockdir).resolve()
    structure=verify_run(lockdir,run); lock=verify(lockdir,source=True)
    root=Path(lock['source_root']); tree=run/'tree'
    reference_lock=Path(reference_lock) if reference_lock else HERE/'comparison_references_20261011'
    comparison_references(reference_lock,root,sha(lockdir/'LOCK.json'))
    binding={'run_manifest_sha256':sha(run/'RUN_MANIFEST.json'),'lock_sha256':sha(lockdir/'LOCK.json')}
    roots={str(root):'<TREE>',str(tree):'<TREE>'}
    roots.update({path:'<TREE>/'+rel for rel,path in lock['input_paths'].items()})
    hashes={}; pairs=[]; tables=[]; npys=[]; jsons=[]; texts=[]
    refs={k:v for k,v in lock['files'].items() if v['role']=='reference'}
    for rel,rec in refs.items():
        p=result_path(tree,rel)
        if not p.is_file(): raise ValueError('Missing reproduced payload: '+rel)
        digest=sha(p)
        if digest!=rec['sha256']:
            hashes[rec['sha256']]=digest
            pairs.append({'path':rel,'historical_sha256':rec['sha256'],'new_sha256':digest})
    for rel,rec in refs.items():
        old=root/rel; p=result_path(tree,rel)
        base={'path':rel,'byte_identical':sha(p)==rec['sha256']}
        if p.name.endswith(('.csv','.csv.gz')):
            a,b,selection=selected_tables(old,p,rel)
            item={**base,**table(a,b),'scope':selection}
            if p.suffix=='.gz':
                with gzip.open(old,'rb') as x,gzip.open(p,'rb') as y: item['decompressed_bytes_equal']=x.read()==y.read()
            tables.append(item)
        elif p.suffix=='.npy':
            # The historic auxiliary cache contains all 456,567 PT cells. Only its
            # 38,299 CD8 rows were consumed by the phenotype analysis being rebuilt.
            meta=pd.read_csv(old.parent/'target_all_metadata.csv.gz',dtype={'cell_uid':str})
            rebuilt=pd.read_csv(p.parent/'target_all_metadata.csv.gz',dtype={'cell_uid':str})
            if not meta.cell_uid.is_unique or not rebuilt.cell_uid.is_unique: raise ValueError('Auxiliary identity duplicate')
            ix=pd.Index(meta.cell_uid).get_indexer(rebuilt.cell_uid)
            if (ix<0).any(): raise ValueError('Missing auxiliary CD8 identity')
            a=np.load(old,mmap_mode='r')[ix]; b=np.load(p)
            if a.shape!=b.shape or a.dtype!=b.dtype or not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError('Auxiliary CD8 representation contract differs')
            npys.append({**base,'scope':'38,299 consumed CD8 rows only, joined through separately locked historical identities',
                'historical_full_shape':list(np.load(old,mmap_mode='r').shape),'shape':list(a.shape),'dtype':str(a.dtype),
                'exact_values':bool(np.array_equal(a,b)),'changed_elements':int(np.count_nonzero(a!=b)),
                'max_abs_difference':float(np.max(abs(a-b),initial=0))})
        elif p.suffix=='.json':
            differences,excluded=json_result(old,p,rel,roots,hashes)
            jsons.append({**base,'exact_after_recorded_provenance_relocation':not differences,'differences':differences,
                          'excluded_preview_hashes':excluded})
        elif p.suffix=='.txt': texts.append({**base,'equal_except_disclosed_os_line':old.read_text()==p.read_text()})
        else: raise ValueError('Unknown payload kind')
    summary=independent_summary(tree)
    verify_run(lockdir,run); verify(lockdir,source=True)
    comparison_references(reference_lock,root,sha(lockdir/'LOCK.json'))
    if binding!={'run_manifest_sha256':sha(run/'RUN_MANIFEST.json'),'lock_sha256':sha(lockdir/'LOCK.json')}:
        raise ValueError('Run changed during comparison')
    exact=all(t['exact_values'] for t in tables+npys) and all(t['exact_after_recorded_provenance_relocation'] for t in jsons) and all(t['equal_except_disclosed_os_line'] for t in texts)
    report={'status':'exact_with_disclosed_reuse_scope' if exact else 'completed_with_differences',**binding,
        'comparison_lock_sha256':COMPARISON_LOCK_SHA,
        'structure':structure,'reference_files_compared':len(refs),'byte_identical_files':sum(t['byte_identical'] for t in tables+npys+jsons+texts),
        'tables':tables,'arrays':npys,'npzs':[],'jsons':jsons,'texts':texts,'provenance_hash_pairs':pairs,
        'independent_summary':summary,'controller':record(Path(__file__)),'execution_controllers':codes(),
        'scope':'PT74 current figure numerical chain; fixed 30pct and ROI; accepted RAMP reuse; not final figure QA'}
    out=reserve(output,[root,lockdir,run]); write(out/'report.json',report)
    write(out/'COMPLETE.json',{'status':'complete','report_sha256':sha(out/'report.json')})
    return {'status':report['status'],'files':len(refs),'byte_identical_files':report['byte_identical_files']}

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--lockdir',required=True); p.add_argument('--run',required=True); p.add_argument('--out',required=True)
    p.add_argument('--reference-lock')
    a=p.parse_args(); print(compare(a.lockdir,a.run,a.out,a.reference_lock))


