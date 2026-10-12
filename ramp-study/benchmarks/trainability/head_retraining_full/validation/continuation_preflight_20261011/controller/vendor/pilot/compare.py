"""Post-fit full prediction and scientific-result comparison; never launches fitting."""
import argparse
import copy as copying
import io
import math
import sys
sys.dont_write_bytecode=True
from contract import *
from study import verify_run

def normalize(result):
    d=copying.deepcopy(result)
    def timing(obj,field):
        v=obj.pop(field);require(type(v) in (int,float) and math.isfinite(v) and v>=0,'Invalid elapsed time')
    for k in ('elapsed_seconds','training_seconds_including_selection','inference_seconds'):timing(d,k)
    timing(d['selection'],'seconds')
    for c in d['candidates']:timing(c,'seconds')
    receipt=d['prediction_receipt'];timing(receipt['provenance']['selection'],'seconds')
    # Separate NPZ arrays, receipt bindings and original/new path checks cover these.
    receipt.pop('prediction_file');receipt.pop('sha256')
    return d

def tree_differences(a,b,path='result'):
    if type(a)!=type(b):return [dict(path=path,reason='type',new=str(type(a)),historical=str(type(b)))]
    if isinstance(a,dict):
        if a.keys()!=b.keys():return [dict(path=path,reason='keys')]
        return sum((tree_differences(a[k],b[k],path+'.'+k) for k in a),[])
    if isinstance(a,list):
        if len(a)!=len(b):return [dict(path=path,reason='length')]
        return sum((tree_differences(x,y,path+f'[{i}]') for i,(x,y) in enumerate(zip(a,b))),[])
    if isinstance(a,float):require(math.isfinite(a) and math.isfinite(b),'Nonfinite scientific result')
    return [] if a==b else [dict(path=path,new=a,historical=b)]

def compare(bundle,run,source):
    import numpy as np
    structural=verify_run(bundle,run);plan=load_plan();rows=[]
    def historical(name):
        p=member(source,name);data=p.read_bytes()
        require(dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())==plan['reference_files'][name],'Historical reference changed')
        return data
    for ident,entry in plan['tasks'].items():
        new=Path(run)/ident/'runtime/results';prefix=entry['row']['folder']+'/'
        old_receipt=json.loads(historical(prefix+'predictions.json'));old_result=json.loads(historical(prefix+'RESULT.json'))
        old_data=historical(prefix+'predictions.npz')
        require(old_receipt['sha256']==hashlib.sha256(old_data).hexdigest() and old_result['prediction_receipt']==old_receipt,'Historical receipt mismatch')
        ledger=read(new/'ledger.json')
        require(old_receipt['provenance']==dict(ledger_sha256=digest(ledger),pool_sha256=ledger['pool_sha256'],selection=old_result['selection']),'Historical ledger/selection mismatch')
        require(old_receipt['prediction_file'].endswith('/'+prefix+'predictions.npz'),'Historical prediction route differs')
        with np.load(io.BytesIO(old_data),allow_pickle=False) as old,np.load(new/'predictions.npz',allow_pickle=False) as current:
            require(set(old.files)==set(current.files)=={'cell_ids','predictions'},'Prediction members differ')
            require(old['cell_ids'].dtype==current['cell_ids'].dtype and np.array_equal(old['cell_ids'],current['cell_ids']),'Target IDs differ')
            a=current['predictions'];b=old['predictions'];require(a.shape==b.shape and a.dtype==b.dtype,'Prediction dtype/shape differs')
            changed=int(np.count_nonzero(a!=b));maxdiff=float(np.max(np.abs(a-b))) if a.dtype.kind=='f' else None
        differences=tree_differences(normalize(read(new/'RESULT.json')),normalize(old_result))
        rows.append(dict(id=ident,job=entry['row']['job'],prediction_count=len(a),changed_predictions=changed,
            max_abs_prediction_difference=maxdiff,npz_bytes_equal=record(new/'predictions.npz')['sha256']==hashlib.sha256(old_data).hexdigest(),
            scientific_result_differences=differences,exact_scientific_match=changed==0 and not differences))
    return dict(status='exact' if all(r['exact_scientific_match'] for r in rows) else 'completed_with_differences',
        scope='pilot45 only; frozen representations and supervision; newly fitted heads',structural=structural,
        jobs=rows,exact_jobs=sum(r['exact_scientific_match'] for r in rows),
        total_predictions=sum(r['prediction_count'] for r in rows),changed_predictions=sum(r['changed_predictions'] for r in rows),
        ignored_fields=['RESULT elapsed_seconds/training_seconds_including_selection/inference_seconds',
                        'selection/candidate seconds (including receipt selection)','receipt prediction_file and NPZ sha256; arrays/bindings checked separately'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--source-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=compare(a.bundle,a.run,a.source_root)
    out=reserve(a.output,[a.bundle,a.run,a.source_root]);write(out/'report.json',result);seal(out,dict(scope='post_fit_comparison',status=result['status']))
    print(dict(status=result['status'],exact_jobs=result['exact_jobs'],predictions=result['total_predictions'],changed=result['changed_predictions']))
