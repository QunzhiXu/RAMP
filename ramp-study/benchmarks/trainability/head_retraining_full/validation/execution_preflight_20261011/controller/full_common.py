"""Full-grid orchestration contracts; frozen pilot/scientific code stays unchanged."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;PARENT=HERE.parent;STUDY=PARENT.parents[1]
sys.path.insert(0,str(HERE/'vendor/pilot'))
import contract as pilot
from contract import require,record,read,write,member,files,copy,seal,verify_seal,digest,environment,child_environment,THREADS,VERSIONS
LOCK_SHA='085b7c3e5bfc974e8b99e774b604e094d74793f51e9b41d244783fedbe50a685'
CODE=('full_common.py','full_worker.py','full_run.py','vendor/pilot/contract.py','vendor/pilot/worker.py','vendor/pilot/study.py','vendor/pilot/compare.py')
RESERVE_BYTES=20*1024**3

def codes():return {n:record(HERE/n) for n in CODE}

def load_lock(directory):
    directory=Path(directory).resolve();require(record(directory/'LOCK.json')['sha256']==LOCK_SHA,'Full lock changed')
    require(read(directory/'COMPLETE.json')==dict(status='complete',lock=record(directory/'LOCK.json')),'Lock incomplete')
    root=read(directory/'LOCK.json');require(root['scope']=='full32370' and root['tasks']==32370 and len(root['groups'])==860,'Full grid changed')
    return root

def group_lock(directory,root,ident):
    require(ident in root['groups'],'Unknown group')
    p=member(directory,'groups/'+ident+'.json');require(record(p)==root['groups'][ident]['lock'],'Group lock changed')
    g=read(p);require(g['id']==ident and list(g['tasks'])==root['groups'][ident]['jobs'],'Group identity changed')
    return g

def reserve(path,protected=()):
    raw=Path(os.path.abspath(path));require(not raw.exists() and not raw.is_symlink(),'Output already exists')
    dest=raw.resolve()
    if dest.is_relative_to(STUDY):
        require(any(dest.is_relative_to(HERE/n) and dest!=HERE/n for n in ('runs','validation')),'Output outside full-phase evidence roots')
    for base in (STUDY.parent/'final_xu',STUDY.parent/'ramp-proteomics',PARENT/'head_retraining',PARENT/'prediction_replay',PARENT/'frozen_20261011_r2',HERE/'locks',HERE/'vendor',*protected):
        base=Path(base).resolve();require(not(dest.is_relative_to(base) or base.is_relative_to(dest)),'Output overlaps protected input')
    dest.parent.mkdir(parents=True,exist_ok=True);dest.mkdir(exist_ok=False);return dest

def check_grid(actual,expected):
    require(isinstance(actual,list) and len(actual)==len(set(actual)) and set(actual)==set(expected),'Full task grid differs')

def disk_check(path,needed=0):
    free=shutil.disk_usage(path).free
    require(free>=RESERVE_BYTES+needed,'Disk reserve insufficient')
    return free

def bound_json(source,name,records):
    p=member(source,name);data=p.read_bytes()
    require(dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())==records[name],'Input changed: '+name)
    return json.loads(data)

def expected_request(ident,lockdir,source,initial):
    return dict(scope='full32370',id=ident,lock_sha256=LOCK_SHA,lockdir=str(Path(lockdir).resolve()),source_root=str(Path(source).resolve()),code_records=initial)

def verify_job(result,row,ledger):
    import numpy as np
    require({p.name for p in result.iterdir()}=={'ledger.json','predictions.npz','predictions.json','RESULT.json'},'Unexpected scientific files')
    require(read(result/'ledger.json')==ledger,'Rebuilt ledger differs')
    receipt=read(result/'predictions.json');r=read(result/'RESULT.json')
    require(receipt['sha256']==record(result/'predictions.npz')['sha256'] and receipt['target_answers_included'] is False,'Prediction receipt mismatch')
    require(r['prediction_receipt']==receipt and receipt['provenance']==dict(ledger_sha256=digest(ledger),pool_sha256=ledger['pool_sha256'],selection=r['selection']),'Result provenance mismatch')
    for role,key in (('fit','fit_cells'),('validation','validation_cells'),('calibration','calibration_cells')):require(r[key]==len(ledger[role]),'Supervision size mismatch')
    require(r['total_supervision']==ledger['total_supervision'],'Supervision total mismatch')
    with np.load(result/'predictions.npz',allow_pickle=False) as z:
        require(set(z.files)=={'cell_ids','predictions'} and z['cell_ids'].tolist()==ledger['target_cell_ids'],'Prediction identity mismatch')
        pred=z['predictions'];require(pred.ndim==1 and len(pred)==receipt['n_cells']==len(ledger['target_cell_ids']),'Prediction shape mismatch')
        require(pred.dtype.kind in ('f' if ledger['marker'] else 'US'),'Prediction dtype mismatch')
        if ledger['marker']:require(np.isfinite(pred).all(),'Nonfinite predictions')
    return dict(predictions=receipt['n_cells'],prediction_sha256=receipt['sha256'],ledger_sha256=digest(ledger),result=record(result/'RESULT.json'))
