"""Create-only source/code locks and preservation census before full retraining."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;PARENT=HERE.parent
sys.path.insert(0,str(PARENT))
from safe_io import record,read,write,require,copy_checked,reserve
import baseline
CATALOG_SHA='82eafb7e4e41652b23844b53247081e3242163cc606004771734d2ba51ea7c4d'
PILOT_SHA='2a2d977541d244630eb5fa307f56eb93e94591f2ad71a9528b9b9c99d7ac7bb0'
PACKAGE='cross_platform_trainability_v2_20260919/src/trainability/'
MODULES=('__init__','core','inputs','real_data','sampling','scoring','targets','validation','heads','runner')

def capture():
    records={}
    for root,dirs,names in os.walk(PARENT):
        dirs[:]=sorted(d for d in dirs if d not in {'head_retraining_full','__pycache__'} and not d.startswith('.'))
        for name in sorted(names):
            if name=='.DS_Store' or name.endswith('.pyc'):continue
            p=Path(root)/name;records[p.relative_to(PARENT).as_posix()]=record(p)
    return dict(trainability_files=records,external=baseline.census(),prior=baseline.prior())

def setup(source):
    pilot=PARENT/'head_retraining';report=read(pilot/'validation/handoff_20261011/report.json')
    require(record(pilot/'validation/handoff_20261011/report.json')['sha256']==PILOT_SHA,'Pilot report changed')
    vendor=reserve(HERE/'vendor/pilot',[source,pilot])
    for n in ('contract.py','worker.py','study.py','compare.py'):
        copy_checked(pilot/n,vendor/n,report['current_code'][n])
    evidence=reserve(HERE/'validation/preflight_20261011',[source,pilot]);write(evidence/'report.json',capture())
    index=PARENT/'prediction_replay/bindings/current_predictions_20261011/catalog.json'
    require(record(index)['sha256']==CATALOG_SHA,'Prediction catalog changed');c=read(index)
    # Catalog was independently audited against the complete published task grid.
    require(len(c['jobs'])==32370 and len(c['representations'])==860,'Full grid differs')
    groups=defaultdict(dict);keys=[]
    for i,row in enumerate(c['jobs']):
        j=row['job'];keys.append(tuple(j.get(k,'') for k in ('task_type','task','method','marker','source','target','family','seed')))
        groups[row['representation']][f'job_{i:05d}']=row
    require(len(set(keys))==32370,'Duplicate task identity')
    lockdir=reserve(HERE/'locks/full_20261011',[source,pilot]);(lockdir/'groups').mkdir()
    scientific={PACKAGE+n+'.py':c['files'][PACKAGE+n+'.py'] for n in MODULES}
    source_files={};root=dict(scope='full32370',catalog_sha256=CATALOG_SHA,pilot_report_sha256=PILOT_SHA,
        vendor={n:record(vendor/n) for n in ('contract.py','worker.py','study.py','compare.py')},scientific=scientific,
        cohorts=c['cohorts'],groups={},task_keys_sha256=hashlib.sha256(json.dumps(keys,separators=(',',':')).encode()).hexdigest())
    counts=Counter()
    for gi,(repkey,tasks) in enumerate(sorted(groups.items())):
        ident=f'group_{gi:04d}';rep=c['representations'][repkey];spec=c['cohorts'][rep['case']]
        names=set(scientific)|{spec['native'],spec['metadata'],rep['values'],rep['ids'],rep['proof']}
        refs=set()
        for row in tasks.values():
            names.add(row['folder']+'/ledger.json')
            if row['plan']:names.add(row['plan'])
            refs.update(row['folder']+'/'+n for n in ('predictions.npz','predictions.json','RESULT.json',row['completion']))
            counts[row['job']['task_type']+'__'+row['job']['task']]+=1
        payload=dict(id=ident,representation_key=repkey,representation=rep,cohort=spec,tasks=tasks,
            inputs={n:c['files'][n] for n in sorted(names)},references={n:c['files'][n] for n in sorted(refs)})
        write(lockdir/'groups'/(ident+'.json'),payload)
        root['groups'][ident]=dict(lock=record(lockdir/'groups'/(ident+'.json')),jobs=list(tasks),representation=repkey)
        source_files.update(payload['inputs'])
    root.update(tasks=32370,representations=860,branches=dict(counts),source_files=source_files)
    root['expected_result_bytes']=sum(c['files'][r['folder']+'/'+n]['bytes'] for r in c['jobs'] for n in ('ledger.json','predictions.npz','predictions.json','RESULT.json'))
    root['storage_policy']='Hash-bound read-only source references; no 25GB representation copy; preserve all previous outputs'
    write(lockdir/'LOCK.json',root);write(lockdir/'COMPLETE.json',dict(status='complete',lock=record(lockdir/'LOCK.json')))
    print(dict(lock=record(lockdir/'LOCK.json'),tasks=root['tasks'],groups=len(groups),training_files=len(source_files),expected_result_bytes=root['expected_result_bytes']),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['setup','check']);p.add_argument('--source-root',type=Path,required=True);a=p.parse_args()
    if a.mode=='setup':setup(a.source_root.resolve())
    else:require(capture()==read(HERE/'validation/preflight_20261011/report.json'),'Accepted files changed');print('Preservation baseline verified')
