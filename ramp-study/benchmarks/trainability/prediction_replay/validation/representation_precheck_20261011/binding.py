"""Pin current primary jobs and all actually used prediction/representation artifacts."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import numpy as np
from common import *

ROOT=phase1.ROOT
LUNG='story_lung_HLA_ABC/releases/senosain140k_20260921'

def routes():
    cat=phase1_anchor(); rows=[]; reps={}; cohorts={}
    def frozen(name):
        p=SNAPSHOT/'tree'/name
        require(record(p)==cat['files'][name],'Phase-1 frozen input changed')
        return read(p)
    def rep(case,marker,method,folder=None,values=None,ids=None,proof=None):
        k=case+'/'+(marker or 'complete')+'/'+method
        if k not in reps:
            folder=folder or f'{ROOT}/representations/formal_v0_5/{k}'
            reps[k]=dict(case=case,marker=marker,method=method,folder=folder,
                         values=values or folder+'/values.npy',ids=ids or folder+'/cell_ids.npy',
                         proof=proof or folder+'/COMPLETE.json')
        return k
    def add(j,folder,repkey,manifest=None,plan=None,completion='JOB_COMPLETE.json',method=None):
        short={k:v for k,v in j.items() if k!='candidates'}
        short['method']=method or j['method']
        rows.append(dict(job=short,candidates_sha256=digest(j['candidates']),original_job_sha256=digest(j),
                         folder=folder,representation=repkey,manifest=manifest,plan=plan,completion=completion))
    suite=frozen(ROOT+'/config/formal_v0_5/SUITE.json')
    for shard in suite['shards']:
        config=frozen(phase1.bind(shard['config']))
        for j in config['jobs']:
            if 'primary' not in j['analysis_scopes']:continue
            if not ((j['task_type']=='J1' and j['task']=='bone_marrow') or (j['task_type']=='J2' and j['task']!='lung_story')):continue
            k=rep(j['task'],j.get('marker',''),j['method'])
            add(j,phase1.bind(config['output'])+'/jobs/'+j['id'],k)
    for case,dirname in phase1.UPDATES.items():
        base=ROOT+'/'+dirname; owner=base+'/MANIFEST.json';m=frozen(owner)
        jobs=m['jobs']['full'] if case=='hnscc_story' else m['jobs']
        if case=='lung_story':jobs=[j for j in jobs if j['arm']=='keep_c02']
        for method in METHODS:
            if case=='lung_story':
                pilot=ROOT+'/'+phase1.PILOT
                if method=='ramp':
                    k=rep(case,'',method,values=phase1.bind(frozen(pilot+'/MANIFEST.json')['representation']),
                          ids='story_lung_HLA_ABC/Comparison_senosain140k_20260921/input/cell_ids.npy',proof=LUNG+'/alignment/COMPLETE.json',folder=LUNG+'/alignment')
                elif method in m['reusable_representations']:
                    info=m['reusable_representations'][method]
                    k=rep(case,'',method,values=phase1.bind(info['values']),ids='story_lung_HLA_ABC/Comparison_senosain140k_20260921/input/cell_ids.npy',proof=phase1.bind(info['receipt']),folder=base+'/representations/'+method)
                else:k=rep(case,'',method,folder=base+'/representations/'+method)
                for j in jobs:
                    folder=(pilot+'/jobs/' if method=='ramp' else base+'/jobs/'+method+'/')+'keep_c02/'+j['id']
                    add(j,folder,k,pilot+'/MANIFEST.json' if method=='ramp' else owner,pilot+'/plans/'+j['id']+'.json','EXPERIMENT_COMPLETE.json',method)
            else:
                arms=[a for a in m['arms'] if a['method']==method and a.get('cohort','full')=='full']
                require(len(arms)==1,'Ambiguous J1 representation')
                values=phase1.bind(arms[0]['values']); k=rep(case,'',method,folder=str(Path(values).parent))
                prefix='full__'+method if case=='hnscc_story' else method
                for j in jobs:
                    plan=base+'/plans/'+('full/' if case=='hnscc_story' else '')+j['id']+'.json'
                    add(j,base+'/jobs/'+prefix+'/'+j['id'],k,owner,plan,'EXPERIMENT_COMPLETE.json',method)
    base=ROOT+'/'+phase1.J2;m=frozen(base+'/MANIFEST.json')
    for j in m['jobs']:
        k=rep('lung_story',j['marker'],j['method'],folder=base+'/representations/'+j['marker']+'/'+j['method'])
        add(j,base+'/jobs/'+j['method']+'/'+j['id'],k,base+'/MANIFEST.json',base+'/'+j['plan_file'])
    for case in phase1.CASES:
        if case=='lung_story':
            c=frozen(ROOT+'/'+phase1.J2+'/INPUT_CONTRACT.json')
            cohorts[case]=dict(native=LUNG+'/inputs/RAW_SHARED9.npy',metadata=LUNG+'/inputs/CELL_METADATA.csv.gz',
                markers=c['markers'],canonical_markers=c['canonical_markers'],pool_id='lung_story:senosain140k_release_20260921:all84000',pool_sha256=m['pool_sha256'])
        else:
            b=ROOT+'/data/real_inputs_v0_3_r2/'+case; c=frozen(b+'/manifest.json')
            cohorts[case]={k:c[k] for k in ('markers','canonical_markers','pool_id','pool_sha256')}
            cohorts[case].update(native=b+'/native.npy',metadata=b+'/metadata.csv.gz')
        cohorts[case]['classes']=frozen(ROOT+'/'+phase1.UPDATES[case]+'/MANIFEST.json')['classes'] if case!='bone_marrow' else c['classes']
    check_grid(rows);require(len(reps)==860,'Representation grid differs')
    return dict(jobs=sorted(rows,key=key),representations=reps,cohorts=cohorts)

def sources(plan, source):
    names=set(phase1_anchor()['files'])
    for row in plan['jobs']:
        names.update(row['folder']+'/'+n for n in ('predictions.npz','predictions.json','ledger.json','RESULT.json',row['completion']))
        for field in ('manifest','plan'):
            if row[field]:names.add(row[field])
    for r in plan['representations'].values():
        names.update(r[n] for n in ('values','ids','proof'))
        for n in ('START.json','COMPLETE.json','MASK_BOUNDARY_AUDIT.json'):
            if member(source,r['folder']+'/'+n).is_file():names.add(r['folder']+'/'+n)
    names.add(LUNG+'/RELEASE_MANIFEST.json')
    names.add('story_lung_HLA_ABC/Comparison_senosain140k_20260921/input/manifest.json')
    return sorted(names)

def collect(source):
    plan=routes(); names=sources(plan,source)
    records={}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,(n,r) in enumerate(zip(names,pool.map(lambda n:record(member(source,n)),names))):
            records[n]=r
            if (i+1)%10000==0:print('HASHED',i+1,'/',len(names),flush=True)
    for n,r in phase1_anchor()['files'].items():require(records[n]==r,'Original frozen source changed: '+n)
    return plan|dict(schema=1,phase1_catalog_sha256=ANCHOR,phase1_handoff_sha256=HANDOFF,files=records,
                     scope='Existing primary predictions only; no new fits; representations hashed and identity checked, not refitted')

def check_job_proof(row,records,source):
    prefix=row['folder']+'/'
    proof=bound_json(source,prefix+row['completion'],records)
    wanted={n:records[prefix+n]['sha256'] for n in ('RESULT.json','predictions.npz','predictions.json','ledger.json')}
    if row['completion']=='JOB_COMPLETE.json':
        require(proof['artifact_sha256']==wanted,'Job artifact proof differs')
        require(proof['job_sha256']==row['original_job_sha256'],'Job identity proof differs')
    else:
        actual={phase1.bind(n):v for n,v in proof['files'].items()}
        require(actual=={prefix+n:v for n,v in wanted.items()},'Experiment artifacts differ')
    if row['manifest']:
        require(proof['manifest_sha256']==records[row['manifest']]['sha256'],'Job manifest proof differs')

def verify_files(catalog,source):
    with ThreadPoolExecutor(max_workers=4) as pool:
        for n,r in zip(catalog['files'],pool.map(lambda n:record(member(source,n)),catalog['files'])):
            require(r==catalog['files'][n],'Source changed: '+n)
    return dict(files=len(catalog['files']),bytes=sum(r['bytes'] for r in catalog['files'].values()))

def load(index,anchor):
    require(record(index)['sha256']==anchor,'Prediction index anchor differs')
    complete=read(Path(index).parent/'COMPLETE.json')
    require(complete['status']=='complete' and complete['catalog']==record(index),'Binding incomplete')
    catalog=read(index);check_grid(catalog['jobs'])
    require(catalog['phase1_catalog_sha256']==ANCHOR and catalog['phase1_handoff_sha256']==HANDOFF,'Wrong phase-1 anchors')
    current=routes()
    for field in ('jobs','representations','cohorts'):require(catalog[field]==current[field],'Routing differs: '+field)
    return catalog

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=output(a.output,a.source_root,[HERE/'common.py',HERE/'binding.py']);before=code_records()
    try:
        write(out/'REQUEST.json',dict(code_records=before,source_root=str(a.source_root.resolve()),phase1_anchor=ANCHOR))
        catalog=collect(a.source_root)
        for n,row in enumerate(catalog['jobs']):
            check_job_proof(row,catalog['files'],a.source_root)
            if (n+1)%5000==0:print('PROOFS',n+1,flush=True)
        write(out/'catalog.json',catalog)
        verification=verify_files(catalog,a.source_root)
        require(before==code_records(),'Controller changed during binding')
        write(out/'COMPLETE.json',dict(status='complete',catalog=record(out/'catalog.json'),verification=verification,jobs=len(catalog['jobs'])))
        print(json.dumps(read(out/'COMPLETE.json')),flush=True)
    except BaseException as e:
        write(out/'FAILED.json',dict(error=repr(e)));raise

if __name__=='__main__':main()
