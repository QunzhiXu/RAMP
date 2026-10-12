"""One-time outcome-blind coverage selection; no predictions/results are opened."""
from contract import *

PACKAGE = 'cross_platform_trainability_v2_20260919/src/trainability/'
MODULES = ('__init__','core','inputs','real_data','sampling','scoring','targets','validation','heads','runner')

def select(catalog):
    jobs = catalog['jobs']; methods = sorted({r['job']['method'] for r in jobs}); selected = []
    for bi, branch in enumerate(sorted({(r['job']['task_type'],r['job']['task']) for r in jobs})):
        for fi, family in enumerate(('knn','linear','mlp')):
            n=3*bi+fi
            eligible=[r for r in jobs if (r['job']['task_type'],r['job']['task'])==branch and r['job']['family']==family and r['job']['method']==methods[n%10] and r['job']['seed']==20260919+n%3]
            selected.append(min(eligible,key=lambda r:tuple(r['job'].get(k,'') for k in ('marker','source','target','id'))))
    # Predeclared within-patient fallback coverage, not chosen by observed scores.
    for kind in ('J1','J2'):
        for fi,family in enumerate(('knn','linear','mlp')):
            eligible=[r for r in jobs if r['job']['task']=='tnbc' and r['job']['task_type']==kind and r['job']['source']=='Tietscher IMC' and r['job']['family']==family and r['job']['method']=='ramp' and r['job']['seed']==20260919+fi]
            selected.append(min(eligible,key=lambda r:tuple(r['job'].get(k,'') for k in ('marker','target','id'))))
    return selected

def build(index, source):
    require(record(index)['sha256']==INDEX_SHA, 'Accepted prediction catalog changed')
    accepted=HERE.parent/'prediction_replay/validation/probability_fix_20261011/report.json'
    require(record(accepted)['sha256']==ACCEPTED_SHA, 'Accepted verifier evidence changed')
    c=read(index); selected=select(c); training={}; refs={}; tasks={}
    def add(name, src):
        rec=c['files'][src]; require(record(member(source,src))==rec,'Frozen source changed')
        training[name]=dict(source=src,record=rec); return name
    for name in MODULES: add('code/src/trainability/'+name+'.py',PACKAGE+name+'.py')
    cohorts={}
    for case in sorted({r['job']['task'] for r in selected}):
        spec=c['cohorts'][case].copy()
        for f in ('native','metadata'): spec[f]=add('cohorts/'+case+'/'+Path(spec[f]).name,spec[f])
        cohorts[case]=spec
    for i,row in enumerate(selected):
        ident=f'head_{i:03d}'; rep=c['representations'][row['representation']]; rfiles={}
        for f in ('values','ids','proof'): rfiles[f]=add('representations/'+row['representation']+'/'+Path(rep[f]).name,rep[f])
        ledger=add('jobs/'+ident+'/ledger.json',row['folder']+'/ledger.json')
        l=read(member(source,row['folder']+'/ledger.json'))
        require(digest(l['candidates'])==row['candidates_sha256'], 'Candidate grid differs')
        require(digest({k:l[k] for k in ('fit','validation','calibration')})==row['job']['supervision_sha256'], 'Supervision differs')
        planfile=add('jobs/'+ident+'/plan.json',row['plan']) if row['plan'] else None
        tasks[ident]=dict(row=row,representation=rfiles,ledger=ledger,plan=planfile)
        for name in ('predictions.npz','predictions.json','RESULT.json',row['completion']):
            src=row['folder']+'/'+name; refs[src]=c['files'][src]
    require(len(tasks)==45, 'Pilot size differs')
    return dict(schema=1,scope='45 preselected frozen-representation head retrainings, not all 32370 jobs',
                selection='Sorted 13 (kind,case) branches x knn/linear/mlp; round-robin sorted ten methods; seeds 20260919+family index; lexicographically first marker/source/target/id. Plus J1/J2 TNBC Tietscher-source RAMP x three families.',
                source_catalog_sha256=INDEX_SHA,accepted_verifier_report_sha256=ACCEPTED_SHA,
                tasks=tasks,cohorts=cohorts,training_files=training,reference_files=refs)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--index',type=Path,required=True);p.add_argument('--source-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    plan=build(a.index,a.source_root)
    # This is a write-once lock, intentionally created before the fit controller is frozen.
    require(a.output.resolve()==HERE/'locks/pilot_20261011.json','Unexpected lock destination')
    a.output.parent.mkdir(parents=True,exist_ok=True);write(a.output,plan)
    print(dict(tasks=len(plan['tasks']),files=len(plan['training_files']),bytes=sum(v['record']['bytes'] for v in plan['training_files'].values()),lock=record(a.output)))
