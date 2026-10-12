"""Read-only validation of completed replay; writes only a new evidence directory."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from common import *
import binding
import preflight
import replay

def invoke(argv,logs,name,env=None):
    with (logs/(name+'.stdout')).open('xb') as out,(logs/(name+'.stderr')).open('xb') as err:
        r=subprocess.run(argv,stdout=out,stderr=err,env=env)
    return dict(command=[str(x) for x in argv],returncode=r.returncode,stdout=record(logs/(name+'.stdout')),stderr=record(logs/(name+'.stderr')))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--index',type=Path,required=True);p.add_argument('--anchor',required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--source-root',type=Path,required=True);p.add_argument('--comparison',type=Path,required=True);p.add_argument('--per-job',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();dest=output(a.output,a.source_root,[a.index,a.run,a.comparison]);commands=[]
    try:
        catalog=binding.load(a.index,a.anchor)
        previous=preflight.check(HERE/'validation/preflight_20261011/report.json')
        verified=replay.verify_run(a.run,catalog,a.anchor)
        source_check=binding.verify_files(catalog,a.source_root)
        with tempfile.TemporaryDirectory(prefix='trainability_prediction_audit_') as tmp:
            scratch=Path(tmp).resolve();parent=scratch/'ramp-study/benchmarks/trainability';local=parent/'prediction_replay';local.mkdir(parents=True)
            # Faithful new controller and accepted read-only prerequisites, no source tree.
            for q in HERE.glob('*.py'):shutil.copy2(q,local/q.name)
            shutil.copytree(HERE/'tests',local/'tests')
            for q in PARENT.glob('*.py'):shutil.copy2(q,parent/q.name)
            shutil.copytree(SNAPSHOT,parent/SNAPSHOT.name)
            handoff=parent/'validation/handoff_20261011_r2';handoff.mkdir(parents=True)
            shutil.copy2(PARENT/'validation/handoff_20261011_r2/report.json',handoff/'report.json')
            r=invoke([sys.executable,'-B','-m','unittest','discover','-s',str(local/'tests'),'-v'],dest,'tests')
            commands.append(r);require(r['returncode']==0,'Tests failed')
            moved=scratch/'moved index with spaces';moved.mkdir()
            for n in ('catalog.json','COMPLETE.json'):shutil.copy2(a.index.parent/n,moved/n)
            base=[sys.executable,'-B',str(local/'replay.py'),'verify-run','--index',str(moved/'catalog.json'),'--anchor',a.anchor]
            r=invoke(base+['--run',str(a.run.resolve())],dest,'moved_index_verify');commands.append(r);require(r['returncode']==0,'Moved verification failed')
            negative={}
            for name in ('duplicate','missing','foreign','unknown_controller','partial'):
                bad=scratch/name;bad.mkdir();(bad/'branches').mkdir()
                manifest=read(a.run/'BATCH_MANIFEST.json');request=read(a.run/'REQUEST.json')
                if name=='duplicate':manifest['branches']=[manifest['branches'][0]]*len(manifest['branches'])
                if name=='missing':manifest['branches']=manifest['branches'][:-1]
                if name=='foreign':manifest['branches'][-1]='J2__foreign'
                if name=='unknown_controller':request['code_records']['scores.py']['sha256']='0'*64
                write(bad/'REQUEST.json',request);write(bad/'BATCH_MANIFEST.json',manifest)
                write(bad/'COMPLETE.json',dict(status='partial' if name=='partial' else 'complete',manifest=record(bad/'BATCH_MANIFEST.json')))
                before={q.name:record(q) for q in bad.iterdir() if q.is_file()}
                r=invoke(base+['--run',str(bad)],dest,'negative_'+name);commands.append(r)
                require(r['returncode']!=0,'Malformed batch accepted: '+name)
                require(before=={q.name:record(q) for q in bad.iterdir() if q.is_file()},'Read-only CLI wrote malformed batch')
                message=(dest/('negative_'+name+'.stderr')).read_text()
                wanted='Batch branch grid differs' if name in ('duplicate','missing','foreign') else 'Batch identity/controller' if name=='unknown_controller' else 'Batch manifest differs'
                require(wanted in message,'Negative failed for wrong reason')
                negative[name]=wanted
        require(preflight.check(HERE/'validation/preflight_20261011/report.json')==previous,'Prior baseline changed')
        comparison=read(a.comparison/'report.json');require(comparison['status']=='passed','Comparison failed')
        import compare
        per_job,table=compare.compare(catalog,a.run,a.anchor)
        require(per_job==read(a.per_job/'report.json'),'Per-job comparison changed')
        report=dict(status='passed',scope='Saved predictions and scores only; no new model fits',commands=commands,run=verified,
                    index_sha256=a.anchor,source_check=source_check,comparison=comparison,per_job=per_job,negative_cli=negative,
                    baseline=previous,code_records=code_records(),current_files={p.name:record(p) for p in HERE.glob('*.py')},
                    tests={p.name:record(p) for p in (HERE/'tests').glob('*.py')},external_audit_approval=False)
        write(dest/'report.json',report);write(dest/'COMPLETE.json',dict(status='complete',files={p.name:record(p) for p in dest.iterdir()}))
        print(dict(status='passed',run=verified,index=a.anchor),flush=True)
    except BaseException as e:write(dest/'FAILED.json',dict(error=repr(e)));raise

if __name__=='__main__':main()
