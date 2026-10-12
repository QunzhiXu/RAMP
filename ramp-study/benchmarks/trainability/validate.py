"""Internal phase-1 handoff checks, not external approval. No training or prediction rescoring."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import baseline
import cohorts
import contract
import inventory as i
from safe_io import require, read, write, record, reserve, files


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();source=a.source_root.resolve(strict=True)
    require(a.output.resolve().is_relative_to(i.HERE/'validation'),'Evidence must stay under this module')
    out=reserve(a.output,[source]);lock=read(i.HERE/'LOCK.json');snapshot=i.HERE/lock['snapshot']
    preflight=i.HERE/'validation/preflight_20261011/report.json'
    try:
        require(record(preflight)['sha256']==lock['preflight_sha256'],'Preflight changed')
        report={'kind':'internal self-validation; external audit pending','before':baseline.verify(preflight),
            'environment':{'python':sys.version,'packages':{p:importlib.metadata.version(p) for p in ('numpy','pandas','scipy','scikit-learn')}},
            'snapshot':i.verify(snapshot,lock['catalog_sha256'],source)}
        result,bench,overall=contract.inspect(snapshot,lock['catalog_sha256'])
        write(out/'aggregation.json',result);write(out/'cohorts.json',cohorts.inspect(source))
        report['aggregation']=record(out/'aggregation.json');report['cohorts']=record(out/'cohorts.json')
        report['comparisons']=result['comparisons']
        for name,df in [('PER_BENCHMARK.csv',bench),('OVERALL.csv',overall)]:
            with (out/name).open('x') as f:df.to_csv(f,index=False)
        with tempfile.TemporaryDirectory(prefix='figure3-inventory-') as temp:
            temp=Path(temp).resolve();module=temp/'module';module.mkdir()
            for path in i.HERE.iterdir():
                if path.is_file():shutil.copy2(path,module/path.name)
            shutil.copytree(i.HERE/'tests',module/'tests',ignore=shutil.ignore_patterns('__pycache__'))
            shutil.copytree(snapshot,module/lock['snapshot'])
            cmd=[sys.executable,'-B','-m','unittest','discover','-s',str(module/'tests'),'-v']
            proc=subprocess.run(cmd,cwd=module,capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
            with (out/'tests.log').open('x') as f:f.write(proc.stdout+proc.stderr)
            require(proc.returncode==0,'Tests in isolated module copy failed')
            report['tests']={'command':cmd,'returncode':proc.returncode,'log':record(out/'tests.log')}
            moved=temp/'moved snapshot with spaces';shutil.move(module/lock['snapshot'],moved)
            other,_,_=contract.inspect(moved,lock['catalog_sha256'])
            require(other==result,'Moved snapshot reaggregation differs')
            report['moved_without_original_tree']={'passed':True,'numeric_report_identical':True}
            rebuilt=temp/'rebuilt';i.freeze(source,rebuilt)
            expected={n:record(snapshot/n) for n in files(snapshot)}
            actual={n:record(rebuilt/n) for n in files(rebuilt)}
            require(expected==actual,'Temporary freeze differs')
            report['byte_identical_rebuilt_snapshot_files']=len(actual)
            # Actual CLI negative: corrupt a payload, retain the true trust anchor.
            (moved/'tree'/i.ROOT/'CURRENT_FIGURE3_RESULTS.json').write_text('{}')
            cmd=[sys.executable,'-B',str(module/'inventory.py'),'verify','--snapshot',str(moved),'--anchor',lock['catalog_sha256']]
            proc=subprocess.run(cmd,cwd=module,capture_output=True,text=True)
            require(proc.returncode!=0 and 'Snapshot file changed' in proc.stderr,'CLI did not reject changed payload')
            report['negative_cli']={'returncode':proc.returncode,'rejected_modified_pointer':True}
        report['after']=baseline.verify(preflight);require(report['before']==report['after'],'Earlier modules changed')
        report['snapshot_after']=i.verify(snapshot,lock['catalog_sha256'],source)
        require(report['snapshot_after']==report['snapshot'],'Original source changed during validation')
        report['current_code']={p.relative_to(i.HERE).as_posix():record(p) for p in sorted(i.HERE.rglob('*'))
            if p.is_file() and p.suffix in {'.py','.md','.json'} and not any(part in p.parts for part in ('validation','__pycache__','frozen_20261011','frozen_20261011_r2'))}
        report['status']='passed';write(out/'report.json',report)
        print(json.dumps({'status':'passed','evidence':str(out),'comparisons':report['comparisons']}))
    except BaseException as error:
        write(out/'FAILED.json',{'error':repr(error)});raise


if __name__=='__main__':main()
