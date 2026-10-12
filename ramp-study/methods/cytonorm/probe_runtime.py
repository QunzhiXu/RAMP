"""Exercise the frozen adapter in temporary directories on synthetic data only."""
import argparse
import csv
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import cytonorm_io as io
import inventory as inv


def environment(parent=None):
    env=dict(os.environ if parent is None else parent)
    for key in ('R_LIBS','R_LIBS_USER','R_LIBS_SITE'):env.pop(key,None)
    env.update(LC_ALL='C.UTF-8',LANG='C.UTF-8')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):env[key]='1'
    return env


def runtime_contract(library=None):
    catalog=inv.verify_local()
    rows=[r for r in catalog['files'] if r.get('copy')=='environment/runtime_lock.csv']
    if len(rows)!=1:raise ValueError('Missing frozen runtime lock')
    row=rows[0];path=inv.SNAPSHOT/row['copy'];io.check(path,row)
    with path.open(newline='') as stream:entries=list(csv.DictReader(stream))
    versions={r['package']:r['version'] for r in entries}
    if not entries or len(versions)!=len(entries):raise ValueError('Invalid frozen runtime version map')
    if library is not None:io.check(Path(library)/'runtime_lock.csv',row)
    return {'lock':{k:row[k] for k in ('bytes','sha256')},'versions':versions}


def check_summary(summary):
    contract=runtime_contract()
    if summary.get('runtime_versions')!=contract['versions']:raise ValueError('Runtime version map differs from frozen lock')
    if summary.get('runtime_lock_sha256')!=contract['lock']['sha256']:raise ValueError('Runtime lock hash differs')
    if summary.get('status')!='passed' or len(summary.get('results',[]))!=2 or any(
        r['maximum_absolute_difference']!=0 or r['preserialization_model_replay_maximum_difference']!=0
        for r in summary['results']):raise ValueError('Synthetic parity failed')
    return contract


def probe(library,rscript,scratch,output):
    library=Path(library).resolve(strict=True);rscript=Path(rscript).absolute();scratch=Path(scratch).resolve(strict=True)
    if not library.is_dir() or not scratch.is_dir() or any(c.isspace() for c in str(scratch)):raise ValueError('Use a library and space-free R scratch directory')
    catalog=inv.verify_local();contract=runtime_contract(library);exe=io.record(rscript);codes={n:io.record(inv.HERE/n) for n in ('probe_runtime.py','check_runtime.R')}
    # The helper also authenticates the vendor archive and lock next to .Rlib.
    root=io.reserve(output,(library.parent,rscript,inv.SNAPSHOT,*(inv.HERE/n for n in codes)))
    commands=[]
    try:
        io.create_json(root/'REQUEST.json',{'library':str(library),'rscript':str(rscript),'executable':exe,
            'codes':codes,'catalog_sha256':inv.CATALOG_SHA,'runtime_contract':contract,
            'locale':{'LC_ALL':'C.UTF-8','LANG':'C.UTF-8'},'real_cohort_fit':False})
        with tempfile.TemporaryDirectory(prefix='cytonorm_runtime_',dir=scratch) as name:
            temp=Path(name).resolve();adapter=temp/'adapter';shutil.copytree(inv.SNAPSHOT/'adapter',adapter)
            (adapter/'.Rlib').symlink_to(library,target_is_directory=True)
            env={**environment(),'TMPDIR':str(temp)}
            def run(command,key):
                with (root/(key+'.stdout.log')).open('x') as out,(root/(key+'.stderr.log')).open('x') as err:
                    result=subprocess.run(list(map(str,command)),cwd=temp,env=env,stdout=out,stderr=err)
                commands.append({'name':key,'command':list(map(str,command)),'returncode':result.returncode})
                if result.returncode:raise RuntimeError(key+' failed; see logs')
                print('PASS '+key,flush=True)
            story=inv.SNAPSHOT/'study_sources'/inv.STORIES['hnscc_story']/'scripts/cytonorm_repro.R'
            run([rscript,'--vanilla',inv.HERE/'check_runtime.R',adapter,story,root,inv.SNAPSHOT/'environment/runtime_lock.csv'], 'two_three_platform')
            for row in catalog['files']:
                if row.get('copy','').startswith('adapter/'):io.check(temp/row['copy'],row)
        io.check(rscript,exe)
        if runtime_contract(library)!=contract:raise ValueError('Runtime lock changed during probe')
        for n,row in codes.items():io.check(inv.HERE/n,row)
        if inv.verify_local()!=catalog:raise ValueError('Catalog changed during probe')
        summary=io.read_json(root/'synthetic.json')
        check_summary(summary)
        io.create_json(root/'report.json',{'status':'passed','real_cohort_fit':False,'commands':commands,
            'summary':summary,'request':io.record(root/'REQUEST.json'),'codes':codes,
            'files':{p.name:io.record(p) for p in sorted(root.iterdir()) if p.is_file()}})
    except BaseException as error:
        io.create_json(root/'FAILED.json',{'error':repr(error),'commands':commands});raise
    return root


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('library','rscript','scratch-parent','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();print(probe(a.library,a.rscript,a.scratch_parent,a.output))


if __name__=='__main__':main()
