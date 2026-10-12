"""Run fixed eligibility, Cox/KM fitting and reporting inside survival only."""
import package_runtime as rt
from package_runtime import BASE
import argparse,json,os,subprocess,shutil


def verify():
    manifest=json.loads((BASE/'results/RUN_MANIFEST.json').read_text())
    failures=[];count=0
    for group in ['inputs','scripts','reference','outputs']:
        for name,h in manifest[group].items():
            p=BASE/name;count+=1
            if not p.exists() or rt.sha(p)!=h:failures.append(str(p))
    assert not failures,failures
    print(json.dumps({'status':'PASS','checked_files':count}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',nargs='?',default='run',choices=['run','verify','plot'])
    args=parser.parse_args()
    if args.command=='verify':verify();return
    if args.command=='plot':
        verify()
        import report_survival
        report_survival.main();verify();return
    if (BASE/'results/RUN_MANIFEST.json').exists():verify();return
    if not (BASE/'results/PREPARATION_VERIFICATION.json').exists():
        import prepare_survival
        prepare_survival.main()
    if not (BASE/'results/COX_MODELS.csv').exists():
        executable=shutil.which('Rscript')
        if executable is None:raise RuntimeError('Rscript with survival and data.table is required')
        env=os.environ.copy();env['R_PROFILE_USER']='/dev/null';env['R_ENVIRON_USER']='/dev/null'
        # R rejects spaces in its temporary-directory string. The child already
        # starts in BASE, so this relative path stays inside survival/tmp.
        for key in ['TMPDIR','TMP','TEMP']:env[key]='tmp'
        subprocess.run([executable,'--vanilla',str(BASE/'scripts/fit_survival.R'),str(BASE)],cwd=BASE,env=env,check=True)
    import report_survival
    report_survival.main();verify()


if __name__=='__main__':main()
