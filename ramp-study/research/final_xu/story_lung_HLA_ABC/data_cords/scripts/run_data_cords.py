"""Run or verify the frozen-reference Cords data package."""
import package_runtime as rt
from package_runtime import BASE
import argparse,json


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
    parser.add_argument('command',nargs='?',default='run',choices=['run','verify'])
    args=parser.parse_args()
    if args.command=='verify' or (BASE/'results/RUN_MANIFEST.json').exists():verify();return
    if not (BASE/'reference/IMPORT_MANIFEST.json').exists():
        import prepare_reference
        prepare_reference.main()
    if not (BASE/'results/PREPARATION_VERIFICATION.json').exists():
        import project
        project.main()
    import verify_data
    verify_data.main();verify()


if __name__=='__main__':main()
