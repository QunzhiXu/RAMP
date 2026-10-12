"""Single entry point for fixed Zheng validation; writes only inside validation."""
import package_runtime as rt
from package_runtime import BASE
import argparse,json

OUT=BASE/'results'


def verify():
    path=OUT/'RUN_MANIFEST.json'
    if not path.exists():raise SystemExit('No completed validation manifest; run: bash run.sh run')
    manifest=json.loads(path.read_text())
    failures=[];checked=0
    for group in ['source_inputs','script_sha256','output_sha256','reference_sha256']:
        for name,expected in manifest[group].items():
            p=BASE/name
            checked+=1
            if not p.exists() or rt.sha(p)!=expected:failures.append(str(p))
    if failures:raise RuntimeError('Files changed since validation: '+json.dumps(failures))
    result={'status':'PASS','checked_files':checked,
            'reproduction':json.loads((OUT/'REPRODUCTION.json').read_text())}
    print(json.dumps(result,indent=2),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',nargs='?',choices=['run','verify','plot'],default='run')
    args=parser.parse_args()
    if args.command=='verify':verify();return
    if args.command=='plot':
        verify()
        import report_validation
        report_validation.main()
        verify();return
    if (OUT/'RUN_MANIFEST.json').exists():verify();return
    if not (OUT/'PREPARATION_VERIFICATION.json').exists():
        import predict
        predict.main()
    import report_validation
    report_validation.main()
    verify()


if __name__=='__main__':main()
