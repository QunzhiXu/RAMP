"""Archive the exact guarded controller before starting the pilot batch."""
import argparse
import sys
sys.dont_write_bytecode=True
from contract import *
from study import fit

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--evidence',type=Path,required=True);p.add_argument('--python',required=True);a=p.parse_args()
    plan=verify_bundle(a.bundle);out=reserve(a.evidence,[a.bundle,a.run]);initial=codes()
    for n,r in initial.items():copy(HERE/n,out/'controller'/n,r)
    write(out/'report.json',dict(code_records=initial,plan=record(HERE/'locks/pilot_20261011.json'),
        baseline=record(HERE/'validation/preflight_20261011/report.json'),bundle_manifest=record(a.bundle/'MANIFEST.json'),
        scope='pilot45',scientific_files={n:v['record'] for n,v in plan['training_files'].items() if n.startswith('code/')}))
    seal(out,dict(scope='pre_fit_controller'))
    print(fit(a.bundle,a.run,a.python,3))
