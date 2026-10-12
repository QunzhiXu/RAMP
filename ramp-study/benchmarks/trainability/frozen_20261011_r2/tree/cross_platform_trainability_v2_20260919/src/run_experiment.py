import sys
sys.dont_write_bytecode=True
import argparse
from trainability.experiment import execute

if __name__=='__main__':
    p=argparse.ArgumentParser(description='Run explicit v2 jobs. Formal mode checks release evidence.')
    p.add_argument('--config',required=True)
    p.add_argument('--pilot',action='store_true')
    a=p.parse_args();results=execute(a.config,pilot=a.pilot)
    import json
    expected=len(json.load(open(a.config))['jobs'])
    raise SystemExit(0 if len(results)==expected and all(r['status'] in {'passed','cached'} for r in results) else 2)
