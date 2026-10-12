import sys
sys.dont_write_bytecode=True
import argparse
from trainability.experiment import execute

if __name__=='__main__':
    p=argparse.ArgumentParser(description='Run explicit v2 jobs. Formal mode checks release evidence.')
    p.add_argument('--config',required=True)
    p.add_argument('--pilot',action='store_true')
    a=p.parse_args();execute(a.config,pilot=a.pilot)
