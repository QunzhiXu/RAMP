import sys
sys.dont_write_bytecode=True
import argparse
from trainability.reporting import summarize_suite
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--suite',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(summarize_suite(a.suite,destination=a.output))
