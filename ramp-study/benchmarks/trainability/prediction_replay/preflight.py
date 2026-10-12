"""Preserve accepted phase-1 and external indexed baselines; no research execution."""
import argparse
import os
from common import *
import baseline

def phase1_files():
    records={}
    for root,dirs,names in os.walk(PARENT):
        dirs[:]=sorted(d for d in dirs if d not in {'prediction_replay','__pycache__'} and not d.startswith('.'))
        for name in sorted(names):
            if name=='.DS_Store' or name.endswith('.pyc'):continue
            p=Path(root)/name;records[p.relative_to(PARENT).as_posix()]=record(p)
    return records

def capture():
    phase1_anchor()
    return dict(phase1_files=phase1_files(),external=baseline.census(),prior=baseline.prior(),phase1_catalog=ANCHOR,phase1_handoff=HANDOFF)

def check(path):
    before=read(path);after=capture();require(before==after,'Previously accepted baseline changed')
    return dict(phase1_files=len(after['phase1_files']),external_non_run_files=len(after['external']),prior=after['prior'])

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['capture','check']);p.add_argument('--path',type=Path,required=True);a=p.parse_args()
    if a.mode=='check':print(check(a.path))
    else:
        out=output(a.path,PARENT.parents[2]/'final_xu',[HERE/'preflight.py'])
        try:
            data=capture();write(out/'report.json',data);write(out/'COMPLETE.json',dict(status='complete',report=record(out/'report.json')))
            print(dict(phase1_files=len(data['phase1_files']),external=len(data['external']),sha256=record(out/'report.json')['sha256']))
        except BaseException as e:write(out/'FAILED.json',dict(error=repr(e)));raise

if __name__=='__main__':main()
