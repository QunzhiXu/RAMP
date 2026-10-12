"""Read-only preservation census, excluding this new phase only."""
import argparse
import os
from pathlib import Path
import sys
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path.insert(0, str(PARENT))
import baseline
from safe_io import record, read, write, reserve, require

def capture():
    records = {}
    for root, dirs, names in os.walk(PARENT):
        dirs[:] = sorted(d for d in dirs if d not in {'head_retraining', '__pycache__'} and not d.startswith('.'))
        for name in sorted(names):
            if name == '.DS_Store' or name.endswith('.pyc'): continue
            p = Path(root) / name
            require(not p.is_symlink(), 'Unexpected old trainability symlink')
            records[p.relative_to(PARENT).as_posix()] = record(p)
    return dict(trainability_files=records, external=baseline.census(), prior=baseline.prior())

def main():
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['capture', 'check']); p.add_argument('path', type=Path)
    a = p.parse_args(); current = capture()
    if a.mode == 'check': require(current == read(a.path / 'report.json'), 'Accepted baseline changed')
    else:
        out = reserve(a.path, [PARENT/'frozen_20261011_r2', PARENT/'prediction_replay', baseline.STUDY.parent/'final_xu', baseline.STUDY.parent/'ramp-proteomics'])
        write(out/'report.json', current)
    print(dict(trainability_files=len(current['trainability_files']), external=len(current['external']), prior=current['prior']))

if __name__ == '__main__': main()
