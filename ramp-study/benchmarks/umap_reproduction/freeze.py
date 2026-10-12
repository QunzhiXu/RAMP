"""One-time, create-only support snapshot and preflight; never runs analysis."""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'integration_evaluation'
sys.path.insert(0, str(SOURCE))
import replay as r
import validate


def main():
    target = r.reserve(HERE / 'support')
    baseline = r.reserve(HERE / 'validation/preflight_20261011')
    files = {}
    for name in ('replay.py', 'replay_worker.py', 'contract.py', 'inventory.py', 'compare_replay.py'):
        rec = r.record(SOURCE / name)
        r.copy(SOURCE / name, target / name, rec)
        files[name] = rec
    lock = {'files': files, 'source_module': str(SOURCE),
            'p2_fix_report': r.record(SOURCE / 'validation/branch_grid_fix_20261011/report.json'),
            'p2_fix_external_review': 'pending; local tests and real CLI validation passed',
            'metric_handoff': r.record(SOURCE / 'validation/replay_handoff_20261011/report.json')}
    r.write(target / 'LOCK.json', lock)
    prior = r.read(SOURCE / 'validation/handoff_20261011/report.json')
    for name, rec in prior['outside_files'].items():
        p = r.STUDY / name
        if 'symlink' in rec: r.require(str(p.resolve()) == rec['symlink'] and p.is_symlink(), 'Changed link')
        else: r.check(p, rec)
    r.write(baseline / 'report.json', {'status': 'passed', 'support_lock': r.record(target / 'LOCK.json'),
        'outside_files': prior['outside_files'], 'integration_module': r.tree(SOURCE),
        'original_indexes': validate.originals()})
    print(json.dumps(r.record(target / 'LOCK.json')))


if __name__ == '__main__': main()
