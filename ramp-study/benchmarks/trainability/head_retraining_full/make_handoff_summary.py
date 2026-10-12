"""Read sealed comparison evidence and produce an exhaustive per-branch summary."""
import argparse
from collections import Counter
import sys
sys.dont_write_bytecode = True
from full_common import *


def summarize(lockdir, comparison, output):
    root = load_lock(lockdir)
    comparison = Path(comparison)
    verify_seal(comparison)
    report = read(comparison / 'report.json')
    check_grid(list(report['groups']), root['groups'])
    branches = {}
    all_ids = []
    for ident, fingerprint in report['groups'].items():
        path = comparison / 'groups' / (ident + '.json')
        require(record(path) == fingerprint, 'Comparison group changed')
        group = read(path)
        expected = group_lock(lockdir, root, ident)
        check_grid([row['id'] for row in group['jobs']], expected['tasks'])
        for row in group['jobs']:
            require(row['job'] == expected['tasks'][row['id']]['job'], 'Comparison job identity changed')
            job = row['job']
            key = job['task_type'] + '__' + job['task']
            counts = branches.setdefault(key, Counter())
            counts.update(jobs=1, predictions=row['predictions'],
                          exact_jobs=int(row['exact']),
                          byte_identical_npz=int(row['prediction_npz_bytes_equal']),
                          changed_predictions=row['changed_predictions'],
                          candidate_records=row['candidate_count'],
                          scientific_result_difference_jobs=int(bool(row['scientific_result_differences'])))
            all_ids.append(row['id'])
    require(len(all_ids) == len(set(all_ids)) == 32370, 'Full summary job grid differs')
    require({k: v['jobs'] for k, v in branches.items()} == root['branches'], 'Summary branch counts differ')
    totals = {name: sum(x[name] for x in branches.values()) for name in
              ('jobs', 'predictions', 'exact_jobs', 'byte_identical_npz', 'changed_predictions', 'candidate_records')}
    require(all(totals[k] == report[k] for k in totals), 'Summary totals differ from comparison')
    result = dict(status=report['status'], scope='All 32370 frozen-representation head retraining jobs',
                  comparison_report=record(comparison / 'report.json'), branches=branches,
                  totals=totals, summary_code=record(Path(__file__)))
    out = reserve(output, [lockdir, comparison])
    write(out / 'report.json', result)
    seal(out, dict(scope='per_branch_comparison_summary'))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lockdir', type=Path, required=True)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(summarize(args.lockdir, args.comparison, args.output))
