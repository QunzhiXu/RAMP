"""Read-only comparison of v0.4 J4 pilot with preserved v0.3 outputs."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
OLD = ROOT / 'validation/e2e_pilot_bone_v0_3_final'


def read(path):
    return json.loads(path.read_text())


def state_equal(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(state_equal(a[k], b[k]) for k in a)
    return torch.equal(a, b)


def main():
    checks = []

    def check(name, passed):
        checks.append(dict(check=name, passed=bool(passed)))

    protected = read(OUT / 'PRIOR_ARTIFACTS_SHA256.json')
    for name, expected in protected.items():
        check('unchanged:' + name, hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected)
    for method in ('unaligned', 'ramp'):
        old = OLD / 'jobs' / ('J4_' + method)
        new = OUT / 'jobs' / ('J4_' + method)
        before, after = read(old / 'RESULT.json'), read(new / 'RESULT.json')
        ledger = read(new / 'ledger.json')
        ledger.pop('readout_groups')
        check(method + ':same_probe_corpus_scaler_and_configuration', ledger == read(old / 'ledger.json'))
        mapping = {name: name for name in before['scores']}
        for arm in ('A-only', 'A+B'):
            mapping[arm + '_random_supervised'] = 'random_supervised'
        check(method + ':six_unique_readouts', len(after['scores']) == len(after['receipts']) == 6)
        check(method + ':shared_baseline', after['shared_baseline']['independent_fits'] == 1
              and not after['shared_baseline']['independent_replicates'])
        for old_name, new_name in mapping.items():
            with np.load(old / (old_name + '.npz'), allow_pickle=False) as a, \
                    np.load(new / (new_name + '.npz'), allow_pickle=False) as b:
                check(method + ':' + old_name + ':identical_ids', np.array_equal(a['cell_ids'], b['cell_ids']))
                check(method + ':' + old_name + ':identical_predictions', np.array_equal(a['predictions'], b['predictions']))
            check(method + ':' + old_name + ':identical_scores', before['scores'][old_name] == after['scores'][new_name])
        checkpoint_mapping = {k: v for k, v in mapping.items() if (old / (k + '.pt')).exists()}
        checkpoint_mapping.update({arm + '_pretrained': arm + '_pretrained' for arm in ('A-only', 'A+B')})
        for old_name, new_name in checkpoint_mapping.items():
            a = torch.load(old / (old_name + '.pt'), weights_only=True)
            b = torch.load(new / (new_name + '.pt'), weights_only=True)
            check(method + ':' + old_name + ':identical_model_weights', state_equal(a, b))
        for old_name, new_name in mapping.items():
            if 'selected_updates' not in before['training'][old_name]:
                continue
            a, b = before['training'][old_name], after['training'][new_name]
            for field in ('selected_updates', 'executed_updates', 'source_validation', 'trainable_parameters'):
                check(method + ':' + old_name + ':' + field, a[field] == b[field])
            history = [{k: row[k] for k in ('updates', 'source_macro_f1')} for row in b['source_history']]
            check(method + ':' + old_name + ':same_source_validation_trajectory', a['source_history'] == history)
        previous_rep = ROOT / 'validation/e2e_pilot_bone_v0_3/representations/bone_marrow/complete' / method
        new_rep = OUT / 'representations/bone_marrow/complete' / method
        for filename in ('values.npy', 'cell_ids.npy'):
            check(method + ':refit_identical_' + filename,
                  np.array_equal(np.load(previous_rep / filename, allow_pickle=False),
                                 np.load(new_rep / filename, allow_pickle=False)))
    result = dict(status='passed' if all(c['passed'] for c in checks) else 'failed',
                  timestamp_utc=datetime.now(timezone.utc).isoformat(),
                  scope='Two 9000-cell J4 regression pilots; not a full performance experiment',
                  prior_artifacts_checked=len(protected), prediction_comparisons=14,
                  checks=checks, formal_experiment_run=False)
    (OUT / 'REGRESSION.json').write_text(json.dumps(result, indent=2) + '\n')
    print(result['status'], len(checks), 'checks;', len(protected), 'prior artifacts unchanged')
    for row in checks:
        if not row['passed']:
            print('FAILED', row['check'])
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
