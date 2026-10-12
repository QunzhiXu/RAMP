"""Compare the two pilot runs to original metrics before launching the full batch."""
from pathlib import Path
import pandas as pd
import numpy as np
import study as s
from bootstrap import r
from compare_replay import DIAGNOSTICS


def main():
    bundle = s.HERE.parent / 'integration_evaluation/runs/replay_inputs_20261011'
    snapshot = s.HERE.parent / 'integration_evaluation/frozen_20261011_r2'
    batch = s.HERE / 'runs/pilot_20261011'
    result = s.verify_run(batch, bundle); manifest = s.inputs(bundle)
    catalog = r.read(bundle / 'trust/catalog.json'); checks = []
    for case, variant in s.branches(manifest, 'pilot'):
        path = Path(case['evaluation_directory']) / ('replicates/' + variant['key'] + '.csv' if case['case'].endswith('_story')
                                                     else 'metric_battery_replicates.csv')
        file = snapshot / 'sources' / path; r.check(file, catalog['sources'][path.as_posix()])
        old = pd.read_csv(file, float_precision='round_trip')
        old = old[old.method.eq(variant['method'])].set_index('sampling_seed')
        names = [row[0] for row in manifest['metric_registry']] + DIAGNOSTICS
        records = [r.read(batch / case['case'] / variant['id'] / 'replicates' / (str(seed) + '.json')) for seed in manifest['seeds']]
        new = np.array([[x['metrics'][name] for name in names] for x in records], dtype=float)
        previous = old.loc[manifest['seeds'], names].to_numpy(float)
        equal = bool(np.array_equal(new, previous)); r.require(equal, 'Pilot metric difference requires investigation')
        checks.append({'case': case['case'], 'method': variant['method'], 'metric_and_diagnostic_values': int(new.size),
                       'exact': equal, 'coordinates': r.read(batch / case['case'] / variant['id'] / 'COORDINATE_COMPARISON.json')})
    out = r.reserve(s.HERE / 'validation/pilot_20261011')
    r.write(out / 'report.json', {'status': 'passed', 'verification': result, 'checks': checks,
                                 'batch_manifest': r.record(batch / 'BATCH_MANIFEST.json'), 'codes': s.codes()})
    print('Two pilots: coordinates and all 1600 metric/diagnostic values exact')


if __name__ == '__main__': main()
