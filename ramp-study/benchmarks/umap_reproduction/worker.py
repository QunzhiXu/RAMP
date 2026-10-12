"""Execute unchanged historical UMAP fit and evaluate_one on fresh coordinates."""
import argparse
import os
from pathlib import Path
import time

import study as s
from bootstrap import HERE, previous_worker as w, r


def compare_coordinates(current, previous, digest):
    import numpy as np
    r.require(current.shape == previous.shape and current.dtype == previous.dtype and np.isfinite(current).all(),
              'Coordinate shape/dtype/finiteness differs')
    return {'array_equal': bool(np.array_equal(current, previous)),
            'changed_elements': int(np.count_nonzero(current != previous)),
            'maximum_absolute_difference': float(np.max(np.abs(current.astype(float) - previous.astype(float)))),
            'new_sha256_array': digest(current), 'historical_sha256_array': digest(previous)}


def fresh_fit(e, values, method, root):
    r.require(root.is_dir() and not any(root.iterdir()), 'UMAP fit requires a reserved empty directory')
    return e.fit_or_load_umap(values, method, root, allow_fit=True)


def run(output):
    output = Path(output).resolve(strict=True)
    request = r.read(output / 'REQUEST.json')
    r.require(output.parent.parent == HERE.parent and output.parent.name == request['case'] and
              output.name == request['variant'], 'Worker outside its reserved batch')
    parent = r.read(output.parent.parent / 'REQUEST.json')
    r.require({k: v for k, v in request.items() if k not in ('case', 'variant')} ==
              {k: v for k, v in parent.items() if k != 'jobs'}, 'Parent request differs')
    r.require(s.codes() == request['code_records'], 'Worker code differs')
    r.write(output / 'STARTED.json', {'pid': os.getpid()})
    try:
        bundle = Path(request['bundle']).resolve(strict=True)
        manifest = s.inputs(bundle); r.check(bundle / 'MANIFEST.json', request['bundle_manifest'])
        case, variant = next((c, v) for c, v in s.branches(manifest, request['scope'])
                             if (c['case'], v['id']) == (request['case'], request['variant']))
        e = w.load_evaluator(bundle)
        before = w.runtime(); r.write(output / 'ENVIRONMENT_BEFORE.json', before)
        import contract
        import numpy as np
        import pandas as pd
        md = pd.read_csv(bundle / case['case'] / 'metadata.csv.gz', low_memory=False)
        matrix = contract.load_matrix(bundle / variant['input'], variant['method'], md, case['markers'])
        values = e.standardized(matrix, variant['method'])
        old_manifest = r.read(bundle / variant['xy_manifest'])
        r.require(e.sha256_array(values) == old_manifest['representation_sha256'] and
                  e.UMAP_PROTOCOL == old_manifest['protocol'] == s.UMAP_PROTOCOL and old_manifest['labels_used_for_fit'] is False,
                  'Historical UMAP representation/protocol differs')
        r.require(manifest['seeds'] == list(e.SEEDS) and manifest['metric_registry'] == [list(x) for x in e.REGISTRY],
                  'Frozen evaluator protocol differs')
        # This directory is exclusively created here and is empty. The original
        # fit_or_load_umap cannot take its cache-return branch under this condition.
        fit_root = r.reserve(output / 'fit')
        start = time.monotonic()
        coordinates = fresh_fit(e, values, variant['method'], fit_root)
        fit_seconds = time.monotonic() - start
        r.require(np.isfinite(coordinates).all() and coordinates.shape == (len(md), 2), 'Invalid fitted coordinates')
        r.write(output / 'FIT.json', {'protocol': e.UMAP_PROTOCOL, 'fitted_from_empty_directory': True,
                                     'labels_used_for_fit': False, 'seconds': fit_seconds,
                                     'evaluator': r.record(bundle / 'evaluator.py'),
                                     'representation_sha256': e.sha256_array(values)})
        old = np.load(bundle / variant['xy'], allow_pickle=False)
        comparison = compare_coordinates(coordinates, old, e.sha256_array)
        r.require(comparison['historical_sha256_array'] == old_manifest['coordinate_sha256'], 'Old coordinate digest differs')
        r.write(output / 'COORDINATE_COMPARISON.json', comparison)
        print('Fresh UMAP', variant['method'], fit_seconds, comparison, flush=True)
        # Labels enter only below, after the UMAP fit and coordinate comparison.
        raw = next(v for v in case['variants'] if v['method'] == 'Unaligned')
        native = e.standardized(np.load(bundle / raw['input'], allow_pickle=False), 'Unaligned')
        r.reserve(output / 'replicates')
        for number, seed in enumerate(e.SEEDS, 1):
            rows = e.sample_rows(md, seed, 5000)
            celltype, dataset, unit, valid, levels = w.labels(e, md, rows, case['kind'])
            tick = time.monotonic()
            metrics = e.evaluate_one(values[rows], native[rows], celltype, dataset, unit, valid, seed, coordinates[rows])
            r.write(output / 'replicates' / (str(seed) + '.json'), {
                'benchmark': case['case'], 'method': variant['method'], 'sampling_seed': seed,
                'n_cells': len(rows), 'n_joint_celltypes': len(levels), 'sampled_rows_sha256': e.sha256_array(rows),
                'metrics': {k: float(v) if np.isfinite(v) else None for k, v in metrics.items()},
                'seconds': time.monotonic() - tick})
            print('resample', number, '/20', round(time.monotonic() - tick, 2), flush=True)
        after = w.runtime(); r.require(after == before, 'Runtime changed')
        r.write(output / 'ENVIRONMENT_AFTER.json', after)
        s.inputs(bundle); r.check(bundle / 'MANIFEST.json', request['bundle_manifest'])
        r.check(Path(request['python']), request['python_record'])
        s.finish_branch(output, request['code_records'])
    except BaseException as error:
        r.write(output / 'FAILED.json', {'error': repr(error)}); raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--run', type=Path, required=True)
    run(p.parse_args().run)
