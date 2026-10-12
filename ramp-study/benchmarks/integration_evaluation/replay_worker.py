"""One method x 20 seeds, using byte-identical evaluator and frozen coordinates."""
import argparse
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import sys
import time

import replay as r


def load_evaluator(bundle):
    spec = importlib.util.spec_from_file_location('frozen_evaluation_core', bundle / 'evaluator.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def runtime():
    import threadpoolctl
    versions = {name: importlib.metadata.version(name) for name in r.PACKAGES}
    r.require(versions == r.PACKAGES, 'Locked Python dependencies differ')
    threads = {k: os.environ.get(k) for k in r.THREADS}
    r.require(all(v == '1' for v in threads.values()), 'Wrong thread environment')
    pools = threadpoolctl.threadpool_info()
    r.require(all(p['num_threads'] == 1 for p in pools), 'Native thread pool is not single-threaded')
    return {'python': sys.version, 'packages': versions, 'threads': threads, 'threadpools': pools}


def labels(e, metadata, rows, kind):
    """Literal caller semantics preserved; metrics still run in frozen evaluate_one."""
    import numpy as np
    import pandas as pd
    local = metadata.iloc[rows].reset_index(drop=True)
    column = 'celltype_joint_evaluation' if kind == 'technical' and 'celltype_joint_evaluation' in local else 'celltype'
    celltype = local[column].fillna('Unannotated').astype(str).to_numpy() if kind == 'technical' else local[column].astype(str).to_numpy()
    dataset = local.dataset.astype(str).to_numpy(); unit = local.unit_key.astype(str).to_numpy()
    flag = local.joint_evaluation.to_numpy(dtype=bool) if 'joint_evaluation' in local else np.asarray([not bool(e.INVALID_LABEL_PATTERN.search(x)) for x in celltype], dtype=bool)
    counts = pd.Series(celltype[flag]).value_counts()
    present = pd.DataFrame({'celltype': celltype[flag], 'dataset': dataset[flag]}).groupby('celltype').dataset.nunique()
    levels = present[(present >= 2) & (counts >= 30)].index.astype(str).to_numpy()
    r.require(len(levels) >= 2, 'Insufficient joint labels')
    return celltype, dataset, unit, flag & np.isin(celltype, levels), levels


def run(output):
    output = output.resolve(strict=True)
    request = r.read(output / 'REQUEST.json')
    r.require(output.parent.parent == Path(__file__).resolve().parents[1] and
              output.parent.name == request['case'] and output.name == request['variant'],
              'Worker output is not a reserved branch of this controller')
    parent = r.read(output.parent.parent / 'REQUEST.json')
    r.require(all(parent.get(k) == v for k, v in request.items() if k not in ('case', 'variant')),
              'Worker request differs from batch')
    r.require(r.code_records() == request['code_records'], 'Unknown worker code before write')
    r.write(output / 'STARTED.json', {'pid': os.getpid()})
    try:
        bundle = Path(request['bundle']).resolve(strict=True)
        r.check(bundle / 'MANIFEST.json', request['bundle_manifest'])
        r.require(r.code_records() == request['code_records'], 'Controller changed before evaluation')
        manifest = r.verify_bundle(bundle)
        case = next(c for c in manifest['cases'] if c['case'] == request['case'])
        variant = next(v for v in case['variants'] if v['id'] == request['variant'])
        e = load_evaluator(bundle)
        before = runtime(); r.write(output / 'ENVIRONMENT_BEFORE.json', before)
        import contract
        import numpy as np
        import pandas as pd
        metadata = pd.read_csv(bundle / case['case'] / 'metadata.csv.gz', low_memory=False)
        matrix = contract.load_matrix(bundle / variant['input'], variant['method'], metadata, case['markers'])
        values = e.standardized(matrix, variant['method'])
        raw_variant = next(v for v in case['variants'] if v['method'] == 'Unaligned')
        native = e.standardized(np.load(bundle / raw_variant['input'], allow_pickle=False), 'Unaligned')
        coordinates = np.load(bundle / variant['xy'], allow_pickle=False)
        umap = r.read(bundle / variant['xy_manifest'])
        r.require(umap['representation_sha256'] == e.sha256_array(values) and
                  umap['coordinate_sha256'] == e.sha256_array(coordinates) and
                  umap['protocol'] == e.UMAP_PROTOCOL and coordinates.shape == (len(values), 2), 'Frozen UMAP input differs')
        r.require(manifest['seeds'] == list(e.SEEDS) and manifest['metric_registry'] == [list(x) for x in e.REGISTRY], 'Evaluator contract differs')
        (output / 'replicates').mkdir(exist_ok=False)
        start = time.monotonic()
        for number, seed in enumerate(e.SEEDS, 1):
            rows = e.sample_rows(metadata, seed, 5000)
            celltype, dataset, unit, valid, levels = labels(e, metadata, rows, case['kind'])
            tick = time.monotonic()
            result = e.evaluate_one(values[rows], native[rows], celltype, dataset, unit, valid, seed, coordinates[rows])
            result = {k: float(v) if np.isfinite(v) else None for k, v in result.items()}
            r.write(output / 'replicates' / (str(seed) + '.json'), {
                'benchmark': case['case'], 'method': variant['method'], 'sampling_seed': seed,
                'n_cells': len(rows), 'n_joint_celltypes': len(levels),
                'sampled_rows_sha256': e.sha256_array(rows), 'metrics': result,
                'seconds': time.monotonic() - tick})
            print(case['case'], variant['method'], f'{number}/20', round(time.monotonic() - tick, 2), flush=True)
        after = runtime(); r.require(after == before, 'Environment changed during evaluation')
        r.write(output / 'ENVIRONMENT_AFTER.json', after)
        r.require(r.code_records() == request['code_records'], 'Controller changed during evaluation')
        r.check(Path(request['python']), request['python_record'])
        r.verify_bundle(bundle); r.check(bundle / 'MANIFEST.json', request['bundle_manifest'])
        r.write(output / 'RUN_MANIFEST.json', {'files': r.tree(output, exclude=('stdout.log', 'stderr.log')),
                                              'elapsed_seconds': time.monotonic() - start})
        r.write(output / 'COMPLETE.json', {'status': 'complete', 'manifest': r.record(output / 'RUN_MANIFEST.json')})
    except BaseException as error:
        r.write(output / 'FAILED.json', {'error': repr(error)}); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--run', type=Path, required=True)
    run(parser.parse_args().run)
