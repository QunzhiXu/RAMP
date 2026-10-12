"""Synthetic contract, explicitly substituted in tests only; never a cohort fit."""
import copy
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import harmony_io as io
import inventory as inv
import study


@contextmanager
def fixture(root, kind='technical', platforms=3):
    import numpy as np
    import pandas as pd
    root = Path(root).resolve(strict=True)
    source, snapshot = root / 'source', root / 'snapshot'
    source.mkdir(); snapshot.mkdir()
    real = inv.verify_local(); old = next(s for s in real['branches'] if s['kind'] == kind)
    spec = copy.deepcopy(old)
    markers = [f'M{i}' for i in range(5)]
    datasets = [f'platform_{i}' for i in range(platforms)]
    n = 120 * platforms; rng = np.random.default_rng(1729)
    groups = np.repeat(np.arange(platforms), 120)
    values = rng.normal(3, 0.7, (n, len(markers))) + groups[:, None] * np.linspace(-0.3, 0.4, len(markers))
    inp = pd.DataFrame(values, columns=markers)
    inp.insert(0, 'unit_key', [f'{datasets[g]}_unit{i % 3}' for i, g in enumerate(groups)])
    inp.insert(0, 'dataset', [datasets[g] for g in groups])
    inp.insert(0, 'cell_id', [f'cell_{i:05}' for i in range(n)])
    spec.update(case='synthetic_' + kind, cells=n, units=3*platforms, markers=markers, datasets=datasets,
                parameters={**inv.PARAMETERS, 'resolved_nclust': min(round(n/30), 100)})
    input_path = source / spec['input']; input_path.parent.mkdir(parents=True)
    inp.to_csv(input_path, index=False, float_format='%.17g', compression={'method': 'gzip', 'mtime': 0})
    (source / spec['marker_file']).write_text('\n'.join(markers) + '\n')
    config = copy.deepcopy(io.read_json(inv.SNAPSHOT / 'study_sources' / old['config']))
    if kind == 'technical':
        (source / spec['record']).write_text('Synthetic continuous data, identity input transform.\n')
        config['batch_levels'] = datasets
        config['input_scale'].update(input_sha256=io.sha(input_path), marker_file_sha256=io.sha(source / spec['marker_file']),
            preprocessing_record_sha256=io.sha(source / spec['record']), preprocessing_record_file='/old/missing/record.txt')
        manifest = {'status': 'complete', 'synthetic': True}
    else:
        inp.iloc[:, :3].to_csv(source / spec['metadata'], index=False, compression={'method': 'gzip', 'mtime': 0})
        with (source / spec['ids']).open('xb') as f: np.save(f, inp.cell_id.to_numpy(dtype='U'))
        manifest = {'synthetic': True, 'output_sha256': {Path(spec[k]).name: io.sha(source / spec[k]) for k in ('input', 'marker_file', 'metadata', 'ids')}}
        config.update(case=spec['case'], datasets=datasets, markers=markers)
    io.create_json(source / spec['input_manifest'], manifest)
    if kind == 'story': config['input_manifest_sha256'] = io.sha(source / spec['input_manifest'])
    (source / spec['config']).parent.mkdir(parents=True, exist_ok=True)
    io.create_json(source / spec['config'], config)
    files = []
    needed = [spec[k] for k in ('input', 'marker_file', 'config', 'input_manifest')]
    needed += [spec['record']] if kind == 'technical' else [spec['metadata'], spec['ids']]
    for name in needed: files.append({'source': name, **io.record(source / name)})
    for row in real['files']:
        if row.get('copy', '').startswith('adapter/') or (kind == 'story' and row['source'] == spec['base'] + '/scripts/run_r.R'):
            io.copy_new(inv.SNAPSHOT / row['copy'], snapshot / row['copy'], row)
            files.append(row)
    io.copy_new(inv.SNAPSHOT / 'registry.json', snapshot / 'registry.json', io.record(inv.SNAPSHOT / 'registry.json'))
    catalog = {'status': 'complete', 'registry_sha256': inv.REGISTRY_SHA, 'branches': [spec], 'files': files, 'links': []}
    io.create_json(snapshot / 'catalog.json', catalog)
    io.create_json(snapshot / 'COMPLETE.json', {'status': 'complete', 'manifest_sha256': io.sha(snapshot / 'catalog.json')})
    with patch.object(inv, 'SNAPSHOT', snapshot), patch.object(inv, 'CATALOG_SHA', io.sha(snapshot / 'catalog.json')):
        yield source, spec
