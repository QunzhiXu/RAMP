"""Read-only real-data inventory; versioned audit tables only, no fitting.

Historical matrices are checked for hashes/shapes, never consumed as J2 inputs.
This is NOT input certification, alignment, training or a cost pilot.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from trainability.core import ROOT, file_sha256, output_path, require, write_json


def inspect(config_path: Path, out: Path, *, verify_hashes: bool = True) -> dict:
    config = json.loads(config_path.read_text())
    contracts = json.loads((ROOT / 'config/input_contracts.json').read_text())
    fx = ROOT.parent
    out = output_path(out)
    require(not out.exists(), 'Preflight output exists; choose a new version')
    inventory, counts, capacity, identities, task_summary = [], [], [], [], []
    for spec in config['tasks']:
        task, refs = spec['name'], spec['historical_references']
        meta = pd.read_csv(fx / refs['metadata']['path'], dtype=str, keep_default_na=False)
        require({'cell_id', 'dataset', 'unit_key', 'celltype'} <= set(meta), 'Metadata contract mismatch')
        require(meta.cell_id.is_unique and meta.cell_id.ne('').all(), 'Duplicate/missing frozen cell ID')
        require(meta.dataset.ne('').all() and meta.unit_key.ne('').all(), 'Missing batch identity')
        require(meta.groupby('unit_key').dataset.nunique().max() == 1, 'Cross-dataset unit collision')
        meta['celltype'] = meta.celltype.replace('', 'Unannotated')
        markers = (fx / refs['markers']['path']).read_text().splitlines()
        require(bool(markers) and all(markers) and len(markers) == len(set(markers)), 'Invalid markers')
        tab = pd.crosstab(meta.celltype, meta.dataset)
        shared = [c for c in tab.index[(tab > 0).all(axis=1)] if c not in config['exclude_labels']]
        datasets = sorted(meta.dataset.unique())
        require(set(datasets) == {c['dataset'] for c in contracts if c['task'] == task},
                'Contracts do not cover all dataset roles')
        for ds in datasets:
            block = meta[meta.dataset == ds]
            col = spec.get('patient_column')
            verified = spec['patient_identity_status'] == 'verified'
            if verified:
                require(col in block and block[col].ne('').all(), 'Verified patient mapping missing')
            identities.append(dict(task=task, dataset=ds, cells=len(block),
                units=block.unit_key.nunique(), patient_identity_status=spec['patient_identity_status'],
                verified_patients=block[col].nunique() if verified else None,
                available_candidate_columns='|'.join(c for c in ['patient_id', 'donor', 'donor_id', 'specimen_id'] if c in block)))
            for label in shared:
                cells = block[block.celltype == label]
                counts.append(dict(task=task, dataset=ds, label=label, cells=len(cells),
                    class_units=cells.unit_key.nunique(), dataset_units=block.unit_key.nunique(),
                    class_verified_patients=cells[col].nunique() if verified else None))
            capacity.append(dict(task=task, dataset=ds, candidate_shared_classes=len(shared),
                smallest_candidate_class=int(tab.loc[shared, ds].min()) if shared else 0,
                candidate_annotated_cells=int(tab.loc[shared, ds].sum()), post_split_budget_verified=False))
        for role, ref in refs.items():
            path = fx / ref['path']
            require(path.is_file(), f'Missing reference: {path}')
            actual = file_sha256(path) if verify_hashes else None
            require(actual is None or actual == ref['expected_sha256'], f'Frozen reference changed: {path}')
            shape, dtype = None, None
            if path.suffix == '.npy':
                matrix = np.load(path, mmap_mode='r', allow_pickle=False)
                shape, dtype = list(matrix.shape), str(matrix.dtype)
                require(shape == [len(meta), len(markers)], f'Reference dimensions differ: {role}')
            inventory.append(dict(task=task, role=role, path=str(path), bytes=path.stat().st_size,
                expected_sha256=ref['expected_sha256'], actual_sha256=actual,
                hash_verified=actual is not None, shape=json.dumps(shape), dtype=dtype,
                use='historical_reference_only', dependencies_verified=False))
        task_summary.append(dict(task=task, cells=len(meta), datasets=len(datasets),
            shared_markers=len(markers), candidate_classes=shared,
            maximum_directions=len(datasets)*(len(datasets)-1)))
    summary = dict(status='inventory_only_not_input_certification', tasks=len(task_summary),
        task_dataset_roles=len(identities), historical_references=len(inventory),
        hashes_verified=verify_hashes, maximum_directions=sum(x['maximum_directions'] for x in task_summary),
        shared_marker_tasks=sum(x['shared_markers'] for x in task_summary),
        pending_dataset_contracts=sum(c['review_status'] == 'pending' for c in contracts),
        formal_training_started=False, real_alignment_run=False, tasks_detail=task_summary,
        config_sha256=file_sha256(config_path), contracts_sha256=file_sha256(ROOT / 'config/input_contracts.json'))
    out.mkdir(parents=True)
    for name, data in [('source_inventory', inventory), ('class_counts', counts),
                       ('budget_feasibility', capacity), ('identity_review', identities),
                       ('normalization_dependency_review', contracts)]:
        pd.DataFrame(data).to_csv(out / (name+'.csv'), index=False)
    write_json(out / 'SUMMARY.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config/protocol.draft.json')
    parser.add_argument('--out', type=Path, default=ROOT / 'data/preflight_v0_1')
    parser.add_argument('--skip-hashes', action='store_true', help='Report hashes as unverified')
    args = parser.parse_args()
    result = inspect(args.config, args.out, verify_hashes=not args.skip_hashes)
    print(json.dumps({k:v for k,v in result.items() if k != 'tasks_detail'}, indent=2))


if __name__ == '__main__':
    main()
