"""Read-only design preflight; no training, no output in historical folders."""
from pathlib import Path
import hashlib
import json
import sys

import pandas as pd

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
FX = ROOT.parent
OLD = FX.parent / 'cross_platform_trainability'
DESIGN = json.loads((OLD / 'design.json').read_text())
STORY = {'lung_story': 'story_lung_HLA_ABC', 'melanoma_story': 'figure_melanoma',
         'hnscc_story': 'figure_HNSCC'}
EXCLUDE = set(DESIGN['jobs']['job1_celltype']['exclude_labels'])


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    counts, inputs, capacity, dependencies = [], [], [], []
    for spec in DESIGN['tasks']:
        task = spec['name']
        if spec['layout'] == 'benchmark':
            b = FX / 'benchmark_results' / task
            meta_path = b / 'ramp/alignment/input/cell_metadata.csv.gz'
            marker_path = b / 'ramp/alignment/input/markers.txt'
            native = b / 'ramp/alignment/input/raw.npy'
            baseline = b / 'ramp/alignment/data/kmeanslow_global.npy'
            aligned = b / 'ramp/alignment/data/ramp_aligned.npy'
            manifest_path = b / 'cycombine/preprocessing/preprocessing_manifest.json'
            manifest = json.loads(manifest_path.read_text())
            for ds in manifest['datasets']:
                formula = ds.get('transform_formula', '')
                dependencies.append(dict(task=task, dataset=ds['dataset'],
                    source=str(manifest_path), formula=formula,
                    needs_cross_marker_dependency_review=('CLR' in json.dumps(ds)
                                                         or 'all_' in formula)))
        else:
            b = FX / STORY[task] / 'Comparison/input'
            meta_path, marker_path = b / 'evaluation_metadata.csv.gz', b / 'markers.txt'
            native, aligned = b / 'raw.npy', b / 'ramp.npy'
            baseline = {
                'lung_story': FX / 'story_lung_HLA_ABC/data_umap/reference/baseline_SHARED9.npy',
                'melanoma_story': FX / 'figure_melanoma/analysis/alignment/output/kmeanslow_baseline.npy',
                'hnscc_story': FX / 'figure_HNSCC/alignment/output/baseline_kmeanslow.float32.npy'
            }[task]
        meta = pd.read_csv(meta_path, low_memory=False)
        assert meta.cell_id.astype(str).is_unique
        meta['celltype'] = meta.celltype.fillna('Unannotated').astype(str)
        tab = pd.crosstab(meta.celltype, meta.dataset)
        shared = [c for c in tab.index[(tab > 0).all(1)] if c not in EXCLUDE]
        for ds in tab.columns:
            unit_count = int(meta.loc[meta.dataset == ds, 'unit_key'].nunique())
            for label in shared:
                counts.append(dict(task=task, dataset=ds, label=label,
                    cells=int(tab.loc[label, ds]), units=unit_count))
            capacity.append(dict(task=task, dataset=ds, shared_classes=len(shared),
                smallest_shared_class=int(tab.loc[shared, ds].min()),
                annotated_shared_cells=int(tab.loc[shared, ds].sum()), units=unit_count))
        for role, path in [('metadata', meta_path), ('markers', marker_path),
                           ('native_values', native), ('kmeanslow', baseline), ('ramp', aligned)]:
            inputs.append(dict(task=task, role=role, path=str(path), exists=path.exists(),
                sha256=sha(path) if path.exists() else None))
    out = ROOT / 'data'
    out.mkdir(exist_ok=True)
    pd.DataFrame(counts).to_csv(out / 'class_counts.csv', index=False)
    pd.DataFrame(capacity).to_csv(out / 'budget_feasibility.csv', index=False)
    pd.DataFrame(inputs).to_csv(out / 'source_inventory.csv', index=False)
    pd.DataFrame(dependencies).to_csv(out / 'normalization_dependency_review.csv', index=False)
    print(json.dumps(dict(status='design_preflight_only', tasks=len(DESIGN['tasks']),
        task_dataset_roles=len(capacity), input_files=len(inputs),
        missing=[x['path'] for x in inputs if not x['exists']],
        smallest_class=min(x['smallest_shared_class'] for x in capacity),
        formal_training_started=False), indent=2))


if __name__ == '__main__':
    main()
