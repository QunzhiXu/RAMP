"""Post-fit figure metrics using frozen label rules and the existing ROI.

No labels enter fitting. This does not reselect the parent ROI or re-extract images.
"""
import argparse
import ast
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from common import HERE, STORY, EXP, record, read, write, sha, reserve
from prepare import verify as verify_lock
from study import verify_run
from compare import table_comparison

EXTRA = [f'{STORY}/sampled_data/{s}/locked_metadata.csv.gz' for s in ('bm_codex', 'bm_cytof', 'triana_abseq')]
EXTRA += [f'{STORY}/figure/panel_B/generate_panel_B_combined.py',
          f'{STORY}/figure/panel_B/data/donor_cd19_specificity.csv',
          f'{EXP}/scripts/assemble_cd19_layout_v11.py']
EXTRA += [f'{EXP}/figure_layout_v11/data/{p}' for p in (
    'B_identity_metrics_by_donor.csv', 'B_identity_metrics_by_scope.csv',
    'geometry_only_zoom_windows.csv', 'zoom_ROI_cells.csv.gz')]
SUPPLEMENT_SHA = 'daa3551312e3424a130492fe6d1d4e8729e713a7ea417de6f3d5c544e34e44a0'


def functions(path, names, namespace):
    tree = ast.parse(Path(path).read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise ValueError('Frozen helper functions missing')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def bind(lockdir, output):
    lock = verify_lock(lockdir, source=True)
    source = Path(lock['source_root'])
    records = {rel: record(source / rel) for rel in EXTRA}
    out = reserve(output, [source, Path(lockdir)])
    write(out / 'LOCK.json', {'base_lock_sha256': sha(Path(lockdir) / 'LOCK.json'), 'files': records})
    write(out / 'COMPLETE.json', {'status': 'complete', 'lock_sha256': sha(out / 'LOCK.json')})
    return {'files': len(records), 'lock_sha256': sha(out / 'LOCK.json')}


def calculate(lockdir, supplementary, run, output):
    verify_run(lockdir, run)
    lock = verify_lock(lockdir, source=True)
    source = Path(lock['source_root']); tree = Path(run).resolve() / 'tree'
    supplementary = Path(supplementary)
    if sha(supplementary / 'LOCK.json') != SUPPLEMENT_SHA:
        raise ValueError('Unrecognized supplementary binding')
    if read(supplementary / 'COMPLETE.json') != {'status': 'complete', 'lock_sha256': sha(supplementary / 'LOCK.json')}:
        raise ValueError('Supplementary binding incomplete')
    binding = read(supplementary / 'LOCK.json')
    if set(binding['files']) != set(EXTRA) or binding['base_lock_sha256'] != sha(Path(lockdir) / 'LOCK.json'):
        raise ValueError('Supplementary grid differs')
    for rel, expected in binding['files'].items():
        if record(source / rel) != expected:
            raise ValueError(f'Post-fit source changed: {rel}')
    namespace = dict(np=np, pd=pd, roc_auc_score=roc_auc_score, SAMPLE=source / STORY / 'sampled_data')
    helpers = functions(source / STORY / 'figure/panel_B/generate_panel_B_combined.py',
                        ['label_masks', 'compute_specificity'], namespace)
    specificity = helpers['compute_specificity']()
    helper = functions(source / EXP / 'scripts/assemble_cd19_layout_v11.py', ['metrics'],
                       dict(np=np, roc_auc_score=roc_auc_score, average_precision_score=average_precision_score))['metrics']
    md = pd.read_csv(source / STORY / 'sampled_data/bm_codex/locked_metadata.csv.gz')
    pm = pd.read_csv(tree / STORY / 'prediction/output/codex_prediction_metadata.csv.gz')
    if not md.cell_id.astype(str).equals(pm.cell_id.astype(str)):
        raise ValueError('Post-fit identity mismatch')
    with np.load(tree / STORY / 'prediction/output/codex_cd19_prediction.npz') as z:
        prediction = z['predicted_kmeanslow'].astype(float)
    truth = md.cluster_anno_l1.eq('B-Cells').to_numpy()
    donor = pd.DataFrame([helper(d, truth[ii], prediction[ii]) for d, ii in md.groupby('donor_id', sort=True).indices.items()])
    full = pd.read_csv(tree / EXP / 'real_spatial0913/data/H14_full_predictions.csv.gz')
    roi = read(source / EXP / 'real_spatial0913/results/selected_roi.json')
    # Reapply the frozen, post-hoc F1-selected parent rectangle. Do not optimize a new one.
    xy = full[['x.coord', 'y.coord']].to_numpy()
    inside = (xy[:, 0] >= roi['x0']) & (xy[:, 0] < roi['x1']) & (xy[:, 1] >= roi['y0']) & (xy[:, 1] < roi['y1'])
    parent = full.loc[inside].copy()
    xy = parent[['x.coord', 'y.coord']].to_numpy() - np.array([roi['x0'], roi['y0']])
    windows = []
    for yy in range(0, 1024 - 512 + 1, 16):
        for xx in range(0, 1024 - 512 + 1, 16):
            inside = (xy[:, 0] >= xx) & (xy[:, 0] < xx + 512) & (xy[:, 1] >= yy) & (xy[:, 1] < yy + 512)
            windows.append({'x0': xx, 'y0': yy, 'n_cells': int(inside.sum())})
    windows = pd.DataFrame(windows).sort_values(['n_cells', 'y0', 'x0'], ascending=[False, True, True])
    best = windows.iloc[0]; cx, cy = int(best.x0), int(best.y0)
    zoom = parent.loc[(xy[:, 0] >= cx) & (xy[:, 0] < cx + 512) & (xy[:, 1] >= cy) & (xy[:, 1] < cy + 512)].copy()
    if (cx, cy, len(zoom), int(zoom.annotated_b.sum())) != (384, 320, 251, 43):
        raise ValueError('Frozen geometry/ROI changed')
    scopes = pd.DataFrame([helper('12 donors pooled', truth, prediction)] + [
        helper(name, f.annotated_b.to_numpy(bool), f.predicted_cd19.to_numpy()) for name, f in (
            ('H14 full donor', full), ('H14 original ROI', parent), ('H14 zoom ROI', zoom))])
    output = reserve(output, [source, Path(run), Path(lockdir), supplementary])
    try:
        frames = {'donor_cd19_specificity.csv': specificity, 'B_identity_metrics_by_donor.csv': donor,
                  'B_identity_metrics_by_scope.csv': scopes, 'geometry_only_zoom_windows.csv': windows,
                  'zoom_ROI_cells.csv.gz': zoom}
        comparisons = {}
        for name, frame in frames.items():
            frame.to_csv(output / name, index=False)
            old = source / (f'{STORY}/figure/panel_B/data/{name}' if name.startswith('donor_cd19') else f'{EXP}/figure_layout_v11/data/{name}')
            comparisons[name] = table_comparison(output / name, old)
        for rel, expected in binding['files'].items():
            if record(source / rel) != expected:
                raise ValueError(f'Post-fit source changed: {rel}')
        report = {'status': 'exact' if all(r['equal'] for r in comparisons.values()) else 'completed_with_differences',
                  'comparison': comparisons, 'mean_B_identity_AUROC': float(donor.auroc.mean()),
                  'specificity_donors': len(specificity), 'post_fit_label_use_only': True,
                  'parent_ROI': 'Previously selected by F1; retained, not reselected',
                  'zoom': {'x0': roi['x0'] + cx, 'y0': roi['y0'] + cy, 'side': 512, 'cells': len(zoom), 'B_cells': int(zoom.annotated_b.sum())},
                  'image_extraction_or_rendering': False, 'supplementary_lock_sha256': sha(supplementary / 'LOCK.json'),
                  'code': record(Path(__file__)), 'base_run_manifest_sha256': sha(Path(run) / 'RUN_MANIFEST.json')}
        write(output / 'report.json', report)
        write(output / 'COMPLETE.json', {'status': 'complete', 'report_sha256': sha(output / 'report.json')})
        return report
    except BaseException as exc:
        write(output / 'FAILED.json', {'error': repr(exc)})
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['bind', 'run']); p.add_argument('--lockdir', required=True)
    p.add_argument('--supplementary'); p.add_argument('--run'); p.add_argument('--out', required=True)
    a = p.parse_args()
    print(json.dumps(bind(a.lockdir, a.out) if a.mode == 'bind' else calculate(a.lockdir, a.supplementary, a.run, a.out), indent=2))
