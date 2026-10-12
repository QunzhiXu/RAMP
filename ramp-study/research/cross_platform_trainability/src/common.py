"""Load every arm for a task as an (n_cells, d) matrix aligned row-for-row with
the task metadata. Alignment is verified by cell_id, never assumed.

Read-only against ../final_xu.
"""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent
import numpy as np
import pandas as pd
import paths

D = paths.D
ARMS = [a for a in D['arms'] if not a.startswith('_')]
AMBIGUOUS = set(D['jobs']['job1_celltype']['exclude_labels'])


# ---------------------------------------------------------------- basic inputs
def load_meta(task):
    m = pd.read_csv(paths.meta_path(task), low_memory=False)
    m['cell_id'] = m['cell_id'].astype(str)
    m['celltype'] = m.get('celltype', pd.Series('Unannotated', index=m.index)) \
                     .fillna('Unannotated').astype(str)
    return m


def load_markers(task):
    return paths.marker_path(task).read_text().split()


# -------------------------------------------------------------- arm assemblers
def _csv_matrix(path, markers):
    """Read a corrected-expression CSV; return (values, cell_ids or None)."""
    df = pd.read_csv(path, low_memory=False)
    ids = None
    for c in ('cell_id', 'cell_ids', 'CellID'):
        if c in df.columns:
            ids = df[c].astype(str).to_numpy()
            df = df.drop(columns=[c])
            break
    if markers:
        cols = [c for c in markers if c in df.columns]
        if len(cols) == len(markers):
            df = df[cols]
    return df.to_numpy(dtype=np.float64), ids


def _reindex(values, ids, order):
    """Put rows in the metadata's cell_id order. Raises if ids do not match."""
    if ids is None:
        if len(values) != len(order):
            raise ValueError(f'no cell ids and row count {len(values)} != {len(order)}')
        return values
    ids = np.asarray(ids).astype(str)
    if set(ids) != set(order):
        raise ValueError(f'cell id sets differ: {len(set(ids))} vs {len(set(order))}, '
                         f'overlap {len(set(ids) & set(order))}')
    pos = pd.Series(np.arange(len(ids)), index=ids)
    return values[pos.reindex(order).to_numpy()]


def _fastmnn(task, files, markers):
    if D['tasks'][[t['name'] for t in D['tasks']].index(task)]['layout'] == 'benchmark':
        emb, ids = _csv_matrix(files[0], [])
        rot = pd.read_csv(files[1], index_col=0)
        # benchmarks store no centres; an additive per-marker constant is
        # removed by the z-scoring applied before every model fit, so
        # reconstruction without centring is equivalent here
        return emb @ rot.to_numpy().T, ids
    comp = np.load(files[0])                                   # values.npy
    rot = pd.read_csv(ROOT / 'data' / task / 'fastmnn_rotation.csv', index_col=0)
    cen = pd.read_csv(ROOT / 'data' / task / 'fastmnn_centers.csv', index_col=0)
    rot = rot.reindex(markers).to_numpy()[:, :comp.shape[1]]
    cen = cen.reindex(markers)['center'].to_numpy()
    return comp @ rot.T + cen, None


def _maxfuse(task, files, train_platform, meta):
    """Amendment 1: use the hub oriented on the training platform."""
    lay = D['tasks'][[t['name'] for t in D['tasks']].index(task)]['layout']
    emb = [f for f in files if 'cell_id' not in f.name]
    ids = [f for f in files if 'cell_id' in f.name]
    if lay == 'benchmark':
        # hub directories are named after the platform but not identically
        # (MEL_Hickey2023_CODEX -> hub_hickey_codex), so match on tokens
        norm = train_platform.lower().replace(' ', '_').replace('-', '_')
        pick = [f for f in emb
                if all(tok in norm for tok in f.parts[-4].lower()[4:].split('_'))]
        if len(pick) != 1:
            raise ValueError(
                f'MaxFuse hub match for {train_platform!r} gave {len(pick)} hits; '
                f'have {sorted({f.parts[-4] for f in emb})}')
        f = pick[0]
        cid = f.parent / 'cell_ids.npy'
        return np.load(f), np.load(cid, allow_pickle=True)
    if task == 'lung_story':
        hub = sorted({f.parent for f in emb})[0]               # hub0 default
        vs = sorted(hub.glob('view*_gcca_embedding.npy'))
        vi = sorted(hub.glob('view*_gcca_cell_ids.npy'))
        return (np.concatenate([np.load(v) for v in vs]),
                np.concatenate([np.load(v, allow_pickle=True) for v in vi]))
    return (np.concatenate([np.load(f) for f in sorted(emb)]),
            np.concatenate([np.load(f, allow_pickle=True) for f in sorted(ids)]))


# ------------------------------------------------------------------ recipes
def virtues_style(X, units, datasets):
    out = np.empty_like(X)
    for u in np.unique(units):
        m = units == u
        out[m] = np.minimum(X[m], np.percentile(X[m], 99, axis=0))
    out = np.log1p(np.clip(out, 0, None))
    for s in np.unique(datasets):
        m = datasets == s
        sd = out[m].std(0, ddof=0)
        out[m] = (out[m] - out[m].mean(0)) / np.where(sd > 0, sd, 1.0)
    return out


def spatium_style(X, units, K=50):
    out = np.empty_like(X)
    for u in np.unique(units):
        m = np.flatnonzero(units == u)
        for j in range(X.shape[1]):
            r = pd.Series(X[m, j]).rank(method='average').to_numpy()
            out[m, j] = np.clip(((r - .5) / len(m) * K).astype(int), 0, K - 1)
    return out


# ---------------------------------------------------------------- entry point
def load_arms(task, train_platform=None, arms=None):
    meta, markers = load_meta(task), load_markers(task)
    order = meta.cell_id.to_numpy()
    units, datasets = meta.unit_key.astype(str).to_numpy(), meta.dataset.to_numpy()
    out, raw = {}, None
    for arm in (arms or ARMS):
        spec, files = D['arms'][arm], paths.arm_files(task, arm)
        if arm == 'Raw':
            v = np.load(files[0]) if files[0].suffix == '.npy' else _csv_matrix(files[0], markers)[0]
            raw = out['Raw'] = np.asarray(v, dtype=np.float64)
        elif spec['kind'] == 'recipe':
            continue                                            # filled in below
        elif arm == 'fastMNN-reconstructed':
            v, ids = _fastmnn(task, files, markers)
            out[arm] = _reindex(np.asarray(v, float), ids, order)
        elif arm == 'MaxFuse':
            v, ids = _maxfuse(task, files, train_platform or datasets[0], meta)
            out[arm] = _reindex(np.asarray(v, float), ids, order)
        else:
            v, ids = ((np.load(files[0]), None) if files[0].suffix == '.npy'
                      else _csv_matrix(files[0], markers))
            out[arm] = _reindex(np.asarray(v, float), ids, order)
    if raw is not None:
        if 'VirTues-style' in (arms or ARMS):
            out['VirTues-style'] = virtues_style(raw, units, datasets)
        if 'Spatium-style' in (arms or ARMS):
            out['Spatium-style'] = spatium_style(raw, units)
    return meta, markers, out
