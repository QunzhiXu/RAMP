"""Read-only verification of frozen Senosain labels; no relabeling or training."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FX = ROOT.parent
SEN = FX / 'story_test_lung/shared_inputs/senosain'
ANN = FX / 'raw_data/lung_cancer/Senosain_2025_CyTOF_and_omics/annotation/rough_annotation'
HASHES = {}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def track(path):
    HASHES[str(path)] = sha(path)
    return path


def save(name, frame):
    frame.to_csv(HERE / name, index=False)


def main():
    old = pd.read_csv(track(SEN / 'senosain_exact70k_metadata.csv.gz'), dtype={'patient_id': str})
    old['raw_row'] = np.arange(len(old))
    story = pd.read_csv(track(FX / 'story_lung_HLA_ABC/data_umap/data/CELL_METADATA.csv.gz'),
                        dtype=str, keep_default_na=False)
    story = story[story.dataset.eq('Senosain CyTOF')].copy()
    md = story[['source_cell_id', 'provider_cell_type', 'cell_key']].merge(old,
              on='source_cell_id', how='left', validate='one_to_one')
    assert len(md) == 28000 and md.raw_row.notna().all()
    assert md.provider_cell_type.eq(md.annotation).all()
    assert md.groupby('patient_id').size().eq(1000).all()
    markers = pd.read_csv(track(SEN / 'senosain_raw35_markers.csv')).marker.tolist()
    full = np.load(track(SEN / 'senosain_exact70k_raw35.npy'), mmap_mode='r')
    x = np.asarray(full[md.raw_row.to_numpy()], dtype=float)
    assert x.shape == (28000, 35) and np.isfinite(x).all()
    for name in ['manifest.json', 'metadata.csv.gz', 'native.npy']:
        track(ROOT / 'data/real_inputs_v0_3_r2/lung_story' / name)
    # Verify native common-marker values through source IDs, not matrix position.
    frozen = pd.read_csv(ROOT / 'data/real_inputs_v0_3_r2/lung_story/metadata.csv.gz', dtype=str)
    frozen['frozen_row'] = np.arange(len(frozen))
    joined = md[['cell_key']].merge(frozen[['cell_id', 'frozen_row']],
                                  left_on='cell_key', right_on='cell_id', validate='one_to_one')
    native = np.load(ROOT / 'data/real_inputs_v0_3_r2/lung_story/native.npy', mmap_mode='r')
    shared = ['CD3', 'CD4', 'CD8', 'CD31', 'KI67', 'VIMENTIN', 'CYTOKERATIN', 'CD45', 'HLA_DR']
    assert np.array_equal(x[:, [markers.index(m) for m in shared]], native[joined.frozen_row])
    # Replicate archived per-patient KMeans-high definition solely as a diagnostic.
    gate_markers = ['VIMENTIN', 'CD45', 'CD31', 'EPCAM', 'CYTOKERATIN', 'CK7', 'CD3', 'CD4', 'CD8', 'CD11B', 'KI67']
    positive = {}
    thresholds = []
    for marker in gate_markers:
        v = x[:, markers.index(marker)]
        high = np.zeros(len(md), dtype=bool)
        for patient, g in md.groupby('patient_id', sort=False):
            ix = g.index.to_numpy()
            values = v[ix]
            median, mean = float(np.median(values)), float(values.mean())
            cofactor = max(mean if median == 0 else median, 1e-6)
            z = np.arcsinh(values / cofactor)
            km = KMeans(n_clusters=2, n_init=20, random_state=20260822,
                        max_iter=500, algorithm='lloyd').fit(z[:, None])
            hi = int(np.argmax(km.cluster_centers_.ravel()))
            high[ix] = km.labels_ == hi
            thresholds.append(dict(patient_id=patient, marker=marker, cofactor=cofactor,
                                   threshold_arcsinh=float(km.cluster_centers_.mean()),
                                   fraction_high=float(high[ix].mean())))
        positive[marker] = high
    save('DIAGNOSTIC_THRESHOLDS.csv', pd.DataFrame(thresholds))
    p = positive
    gates = {
        'archived_strict_mesenchymal': p['VIMENTIN'] & ~p['CD45'] & ~p['CD31'] & ~p['EPCAM'] & ~p['CYTOKERATIN'] & ~p['CK7'],
        'epithelial_any': p['EPCAM'] | p['CYTOKERATIN'] | p['CK7'],
        'epithelial_any_CD45low': (p['EPCAM'] | p['CYTOKERATIN'] | p['CK7']) & ~p['CD45'],
        'CD45_CD3': p['CD45'] & p['CD3'],
        'CD45_CD3low_CD11B': p['CD45'] & ~p['CD3'] & p['CD11B'],
        'CD31_CD45low': p['CD31'] & ~p['CD45'],
    }
    expected = md.paper_mesenchymal_gate_recomputed.to_numpy(dtype=bool)
    assert np.array_equal(gates['archived_strict_mesenchymal'], expected)
    cell_diagnostics = pd.DataFrame({'source_cell_id': md.source_cell_id,
                                    'patient_id': md.patient_id, 'annotation': md.annotation})
    for name, vals in {**p, **gates}.items():
        cell_diagnostics[name] = vals.astype(int)
    save('CELL_DIAGNOSTICS.csv.gz', cell_diagnostics)
    bypatient = cell_diagnostics.groupby(['annotation', 'patient_id']).agg(
        cells=('source_cell_id', 'size'), **{name: (name, 'mean') for name in list(p) + list(gates)}).reset_index()
    save('GATE_FRACTIONS_PER_PATIENT.csv', bypatient)
    rows = []
    for label, group in cell_diagnostics.groupby('annotation'):
        patient = bypatient[bypatient.annotation.eq(label)]
        row = dict(annotation=label, cells=len(group), patients=group.patient_id.nunique())
        for name in list(p) + list(gates):
            row[name + '_cell_fraction'] = group[name].mean()
            row[name + '_patient_mean_fraction'] = patient[name].mean()
        rows.append(row)
    save('GATE_FRACTIONS_SUMMARY.csv', pd.DataFrame(rows))
    # Continuous raw-marker comparisons avoid drawing conclusions from one gate.
    contrasts = [
        ('Cycling tumor / epithelial', 'Tumor / epithelial', ['KI67', 'EPCAM', 'CYTOKERATIN', 'CK7', 'CD45', 'CD3']),
        ('Stromal / mesenchymal', 'Tumor / epithelial', ['VIMENTIN', 'EPCAM', 'CYTOKERATIN', 'CK7', 'CD45', 'CD3']),
        ('CD8 T', 'CD4 T', ['CD8', 'CD4', 'CD3']),
    ]
    aucrows = []
    for pos, neg, ms in contrasts:
        for patient, group in md.groupby('patient_id'):
            ix = group.index[group.annotation.isin([pos, neg])].to_numpy()
            y = md.loc[ix, 'annotation'].eq(pos).to_numpy()
            if min(y.sum(), (~y).sum()) < 5:
                continue
            for marker in ms:
                aucrows.append(dict(positive=pos, negative=neg, patient_id=patient,
                                    marker=marker, n_positive=int(y.sum()), n_negative=int((~y).sum()),
                                    auc=roc_auc_score(y, x[ix, markers.index(marker)])))
    auc = pd.DataFrame(aucrows)
    save('RAW_MARKER_AUC_PER_PATIENT.csv', auc)
    save('RAW_MARKER_AUC_SUMMARY.csv', auc.groupby(['positive', 'negative', 'marker']).agg(
        patients=('auc', 'size'), mean_auc=('auc', 'mean'), median_auc=('auc', 'median')).reset_index())
    newer = pd.read_csv(track(ANN / 'senosain_140k_final_cell_annotations.csv.gz'), dtype=str)
    overlap = md.merge(newer[['source_cell_id', 'final_cell_type', 'annotation_confidence',
                             'lineage_reference_use', 'qc_recommendation']], on='source_cell_id', validate='one_to_one')
    save('ANNOTATION_OVERLAP.csv', overlap[['source_cell_id', 'annotation', 'final_cell_type',
                                        'annotation_confidence', 'lineage_reference_use', 'qc_recommendation']])
    save('ANNOTATION_CROSSTAB.csv', overlap.groupby(['annotation', 'final_cell_type']).size().rename('cells').reset_index())
    counts = md.groupby('annotation').size().rename('cells').reset_index()
    counts['fraction_all_28k'] = counts.cells / len(md)
    counts['fraction_J1_eligible'] = counts.cells / md.annotation.ne('Ambiguous / mixed').sum()
    save('LABEL_COUNTS.csv', counts)
    checks = dict(cells=len(md), patients=md.patient_id.nunique(), labels_exact_match_to_archived=True,
                  full_panel_to_common9_exact_match=True, mesenchymal_gate_exact_match=True,
                  strict_mesenchymal_cells=int(expected.sum()), newer_overlap=len(overlap),
                  old_annotation_generator_recovered=False, independent_ground_truth_available=False,
                  original_data_modified=False, models_refitted=False,
                  diagnostic_gate_caveat='KMeans high/low is a distribution split, not an independent biological truth or a relabeling rule.',
                  input_hashes=HASHES, input_hashes_unchanged=all(sha(path) == before for path, before in HASHES.items()))
    assert checks['input_hashes_unchanged']
    (HERE / 'CHECKS.json').write_text(json.dumps(checks, indent=2))
    print(pd.DataFrame(rows).to_string(index=False))
    print(auc.groupby(['positive', 'marker']).auc.mean().to_string())
    print(json.dumps({k:v for k,v in checks.items() if k != 'input_hashes'}, indent=2))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
