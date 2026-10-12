"""Reconstruct consumed observed fields, not the lost historical broad-table producer.

No prior predicted PD1/CD69 columns are regenerated or supplied to the new fit.
The cohort readers and survival functions are frozen historical code. Spatial
joins and nearest-centroid distances are explicit new code, checked against
all frozen consumed cells/patients only after execution.
"""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from common import HERE, STORY, read, write, record, stage_root
from specs import A, R, ALIGN

def load(path):
    sys.path.insert(0, str(Path(path).parent))
    spec = importlib.util.spec_from_file_location('frozen_' + Path(path).stem, path)
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

def observed_inputs(run):
    import numpy as np
    import pandas as pd
    from scipy.spatial import cKDTree
    tree = Path(run) / 'tree'; story = tree / STORY
    builder = load(story / 'inputs/cohort_package/build_story_sample.py')
    _, mold, clinical = builder.build_moldoveanu()
    _, eng, _ = builder.build_engelbrecht()
    for name, frame in [('moldoveanu_imc', mold), ('engelbrecht_citeseq', eng)]:
        folder = story / 'inputs/cohort_package' / name; folder.mkdir()
        builder.write_csv_gz(frame, folder / 'locked_metadata.csv.gz')
    identity = pd.read_csv(tree / ALIGN / 'cell_metadata.csv.gz')
    if len(identity) != 165686 or not identity.cell_uid.is_unique:
        raise ValueError('Alignment identity differs')
    fields = ['Cluster.v2', 'Response', 'Known_Deceased', 'Survival_from_Rx_Start', 'Treatment', 'prob.Ki67']
    observed = identity[['dataset', 'donor', 'cell_uid']].rename(columns={'cell_uid': 'cell_uid_ordered'})
    observed = observed.merge(mold[['cell_uid', *fields]], left_on='cell_uid_ordered', right_on='cell_uid',
                              how='left', sort=False, validate='one_to_one').drop(columns='cell_uid')
    distance = pd.Series(np.nan, index=mold.cell_uid, dtype=float)
    for patient, group in mold.groupby('Sample_ID', sort=True):
        xy = group[['Location_Center_X', 'Location_Center_Y']].to_numpy(float)
        if not np.isfinite(xy).all():
            raise ValueError('Nonfinite provider coordinates')
        tumor = group['Cluster.v2'].eq('melano').to_numpy()
        cd8 = group['Cluster.v2'].isin(['Tc.ae', 'Tc.naive']).to_numpy()
        if not tumor.any():
            raise ValueError('No tumor centroid in ' + patient)
        distance.loc[group.loc[cd8, 'cell_uid']] = cKDTree(xy[tumor]).query(xy[cd8], k=1)[0]
    observed['distance_to_nearest_tumor'] = observed.cell_uid_ordered.map(distance)
    if not observed.cell_uid_ordered.equals(identity.cell_uid):
        raise ValueError('Observed row order changed')
    builder.write_csv_gz(observed, story / 'inputs/spatial_cell_metadata.csv.gz')
    patient_rows = []
    for patient, group in observed.loc[observed.dataset.eq('Moldoveanu IMC')].groupby('donor', sort=True):
        n = len(group); tcae = group['Cluster.v2'].eq('Tc.ae')
        cd8 = group['Cluster.v2'].isin(['Tc.ae', 'Tc.naive'])
        ki67 = tcae & group['prob.Ki67'].ge(.5)
        target = ki67 & group.distance_to_nearest_tumor.le(30)
        row = {'patient': patient, 'n_all_cells': n, 'n_cd8': int(cd8.sum()), 'n_tcae': int(tcae.sum()),
               'n_observed_ki67_tcae': int(ki67.sum()), 'n_observed_30um': int(target.sum()),
               'cd8_per_all': cd8.sum()/n, 'tcae_per_all': tcae.sum()/n,
               'observed_ki67_tcae_per_all': ki67.sum()/n,
               'observed_ki67_contact30_tcae_per_all': target.sum()/n,
               'observed_30um_per_all': target.sum()/n,
               'observed_ki67_contact30_fraction_within_tcae': target.sum()/tcae.sum() if tcae.sum() else np.nan}
        c = clinical.set_index('Sample_ID').loc[patient]
        for key in ['Known_Deceased', 'Survival_from_Rx_Start', 'Response', 'irRC', 'Treatment',
                    'Age_last_alive', 'Sex', 'Tissue_Source_Simplified', 'Lymphocyte_Infiltration']:
            row[key] = c[key]
        patient_rows.append(row)
    features = pd.DataFrame(patient_rows)
    if len(features) != 30 or features.n_observed_30um.sum() != 478 or features.Known_Deceased.sum() != 20:
        raise ValueError('Observed cohort contract differs')
    features.to_csv(story / 'inputs/observed_patient_level_features.csv', index=False)
    write(story / 'inputs/REBUILD.json', {'cells': len(observed), 'patients': 30, 'target_cells': 478,
         'consumed_spatial_columns': observed.columns.tolist(), 'cohort_readers': 'unchanged historical functions',
         'spatial_join_and_distance': 'new explicit reconstruction, not the original missing producer',
         'not_rebuilt': ['legacy predicted PD1/CD69 fields', 'legacy response and feature-screen tables']})

def observed_survival(run):
    tree = Path(run) / 'tree'; out = tree / STORY / 'analysis/observed_survival'
    out.mkdir(parents=True)
    lock = read(Path(run) / 'LOCK.json'); r = lock['r_executable']
    if record(Path(r['path']).resolve()) != {k: r[k] for k in ('bytes', 'sha256')}:
        raise ValueError('R executable differs')
    command = [r['path'], '--vanilla', str(HERE / 'survival.R'), str(tree / R / 'run_observed20_sensitivity.R'),
               str(tree / STORY / 'inputs/observed_patient_level_features.csv'), str(out)]
    write(out / 'COMMAND.json', command)
    env = os.environ.copy()
    env['TMPDIR'] = '/private/tmp'  # R 4.3.2 rejects spaces in R_TempDir.
    write(out / 'R_ENVIRONMENT_POLICY.json', {'TMPDIR': env['TMPDIR'], 'LC_ALL': env['LC_ALL'], 'LANG': env['LANG']})
    result = subprocess.run(command, text=True, capture_output=True, env=env)
    (out / 'stdout.log').write_text(result.stdout); (out / 'stderr.log').write_text(result.stderr)
    if result.returncode:
        raise ValueError('Frozen 30um survival failed: ' + result.stderr)
    env = read(out / 'environment.json')
    if env != {'R': '4.3.2', 'survival': '3.5.7', 'jsonlite': '2.0.0'}:
        raise ValueError('R package environment differs')

def fixed_roi(run):
    import numpy as np
    import pandas as pd
    tree = Path(run) / 'tree'; story = tree / STORY
    module = load(tree / A / 'make_all_cd8_spatial_figure.py')
    if module.SAMPLE_ID != '22RD' or tuple(module.ROI) != (600, 700, 900, 995):
        raise ValueError('Historical ROI contract changed')
    prediction = pd.read_csv(tree / A / 'outer_patient_predictions.csv.gz')
    mold = pd.read_csv(story / 'inputs/cohort_package/moldoveanu_imc/locked_metadata.csv.gz')
    frame = prediction.loc[prediction.patient.eq(module.SAMPLE_ID)].merge(
        mold[['cell_uid', 'Location_Center_X', 'Location_Center_Y']],
        left_on='cell_uid_ordered', right_on='cell_uid', how='left', validate='one_to_one', sort=False)
    x0, y0, x1, y1 = module.ROI
    inside = frame.Location_Center_X.between(x0, x1) & frame.Location_Center_Y.between(y0, y1)
    shown = frame.loc[inside]; target = shown.target_30um.to_numpy(bool)
    answer = {'sample_id': module.SAMPLE_ID, 'roi_bounds_xyxy': list(module.ROI),
              'n_all_cd8_whole': len(frame), 'n_ground_truth_whole': int(frame.target_30um.sum()),
              'n_all_cd8_shown': len(shown), 'n_tcae_shown': int(shown.is_tcae.sum()),
              'n_tcnaive_shown': int(shown.is_tcnaive.sum()), 'n_ground_truth_shown': int(target.sum())}
    for rule in ('top15', 'top20'):
        called = shown['called_' + rule].to_numpy(bool)
        tp = int((target & called).sum()); fp = int((~target & called).sum()); fn = int((target & ~called).sum())
        answer[rule] = {'n_called_whole': int(frame['called_' + rule].sum()), 'n_called_shown': int(called.sum()),
                        'tp_shown': tp, 'fp_shown': fp, 'fn_shown': fn}
    answer['precision_shown'] = answer['top20']['tp_shown']/answer['top20']['n_called_shown']
    answer['recall_shown'] = answer['top20']['tp_shown']/answer['n_ground_truth_shown']
    write(story / 'analysis/fixed_roi.json', answer)
