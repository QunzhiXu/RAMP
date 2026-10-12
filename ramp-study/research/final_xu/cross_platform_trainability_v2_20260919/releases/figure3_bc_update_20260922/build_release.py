"""Publish the completed J1/J2 endpoints without refitting or score selection."""
from pathlib import Path
import hashlib
import json
import shutil
import pandas as pd
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
J1 = ROOT / 'releases/j1_scope_update_20260921_r3'
J2 = ROOT / 'exploration_lung_new140k_j2_20260922'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    checks = json.loads((J2 / 'summary/CHECKS.json').read_text())
    complete = json.loads((J2 / 'COMPLETE.json').read_text())
    status = json.loads((J2 / 'RUN_STATUS.json').read_text())
    assert checks['status'] == 'complete'
    assert checks['prediction_jobs'] == 4860 and checks['representations'] == 90
    assert checks['all_predictions_rescored'] and checks['same_target_and_truth_support_all_methods']
    assert complete['manifest_sha256'] == checks['manifest_sha256'] == sha(J2/'MANIFEST.json')
    assert status['passed_shards'] == status['total_shards'] == 90 and status['failed_shards'] == 0
    for name, expected in checks['summary_files'].items():
        assert sha(J2/'summary'/name) == expected, name
    jobs = pd.read_csv(J2/'summary/SCORES_PER_JOB.csv')
    assert len(jobs) == 4860 and jobs.job_id.is_unique
    assert jobs.groupby(['method','marker']).size().eq(54).all()
    assert jobs.groupby('method').size().eq(486).all()
    assert jobs.target_cells.eq(28000).all()
    assert np.isfinite(jobs[['effective_spearman','mae']].to_numpy()).all()
    for row in jobs.itertuples():
        assert Path(row.result).exists()
        assert (Path(row.result).parent/'JOB_COMPLETE.json').exists()
    new = pd.read_json(J2/'summary/primary_means.json')
    old = pd.read_json(J1/'primary_means.json')
    keys = ['task_type','task','family','metric','method']
    assert not new.duplicated(keys).any()
    keep = lambda d: d[~(d.task_type.eq('J2') & d.task.isin(['lung_story','ALL']))].sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(keep(new), keep(old), atol=1e-14, rtol=0)
    shutil.copy2(J2/'summary/primary_means.json', OUT/'primary_means.json')
    for name in ['J1_PER_BENCHMARK.csv','J1_PER_MODEL.csv','J1_OVERALL.csv']:
        shutil.copy2(J1/name, OUT/name)
    for name in ['J2_PER_BENCHMARK.csv','J2_OVERALL.csv','LUNG_BEFORE_AFTER.csv','OVERALL_BEFORE_AFTER.csv']:
        shutil.copy2(J2/'summary'/name, OUT/name)
    task_rows = new[new.task.ne('ALL')]
    panel = task_rows.groupby(['task_type','task','metric','method'], as_index=False).score.mean()
    panel.to_csv(OUT/'PER_BENCHMARK.csv', index=False)
    means = panel.groupby(['task_type','metric','method'], as_index=False).score.mean()
    means.to_csv(OUT/'OVERALL.csv', index=False)
    report = dict(status='complete', release=OUT.name, j1_primary_jobs=3060,
        new_lung_j2_jobs=4860, new_lung_j2_representations=90,
        primary_j2_jobs=int(new[new.task_type.eq('J2') & new.task.ne('ALL') & new.metric.eq('mae')].rows.sum()),
        all_new_lung_predictions_previously_independently_rescored=True,
        current_summary_hashes_verified=True, all_4860_completion_records_present=True,
        accepted_j1_unchanged=True, other_five_j2_unchanged=True,
        j1_class_scope_revisions_exploratory=True,
        panels_def='Original v0.5 scope and Lung input; preserved, not updated by this release.',
        source_files={str(p):sha(p) for p in [J1/'REPORT.json',J1/'primary_means.json',
            J2/'COMPLETE.json', J2/'summary/CHECKS.json',J2/'summary/primary_means.json',
            J2/'summary/SCORES_PER_JOB.csv']},
        no_models_refitted=True, no_marker_or_method_selection=True)
    save(OUT/'REPORT.json', report)
    save(ROOT/'CURRENT_FIGURE3_RESULTS.json', dict(release=OUT.name,directory=str(OUT),
        primary_means_sha256=sha(OUT/'primary_means.json'),report_sha256=sha(OUT/'REPORT.json'),
        panel_B='Updated seven-task J1; exploratory classification scopes.',
        panel_C='Six-task J2 with new Lung reference; 79 benchmark-marker combinations.',
        panels_DEF='Original v0.5 resource experiments; separately identified in caption and Methods.',
        figure_updated=False,manuscript_updated=False))
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
