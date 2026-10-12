"""Promote six audited J1 updates; original experiments remain immutable."""
from pathlib import Path
import hashlib
import json
import pandas as pd
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
UPDATES = {
    'tnbc': 'exploration_tnbc_five_classes_20260921',
    'melanoma': 'exploration_melanoma_three_classes_20260921',
    'lung_cancer': 'exploration_lung_original_four_classes_20260921',
    'lung_story': 'exploration_lung_new140k_comparison_20260921',
    'melanoma_story': 'exploration_melanoma_story_lymphocytes_20260921',
    'hnscc_story': 'exploration_hnscc_cd4_cd8_b_20260921',
}
METRICS = ['macro_f1', 'balanced_accuracy', 'accuracy']


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    base = Path((ROOT / 'results/formal_v0_5/LATEST_SUMMARY.txt').read_text().strip())
    raw = pd.read_json(base / 'primary_means.json')
    original = raw.copy()
    source_records = {}
    jobs = []
    for task, directory in UPDATES.items():
        folder = ROOT / directory
        checks = json.loads((folder / 'CHECKS.json').read_text())
        assert checks['original_files_unchanged']
        assert json.loads((folder / 'RUN_COMPLETE.json').read_text())['all_passed']
        df = pd.read_csv(folder / 'SCORES_PER_JOB.csv')
        if task == 'lung_story': df = df[df.arm.eq('keep_c02')].copy()
        if task == 'hnscc_story': df = df[df.cohort.eq('full')].copy()
        expected = 540 if task in ['tnbc', 'melanoma', 'lung_cancer', 'lung_story'] else 180
        assert len(df) == expected
        assert df.method.nunique() == 10 and set(df.family) == {'linear', 'knn', 'mlp'}
        assert set(df.seed) == {20260919, 20260920, 20260921}
        assert not df.duplicated(['method', 'family', 'seed', 'source', 'target']).any()
        grouped = df.groupby(['source', 'target', 'family', 'seed'])
        assert grouped.size().eq(10).all()
        assert grouped.supervision_sha256.nunique().eq(1).all()
        assert grouped.target_ids_sha256.nunique().eq(1).all()
        df['task'] = task
        jobs.append(df)
        rows = []
        for (method, family), g in df.groupby(['method', 'family']):
            for metric in METRICS:
                rows.append(dict(task=task, task_type='J1', family=family,
                    metric=metric, method=method, score=g['patient_'+metric].mean(), rows=len(g)))
        raw = pd.concat([raw[~(raw.task.eq(task) & raw.task_type.eq('J1'))], pd.DataFrame(rows)], ignore_index=True)
        summary_name = 'PRIMARY_SUMMARY.csv' if task == 'lung_story' else 'FULL_COHORT_SUMMARY.csv'
        published = pd.read_csv(folder / summary_name).set_index('method')
        for method, g in df.groupby('method'):
            for metric in METRICS:
                assert np.isclose(g['patient_'+metric].mean(), published.loc[method, 'patient_'+metric], atol=1e-12)
        source_records[task] = dict(directory=str(folder), selected_jobs=expected,
            selection='keep_c02' if task == 'lung_story' else 'full cohort',
            sources={str(folder/f): sha(folder/f) for f in ['SCORES_PER_JOB.csv', summary_name, 'CHECKS.json', 'MANIFEST.json', 'RUN_COMPLETE.json']})
    raw = raw[~(raw.task.eq('ALL') & raw.task_type.eq('J1'))].copy()
    j1 = raw[raw.task_type.eq('J1')]
    totals = j1.groupby(['task_type', 'family', 'metric', 'method'], as_index=False).agg(score=('score', 'mean'), rows=('rows', 'sum'))
    totals['task'] = 'ALL'
    raw = pd.concat([raw, totals], ignore_index=True)[original.columns]
    assert len(raw) == len(original)
    keys = ['task', 'task_type', 'family', 'metric', 'method']
    assert not raw.duplicated(keys).any()
    unchanged = ~original.task_type.eq('J1') | (~original.task.isin(list(UPDATES)+['ALL']))
    expected = original[unchanged].set_index(keys).sort_index()
    actual = raw.set_index(keys).loc[expected.index].sort_index()
    pd.testing.assert_frame_equal(actual, expected)
    raw.sort_values(keys).to_json(OUT / 'primary_means.json', orient='records', indent=2, double_precision=15)
    j1.sort_values(keys).to_csv(OUT / 'J1_PER_MODEL.csv', index=False)
    bench = j1.groupby(['task', 'method', 'metric'], as_index=False).score.mean()
    bench['rank'] = bench.groupby(['task', 'metric']).score.rank(ascending=False, method='min').astype(int)
    bench.to_csv(OUT / 'J1_PER_BENCHMARK.csv', index=False)
    overall = bench.groupby(['method', 'metric'], as_index=False).score.mean()
    overall['rank'] = overall.groupby('metric').score.rank(ascending=False, method='min').astype(int)
    overall.to_csv(OUT / 'J1_OVERALL.csv', index=False)
    pd.concat(jobs, ignore_index=True).to_csv(OUT / 'UPDATED_J1_JOBS.csv', index=False)
    report = dict(status='complete', release='j1_scope_update_20260921_r3', primary_annotation_jobs=3060,
        reused_original_j1_jobs=540, promoted_updated_j1_jobs=2520, benchmark_count=7, directions=34,
        methods=10, families=3, seeds=3, updated_tasks=list(UPDATES),
        update_scope='Accepted primary J1 results only. Figure 3 and manuscript remain on the original version pending dependent J2/curve synchronization and figure integration; all seven J1 benchmark reviews are complete.',
        figure_and_manuscript_updated=False,
        publication_status='Deferred at user request; no active figure data-source switch.',
        original_summary=str(base), original_summary_sha256=sha(base/'primary_means.json'),
        original_report_sha256=sha(base/'REPORT.json'), sources=source_records,
        j2_and_unupdated_j1_unchanged=True, paired_methods_verified=True,
        no_experiments_refitted_during_promotion=True, class_revisions_post_hoc=True,
        user_authorized_promotion=True, script_sha256=sha(__file__))
    (OUT / 'REPORT.json').write_text(json.dumps(report, indent=2)+'\n')
    print(overall.pivot(index='method', columns='metric', values='score').sort_values('macro_f1', ascending=False).to_string())


if __name__ == '__main__': main()
