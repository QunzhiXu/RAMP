"""Verify predictions, then replace Lung J2 only in a NEW summary artifact."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from run import OUT, ROOT, METHODS, task_for, save, verify, verify_job
from trainability.core import file_sha256, digest
from trainability.scoring import load_frozen_predictions, regression_score
from trainability.reporting import nested_average
from trainability.runner import eligible_ids


def main():
    manifest = json.loads((OUT / 'MANIFEST.json').read_text())
    verify(manifest)
    mh = file_sha256(OUT / 'MANIFEST.json')
    task = task_for()
    lookup = {c:i for i,c in enumerate(task.pool.cell_ids)}
    md = task.metadata.set_index('cell_id')
    rows, support, coverage = [], {}, []
    for n, job in enumerate(manifest['jobs']):
        dest = OUT / 'jobs' / job['method'] / job['id']
        assert (dest / 'JOB_COMPLETE.json').exists(), f'Missing job: {dest}'
        verify_job(dest, job, mh)
        result = json.loads((dest / 'RESULT.json').read_text())
        ledger = json.loads((dest / 'ledger.json').read_text())
        frozen_plan = json.loads((OUT / job['plan_file']).read_text())
        for k in ['fit', 'validation', 'calibration']:
            assert ledger[k] == frozen_plan[k]
        assert ledger['mask'] == [task.receipt['canonical_markers'][job['marker']]]
        assert ledger['pool_sha256'] == manifest['pool_sha256']
        target_ids = eligible_ids(task, job['target'], False)
        assert ledger['target_cell_ids'] == list(target_ids) and len(target_ids) == 28000
        pred = load_frozen_predictions(result['prediction_receipt'], target_ids)
        truth = task.native[[lookup[c] for c in target_ids], task.markers.index(job['marker'])]
        target_md = md.loc[list(target_ids)]
        score = regression_score(truth, pred, target_md.unit_key.to_numpy(), target_md.patient_id.to_numpy())
        assert score['status'] == 'ok'
        for metric in ['effective_spearman', 'mae', 'mae_rho_support']:
            a, b = score[metric], result['score'][metric]
            assert (a is None and b is None) or (a is not None and b is not None and np.isclose(a,b,atol=1e-12,rtol=0))
        assert score['units'] == result['score']['units']
        key = (job['marker'], job['source'], job['target'], job['seed'], job['family'])
        signature = [(r['unit'], r['patient'], r['cells'], r['effective_rho'] is not None) for r in score['units']]
        assert support.setdefault(key, signature) == signature, 'Method-specific scoring support'
        rows.append(dict(job_id=job['id'], task='lung_story', task_type='J2', method=job['method'],
            marker=job['marker'], source=job['source'], target=job['target'], family=job['family'],
            seed=job['seed'], direction=job['source']+' -> '+job['target'],
            effective_spearman=score['effective_spearman'], mae=score['mae'],
            rho_patients=score['rho_patients'], mae_patients=score['mae_patients'],
            target_cells=len(target_ids), target_ids_sha256=digest(target_ids),
            supervision_sha256=job['supervision_sha256'],
            constant_prediction_units=score['rho_status_counts'].get('constant_prediction',0),
            constant_truth_units=score['rho_status_counts'].get('constant_truth',0),
            result=str(dest/'RESULT.json')))
        coverage.append(dict(job_id=job['id'], status='passed', method=job['method'],
                             marker=job['marker'], prediction_rescored=True))
        if (n+1) % 486 == 0: print(f'RESCORED {n+1}/4860', flush=True)
    assert len(rows) == 4860 and len(support) == 486
    reps = list((OUT / 'representations').glob('*/*/COMPLETE.json'))
    assert len(reps) == 90
    for method in METHODS:
        for marker in task.markers:
            folder = OUT/'representations'/marker/method
            assert json.loads((folder/'MASK_BOUNDARY_AUDIT.json').read_text())['passed']
            proof = json.loads((folder/'COMPLETE.json').read_text())
            assert proof['pool_sha256'] == task.pool.sha256
            for name, key in [('values.npy','values_sha256'), ('cell_ids.npy','cell_ids_sha256')]:
                assert file_sha256(folder/name) == proof[key]
    out = OUT / 'summary'
    out.mkdir(exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out/'SCORES_PER_JOB.csv', index=False)
    pd.DataFrame(coverage).to_csv(out/'COVERAGE.csv', index=False)
    new_means = []
    for (method, family), group in frame.groupby(['method', 'family']):
        for metric in ['effective_spearman', 'mae']:
            eligible = group[group[metric].notna()]
            assert len(eligible), 'No scoreable target units'
            rr = [dict(r, score=r[metric]) for r in eligible.to_dict('records')]
            new_means.append(dict(task='lung_story', task_type='J2', family=family, metric=metric,
                                  method=method, score=float(nested_average(rr)), rows=len(rr)))
    prior = json.loads(Path(manifest['accepted_j1_primary_means']).read_text())
    baseline = json.loads(Path(manifest['original_primary_means']).read_text())
    key = lambda r: tuple(r[k] for k in ['task', 'task_type', 'family', 'metric', 'method'])
    def changed(r): return r['task_type'] == 'J2' and r['task'] in ['lung_story', 'ALL']
    updated = [dict(r) for r in prior if not changed(r)] + new_means
    j2_task_rows = [r for r in updated if r['task_type']=='J2' and r['task']!='ALL']
    tasks = sorted({r['task'] for r in j2_task_rows})
    assert tasks == ['bone_marrow','hnscc_story','lung_cancer','lung_story','melanoma_story','tnbc']
    for method in METHODS:
        for family in ['linear', 'knn', 'mlp']:
            for metric in ['effective_spearman','mae']:
                selected = [r for r in j2_task_rows if r['method']==method and r['family']==family and r['metric']==metric]
                assert len(selected) == 6
                updated.append(dict(task='ALL', task_type='J2', family=family, metric=metric,
                    method=method, score=float(np.mean([r['score'] for r in selected])), rows=sum(r['rows'] for r in selected)))
    assert len(updated) == len(prior) and len({key(r) for r in updated}) == len(updated)
    assert {key(r):r for r in updated if not changed(r)} == {key(r):r for r in prior if not changed(r)}
    old_j2 = {key(r):r for r in baseline if r['task_type']=='J2' and r['task'] not in ['ALL','lung_story']}
    for r in updated:
        if key(r) in old_j2:
            assert np.isclose(r['score'], old_j2[key(r)]['score'], atol=1e-14, rtol=0)
    save(out/'primary_means.json', sorted(updated, key=key))
    pd.DataFrame(new_means).to_csv(out/'LUNG_PER_MODEL.csv', index=False)
    panel = pd.DataFrame(j2_task_rows).groupby(['task','metric','method'], as_index=False).score.mean()
    panel['rank'] = panel.groupby(['task','metric']).score.transform(
        lambda s:s.rank(ascending=s.name[1]=='mae', method='min')).astype(int)
    panel.to_csv(out/'J2_PER_BENCHMARK.csv', index=False)
    overall = panel.groupby(['method','metric'], as_index=False).score.mean()
    overall['rank'] = overall.groupby('metric').score.transform(lambda s:s.rank(ascending=s.name=='mae',method='min')).astype(int)
    overall.to_csv(out/'J2_OVERALL.csv', index=False)
    old_frame = pd.DataFrame(baseline)
    old_panel = old_frame[old_frame.task_type.eq('J2') & ~old_frame.task.eq('ALL')].groupby(['task','metric','method'],as_index=False).score.mean()
    old_overall = old_panel.groupby(['method','metric'],as_index=False).score.mean()
    delta = panel[panel.task.eq('lung_story')].merge(old_panel[old_panel.task.eq('lung_story')],
        on=['task','metric','method'], suffixes=('_new','_old'), validate='one_to_one')
    delta['change'] = delta.score_new-delta.score_old
    delta.to_csv(out/'LUNG_BEFORE_AFTER.csv', index=False)
    total_delta = overall.merge(old_overall,on=['method','metric'],suffixes=('_new','_old'),validate='one_to_one')
    total_delta['change'] = total_delta.score_new-total_delta.score_old
    total_delta.to_csv(out/'OVERALL_BEFORE_AFTER.csv', index=False)
    verify(manifest)
    save(out/'CHECKS.json', dict(status='complete', prediction_jobs=4860, representations=90,
        all_predictions_rescored=True, same_target_and_truth_support_all_methods=True,
        original_source_files_unchanged=True, all_J1_unchanged=True, other_five_J2_unchanged=True,
        original_melanoma_still_excluded=True, manuscript_and_figures_updated=False,
        manifest_sha256=mh, completed_unix=time.time(),
        summary_files={p.name:file_sha256(p) for p in out.iterdir() if p.is_file() and p.suffix in ['.csv','.json'] and p.name!='CHECKS.json'}))
    lines = ['# 新版 Lung story J2 结果', '',
        '4,860/4,860 个主作业、90/90 个隐藏-marker 表示均完成并通过预测复算。', '',
        '本目录总汇总保留最新七组 J1，仅替换 Lung story J2；其余五组 J2 保留，原三平台 Melanoma 仍未纳入 J2。图与稿件尚未改动。', '',
        '## 六组 J2 总体（benchmark 等权、三个模型等权）', '',
        '| 方法 | Spearman | 排名 | MAE | 排名 |', '|---|---:|---:|---:|---:|']
    for method in METHODS:
        g = overall[overall.method.eq(method)].set_index('metric')
        lines.append(f"| {method} | {g.loc['effective_spearman','score']:.4f} | {int(g.loc['effective_spearman','rank'])} | {g.loc['mae','score']:.4f} | {int(g.loc['mae','rank'])} |")
    lines += ['', 'Spearman 越高越好；MAE 越低越好。源校准和目标评分使用原相对尺度；这不是不同平台绝对强度的 MAE。', '',
        '运行边界和源数据限制见 ../DESIGN_ZH.md；逐任务及新旧对比见本目录 CSV。没有根据结果删除 marker、方向或细胞类别。']
    (out/'README_ZH.md').write_text('\n'.join(lines)+'\n')
    print('SUMMARY COMPLETE: new Lung J2; five unchanged J2 tasks; J1 unchanged.', flush=True)


if __name__ == '__main__': main()
