"""Complete-coverage reporting; retain every fixed weight and source selection."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'src'))
from trainability.core import file_sha256


def md_table(frame):
    columns=list(frame.columns)
    def fmt(x):
        return f'{x:.6f}' if isinstance(x,(float,np.floating)) else str(x)
    rows=['| '+' | '.join(map(str,columns))+' |','| '+' | '.join(['---']*len(columns))+' |']
    rows += ['| '+' | '.join(fmt(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)]
    return '\n'.join(rows)


def main():
    manifest=json.loads((HERE/'MANIFEST.json').read_text())
    for p,h in {**manifest['frozen_scripts'],**manifest['original_code_hashes']}.items():
        assert file_sha256(p)==h,('Changed protected input/code',p)
    rows=[];selection=[];errors=[];verification=[]
    for job in manifest['jobs']:
        assert (HERE/'results'/job['key']/'COMPLETE.json').exists(),f'Incomplete job: {job["key"]}'
        for method in ['ramp','percentile']:
            p=HERE/'results'/job['key']/method
            d=json.loads((p/'RESULT.json').read_text())
            assert file_sha256(p/'predictions.npz')==d['prediction_receipt']['sha256']
            assert file_sha256(p/'SELECTION.json')==d['prediction_receipt']['provenance']['selection_sha256']
            old=ROOT/'results/formal_v0_5'/job['task']/method/'jobs'/job['methods'][method]['id']
            assert file_sha256(old/'ledger.json')==d['ledger_sha256']
            assert file_sha256(old/'RESULT.json')==d['original_result_sha256']
            errors.append(d['original_control_max_abs_error'])
            base={k:job[k] for k in ['task','marker','source','target','seed','key']};base['method']=method
            for i,s in enumerate(d['scores']):
                assert s['status']=='ok' and s['effective_spearman'] is not None
                rows.append(dict(**base,variant=f'lambda_{manifest["lambdas"][i]:g}',
                    spearman=s['effective_spearman'],mae=s['mae']))
            s=d['selected_score']
            rows.append(dict(**base,variant='source_selected',spearman=s['effective_spearman'],mae=s['mae']))
            selection.append(dict(**base,selected_lambda=d['selected_lambda'],
                source_gain=d['candidates'][d['selected_column']]['source_validation_score']-d['candidates'][0]['source_validation_score'],
                target_spearman_gain=s['effective_spearman']-d['scores'][0]['effective_spearman'],
                target_mae_change=s['mae']-d['scores'][0]['mae'],seconds=d['elapsed_seconds']))
            verification.append(dict(key=job['key'],method=method,prediction_sha256=d['prediction_receipt']['sha256'],
                                     mse_control_max_abs_error=d['original_control_max_abs_error']))
    df=pd.DataFrame(rows)
    # Match the frozen J2 averaging order exactly; never pool all jobs or cells.
    marker=df.groupby(['task','source','target','marker','method','variant'])[['spearman','mae']].mean().reset_index()
    directions=marker.groupby(['task','source','target','method','variant'])[['spearman','mae']].mean().reset_index()
    tasks=directions.groupby(['task','method','variant'])[['spearman','mae']].mean().reset_index()
    overall=tasks.groupby(['method','variant'])[['spearman','mae']].mean().reset_index()
    prior=Path((ROOT/'results/formal_v0_5/LATEST_SUMMARY.txt').read_text().strip())
    primary=pd.read_json(prior/'primary_means.json')
    for r in overall[overall.variant=='lambda_0'].itertuples():
        for ourmetric,oldmetric in [('spearman','effective_spearman'),('mae','mae')]:
            q=primary[(primary.task=='ALL')&(primary.task_type=='J2')&(primary.family=='mlp')&(primary.method==r.method)&(primary.metric==oldmetric)]
            assert len(q)==1 and abs(getattr(r,ourmetric)-q.iloc[0].score)<1e-10,(r.method,ourmetric)
    select=pd.DataFrame(selection)
    for name,frame in [('job_metrics',df),('marker_means',marker),('direction_means',directions),('task_means',tasks),('overall_means',overall),('selection',select),('verification',pd.DataFrame(verification))]:
        frame.to_csv(HERE/(name+'.csv'),index=False)
    table=overall.pivot(index='variant',columns='method',values=['spearman','mae'])
    comparisons=pd.DataFrame(dict(variant=table.index,
        ramp_spearman=table[('spearman','ramp')].values,
        percentile_spearman=table[('spearman','percentile')].values,
        ramp_minus_percentile=table[('spearman','ramp')].values-table[('spearman','percentile')].values,
        ramp_mae=table[('mae','ramp')].values,percentile_mae=table[('mae','percentile')].values))
    comparisons.to_csv(HERE/'comparisons.csv',index=False)
    t=tasks[tasks.variant=='source_selected'].pivot(index='task',columns='method',values='spearman')
    t['ramp_minus_percentile']=t.ramp-t.percentile
    t=t.reset_index()
    fixed=tasks[tasks.variant=='lambda_0'].pivot(index='task',columns='method',values='spearman')
    change=t.set_index('task')[['ramp','percentile']]-fixed
    change=change.rename(columns={'ramp':'ramp_change_from_original','percentile':'percentile_change_from_original'})
    t=t.merge(change.reset_index(),on='task')
    freq=select.groupby(['method','selected_lambda']).size().rename('jobs').reset_index()
    chosen=comparisons[comparisons.variant=='source_selected'].iloc[0]
    baseline=comparisons[comparisons.variant=='lambda_0'].iloc[0]
    decision=('RAMP 的总体 Spearman 描述性均值超过 percentile。' if chosen.ramp_minus_percentile>0
              else 'RAMP 的总体 Spearman 描述性均值仍未超过 percentile。')
    lines=['# 缺失 marker 排序训练试验结果','',decision,'',
        f'源选择后的 RAMP Spearman 从 {baseline.ramp_spearman:.6f} 变为 {chosen.ramp_spearman:.6f}；percentile 从 {baseline.percentile_spearman:.6f} 变为 {chosen.percentile_spearman:.6f}。',
        '本次两者均略降；差距缩小并不是 RAMP 自身提高。两个固定正权重也没有使 RAMP 反超，MAE 没有改善。本轮设置不建议替代原预测器。','',
        f'全部 {manifest["n_pairs"]} 组配对任务已完成，即两方法共 {2*manifest["n_pairs"]} 个训练任务，每任务三个损失权重。覆盖六任务、{manifest["task_markers"]} 个 task×marker。',
        f'关闭排序损失的控制组复现了原正式总体分数；所有任务预测最大绝对差为 {max(errors):.3g}。','',
        '## 总体结果','',md_table(comparisons),'',
        '`source_selected` 是预先规定的主读数：每个任务的权重只由源验证 Spearman 选择。固定权重结果全部展示，用于解释；不根据目标成绩从中挑一个替换主结果。MAE 越低越好。','',
        '## 源验证选择后的逐任务结果','',md_table(t),'',
        '## 源验证选择的权重','',md_table(freq),'',
        '## 解释范围','',
        '- RAMP 算法和输入缓存完全未改；没有混入 percentile 特征，没有删除 HNSCC 或表现较差的 marker。',
        '- 网络超参数沿用原源验证选择；本轮只搜索三种损失权重，不代表所有可能的预测器都已尝试。',
        '- 排序仅使用既有 fit 细胞的同 unit 非平局实测顺序。监督预算、划分、初始化、训练上限和评分规则一致。',
        '- MSE 与两个混合损失的模型分别早停；先保存源选择和三个候选预测，再读取目标隐藏 marker 答案评分。',
        '- 先重复均值，再方向内 marker 均值、任务内方向均值、任务等权均值；不把细胞/种子当作独立队列。',
        '- 这是看过原结果后的探索性实验；六个任务也非完全独立来源。没有进行显著性宣称或将其当作外部验证。',
        '- 冻结原脚本、监督账本及原 RESULT 文件哈希已经复核。缓存哈希在每个任务加载时验证。','',
        '## 文件','',
        '- `PLAN.md`：开始运行前冻结的实验设计。',
        '- `MANIFEST.json`：全部任务及代码哈希。',
        '- `results/`：逐任务源选择、模型权重、冻结预测和评分。',
        '- `comparisons.csv`、`task_means.csv`、`marker_means.csv`：不同层级完整成绩。',
        '- `selection.csv`：选择权重和相对原模型变化。',
        '- `verification.csv`：原模型复现误差及预测哈希。','']
    (HERE/'RESULTS_ZH.md').write_text('\n'.join(lines))
    report=dict(complete=True,pairs=manifest['n_pairs'],method_jobs=2*manifest['n_pairs'],task_markers=manifest['task_markers'],
        max_control_prediction_error=max(errors),original_primary_means_reproduced=True,
        source_selected=chosen.to_dict(),baseline=baseline.to_dict(),protected_code_and_results_verified=True)
    (HERE/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2));print(t.to_string(index=False));print(freq.to_string(index=False))


if __name__=='__main__':main()
