"""Verify paired supervision, summarize scores and actual curve coverage."""
from run import ROOT, OUT, TASKS, VARIANTS, save, file_sha256
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METRICS=['macro_f1','balanced_accuracy','accuracy']
NAMES={'hnscc_story':'HNSCC story','melanoma_story':'Melanoma story','lung_story':'Lung story'}

def table(frame, digits=4):
    fmt=frame.copy()
    for c in fmt:
        if pd.api.types.is_float_dtype(fmt[c]):fmt[c]=fmt[c].map(lambda x:f'{x:.{digits}f}')
    rows=[list(fmt.columns)]+fmt.astype(str).values.tolist()
    return '\n'.join(['| '+' | '.join(rows[0])+' |',
        '| '+' | '.join(['---']*len(rows[0]))+' |']+['| '+' | '.join(r)+' |' for r in rows[1:]])

def main():
    complete=json.loads((OUT/'RUN_COMPLETE.json').read_text());assert complete['all_passed']
    manifest=json.loads((OUT/'MANIFEST.json').read_text())
    data=pd.concat([pd.read_csv(OUT/f'{n}_SCORES.csv') for n in TASKS],ignore_index=True)
    assert len(data)==180
    originals=pd.read_csv(OUT/'ORIGINAL_FULL_J1_SCORES.csv')
    all_scores=pd.concat([originals[originals.task.isin(TASKS)],data],ignore_index=True)
    all_scores.to_csv(OUT/'J1_ALL_SCORES.csv',index=False)
    summary=all_scores.groupby(['task','variant'])[METRICS].mean().reset_index()
    summary.to_csv(OUT/'J1_SUMMARY.csv',index=False)
    for group in [['task','family','variant'],['task','source','target','variant'],['task','seed','variant']]:
        all_scores.groupby(group)[METRICS].mean().reset_index().to_csv(OUT/('J1_BY_'+group[1].upper()+'.csv'),index=False)
    keys=['task','id','source','target','family','seed']
    paired=data[data.variant.eq(VARIANTS[0])].merge(data[data.variant.eq(VARIANTS[1])],
        on=keys,suffixes=('_old','_new'),validate='one_to_one')
    for m in METRICS:paired[m+'_delta']=paired[m+'_new']-paired[m+'_old']
    paired.to_csv(OUT/'J1_PAIRED_DELTAS.csv',index=False)
    for n in TASKS:
        for j in manifest['tasks'][n]['jobs']:
            ledgers=[json.loads((OUT/'jobs'/n/v/j['id']/'ledger.json').read_text()) for v in VARIANTS]
            assert ledgers[0]==ledgers[1],(n,j['id'])
    original=pd.read_csv(OUT/'ALL_SEVEN_ORIGINAL_COVERAGE.csv')
    original_summary=[]
    for (n,ds),g in original.groupby(['task','dataset']):
        by=g.groupby('unit_key').agg(any_support=('support_passes','any'),all_support=('support_passes','all'),
            changed=('changed_markers','max'),exact_baseline=('exact_baseline','all'))
        original_summary.append(dict(task=n,dataset=ds,donors=len(by),all_references_supported=int(by.all_support.sum()),
            any_reference_supported=int(by.any_support.sum()),donors_changed=int(by.changed.gt(0).sum()),
            donors_exact_baseline=int(by.exact_baseline.sum())))
    original_summary=pd.DataFrame(original_summary)
    original_summary.to_csv(OUT/'ALL_SEVEN_ORIGINAL_COVERAGE_SUMMARY.csv',index=False)
    coverage=[]; profiles=[]; deltas=[]; failed=[]
    for n in TASKS:
        info=manifest['tasks'][n];inp=OUT/'inputs'/n;rep=OUT/'representations'/n
        md=pd.read_csv(inp/'metadata.csv.gz',dtype=str,keep_default_na=False)
        native=np.load(inp/'native.npy');base=np.load(rep/'kmeanslow.npy');new=np.load(rep/'aligned.npy')
        old=np.load(inp/'old_aligned_same_subset.npy')
        audit=pd.read_csv(rep/'profile_audit.csv')
        support=pd.read_csv(inp/'support.csv')
        assert len(audit)==len(support)*len(info['markers'])
        audit['marker']=[info['markers'][int(i)] for i in audit.marker_index]
        audit.insert(0,'task',n);profiles.append(audit)
        assert audit.curve_cells.ge(60).all()
        for row in audit[~audit.curve_profile_fitted].to_dict('records'):
            assert row['curve_unique_bin_centres']<4
            x=native[md.unit_key.eq(row['unit_key']),row['marker_index']]
            vals,counts=np.unique(x,return_counts=True)
            failed.append(dict(**row,donor_cells=len(x),raw_distinct_values=len(vals),
                raw_zero_fraction=float(np.mean(x==0)),raw_most_common_value=float(vals[counts.argmax()]),
                raw_most_common_fraction=float(counts.max()/len(x))))
        for (ds,u),g in md.groupby(['dataset','unit_key']):
            ix=g.index.to_numpy()
            for k,marker in enumerate(info['markers']):
                deltas.append(dict(task=n,dataset=ds,unit_key=u,marker=marker,
                    old_max_change=float(np.abs(old[ix,k]-base[ix,k]).max()),
                    new_max_change=float(np.abs(new[ix,k]-base[ix,k]).max())))
        dd=pd.DataFrame([r for r in deltas if r['task']==n])
        for ds,g in md.groupby('dataset'):
            pa=audit[audit.query_dataset.eq(ds)];sel=dd[dd.dataset.eq(ds)]
            unit=sel.groupby('unit_key')[['old_max_change','new_max_change']].max()
            cov=support[support.dataset.eq(ds)]
            coverage.append(dict(task=n,dataset=ds,donors=len(unit),
                minimum_curve_cells=int(cov.curve_cells.min()),maximum_curve_cells=int(cov.curve_cells.max()),
                old_subset_donors_changed=int(unit.old_max_change.gt(1e-8).sum()),
                new_donors_changed=int(unit.new_max_change.gt(1e-8).sum()),
                all_profiles=len(pa),successful_profiles=int(pa.curve_profile_fitted.sum()),
                failed_profiles=int((~pa.curve_profile_fitted).sum()),
                old_changed_donor_markers=int(sel.old_max_change.gt(1e-8).sum()),
                new_changed_donor_markers=int(sel.new_max_change.gt(1e-8).sum())))
    coverage=pd.DataFrame(coverage);coverage.to_csv(OUT/'REFIT_COVERAGE_SUMMARY.csv',index=False)
    pd.concat(profiles).to_csv(OUT/'REFIT_ALL_PROFILES.csv',index=False)
    pd.DataFrame(deltas).to_csv(OUT/'REFIT_PER_DONOR_MARKER_CHANGES.csv',index=False)
    pd.DataFrame(failed).to_csv(OUT/'REFIT_TIED_PROFILES.csv',index=False)
    failed_stats=pd.DataFrame(failed).groupby('marker').agg(failed_profiles=('unit_key','count'),
        minimum_zero_fraction=('raw_zero_fraction','min'),maximum_zero_fraction=('raw_zero_fraction','max')).reset_index()
    failed_stats.to_csv(OUT/'REFIT_TIED_PROFILE_SUMMARY.csv',index=False)

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titleweight':'bold',
        'axes.labelweight':'bold','pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(2,3,figsize=(9.1,5.2),sharey='row')
    colors=['#a3a3a3','#c88b37','#176c8e']
    for c,n in enumerate(TASKS):
        for r,(metric,label) in enumerate([('macro_f1','Macro-F1'),('balanced_accuracy','Balanced accuracy')]):
            ax=axes[r,c];s=summary[summary.task.eq(n)].set_index('variant')
            variants=['original_full_cohort',*VARIANTS]
            values=[s.at[v,metric] for v in variants]
            ax.bar(range(3),values,width=.65,color=colors,zorder=3)
            for x,y in enumerate(values):ax.text(x,y+.012,f'{y:.3f}',ha='center',va='bottom',fontsize=9,fontweight='bold')
            ax.set_xticks(range(3),['Original\nfull cohort','Old values\nsame subset','Refit\nsame subset'])
            ax.set_ylim(0,.72);ax.spines[['top','right']].set_visible(False)
            ax.grid(axis='y',alpha=.15,zorder=0)
            ax.tick_params(axis='x',length=0)
            if r==0:ax.set_title(NAMES[n])
            if c==0:ax.set_ylabel(label)
    fig.tight_layout(pad=1.2,w_pad=1,h_pad=1.3)
    fig.savefig(OUT/'J1_PAIRED_PERFORMANCE.png',dpi=220,facecolor='white')
    fig.savefig(OUT/'J1_PAIRED_PERFORMANCE.pdf',facecolor='white');plt.close(fig)

    masked=json.loads((OUT/'MASKED_AUDIT_CHECKS.json').read_text())
    comparisons=[]
    for n in TASKS:
        s=summary[summary.task.eq(n)].set_index('variant');row={'Benchmark':NAMES[n]}
        for m in METRICS:
            for v,key in [('original_full_cohort','full'),(VARIANTS[0],'old_subset'),(VARIANTS[1],'refit')]:row[f'{m}_{key}']=s.at[v,m]
            row[m+'_paired_delta']=s.at[VARIANTS[1],m]-s.at[VARIANTS[0],m]
        comparisons.append(row)
    comparisons=pd.DataFrame(comparisons);comparisons.to_csv(OUT/'PERFORMANCE_COMPARISON.csv',index=False)
    checks=dict(all_180_head_jobs_completed=True,all_90_paired_ledgers_identical=True,
        algorithm_and_input_hashes_unchanged=True,all_new_curve_sample_counts_at_least_60=True,
        all_remaining_profile_failures_have_fewer_than_four_unique_bin_centres=True,
        masked_existing_representations_checked=masked['representations'],no_J2_new_performance_claim=True)
    for p,h in {**manifest['protected_files'],**manifest['code_hashes']}.items():assert file_sha256(p)==h,p
    save(OUT/'RESULT_CHECKS.json',checks)
    report='''# Donor-cap 20 的实际结果

算法和参数没有修改。固定使用先前已保存的抽样种子 20260921，没有根据表现更换 donor。HNSCC、Melanoma story、Lung story 三组均完成原 J1 注释设置的试跑：三个预测模型（KNN、线性模型、neural network）、三个种子、所有有向平台组合。两种表示共 180 个训练/评价任务；90 对任务的监督及目标细胞 ledger 完全一致。训练 500 个 source 细胞、source 验证 512 个细胞，按患者等权，沿用原类别和 source-only 超参数选择。

## 性能

`full` 为旧全队列原结果；`old_subset` 为在本次子集上使用旧全队列对齐值重新训练/评价；`refit` 为在相同子集重新运行 RAMP 后训练/评价。只有后两者使用完全相同的细胞与监督拆分，适合衡量本次重新对齐的收益。队列改变的结果单列，不能与旧全队列其他比较方法直接排名。

'''
    report+=table(comparisons[['Benchmark','macro_f1_full','macro_f1_old_subset','macro_f1_refit','macro_f1_paired_delta']])+'\n\n'
    report+=table(comparisons[['Benchmark','balanced_accuracy_full','balanced_accuracy_old_subset','balanced_accuracy_refit','balanced_accuracy_paired_delta']])+'\n\n'
    report+=table(comparisons[['Benchmark','accuracy_full','accuracy_old_subset','accuracy_refit','accuracy_paired_delta']])+'\n\n'
    report+='''同一子集内三组宏 F1 均有所改善，但幅度有限。Melanoma story 新结果仍低于旧全队列的原结果，因此不能说减少 donor 后整体性能必然提高；其子集组成和新的监督抽样本身影响了难度。本次数据只能说明重新对齐相对于同子集旧值有一定收益。

模型分层也有差异：Melanoma story 的 KNN macro-F1 从 0.286 提高到 0.320、线性模型从 0.301 提高到 0.320；neural network 为 0.28827 → 0.28821，基本不变，其 balanced accuracy 从 0.548 降为 0.540。不能说所有模型、所有指标均改善。完整分层数据见 `J1_BY_FAMILY.csv`、`J1_BY_SOURCE.csv` 和 `J1_BY_SEED.csv`。

## 七组原结果覆盖审计

以下单位为 donor。`all_references_supported` 要求所有参考平台方向都有至少 60 个拟合细胞；`any_reference_supported` 只要求至少一个方向满足。`donors_exact_baseline` 是实际逐值比较，意味着整个 donor 的全部 marker 都与 Kmeanslow 初始值完全一致。

'''+table(original_summary)+'\n\n'
    report+='''原来的四组 benchmark 没有发现相同的拟合细胞数量不足问题。三个 story 配置有此问题，HNSCC 最严重；Lung story 较轻，但多个 donor 缺少其中一个参考方向。这里不能把“有足够细胞”等同于每个 marker 都有可用的曲线。

## 重新拟合的实际覆盖

'''+table(coverage)+'\n\n'
    report+='''新三组所有 donor/参考方向都通过了数量门槛，所有保留 donor 都有实际数值变换。HNSCC 和 Lung 的全部预期 marker 曲线成功拟合。

Melanoma IMC 仍有 31/300 个 donor×marker 曲线无法拟合，原因是大量重复表达值导致有效分位数分箱中心少于 4 个，而不是拟合细胞数不足。CITE-seq 120/120 成功。不能通过简单更换 donor 保证消除这种表达分布自身的问题。具体 marker 和原始零值比例见下表及 `REFIT_TIED_PROFILES.csv`。

'''+table(failed_stats)+'\n\n'
    report+=f'''## 对缺失 marker 实验的影响

已检查七组存档，其中 {len(masked['tasks_with_masked_representations'])} 组有已保存的 masked-marker RAMP 表示，合计 {masked['representations']} 个，全部完成核查。原三平台 melanoma 没有对应的已保存 masked 表示，本轮没有为其新建。已有表示的细胞身份、顺序、family、平台列表与对应 complete 表示一致，所以内部数量不足落在同一组 donor 上；并逐值确认没有任何有效参考方向的 donor 在每个 masked 表示中都完全等于初始值。这一覆盖问题也影响旧 J2 输入。

本轮没有重新运行 masked-marker 对齐及 J2 预测，不能据此宣称缺失 marker 的 Spearman 或 MAE 已提高。若继续 J2，需要在固定新子集内，对每个被遮蔽 marker 重新执行对齐，不能复用 complete 表示。

## 结论和范围

这个方案确实在不改算法的情况下恢复了 donor 级曲线覆盖，但目前 J1 的增幅不足以解释或解决全部低分。类别间标记区分度、稀有源类别、跨平台标签定义差异及零值 ties 等仍可能限制表现。没有重跑其他比较方法，不能据此更新方法排名。原有七组正式结果、图和稿件均保持不变。

复现：`python3 run.py --workers 2`；审计：`python3 audit_masked.py`；汇总：`python3 summarize.py`。`MANIFEST.json` 记录固定输入和代码哈希，`RESULT_CHECKS.json` 记录配对及完整性检查。
'''
    (OUT/'RESULTS_ZH.md').write_text(report)
    print(comparisons.to_string(index=False))

if __name__=='__main__':main()
