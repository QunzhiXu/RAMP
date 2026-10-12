"""Register accepted five-task J1 update; preserve old release and figure files."""
from pathlib import Path
import hashlib,json,shutil
import pandas as pd
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
REPORTS=ROOT/'reports/seven_benchmark_latest_summary_20260921'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    report=json.loads((OUT/'REPORT.json').read_text())
    old=json.loads((ROOT/'CURRENT_J1_RESULTS.json').read_text()); prev=Path(old['directory'])
    before=OUT/'before_activation';before.mkdir(exist_ok=False)
    shutil.copy2(ROOT/'CURRENT_J1_RESULTS.json',before/'CURRENT_J1_RESULTS.json')
    shutil.copy2(ROOT/'README.md',before/'README.md')
    shutil.copytree(REPORTS,before/'seven_benchmark_summary')
    # Only melanoma and the J1 ALL roll-up may change relative to the accepted release.
    a=pd.read_json(prev/'primary_means.json');b=pd.read_json(OUT/'primary_means.json')
    keys=['task','task_type','family','metric','method']
    unchanged=~(a.task_type.eq('J1') & a.task.isin(['melanoma','ALL']))
    pd.testing.assert_frame_equal(a[unchanged].set_index(keys).sort_index(),b.set_index(keys).loc[a[unchanged].set_index(keys).index].sort_index())
    figure_hashes={str(p):sha(p) for p in (ROOT/'Figure').rglob('*') if p.is_file() and 'archive' not in str(p.relative_to(ROOT/'Figure'))}
    bench=pd.read_csv(OUT/'J1_PER_BENCHMARK.csv');overall=pd.read_csv(OUT/'J1_OVERALL.csv')
    sources={t:str(Path(v['directory'])/('PRIMARY_SUMMARY.csv' if t=='lung_story' else 'FULL_COHORT_SUMMARY.csv')) for t,v in report['sources'].items()}
    for t in ['bone_marrow','lung_cancer']:sources[t]=str(Path(report['original_summary'])/'primary_means.json')
    bench['source']=bench.task.map(sources)
    bench.to_csv(REPORTS/'LATEST_COMPLETED_TRAINING_BY_BENCHMARK.csv',index=False)
    overall.assign(benchmark_count=7).to_csv(REPORTS/'OVERALL_SEVEN_EQUAL_WEIGHT.csv',index=False)
    source_manifest=dict(date='2026-09-21',task='J1 annotation',accepted_release=str(OUT),figure_updated=False,
        sources={p:sha(p) for p in sorted(set(sources.values()))})
    (REPORTS/'SOURCE_MANIFEST.json').write_text(json.dumps(source_manifest,indent=2)+'\n')
    names={'bone_marrow':'Bone marrow','tnbc':'TNBC','melanoma':'Melanoma（原三平台）','lung_cancer':'Lung cancer（原 benchmark）','lung_story':'Lung story','melanoma_story':'Melanoma story','hnscc_story':'HNSCC story'}
    scopes={'bone_marrow':'CD4 T、CD8 T、B、Monocyte','tnbc':'CD4 T、CD8 T、Myeloid、Stroma、Tumor/epithelial','melanoma':'Myeloid、Stroma、Tumor','lung_cancer':'原六类','lung_story':'CD4 T、CD8 T、Myeloid/APC、Tumor/Epithelial','melanoma_story':'B lineage、CD4 T（含 Treg）、CD8 T','hnscc_story':'B、CD4 T、CD8 T'}
    lines=['# 七组 benchmark 当前采用的 J1 结果','','更新：2026-09-21。TNBC、原三平台 Melanoma、Lung story、Melanoma story、HNSCC story 的重训结果已采用。Bone marrow 保持原四类，原 Lung benchmark 保持原六类并正在审查。Figure 3 与 manuscript 等审查完成后统一更新。','','患者等权构造混淆矩阵，再计算指标；方向、seed、模型等权，最后七组等权。10 方法、3 模型、3 seeds，源训练/验证预算 500/512。','','| Benchmark | 监督与评价范围 | RAMP F1（排名） | RAMP BA（排名） |','|---|---|---:|---:|']
    for t in names:
        z=bench[bench.task.eq(t)&bench.method.eq('ramp')].set_index('metric')
        lines.append(f"| {names[t]} | {scopes[t]} | {z.at['macro_f1','score']:.4f}（{z.at['macro_f1','rank']}/10） | {z.at['balanced_accuracy','score']:.4f}（{z.at['balanced_accuracy','rank']}/10） |")
    lines+=['','## 七组等权平均','','| 方法 | Macro-F1 | Balanced accuracy | Accuracy |','|---|---:|---:|---:|']
    pivot=overall.pivot(index='method',columns='metric',values='score').sort_values('macro_f1',ascending=False)
    for m,v in pivot.iterrows():lines.append(f"| {m} | {v.macro_f1:.4f} | {v.balanced_accuracy:.4f} | {v.accuracy:.4f} |")
    lines+=['','RAMP 三项总体均第一，F1 和 BA 均在 6/7 组第一；HNSCC 两项第二。','','## 版本与解释','','- 本次新增原三平台 Melanoma 三类重训：同一批三类目标上，旧分类器→新分类器 F1 为 0.6146→0.6193，BA 为 0.6453→0.6563。原完整四类 F1 为 0.5430。范围变化不能全部归因为重训或对齐改善。','- Melanoma 移出 CD8 监督/评价的背景是抽样训练覆盖不足；CD8 仍是共享类别，其细胞与通道保留在冻结对齐输入。','- TNBC 采用五类重训；同目标旧→新 F1 为 0.6222→0.6411。','- Lung story 采用新版 Senosain 140k 池重抽的原 28 位患者各 1,000 个细胞，Cords/Zheng 不变；84k 对齐细胞，9 markers，保留 C02 主分析。','- Melanoma story 采用 B lineage/CD4（含 Treg）/CD8 三类；HNSCC 采用 full 队列 B/CD4/CD8。敏感性试验不混入主排名。','- 原 Lung 的 0.7007 是旧完整混淆矩阵的四类 F1 平均，尚不是四类重训成绩。FIRST_FOUR_RESCORE_ONLY_NOT_RETRAINED.csv 保留为历史诊断表，不是当前结果入口。','- 后续类别范围调整是探索性的；保留所有完整原结果。J2、标签量/容量曲线与多来源任务仍是原 v0.5 范围。','','## 来源','']
    for t in names:lines.append(f'- {names[t]}：[结果]({sources[t]})')
    lines+=['',f'当前结果登记：[CURRENT_J1_RESULTS.json]({ROOT}/CURRENT_J1_RESULTS.json)。当前发布记录：[README]({OUT}/README.md)。']
    (REPORTS/'README.md').write_text('\n'.join(lines)+'\n')
    readme=f'''# 当前采用的 J1 结果版本\n\n版本 `{OUT.name}`。本次在上一版本的四项更新上，新增原三平台 Melanoma 的 Myeloid/Stroma/Tumor 三类重训。原版本保留在 `{prev.name}`；本次登记前的索引与汇总保存在 `before_activation/`。\n\n共 3,060 个主作业：5 个更新任务 1,980 个，2 个原任务（Bone marrow、原 Lung）1,080 个。七组等权，RAMP macro-F1 0.7219、balanced accuracy 0.7492、accuracy 0.7747，三项第一。\n\n- [七组完整汇总](../../reports/seven_benchmark_latest_summary_20260921/README.md)\n- [Melanoma 详细审查](../../exploration_melanoma_three_classes_20260921/README.md)\n- `primary_means.json`、`J1_PER_BENCHMARK.csv`、`J1_PER_MODEL.csv`、`J1_OVERALL.csv`：当前数表。\n- `UPDATED_J1_JOBS.csv`：1,980 个已采用的更新作业。\n- `REPORT.json`、`ACTIVATION_CHECKS.json`：来源与核验。\n- `build_release.py` 重建数表；`activate_release.py` 登记来源与摘要，不重训。\n\n此次仅更新结果；Figure 3、manuscript、生物学 story 均未改。J2、学习曲线、容量曲线、多来源实验不随 J1 自动更新。探索性范围修订不改称预注册。\n'''
    (OUT/'README.md').write_text(readme)
    ptr=dict(release=OUT.name,directory=str(OUT),report_sha256=sha(OUT/'REPORT.json'),primary_means_sha256=sha(OUT/'primary_means.json'),accepted_updated_tasks=report['updated_tasks'],unchanged_tasks=['bone_marrow','lung_cancer'],figure3_updated=False,manuscript_updated=False,pending_scope='Original Lung benchmark review before figure/manuscript integration',human_summary=str(REPORTS/'README.md'))
    (ROOT/'CURRENT_J1_RESULTS.json').write_text(json.dumps(ptr,indent=2)+'\n')
    text=(ROOT/'README.md').read_text().replace('TNBC 和三个 story','TNBC、原三平台 Melanoma 和三个 story').replace('[四组更新的结果发布记录](releases/j1_scope_update_20260921/README.md)',f'[五组更新的结果发布记录](releases/{OUT.name}/README.md)')
    (ROOT/'README.md').write_text(text)
    exp=ROOT/'exploration_melanoma_three_classes_20260921/README.md'
    text=exp.read_text().replace('当前为探索性候选结果，尚未替换正式结果汇总、Figure 3 或 manuscript，也未改动 Melanoma 生物学 story。',f'经用户确认，已纳入当前 J1 benchmark 结果版本 `{OUT.name}`。Figure 3、manuscript 和 Melanoma 生物学 story 未改动；此次类别调整仍属探索性分析。')
    exp.write_text(text)
    assert all(sha(p)==h for p,h in figure_hashes.items())
    (OUT/'ACTIVATION_CHECKS.json').write_text(json.dumps(dict(previous_release=str(prev),only_melanoma_and_j1_all_numeric_changes=True,figure_files_unchanged=True,figure_file_count=len(figure_hashes),manuscript_not_written=True,source_hashes_verified=True,script_sha256=sha(__file__)),indent=2)+'\n')
    print('Activated',OUT.name,'; figure unchanged:',len(figure_hashes),'files')
if __name__=='__main__':main()
