"""Read frozen v0.5 outputs and produce a descriptive result review; no refitting."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SUMMARY = Path((ROOT / 'results/formal_v0_5/LATEST_SUMMARY.txt').read_text().strip())
METHODS = ['unaligned', 'ramp', 'clipped_zscore', 'percentile', 'minmax',
           'harmony_direct', 'cycombine', 'cytonorm', 'fastmnn', 'maxfuse']
LABELS = dict(zip(METHODS, ['Unaligned', 'RAMP', 'Clipped z-score', 'Percentile', 'Min-max',
                          'Harmony direct', 'cyCombine', 'CytoNorm', 'fastMNN', 'MaxFuse']))
COLORS = dict(zip(METHODS, ['#656565', '#007C78', '#A07617', '#8557A4', '#AB7474',
                          '#A4A44D', '#4C8DAE', '#C77638', '#5266AB', '#AF5580']))
FAMILIES = ['linear', 'knn', 'mlp']
report = json.loads((SUMMARY / 'REPORT.json').read_text())
assert report['status'] == 'complete' and report['passed'] == report['jobs'] == 53320
means = pd.read_json(SUMMARY / 'primary_means.json')
metrics = pd.read_csv(SUMMARY / 'metrics.csv', low_memory=False).fillna({'marker': '', 'arm': ''})
contrasts = pd.read_json(SUMMARY / 'cohort_pretraining_contrasts.json')
suite = json.loads((ROOT / 'config/formal_v0_5/SUITE.json').read_text())
jobs = {}
for shard in suite['shards']:
    for job in json.loads(Path(shard['config']).read_text())['jobs']:
        assert job['id'] not in jobs
        jobs[job['id']] = job
metrics['scopes'] = metrics.job_id.map(lambda x: jobs[x]['analysis_scopes'])


def aggregate(frame, groups):
    """Same nesting as the frozen exporter: repeats, markers, directions, tasks."""
    groups = list(groups)
    x = frame.groupby(groups + ['task', 'direction', 'marker'], dropna=False).score.mean()
    x = x.groupby(groups + ['task', 'direction'], dropna=False).mean()
    x = x.groupby(groups + ['task'], dropna=False).mean()
    return x.groupby(groups, dropna=False).mean().reset_index()


def primary_table(kind, metric):
    a = means[(means.task == 'ALL') & (means.task_type == kind) & (means.metric == metric)]
    return a.pivot(index='method', columns='family', values='score').reindex(METHODS)[FAMILIES]


def md_table(frame, digits=4):
    f = frame.copy()
    cols = [str(c) for c in f.columns]
    def fmt(x):
        if pd.isna(x): return '—'
        if isinstance(x, (float, np.floating)): return f'{x:.{digits}f}'
        return str(x).replace('|', '/')
    rows = ['| ' + ' | '.join(cols) + ' |', '| ' + ' | '.join(['---'] * len(cols)) + ' |']
    rows += ['| ' + ' | '.join(fmt(x) for x in row) + ' |' for row in f.itertuples(index=False, name=None)]
    return '\n'.join(rows)


def named(frame):
    x = frame.rename(index=LABELS).copy()
    return x.rename_axis('Method').reset_index()


# Independently confirm that the primary tables follow the frozen hierarchy.
for kind, metric in [('J1', 'macro_f1'), ('J2', 'effective_spearman')]:
    a = metrics[(metrics.task_type == kind) & (metrics.metric == metric)
                & metrics.scopes.map(lambda x: 'primary' in x)]
    actual = aggregate(a, ['method', 'family'])
    expected = means[(means.task == 'ALL') & (means.task_type == kind) & (means.metric == metric)]
    check = actual.merge(expected, on=['method', 'family'], suffixes=('_actual', '_expected'))
    assert len(check) == 30 and np.allclose(check.score_actual, check.score_expected, atol=1e-12)

j1 = primary_table('J1', 'macro_f1')
j2 = primary_table('J2', 'effective_spearman')
j2_mae = primary_table('J2', 'mae')
endpoints = {}
for kind in ['J3', 'J4']:
    x = metrics[(metrics.task_type == kind) & (metrics.metric == 'macro_f1')]
    endpoints[kind] = aggregate(x, ['method', 'family', 'arm'])
    endpoints[kind].to_csv(OUT / f'{kind}_endpoints.csv', index=False)
j3 = endpoints['J3'].query("arm == 'A+B'").pivot(index='method', columns='family', values='score').reindex(METHODS)[FAMILIES]
j3_mlp = endpoints['J3'].query("family == 'mlp'").pivot(index='method', columns='arm', values='score').reindex(METHODS)
j3_mlp['gain_vs_mean_single'] = j3_mlp['A+B'] - (j3_mlp['A-only'] + j3_mlp['B-only']) / 2
j4 = endpoints['J4'].pivot(index='method', columns='arm', values='score').reindex([m for m in METHODS if m not in ['fastmnn', 'maxfuse']])
j4['gain_A+B_vs_no_pretrain'] = j4['A+B_finetuned'] - j4.random_supervised

curves = {}
curve_support = []
for kind, metric in [('J1', 'macro_f1'), ('J2', 'effective_spearman')]:
    for scope, resource, count in [('learning', 'fit_budget', 4), ('capacity', 'width', 3)]:
        a = metrics[(metrics.task_type == kind) & (metrics.metric == metric)
                    & metrics.scopes.map(lambda x: scope in x)].copy()
        keys = ['method', 'task', 'direction', 'marker', resource]
        points = a.groupby(keys).agg(score=('score', 'mean'), n=('score', 'count'),
                                    seeds=('seed', 'nunique'), parameters=('parameters', 'mean')).reset_index()
        points = points[(points.n == 3) & (points.seeds == 3)]
        unit = ['method', 'task', 'direction', 'marker']
        complete = points.groupby(unit)[resource].nunique()
        keep = complete[complete == count].reset_index()[unit]
        points = points.merge(keep, on=unit)
        supports = {method: set(map(tuple, g[['task', 'direction', 'marker']].drop_duplicates().to_numpy()))
                    for method, g in points.groupby('method')}
        assert len(supports) == 10 and all(v == supports['ramp'] for v in supports.values())
        avg = aggregate(points, ['method', resource])
        curves[(kind, scope)] = avg.pivot(index=resource, columns='method', values='score').reindex(columns=METHODS)
        curves[(kind, scope)].to_csv(OUT / f'{kind}_{scope}_common_support.csv')
        curve_support.append(dict(task_type=kind, scope=scope, tasks=points.task.nunique(),
                                  task_direction_markers=len(supports['ramp']), complete_seeds=3))

# Donor comparisons use the same directions at BOTH budgets, never shifting cohorts.
donor_points = metrics[(metrics.task_type == 'J1') & (metrics.metric == 'macro_f1')
                      & metrics.scopes.map(lambda x: 'donors' in x)]
donor_points = donor_points.groupby(['method', 'task', 'direction', 'labelled_patient_count']).agg(
    score=('score', 'mean'), n=('score', 'count')).reset_index().query('n == 3')
donor_rows = []
for lo, hi in [(2, 4), (4, 8), (8, 16), (16, 32)]:
    p = donor_points[donor_points.labelled_patient_count.isin([lo, hi])].pivot(
        index=['method', 'task', 'direction'], columns='labelled_patient_count', values='score').dropna().reset_index()
    p['gain'] = p[hi] - p[lo]
    for method, g in p.groupby('method'):
        donor_rows.append(dict(method=method, lower_donors=lo, higher_donors=hi,
                               tasks=g.task.nunique(), directions=len(g),
                               gain=g.groupby('task').gain.mean().mean()))
pd.DataFrame(donor_rows).to_csv(OUT / 'donor_matched_comparisons.csv', index=False)

by_task = {}
for kind, metric in [('J1', 'macro_f1'), ('J2', 'effective_spearman')]:
    a = means[(means.task != 'ALL') & (means.task_type == kind) & (means.metric == metric) & (means.family == 'mlp')]
    p = a.pivot(index='task', columns='method', values='score')
    p['RAMP_rank'] = p.rank(axis=1, ascending=False)['ramp'].astype(int)
    p.to_csv(OUT / f'{kind}_MLP_by_task.csv')
    by_task[kind] = p[['unaligned', 'ramp', 'fastmnn', 'clipped_zscore', 'percentile', 'RAMP_rank']].rename(columns=LABELS)

cost = []
for p in (ROOT / 'representations/formal_v0_5').glob('*/*/*/COMPLETE.json'):
    d = json.loads(p.read_text())
    cost.append(dict(task=p.parents[2].name, marker=p.parents[1].name, method=p.parent.name,
                     seconds=d['elapsed_seconds']))
cost = pd.DataFrame(cost).groupby('method').agg(fits=('seconds', 'size'), seconds=('seconds', 'sum'))
cost['worker_hours'] = cost.seconds / 3600
cost.reindex(METHODS).to_csv(OUT / 'cold_alignment_cost.csv')

parts = [
    '**RAMP cross-platform trainability：正式实验结果解读（v0.5，2026-09-21）**',
    '53,320 项任务全部完成，0 失败、0 未运行；70 个方法×数据任务批次，3 个随机重复。以下只读取冻结产物，不重新拟合或修改评价范围。',
    '**核心判断**：证据最强的是跨平台细胞类型迁移、有限标注与小模型下的表现，以及固定总监督预算下多来源训练的最终成绩。缺失 marker 预测中 RAMP 总体第二，percentile 标准化第一。小型预训练的额外增益接近零，不能据此声称 RAMP 已证明能够显著提高 foundation-model 预训练收益。',
    '**1. 比较范围和指标**',
    'J1 覆盖 7 个冻结数据任务、34 个有向平台对；固定预算主比较为 fit=500、source validation=512。J2 覆盖 6 个合格数据任务的全部 79 个 task×marker，逐个从所有平台的输入中隐藏该 marker，然后重新对齐，再由源平台训练预测目标平台同一 marker；额外包含源 calibration=64 cells/unit。Ptacek 上游处理依赖未解决，因此原 melanoma benchmark 预先排除于严格 J2。',
    'J3 覆盖 5 个三队列数据任务中的 13 个合格目标设置。A-only、B-only、A+B 的总 fit/validation 细胞数与逐类数量分别匹配。J4 覆盖 5 个三队列任务、8 个保留 marker 轴的方法；固定小型 encoder、相同监督更新预算，对比无预训练和预训练后微调。fastMNN/MaxFuse 的 embedding 在 J1/J2/J3 中正常比较，在 J4 的 marker 重建任务中按预先设计不纳入。',
    '主均值按重复→方向内 marker（J2）→任务内方向→任务等权汇总，不把 53,320 项当作独立样本。J1/J3/J4 用患者等权混淆矩阵上的 macro-F1，越高越好。J2 用患者/unit 内 Spearman 的等权汇总，越高越好；常数预测按冻结规则记 effective rho=0。J2 MAE 针对共同 0–1 percentile 目标，越低越好，不是原始荧光强度误差。',
    '全部流程的无标签对齐集合 U 包括目标表达；目标标签和目标被隐藏 marker 不用于训练/调参。因此这是固定 transductive 协议下的结果，不是完全未见目标数据的 inductive 实验。7 个任务有部分来源重叠，并不等于 7 项独立研究。',
    '**2. J1：跨平台细胞类型迁移**',
    md_table(named(j1)),
    'RAMP 在线性模型和 MLP 上总体第一；kNN 的 fastMNN=0.5208、RAMP=0.5170，RAMP 第二。MLP 相对未对齐基线从 0.3795 提高至 0.5289（+0.1494，即 +14.94 个百分点），相对最强对照 fastMNN 为 +0.0152。不能表述为所有模型、所有数据任务或所有分类指标均第一：例如 accuracy 在线性与 kNN 下由 fastMNN 领先。',
    '下表是固定 MLP 的各数据任务结果，排名基于完整 10 方法：',
    md_table(by_task['J1'].rename_axis('Task').reset_index()),
    '骨髓最明显；肺 benchmark 也领先。原 melanoma benchmark、HNSCC 和 melanoma story 中 fastMNN 的 MLP 分数更高；Lung story 相对未对齐增益仅约 0.008。因此整体提升不能替代逐任务解释。',
    '**3. J2：隐藏 marker 后跨平台预测**',
    md_table(named(j2)),
    'RAMP 在三类模型的总体 Spearman 均为第二，percentile 均第一。MLP：未对齐 0.2821 → RAMP 0.3594；percentile=0.3629，差 0.0035。线性和 kNN 的 RAMP 与 percentile 差距分别约 0.0147、0.0122。RAMP 在该任务的描述性平均分明显高于 fastMNN/MaxFuse，但这不能说明 embedding 方法在其他任务中普遍无效。',
    '共同 percentile 目标的 MAE：',
    md_table(named(j2_mae)),
    'MLP MAE：未对齐 0.3524 → RAMP 0.2255；percentile=0.2221。预测对象本身是相对表达/percentile；percentile 输入的优势可能与这个目标有关，但这一解释尚未单独验证。当前结果不能证明恢复了跨平台绝对蛋白浓度。',
    md_table(by_task['J2'].rename_axis('Task').reset_index()),
    'MLP 下 RAMP 在骨髓与 melanoma story 第一，其余 4 个任务第二；这两个第一也不是所有模型均成立（例如 melanoma story 的线性模型为第六）。',
    '**4. 标注量和模型容量**',
    'J1 学习曲线：固定 MLP width=32；34 个方向和全部 3 个重复支持一致。下表只列 fit 细胞数；每个点还使用 512 个源验证细胞。',
    md_table(curves[('J1', 'learning')].rename(columns=LABELS).reset_index()),
    'RAMP fit=100 时 macro-F1=0.4796，未对齐 fit=2000 时仅 0.3807。对应总监督数是 612 对 2512，比例约 4.1，而不是总标注量降低 20 倍。RAMP 在全部 4 个测试预算的总体均值均第一；这是冻结任务上的描述性曲线，不是任意数据均可省同样标注量的保证。',
    'J1 容量曲线：固定 fit=500、validation=512。width 是单隐层 MLP 宽度；同一任务中实际可训练参数数按输入维度/类别数记录。',
    md_table(curves[('J1', 'capacity')].rename(columns=LABELS).reset_index()),
    'RAMP width=8 的 F1=0.5148，高于未对齐 width=128 的 0.3919。RAMP width=8 的实际 head 参数范围为 125–265；width=128 为 1925–4105，范围来自不同任务的输入/类别数。不能把这些参数数当成对齐器或完整 pipeline 的总参数/总成本。',
    'J2 的额外学习/容量曲线每个任务只用预先指定的第一登记数据集→最后登记数据集方向。要求完整 3 seeds 和完整预算网格后，有 78 个 task-direction-marker（主比较为 79 个 task-marker）。不能把该方向子集的 0.3905 与全方向主均值 0.3594 混用。',
    md_table(curves[('J2', 'learning')].rename(columns=LABELS).reset_index()),
    md_table(curves[('J2', 'capacity')].rename(columns=LABELS).reset_index()),
    'J2 的 percentile 在所有测试资源点仍领先；小标注量时优势比固定主预算更大。',
    'Donor 数量曲线不同点可用的数据方向不同，因此不能直接把所有点的总体均值连成“donor 越多越好/越差”的因果曲线。配对共同支持的结果保存在 donor_matched_comparisons.csv。RAMP 2→4、4→8、8→16、16→32 donors 的平均 F1 改变量分别约 +0.0106、+0.0114、+0.0281、+0.0419；对应方向数为 20、18、12、1。最后一点仅一个方向，不作普遍结论。Donor 数也不等于独立 dataset 数。',
    '**5. J3：相同总预算下，多来源训练的最终成绩**',
    'A+B 的 macro-F1：',
    md_table(named(j3)),
    'RAMP 在三种模型的 A+B 最终成绩均第一。MLP：RAMP=0.6712、clipped z-score=0.6559、MaxFuse=0.6341、fastMNN=0.6258、未对齐=0.5625。',
    'MLP 的三个来源条件和相对两个单源平均值的增益：',
    md_table(named(j3_mlp[['A-only', 'B-only', 'A+B', 'gain_vs_mean_single']])),
    'RAMP 的 A-only=0.6497、B-only=0.6449、A+B=0.6712；增加来源的平均增益为 +0.0239。未对齐的平均增益更大（+0.1155），但最终成绩仍低。这支持“对齐后多来源模型最终更好”，不能写成“RAMP 使加入第二来源的边际收益最大”。J3 与 J1 的支持集合和类配额不同，二者原始均值不能直接作前后比较。',
    '**6. J4：预训练收益有限，应与输入处理收益分开**',
    md_table(named(j4[['random_supervised', 'A-only_finetuned', 'A+B_finetuned', 'gain_A+B_vs_no_pretrain']])),
    'RAMP 无预训练端到端监督模型 F1=0.5942；单来源预训练后为 0.5922；双来源预训练后为 0.5982。双来源预训练相对无预训练仅 +0.0041（约 0.41 个百分点）。RAMP 双来源微调的最终分数在 8 方法中第一，但它的优势主要已存在于无预训练输入处理阶段。',
    '作为对照，未对齐无预训练=0.5258、双来源预训练=0.5209（−0.0049）。RAMP 与未对齐的预训练差中差约 +0.0089；相对 cyCombine 的差中差仅约 +0.0004。这些小幅变化不能支持强的 foundation-model 训练收益结论。RAMP 在原 melanoma benchmark 的双来源预训练相对无预训练反而下降约 0.0093。',
    '冻结 encoder 的线性 probe 仅为不同优化协议下的诊断，不能拿它与随机初始化端到端训练的差异当作纯预训练收益。本轮也没有训练或直接比较真正的大型 foundation model。',
    '**7. 不确定性与计算成本**',
    '所有正式任务成功，指标表没有缺失 score。部分任务包含 unit-descriptive 身份或少于两个合格患者，因此完整总体比较没有统一的患者级 95% 区间；三 seeds 的标准差不是患者不确定性。可评价任务的条件性患者 bootstrap 区间已保存于 paired_comparisons.json，不能把完整总体点估计的第一名直接称为统计显著优于所有对照。',
    '例如骨髓 MLP，RAMP−未对齐 F1=+0.2970，条件性患者 bootstrap 95% 区间约 [0.2815,0.3115]；Lung story 的同一差值 +0.0080，区间约 [−0.0077,0.0214]。这些区间条件于固定训练模型/拆分，未重复训练抽样，且不代表新研究之间的不确定性。',
    '以下是缓存记录中的 86 次 cold representation fit 的累计 worker elapsed hours，包含完整面板和逐 marker 隐藏后重拟合，不是并行实验总 wall time，也不含后续 head 训练：',
    md_table(named(cost.reindex(METHODS)[['fits', 'worker_hours']])),
    '**8. 适合写入论文的结论**',
    '推荐主线：RAMP 使不同平台的数据更适合共享预测规则，并在有限标注和小型模型下提高跨平台学习表现；对多来源训练的最终预测性能也有帮助。缺失 marker 预测说明保留 marker 轴的处理具有价值，但简单 percentile 标准化仍是强对照。',
    '不支持的强结论：所有任务/指标全面第一；RAMP 相对所有方法都大幅提高预测；已经证明 RAMP 显著提高大型 foundation-model 预训练；已经建立 transformed biomarker 的绝对真值。',
    '新增 figure 可用：J1 主性能、J2 主性能、完整标注量曲线、完整容量曲线和 J3 终点/增益。J4 若保留，应明确显示无预训练基线和小幅增益，作为探索性结果；无论版面如何安排，完整 J1–J4 结果均应随文可查。',
    '**产物与复现**',
    f'冻结汇总来源：{SUMMARY}',
    '本目录包含 analyze_results.py、RESULTS_REVIEW_ZH.md、results_overview.png/.pdf，以及资源曲线、逐任务结果、J3/J4 endpoints、配对 donor 比较和成本 CSV。运行脚本不改动冻结输入、配置、代码或正式结果。',
]
(OUT / 'RESULTS_REVIEW_ZH.md').write_text('\n\n'.join(parts) + '\n')

# Exploratory overview only, not a replacement for a manuscript figure.
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 12,
                     'axes.labelsize': 10, 'pdf.fonttype': 42, 'ps.fonttype': 42})
fig, axes = plt.subplots(3, 2, figsize=(16, 15), constrained_layout=True)
family_symbols = {'linear': 'o', 'knn': 's', 'mlp': '^'}


def dot_panel(ax, table, title, xlabel):
    order = table.index.tolist()
    y = np.arange(len(order))
    for k, family in enumerate(FAMILIES):
        for i, method in enumerate(order):
            ax.scatter(table.loc[method, family], y[i] + (k - 1) * .19,
                       marker=family_symbols[family], s=33, color=COLORS[method], zorder=3)
    ax.set_yticks(y, [LABELS[x] for x in order]); ax.invert_yaxis()
    ax.set_xlabel(xlabel); ax.set_title(title, loc='left', fontweight='bold')
    ax.set_xlim(0, 1); ax.grid(axis='x', color='#E9E9E9', zorder=0)
    ax.legend(handles=[Line2D([], [], marker=family_symbols[f], color='#555555', linestyle='',
                             label={'linear': 'Linear', 'knn': 'kNN', 'mlp': 'MLP'}[f]) for f in FAMILIES],
              loc='upper right', frameon=False, ncol=1, fontsize=9)
    for tick, method in zip(ax.get_yticklabels(), order):
        if method == 'ramp':tick.set_fontweight('bold');tick.set_color(COLORS['ramp'])


dot_panel(axes[0, 0], j1, 'A  Cross-platform cell-type transfer (7 tasks)', 'Macro-F1 (higher is better)')
dot_panel(axes[0, 1], j2, 'B  Masked-marker prediction (6 tasks)', 'Mean within-patient Spearman (higher is better)')
for ax, scope, title, xlabel in [
    (axes[1, 0], 'learning', 'C  Label budget: fixed MLP width 32', 'Source training cells (+512 validation cells)'),
    (axes[1, 1], 'capacity', 'D  Model capacity: fixed 500 training cells', 'MLP hidden width')]:
    table = curves[('J1', scope)]
    for method in METHODS:
        ax.plot(table.index, table[method], marker='o', ms=4, color=COLORS[method],
                lw=2.5 if method == 'ramp' else 1.1, alpha=1 if method == 'ramp' else .8,
                label=LABELS[method], zorder=4 if method == 'ramp' else 2)
    ax.set_xscale('log');ax.set_xticks(table.index, [str(int(x)) for x in table.index]);ax.minorticks_off()
    ax.set_ylim(0, 1);ax.set_xlabel(xlabel);ax.set_ylabel('Macro-F1');ax.grid(color='#EEEEEE')
    ax.set_title(title, loc='left', fontweight='bold')
axes[1, 0].legend(loc='upper left', ncol=2, frameon=False, fontsize=8.5)
dot_panel(axes[2, 0], j3, 'E  Two-source training, matched label budget (5 tasks)', 'A+B macro-F1 (higher is better)')
ax = axes[2, 1]
for i, method in enumerate(j4.index):
    a = j4.loc[method, 'random_supervised']; b = j4.loc[method, 'A+B_finetuned']
    ax.plot([a, b], [i, i], color=COLORS[method], lw=2)
    ax.scatter(a, i, s=68, facecolor='white', edgecolor=COLORS[method], linewidth=1.4, zorder=3)
    ax.scatter(b, i, s=28, color=COLORS[method], marker='D', zorder=4)
ax.set_yticks(range(len(j4)), [LABELS[x] for x in j4.index]);ax.invert_yaxis();ax.set_xlim(0, 1)
ax.set_xlabel('Macro-F1');ax.set_title('F  Pretraining: small extra gains (5 tasks)', loc='left', fontweight='bold')
ax.grid(axis='x', color='#E9E9E9')
ax.legend(handles=[Line2D([], [], color='#555555', marker='o', markerfacecolor='white', linestyle='', label='No pretraining'),
                   Line2D([], [], color='#555555', marker='D', linestyle='', label='A+B pretraining + fine-tuning')],
          loc='lower right', frameon=False, fontsize=9)
for ax in axes.flat:
    ax.spines[['top', 'right']].set_visible(False)
fig.suptitle('Cross-platform trainability: complete frozen v0.5 experiment\nTask-equal means; all methods shown; no target-based model selection', fontsize=15)
fig.savefig(OUT / 'results_overview.png', dpi=150)
fig.savefig(OUT / 'results_overview.pdf')
plt.close(fig)
provenance = {'source_summary': str(SUMMARY), 'suite_sha256': report['suite_sha256'],
              'source_files_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in [SUMMARY / x for x in ['REPORT.json', 'primary_means.json', 'metrics.csv',
                                               'cohort_pretraining_contrasts.json']]},
              'primary_means_independently_reproduced': True, 'curves': curve_support,
              'training_or_frozen_results_modified': False}
(OUT / 'REVIEW_PROVENANCE.json').write_text(json.dumps(provenance, indent=2) + '\n')
print(json.dumps({'report': str(OUT / 'RESULTS_REVIEW_ZH.md'), 'overview': str(OUT / 'results_overview.png'),
                  'primary_means_verified': True, 'curve_support': curve_support}, indent=2))
