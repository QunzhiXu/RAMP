"""Plot one donor per point: arithmetic mean of 14 marker-specific scores."""
from runtime import BASE, save, sha, now
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

SOURCE = BASE / 'validation_subpanels/all_markers_four_metrics/metrics_by_marker_and_donor.csv'
OUT = BASE / 'validation_subpanels/donor_mean_four_metrics'
OUT.mkdir(parents=True, exist_ok=True)
METRICS = ['spearman', 'pearson', 'auroc', 'average_precision']
source_hash = sha(SOURCE)
full = pd.read_csv(SOURCE)
assert len(full) == 168 and not full.duplicated(['donor', 'marker']).any()
assert full.donor.nunique() == 12 and full.marker.nunique() == 14
assert full.groupby('donor').marker.nunique().eq(14).all()
assert full.n_cells.eq(2500).all() and np.isfinite(full[METRICS]).all().all()

# Equal weight per marker, separately for each metric and donor.
summary = full.groupby('donor', sort=True)[METRICS].mean()
summary.insert(0, 'n_markers', 14)
summary.insert(1, 'n_cells', 2500)
summary.to_csv(OUT / 'donor_mean_metrics.csv')
full.to_csv(OUT / 'source_metrics_by_marker_and_donor.csv', index=False)
summary[METRICS].describe().to_csv(OUT / 'donor_distribution_summary.csv')

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
    'svg.fonttype': 'none', 'text.color': '#28343E',
    'axes.labelcolor': '#45515B', 'xtick.color': '#45515B', 'ytick.color': '#65717A'})
colors = ['#397CAB', '#5A9993', '#A27BA9', '#CF9856']
values = [summary[m].to_numpy() for m in METRICS]
fig, ax = plt.subplots(figsize=(5.5, 3.25))
fig.subplots_adjust(left=.14, right=.97, bottom=.205, top=.84)
fig.text(.065, .962, 'Prediction Accuracy (12 donors)', fontsize=14, weight='bold', va='top')
bp = ax.boxplot(values, positions=np.arange(4), widths=.48, patch_artist=True,
    whis=1.5, showfliers=False,
    medianprops={'color': '#2D3942', 'linewidth': 1.7},
    whiskerprops={'linewidth': 1.05}, capprops={'linewidth': 1.05})
for i, color in enumerate(colors):
    bp['boxes'][i].set(facecolor=to_rgba(color, .17), edgecolor=color, linewidth=1.2)
    for item in bp['whiskers'][2*i:2*i+2] + bp['caps'][2*i:2*i+2]:
        item.set_color(color)

rng = np.random.default_rng(42)
plotted = []
for i, (metric, color, y) in enumerate(zip(METRICS, colors, values)):
    jitter = rng.uniform(-.145, .145, len(y))
    ax.scatter(i + jitter, y, s=24, facecolor=color, edgecolor='white',
               linewidth=.55, alpha=.95, zorder=4)
    plotted += [{'donor': donor, 'metric': metric, 'x': float(i + j), 'y': float(v)}
                for donor, j, v in zip(summary.index, jitter, y)]
low = min(float(v.min()) for v in values)
ymin = -.025 if low >= 0 else np.floor((low - .04) * 5) / 5
ax.set(xlim=(-.55, 3.55), ylim=(ymin, 1.025), ylabel='Mean score across markers')
ax.set_xticks(range(4), ['Spearman ρ', 'Pearson r', 'AUROC', 'AUPRC\n(AP)'], fontsize=10, fontweight='bold')
ax.set_yticks(np.arange(0 if low >= 0 else np.ceil(ymin * 5) / 5, 1.01, .2))
ax.tick_params(axis='x', length=0, pad=9)
ax.tick_params(axis='y', length=0, pad=6)
ax.yaxis.grid(True, color='#E8EBEE', linewidth=.7)
ax.set_axisbelow(True)
ax.spines[['top', 'right', 'left']].set_visible(False)
ax.spines['bottom'].set(color='#B9C2C9', linewidth=.7)
ax.set_ylabel('Mean score across markers', labelpad=8, fontsize=10)
for extension in ['png', 'svg']:
    fig.savefig(OUT / f'donor_mean_four_metric_boxplot.{extension}', dpi=300, facecolor='white')
plt.close(fig)
pd.DataFrame(plotted).to_csv(OUT / 'plotted_points.csv', index=False)
assert len(plotted) == 48 and sha(SOURCE) == source_hash
save(OUT / 'AUDIT.json', {'time_utc': now(), 'source': str(SOURCE),
    'source_sha256': source_hash, 'script_sha256': sha(__file__),
    'aggregation': 'Arithmetic mean of 14 marker-specific scores, separately within each donor and metric',
    'marker_weight': 'equal (1/14)', 'point_unit': 'donor',
    'n_donors': 12, 'n_markers_per_donor': 14, 'points_per_metric': 12,
    'metric_order': METRICS, 'donor_order': summary.index.to_list(),
    'markers_excluded': [], 'donors_excluded': [], 'all_points_shown': True,
    'box': 'median, 25th and 75th percentiles across 12 donor means; 1.5 IQR whiskers',
    'fliers': 'All donor points overlaid, including outliers; no duplicate flier glyphs',
    'y_limits': [ymin, 1.025], 'jitter_seed': 42,
    'figure_size_inches': [5.5, 3.25],
    'title': 'Prediction Accuracy (12 donors)',
    'subtitle': None, 'footer': None, 'metric_labels_bold': True,
    'model_refit': False, 'prediction_changed': False, 'source_unchanged': True})
(OUT / 'README_ZH.md').write_text('''# 每位 donor 的平均 marker 预测准确性

主图：`donor_mean_four_metric_boxplot.png`；矢量版：同名 SVG。

图面采用紧凑高度，标题为 Prediction Accuracy (12 donors)，四个横轴指标名称加粗，移除副标题与底部小字。纵轴保留 0–1 的刻度范围；每个点的定义保留在此说明与图注中。

每列对应 Spearman、Pearson、AUROC 或 AUPRC（以 AP 计算），每个点代表一位 CODEX donor，共 12 个点。先在每位 donor 内分别计算 14 个 marker 的单项指标，再对 14 个数值取等权算术平均。因此纵轴是 Mean score across markers。平均的是指标分数，不是先合并或平均 marker 的表达值。

箱体呈现 12 个 donor 均值的四分位距，中线为中位数，须延伸至 1.5 IQR 内最远观测值。全部 donor 点均叠加显示。每位 donor 包含 2,500 个细胞，全部 14 个 marker 均纳入。

读取上一版已经核对过的 168 行 marker × donor 指标，未重新训练或改变预测。数据来自永久 marker 留出验证：待验证 marker 从所有平台输入和 CODEX 训练目标中移除。Spearman 和 Pearson 比较 KMEANSLOW 标准化的实测值与预测值；AUROC 和 AP 评价实测高表达状态识别，阳性标签为每位 donor 每个 marker 的实测表达经两类 KMeans 后中心较高的一类。AP 是非插值 precision–recall 汇总。这里呈现的是 donor 的跨 marker 平均表现；单个 marker 的表现保留在源数据表中。

数据：`donor_mean_metrics.csv` 为每位 donor 的四个平均指标；`source_metrics_by_marker_and_donor.csv` 为完整逐 marker 结果；`plotted_points.csv` 记录所有实际绘制点。

复现脚本：`Exploration0909/scripts/draw_donor_mean_accuracy.py`。所有写入仅在 Exploration0909 内。

## 英文图注

Prediction accuracy across 12 CODEX donors. Each dot represents one donor, with each metric averaged equally over 14 held-out markers. Spearman correlation, Pearson correlation, AUROC and average precision (AP; a non-interpolated precision–recall summary) were first calculated for each marker within each donor using frozen permanent-marker-holdout neural network predictions. The 14 marker-specific scores were then arithmetically averaged for each donor. Boxes indicate the interquartile range across donor means, center lines indicate medians, and whiskers extend to the most extreme observations within 1.5 times the interquartile range. All 12 donor points are shown. Each donor contributes 2,500 cells. AUROC and AP assess discrimination of measured high-expression states defined by two-cluster KMeans within each donor and marker.
''')
print(summary.to_string())
print(OUT / 'donor_mean_four_metric_boxplot.png')
