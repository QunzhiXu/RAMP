"""Evaluate frozen permanent-holdout predictions; plot 14 marker dots per metric."""
from runtime import BASE, STORY, save, sha, now
from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

sys.path.insert(0, str(STORY / 'prediction'))
import masked_marker_model as model_def

OUT = BASE / 'validation_subpanels/all_markers_four_metrics'
OUT.mkdir(parents=True, exist_ok=True)
MANIFEST = BASE / 'validation_subpanels/all_marker_validation/COMPLETION.json'
RAW = STORY / 'sampled_data/bm_codex/source_input_15.csv.gz'
ID = ['cell_id', 'dataset', 'donor', 'unit_key']
MARKERS = list(model_def.NORMAL_MARKERS)
METRICS = ['spearman', 'pearson', 'auroc', 'average_precision']


def evaluate():
    manifest = json.loads(MANIFEST.read_text())
    input_hash = sha(RAW)
    raw = pd.read_csv(RAW, usecols=ID + MARKERS)
    assert len(raw) == 30000 and raw.unit_key.nunique() == 12
    records, origins, checks = [], {}, {}
    for marker in MARKERS:
        origin = manifest['origins'][marker]
        root = Path(origin['metrics_path']).parent
        prediction_path = root / f'{marker}_prediction.npz'
        assert sha(prediction_path) == origin['prediction_sha256']
        assert sha(root / 'donor_metrics.csv') == origin['metrics_sha256']
        md = pd.read_csv(root / 'codex_prediction_metadata.csv.gz')
        assert raw[ID].astype(str).equals(md[ID].astype(str))
        pred = np.load(prediction_path)['prediction']
        assert pred.shape == (len(raw),) and np.isfinite(pred).all()
        # Same donor/marker-independent normalization as the existing evaluations.
        truth = model_def.ramp.kmeanslow_global(
            raw[[marker]].to_numpy(float), md.unit_key.to_numpy(),
            md.dataset.to_numpy(), [marker])[:, 0]
        previous = pd.read_csv(root / 'donor_metrics.csv').set_index('donor')
        delta_rho, delta_auc = [], []
        for unit, ix in md.groupby('unit_key', sort=True).indices.items():
            kmeans = KMeans(n_clusters=2, n_init=20, max_iter=500,
                random_state=model_def.stable_seed('HIGH', marker, unit)).fit(truth[ix, None])
            high = kmeans.labels_ == int(np.argmax(kmeans.cluster_centers_.ravel()))
            row = {'marker': marker, 'donor': unit, 'n_cells': len(ix),
                'n_high': int(high.sum()), 'high_fraction': float(high.mean()),
                'spearman': float(spearmanr(truth[ix], pred[ix]).statistic),
                'pearson': float(pearsonr(truth[ix], pred[ix]).statistic),
                'auroc': float(roc_auc_score(high, pred[ix])),
                'average_precision': float(average_precision_score(high, pred[ix]))}
            assert np.isfinite([row[m] for m in METRICS]).all()
            records.append(row)
            delta_rho.append(abs(row['spearman'] - previous.loc[unit, 'spearman']))
            delta_auc.append(abs(row['auroc'] - previous.loc[unit, 'high_state_auroc']))
        checks[marker] = {'max_abs_difference_existing_spearman': max(delta_rho),
                          'max_abs_difference_existing_auroc': max(delta_auc)}
        assert max(delta_rho + delta_auc) < 1e-10, (marker, checks[marker])
        origins[marker] = {'prediction_path': str(prediction_path),
                           'prediction_sha256': origin['prediction_sha256']}
        print(f'{marker}: 12 donors evaluated; prior metrics reproduced', flush=True)

    full = pd.DataFrame(records)
    assert len(full) == 168 and not full.duplicated(['marker', 'donor']).any()
    assert full.groupby('marker').donor.nunique().eq(12).all()
    full.to_csv(OUT / 'metrics_by_marker_and_donor.csv', index=False)
    summary = full.groupby('marker')[METRICS + ['high_fraction']].median().loc[MARKERS]
    summary.insert(0, 'n_donors', 12)
    summary.to_csv(OUT / 'marker_median_metrics.csv')
    assert sha(RAW) == input_hash
    save(OUT / 'EVALUATION.json', {'time_utc': now(),
        'raw_input': str(RAW), 'raw_sha256': input_hash,
        'manifest_sha256': sha(MANIFEST), 'origins': origins,
        'n_markers': 14, 'n_donors': 12, 'n_codex_cells': 30000,
        'protocol': 'Reuse fixed permanent-marker-holdout NN predictions; no retraining',
        'truth_scale': 'KMEANSLOW, independent normalization per donor and marker',
        'high_state': 'Higher-center cluster from within-donor measured-expression KMeans (k=2); same rule as previous evaluation',
        'AP_definition': 'sklearn average_precision_score, non-interpolated precision-recall summary; not trapezoidal PR area',
        'aggregation': 'For each metric, median of 12 donor scores per marker; one point per marker',
        'donor_equal_weight': True, 'markers_excluded': [],
        'existing_metric_reproduction': checks,
        'aggregation_note': 'Old AUROC summaries used donor means. This new figure consistently uses donor medians for all four metrics.',
        'input_unchanged': True})
    return summary


def plot(summary):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'svg.fonttype': 'none', 'pdf.fonttype': 42, 'text.color': '#28343E',
        'axes.labelcolor': '#45515B', 'xtick.color': '#45515B', 'ytick.color': '#65717A'})
    colors = ['#397CAB', '#5A9993', '#A27BA9', '#CF9856']
    values = [summary[m].to_numpy() for m in METRICS]
    fig, ax = plt.subplots(figsize=(5.5, 4.05))
    fig.subplots_adjust(left=.14, right=.97, bottom=.21, top=.83)
    fig.text(.065, .958, 'Prediction accuracy', fontsize=15, weight='bold', va='top')
    fig.text(.065, .886, '14 held-out markers', fontsize=9.5, color='#73808A')
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
        plotted += [{'marker': marker, 'metric': metric, 'x': float(i + j), 'y': float(v)}
                    for marker, j, v in zip(MARKERS, jitter, y)]
    low = min(float(v.min()) for v in values)
    ymin = -.025 if low >= 0 else np.floor((low - .04) * 5) / 5
    ax.set(xlim=(-.55, 3.55), ylim=(ymin, 1.025), ylabel='Score')
    ax.set_xticks(range(4), ['Spearman ρ', 'Pearson r', 'AUROC', 'AUPRC\n(AP)'], fontsize=10)
    ax.set_yticks(np.arange(0 if low >= 0 else np.ceil(ymin * 5) / 5, 1.01, .2))
    ax.tick_params(axis='x', length=0, pad=9)
    ax.tick_params(axis='y', length=0, pad=6)
    ax.yaxis.grid(True, color='#E8EBEE', linewidth=.7)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set(color='#B9C2C9', linewidth=.7)
    ax.set_ylabel('Score', labelpad=8, fontsize=10)
    fig.text(.14, .032, 'Each point: one marker · median across 12 CODEX donors',
             fontsize=8.3, color='#6B7780')
    for extension in ['png', 'svg']:
        fig.savefig(OUT / f'all_marker_four_metric_boxplot.{extension}', dpi=300, facecolor='white')
    plt.close(fig)
    pd.DataFrame(plotted).to_csv(OUT / 'plotted_points.csv', index=False)
    save(OUT / 'PLOT_AUDIT.json', {'metric_order': METRICS, 'marker_order_in_data': MARKERS,
        'n_points_per_metric': 14, 'all_points_shown': True, 'marker_names_on_plot': False,
        'box': 'median, 25th and 75th percentiles; whiskers within 1.5 IQR',
        'fliers': 'All marker points overlaid; separate duplicate flier glyphs suppressed',
        'y_limits': [ymin, 1.025], 'metric_ranges': {m: [float(summary[m].min()), float(summary[m].max())] for m in METRICS},
        'jitter_seed': 42, 'pairwise_inference': 'none; descriptive marker distributions',
        'summary_sha256': sha(OUT / 'marker_median_metrics.csv')})


def documentation():
    (OUT / 'README_ZH.md').write_text('''# 全 marker 预测准确性：四指标箱线图

主图为 `all_marker_four_metric_boxplot.png`，可编辑矢量版为同名 SVG。

横轴依次为 Spearman ρ、Pearson r、AUROC、AUPRC（以 average precision，AP，计算）。每列 14 个点，每点代表一个 marker 的 12 位 CODEX donor 内指标中位数。全部 14 个 marker 均保留，图中省略名称，名称及数值保留在 `marker_median_metrics.csv`。箱体为四分位距，中线为中位数，须延伸至 1.5 IQR 内最远观测值；所有点（包括离群点）都叠加显示。

Spearman 与 Pearson 在 KMEANSLOW 标准化的实测值和预测值之间计算。AUROC 与 AP 使用实测表达在各 donor 内进行两类 KMeans 后确定的高表达状态为阳性标签，预测的连续表达值为分数。这是高表达状态识别，未使用细胞类型标签。AP 是非插值的 precision-recall 汇总，与梯形积分 PR 面积有区别；其参考水平随高表达状态比例变化，比例已保存到结果表。

直接读取已完成的永久 marker 留出神经网络预测，没有重新训练或选择模型参数。每个待验证 marker 均从所有平台预测输入以及 CODEX 训练目标中移除，参考平台保留该 marker 的训练目标。每个 marker 评估相同的 30,000 个 CODEX 细胞（12 位 donor，每位 2,500 个）。现有 Spearman 和 AUROC 的逐 donor 结果已复算核对一致。

此次四列均取 donor 中位数；旧图 AUROC 汇总采用 donor 均值，因此新图 AUROC 的聚合口径不同。旧文件保持原样。

复现：运行 `Exploration0909/scripts/draw_four_metric_accuracy.py`。脚本采用 runtime 写入保护，所有输出仅在 Exploration0909 内。

## 英文图注

Prediction accuracy across 14 held-out markers. Spearman correlation, Pearson correlation, AUROC and average precision (AP; a non-interpolated precision–recall summary) were computed within each of 12 CODEX donors using frozen permanent-marker-holdout neural network predictions. Each dot represents one marker, summarized by the median across donors. Boxes indicate the interquartile range, center lines indicate medians, and whiskers extend to the most extreme observations within 1.5 times the interquartile range. All 14 marker-level points are shown. Correlations compare measured and predicted KMEANSLOW-normalized expression. AUROC and AP assess discrimination of the measured high-expression state, defined independently within each donor and marker by two-cluster KMeans. Each donor contributes 2,500 cells. No cell-type labels were used in these metrics.
''')


if __name__ == '__main__':
    summary = evaluate()
    plot(summary)
    documentation()
    print(summary.to_string(), flush=True)
    print(OUT / 'all_marker_four_metric_boxplot.png', flush=True)
