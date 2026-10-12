"""Read-only benchmark results -> weighted cumulative performance curves."""
from pathlib import Path
import hashlib
import json
import os

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
os.environ.setdefault('MPLCONFIGDIR', str(HERE/'.cache'))
os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

SOURCE = ROOT/'benchmark_results/comparison_evaluation/summary/primary_method_family_metric_ranks.csv'
METHODS = ['RAMP', 'MaxFuse', 'cyCombine', 'fastMNN', 'CytoNorm', 'Harmony-direct']
COLORS = dict(zip(METHODS, ['#008866', '#D09039', '#9772B4', '#397CA8', '#479DAD', '#858B91']))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    before = sha(SOURCE)
    original = pd.read_csv(SOURCE)
    assert len(original) == 4*31*6
    assert not original.duplicated(['benchmark', 'metric', 'method_family']).any()
    excluded = original[original.metric_family.eq('geometry')].copy()
    assert set(excluded.metric) == {'native_knn_jaccard'} and len(excluded) == 4*6
    df = original[original.metric_family.ne('geometry')].copy()
    assert len(df) == 4*30*6 and df.metric.nunique() == 30 and df.metric_family.nunique() == 5
    blocks = ['benchmark', 'metric_family', 'method_family']
    n_blocks = df[['benchmark', 'metric_family']].drop_duplicates().shape[0]
    assert n_blocks == 20
    df['weight'] = 1/(n_blocks*df.groupby(blocks)['metric'].transform('size'))
    quality = np.where(df.direction.eq('higher'), df['mean'], -df['mean'])
    df['quality'] = quality
    group = df.groupby(['benchmark', 'metric'])['quality']
    best, worst = group.transform('max'), group.transform('min')
    df['range_normalized_gap'] = np.divide(best-df.quality, best-worst,
        out=np.zeros(len(df)), where=(best-worst).ne(0))
    assert df.range_normalized_gap.between(0, 1).all()
    assert np.allclose(df.groupby('method_family').weight.sum(), 1)
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 7,
        'axes.titlesize': 8, 'axes.labelsize': 7, 'axes.linewidth': .55,
        'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(180/25.4, 82/25.4), dpi=420)
    fig.subplots_adjust(left=.072, right=.987, bottom=.23, top=.80, wspace=.25)
    fig.suptitle('Overall benchmark performance', x=.025, y=.975,
                 ha='left', fontsize=10, weight='bold')
    fig.text(.985, .965, '4 benchmarks | 30 metrics | 6 methods',
             ha='right', va='top', fontsize=7)
    tables = []
    handles = []
    for method in METHODS:
        g = df[df.method_family.eq(method)]
        lw = 1.8 if method == 'RAMP' else 1.0
        zorder = 8 if method == 'RAMP' else 3
        x = np.arange(1, 7)
        y = np.array([g.loc[g['rank'].le(k), 'weight'].sum() for k in x])
        line, = axes[0].step(x, y, where='post', color=COLORS[method], linewidth=lw,
                            label=method, zorder=zorder)
        axes[0].scatter(x, y, s=8 if method == 'RAMP' else 5, color=COLORS[method], zorder=zorder)
        handles.append(line)
        for k, value in zip(x, y):
            tables.append(dict(method=method, curve='cumulative_rank', threshold=float(k), coverage=float(value)))
        grouped = g.groupby('range_normalized_gap', sort=True).weight.sum()
        xx, yy = grouped.index.to_numpy(), grouped.cumsum().to_numpy()
        # Explicit zero coverage at the origin for methods without a best-score case.
        if xx[0] > 0:
            xx, yy = np.r_[0, xx], np.r_[0, yy]
        axes[1].step(xx, yy, where='post', color=COLORS[method], linewidth=lw, zorder=zorder)
        for threshold, value in zip(xx, yy):
            tables.append(dict(method=method, curve='normalized_gap', threshold=float(threshold), coverage=float(value)))
    axes[0].set(title='A   Cumulative rank coverage', xlabel='Rank threshold k',
                xlim=(.9, 6.1), xticks=range(1, 7), ylabel='Weighted fraction within threshold')
    axes[1].set(title='B   Distance from the best method', xlabel='Gap / observed method range',
                xlim=(-.02, 1.02), xticks=np.arange(0, 1.01, .2), ylabel='Weighted fraction within threshold')
    for ax in axes:
        ax.set_ylim(0, 1.03)
        ax.set_yticks(np.arange(0, 1.01, .25))
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', color='#E5E9EC', linewidth=.5)
        ax.set_axisbelow(True)
        ax.tick_params(length=2.5, width=.5)
        ax.set_title(ax.get_title(), loc='left', weight='bold', pad=7)
        ax.set_title('')
    fig.legend(handles, METHODS, ncol=6, loc='lower center', bbox_to_anchor=(.51, .072),
               frameon=False, handlelength=1.8, columnspacing=1.4, fontsize=7)
    fig.text(.5, .025, 'Geometry excluded; equal weight per benchmark and each of the 5 retained categories.',
             ha='center', fontsize=6.5)
    for ext in ['png', 'svg']:
        fig.savefig(HERE/f'performance_curve_preview.{ext}', dpi=420)
    plt.close(fig)
    df.to_csv(HERE/'source_values_and_weights.csv', index=False)
    excluded.to_csv(HERE/'excluded_geometry_values.csv', index=False)
    pd.DataFrame(tables).to_csv(HERE/'curve_coordinates.csv', index=False)
    # The removed category previously had one sixth of the total weight.
    # Recompute category-balanced ranks, retaining the full-protocol comparison.
    before_ranks = original.groupby(blocks)['rank'].mean().groupby('method_family').mean()
    after_ranks = df.groupby(blocks)['rank'].mean().groupby('method_family').mean()
    ranking = pd.DataFrame({'mean_rank_31_metrics': before_ranks,
                            'mean_rank_30_metrics': after_ranks})
    ranking['position_31_metrics'] = ranking.mean_rank_31_metrics.rank(method='min').astype(int)
    ranking['position_30_metrics'] = ranking.mean_rank_30_metrics.rank(method='min').astype(int)
    ranking.sort_values('position_30_metrics').to_csv(HERE/'overall_ranking_before_after.csv')
    case_rank = df.groupby(blocks)['rank'].mean().groupby(['benchmark', 'method_family']).mean().rename('mean_rank').reset_index()
    case_rank['overall_position'] = case_rank.groupby('benchmark').mean_rank.rank(method='min').astype(int)
    case_rank.to_csv(HERE/'benchmark_overall_ranks_30_metrics.csv', index=False)
    stats = {}
    for method in METHODS:
        g = df[df.method_family.eq(method)]
        stats[method] = {f'top{k}': float(g.loc[g['rank'].le(k), 'weight'].sum()) for k in [1, 2, 3]}
    assert sha(SOURCE) == before
    (HERE/'manifest.json').write_text(json.dumps(dict(status='PASS', source=str(SOURCE),
        source_sha256=before, source_unchanged=True, n_metric_cases_per_method=120,
        n_retained_metrics=30, n_retained_categories=5,
        exclusion=dict(category='geometry', metric='native_knn_jaccard', n_rows=24,
            decision='User-requested Figure 2 scope change after inspection of performance; not a prespecified exclusion.',
            prior_category_weight=1/6, remaining_categories_reweighted_equally=True,
            original_31_metric_results_preserved=True),
        weighting='1 / (20 * number of retained metrics in the benchmark/category)',
        rank_curve='Weighted fraction of cases with existing mean rank <= k; average-rank ties retained.',
        gap_curve='For each benchmark/metric, orient score so higher is better; (best-score)/(best-worst) across six methods. All-equal case -> zero gap.',
        gap_caveat='Normalized by the observed six-method range; 0.1 does not mean a 10% change in the original metric.',
        smoothing='None. Exact step curves; no interpolation or KDE.',
        per_benchmark_outputs_not_replaced=True, statistics=stats), indent=2)+'\n')
    (HERE/'README_ZH.md').write_text(
        '# Figure 2 累计曲线小样\n\n'
        '当前为按用户要求排除 Geometry 的 30 指标版，仅更新新版 Figure 2 的曲线与排名草稿；'
        '尚未替换正式旧版整图或 manuscript。\n\n'
        '本次排除发生在查看结果之后，是展示/评价范围调整，并非预先设定的排除。'
        'Geometry 的 native_knn_jaccard 衡量源数据集内 15-NN 邻域保留，和 marker 的单变量保序不同。'
        '原 31 指标曲线、脚本及表格保存在 versions/all31_before_geometry_exclusion/；'
        '原始 benchmark 评估文件保持只读。排除项数值另存 excluded_geometry_values.csv。\n\n'
        '- A：累计排名曲线。横轴 k，纵轴为名次不超过 k 的加权结果比例。'
        '包括四套 benchmark、30 项指标及六种方法；同一 benchmark/类别内指标等权，'
        '20 个 benchmark × 剩余类别组合等权。原 Geometry 虽只有一项指标，但占旧汇总总权重的 1/6；'
        '移除后重新归一化为五类各占 1/5，并非简单移除 1/31 权重。\n'
        '- B：相对最佳值的差距分布。每个 benchmark/指标按方向统一后，'
        '计算 (最佳得分−方法得分)/(最佳得分−最差得分)。0 为最佳，1 为当前六方法中最差。'
        '例如 0.1 是方法间观测范围的 10%，不是原指标下降 10%。'
        '该尺度依赖当前方法集合；未改动原始得分。\n'
        '- 曲线越靠左上，表示在更严格阈值下覆盖的结果越多；曲线可能交叉。'
        '剩余 30 项指标的所有结果均包含在曲线中，无其他筛选；未平滑、未切掉尾部。\n'
        '- 120 个 benchmark × metric 项相互并非独立试验；曲线用于描述总体分布，'
        '未计算把这些指标当独立样本的显著性检验。\n\n'
        'A 和 B 是同一组结果的两个候选展示，正式整图建议选用其一。'
        '具体指标数值仍应配套保留。源值、权重和曲线坐标均为 CSV；复现入口 build_curve_preview.py。\n\n'
        '总榜调整前后见 overall_ranking_before_after.csv；四套 benchmark 的新综合名次见 '
        'benchmark_overall_ranks_30_metrics.csv。正文或 caption 应写明 30 指标、5 类，排除源邻域 Geometry，'
        '并保留完整 31 指标版本供追溯；不能继续称为原 31 项协议的完整结果。\n')
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
