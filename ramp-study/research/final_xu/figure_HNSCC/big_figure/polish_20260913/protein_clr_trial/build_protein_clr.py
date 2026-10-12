"""Plot the original selected CD8 cells using measured CLR protein values."""
from pathlib import Path
import hashlib
import json
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/hnscc_protein_clr_plot')
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import fitz

HERE = Path(__file__).resolve().parent
TRIAL = HERE.parent
ROOT = TRIAL.parent.parent
SOURCE = ROOT / 'program_return_exploration_20260913/output/source_features_and_programs.csv.gz'
GROUPS = TRIAL / 'cell_distribution_trial/group_assignments.csv.gz'
OLD_MEANS = TRIAL / 'cell_distribution_trial/panel_E/donor_means.csv'
LOW, HIGH, INK, GRID = '#2B6CB0', '#C9433C', '#182027', '#E7EBEE'
FEATURES = [('LAG3', 'LAG-3'), ('CD127', 'CD127'), ('CD62L', 'CD62L')]
DONORS = ['126', 'indiv_1', 'indiv_2']


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    before = {str(p): sha(p) for p in [SOURCE, GROUPS, OLD_MEANS]}
    d = pd.read_csv(SOURCE, dtype={'cell_uid': str, 'donor': str})
    groups = pd.read_csv(GROUPS, dtype={'cell_uid': str, 'donor': str})
    d = d.merge(groups[['cell_uid', 'donor', 'group', 'state_rank']],
                on=['cell_uid', 'donor'], validate='one_to_one')
    assert len(d) == 4788 and set(d.donor) == set(DONORS)
    rank = d.groupby('donor').pair.transform(lambda x: (rankdata(x)-.5)/len(x))
    np.testing.assert_allclose(rank, d.state_rank, atol=1e-14, rtol=0)
    expected = np.select([rank <= .30, rank >= .70], ['Lower', 'Higher'], default='Middle')
    np.testing.assert_array_equal(expected, d.group)
    assert d.group.value_counts().to_dict() == {'Middle': 1912, 'Lower': 1439, 'Higher': 1437}
    old = pd.read_csv(OLD_MEANS, dtype={'donor': str})

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none', 'text.color': INK})
    width, height = 4.2, 3.45
    fig = plt.figure(figsize=(width, height), facecolor='white')
    fig.text(.08/width, .976, 'E', fontsize=23, weight='bold', va='top')
    fig.text(.51/width, .947, 'PROTEIN PHENOTYPE', fontsize=12, weight='bold', va='top')
    fig.legend(handles=[
        Line2D([], [], marker='o', linestyle='', markersize=4.2, color=LOW, label='Lower 30%'),
        Line2D([], [], marker='o', linestyle='', markersize=4.2, color=HIGH, label='Higher 30%'),
        Line2D([], [], marker='D', linestyle='', markersize=4, markerfacecolor=INK,
               markeredgecolor='white', markeredgewidth=.45, label='Donor mean')],
        loc='center', bbox_to_anchor=(.55, .839), ncol=3, frameon=False, fontsize=8.6,
        handlelength=.7, handletextpad=.35, columnspacing=.9)
    ax = fig.add_axes([.168, .160, .801, .575])
    points_out, means_out, summaries = [], [], []
    for j, (key, name) in enumerate(FEATURES):
        # Check that this is the exact pre-percentile measurement used previously.
        percentiles = d.groupby('donor')[key+'_measured'].transform(lambda x: 100*(rankdata(x)-.5)/len(x))
        for group, dx, color in [('Lower', -.215, LOW), ('Higher', .215, HIGH)]:
            b = d[d.group.eq(group)]
            values = b[key+'_measured'].to_numpy(float)
            assert len(values) == (1439 if group == 'Lower' else 1437)
            assert np.isfinite(values).all()
            x = j+dx
            vio = ax.violinplot([values], positions=[x], widths=.39, showmeans=False,
                                showextrema=False, points=180, bw_method=.22)
            for body in vio['bodies']:
                body.set(facecolor=color, edgecolor=color, linewidth=.45, alpha=.09, zorder=1)
            jitter = np.random.default_rng(20260913+100*j+(group == 'Higher')).uniform(-.095, .095, len(values))
            dots = ax.scatter(x+jitter, values, s=2, color=color, alpha=.13,
                              linewidths=0, rasterized=True, zorder=2)
            np.testing.assert_array_equal(np.asarray(dots.get_offsets())[:, 1], values)
            ax.boxplot([values], positions=[x], widths=.14, patch_artist=True, manage_ticks=False,
                       showfliers=False, whis=1.5,
                       boxprops=dict(facecolor='white', edgecolor=color, linewidth=.9, alpha=.85),
                       medianprops=dict(color=color, linewidth=1.2),
                       whiskerprops=dict(color=color, linewidth=.8),
                       capprops=dict(color=color, linewidth=.8), zorder=3)
            means = b.groupby('donor')[key+'_measured'].mean().loc[DONORS]
            ax.scatter(x+np.array([-.037, 0, .037]), means.to_numpy(), s=18, marker='D',
                       facecolor=INK, edgecolor='white', linewidth=.5, zorder=5)
            for donor, mean in means.items():
                idx = b.index[b.donor.eq(donor)]
                prior = old[old.marker.eq(key) & old.group.eq(group) & old.donor.eq(donor)].iloc[0]
                np.testing.assert_allclose(percentiles.loc[idx].mean(), prior['mean'], atol=1e-10, rtol=0)
                means_out.append(dict(marker=key, group=group, donor=donor, cells=len(idx), mean_CLR=mean))
            q1, med, q3 = np.quantile(values, [.25, .5, .75])
            summaries.append(dict(marker=key, group=group, cells=len(values),
                                  q1=q1, median=med, q3=q3, minimum=values.min(), maximum=values.max(),
                                  pooled_mean=values.mean(), equal_donor_mean=means.mean()))
            points_out.append(b[['cell_uid', 'donor', 'group']].assign(marker=key, value_CLR=values))

    all_values = pd.concat(points_out, ignore_index=True)
    assert len(all_values) == 8628
    values = all_values.value_CLR.to_numpy()
    y_low = np.floor((values.min()-.1)*2)/2
    y_high = np.ceil((values.max()+.1)*2)/2
    ax.set(xlim=(-.65, 2.65), ylim=(y_low, y_high))
    ax.set_xticks([0, 1, 2], [name for _, name in FEATURES])
    ax.set_yticks(np.arange(np.ceil(y_low/2)*2, y_high+.01, 2))
    ax.set_ylabel('Normalized protein\nexpression (CLR)', fontsize=9.5, labelpad=5)
    ax.tick_params(axis='x', labelsize=10.2, length=0, pad=7)
    ax.tick_params(axis='y', labelsize=8.6, length=3, pad=3)
    ax.yaxis.grid(True, color=GRID, lw=.6)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    for side in ['left', 'bottom']:
        ax.spines[side].set_color('#7A838A')
        ax.spines[side].set_linewidth(.8)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    text_items = list(fig.texts)+[ax.yaxis.label]+ax.get_xticklabels()+ax.get_yticklabels()
    for legend in fig.legends:
        text_items += legend.get_texts()
    for text in text_items:
        box = text.get_window_extent(renderer)
        assert box.x0 >= -1 and box.y0 >= -1 and box.x1 <= fig.bbox.width+1 and box.y1 <= fig.bbox.height+1, text.get_text()
    stem = HERE/'panel_E_protein_CLR'
    for ext in ['pdf', 'png', 'svg']:
        fig.savefig(stem.with_suffix('.'+ext), dpi=320, facecolor='white')
    plt.close(fig)
    with fitz.open(stem.with_suffix('.pdf')) as pdf:
        assert 'expression (CLR)' in pdf[0].get_text()
        assert 'percentile' not in pdf[0].get_text().lower()
        pdf[0].get_pixmap(dpi=250, alpha=False).save(HERE/'pdf_preview.png')
    all_values.to_csv(HERE/'plotted_cell_values.csv.gz', index=False)
    pd.DataFrame(means_out).to_csv(HERE/'donor_means.csv', index=False)
    pd.DataFrame(summaries).to_csv(HERE/'box_summaries.csv', index=False)
    assert before == {p: sha(Path(p)) for p in before}
    (HERE/'AUDIT.json').write_text(json.dumps(dict(
        status='PASS', input_sha256=before, inputs_unchanged=True,
        grouping_unchanged=True, source_CD8=4788, lower_cells=1439, higher_cells=1437,
        displayed_cell_marker_points=8628, displayed_donor_means=18,
        expression='log1p ADT count minus per-cell mean log1p ADT count over non-isotype markers excluding TIGIT and CD39',
        uses_percentile_axis=False, original_percentile_donor_means_reproduced=True,
        all_selected_cells_shown=True, expression_clipping=False, y_limits=[y_low, y_high],
        biological_replicates=3, cell_level_inferential_tests=False,
        existing_figures_replaced=False, manuscript_replaced=False), indent=2)+'\n')
    (HERE/'README_ZH.md').write_text('''# Protein：CLR 表达值版

`panel_E_protein_CLR.png/.pdf/.svg` 使用每枚细胞的 CLR 归一化 ADT 表达值，替换旧细胞分布图中的表达百分位。三个 marker 为 LAG3、CD127、CD62L。

分组保持原 CD39/TIGIT 状态的 donor 内低/高 30%，分别 1,439/1,437 个肿瘤 CD8。三个 donor、细胞名单和测量缓存均不变。浅色点为全部细胞；箱体为合并细胞的中位数和四分位距，须为 1.5 IQR；浅色轮廓为 KDE；黑色菱形为三个 donor 的组内 CLR 均值。细胞数较多的 donor 对箱体权重较大，未进行细胞级显著性检验。

CLR 数值直接读取原先已验证的 `source_features_and_programs.csv.gz` 中三项 `_measured` 列：log1p(ADT count) 减去同一细胞中非 isotype、且排除 TIGIT/CD39 后各 ADT 的平均 log1p count。这是归一化相对表达值，不是原始计数或绝对蛋白浓度。负数表示低于该细胞的归一化参考水平。没有再次取秩、做 z-score、裁剪或新增归一化。

脚本重算同一批细胞原百分位并复现此前的 18 个 donor/group/marker 均值，以核对数据对应；绘图改用其取秩前 CLR 数值。RNA 图和目前包含 343-cell 空间双图的整图未修改，等待确定展示形式。
''')
    print(pd.DataFrame(summaries).round(3).to_string(index=False))
    print('PASS: exact original cells and grouping; CLR expression plotted without clipping.')


if __name__ == '__main__':
    main()
