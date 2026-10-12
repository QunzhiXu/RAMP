#!/usr/bin/env python3
"""Compare fixed donor-wise CD39/TIGIT tail fractions for six fixed features.

Read-only biological inputs. Descriptive effects, no inferential DE or optimal
threshold search. All writes stay in this exploration directory.
"""
from pathlib import Path
import hashlib
import json
import os

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
os.environ.setdefault('MPLCONFIGDIR', str(HERE / '.cache' / 'matplotlib'))
os.environ.setdefault('MPLBACKEND', 'Agg')

import anndata as ad
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata

INPUTS = {
    'scores': ROOT / 'test/hnscc_tigit_cd39_molecular_spatial_refinement_20260904/citeseq_state_validation/results/tumour_cd8_molecular_scores.csv.gz',
    'formal_scores': ROOT / 'figure_HNSCC/biomarker_discovery/screen/rahim_tumor_cd8_biomarker_programs.csv.gz',
    'adt': ROOT / 'figure_HNSCC/sampled_data/rahim_citeseq/complete/GSE212797_adata_ADT.h5ad',
    'rna': ROOT / 'figure_HNSCC/sampled_data/rahim_citeseq/complete/GSE212797_adata.h5ad',
    'original_contrasts': ROOT / 'story_test_hnscc/ending_exploration_20260910/individual_marker_donor_contrasts.csv',
}
FRACTIONS = [.10, .20, .25, .30, .40, .50]
FEATURES = [
    ('ADT', 'CD223--LAG-3', 'LAG-3', 1),
    ('ADT', 'CD127--IL-7Ra', 'CD127', -1),
    ('ADT', 'CD62L', 'CD62L', -1),
    ('RNA', 'CTLA4', 'CTLA4', 1),
    ('RNA', 'CXCL13', 'CXCL13', 1),
    ('RNA', 'IL7R', 'IL7R', -1),
]
SCORE = 'compact_top5pct_clr_program'
LOW, HIGH, INK, GRID = '#2775B6', '#D95F02', '#182027', '#E7EBEE'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while b := f.read(8 * 1024 * 1024):
            h.update(b)
    return h.hexdigest()


def dense(x):
    return x.toarray() if sparse.issparse(x) else np.asarray(x)


def save_figure(fig, stem):
    fig.canvas.draw()
    for fmt in ['png', 'svg']:
        fig.savefig(HERE / f'{stem}.{fmt}', dpi=240, facecolor='white')
    plt.close(fig)


def load_features(d):
    adt = ad.read_h5ad(INPUTS['adt'])
    pos = adt.obs_names.get_indexer(d.cell_uid)
    assert (pos >= 0).all()
    assert adt.obs.iloc[pos].patient.astype(str).tolist() == d.donor.tolist()
    counts = dense(adt.X[pos, :]).astype(float)
    names = np.asarray(adt.var_names, dtype=str)
    background = (np.char.find(np.char.lower(names), 'isotype') < 0) & ~np.isin(names, ['TIGIT--VSTM3', 'CD39'])
    log = np.log1p(counts)
    clr = log - log[:, background].mean(axis=1, keepdims=True)
    values = {}
    for modality, marker, label, direction in FEATURES:
        if modality == 'ADT':
            values[(modality, marker)] = clr[:, adt.var_names.get_loc(marker)]
    rna = ad.read_h5ad(INPUTS['rna'], backed='r')
    pos = rna.obs_names.get_indexer(d.cell_uid)
    assert (pos >= 0).all()
    obs = rna.obs.iloc[pos]
    assert obs.patient.astype(str).tolist() == d.donor.tolist()
    assert obs.tissue.astype(str).eq('tumor').all()
    assert obs.cell_type.astype(str).eq('CD8 T cell').all()
    order = np.argsort(pos)
    genes = [marker for modality, marker, *_ in FEATURES if modality == 'RNA']
    cols = rna.raw.var_names.get_indexer(genes)
    assert (cols >= 0).all()
    x = dense(rna.raw.X[pos[order], :][:, cols])[np.argsort(order)]
    for j, gene in enumerate(genes):
        values[('RNA', gene)] = x[:, j]
    rna.file.close()
    assert all(np.isfinite(v).all() for v in values.values())
    return values


def compare(d, values):
    counts, rows = [], []
    assignment = d[['cell_uid', 'donor', SCORE, 'state_rank']].copy()
    for q in FRACTIONS:
        # At 50%, assign an exact median tie to Lower once; no overlapping cells.
        low_mask = d.state_rank.le(q)
        high_mask = d.state_rank.gt(.5) if q == .5 else d.state_rank.ge(1-q)
        assert not (low_mask & high_mask).any()
        if q == .5:
            assert (low_mask | high_mask).all()
        assignment[f'group_{int(q*100)}pct'] = np.select([low_mask, high_mask], ['Lower', 'Higher'], default='Middle')
        for donor, b in d.groupby('donor', sort=True):
            ix = b.index
            low = ix[low_mask.loc[ix]]
            high = ix[high_mask.loc[ix]]
            assert min(len(low), len(high)) > 0
            counts.append(dict(fraction=q, donor=donor, n_total=len(b), n_low=len(low), n_high=len(high), n_middle=len(b)-len(low)-len(high)))
            for modality, marker, label, direction in FEATURES:
                v = values[(modality, marker)]
                pct = pd.Series((rankdata(v[ix])-.5)/len(ix), index=ix)
                lo_pct, hi_pct = pct.loc[low].mean(), pct.loc[high].mean()
                lo_det, hi_det = (v[low] > 0).mean(), (v[high] > 0).mean()
                lo, hi = (lo_pct, hi_pct) if modality == 'ADT' else (lo_det, hi_det)
                rows.append(dict(fraction=q, donor=donor, modality=modality, marker=marker, label=label,
                                 expected_direction=direction, n_low=len(low), n_high=len(high),
                                 low=100*lo, high=100*hi, delta=100*(hi-lo),
                                 low_mean_percentile=lo_pct, high_mean_percentile=hi_pct,
                                 low_detection_fraction=lo_det if modality == 'RNA' else np.nan,
                                 high_detection_fraction=hi_det if modality == 'RNA' else np.nan))
    donor = pd.DataFrame(rows)
    donor['direction_consistent'] = donor.delta * donor.expected_direction > 0
    summary = donor.groupby(['fraction', 'modality', 'marker', 'label', 'expected_direction'], sort=False).agg(
        low=('low', 'mean'), high=('high', 'mean'), delta=('delta', 'mean'),
        min_donor_delta=('delta', 'min'), max_donor_delta=('delta', 'max'),
        donors_consistent=('direction_consistent', 'sum'), n_donors=('donor', 'nunique'),
    ).reset_index()
    baseline = summary[summary.fraction.eq(.2)].set_index(['modality', 'marker']).delta
    summary['effect_relative_to_20pct'] = [r.delta / baseline.loc[(r.modality, r.marker)] for r in summary.itertuples()]
    return pd.DataFrame(counts), donor, summary, assignment


def verify_baseline(donor):
    old = pd.read_csv(INPUTS['original_contrasts'], dtype={'donor': str})
    check = donor[donor.fraction.eq(.2)].merge(old, on=['donor', 'modality', 'marker'], validate='one_to_one', suffixes=('_new', '_old'))
    assert len(check) == 18
    cols = ['n_low', 'n_high', 'low_mean_percentile', 'high_mean_percentile', 'low_detection_fraction', 'high_detection_fraction']
    for col in cols:
        np.testing.assert_allclose(check[col+'_new'], check[col+'_old'], rtol=0, atol=1e-12, equal_nan=True)
    return len(check)


def plots(summary, donor, counts):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none', 'font.size': 10})
    handles = [Line2D([0], [0], color=c, marker='o', label=l, lw=2) for l, c in [('Lower state', LOW), ('Higher state', HIGH)]]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.5))
    fig.subplots_adjust(left=.072, right=.98, bottom=.10, top=.81, hspace=.59, wspace=.29)
    fig.suptitle('CD39/TIGIT group-fraction comparison', fontsize=17, weight='bold', y=.98)
    fig.text(.5, .917, 'Same six features and score; fraction selected from each donor at each tail', ha='center', fontsize=10.8)
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .902), ncol=2, frameon=False)
    for ax, (modality, marker, label, direction) in zip(axes.flat, FEATURES):
        s = summary[summary.marker.eq(marker) & summary.modality.eq(modality)].sort_values('fraction')
        b = donor[donor.marker.eq(marker) & donor.modality.eq(modality)]
        for _, g in b.groupby('donor'):
            g = g.sort_values('fraction')
            for col, color in [('low', LOW), ('high', HIGH)]:
                ax.plot(g.fraction*100, g[col], color=color, alpha=.22, lw=.9)
        for col, color in [('low', LOW), ('high', HIGH)]:
            ax.plot(s.fraction*100, s[col], color=color, marker='o', markersize=4.7, lw=2.2)
        ax.axvline(20, color='#929BA3', ls=':', lw=1, zorder=0)
        ax.set_title(f'{label} | {"Protein" if modality == "ADT" else "RNA"}', fontsize=12, weight='bold', pad=8)
        ax.set(xlim=(8, 52), ylim=(0, 100), xticks=[10, 20, 25, 30, 40, 50], yticks=[0, 25, 50, 75, 100])
        ax.set_xlabel('Cells in each tail (%)', fontsize=10)
        ax.set_ylabel('Mean expression percentile' if modality == 'ADT' else 'RNA detected (%)', fontsize=9)
        ax.grid(axis='y', color=GRID)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=9)
    fig.text(.98, .012, 'Bold: equal-donor mean  |  Thin: individual donors  |  Dotted: original 20%', ha='right', fontsize=9, color='#5F6972')
    save_figure(fig, 'six_marker_fraction_trends')

    # Identically scaled candidates allow a direct presentation comparison.
    for qs, stem in [([.2, .3, .5], 'panel_E_20_30_50_comparison'), (FRACTIONS, 'panel_E_all_fractions')]:
        nrows = int(np.ceil(len(qs)/3))
        fig = plt.figure(figsize=(12.9, 3.35*nrows + .35), facecolor='white')
        fig.suptitle('Measured CD39/TIGIT state: lower versus higher', y=.985, fontsize=15, weight='bold')
        for k, q in enumerate(qs):
            col, row = k % 3, k // 3
            x0, y0 = col/3, 1-(row+1)/nrows
            height = .85/nrows
            y0 = .04 + (nrows-1-row)*.85/nrows
            nt = counts[counts.fraction.eq(q)][['n_low', 'n_high']].sum()
            title_y = y0 + height*.97
            fig.text(x0+1/6, title_y, f'Lower {q:.0%} vs higher {q:.0%}', ha='center', fontsize=13, weight='bold')
            fig.text(x0+1/6, title_y-height*.088, f'{int(nt.n_low):,} / {int(nt.n_high):,} cells', ha='center', fontsize=9.5, color='#5F6972')
            legend = fig.legend(handles=handles, loc='center', bbox_to_anchor=(x0+1/6, title_y-height*.185), ncol=2, frameon=False, fontsize=9, columnspacing=1.0, handlelength=1)
            for modality, ax_x, ax_w, xlabel in [('ADT', .049, .100, 'Expression percentile'), ('RNA', .234, .085, 'RNA detected (%)')]:
                ax = fig.add_axes([x0+ax_x, y0+height*.17, ax_w, height*.45])
                features = [f for f in FEATURES if f[0] == modality]
                for j, (_, marker, label, _) in enumerate(features):
                    s = summary[summary.fraction.eq(q) & summary.marker.eq(marker) & summary.modality.eq(modality)].iloc[0]
                    y = len(features)-1-j
                    ax.plot([s.low, s.high], [y, y], color='#BAC4CC', lw=1.6)
                    ax.scatter(s.low, y, color=LOW, s=44, edgecolors='white', linewidths=.5, zorder=3)
                    ax.scatter(s.high, y, color=HIGH, s=44, edgecolors='white', linewidths=.5, zorder=3)
                ax.set(xlim=(-4, 104), ylim=(-.6, 2.6), xticks=[0, 50, 100], yticks=[2, 1, 0])
                ax.set_yticklabels([f[2] for f in features], fontsize=10)
                ax.set_title('Protein' if modality == 'ADT' else 'RNA', fontsize=11, weight='bold', pad=9)
                ax.set_xlabel(xlabel, fontsize=9, labelpad=7)
                ax.spines[['top', 'right', 'left']].set_visible(False)
                ax.spines['bottom'].set_color('#7A838A')
                ax.tick_params(axis='y', length=0, pad=4)
                ax.tick_params(axis='x', labelsize=9)
                ax.grid(axis='x', color=GRID)
                ax.set_axisbelow(True)
        save_figure(fig, stem)


def write_report(counts, summary):
    lines = [
        '# Panel E：高低组细胞比例比较', '',
        '日期：2026-09-10。固定六项候选、细胞集合和实测 CD39/TIGIT 联合得分，仅改变每个 donor 内两端各取多少细胞。', '',
        '## 结果与建议', '',
        '10%、20%、25%、30%、40%、50% 六种分组下，六项指标的高低组差异方向均在三个 donor 中一致。随着纳入更多中间状态细胞，组间差异逐渐缩小，没有观察到某个比例发生方向反转。这里的一致是方向一致，不代表每项差异都经过显著性检验。', '',
        '如果目的是扩大细胞覆盖范围，我建议先看各 30% 的展示：共纳入 2,876 个细胞，比原来 1,914 个多约一半，六项差异仍保留原来的约 80%–84%。若希望使用全部细胞，各 50% 仍保留同向差异，但幅度约为原来的 54%–63%。', '',
        '如果目的是让两组距离更大，各 10% 的差异比原来增加约 15%–34%，但只纳入 959 个细胞。这是选择更极端状态的结果，不能视为预测模型或生物学机制得到改善。没有定义或自动选择所谓最优阈值。', '',
        '| 每端比例 | Lower 细胞 | Higher 细胞 | 中间未纳入 | 六项差异相对原 20% 的范围 |',
        '|---|---:|---:|---:|---:|',
    ]
    for q, b in counts.groupby('fraction'):
        c = b[['n_low', 'n_high', 'n_middle']].sum()
        rel = summary.loc[summary.fraction.eq(q), 'effect_relative_to_20pct']*100
        lines.append(f'| {q:.0%} | {int(c.n_low):,} | {int(c.n_high):,} | {int(c.n_middle):,} | {rel.min():.1f}%–{rel.max():.1f}% |')
    lines += ['', '差异保留比例逐项计算，再列出六项的最小值至最大值；没有把蛋白百分位差和 RNA 检出率差混合成一个总分。', '',
              '## 同一尺度的展示', '',
              '![20%、30%、50% 对比](panel_E_20_30_50_comparison.png)', '',
              '![六种比例的变化](six_marker_fraction_trends.png)', '',
              '细线表示单个 donor，粗线是三个 donor 等权均值。蓝色为 Lower，橙色为 Higher。', '',
              '## 六项的高低组差异', '',
              '下表均为 Higher − Lower。蛋白单位是平均表达百分位点，RNA 单位是检出率百分点。', '',
              '| 测量 | 指标 | 10% | 20% | 25% | 30% | 40% | 50% |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for modality, marker, label, _ in FEATURES:
        b = summary[summary.modality.eq(modality) & summary.marker.eq(marker)].set_index('fraction')
        vals = ' | '.join(f'{b.loc[q, "delta"]:+.2f}' for q in FRACTIONS)
        lines.append(f'| {"蛋白" if modality == "ADT" else "RNA"} | {label} | {vals} |')
    lines += ['', '## 分组与复现', '',
              '- 每位 donor 内对原始固定的 CD39/TIGIT 联合得分计算 midrank；一般使用 <=q 与 >=1−q。',
              '- 50% 使用 <=0.5 与 >0.5，使两组互斥且覆盖所有细胞。恰好位于中位数的细胞只分配给 Lower；秩的并列可使实际组大小略偏离精确百分比。',
              '- 六项指标沿用原来的三个蛋白与三个 RNA，未依据此次比较重新选择。',
              '- 蛋白使用 pair-free CLR 在全部 donor CD8 内的表达百分位，不在每个高低组中重新排序。RNA 使用 raw.X >0 的检出比例。',
              '- 每位 donor 分别计算，再等权平均；没有按 donor 细胞数加权。',
              '- 20% 的 18 个 donor×指标结果与原始结果在 1e-12 误差范围内重现；全部 4,788 个细胞的联合得分与正式来源一致。',
              '- 所有生物学输入的 SHA-256 在运行前后相同；原来的 Panel E 和合图未替换。',
              '- 这是同一数据集内的分组比例比较，没有新增 DE、P 值、FDR 或独立验证。', '',
              '## 文件', '',
              '- `scan_group_fractions.py`：完整重现脚本。',
              '- `group_counts.csv`：每种比例、每个 donor 的细胞数。',
              '- `cell_group_assignments.csv.gz`：全部细胞在各比例下的分组。',
              '- `donor_contrasts.csv`：每个 donor 的原始对比。',
              '- `equal_donor_contrasts.csv`：等权汇总及相对 20% 的差异保留量。',
              '- `effect_size_by_fraction.csv`：六项差异随比例变化。',
              '- `panel_E_20_30_50_comparison.png/.svg`：三种比例的图形对照。',
              '- `panel_E_all_fractions.png/.svg`：全部六种比例的图形对照。',
              '- `six_marker_fraction_trends.png/.svg`：各 donor 和等权均值变化。',
              '- `run_manifest.json`：来源、哈希、分组规则与重现检查。', '',
              '运行：`python3 scan_group_fractions.py`。']
    (HERE / 'README_ZH.md').write_text('\n'.join(lines)+'\n')


def main():
    hashes = {k: sha(p) for k, p in INPUTS.items()}
    d = pd.read_csv(INPUTS['scores'], dtype={'donor': str})
    assert len(d) == 4788 and d.cell_uid.is_unique and set(d.donor) == {'126', 'indiv_1', 'indiv_2'}
    formal = pd.read_csv(INPUTS['formal_scores'], dtype={'donor': str})
    check = d[['cell_uid', 'donor', SCORE]].merge(formal[['cell_uid', 'donor', SCORE]], on=['cell_uid', 'donor'], validate='one_to_one', suffixes=('_cached', '_formal'))
    assert len(check) == len(d)
    np.testing.assert_allclose(check[SCORE+'_cached'], check[SCORE+'_formal'], rtol=0, atol=1e-14)
    d['state_rank'] = d.groupby('donor')[SCORE].transform(lambda v: (rankdata(v)-.5)/len(v))
    values = load_features(d)
    counts, donor, summary, assignment = compare(d, values)
    matched = verify_baseline(donor)
    counts.to_csv(HERE / 'group_counts.csv', index=False)
    donor.to_csv(HERE / 'donor_contrasts.csv', index=False)
    summary.to_csv(HERE / 'equal_donor_contrasts.csv', index=False)
    assignment.to_csv(HERE / 'cell_group_assignments.csv.gz', index=False)
    summary.pivot(index=['modality', 'label'], columns='fraction', values='delta').to_csv(HERE / 'effect_size_by_fraction.csv')
    plots(summary, donor, counts)
    write_report(counts, summary)
    assert hashes == {k: sha(p) for k, p in INPUTS.items()}
    manifest = dict(inputs={k: dict(path=str(p), sha256=hashes[k]) for k, p in INPUTS.items()}, inputs_unchanged=True,
                    fractions=FRACTIONS, features=FEATURES, cells=len(d), donor_count=3,
                    grouping='Within-donor joint measured CD39/TIGIT score midrank: <=q and >=1-q; at q=0.5 use <=0.5 and >0.5, assigning exact median ties only to Lower',
                    group_score_matches_formal=True, original_20pct_donor_feature_rows_reproduced=matched,
                    aggregation='Equal-weight donor means', protein_endpoint='Mean donor-wide pair-free CLR expression percentile (0-100)',
                    rna_endpoint='RNA raw.X >0 detection percent', inference='Descriptive; no DE p values, FDR, or optimization objective',
                    original_panel_unchanged=True)
    (HERE / 'run_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('COUNTS\n'+counts.groupby('fraction')[['n_low', 'n_high', 'n_middle']].sum().to_string())
    print('\nEFFECTS (Higher minus Lower)\n'+summary.pivot(index=['modality', 'label'], columns='fraction', values='delta').round(2).to_string())
    print('\nDIRECTION CONSISTENCY\n'+summary.pivot(index=['modality', 'label'], columns='fraction', values='donors_consistent').to_string())
    print('\nRELATIVE EFFECT vs 20%\n'+summary.pivot(index=['modality', 'label'], columns='fraction', values='effect_relative_to_20pct').round(3).to_string())
    print(f'Baseline replicated for {matched} donor-feature rows; all source hashes unchanged.')


if __name__ == '__main__':
    main()
