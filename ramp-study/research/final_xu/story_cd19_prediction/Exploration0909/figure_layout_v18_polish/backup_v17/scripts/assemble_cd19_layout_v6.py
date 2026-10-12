"""Use the familiar low-anomaly CD3 marker as Panel B's third column."""
from runtime import BASE, save, sha, now
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import assemble_cd19_layout_v5 as v5
v4, v3, v = v5.v4, v5.v3, v5.v

OUT = BASE / 'figure_layout_v6'
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'panels').mkdir(exist_ok=True)
SCORE_SOURCE = v.FORMAL / 'panel_B/data/all_marker_qc_scores.csv'
score_hash = sha(SCORE_SOURCE)
all_scores = pd.read_csv(SCORE_SOURCE)
codex = all_scores.loc[all_scores.dataset.eq('BM CODEX')].copy()
assert len(codex) == 180 and codex.unit_key.nunique() == 12 and codex.marker.nunique() == 15
summary = codex.groupby('marker', as_index=False).agg(
    mean_qc_alarm=('qc_alarm', 'mean'), median_rank=('qc_rank_within_unit', 'median'),
    top1_donors=('qc_rank_within_unit', lambda s: int(s.eq(1).sum())),
    top3_donors=('qc_rank_within_unit', lambda s: int(s.le(3).sum())))
summary = summary.sort_values(['top1_donors', 'top3_donors', 'mean_qc_alarm'],
                               ascending=False).reset_index(drop=True)
assert summary.marker.to_list() == v.qc_summary.marker.to_list()
summary.insert(0, 'original_summary_position', np.arange(1, 16))
ORDER = ['CD19', 'CD61', 'CD3']
selected = codex.loc[codex.marker.isin(ORDER)].copy()
assert len(selected) == 36 and selected.groupby('marker').unit_key.nunique().eq(12).all()
cd3 = summary.loc[summary.marker.eq('CD3')].iloc[0]
assert cd3.original_summary_position == 14 and cd3.top3_donors == 0


def panel_b(fig, rect, letter='B'):
    before = len(fig.axes)
    old_scores, old_order = v.qc_scores, v.qc_order
    old_colors = v.qc['MARKER_COLORS']
    try:
        v.qc_scores, v.qc_order = selected, ORDER
        v.qc['MARKER_COLORS'] = {**old_colors, 'CD3': '#66A61E'}
        v5.panel_b(fig, rect, letter)
    finally:
        v.qc_scores, v.qc_order = old_scores, old_order
        v.qc['MARKER_COLORS'] = old_colors
    left, right = fig.axes[before:]
    # Include the low CD3 scores and leave room for the existing rank labels.
    left.set_ylim(.4, 1.16)
    left.set_yticks([.4, .6, .8, 1.0])
    for text in left.texts:
        text.set_y(1.15)
    assert selected.qc_alarm.between(.4, 1).all()


def export(fig, name):
    for ext in ['png', 'svg']:
        fig.savefig(OUT / f'{name}.{ext}', dpi=300, facecolor='white')


def main():
    fig = plt.figure(figsize=(v3.W, v3.H), facecolor='white')
    panels = [
        (v.panel_a, (.15, .13, 2.50, 2.55)),
        (panel_b, (2.76, .13, 5.70, 2.55)),
        (v3.panel_c, (8.57, .13, 5.78, 2.55)),
        (v3.panel_d, (.15, 2.79, 5.45, 3.50)),
        (v4.panel_e, (5.695, 2.79, 4.30, 3.50)),
        (v5.panel_f, (10.09, 2.79, 4.26, 3.50)),
        (v4.panel_g, (.15, 6.40, 6.72, 3.95)),
        (v3.panel_h, (7.00, 6.40, 7.35, 3.95)),
    ]
    prior_path = BASE / 'figure_layout_v5/LAYOUT_AUDIT.json'
    prior = json.loads(prior_path.read_text())
    rectangles = {draw.__name__: list(rect) for draw, rect in panels}
    assert rectangles == prior['panel_rectangles_inches_from_top_left']
    for draw, rect in panels:
        draw(fig, rect)
    export(fig, 'cd19_story_three_row_v6')
    plt.close(fig)
    fig = plt.figure(figsize=(6.00, 2.65), facecolor='white')
    panel_b(fig, (.23, .05, 5.70, 2.55))
    export(fig, 'panels/B_CD19_CD61_CD3')
    plt.close(fig)
    summary.to_csv(OUT / 'all_15_markers_original_ranking.csv', index=False)
    selected.to_csv(OUT / 'B_selected_markers_all_12_donors.csv', index=False)
    summary.loc[summary.marker.isin(ORDER)].to_csv(OUT / 'B_selected_marker_summary.csv', index=False)
    assert sha(SCORE_SOURCE) == score_hash
    assert all(sha(path) == value for path, value in v.input_hashes.items())
    save(OUT / 'LAYOUT_AUDIT.json', {'time_utc': now(), 'script_sha256': sha(__file__),
        'source_sha256': {**v.input_hashes, str(SCORE_SOURCE): score_hash},
        'base_layout_audit_sha256': sha(prior_path),
        'base_script_sha256': {str(BASE / f'scripts/assemble_cd19_layout_v{i}.py'):
                               sha(BASE / f'scripts/assemble_cd19_layout_v{i}.py') for i in range(1, 6)},
        'canvas_inches': [v3.W, v3.H], 'panel_rectangles_inches_from_top_left': rectangles,
        'layout_matches_v5': True,
        'B_left': {'marker_order': ORDER, 'n_donors_per_marker': 12, 'all_36_points_visible': True,
                   'existing_summary_sort': 'descending top1_donors, top3_donors, mean_qc_alarm',
                   'chosen_control': 'CD3', 'original_summary_position': 14, 'total_markers': 15,
                   'control_top3_donors': 0,
                   'selection_reason': 'Familiar T-cell marker with low anomaly ranking, also used in Panel D',
                   'control_description': 'low-anomaly comparator; not proof of complete assay normality',
                   'lowest_marker_existing_summary': summary.iloc[-1].marker,
                   'lowest_marker_mean_alarm': summary.sort_values('mean_qc_alarm').iloc[0].marker,
                   'y_limits': [.4, 1.16], 'scores_recomputed': False},
        'all_input_files_unchanged': True})
    (OUT / 'README_ZH.md').write_text('''# CD19 story v6：Panel B 使用 CD3 低异常对照

主图为 cd19_story_three_row_v6.png，同名 SVG 为矢量版。B 单图见 panels/B_CD19_CD61_CD3.png。

Panel B 左图保留 CD19 和 CD61，将第三列 CD45RA 换成 CD3。每列仍为同一批 12 位 CODEX donor，连线连接同一 donor，指标直接读取现有 QC 结果。为了显示全部 CD3 点，纵轴下限调至 0.4。

原图汇总顺序先按进入异常第 1 名的 donor 次数排序，再按进入 Top 3 的次数、最后按平均异常分数排序。该排序中 CD19、CD61 是前两名；CD3 排第 14/15，12 位 donor 都未进入异常 Top 3。CD3 的平均异常分数为 0.6417，逐 donor 范围为 0.4667–0.7667。

选择 CD3 是因为它是常见 T 细胞标记，异常排名靠后，并能呼应 D 的 CD3 验证。图中视为低异常对照，不由这一相对排名宣称通道绝对正常。现有汇总顺序的最后一名为 CD14；若单纯按平均异常分数，最低为 CD45。这些排序区别保留在审计文件和完整 15-marker 表中。

all_15_markers_original_ranking.csv 为完整原规则排名，B_selected_markers_all_12_donors.csv 保存展示点的逐 donor 数据。其他 panels 沿用 v5，包括 F 的 0.35 带宽、E 的 0.4–0.9 纵轴以及 G 的绿色背景细胞。

复现脚本为 Exploration0909/scripts/assemble_cd19_layout_v6.py。只读取既有数据，所有新文件在 figure_layout_v6 内。
''')
    print(summary.loc[summary.marker.isin(ORDER)].to_string(index=False), flush=True)
    print(OUT / 'cd19_story_three_row_v6.png', flush=True)


if __name__ == '__main__':
    main()
