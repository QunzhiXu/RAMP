"""Shorten B's subtitle and try smoother CD19 ridges with KDE factor 0.35."""
from runtime import BASE, save, sha, now
import json
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import assemble_cd19_layout_v4 as v4
v3, v = v4.previous, v4.v

OUT = BASE / 'figure_layout_v5'
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'panels').mkdir(exist_ok=True)
BANDWIDTH = .35
smoothed_curves = {}
curve_rows = []
for kind, groups in [('Observed', v.observed), ('Predicted', v.predicted)]:
    smoothed_curves[kind] = {}
    for group, _, _ in v.ridge['GROUPS']:
        values = groups[group]
        values = values[np.isfinite(values)]
        density = gaussian_kde(values, bw_method=BANDWIDTH)(v.ridge_grid)
        assert np.isfinite(density).all() and density.max() > 0
        smoothed_curves[kind][group] = density
        curve_rows += [{'condition': kind, 'group': group, 'x': float(x),
                        'density': float(d), 'display_height': float(.78*d/density.max())}
                       for x, d in zip(v.ridge_grid, density)]


def panel_b(fig, rect, letter='B'):
    before = len(fig.axes)
    v3.panel_b(fig, rect, letter)
    left, right = fig.axes[before:]
    left.set_title('Label-free detection', loc='left', fontsize=10.8, fontweight='bold', pad=8)


def panel_f(fig, rect, letter='F'):
    original = v.ridge_curves
    try:
        v.ridge_curves = smoothed_curves
        v.panel_f(fig, rect, letter)
    finally:
        v.ridge_curves = original


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
        (panel_f, (10.09, 2.79, 4.26, 3.50)),
        (v4.panel_g, (.15, 6.40, 6.72, 3.95)),
        (v3.panel_h, (7.00, 6.40, 7.35, 3.95)),
    ]
    prior_path = BASE / 'figure_layout_v4/LAYOUT_AUDIT.json'
    prior = json.loads(prior_path.read_text())
    rectangles = {draw.__name__: list(rect) for draw, rect in panels}
    assert rectangles == prior['panel_rectangles_inches_from_top_left']
    for draw, rect in panels:
        draw(fig, rect)
    export(fig, 'cd19_story_three_row_v5')
    fig.savefig(OUT / 'second_row_preview.png', dpi=300, facecolor='white',
        bbox_inches=Bbox.from_extents(.10, v3.H-6.34, 14.42, v3.H-2.74))
    plt.close(fig)
    for draw, name, size in [
        (panel_b, 'B_label_free_detection', (5.80, 2.65)),
        (panel_f, 'F_CD19_ridges_bandwidth_035', (4.36, 3.60)),
    ]:
        fig = plt.figure(figsize=size, facecolor='white')
        draw(fig, (.05, .05, size[0]-.1, size[1]-.1))
        export(fig, f'panels/{name}')
        plt.close(fig)
    pd.DataFrame(curve_rows).to_csv(OUT / 'F_ridge_curves_bandwidth_035.csv', index=False)
    assert all(sha(path) == value for path, value in v.input_hashes.items())
    save(OUT / 'LAYOUT_AUDIT.json', {'time_utc': now(), 'script_sha256': sha(__file__),
        'source_sha256': v.input_hashes, 'base_layout_audit_sha256': sha(prior_path),
        'base_script_sha256': {str(BASE / f'scripts/assemble_cd19_layout_v{i}.py'):
                               sha(BASE / f'scripts/assemble_cd19_layout_v{i}.py') for i in [1, 2, 3, 4]},
        'canvas_inches': [v3.W, v3.H], 'panel_rectangles_inches_from_top_left': rectangles,
        'layout_matches_v4': True, 'B_first_subplot_title': 'Label-free detection',
        'F': {'old_KDE_factor': .20, 'new_KDE_factor': BANDWIDTH,
              'implementation': 'scipy.stats.gaussian_kde(values, bw_method=0.35)',
              'same_factor_for_all_six_curves': True, 'grid_points': len(v.ridge_grid),
              'display_peak_height': .78, 'cell_values_changed': False,
              'median_ticks_changed': False, 'x_limits_changed': False,
              'group_cell_counts': {group: len(v.observed[group]) for group, _, _ in v.ridge['GROUPS']}},
        'all_input_files_unchanged': True})
    (OUT / 'README_ZH.md').write_text('''# CD19 story v5：B 标题简化与 F 曲线平滑尝试

主图：cd19_story_three_row_v5.png；可编辑 SVG 为同名文件。

- Panel B 第一张小图标题缩短为 Label-free detection。
- Panel F 的六条分布曲线统一将 scipy gaussian_kde 的 bw_method 从 0.20 调至 0.35，使峰顶更平滑。此数值是 KDE 带宽因子。实测/预测值、细胞集合、横轴范围、网格和每条曲线的峰值归一化高度均沿用原版；仅做这一次更大带宽的展示尝试。

B、F 单图位于 panels。F_ridge_curves_bandwidth_035.csv 保存新曲线数值。其余 panel 与布局沿用 v4，包括 E 的 0.4–0.9 纵轴与 G 的绿色背景细胞。v4 保留用于比较。

复现：Exploration0909/scripts/assemble_cd19_layout_v5.py。输入哈希和参数记录见 LAYOUT_AUDIT.json。
''')
    print(OUT / 'cd19_story_three_row_v5.png', flush=True)


if __name__ == '__main__':
    main()
