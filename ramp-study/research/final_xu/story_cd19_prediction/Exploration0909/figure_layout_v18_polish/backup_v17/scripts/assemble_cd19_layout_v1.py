"""Three-row CD19 figure preview, using frozen data and layout-only changes."""
from runtime import BASE, STORY, save, sha, now
from pathlib import Path
import ast, json
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Ellipse
from matplotlib.transforms import Bbox

OUT = BASE / 'figure_layout_v1'
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'panels').mkdir(exist_ok=True)
FORMAL = STORY / 'figure'
SOURCES = set()


def source(path):
    path = Path(path)
    SOURCES.add(path)
    return path


def csv(path, **kwargs):
    return pd.read_csv(source(path), **kwargs)


def borrow(path, functions, constants=(), extra=None):
    """Read existing plotting functions without running module-level mkdir/main."""
    path = source(path)
    tree = ast.parse(path.read_text())
    namespace = dict(globals())
    namespace.update(extra or {})
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in constants:
                namespace[target.id] = ast.literal_eval(node.value)
    selected = [node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name in functions]
    assert len(selected) == len(functions)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


# Read each frozen input once. The final figure performs no model/UMAP fitting.
cells = csv(BASE / 'CD3_NN_UMAP_validation/cell_predictions_and_umap.csv.gz')
xy_cd3 = cells[['UMAP1', 'UMAP2']].to_numpy()
assert len(cells) == 30000 and cells.donor.nunique() == 12
cd3_audit = json.loads(source(BASE / 'CD3_NN_UMAP_validation/PLOT_AUDIT.json').read_text())
norm = Normalize(cd3_audit['vmin'], cd3_audit['vmax'], clip=True)
cmap = LinearSegmentedColormap.from_list('CD3', ['#DCE1E6', '#F5E7B2', '#EDB34E', '#CF572D', '#861E41'])
donor_scores = csv(BASE / 'validation_subpanels/donor_mean_four_metrics/donor_mean_metrics.csv')
assert len(donor_scores) == 12 and donor_scores.n_markers.eq(14).all()

qc_data = FORMAL / 'panel_B/data'
qc = borrow(FORMAL / 'panel_B/generate_panel_B_combined.py',
            ['draw_qc', 'draw_specificity'], ['DATASETS', 'DATASET_COLORS', 'MARKER_COLORS'])
qc_scores = csv(qc_data / 'codex_top3_marker_donor_scores.csv')
qc_summary = csv(qc_data / 'codex_marker_qc_summary.csv')
qc_specificity = csv(qc_data / 'donor_cd19_specificity.csv')
qc_order = json.loads(source(qc_data / 'manifest.json').read_text())['top3_markers']

workflow = borrow(FORMAL / 'panel_D/generate_panel_D.py', ['rounded_box', 'arrow'],
                  ['INK', 'BLUE', 'GREEN', 'ORANGE', 'MAGENTA'])
ridge = borrow(FORMAL / 'panel_C/generate_panel_C.py', ['load_values', 'density'], ['GROUPS'],
               {'PREDICTION_DIR': STORY / 'prediction/output', 'CODEX_DIR': STORY / 'sampled_data/bm_codex'})
for path in [STORY / 'prediction/output/codex_cd19_prediction.npz',
             STORY / 'prediction/output/codex_prediction_metadata.csv.gz',
             STORY / 'sampled_data/bm_codex/locked_metadata.csv.gz']:
    source(path)
observed, predicted = ridge['load_values']()
pooled = np.concatenate([*observed.values(), *predicted.values()])
lo, hi = np.quantile(pooled[np.isfinite(pooled)], [.002, .998])
lo, hi = min(-1.5, float(lo)), max(3.2, float(hi))
pad = .04 * (hi - lo)
ridge_grid = np.linspace(lo - pad, hi + pad, 500)
ridge_curves = {kind: {group: ridge['density'](values[group], ridge_grid)
                       for group, _, _ in ridge['GROUPS']}
                for kind, values in [('Observed', observed), ('Predicted', predicted)]}

spatial = borrow(FORMAL / 'panel_F/generate_panel_F.py', ['load', 'format_spatial_axis'],
                 ['DONOR', 'THRESHOLD', 'PREDICTED_COLOR', 'B_COLOR'])
spatial_md, spatial_prediction = spatial['load']()
spatial_xy = spatial_md[['x.coord', 'y.coord']].to_numpy(float)

umap_dir = STORY / 'umap/data'
joint = borrow(FORMAL / 'panel_G/generate_panel_G.py', ['load', 'draw_umap', 'highlight_separation'],
               ['CONDITIONS', 'CONDITION_TITLES', 'LABEL_ORDER', 'COLORS'],
               {'COORDINATES': source(umap_dir / 'umap_coordinates.npz'),
                'METADATA': source(STORY / 'alignment/output/cell_metadata.csv.gz'),
                'METRICS': source(umap_dir / 'metrics.csv')})
joint_xy, joint_md, joint_metrics = joint['load']()
raw_image = plt.imread(source(FORMAL / 'panel_A/panel_A_observed_codex_cd19.png'))
input_hashes = {str(path): sha(path) for path in sorted(SOURCES)}

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
    'svg.fonttype': 'none', 'text.color': '#252E35',
    'axes.labelcolor': '#414C56', 'xtick.color': '#414C56', 'ytick.color': '#58636C'})


def ax_at(fig, x, y, w, h):
    fw, fh = fig.get_size_inches()
    return fig.add_axes([x/fw, 1-(y+h)/fh, w/fw, h/fh])


def text_at(fig, x, y, text, **kwargs):
    fw, fh = fig.get_size_inches()
    return fig.text(x/fw, 1-y/fh, text, va='top', **kwargs)


def title(fig, rect, letter, text):
    x, y, w, h = rect
    return text_at(fig, x+.02, y+.02, f'{letter}.  {text}' if letter else text,
                   fontsize=13.5, fontweight='bold')


def panel_a(fig, rect, letter='A'):
    x, y, w, h = rect
    title(fig, rect, letter, 'Failed CODEX CD19')
    ax = ax_at(fig, x, y+.42, w, h-.44)
    ax.imshow(raw_image)
    ax.axis('off')


def panel_b(fig, rect, letter='B'):
    x, y, w, h = rect
    title(fig, rect, letter, 'QC identifies failed CD19')
    left = ax_at(fig, x+.43, y+.66, 2.65, h-1.00)
    right = ax_at(fig, x+3.85, y+.66, 2.65, h-1.00)
    qc['draw_qc'](left, qc_scores, qc_summary, qc_order)
    qc['draw_specificity'](right, qc_specificity)
    for ax in [left, right]:
        for text in ax.texts:
            text.set_fontsize(8.6)
        ax.tick_params(axis='y', labelsize=9)
    for text in left.texts:
        text.set_text(text.get_text().replace(' ranked #1', '\nranked #1'))
        text.set_y(1.095)
        text.set_verticalalignment('top')
        text.set_linespacing(.95)


def panel_c(fig, rect, letter='C'):
    x, y, w, h = rect
    title(fig, rect, letter, 'Cross-platform masked-marker prediction')
    ax = ax_at(fig, x, y+.43, w, h-.48)
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis('off')
    box, arrow = workflow['rounded_box'], workflow['arrow']
    box(ax, .02, .57, .25, .32, '#EEF4FB', '#3E70B2', 'CODEX',
        '14 shared markers\nfailed CD19 withheld', title_size=11, body_size=9.5)
    box(ax, .02, .10, .25, .32, '#F1F8F4', '#31916E', 'References',
        '14 shared markers\nmeasured CD19', title_size=11, body_size=9.5)
    box(ax, .385, .28, .275, .42, '#FFF7E9', '#DF8424', 'Masked-marker NN',
        'Shared protein-state\nrelationships', title_size=11, body_size=10)
    box(ax, .775, .60, .205, .28, '#FCEEF4', '#C51B5D', 'Completed CODEX',
        '+ predicted CD19', title_color='#C51B5D', title_size=10, body_size=9.5)
    box(ax, .775, .10, .205, .28, '#F3F0FA', '#7355A3', 'RAMP alignment',
        'Three datasets', title_color='#5C4386', title_size=10, body_size=9.5)
    arrow(ax, (.288, .71), (.368, .59))
    arrow(ax, (.288, .27), (.368, .40))
    arrow(ax, (.678, .60), (.757, .73))
    arrow(ax, (.877, .575), (.877, .405))


def format_cd3(ax):
    low, high = xy_cd3.min(axis=0), xy_cd3.max(axis=0)
    pad = .035 * (high-low)
    ax.set(xlim=(low[0]-pad[0], high[0]+pad[0]), ylim=(low[1]-pad[1], high[1]+pad[1]))
    ax.set_aspect('equal'); ax.axis('off')


def panel_d(fig, rect, letter='D'):
    x, y, w, h = rect
    title(fig, rect, letter, 'Held-out CD3 validation')
    axes = [ax_at(fig, x+.03, y+.62, 3.20, h-.69),
            ax_at(fig, x+3.61, y+.62, 3.20, h-.69)]
    t = cells.celltype.isin(cd3_audit['T_labels']).to_numpy()
    for mask, color in [(~t, '#E7C36B'), (t, '#266BAE')]:
        axes[0].scatter(*xy_cd3[mask].T, s=1.05, color=color, alpha=1, lw=0, rasterized=True)
    handles = [Line2D([], [], marker='o', linestyle='', color=color, markersize=5)
               for color in ['#266BAE', '#E7C36B']]
    axes[0].legend(handles, ['T cells', 'Other cells'], loc='upper left',
        bbox_to_anchor=(.00, .95), frameon=False, fontsize=9.3,
        handletextpad=.35, labelspacing=.3, borderpad=0)
    values = cells.predicted_CD3.to_numpy()
    order = np.argsort(values, kind='stable')
    sc = axes[1].scatter(*xy_cd3[order].T, c=values[order], s=1.05, cmap=cmap,
                         norm=norm, alpha=1, lw=0, rasterized=True)
    for ax, heading in zip(axes, ['Cell annotation', 'Predicted CD3']):
        format_cd3(ax)
        ax.set_title(heading, fontsize=11.5, fontweight='bold', pad=7)
    cb = fig.colorbar(sc, cax=ax_at(fig, x+7.0, y+1.18, .10, 1.55), ticks=[-1, 0, 1, 2, 3])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8.5, length=2)
    cb.ax.set_title('CD3', fontsize=9.5, pad=7)


def panel_e(fig, rect, letter='E'):
    x, y, w, h = rect
    title(fig, rect, letter, 'Prediction Accuracy (12 donors)')
    ax = ax_at(fig, x+.58, y+.57, w-.70, h-1.11)
    metrics = ['spearman', 'pearson', 'auroc', 'average_precision']
    colors = ['#397CAB', '#5A9993', '#A27BA9', '#CF9856']
    values = [donor_scores[m].to_numpy() for m in metrics]
    bp = ax.boxplot(values, positions=np.arange(4), widths=.48, patch_artist=True,
        whis=1.5, showfliers=False, medianprops={'color': '#2D3942', 'linewidth': 1.6},
        whiskerprops={'linewidth': 1.05}, capprops={'linewidth': 1.05})
    rng = np.random.default_rng(42)
    for i, (color, values) in enumerate(zip(colors, values)):
        bp['boxes'][i].set(facecolor=to_rgba(color, .17), edgecolor=color, linewidth=1.2)
        for item in bp['whiskers'][2*i:2*i+2] + bp['caps'][2*i:2*i+2]:
            item.set_color(color)
        ax.scatter(i+rng.uniform(-.145, .145, len(values)), values, s=24,
                   color=color, edgecolor='white', linewidth=.55, alpha=.95, zorder=4)
    ax.set(xlim=(-.55, 3.55), ylim=(-.025, 1.025))
    ax.set_xticks(range(4), ['Spearman ρ', 'Pearson r', 'AUROC', 'AUPRC\n(AP)'],
                  fontsize=9.4, fontweight='bold')
    ax.set_yticks(np.arange(0, 1.01, .2))
    ax.tick_params(axis='x', length=0, pad=8)
    ax.tick_params(axis='y', length=0, labelsize=9)
    ax.yaxis.grid(True, color='#E8EBEE', linewidth=.7)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set(color='#B9C2C9', linewidth=.7)
    ax.set_ylabel('Mean score across markers', fontsize=9.5, labelpad=7)


def panel_f(fig, rect, letter='F'):
    x, y, w, h = rect
    title(fig, rect, letter, 'CD19 restoration')
    baseline = [2., 1., 0.]
    for column, (kind, values) in enumerate([('Observed', observed), ('Predicted', predicted)]):
        ax = ax_at(fig, x+.67+column*1.84, y+.68, 1.68, h-1.17)
        for base, (display, _, color) in zip(baseline, ridge['GROUPS']):
            density = ridge_curves[kind][display]
            height = .78*density/density.max()
            ax.fill_between(ridge_grid, base, base+height, color=color, alpha=.86, lw=0)
            ax.plot(ridge_grid, base+height, color=color, lw=1.2)
            ax.hlines(base, ridge_grid[0], ridge_grid[-1], color='#D8D8D8', lw=.6, zorder=0)
            ax.plot(np.median(values[display]), base, marker='|', color='#222222', ms=6, mew=1.2)
        ax.set(xlim=(ridge_grid[0], ridge_grid[-1]), ylim=(-.12, 2.96))
        ax.set_yticks(np.array(baseline)+.18, ['B cells', 'CD4 T', 'CD8 T'] if column==0 else ['', '', ''],
                      fontsize=9.5, fontweight='bold')
        ax.set_xticks([0, 2, 4])
        ax.tick_params(axis='x', labelsize=9, length=2)
        ax.tick_params(axis='y', length=0, pad=5)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.spines['bottom'].set_color('#7C858C')
        ax.set_title(kind, fontsize=11.5, fontweight='bold', pad=8)
    text_at(fig, x+w/2+.2, y+h-.21, 'CD19 (KMEANSLOW)', ha='center', fontsize=9.5)


def panel_g(fig, rect, letter='G'):
    x, y, w, h = rect
    title(fig, rect, letter, 'Predicted CD19 recovers spatial B-cell locations')
    provider_b = spatial_md.cluster_anno_l1.eq('B-Cells').to_numpy()
    predicted_high = spatial_prediction > spatial['THRESHOLD']
    width = (w-.18)/2
    pad = .015*np.ptp(spatial_xy, axis=0)
    lims = np.array([spatial_xy.min(axis=0)-pad, spatial_xy.max(axis=0)+pad])
    for i, (mask, color, heading) in enumerate([
        (predicted_high, spatial['PREDICTED_COLOR'], f'Predicted CD19 > 1.4 (n = {predicted_high.sum()})'),
        (provider_b, spatial['B_COLOR'], f'Annotated B cells (n = {provider_b.sum()})')]):
        ax = ax_at(fig, x+i*(width+.18), y+.65, width, h-.69)
        ax.scatter(*spatial_xy[~mask].T, s=5, color='#FFD92F', alpha=.62, lw=0, rasterized=True)
        ax.scatter(*spatial_xy[mask].T, s=10.5, color=color, alpha=.92, lw=0, rasterized=True)
        spatial['format_spatial_axis'](ax, tuple(lims[:, 0]), tuple(lims[:, 1]))
        ax.set_title(heading, fontsize=10.8, fontweight='bold', pad=7)


def panel_h(fig, rect, letter='H'):
    x, y, w, h = rect
    title(fig, rect, letter, 'Predicted CD19 restores Mature B–NK separation')
    labels = joint_md.celltype.astype(str).to_numpy()
    before = joint_metrics.loc['without_cd19', 'matureB_vs_NK_silhouette_umap']
    after = joint_metrics.loc['with_predicted_cd19', 'matureB_vs_NK_silhouette_umap']
    text_at(fig, x+5.67, y+.38, f'Mature B–NK silhouette: {before:.3f} → {after:.3f}',
             ha='center', fontsize=9.5, fontweight='bold')
    for i, (condition, heading) in enumerate(zip(joint['CONDITIONS'], ['14 shared markers', '+ predicted CD19'])):
        ax = ax_at(fig, x+2.30+i*3.48, y+.88, 3.43, h-.94)
        joint['draw_umap'](ax, joint_xy[condition], labels)
        joint['highlight_separation'](ax, condition)
        ax.set_title(heading, fontsize=11, fontweight='bold', pad=7)
    active = [label for label in joint['LABEL_ORDER'] if np.any(labels==label)]
    handles = [Line2D([], [], marker='o', linestyle='', color=joint['COLORS'][label], markersize=5)
               for label in active]
    fw, fh = fig.get_size_inches()
    fig.legend(handles, active, loc='center left', bbox_to_anchor=((x+.015)/fw, 1-(y+2.33)/fh),
               frameon=False, fontsize=9.0, labelspacing=.16, handletextpad=.33,
               borderpad=0, borderaxespad=0)


def export(fig, name, dpi=280):
    for ext in ['png', 'svg']:
        fig.savefig(OUT / f'{name}.{ext}', dpi=dpi, facecolor='white')


def main():
    fig = plt.figure(figsize=(18, 11.40), facecolor='white')
    panels = [
        (panel_a, (.25, .20, 3.05, 2.85)),
        (panel_b, (3.70, .20, 6.70, 2.85)),
        (panel_c, (10.80, .20, 6.95, 2.85)),
        (panel_d, (.25, 3.35, 7.60, 3.65)),
        (panel_e, (8.20, 3.35, 4.85, 3.65)),
        (panel_f, (13.40, 3.35, 4.35, 3.65)),
        (panel_g, (.25, 7.30, 7.90, 3.85)),
        (panel_h, (8.50, 7.30, 9.25, 3.85)),
    ]
    for draw, rect in panels:
        draw(fig, rect)
    export(fig, 'cd19_story_three_row_v1')
    # A larger second-row crop helps review the four small plots together.
    fig.savefig(OUT / 'second_row_preview.png', dpi=280, facecolor='white',
        bbox_inches=Bbox.from_extents(.18, 11.4-7.08, 17.85, 11.4-3.28))
    plt.close(fig)
    # Regenerate the newly arranged subpanels as standalone editable files.
    for draw, name, size in [(panel_d, 'D_CD3_validation_pair', (7.70, 3.75)),
                             (panel_e, 'E_donor_mean_accuracy', (4.95, 3.75)),
                             (panel_f, 'F_square_CD19_ridges', (4.45, 3.75))]:
        fig = plt.figure(figsize=size, facecolor='white')
        draw(fig, (.05, .05, size[0]-.1, size[1]-.1))
        export(fig, f'panels/{name}')
        plt.close(fig)
    assert all(sha(path)==value for path, value in input_hashes.items())
    save(OUT / 'LAYOUT_AUDIT.json', {'time_utc': now(), 'source_sha256': input_hashes,
        'script_sha256': sha(__file__), 'canvas_inches': [18, 11.4],
        'rows': [['A_failed_CD19', 'B_QC', 'C_workflow'],
                 ['D_CD3_annotation_and_prediction', 'E_donor_accuracy', 'F_CD19_ridges'],
                 ['G_spatial_CD19', 'H_joint_UMAP_comparison']],
        'panel_rectangles_inches_from_top_left': {draw.__name__: list(rect) for draw, rect in panels},
        'CD3': {'n_cells': len(cells), 'n_donors': 12, 'T_cells': int(cells.celltype.isin(cd3_audit['T_labels']).sum()),
                'coordinates_refit': False, 'all_cells_shown': True,
                'label_layer_order': ['Other cells', 'T cells'], 'color_limits': [norm.vmin, norm.vmax]},
        'accuracy': {'point_unit': 'donor', 'n_points_per_metric': 12, 'aggregation': 'arithmetic mean across 14 markers'},
        'ridges': {'layout': 'Observed left, Predicted right', 'group_n': {g: len(observed[g]) for g, _, _ in ridge['GROUPS']},
                   'same_KDE_as_original': True, 'bandwidth': .20, 'x_limits': [float(ridge_grid[0]), float(ridge_grid[-1])]},
        'spatial': {'donor': spatial['DONOR'], 'n_cells': len(spatial_md), 'threshold': spatial['THRESHOLD']},
        'joint_UMAP': {'n_cells': len(joint_md), 'coordinate_refit': False},
        'input_files_unchanged': True, 'all_writes_under_Exploration0909': True})
    (OUT / 'README_ZH.md').write_text('''# CD19 story 三行排版预览 v1

主图：`cd19_story_three_row_v1.png`；可编辑图形版：同名 SVG。`second_row_preview.png` 为第二行单独预览。

- 第一行 A–C：原始 CODEX CD19 图、QC 与 CD19 特异性检查、跨平台神经网络预测 workflow。
- 第二行 D–F：D 包含细胞类型注释与 Predicted CD3 两张 UMAP；E 为每位 donor 的跨 14 marker 平均准确性；F 为左右并排的 Observed / Predicted CD19 ridge plot。
- 第三行 G–H：原有 CD19 空间对照与补全前后联合 UMAP。

所有图形由冻结数据重新绘制。CD3 图沿用 13 marker 三平台联合 UMAP，仅显示同一批 30,000 个 CODEX 细胞；左图 T cells 覆盖 Other cells，全部点保留。右图沿用既定预测值及色阶。两张图共同归入 D，但左图是作者注释，不是实测 CD3 强度图。E 每列 12 点，每点是一位 donor 的 14 个 marker 指标算术平均。计算规则及逐 donor 数据见 validation_subpanels/donor_mean_four_metrics。

Ridge 图重排为近方形，左列 Observed，右列 Predicted；三个群体依次为 B、CD4 T、CD8 T。细胞集合、KDE 带宽、每条曲线峰值归一化规则、实测与预测值以及横轴范围均沿用原图。灰色小字未加入新准确性图；点的定义保存在文档中。

G 和 H 沿用原 CD19 神经网络结果、空间 donor/阈值和冻结 UMAP；它们与 CD3 留出验证属于不同的任务。此次只调整排版，没有重新训练、筛选 donor 或拟合 UMAP。原正式 figure、prediction、sampled_data、alignment 和 umap 文件保持原样。

复现：运行 `Exploration0909/scripts/assemble_cd19_layout_v1.py`。写入范围由 runtime 审计钩子限制在 Exploration0909。原脚本中的绘图函数通过 AST 单独读取，未运行其模块级输出或 main。输入哈希和绘图参数记录于 LAYOUT_AUDIT.json。
''')
    print(OUT / 'cd19_story_three_row_v1.png', flush=True)


if __name__ == '__main__':
    main()
