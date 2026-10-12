"""Add cell-label overlap precision and recall below the two spatial maps."""
from runtime import BASE, save, sha, now
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import assemble_cd19_layout_v7 as v7

v6, v5, v4, v3, v = v7.v6, v7.v5, v7.v4, v7.v3, v7.v
OUT = BASE / 'figure_layout_v8'
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'panels').mkdir(exist_ok=True)

METRIC_SOURCES = [
    v.STORY / 'sampled_data/bm_codex/locked_metadata.csv.gz',
    v.STORY / 'prediction/output/codex_prediction_metadata.csv.gz',
    v.STORY / 'prediction/output/codex_cd19_prediction.npz',
]
metric_hashes = {str(path): sha(path) for path in METRIC_SOURCES}
assert len(v.spatial_md) == len(v.spatial_prediction) == 2500
assert v.spatial_md.cell_id.is_unique and np.isfinite(v.spatial_prediction).all()
truth = v.spatial_md.cluster_anno_l1.eq('B-Cells').to_numpy()
positive = v.spatial_prediction > v.spatial['THRESHOLD']
tp = int(np.sum(truth & positive))
fp = int(np.sum(~truth & positive))
fn = int(np.sum(truth & ~positive))
tn = int(np.sum(~truth & ~positive))
precision = tp / (tp + fp)
recall = tp / (tp + fn)
assert (tp, fp, fn, tn) == (209, 86, 62, 2143)
metrics = {
    'donor': v.spatial['DONOR'],
    'n_cells': len(truth),
    'prediction_threshold': v.spatial['THRESHOLD'],
    'prediction_rule': 'predicted_kmeanslow > 1.4',
    'reference_rule': 'provider cluster_anno_l1 == B-Cells',
    'n_predicted_high': int(positive.sum()),
    'n_annotated_b': int(truth.sum()),
    'true_positive_overlap': tp, 'false_positive': fp,
    'false_negative': fn, 'true_negative': tn,
    'precision': precision, 'recall': recall,
    'dice_f1': 2 * tp / (2 * tp + fp + fn),
    'jaccard_iou': tp / (tp + fp + fn),
    'displayed_metrics': ['precision', 'recall'],
    'scope': 'same 2500 sampled cells in the displayed donor; exact cell-ID label overlap',
    'spatial_tolerance_or_nearest_neighbor_matching': False,
    'threshold_changed_or_optimized': False,
}


def panel_g(fig, rect, letter='G'):
    v7.panel_g(fig, rect, letter)
    x, y, w, h = rect
    width = (w-v7.G_GAP)/2
    for i, label in enumerate([
        f'Precision: {precision:.1%}',
        f'Recall: {recall:.1%}',
    ]):
        v.text_at(fig, x+i*(width+v7.G_GAP)+width/2, y+h-.35,
                  label, ha='center', fontsize=12, fontweight='bold')


def export(fig, name):
    for ext in ['png', 'svg']:
        fig.savefig(OUT / f'{name}.{ext}', dpi=300, facecolor='white')


def main():
    fig = plt.figure(figsize=(v3.W, v3.H), facecolor='white')
    panels = [
        (v.panel_a, (.15, .13, 2.50, 2.55)),
        (v6.panel_b, (2.76, .13, 5.70, 2.55)),
        (v3.panel_c, (8.57, .13, 5.78, 2.55)),
        (v3.panel_d, (.15, 2.79, 5.45, 3.50)),
        (v4.panel_e, (5.695, 2.79, 4.30, 3.50)),
        (v7.panel_f, (10.09, 2.79, 4.26, 3.50)),
        (panel_g, (.15, 6.40, 6.72, 3.95)),
        (v3.panel_h, (7.00, 6.40, 7.35, 3.95)),
    ]
    prior_path = BASE / 'figure_layout_v7/LAYOUT_AUDIT.json'
    prior = json.loads(prior_path.read_text())
    rectangles = {draw.__name__: list(rect) for draw, rect in panels}
    assert rectangles == prior['panel_rectangles_inches_from_top_left']
    for draw, rect in panels:
        draw(fig, rect)
    export(fig, 'cd19_story_three_row_v8')
    plt.close(fig)
    fig = plt.figure(figsize=(6.82, 4.05), facecolor='white')
    panel_g(fig, (.05, .05, 6.72, 3.95))
    export(fig, 'panels/G_spatial_overlap_metrics')
    plt.close(fig)

    save(OUT / 'G_overlap_metrics.json', metrics)
    pd.DataFrame({
        'cell_id': v.spatial_md.cell_id.astype(str),
        'predicted_cd19': v.spatial_prediction,
        'predicted_high': positive,
        'annotated_b_cell': truth,
        'overlap': truth & positive,
    }).to_csv(OUT / 'G_cell_overlap.csv', index=False)
    all_sources = {**v.input_hashes, **metric_hashes, str(v6.SCORE_SOURCE): v6.score_hash}
    assert all(sha(path) == value for path, value in all_sources.items())
    save(OUT / 'LAYOUT_AUDIT.json', {
        'time_utc': now(), 'script_sha256': sha(__file__),
        'source_sha256': all_sources,
        'base_layout_audit_sha256': sha(prior_path),
        'base_script_sha256': {str(BASE / f'scripts/assemble_cd19_layout_v{i}.py'):
                              sha(BASE / f'scripts/assemble_cd19_layout_v{i}.py') for i in range(1, 8)},
        'canvas_inches': [v3.W, v3.H],
        'panel_rectangles_inches_from_top_left': rectangles,
        'G_metrics': metrics,
        'other_panels_unchanged_from_v7': True,
        'all_input_files_unchanged': True,
    })
    (OUT / 'README_ZH.md').write_text(f'''# CD19 story v8

主图：cd19_story_three_row_v8.png；同名 SVG 为矢量版。
G 单图：panels/G_spatial_overlap_metrics.png 和 .svg。

在 G 的两张空间图下方增加两个标签重合指标：

- 左侧 Precision：209/295 = {precision:.1%}。预测 CD19 > 1.4 的细胞中，作者注释 B 细胞所占比例。
- 右侧 Recall：209/271 = {recall:.1%}。作者注释 B 细胞中，被预测 CD19 > 1.4 覆盖的比例。

计算限定为当前展示的同一位 donor（{v.spatial['DONOR']}）的 2,500 个抽样细胞，按同一 cell_id 逐细胞比较预测标签与作者标签；不做空间距离容差匹配。沿用固定阈值 1.4。这里的参考标签是作者注释 B 细胞。

混淆矩阵：TP = {tp}、FP = {fp}、FN = {fn}、TN = {tn}。Dice/F1 = {metrics['dice_f1']:.6f}，Jaccard/IoU = {metrics['jaccard_iou']:.6f}，一并保存于 G_overlap_metrics.json；图上展示 Precision 和 Recall，分别对应两张图的细胞集合。G_cell_overlap.csv 保存逐细胞计算结果。

G 的图形、颜色和标题间距沿用 v7，F 横轴仍为 CD19。其余 panels 不变。

复现：Exploration0909/scripts/assemble_cd19_layout_v8.py。所有输出仅写入 Exploration0909，输入哈希见 LAYOUT_AUDIT.json。
''')
    print(json.dumps(metrics, indent=2), flush=True)
    print(OUT / 'cd19_story_three_row_v8.png', flush=True)


if __name__ == '__main__':
    main()
