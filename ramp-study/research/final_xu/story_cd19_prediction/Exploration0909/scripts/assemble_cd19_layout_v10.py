"""Assemble the CD3 Panel E with real CODEX tissue images in Panel G."""
from runtime import BASE, save, sha, now
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import assemble_cd19_layout_v9 as previous

v = previous.v
v8 = previous.previous
OUT = BASE / 'figure_layout_v10'
for path in [OUT, OUT / 'panels', OUT / 'data']:
    path.mkdir(parents=True, exist_ok=True)
REAL = BASE / 'real_spatial0913'
SOURCES = {}


def track(path):
    SOURCES[str(path)] = sha(path)
    return path


prior = json.loads(track(BASE / 'figure_layout_v9/LAYOUT_AUDIT.json').read_text())
example = prior['examples']['CD3']
scatter = pd.read_csv(track(BASE / 'figure_layout_v9/data/CD3_scatter_cells.csv.gz'))
assert len(scatter) == 2500 and scatter.cell_id.is_unique
assert abs(spearmanr(scatter.measured, scatter.predicted).statistic - example['spearman']) < 1e-12
previous.examples['CD3'] = {'cells': scatter, 'donor': example['selected_donor'],
                           'short': example['donor_short_label'], 'rho': example['spearman'],
                           'pearson': example['pearson']}
previous.checks['CD3'] = dict(example)
roi = json.loads(track(REAL / 'results/selected_roi.json').read_text())
grey_audit = json.loads(track(REAL / 'results/gray_background_audit.json').read_text())
assert grey_audit['status'] == 'PASS'
roi_cells = pd.read_csv(track(REAL / 'data/selected_roi_cells.csv.gz'))
truth = roi_cells.annotated_b.to_numpy(bool)
positive = roi_cells.predicted_high.to_numpy(bool)
tp = int((truth & positive).sum())
fp = int((~truth & positive).sum())
fn = int((truth & ~positive).sum())
assert (len(roi_cells), tp, fp, fn) == (330, 44, 7, 3)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
assert precision == roi['precision'] and recall == roi['recall']
assert int(positive.sum()) == roi['n_high'] and int(truth.sum()) == roi['n_b']
images = [np.asarray(Image.open(track(REAL / 'figures' / name)).convert('RGB')) for name in [
    'H14_ROI_predicted_CD19_masks_gray_background.png',
    'H14_ROI_annotated_B_masks_gray_background.png',
]]
assert all(image.shape == (1024, 1024, 3) for image in images)


def panel_g(fig, rect, letter='G'):
    x, y, w, h = rect
    v.title(fig, rect, letter, 'Predicted CD19 recovers spatial B-cell locations')
    gap = .10
    side = min((w - gap) / 2, h - .87)
    start = x + (w - 2 * side - gap) / 2
    titles = [f'Predicted CD19 > 1.4 (n = {int(roi["n_high"])})',
              f'Annotated B cells (n = {int(roi["n_b"])})']
    metrics = [f'Precision: {precision:.1%}', f'Recall: {recall:.1%}']
    for i, (image, title, metric) in enumerate(zip(images, titles, metrics)):
        ax = v.ax_at(fig, start + i * (side + gap), y + .51, side, side)
        ax.imshow(image, interpolation='nearest')
        ax.set_axis_off()
        ax.set_title(title, fontsize=10.5, fontweight='bold', pad=6)
        length = 50 / roi['pixel_size_um']
        xx, yy = image.shape[1] - length - 45, image.shape[0] - 55
        ax.plot([xx, xx + length], [yy, yy], color='white', lw=2.5, solid_capstyle='butt')
        ax.text(xx + length / 2, yy - 17, '50 µm', color='white', fontsize=9,
                fontweight='bold', ha='center', va='bottom')
        v.text_at(fig, start + i * (side + gap) + side / 2, y + h - .22,
                  metric, ha='center', fontsize=11.5, fontweight='bold')
        assert np.array_equal(np.asarray(ax.images[0].get_array()), image)
    return side


def export(fig, name):
    for ext in ['png', 'svg', 'pdf']:
        fig.savefig(OUT / f'{name}.{ext}', dpi=300, facecolor='white')


def main():
    plt.rcParams['pdf.fonttype'] = 42
    panels = [
        (v.panel_a, (.15, .13, 2.50, 2.55)),
        (v8.v6.panel_b, (2.76, .13, 5.70, 2.55)),
        (v8.v3.panel_c, (8.57, .13, 5.78, 2.55)),
        (v8.v3.panel_d, (.15, 2.79, 5.45, 3.50)),
        (previous.panel_e, (5.695, 2.79, 4.30, 3.50)),
        (v8.v7.panel_f, (10.09, 2.79, 4.26, 3.50)),
        (panel_g, (.15, 6.40, 6.72, 3.95)),
        (v8.v3.panel_h, (7.00, 6.40, 7.35, 3.95)),
    ]
    rectangles = {draw.__name__: list(rect) for draw, rect in panels}
    assert rectangles == prior['panel_rectangles_inches_from_top_left']
    fig = plt.figure(figsize=(14.5, 10.5), facecolor='white')
    for draw, rect in panels:
        draw(fig, rect)
    export(fig, 'cd19_story_three_row_v10')
    fig.savefig(OUT / 'third_row_preview.png', dpi=300, facecolor='white',
                bbox_inches=Bbox.from_extents(.10, .08, 14.42, 10.5 - 6.35))
    plt.close(fig)
    for draw, name, size in [
        (previous.panel_e, 'E_CD3_scatter_and_accuracy', (4.40, 3.60)),
        (panel_g, 'G_real_CODEX_spatial', (6.82, 4.05)),
    ]:
        fig = plt.figure(figsize=size, facecolor='white')
        draw(fig, (.05, .05, size[0] - .10, size[1] - .10))
        export(fig, f'panels/{name}')
        plt.close(fig)
    source_hashes = {**prior['source_sha256'], **SOURCES,
                     **grey_audit['source_sha256'],
                     str(BASE / 'scripts/assemble_cd19_layout_v9.py'): sha(BASE / 'scripts/assemble_cd19_layout_v9.py')}
    assert all(sha(path) == digest for path, digest in source_hashes.items())
    roi_cells.to_csv(OUT / 'data/G_ROI_cells.csv.gz', index=False)
    save(OUT / 'data/G_ROI_metrics.json', roi)
    save(OUT / 'LAYOUT_AUDIT.json', {
        'status': 'PASS', 'time_utc': now(), 'script_sha256': sha(__file__),
        'canvas_inches': [14.5, 10.5], 'panel_rectangles_inches_from_top_left': rectangles,
        'changed_panels_relative_to_v8': ['E', 'G'],
        'E': {'source': 'v9 CD3 scatter and compact four-metric summary',
              'example': example, 'summary': prior['lower_summary']},
        'G': {'source': 'Real H14 CODEX DAPI plus genuine Mesmer segmentation overlays',
              'source_roi': roi, 'dapi_context': 'neutral white-grey',
              'positive_overlay': 'green/cyan, identical to approved standalone images',
              'raster_arrays_identical_to_standalone_images': True,
              'image_cropped_further': False, 'image_aspect': 'equal',
              'scale_bar_um': 50, 'metrics_recomputed_from_same_cell_ID_membership': True,
              'full_donor_prediction_n_cells': 83384,
              'roi_metric_n_cells': 330, 'n_predicted_high': 51, 'n_annotated_b': 47,
              'tp': tp, 'fp': fp, 'fn': fn, 'precision': precision, 'recall': recall,
              'selection': 'Previously selected highest-F1 ROI among 1331 eligible H14 windows',
              'simulation_used': False},
        'other_panels_match_v8': True, 'all_input_files_unchanged': True,
        'no_model_or_UMAP_refit': True, 'source_sha256': source_hashes,
    })
    print(OUT / 'cd19_story_three_row_v10.png')
    print(f'PASS: E CD3 rho={example["spearman"]:.6f}; G n=330, precision={precision:.6f}, recall={recall:.6f}')


if __name__ == '__main__':
    main()
