"""Native A4 portrait layout within the supplement's 18-mm page margins."""
from pathlib import Path
import hashlib
import json
import os

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PROJECT = ROOT / 'cross_platform_trainability_v2_20260919'
os.environ.setdefault('MPLCONFIGDIR', str(HERE / '.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import fitz

OUT = HERE / 'output'
(OUT / 'pdf').mkdir(parents=True, exist_ok=True)
WIDTH, HEIGHT, MARGIN = 210, 297, 18
INK, GREEN, GRAY = '#223037', '#008568', '#78868E'
METHODS = ['ramp', 'unaligned', 'harmony_direct', 'cycombine', 'cytonorm',
           'fastmnn', 'maxfuse', 'clipped_zscore', 'minmax', 'percentile']
METHOD_NAMES = ['RAMP', 'Unaligned', 'Harmony direct', 'cyCombine', 'CytoNorm',
                'fastMNN', 'MaxFuse', 'Clipped z-score', 'Min-max', 'Percentile']
TASKS = ['bone_marrow', 'tnbc', 'melanoma', 'lung_cancer']
TASK_NAMES = ['Bone marrow', 'TNBC', 'Melanoma 1', 'Lung benchmark']
MODELS = ['linear', 'knn', 'mlp']
MODEL_NAMES = ['Linear', 'kNN', 'Neural network']
METRICS = ['balanced_accuracy', 'macro_f1']

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

source = PROJECT / 'Figure' / 'Panel B' / 'per_model_scores.csv'
pointer_path = PROJECT / 'CURRENT_FIGURE3_RESULTS.json'
pointer = json.loads(pointer_path.read_text())
release = Path(pointer['directory']) / 'primary_means.json'
assert sha(release) == pointer['primary_means_sha256']
sources = {str(p): sha(p) for p in [source, pointer_path, release]}
data = pd.read_csv(source, float_precision='round_trip')
data = data[data.task.isin(TASKS) & data.metric.isin(METRICS)].copy()
keys = ['task_type', 'task', 'family', 'metric', 'method']
assert len(data) == 240 and not data.duplicated(keys).any()
assert data.groupby(['task', 'family', 'metric']).size().eq(10).all()
assert data.score.between(.2, 1).all()
reference = pd.read_json(release)
matched = data.merge(reference, on=keys, validate='one_to_one', suffixes=('_plot', '_release'))
assert len(matched) == 240
np.testing.assert_allclose(matched.score_plot, matched.score_release, atol=1e-14, rtol=0)

plt.rcParams.update({'font.family': 'Arial', 'font.weight': 'bold', 'font.size': 8,
                     'text.color': INK, 'xtick.color': INK, 'ytick.color': INK,
                     'pdf.fonttype': 42, 'svg.fonttype': 'none'})
fig = plt.figure(figsize=(WIDTH / 25.4, HEIGHT / 25.4), dpi=160, facecolor='white')
def text(x, y, label, **kw):
    return fig.text(x / WIDTH, 1 - y / HEIGHT, label, va='top', **kw)
text(18, 18, 'Cell-type annotation', fontsize=14)
text(18, 27, '4 benchmarks  |  3 models  |  10 methods', fontsize=8.7)
text(192, 27, '↑ Higher is better', fontsize=7.8, ha='right')
legend = fig.legend(handles=[
    Line2D([], [], ls='', marker='o', ms=4.3, color=GRAY, label='Balanced accuracy'),
    Line2D([], [], ls='', marker='^', ms=4.7, markerfacecolor='white', markeredgecolor=GRAY,
           markeredgewidth=.85, label='Macro-F1')],
    loc='upper center', bbox_to_anchor=(.5, 1 - 35.5 / HEIGHT), ncol=2,
    frameon=False, prop={'size': 8.4, 'weight': 'bold'}, handletextpad=.5,
    columnspacing=2.1, borderaxespad=0)
for col, name in enumerate(MODEL_NAMES):
    text(67 + 51 * col, 46, name, fontsize=10, ha='center')

records, layout = [], []
for row, (task, name) in enumerate(zip(TASKS, TASK_NAMES)):
    top = 63 + 56 * row
    text(18, top - 7, f"{'ABCD'[row]}  {name}", fontsize=10.5)
    for col, model in enumerate(MODELS):
        x, y, w, h = 45 + 51 * col, top, 44, 41
        ax = fig.add_axes([x / WIDTH, 1 - (y + h) / HEIGHT, w / WIDTH, h / HEIGHT])
        ax.axhspan(-.5, .5, color=GREEN, alpha=.065, lw=0, zorder=0)
        for metric, offset, marker in [('balanced_accuracy', -.22, 'o'), ('macro_f1', .22, '^')]:
            values = data[data.task.eq(task) & data.family.eq(model) & data.metric.eq(metric)].set_index('method').loc[METHODS, 'score']
            for i, (method, value) in enumerate(values.items()):
                color = GREEN if method == 'ramp' else GRAY
                size = (22 if marker == 'o' else 26) if method == 'ramp' else (17 if marker == 'o' else 20)
                ax.scatter(value, i + offset, s=size, marker=marker,
                           facecolors=color if marker == 'o' else 'white',
                           edgecolors='white' if marker == 'o' else color,
                           linewidths=.4 if marker == 'o' else .85, zorder=4)
                records.append(dict(task=task, family=model, metric=metric, method=method,
                                    score=float(value), y_position=i + offset, marker=marker))
        ax.set_xlim(.2, 1)
        ax.set_ylim(9.65, -.65)
        ticks = [.2, .4, .6, .8, 1]
        ax.set_xticks(ticks, [f'{t:g}' for t in ticks], fontsize=7.8)
        ax.set_yticks(range(10), METHOD_NAMES if col == 0 else [''] * 10, fontsize=8.2)
        if col == 0:
            ax.get_yticklabels()[0].set_color(GREEN)
        ax.tick_params(axis='y', length=0, pad=5)
        ax.tick_params(axis='x', length=2.5, pad=3, color='#869398', width=.6)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.spines['bottom'].set_color('#869398')
        ax.spines['bottom'].set_linewidth(.6)
        ax.grid(axis='x', color='#E4E9EC', linewidth=.5)
        ax.grid(axis='y', color='#EEF1F3', linewidth=.4)
        ax.set_axisbelow(True)
        layout.append(dict(task=task, model=model, axes_mm=[x, y, w, h],
                           score_limits=[.2, 1], points=20))

fig.canvas.draw()
renderer = fig.canvas.get_renderer()
objects = list(fig.texts) + list(legend.get_texts())
for ax in fig.axes:
    objects.extend(ax.get_xticklabels())
    objects.extend(ax.get_yticklabels())
text_boxes = []
for obj in objects:
    if not obj.get_visible() or not obj.get_text():
        continue
    b = obj.get_window_extent(renderer)
    # All labels lie inside the 174 x 261 mm text block, allowing 0.5 mm
    # for edge-aligned tick labels at the far-right plot boundary.
    mm = np.array(b.extents) / fig.dpi * 25.4
    assert mm[0] >= 17.5 and mm[1] >= 17.5 and mm[2] <= 192.5 and mm[3] <= 279.5, (obj.get_text(), mm)
    text_boxes.append(dict(text=obj.get_text(), bounds_mm=mm.tolist()))

stem = 'annotation_four_benchmarks_a4_portrait'
for ext in ['png', 'svg', 'pdf']:
    target = (OUT / 'pdf' if ext == 'pdf' else OUT) / f'{stem}.{ext}'
    fig.savefig(target, dpi=300, facecolor='white')
plt.close(fig)
plotted = pd.DataFrame(records)
plotted.to_csv(HERE / 'plotted_values.csv', index=False)
check = plotted.merge(data, on=['task', 'family', 'metric', 'method'], validate='one_to_one')
np.testing.assert_array_equal(check.score_x, check.score_y)
assert len(check) == 240
assert all(sha(p) == expected for p, expected in sources.items())
doc = fitz.open(OUT / 'pdf' / f'{stem}.pdf')
assert len(doc) == 1
page = doc[0]
np.testing.assert_allclose([page.rect.width, page.rect.height], np.array([210, 297]) * 72 / 25.4, atol=.01, rtol=0)
assert all(page.rect.contains(fitz.Rect(*word[:4])) for word in page.get_text('words'))
qa = HERE / 'visual_qa'
qa.mkdir(exist_ok=True)
page.get_pixmap(dpi=160).save(qa / f'{stem}_pdf.png')
manifest = dict(status='independent_native_portrait_preview', benchmarks=TASKS,
                models=MODELS, methods=METHODS, metrics=METRICS, scores=240, axes=layout,
                main_figure_modified=False, formal_supplement_modified=False, no_refitting=True,
                all_scores_match_frozen_release=True, all_points_visible=True,
                common_axes_across_benchmarks_models_metrics=True,
                metrics_encoded_by_shape={'balanced_accuracy': 'filled circle, upper', 'macro_f1': 'open triangle, lower'},
                page_size_mm=[WIDTH, HEIGHT], content_width_mm=174, page_margin_mm=18,
                method_font_pt=8.2, tick_font_pt=7.8, rendered_directly_at_final_size=True,
                text_boxes=text_boxes, source_sha256=sources, script_sha256=sha(__file__),
                output_sha256={str(p.relative_to(HERE)): sha(p) for p in OUT.rglob('*') if p.is_file()})
(HERE / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
print('Native A4 portrait preview generated; 240 scores verified; no raster-panel merging.')
