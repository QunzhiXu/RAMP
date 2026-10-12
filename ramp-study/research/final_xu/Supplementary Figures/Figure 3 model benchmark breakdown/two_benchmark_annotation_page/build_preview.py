"""One landscape page: two benchmarks, three models and two annotation metrics."""
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
import numpy as np
import pandas as pd
import fitz

OUT = HERE / 'output'
(OUT / 'pdf').mkdir(parents=True, exist_ok=True)
WIDTH, HEIGHT = 297, 210
INK, GREEN, GRAY = '#223037', '#008568', '#78868E'
METHODS = ['ramp', 'unaligned', 'harmony_direct', 'cycombine', 'cytonorm',
           'fastmnn', 'maxfuse', 'clipped_zscore', 'minmax', 'percentile']
METHOD_NAMES = ['RAMP', 'Unaligned', 'Harmony direct', 'cyCombine', 'CytoNorm',
                'fastMNN', 'MaxFuse', 'Clipped z-score', 'Min-max', 'Percentile']
TASKS = ['bone_marrow', 'tnbc']
TASK_NAMES = ['Bone marrow', 'TNBC']
MODELS = ['linear', 'knn', 'mlp']
MODEL_NAMES = ['Linear', 'kNN', 'Neural network']
METRICS = ['balanced_accuracy', 'macro_f1']
METRIC_NAMES = ['Balanced accuracy ↑', 'Macro-F1 ↑']

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
assert len(data) == 120
keys = ['task_type', 'task', 'family', 'metric', 'method']
assert not data.duplicated(keys).any()
assert set(data.method) == set(METHODS) and set(data.family) == set(MODELS)
assert data.groupby(['task', 'family', 'metric']).size().eq(10).all()
assert np.isfinite(data.score).all() and data.score.between(.2, 1).all()
reference = pd.read_json(release)
comparison = data.merge(reference, on=keys, validate='one_to_one', suffixes=('_plot', '_release'))
assert len(comparison) == 120
np.testing.assert_allclose(comparison.score_plot, comparison.score_release, atol=1e-14, rtol=0)

plt.rcParams.update({'font.family': 'Arial', 'font.weight': 'bold', 'font.size': 8,
                     'text.color': INK, 'xtick.color': INK, 'ytick.color': INK,
                     'pdf.fonttype': 42, 'svg.fonttype': 'none'})
fig = plt.figure(figsize=(WIDTH / 25.4, HEIGHT / 25.4), dpi=160, facecolor='white')
def text(x, y, label, **kw):
    return fig.text(x / WIDTH, 1 - y / HEIGHT, label, va='top', **kw)
text(9, 8, 'Cell-type annotation', fontsize=15)
text(9, 18, '2 benchmarks  |  3 models  |  10 methods', fontsize=9)
text(288, 18, 'Model-specific performance', fontsize=9, ha='right')
records, layout = [], []
for row, (task, task_name) in enumerate(zip(TASKS, TASK_NAMES)):
    top = 33 if row == 0 else 122
    text(9, top, f"{'AB'[row]}  {task_name}", fontsize=11.3)
    for model_column, (model, model_name) in enumerate(zip(MODELS, MODEL_NAMES)):
        group_x = 36 + 87 * model_column
        text(group_x + 39, top, model_name, fontsize=11, ha='center')
        for metric_column, (metric, metric_name) in enumerate(zip(METRICS, METRIC_NAMES)):
            x, y, w, h = group_x + 42.5 * metric_column, top + 18, 35.5, 52
            text(x + w / 2, top + 10, metric_name, fontsize=8.4, ha='center')
            ax = fig.add_axes([x / WIDTH, 1 - (y + h) / HEIGHT, w / WIDTH, h / HEIGHT])
            values = data[data.task.eq(task) & data.family.eq(model) & data.metric.eq(metric)].set_index('method').loc[METHODS, 'score']
            ax.axhspan(-.49, .49, color=GREEN, alpha=.07, lw=0, zorder=0)
            for i, (method, value) in enumerate(values.items()):
                ax.scatter(value, i, s=30 if method == 'ramp' else 23,
                           color=GREEN if method == 'ramp' else GRAY,
                           edgecolor='white', linewidth=.45, zorder=4)
                records.append(dict(task=task, family=model, metric=metric,
                                    method=method, score=float(value), row=i))
            ax.set_xlim(.2, 1)
            ax.set_ylim(9.6, -.6)
            ticks = [.2, .4, .6, .8, 1]
            ax.set_xticks(ticks, [f'{t:g}' for t in ticks], fontsize=7.8)
            first = model_column == 0 and metric_column == 0
            ax.set_yticks(range(10), METHOD_NAMES if first else [''] * 10, fontsize=8.2)
            if first:
                ax.get_yticklabels()[0].set_color(GREEN)
            ax.tick_params(axis='y', length=0, pad=5)
            ax.tick_params(axis='x', length=2.5, pad=3, color='#869398', width=.6)
            ax.spines[['top', 'right', 'left']].set_visible(False)
            ax.spines['bottom'].set_color('#869398')
            ax.spines['bottom'].set_linewidth(.6)
            ax.grid(axis='x', color='#E4E9EC', linewidth=.5)
            ax.grid(axis='y', color='#EEF1F3', linewidth=.4)
            ax.set_axisbelow(True)
            layout.append(dict(task=task, model=model, metric=metric, axes_mm=[x, y, w, h],
                               score_limits=[.2, 1], points=10))
text(148.5, 204, '↑ Higher is better', fontsize=8, ha='center')

# Check actual visible text, including method names and on-axis ticks.
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
objects = list(fig.texts)
for ax in fig.axes:
    objects.extend(ax.get_xticklabels())
    objects.extend(ax.get_yticklabels())
for obj in objects:
    if not obj.get_visible() or not obj.get_text():
        continue
    b = obj.get_window_extent(renderer)
    assert fig.bbox.contains(b.x0, b.y0) and fig.bbox.contains(b.x1, b.y1), obj.get_text()

stem = 'annotation_bone_marrow_tnbc_three_models'
for ext in ['png', 'svg', 'pdf']:
    target = (OUT / 'pdf' if ext == 'pdf' else OUT) / f'{stem}.{ext}'
    fig.savefig(target, dpi=300, facecolor='white')
plt.close(fig)
plotted = pd.DataFrame(records)
plotted.to_csv(HERE / 'plotted_values.csv', index=False)
matched = plotted.merge(data, on=['task', 'family', 'metric', 'method'], validate='one_to_one')
np.testing.assert_array_equal(matched.score_x, matched.score_y)
assert len(matched) == 120
assert all(sha(p) == expected for p, expected in sources.items())

doc = fitz.open(OUT / 'pdf' / f'{stem}.pdf')
assert len(doc) == 1
page = doc[0]
np.testing.assert_allclose([page.rect.width, page.rect.height], np.array([297, 210]) * 72 / 25.4, atol=.01, rtol=0)
assert all(page.rect.contains(fitz.Rect(*word[:4])) for word in page.get_text('words'))
qa = HERE / 'visual_qa'
qa.mkdir(exist_ok=True)
page.get_pixmap(dpi=160).save(qa / f'{stem}_pdf.png')
manifest = dict(status='independent_full_page_preview', benchmarks=TASKS,
                models=MODELS, methods=METHODS, metrics=METRICS, scores=120, axes=layout,
                main_figure_modified=False, formal_supplement_modified=False, no_refitting=True,
                all_scores_match_frozen_release=True, all_points_visible=True,
                all_text_inside_page=True, common_axes_across_benchmarks_and_models=True,
                figure_size_mm=[WIDTH, HEIGHT], source_sha256=sources, script_sha256=sha(__file__),
                output_sha256={str(p.relative_to(HERE)): sha(p) for p in OUT.rglob('*') if p.is_file()})
(HERE / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
print('One A4 landscape preview: 2 benchmarks, 3 models, 2 metrics, all 120 scores verified.')
