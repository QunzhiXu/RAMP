"""Separate score scales: one A4 page per missing-marker endpoint."""
from pathlib import Path
import argparse
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

parser = argparse.ArgumentParser()
parser.add_argument('--metric', choices=['effective_spearman', 'mae'], required=True)
METRIC = parser.parse_args().metric
FIGURE = 'S12' if METRIC == 'effective_spearman' else 'S13'
TITLE = 'Missing-marker Spearman' if FIGURE == 'S12' else 'Missing-marker MAE'
LIMITS = (.05, .55) if FIGURE == 'S12' else (.18, .46)
TICKS = [.1, .2, .3, .4, .5] if FIGURE == 'S12' else [.2, .3, .4]
DIRECTION = '↑ Higher is better' if FIGURE == 'S12' else '↓ Lower is better'
OUT = HERE / 'output'
(OUT / 'pdf').mkdir(parents=True, exist_ok=True)
WIDTH, HEIGHT = 210, 297
INK, GREEN, GRAY = '#223037', '#008568', '#78868E'
METHODS = ['ramp', 'unaligned', 'harmony_direct', 'cycombine', 'cytonorm',
           'fastmnn', 'maxfuse', 'clipped_zscore', 'minmax', 'percentile']
METHOD_NAMES = ['RAMP', 'Unaligned', 'Harmony direct', 'cyCombine', 'CytoNorm',
                'fastMNN', 'MaxFuse', 'Clipped z-score', 'Min-max', 'Percentile']
TASKS = ['bone_marrow', 'tnbc', 'lung_cancer', 'lung_story', 'melanoma_story', 'hnscc_story']
TASK_NAMES = ['Bone marrow', 'TNBC', 'Lung cancer', 'Lung HLA-ABC', 'Melanoma 2', 'HNSCC']
MODELS, MODEL_NAMES = ['linear', 'knn', 'mlp'], ['Linear', 'kNN', 'Neural network']

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

source = PROJECT / 'Figure' / 'Panel C' / 'per_model_scores.csv'
pointer_path = PROJECT / 'CURRENT_FIGURE3_RESULTS.json'
pointer = json.loads(pointer_path.read_text())
release = Path(pointer['directory']) / 'primary_means.json'
assert sha(release) == pointer['primary_means_sha256']
sources = {str(p): sha(p) for p in [source, pointer_path, release]}
data = pd.read_csv(source, float_precision='round_trip')
data = data[data.metric.eq(METRIC)].copy()
assert set(data.task) == set(TASKS)
keys = ['task_type', 'task', 'family', 'metric', 'method']
assert len(data) == 180 and not data.duplicated(keys).any()
assert data.groupby(['task', 'family']).size().eq(10).all()
assert data.score.between(*LIMITS).all()
reference = pd.read_json(release)
matched = data.merge(reference, on=keys, validate='one_to_one', suffixes=('_plot', '_release'))
assert len(matched) == 180
np.testing.assert_allclose(matched.score_plot, matched.score_release, atol=1e-14, rtol=0)

plt.rcParams.update({'font.family': 'Arial', 'font.weight': 'bold', 'font.size': 8,
                     'text.color': INK, 'xtick.color': INK, 'ytick.color': INK,
                     'pdf.fonttype': 42, 'svg.fonttype': 'none'})
fig = plt.figure(figsize=(WIDTH / 25.4, HEIGHT / 25.4), dpi=160, facecolor='white')
def text(x, y, label, **kw):
    return fig.text(x / WIDTH, 1 - y / HEIGHT, label, va='top', **kw)
text(18, 18, f'{FIGURE}  {TITLE}', fontsize=14)
text(18, 27, '6 benchmarks  |  3 models  |  10 methods', fontsize=8.7)
text(192, 27, DIRECTION, fontsize=7.8, ha='right')
for col, name in enumerate(MODEL_NAMES):
    text(67 + 51 * col, 35, name, fontsize=10, ha='center')
records, layout = [], []
for row, (task, name) in enumerate(zip(TASKS, TASK_NAMES)):
    top = 51 + 38 * row
    text(18, top - 7, f"{'ABCDEF'[row]} {name}", fontsize=9.0)
    for col, model in enumerate(MODELS):
        x, y, w, h = 45 + 51 * col, top, 44, 29.8
        ax = fig.add_axes([x / WIDTH, 1 - (y + h) / HEIGHT, w / WIDTH, h / HEIGHT])
        ax.axhspan(-.5, .5, color=GREEN, alpha=.065, lw=0, zorder=0)
        values = data[data.task.eq(task) & data.family.eq(model)].set_index('method').loc[METHODS, 'score']
        for i, (method, value) in enumerate(values.items()):
            ax.scatter(value, i, s=24 if method == 'ramp' else 18,
                       color=GREEN if method == 'ramp' else GRAY, edgecolor='white', linewidth=.4, zorder=4)
            records.append(dict(task=task, family=model, metric=METRIC, method=method, score=float(value)))
        ax.set_xlim(*LIMITS)
        ax.set_ylim(9.6, -.6)
        ax.set_xticks(TICKS, [f'{t:g}' for t in TICKS], fontsize=7.8)
        ax.set_yticks(range(10), METHOD_NAMES if col == 0 else [''] * 10, fontsize=8.2)
        if col == 0:
            ax.get_yticklabels()[0].set_color(GREEN)
        ax.tick_params(axis='y', length=0, pad=5)
        ax.tick_params(axis='x', length=2, pad=2, color='#869398', width=.6)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.spines['bottom'].set_color('#869398')
        ax.spines['bottom'].set_linewidth(.6)
        ax.grid(axis='x', color='#E4E9EC', linewidth=.5)
        ax.grid(axis='y', color='#EEF1F3', linewidth=.4)
        ax.set_axisbelow(True)
        layout.append(dict(task=task, model=model, axes_mm=[x, y, w, h], score_limits=LIMITS, points=10))
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
objects = list(fig.texts)
for ax in fig.axes:
    objects.extend(ax.get_xticklabels())
    objects.extend(ax.get_yticklabels())
for obj in objects:
    if obj.get_visible() and obj.get_text():
        b = np.array(obj.get_window_extent(renderer).extents) / fig.dpi * 25.4
        assert b[0] >= 17.5 and b[1] >= 17.5 and b[2] <= 192.5 and b[3] <= 279.5, (obj.get_text(), b)
stem = f'{FIGURE.lower()}_missing_marker_{"spearman" if FIGURE == "S12" else "mae"}'
for ext in ['png', 'svg', 'pdf']:
    target = (OUT / 'pdf' if ext == 'pdf' else OUT) / f'{stem}.{ext}'
    fig.savefig(target, dpi=300, facecolor='white')
plt.close(fig)
plotted = pd.DataFrame(records)
plotted.to_csv(HERE / f'{FIGURE}_plotted_values.csv', index=False)
check = plotted.merge(data, on=['task', 'family', 'metric', 'method'], validate='one_to_one')
np.testing.assert_array_equal(check.score_x, check.score_y)
assert len(check) == 180 and all(sha(p) == expected for p, expected in sources.items())
doc = fitz.open(OUT / 'pdf' / f'{stem}.pdf')
assert len(doc) == 1
page = doc[0]
np.testing.assert_allclose([page.rect.width, page.rect.height], np.array([210, 297]) * 72 / 25.4, atol=.01, rtol=0)
assert all(page.rect.contains(fitz.Rect(*word[:4])) for word in page.get_text('words'))
qa = HERE / 'visual_qa'
qa.mkdir(exist_ok=True)
page.get_pixmap(dpi=160).save(qa / f'{stem}_pdf.png')
manifest = dict(status='publication_candidate', figure=FIGURE, stem=stem, benchmarks=TASKS,
                models=MODELS, methods=METHODS, metrics=[METRIC], scores=180, axes=layout,
                all_scores_match_frozen_release=True, all_points_visible=True,
                common_axes_across_benchmarks_models=True, no_refitting=True,
                page_size_mm=[WIDTH, HEIGHT], content_width_mm=174, page_margin_mm=18,
                method_font_pt=8.2, tick_font_pt=7.8, rendered_directly_at_final_size=True,
                source_sha256=sources, script_sha256=sha(__file__),
                output_sha256={str(p.relative_to(HERE)): sha(p) for p in OUT.rglob('*') if p.is_file()})
(HERE / f'{FIGURE}_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(f'{FIGURE}: native A4 portrait, 180 scores verified.')
