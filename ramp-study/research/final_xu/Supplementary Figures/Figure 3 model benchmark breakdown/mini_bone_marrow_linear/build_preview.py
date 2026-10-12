"""One benchmark/model dot-plot preview, using frozen Figure 3 scores."""
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
WIDTH, HEIGHT = 210, 181
METHODS = ['ramp', 'unaligned', 'harmony_direct', 'cycombine', 'cytonorm',
           'fastmnn', 'maxfuse', 'clipped_zscore', 'minmax', 'percentile']
NAMES = ['RAMP', 'Unaligned', 'Harmony direct', 'cyCombine', 'CytoNorm',
         'fastMNN', 'MaxFuse', 'Clipped z-score', 'Min-max', 'Percentile']
INK, GREEN, GRAY = '#223037', '#008568', '#78868E'
SPECS = [
    ('B', 'balanced_accuracy', 'Balanced accuracy ↑', (.50, .99), [.5, .6, .7, .8, .9]),
    ('B', 'macro_f1', 'Macro-F1 ↑', (.50, .99), [.5, .6, .7, .8, .9]),
    ('C', 'effective_spearman', 'Spearman ρ ↑', (.12, .46), [.15, .25, .35, .45]),
    ('C', 'mae', 'Relative-expression MAE ↓', (.18, .40), [.20, .25, .30, .35, .40]),
]

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

sources, tables = {}, []
for panel in ['B', 'C']:
    source = PROJECT / 'Figure' / f'Panel {panel}' / 'per_model_scores.csv'
    sources[str(source)] = sha(source)
    data = pd.read_csv(source, float_precision='round_trip')
    tables.append(data[data.task.eq('bone_marrow') & data.family.eq('linear')])
data = pd.concat(tables, ignore_index=True)
assert len(data) == 40 and set(data.method) == set(METHODS)
assert data.groupby('metric').size().eq(10).all()
assert not data.duplicated(['metric', 'method']).any()
assert np.isfinite(data.score).all()
pointer_path = PROJECT / 'CURRENT_FIGURE3_RESULTS.json'
pointer = json.loads(pointer_path.read_text())
release = Path(pointer['directory']) / 'primary_means.json'
assert sha(release) == pointer['primary_means_sha256']
source_scores = pd.read_json(release)
reference = source_scores[source_scores.task.eq('bone_marrow') & source_scores.family.eq('linear')]
matched = data.merge(reference, on=['task', 'task_type', 'family', 'method', 'metric'],
                     validate='one_to_one', suffixes=('_plot', '_release'))
assert len(matched) == 40
np.testing.assert_allclose(matched.score_plot, matched.score_release, rtol=0, atol=1e-14)
sources.update({str(pointer_path): sha(pointer_path), str(release): sha(release)})

plt.rcParams.update({'font.family': 'Arial', 'font.weight': 'bold', 'font.size': 8.5,
                     'text.color': INK, 'xtick.color': INK, 'ytick.color': INK,
                     'pdf.fonttype': 42, 'svg.fonttype': 'none'})
fig = plt.figure(figsize=(WIDTH / 25.4, HEIGHT / 25.4), dpi=160, facecolor='white')
def text(x, y, label, **kw):
    return fig.text(x / WIDTH, 1 - y / HEIGHT, label, va='top', **kw)
text(9, 8, 'Bone marrow | Linear', fontsize=15)
text(200, 10, '10 methods', fontsize=9, ha='right')
text(9, 25, 'Cell-type annotation', fontsize=11)
text(9, 99, 'Missing-marker prediction', fontsize=11)
labels, rankings = [], []
for index, (panel, metric, title, bounds, ticks) in enumerate(SPECS):
    row, col = divmod(index, 2)
    x, y, w, h = (36 if col == 0 else 124), (43 if row == 0 else 117), 75, 44
    text(x, y - 9, title, fontsize=9.5)
    ax = fig.add_axes([x / WIDTH, 1 - (y + h) / HEIGHT, w / WIDTH, h / HEIGHT])
    values = data[data.metric.eq(metric)].set_index('method').loc[METHODS, 'score']
    assert values.between(*bounds).all()
    ranks = values.rank(ascending=metric == 'mae', method='min')
    ax.axhspan(-.48, .48, color=GREEN, alpha=.055, lw=0, zorder=0)
    for i, (method, value) in enumerate(values.items()):
        color = GREEN if method == 'ramp' else GRAY
        ax.scatter(value, i, s=37 if method == 'ramp' else 27, color=color,
                   edgecolor='white', linewidth=.45, zorder=3)
        annotation = ax.annotate(f'{value:.3f}', (value, i), xytext=(5, 0),
                                 textcoords='offset points', ha='left', va='center',
                                 fontsize=8, color=GREEN if method == 'ramp' else INK)
        labels.append((annotation, ax))
        rankings.append(dict(metric=metric, method=method, score=float(value), rank=int(ranks[method])))
    ax.set_xlim(*bounds)
    ax.set_ylim(9.65, -.65)
    ax.set_yticks(range(10), NAMES if col == 0 else [''] * 10, fontsize=8.3)
    if col == 0:
        ax.get_yticklabels()[0].set_color(GREEN)
    ax.set_xticks(ticks, [f'{v:g}' for v in ticks], fontsize=8)
    ax.tick_params(axis='y', length=0, pad=5)
    ax.tick_params(axis='x', length=2.5, pad=3, color='#869398', width=.6)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_color('#869398')
    ax.spines['bottom'].set_linewidth(.6)
    ax.grid(axis='x', color='#E4E9EC', lw=.6)
    ax.set_axisbelow(True)
text(105, 174, '↑ Higher is better     ↓ Lower is better', fontsize=8, ha='center')
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
for obj, ax in labels:
    box = obj.get_window_extent(renderer)
    assert ax.bbox.contains(box.x0, box.y0) and ax.bbox.contains(box.x1, box.y1), obj.get_text()
stem = 'bone_marrow_linear_dotplot'
for ext in ['png', 'svg', 'pdf']:
    target = (OUT / 'pdf' if ext == 'pdf' else OUT) / f'{stem}.{ext}'
    fig.savefig(target, dpi=300, facecolor='white')
plt.close(fig)
data.to_csv(HERE / 'plotted_values.csv', index=False)
pd.DataFrame(rankings).to_csv(HERE / 'scores_and_ranks.csv', index=False)
assert all(sha(path) == expected for path, expected in sources.items())
pdf = fitz.open(OUT / 'pdf' / f'{stem}.pdf')
assert len(pdf) == 1
page = pdf[0]
assert all(page.rect.contains(fitz.Rect(*word[:4])) for word in page.get_text('words'))
qa = HERE / 'visual_qa'
qa.mkdir(exist_ok=True)
page.get_pixmap(dpi=160).save(qa / f'{stem}_pdf.png')
manifest = dict(status='independent_mini_preview', benchmark='bone_marrow', model='linear',
                scores=40, main_figure_modified=False, formal_supplement_modified=False,
                no_refitting=True, all_values_match_frozen_release=True,
                all_annotations_inside_axes=True, pdf_text_inside_page=True,
                figure_size_mm=[WIDTH, HEIGHT], source_sha256=sources,
                script_sha256=sha(__file__),
                output_sha256={str(p.relative_to(HERE)): sha(p) for p in OUT.rglob('*') if p.is_file()})
(HERE / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(pd.DataFrame(rankings).query("method == 'ramp'").to_string(index=False))
