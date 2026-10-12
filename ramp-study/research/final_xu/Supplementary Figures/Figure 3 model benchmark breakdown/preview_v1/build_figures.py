"""S10/S11 previews: frozen Figure 3 endpoints without model/task averaging."""
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
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

OUT = HERE / 'output'
(OUT / 'pdf').mkdir(parents=True, exist_ok=True)
WIDTH, HEIGHT = 210, 297
INK, GREEN = '#202B30', '#008568'
MODELS = ['linear', 'knn', 'mlp']
MODEL_NAMES = ['Linear', 'kNN', 'Neural network']
METHODS = ['ramp', 'unaligned', 'harmony_direct', 'cycombine', 'cytonorm',
           'fastmnn', 'maxfuse', 'clipped_zscore', 'minmax', 'percentile']
METHOD_NAMES = ['RAMP', 'Unaligned', 'Harmony direct', 'cyCombine', 'CytoNorm',
                'fastMNN', 'MaxFuse', 'Clipped z-score', 'Min-max', 'Percentile']
TASKS = ['bone_marrow', 'tnbc', 'melanoma', 'lung_cancer', 'lung_story', 'melanoma_story', 'hnscc_story']
TASK_LABELS = {'bone_marrow': 'Bone\nmarrow', 'tnbc': 'TNBC', 'melanoma': 'Melan.\n1',
               'lung_cancer': 'Lung\nbench.', 'lung_story': 'Lung\nHLA-ABC',
               'melanoma_story': 'Melan.\n2', 'hnscc_story': 'HNSCC'}
METRICS = {
    'balanced_accuracy': dict(title='Balanced accuracy', direction='higher', limits=[0, 1], ticks=[0, .25, .5, .75, 1]),
    'macro_f1': dict(title='Macro-F1', direction='higher', limits=[0, 1], ticks=[0, .25, .5, .75, 1]),
    'effective_spearman': dict(title='Spearman ρ', direction='higher', limits=[0, .5], ticks=[0, .1, .2, .3, .4, .5]),
    'mae': dict(title='Relative-expression MAE', direction='lower', limits=[.18, .45], ticks=[.20, .25, .30, .35, .40, .45]),
}
SETTINGS = {
    'S10': dict(task_type='J1', main_panel='B', title='Cell-type annotation',
                metrics=['balanced_accuracy', 'macro_f1'], tasks=TASKS,
                stem='supplement_s10_annotation_by_model_benchmark'),
    'S11': dict(task_type='J2', main_panel='C', title='Missing-marker prediction',
                metrics=['effective_spearman', 'mae'], tasks=[t for t in TASKS if t != 'melanoma'],
                stem='supplement_s11_missing_marker_by_model_benchmark'),
}
CMAP = LinearSegmentedColormap.from_list('performance', ['#F4F7FA', '#DDE9F0', '#9FC5D7', '#4F8CAA', '#195777', '#0B3048'])
plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'font.weight': 'bold',
                     'text.color': INK, 'axes.labelcolor': INK,
                     'xtick.color': INK, 'ytick.color': INK,
                     'pdf.fonttype': 42, 'svg.fonttype': 'none'})

def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

def load():
    pointer_path = PROJECT / 'CURRENT_FIGURE3_RESULTS.json'
    pointer = json.loads(pointer_path.read_text())
    source = Path(pointer['directory']) / 'primary_means.json'
    report = Path(pointer['directory']) / 'REPORT.json'
    assert sha(source) == pointer['primary_means_sha256']
    assert sha(report) == pointer['report_sha256']
    assert json.loads(report.read_text())['status'] == 'complete'
    raw = pd.read_json(source)
    inputs = {str(p): sha(p) for p in [pointer_path, source, report]}
    frames = {}
    for which, cfg in SETTINGS.items():
        path = PROJECT / 'Figure' / f"Panel {cfg['main_panel']}" / 'per_model_scores.csv'
        d = pd.read_csv(path, float_precision='round_trip')
        assert set(d.task) == set(cfg['tasks'])
        assert set(d.method) == set(METHODS) and set(d.family) == set(MODELS)
        assert set(d.metric) == set(cfg['metrics'])
        assert d.task_type.eq(cfg['task_type']).all()
        keys = ['task_type', 'task', 'family', 'metric', 'method']
        assert not d.duplicated(keys).any() and np.isfinite(d.score).all()
        assert len(d) == len(cfg['tasks']) * 3 * 2 * 10
        reference = raw[(raw.task_type == cfg['task_type']) & raw.task.isin(cfg['tasks']) & raw.metric.isin(cfg['metrics'])]
        matched = d.merge(reference, on=keys, suffixes=('_plot', '_release'), validate='one_to_one')
        assert len(matched) == len(d)
        np.testing.assert_allclose(matched.score_plot, matched.score_release, atol=1e-14, rtol=0)
        for metric in cfg['metrics']:
            lo, hi = METRICS[metric]['limits']
            assert d[d.metric.eq(metric)].score.between(lo, hi).all()
        inputs[str(path)] = sha(path)
        frames[which] = d
    return inputs, frames

def text(fig, x, y, s, **kwargs):
    return fig.text(x / WIDTH, 1 - y / HEIGHT, s, va='top', **kwargs)

def axes(fig, x, y, w, h):
    return fig.add_axes([x / WIDTH, 1 - (y + h) / HEIGHT, w / WIDTH, h / HEIGHT])

def annotation_color(rgb):
    rgb = np.asarray(rgb[:3])
    linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
    luminance = float(linear @ np.array([.2126, .7152, .0722]))
    black_contrast = (luminance + .05) / .05
    white_contrast = 1.05 / (luminance + .05)
    return ('white' if white_contrast > black_contrast else '#111111'), max(black_contrast, white_contrast)

def build(which, cfg, data):
    fig = plt.figure(figsize=(WIDTH / 25.4, HEIGHT / 25.4), dpi=160, facecolor='white')
    text(fig, 9, 8, f"{which}  {cfg['title']}", fontsize=14)
    text(fig, 9, 18, f"3 models  |  {len(cfg['tasks'])} benchmarks  |  10 methods", fontsize=8.7)
    text(fig, 200, 18, 'Darker = better', ha='right', fontsize=8)
    records, checks = [], []
    for row, (model, name) in enumerate(zip(MODELS, MODEL_NAMES)):
        top = 33 + row * 80
        text(fig, 9, top, name, fontsize=11)
        for col, metric in enumerate(cfg['metrics']):
            info = METRICS[metric]
            lo, hi = info['limits']
            norm = Normalize(lo, hi)
            cmap = CMAP if info['direction'] == 'higher' else CMAP.reversed()
            x, w = (33, 81) if col == 0 else (119, 81)
            letter = chr(ord('A') + row * 2 + col)
            arrow = '↑' if info['direction'] == 'higher' else '↓'
            text(fig, x, top + 10, f"{letter}  {info['title']} {arrow}", fontsize=9.1)
            ax = axes(fig, x, top + 20, w, 51)
            d = data[data.family.eq(model) & data.metric.eq(metric)]
            matrix = d.pivot(index='method', columns='task', values='score').loc[METHODS, cfg['tasks']]
            image = ax.imshow(matrix.to_numpy(), cmap=cmap, norm=norm, aspect='auto', interpolation='nearest')
            ax.set_xticks(range(len(cfg['tasks'])), [TASK_LABELS[t] for t in cfg['tasks']], fontsize=7.3)
            ax.set_yticks(range(10), METHOD_NAMES if col == 0 else [''] * 10, fontsize=7.9)
            ax.tick_params(axis='both', length=0)
            ax.tick_params(axis='x', pad=5)
            ax.tick_params(axis='y', pad=5)
            if col == 0:
                ax.get_yticklabels()[0].set_color(GREEN)
            ax.set_xticks(np.arange(-.5, len(cfg['tasks']), 1), minor=True)
            ax.set_yticks(np.arange(-.5, 10, 1), minor=True)
            ax.grid(which='minor', color='white', linewidth=.65)
            ax.tick_params(which='minor', bottom=False, left=False)
            ax.spines[:].set_visible(False)
            ax.add_patch(Rectangle((-.5, -.5), len(cfg['tasks']), 1, fill=False,
                                   edgecolor=GREEN, linewidth=1.05, clip_on=False, zorder=4))
            for i, method in enumerate(METHODS):
                for j, task in enumerate(cfg['tasks']):
                    value = float(matrix.loc[method, task])
                    color, contrast = annotation_color(cmap(norm(value)))
                    obj = ax.text(j, i, f'{value:.3f}', ha='center', va='center',
                                  fontsize=7.6, color=color, weight='bold', zorder=5)
                    records.append(dict(figure=which, panel=letter, family=model, task=task,
                                        method=method, metric=metric, score=value, displayed=f'{value:.3f}'))
                    checks.append((obj, ax, contrast))
            if row == 2:
                cax = axes(fig, x, 284, w, 2.7)
                cb = fig.colorbar(image, cax=cax, orientation='horizontal', ticks=info['ticks'])
                cb.outline.set_visible(False)
                cb.ax.tick_params(labelsize=7.3, length=2, pad=3, width=.5)
                cb.ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    page = fig.bbox
    for obj in fig.texts:
        b = obj.get_window_extent(renderer)
        assert page.contains(b.x0, b.y0) and page.contains(b.x1, b.y1), obj.get_text()
    for obj, ax, contrast in checks:
        b = obj.get_window_extent(renderer)
        assert ax.bbox.contains(b.x0, b.y0) and ax.bbox.contains(b.x1, b.y1)
        assert contrast >= 4.5
    for ext in ['png', 'svg', 'pdf']:
        target = (OUT / 'pdf' if ext == 'pdf' else OUT) / f"{cfg['stem']}.{ext}"
        fig.savefig(target, dpi=300, facecolor='white',
                    metadata={'Creator': 'RAMP supplementary figures'} if ext == 'pdf' else None)
    plt.close(fig)
    frame = pd.DataFrame(records)
    frame.to_csv(HERE / f'{which}_plotted_values.csv', index=False)
    return dict(figure=which, stem=cfg['stem'], cells=len(records), tasks=cfg['tasks'],
                models=MODELS, metrics=cfg['metrics'],
                all_cell_annotations_inside_heatmaps=True, minimum_text_contrast=min(t[2] for t in checks))

if __name__ == '__main__':
    inputs, frames = load()
    results = [build(which, cfg, frames[which]) for which, cfg in SETTINGS.items()]
    assert sum(r['cells'] for r in results) == 780
    assert all(sha(p) == expected for p, expected in inputs.items())
    manifest = dict(status='independent_preview', formal_supplement_modified=False,
                    main_figure_modified=False, figure_size_mm=[WIDTH, HEIGHT],
                    source_sha256=inputs, source_inputs_unchanged=True,
                    script_sha256=sha(__file__), figures=results, metric_display=METRICS,
                    no_model_averaging=True, no_benchmark_averaging=True, no_refitting=True,
                    methods=METHODS, model_labels=dict(zip(MODELS, MODEL_NAMES)),
                    output_sha256={str(p.relative_to(HERE)): sha(p) for p in OUT.rglob('*') if p.is_file()})
    (HERE / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('S10/S11 rendered with all 780 model-specific benchmark scores.')
