"""Three-row melanoma redesign using frozen results and display-only changes."""
from pathlib import Path
import os
import sys
import json
import hashlib
import ast

HERE = Path(__file__).resolve().parent
STORY = HERE.parent
sys.dont_write_bytecode = True
os.environ['MPLCONFIGDIR'] = str(HERE / '.mpl-cache')
os.environ['MPLBACKEND'] = 'Agg'
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
(HERE / 'panels').mkdir(exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, PathPatch
from matplotlib.path import Path as MplPath
from PIL import Image

W, H = 13.5, 10.5
INK = '#252E35'
SOURCES = {}
DRAW_AUDIT = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source(relative):
    path = STORY / relative
    SOURCES[str(path)] = sha(path)
    return path


def table(relative):
    return pd.read_csv(source(relative), low_memory=False)


def read_json(relative):
    return json.loads(source(relative).read_text())


ALIGN_XY = np.load(source('cache/ramp230_clean15_display_umap.npy'))
ALIGN_MD = table('cache/ramp230_clean15_display_metadata.csv.gz')
ALIGN_MANIFEST = read_json('analysis/alignment/output/manifest.json')
ALIGN_CACHE = read_json('cache/ramp230_clean15_display_manifest.json')
assert len(ALIGN_MD) == len(ALIGN_XY) == 40000
assert ALIGN_MD.dataset.value_counts().to_dict() == {'Moldoveanu IMC': 20000, 'Engelbrecht CITE-seq': 20000}
assert sha(source('analysis/alignment/output/ramp_aligned.npy')) == ALIGN_CACHE['alignment_values_sha256']
assert sha(source('analysis/alignment/output/cell_metadata.csv.gz')) == ALIGN_CACHE['alignment_metadata_sha256']
assert sha(source('inputs/harmonized_display_labels.csv.gz')) == ALIGN_CACHE['display_labels_sha256']
SURVIVAL = table('inputs/observed_patient_level_features.csv')
SURVIVAL_RESULTS = table('inputs/observed_survival_results.csv')
PATIENT_METRICS = table('analysis/all_cd8_prediction/patient_metrics.csv')
PROJECTION = table('analysis/all_cd8_prediction/citeseq_projection/engelbrecht_all_cd8_projection.csv.gz')
EFFECTS = table('analysis/all_cd8_prediction/biological_interpretation/all_heldout_adt_effects_by_patient.csv')
MODULES = table('analysis/all_cd8_prediction/biological_interpretation/candidate_biomarker_modules.csv')
SPATIAL_INFO = read_json('analysis/all_cd8_prediction/spatial_figure/manifest.json')['recommended_top20']
A_SOURCE = source('inputs/panel_a_spatial_state_definition_30um.png')
PAIR_SOURCE = source('analysis/all_cd8_prediction/spatial_figure/22RD_allCD8_fixed30um_global_top20_recommended.png')
WORKFLOW_SCRIPT = source('Panel D/draw_panel_D.py')
WORKFLOW_ICON = source('Panel D/assets/ramp_publication_icon.png')

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9.5,
                     'svg.fonttype': 'none', 'text.color': INK,
                     'axes.labelcolor': '#414C56', 'xtick.color': '#414C56',
                     'ytick.color': '#58636C', 'axes.titlecolor': INK})


def axis(fig, x, y, w, h):
    fw, fh = fig.get_size_inches()
    return fig.add_axes([x/fw, 1-(y+h)/fh, w/fw, h/fh])


def text(fig, x, y, label, **kwargs):
    fw, fh = fig.get_size_inches()
    return fig.text(x/fw, 1-y/fh, label, va='top', **kwargs)


def title(fig, rect, letter, label):
    x, y, w, h = rect
    text(fig, x+.02, y+.02, f'{letter}.  {label}', fontsize=13.5, fontweight='bold')


def legend(ax, names, colors, ncol, fontsize=8.0, y=-.04):
    handles = [Line2D([], [], marker='o', linestyle='', color=c, markersize=4.6)
               for c in colors]
    return ax.legend(handles, names, loc='upper center', bbox_to_anchor=(.5, y),
                     ncol=ncol, frameon=False, fontsize=fontsize, handletextpad=.30,
                     columnspacing=.72, labelspacing=.28, borderpad=0, borderaxespad=0)


def format_umap(ax, xy):
    low, high = xy.min(axis=0), xy.max(axis=0)
    pad = .014*(high-low)
    ax.set(xlim=(low[0]-pad[0], high[0]+pad[0]),
           ylim=(low[1]-pad[1], high[1]+pad[1]))
    ax.set_aspect('equal', adjustable='box')
    ax.set_anchor('N')
    ax.axis('off')


def panel_a(fig, rect):
    x, y, w, h = rect
    title(fig, rect, 'A', 'Define the spatial CD8 state')
    with Image.open(A_SOURCE) as im:
        image = np.asarray(im.convert('RGB').crop((0, 105, 1688, 1439)))
    ax = axis(fig, x, y+.35, w, h-.37)
    ax.imshow(image)
    ax.set_anchor('N')
    ax.axis('off')


def km_curve(times, events):
    x, s = [0.], [1.]
    value = 1.
    for t in np.unique(times):
        deaths = int(np.sum((times == t) & (events == 1)))
        if deaths:
            value *= 1-deaths/np.sum(times >= t)
            x.append(float(t)); s.append(value)
    if max(times) > x[-1]:
        x.append(float(max(times))); s.append(value)
    return np.array(x), np.array(s)


def panel_b(fig, rect):
    x, y, w, h = rect
    title(fig, rect, 'B', 'Survival association')
    feature = 'observed_ki67_contact30_tcae_per_all'
    columns = [feature, 'Survival_from_Rx_Start', 'Known_Deceased']
    d = SURVIVAL[columns].apply(pd.to_numeric, errors='coerce').dropna()
    t, e = d.iloc[:, 1].to_numpy(), d.iloc[:, 2].to_numpy(int)
    high = d[feature].to_numpy() >= d[feature].median()
    assert len(d) == 30 and int(e.sum()) == 20 and int(high.sum()) == 15
    ax = axis(fig, x+.44, y+.47, w-.50, h-1.28)
    medians = {}
    for name, mask, color in [('High', high, '#C92035'), ('Low', ~high, '#3B82C4')]:
        tx, sy = km_curve(t[mask], e[mask])
        ax.step(tx, sy, where='post', color=color, lw=1.9, label=f'{name} (n=15)')
        ct = t[mask & (e == 0)]
        cs = sy[np.searchsorted(tx, ct, side='right')-1]
        ax.plot(ct, cs, '|', color=color, ms=5, mew=1)
        medians[name] = float(tx[np.flatnonzero(sy <= .5)[0]])
    p = float(SURVIVAL_RESULTS.set_index('feature').loc[feature, 'logrank_p'])
    assert medians == {'High': 554., 'Low': 158.}
    ax.text(.97, .94, 'Median OS: 554 vs 158 d\n'+f'Log-rank P = {p:.3g}',
            transform=ax.transAxes, ha='right', va='top', fontsize=8.3,
            fontweight='bold', linespacing=1.35)
    ax.set(xlim=(0, 2100), ylim=(0, 1.04), yticks=np.arange(0, 1.01, .2))
    ax.set_ylabel('Overall survival probability', fontsize=9.0, labelpad=5)
    ax.tick_params(axis='both', labelsize=8, length=2)
    ax.tick_params(axis='x', labelbottom=False)
    ax.yaxis.grid(True, color='#E3E7EB', lw=.6)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='lower right', fontsize=8.2, frameon=False, labelspacing=.25,
              borderpad=.1, handlelength=1.5)
    rx = axis(fig, x+.44, y+h-.59, w-.50, .25)
    ticks = [0, 500, 1000, 1500, 2000]
    rx.set(xlim=(0, 2100), ylim=(-.5, 1.5), xticks=ticks, yticks=[1, 0])
    rx.set_yticklabels(['High', 'Low'], fontsize=7.6)
    rx.tick_params(axis='y', length=0, pad=6)
    rx.tick_params(axis='x', labelsize=7.6, length=2, pad=2)
    rx.spines[['left', 'top', 'right']].set_visible(False)
    for level, mask, color in [(1, high, '#C92035'), (0, ~high, '#3B82C4')]:
        for tick in ticks:
            rx.text(tick, level, str(int(np.sum(t[mask] >= tick))), ha='center',
                    va='center', fontsize=7.5, color=color)
    rx.set_xlabel('Days from ICI initiation', fontsize=8.6, labelpad=3)
    text(fig, x+.44, y+h-.74, 'Number at risk', fontsize=7.6, fontweight='bold')
    DRAW_AUDIT['B'] = {'n_patients': 30, 'events': 20, 'high_low_counts': [15, 15],
                        'median_OS_days': medians, 'logrank_p': p, 'x_limits': [0, 2100],
                        'risk_times': ticks, 'feature': feature}


def panel_c(fig, rect):
    # Reuse only the plotting portion of the current v4 workflow. Its file I/O
    # and pyplot figure creation are not executed; all text remains vector.
    module = ast.parse(WORKFLOW_SCRIPT.read_text())
    main = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    statements, active = [], False
    for node in main.body:
        names = [t.id for t in node.targets if isinstance(t, ast.Name)] if isinstance(node, ast.Assign) else []
        if 'colors' in names:
            active = True
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == 'extension':
            break
        if not active or set(names) & {'fig', 'ax'}:
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            if any(isinstance(a, ast.Constant) and isinstance(a.value, str) and
                   a.value.startswith('D.  Cross-platform') for a in node.value.args):
                continue
        statements.append(node)
    ax = axis(fig, *rect)
    environment = dict(globals(), fig=fig, ax=ax, W=6.16, H=2.95, version=4,
                       alignment=ALIGN_MANIFEST, donor_count=38, ICON=WORKFLOW_ICON)
    exec(compile(ast.Module(body=statements, type_ignores=[]), str(WORKFLOW_SCRIPT), 'exec'), environment)
    title(fig, rect, 'C', 'Cross-platform spatial-state transfer')


DATASET_COLORS = {'Moldoveanu IMC': '#3F78B5', 'Engelbrecht CITE-seq': '#E68A2E'}
CELL_COLORS = {'Tumor': '#D95D64', 'CD8 T': '#226C9A', 'Other T / CD8-': '#83ADD2',
               'B lineage': '#E6B547', 'NK': '#12A69B', 'Myeloid': '#D88036',
               'Endothelial': '#569860', 'Stroma': '#A676B5', 'Non-joint / ambiguous': '#CCD1D6'}
CELL_ORDER = ['Non-joint / ambiguous', 'Tumor', 'Stroma', 'Endothelial',
              'NK', 'Myeloid', 'B lineage', 'Other T / CD8-', 'CD8 T']


def panel_d(fig, rect):
    x, y, w, h = rect
    title(fig, rect, 'D', 'RAMP alignment')
    gap = .06
    pw = (w-gap)/2
    axes = [axis(fig, x+i*(pw+gap), y+.52, pw, h-1.00) for i in range(2)]
    ds = ALIGN_MD.dataset.to_numpy(str)
    ct = ALIGN_MD.harmonized_coarse_label.to_numpy(str)
    order = np.random.default_rng(20260910).permutation(len(ds))
    axes[0].scatter(*ALIGN_XY[order].T, c=[DATASET_COLORS[d] for d in ds[order]],
                    s=1.35, alpha=.67, lw=0, rasterized=True)
    for category in CELL_ORDER:
        keep = ct == category
        axes[1].scatter(*ALIGN_XY[keep].T, color=CELL_COLORS[category],
                        s=1.35, alpha=.85, lw=0, rasterized=True)
    for ax, name in zip(axes, ['Dataset', 'Cell type']):
        format_umap(ax, ALIGN_XY)
        ax.set_title(name, fontsize=10.8, fontweight='bold', pad=6)
    legend(axes[0], ['IMC', 'CITE-seq'], list(DATASET_COLORS.values()), 2, 8.5, -.025)
    levels = list(CELL_COLORS)
    names = [{'Other T / CD8-': 'Other T', 'Non-joint / ambiguous': 'Others'}.get(c, c) for c in levels]
    legend(axes[1], names, [CELL_COLORS[c] for c in levels], 3, 7.5, -.025)
    assert len(axes[0].collections[0].get_offsets()) == 40000
    assert sum(len(c.get_offsets()) for c in axes[1].collections) == 40000
    DRAW_AUDIT['D'] = {'n_each_map': 40000, 'n_each_dataset': 20000,
                       'coordinates_refitted': False, 'aspect': 'equal',
                       'actual_map_gap_inches': float((axes[1].get_position().x0-axes[0].get_position().x1)*fig.get_figwidth()),
                       'dataset_draw_order': 'one fixed random permutation across both platforms',
                       'dot_size': 1.35, 'dataset_alpha': .67, 'celltype_alpha': .85,
                       'celltype_counts': ALIGN_MD.harmonized_coarse_label.value_counts().to_dict()}


def panel_e(fig, rect):
    x, y, w, h = rect
    title(fig, rect, 'E', 'Held-out spatial recovery')
    boxes = [(126, 220, 1207, 1283), (1478, 220, 2559, 1283)]
    gap = .16
    pw = (w-gap)/2
    with Image.open(PAIR_SOURCE) as im:
        crops = [np.asarray(im.convert('RGB').crop(b)) for b in boxes]
    for i, (crop, name) in enumerate(zip(crops, ['Fixed spatial ground truth', 'Patient-held-out prediction'])):
        ax = axis(fig, x+i*(pw+gap), y+.52, pw, h-.70)
        ax.imshow(crop)
        ax.set_anchor('N')
        ax.axis('off')
        ax.set_title(name, fontsize=10.2, fontweight='bold', pad=7)
    text(fig, x+w/2, y+h-.12, '22RD · same field · whole-section top 20%',
         ha='center', fontsize=8.1)
    DRAW_AUDIT['E'] = {'source_pixel_crops': boxes, 'same_as_original_combined_figure': True,
                       'roi': SPATIAL_INFO['roi_bounds_xyxy'], 'shown_cells': 46,
                       'target_cells': 23, 'predicted_calls': 18, 'tp': 17, 'fp': 1,
                       'source_image_color_processing_unchanged': True}


def panel_f(fig, rect):
    x, y, w, h = rect
    title(fig, rect, 'F', 'Held-out validation')
    d = PATIENT_METRICS.loc[PATIENT_METRICS.evaluable_discrimination.astype(bool)]
    assert len(d) == 14
    text(fig, x+.02, y+.29, '14 patients', fontsize=9.1, color='#58636C')
    rng = np.random.default_rng(20260903)
    for i, (key, name, color, baseline, limits) in enumerate([
        ('AUROC', 'AUROC', '#3F78B5', .5, (.38, .92)),
        ('AUPRC_over_prevalence', 'AUPRC / prevalence', '#E68A2E', 1., (.8, 11.5)),
    ]):
        ax = axis(fig, x+.20, y+.75+i*1.13, w-.31, .73)
        values = d[key].to_numpy(float)
        median = float(np.median(values))
        ax.scatter(values, rng.uniform(-.065, .065, len(values)), s=25,
                   c=color, edgecolor='white', lw=.5, zorder=3)
        ax.axvline(baseline, color='#8A939B', ls=(0, (3, 2)), lw=1)
        ax.scatter(median, .17, marker='D', s=29, c=INK, zorder=4)
        ax.text(median, .25, f'median {median:.3f}' if i == 0 else f'median {median:.2f}×',
                ha='center', va='bottom', fontsize=8.8, fontweight='bold')
        ax.set(xlim=limits, ylim=(-.15, .42), yticks=[])
        if i == 1:
            ax.set_xscale('log'); ax.set_xticks([1, 2, 5, 10], ['1', '2', '5', '10'])
        ax.set_xlabel(name, fontsize=9.3, fontweight='bold', labelpad=3)
        ax.tick_params(axis='x', labelsize=8, length=2)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.xaxis.grid(True, color='#E8EBEF', lw=.55)
    DRAW_AUDIT['F'] = {'n_patients': 14, 'median_AUROC': float(d.AUROC.median()),
                       'median_AUPRC_over_prevalence': float(d.AUPRC_over_prevalence.median())}


def query_display_coordinates():
    xy = PROJECTION[['umap_1', 'umap_2']].to_numpy(float)
    centered = xy-xy.mean(axis=0)
    _, vectors = np.linalg.eigh(np.cov(centered.T))
    a = np.arctan2(vectors[1, -1], vectors[0, -1])
    a = (a+np.pi/2) % np.pi-np.pi/2
    rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    assert np.allclose(rotation.T@rotation, np.eye(2))
    displayed = centered@rotation
    assert np.allclose(np.linalg.norm(np.diff(displayed, axis=0), axis=1),
                       np.linalg.norm(np.diff(xy, axis=0), axis=1))
    return displayed, float(np.degrees(a)), rotation


def panel_g(fig, rect):
    x, y, w, h = rect
    title(fig, rect, 'G', 'Transfer to CITE-seq CD8 cells')
    xy, angle, rotation = query_display_coordinates()
    assert len(xy) == 5768
    keep = {'CD8ct', 'CD8ct.CCL4', 'CD8ct.MT1E+', 'CD8nv'}
    labels = PROJECTION['clust.name'].where(PROJECTION['clust.name'].isin(keep), 'Other CD8').to_numpy(str)
    colors = {'Other CD8': '#C1C7CD', 'CD8nv': '#397FAE', 'CD8ct': '#D9575B',
              'CD8ct.CCL4': '#EBA15E', 'CD8ct.MT1E+': '#CD86B5'}
    gap = .06
    pw = (w-gap)/2
    axes = [axis(fig, x+i*(pw+gap), y+.53, pw, 2.20) for i in range(2)]
    for name, color in colors.items():
        mask = labels == name
        axes[0].scatter(*xy[mask].T, color=color, s=2.65, alpha=.88, lw=0, rasterized=True)
    score = PROJECTION.all_cd8_spatial_state_score.to_numpy(float)
    order = np.argsort(score, kind='stable')
    dots = axes[1].scatter(*xy[order].T, c=score[order], cmap='magma', vmin=0, vmax=.9,
                           s=2.65, alpha=.95, lw=0, rasterized=True)
    for ax, name in zip(axes, ['Provider CD8 states', 'Transferred spatial-state score']):
        format_umap(ax, xy)
        ax.set_title(name, fontsize=10.6, fontweight='bold', pad=6)
    legend(axes[0], list(colors), list(colors.values()), 3, 7.7, -.025)
    cax = axis(fig, x+pw+gap+.40, y+h-.31, pw-.80, .08)
    cb = fig.colorbar(dots, cax=cax, orientation='horizontal', extend='max')
    cb.set_ticks([0, .3, .6, .9]); cb.ax.tick_params(labelsize=7.8, length=2, pad=2)
    cb.outline.set_linewidth(.5)
    assert sum(len(c.get_offsets()) for c in axes[0].collections) == len(xy)
    assert len(axes[1].collections[0].get_offsets()) == len(xy)
    DRAW_AUDIT['G'] = {'n_each_map': len(xy), 'n_patients': int(PROJECTION.Patient_ID.nunique()),
                       'UMAP_refitted': False, 'aspect': 'equal', 'display_rotation_degrees': angle,
                       'actual_map_gap_inches': float((axes[1].get_position().x0-axes[0].get_position().x1)*fig.get_figwidth()),
                       'rotation_matrix': rotation.tolist(),
                       'rotation_selected_from': 'coordinate covariance only, without labels or scores',
                       'pairwise_distances_preserved': True, 'score_limits': [0, .9],
                       'all_original_labels_retained_or_same_other_group': True,
                       'dot_size': 2.65}


def panel_h(fig, rect):
    x, y, w, h = rect
    title(fig, rect, 'H', 'Held-out protein program')
    text(fig, x+.02, y+.30, 'LFA-1/CD2 adhesion with PD-1 checkpoint',
         fontsize=8.5, color='#58636C')
    marker_map = {'CD11a-AB': 'CD11a', 'CD18-AB': 'CD18', 'CD2-AB': 'CD2', 'CD279-AB': 'PD-1'}
    d = EFFECTS.loc[EFFECTS.normalization.eq('classical_clr') & EFFECTS.adt_feature.isin(marker_map)].copy()
    d['marker'] = d.adt_feature.map(marker_map)
    ax = axis(fig, x+.46, y+.68, w-.54, h-1.00)
    colors = ['#0072B2', '#009E73', '#CC79A7', '#D55E00']
    rng = np.random.default_rng(20260831)
    for i, (marker, color) in enumerate(zip(marker_map.values(), colors)):
        values = d.loc[d.marker.eq(marker), 'top25_minus_rest75_rank'].to_numpy(float)
        assert len(values) == 8 and np.all(values > 0)
        ax.scatter(i+rng.uniform(-.10, .10, len(values)), values, s=28,
                   color=color, edgecolor='white', lw=.55, zorder=4)
        ax.plot([i-.21, i+.21], [values.mean()]*2, color=INK, lw=1.7)
        ax.text(i, max(.30, float(values.max())+.025), '8/8 up', fontsize=8.2,
                fontweight='bold', color=color, ha='center', va='bottom')
    ax.axhline(0, color='#8A939B', ls=(0, (3, 2)), lw=.8)
    ax.set(ylim=(-.035, .45), xticks=range(4))
    ax.set_xticklabels(list(marker_map.values()), fontsize=9.2, fontweight='bold')
    ax.set_ylabel('High-state − other CD8 (rank)', fontsize=8.7, labelpad=5)
    ax.tick_params(axis='both', labelsize=8, length=2)
    ax.yaxis.grid(True, color='#E3E7EB', lw=.6)
    ax.spines[['top', 'right']].set_visible(False)
    module = MODULES.loc[MODULES.module.eq('core_4_LFA1_CD2_PD1') & MODULES.population.eq('all_CD8')].iloc[0]
    ax.text(.98, .035, f"4-marker module\nmean patient ρ = {module.mean_patient_spearman:.3f}",
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8.0,
            bbox={'facecolor': 'white', 'edgecolor': '#CBD2D8', 'pad': 2.8})
    DRAW_AUDIT['H'] = {'n_patients_each_marker': 8, 'normalization': 'classical_clr',
                       'contrast': 'within-patient top score quartile minus remaining cells, marker ranks',
                       'markers': list(marker_map.values()),
                       'mean_patient_module_spearman': float(module.mean_patient_spearman)}


PANELS = [
    ('A', panel_a, (.15, .13, 3.45, 3.02)),
    ('B', panel_b, (3.66, .13, 3.57, 3.02)),
    ('C', panel_c, (7.29, .13, 6.06, 3.02)),
    ('D', panel_d, (.15, 3.29, 6.30, 3.80)),
    ('E', panel_e, (6.55, 3.29, 6.80, 3.80)),
    ('F', panel_f, (.15, 7.23, 2.93, 3.12)),
    ('G', panel_g, (3.16, 7.23, 6.68, 3.12)),
    ('H', panel_h, (9.92, 7.23, 3.43, 3.12)),
]


def export(fig, stem):
    for extension in ['png', 'svg']:
        fig.savefig(HERE / f'{stem}.{extension}', dpi=300, facecolor='white')


def main():
    fig = plt.figure(figsize=(W, H), facecolor='white')
    for letter, draw, rect in PANELS:
        draw(fig, rect)
    export(fig, 'melanoma_three_row_redesign_v2')
    plt.close(fig)
    for letter, draw, (_, _, w, h) in PANELS:
        fig = plt.figure(figsize=(w+.16, h+.16), facecolor='white')
        draw(fig, (.08, .08, w, h))
        export(fig, f'panels/{letter}')
        plt.close(fig)
    assert all(sha(path) == value for path, value in SOURCES.items())
    (HERE / 'LAYOUT_AUDIT.json').write_text(json.dumps({
        'canvas_inches': [W, H], 'matches_CD19_canvas': False,
        'previous_canvas_inches': [14.5, 10.5],
        'canvas_width_reduction_percent': 100*(14.5-W)/14.5,
        'new_to_old_panel': {'A': 'A', 'B': 'B', 'C': 'D', 'D': 'E', 'E': 'C', 'F': 'F', 'G': 'G', 'H': 'H'},
        'rectangles_inches_from_top_left': {letter: rect for letter, _, rect in PANELS},
        'workflow_source_version': 4, 'workflow_has_vector_text': True,
        'B_width_inches_old_v1_v2': [2630/300, 4.15, 3.57],
        'panels': DRAW_AUDIT, 'source_sha256': SOURCES,
        'all_sources_unchanged': True, 'script_sha256': sha(__file__),
        'analysis_refitted': False,
    }, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(DRAW_AUDIT, ensure_ascii=False, indent=2), flush=True)
    print(HERE / 'melanoma_three_row_redesign_v2.png', flush=True)


if __name__ == '__main__':
    main()
