"""Fixed patient-group composition results; no models or thresholds are refit."""
from runtime import BASE, STORY
from pathlib import Path
import hashlib, json, shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

COLORS = {'Lower': '#588299', 'Higher': '#cb506a'}
SUMMARY = STORY / 'summary_steps3_6'
FILES = {
    'data/PATIENT_SUMMARY.csv': SUMMARY / '05_immune_environment/PATIENT_SUMMARY.csv',
    'data/COMPOSITION_COMPARISONS.csv': SUMMARY / '05_immune_environment/COMPOSITION_COMPARISONS.csv',
    'data/COMPOSITION_ADJUSTED.csv': SUMMARY / '05_immune_environment/COMPOSITION_ADJUSTED.csv',
    'data/PATIENT_IMMUNE_COMPOSITION.csv': STORY / 'mechanism_support/patient_immune/results/PATIENT_IMMUNE_COMPOSITION.csv',
    'provenance/KM_PATIENTS.csv': SUMMARY / '06_DFS/KM_PATIENTS.csv',
    'provenance/KM_SUMMARY.csv': SUMMARY / '06_DFS/KM_SUMMARY.csv',
}

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p, v): p.write_text(json.dumps(v, indent=2, ensure_ascii=False) + '\n')

def main():
    sources = []
    for dest, source in FILES.items():
        out = BASE / dest
        if out.exists(): assert sha(out) == sha(source)
        else: shutil.copyfile(source, out)
        sources.append({'source': str(source), 'local': dest, 'sha256': sha(source)})
    js(BASE / 'provenance/SOURCES.json', sources)
    md = pd.read_csv(BASE / 'data/PATIENT_SUMMARY.csv', float_precision='round_trip')
    stats = pd.read_csv(BASE / 'data/COMPOSITION_COMPARISONS.csv', float_precision='round_trip').set_index('endpoint')
    adjusted = pd.read_csv(BASE / 'data/COMPOSITION_ADJUSTED.csv', float_precision='round_trip')
    adjusted = adjusted[~adjusted.full_CAF_covariates].set_index('endpoint')
    patients = pd.read_csv(BASE / 'provenance/KM_PATIENTS.csv')
    joined = md.merge(patients[['patient_id', 'group']], on='patient_id', validate='one_to_one', suffixes=('', '_DFS'))
    assert len(joined) == len(md) == len(patients) == 280
    assert joined.group.equals(joined.group_DFS)
    assert md.group.value_counts().to_dict() == {'Lower': 140, 'Higher': 140}
    km = pd.read_csv(BASE / 'provenance/KM_SUMMARY.csv').iloc[0]
    assert np.array_equal(md.group.eq('Higher'), md.all_mean.ge(km.median_cut))
    full = pd.read_csv(BASE / 'data/PATIENT_IMMUNE_COMPOSITION.csv', float_precision='round_trip').set_index('patient_id')
    assert len(full) == len(md) and full.index.is_unique
    np.testing.assert_allclose(md.CD8_of_immune, md.CD8 / md.immune_cells, rtol=0, atol=1e-14)
    np.testing.assert_allclose(md.CD8_of_immune, full.loc[md.patient_id, 'CD8_of_immune'], rtol=0, atol=1e-14)
    checked = stats.reset_index().merge(adjusted[['q_bh']].rename(columns={'q_bh': 'adjusted_q_bh'}),
                                       left_on='endpoint', right_index=True, validate='one_to_one')
    checked['difference_percentage_points'] = 100 * (checked.higher_mean - checked.lower_mean)
    checked['relative_change_percent'] = 100 * (checked.higher_mean / checked.lower_mean - 1)
    checked.to_csv(BASE / 'data/CHECKED_COMPOSITION.csv', index=False)
    endpoints = [
        ('immune_of_all', 'Immune\ncells', 'immune_cells', 'all_cells'),
        ('CD4_of_immune', 'CD4 T', 'CD4', 'immune_cells'),
        ('CD8_of_immune', 'CD8 T', 'CD8', 'immune_cells'),
        ('Bcell_of_immune', 'B cells', 'Bcell', 'immune_cells'),
        ('Myeloid_of_immune', 'Myeloid', 'Myeloid', 'immune_cells'),
        ('Neutrophil_of_immune', 'Neutrophils', 'Neutrophil', 'immune_cells'),
    ]
    lineage_counts = [e[2] for e in endpoints[1:]]
    lineage_fractions = [e[0] for e in endpoints[1:]]
    assert np.array_equal(md[lineage_counts].sum(axis=1), md.immune_cells)
    np.testing.assert_allclose(md[lineage_fractions].sum(axis=1), 1, rtol=0, atol=1e-14)
    rows = []
    for endpoint, title, numerator, denominator in endpoints:
        np.testing.assert_allclose(md[endpoint], md[numerator] / md[denominator], rtol=0, atol=1e-14)
        np.testing.assert_allclose(md[endpoint], full.loc[md.patient_id, endpoint], rtol=0, atol=1e-14)
        for group in COLORS:
            value = float(md.loc[md.group.eq(group), endpoint].mean())
            assert abs(value - stats.loc[endpoint, group.lower() + '_mean']) < 1e-12
            rows.append({'endpoint': endpoint, 'label': title.replace('\n', ' '), 'group': group,
                         'numerator': numerator, 'denominator': denominator,
                         'n_patients': 140, 'mean_fraction': value, 'mean_percent': value * 100,
                         'adjusted_q_bh': float(adjusted.loc[endpoint, 'q_bh'])})
    plotted = pd.DataFrame(rows)
    plotted.to_csv(BASE / 'data/PLOT_SUMMARY.csv', index=False)
    md[['patient_id', 'group', 'all_mean', *(e[0] for e in endpoints)]].to_csv(BASE / 'data/PLOT_PATIENTS.csv', index=False)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'svg.fonttype': 'none',
                         'text.color': '#243745', 'axes.labelcolor': '#43515a',
                         'axes.edgecolor': '#9ba8b0', 'axes.linewidth': .7,
                         'savefig.facecolor': 'white'})
    width, height = 4.50, 2.75
    fig = plt.figure(figsize=(width, height))
    fig.text(.017, .960, 'E', fontsize=22, fontweight='bold', va='top')
    fig.text(.50 / width, .957, 'Immune composition', fontsize=12.3, fontweight='bold', va='top')
    # Use the former legend strip to explain the existing patient-level grouping.
    workflow_artists = []
    flow_y_offset = -.06
    def xy(x, y): return x / width, (y + flow_y_offset) / height
    def box(x, w):
        patch = FancyBboxPatch(xy(x, 2.05), w / width, .36 / height,
                              boxstyle='round,pad=0.004,rounding_size=0.012',
                              facecolor='white', edgecolor='#c8d1d5', linewidth=.8,
                              transform=fig.transFigure, zorder=3)
        fig.add_artist(patch); workflow_artists.append(patch)
    def line(x1, y1, x2, y2):
        artist = Line2D([x1 / width, x2 / width], [(y1 + flow_y_offset) / height, (y2 + flow_y_offset) / height],
                        color='#67747c', linewidth=.8, transform=fig.transFigure, zorder=2)
        fig.add_artist(artist); workflow_artists.append(artist)
    def arrow(x1, y1, x2, y2, color='#67747c'):
        patch = FancyArrowPatch(xy(x1, y1), xy(x2, y2), arrowstyle='-|>',
                                mutation_scale=6.5, linewidth=.8, color=color,
                                shrinkA=0, shrinkB=0, transform=fig.transFigure, zorder=2)
        fig.add_artist(patch); workflow_artists.append(patch)
    box(.18, 1.25); box(1.61, 1.25)
    fig.text(*xy(.805, 2.295), 'Predicted HLA-ABC', ha='center', va='center', fontsize=7.2, fontweight='bold')
    fig.text(*xy(.805, 2.155), 'in vCAF', ha='center', va='center', fontsize=7.2)
    fig.text(*xy(2.235, 2.295), 'Mean per patient', ha='center', va='center', fontsize=7.2, fontweight='bold')
    fig.text(*xy(2.235, 2.155), 'Median split', ha='center', va='center', fontsize=7.0, color='#56646d')
    arrow(1.45, 2.23, 1.59, 2.23)
    line(2.88, 2.23, 3.02, 2.23)
    line(3.02, 2.12, 3.02, 2.34)
    for group, y in [('Lower', 2.34), ('Higher', 2.12)]:
        arrow(3.02, y, 3.285, y, COLORS[group])
        swatch = Rectangle(xy(3.31, y - .045), .10 / width, .09 / height,
                           facecolor=COLORS[group], edgecolor='none', transform=fig.transFigure)
        fig.add_artist(swatch); workflow_artists.append(swatch)
        fig.text(*xy(3.48, y), f'{group} (n = 140)', color=COLORS[group],
                 fontsize=7.4, ha='left', va='center')
    ax = fig.add_axes([.51 / width, .54 / height, 3.81 / width, 1.38 / height])
    text_artists = []
    label_pairs = []
    for x, (endpoint, label, _, _) in enumerate(endpoints):
        pair = []
        for group, offset in [('Lower', -.17), ('Higher', .17)]:
            row = plotted[plotted.endpoint.eq(endpoint) & plotted.group.eq(group)].iloc[0]
            ax.bar(x + offset, row.mean_percent, width=.29, color=COLORS[group], linewidth=0)
            item = ax.text(x + offset, row.mean_percent + .9, f'{row.mean_percent:.1f}%',
                           ha='center', va='bottom', fontsize=6.3, color=COLORS[group])
            text_artists.append(item); pair.append(item)
        label_pairs.append(pair)
    ax.set(xlim=(-.58, len(endpoints) - .42), ylim=(0, 55), yticks=[0, 20, 40],
           xticks=range(len(endpoints)), xticklabels=[e[1] for e in endpoints], ylabel='Mean fraction (%)')
    ax.tick_params(axis='y', length=3, pad=3, labelsize=7.5, color='#9ba8b0')
    ax.tick_params(axis='x', length=0, pad=5, labelsize=6.5)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_axisbelow(True); ax.grid(axis='y', color='#e8edef', linewidth=.5)
    # Shared denominator brackets keep all five lineage labels readable.
    denominator_artists = []
    for left, right, label in [(-.40, .40, '/ all cells'), (.60, 5.40, '/ immune cells')]:
        artist, = ax.plot([left, left, right, right], [-.215, -.245, -.245, -.215],
                          transform=ax.get_xaxis_transform(), clip_on=False,
                          color='#9ba8b0', linewidth=.6)
        denominator_artists.append(artist)
        denominator_artists.append(ax.text((left + right) / 2, -.28, label,
                                   transform=ax.get_xaxis_transform(), clip_on=False,
                                   ha='center', va='top', fontsize=6.5, color='#43515a'))
    # Stagger close pair labels without moving or changing the bars.
    fig.canvas.draw(); renderer = fig.canvas.get_renderer()
    for pair in label_pairs:
        a, b = [item.get_window_extent(renderer) for item in pair]
        if a.padded(.7).overlaps(b.padded(.7)):
            upper = max(pair, key=lambda item: item.get_position()[1])
            lower = next(item for item in pair if item is not upper)
            upper_box = upper.get_window_extent(renderer)
            lower_box = lower.get_window_extent(renderer)
            shift_px = lower_box.y1 + 1.5 - upper_box.y0
            delta = shift_px * (ax.get_ylim()[1] - ax.get_ylim()[0]) / ax.bbox.height
            xpos, ypos = upper.get_position()
            upper.set_position((xpos, ypos + max(delta, 0)))
    fig.canvas.draw(); renderer = fig.canvas.get_renderer()
    for a, b in label_pairs:
        assert not a.get_window_extent(renderer).overlaps(b.get_window_extent(renderer))
    for item in [*fig.texts, *workflow_artists, *text_artists, *denominator_artists, *ax.get_yticklabels(), *ax.get_xticklabels(), ax.yaxis.label]:
        box = item.get_window_extent(renderer).transformed(fig.transFigure.inverted())
        assert box.x0 >= 0 and box.x1 <= 1 and box.y0 >= 0 and box.y1 <= 1, box
    for ext in ['png', 'svg']: fig.savefig(BASE / f'panel_E_immune_composition.{ext}', dpi=300)
    plt.close(fig)
    for r in sources: assert sha(Path(r['source'])) == sha(BASE / r['local']) == r['sha256']
    js(BASE / 'provenance/VERIFICATION.json', {
        'status': 'PASS', 'version': 'v5_all_five_immune_lineages', 'title': 'Immune composition',
        'bar_orientation': 'Vertical', 'axis_label': 'Mean fraction (%)',
        'adjusted_q_displayed_on_figure': False,
        'CD8_denominator': 'All immune cells, consistent with CD4, B cells, Myeloid and neutrophils',
        'lineage_denominators_consistent': True,
        'all_five_author_immune_lineages_displayed': True,
        'five_lineage_fractions_sum_to_one_per_patient': True,
        'shared_denominator_brackets': True,
        'percentage_label_pairs_do_not_overlap': True,
        'workflow': ['Predicted HLA-ABC in vCAF', 'Mean per patient', 'Median split', 'Lower / Higher'],
        'workflow_location': 'Former legend strip above the bars',
        'legend_orientation': 'Vertical; Lower above Higher',
        'workflow_arrows_point_to_group_swatch_labels': True,
        'grouping_unit': 'Patient, not individual cells',
        'n_patients': 280, 'group_sizes': {'Lower': 140, 'Higher': 140},
        'same_patients_and_groups_as_survival': True, 'median_cut': float(km.median_cut),
        'endpoints': [e[0] for e in endpoints], 'patient_equal_means_verified': True,
        'denominators_verified': True, 'adjusted_q_source': 'Existing TMA, age, sex, stage model; BH family of 12 composition endpoints',
        'new_model_or_threshold_search': False, 'statistics_refit': False,
        'source_files_unchanged': True, 'all_text_inside_canvas': True,
        'group_colors': COLORS, 'figure_size_inches': [width, height], 'plotted_statistics': rows})
    js(BASE / 'provenance/MANIFEST.json', [{'file': str(p.relative_to(BASE)), 'sha256': sha(p)}
       for p in sorted(BASE.rglob('*')) if p.is_file() and p.relative_to(BASE).parts[0] not in {'cache', 'tmp', 'versions'} and p.name != 'MANIFEST.json'])
    print('Panel E complete: total immune fraction plus all five annotated immune lineages.', flush=True)

if __name__ == '__main__': main()
