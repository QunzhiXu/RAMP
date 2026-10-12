"""Fixed patient-group composition results; no models or thresholds are refit."""
from runtime import BASE, STORY
from pathlib import Path
import hashlib, json, shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

COLORS = {'Lower': '#588299', 'Higher': '#cb506a'}
SUMMARY = STORY / 'summary_steps3_6'
FILES = {
    'data/PATIENT_SUMMARY.csv': SUMMARY / '05_immune_environment/PATIENT_SUMMARY.csv',
    'data/COMPOSITION_COMPARISONS.csv': SUMMARY / '05_immune_environment/COMPOSITION_COMPARISONS.csv',
    'data/COMPOSITION_ADJUSTED.csv': SUMMARY / '05_immune_environment/COMPOSITION_ADJUSTED.csv',
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
    endpoints = [
        ('immune_of_all', 'Immune cells\n/ all cells', 'immune_cells', 'all_cells'),
        ('CD4_of_immune', 'CD4 T\n/ immune cells', 'CD4', 'immune_cells'),
        ('Neutrophil_of_immune', 'Neutrophils\n/ immune cells', 'Neutrophil', 'immune_cells'),
    ]
    rows = []
    for endpoint, title, numerator, denominator in endpoints:
        np.testing.assert_allclose(md[endpoint], md[numerator] / md[denominator], rtol=0, atol=1e-14)
        for group in COLORS:
            value = float(md.loc[md.group.eq(group), endpoint].mean())
            assert abs(value - stats.loc[endpoint, group.lower() + '_mean']) < 1e-12
            rows.append({'endpoint': endpoint, 'label': title.replace('\n', ' '), 'group': group,
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
    legend = fig.legend(handles=[Patch(facecolor=c, label=f'{g} vCAF HLA') for g, c in COLORS.items()],
                        loc='center', bbox_to_anchor=(2.70 / width, 2.29 / height), ncol=2,
                        frameon=False, fontsize=7.5, handlelength=1.0, handletextpad=.45,
                        columnspacing=1.0, borderaxespad=0, borderpad=0)
    ax = fig.add_axes([1.10 / width, .50 / height, 3.20 / width, 1.59 / height])
    text_artists = []
    for y, (endpoint, label, _, _) in zip([2, 1, 0], endpoints):
        for group, offset in [('Lower', .145), ('Higher', -.145)]:
            row = plotted[plotted.endpoint.eq(endpoint) & plotted.group.eq(group)].iloc[0]
            ax.barh(y + offset, row.mean_percent, height=.23, color=COLORS[group], linewidth=0)
            text_artists.append(ax.text(row.mean_percent + .65, y + offset, f'{row.mean_percent:.1f}%',
                                       va='center', fontsize=7.4, color=COLORS[group]))
        q = float(adjusted.loc[endpoint, 'q_bh'])
        qtext = f'{q:.2g}'.replace('e-0', 'e−').replace('e-', 'e−')
        text_artists.append(ax.text(29, y + .34, f'Adj. q = {qtext}', ha='right', va='bottom',
                                   fontsize=7.0, color='#56646d'))
    ax.set(xlim=(0, 29), ylim=(-.50, 2.61), xticks=[0, 10, 20],
           yticks=[2, 1, 0], yticklabels=[e[1] for e in endpoints], xlabel='Mean patient fraction (%)')
    ax.tick_params(axis='y', length=0, pad=6, labelsize=8)
    ax.tick_params(axis='x', length=3, pad=3, labelsize=7.5, color='#9ba8b0')
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.set_axisbelow(True); ax.grid(axis='x', color='#e8edef', linewidth=.5)
    fig.canvas.draw(); renderer = fig.canvas.get_renderer()
    for item in [*fig.texts, legend, *text_artists, *ax.get_yticklabels(), *ax.get_xticklabels(), ax.xaxis.label]:
        box = item.get_window_extent(renderer).transformed(fig.transFigure.inverted())
        assert box.x0 >= 0 and box.x1 <= 1 and box.y0 >= 0 and box.y1 <= 1, box
    for ext in ['png', 'svg']: fig.savefig(BASE / f'panel_E_immune_composition.{ext}', dpi=300)
    plt.close(fig)
    for r in sources: assert sha(Path(r['source'])) == sha(BASE / r['local']) == r['sha256']
    js(BASE / 'provenance/VERIFICATION.json', {
        'status': 'PASS', 'version': 'v1_three_composition_readouts', 'title': 'Immune composition',
        'n_patients': 280, 'group_sizes': {'Lower': 140, 'Higher': 140},
        'same_patients_and_groups_as_survival': True, 'median_cut': float(km.median_cut),
        'endpoints': [e[0] for e in endpoints], 'patient_equal_means_verified': True,
        'denominators_verified': True, 'adjusted_q_source': 'Existing TMA, age, sex, stage model; BH family of 12 composition endpoints',
        'new_model_or_threshold_search': False, 'statistics_refit': False,
        'source_files_unchanged': True, 'all_text_inside_canvas': True,
        'group_colors': COLORS, 'figure_size_inches': [width, height], 'plotted_statistics': rows})
    js(BASE / 'provenance/MANIFEST.json', [{'file': str(p.relative_to(BASE)), 'sha256': sha(p)}
       for p in sorted(BASE.rglob('*')) if p.is_file() and p.relative_to(BASE).parts[0] not in {'cache', 'tmp', 'versions'} and p.name != 'MANIFEST.json'])
    print('Panel E complete: fixed 280 patients, three existing composition contrasts.', flush=True)

if __name__ == '__main__': main()
