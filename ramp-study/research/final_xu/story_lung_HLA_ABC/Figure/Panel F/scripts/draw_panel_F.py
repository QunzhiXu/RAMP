"""Render the fixed primary DFS curves and risk table; no survival model refit."""
from runtime import BASE, STORY
from pathlib import Path
import hashlib, json, shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

COLORS = {'Lower': '#588299', 'Higher': '#cb506a'}
SUMMARY = STORY / 'summary_steps3_6'
FILES = {f'data/{name}.csv': SUMMARY / f'06_DFS/{name}.csv' for name in
         ['KM_CURVE', 'KM_PATIENTS', 'KM_SUMMARY', 'FIVE_YEAR_DFS', 'COX_TERMS', 'NUMBER_AT_RISK']}
FILES['provenance/IMMUNE_PATIENTS.csv'] = SUMMARY / '05_immune_environment/PATIENT_SUMMARY.csv'

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
    def read(name): return pd.read_csv(BASE / f'data/{name}.csv', float_precision='round_trip')
    curve, patients, km, five, cox, risk = [read(n) for n in
        ['KM_CURVE', 'KM_PATIENTS', 'KM_SUMMARY', 'FIVE_YEAR_DFS', 'COX_TERMS', 'NUMBER_AT_RISK']]
    assert len(km) == 1 and km.iloc[0].id == 'RAMP_k30__mean__all280'
    k = km.iloc[0]
    assert len(patients) == 280 and patients.patient_id.is_unique and patients.event.sum() == 96
    assert patients.group.value_counts().to_dict() == {'Lower': 140, 'Higher': 140}
    immune = pd.read_csv(BASE / 'provenance/IMMUNE_PATIENTS.csv')
    both = patients.merge(immune[['patient_id', 'group']], on='patient_id', validate='one_to_one', suffixes=('', '_immune'))
    assert len(both) == 280 and both.group.equals(both.group_immune)
    for row in risk.itertuples():
        assert int((patients.group.eq(row.group) & patients.time_years.ge(row.time_years)).sum()) == row.n_at_risk
    for row in five.itertuples():
        c = curve[curve.group.eq(row.group)]
        assert c.time.is_monotonic_increasing and c.survival.is_monotonic_decreasing
        assert abs(c[c.time.le(5)].iloc[-1].survival - row.five_year_survival) < 1e-12
    hr = cox[cox.term.eq('state_z')].iloc[0]
    assert hr.model == 'matched_full' and hr.config == 'RAMP_k30' and hr.state == 'mean'
    fs = five.set_index('group')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'svg.fonttype': 'none',
                         'text.color': '#243745', 'axes.labelcolor': '#43515a',
                         'axes.edgecolor': '#9ba8b0', 'axes.linewidth': .7,
                         'savefig.facecolor': 'white'})
    # Completes a row of exactly the approved ABC width with D=4.9, E=4.5 and two 3-pt gaps.
    width, height = (1025.04 - 4.9 * 72 - 4.5 * 72 - 6) / 72, 2.75
    fig = plt.figure(figsize=(width, height))
    fig.text(.017, .960, 'F', fontsize=22, fontweight='bold', va='top')
    fig.text(.50 / width, .957, 'Survival association', fontsize=12.3, fontweight='bold', va='top')
    fig.text(.57 / width, 2.29 / height,
             f'Adjusted HR per SD: {hr.hr:.2f} (95% CI {hr.ci_low:.2f}–{hr.ci_high:.2f})',
             fontsize=7.8, va='center', color='#56646d')
    left, bottom, plot_w, plot_h = .57, .84, width - .82, 1.32
    ax = fig.add_axes([left / width, bottom / height, plot_w / width, plot_h / height])
    for group, color in COLORS.items():
        c = curve[curve.group.eq(group)]
        x = np.r_[0., c.time.to_numpy()]; y = np.r_[1., c.survival.to_numpy()]
        artist, = ax.step(x, y, where='post', color=color, linewidth=1.4, label=group)
        assert np.array_equal(artist.get_xdata(), x) and np.array_equal(artist.get_ydata(), y)
        ax.fill_between(x, np.r_[1., c.lower], np.r_[1., c.upper], step='post', color=color, alpha=.13, linewidth=0)
        censor = c[c.n_censor.gt(0)]
        ax.scatter(censor.time, censor.survival, marker='+', s=9, color=color, linewidths=.6)
    ax.set(xlim=(0, 8), ylim=(0, 1.04), xticks=[0, 2, 4, 6, 8], yticks=[0, .25, .5, .75, 1],
           xlabel='Years', ylabel='Disease-free survival (%)')
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0, symbol=''))
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=7.2, length=3, pad=2, color='#9ba8b0')
    ax.xaxis.label.set_size(8); ax.yaxis.label.set_size(8)
    legend = ax.legend(loc='lower left', bbox_to_anchor=(.01, .015), frameon=False,
                       fontsize=7.2, handlelength=1.4, labelspacing=.35, borderpad=0, handletextpad=.4)
    t1 = ax.text(.52, .27, f'Log-rank p = {k.logrank_p:.4f}', transform=ax.transAxes, fontsize=7.2)
    t2 = ax.text(.52, .13, f'5-year DFS: {fs.loc["Lower", "five_year_survival"]:.1%} / {fs.loc["Higher", "five_year_survival"]:.1%}',
                 transform=ax.transAxes, fontsize=7.2)
    assert 0 < k.logrank_p < 1
    fig.text(left / width, .40 / height, 'Number at risk', fontsize=7.2, va='center')
    for group, y in [('Lower', .24), ('Higher', .08)]:
        fig.text(.40 / width, y / height, group, color=COLORS[group], fontsize=7.2, ha='right', va='center')
        for row in risk[risk.group.eq(group)].itertuples():
            fig.text((left + row.time_years / 8 * plot_w) / width, y / height, str(row.n_at_risk),
                     fontsize=7.2, color=COLORS[group], ha='center', va='center')
    fig.canvas.draw(); renderer = fig.canvas.get_renderer()
    for item in [*fig.texts, legend, t1, t2, ax.xaxis.label, ax.yaxis.label,
                 *ax.get_xticklabels(), *ax.get_yticklabels()]:
        box = item.get_window_extent(renderer).transformed(fig.transFigure.inverted())
        assert box.x0 >= 0 and box.x1 <= 1 and box.y0 >= 0 and box.y1 <= 1, box
    for ext in ['png', 'svg']: fig.savefig(BASE / f'panel_F_survival.{ext}', dpi=300)
    plt.close(fig)
    for r in sources: assert sha(Path(r['source'])) == sha(BASE / r['local']) == r['sha256']
    js(BASE / 'provenance/VERIFICATION.json', {
        'status': 'PASS', 'version': 'v3_survival_association_title', 'title': 'Survival association',
        'legend_labels': ['Lower', 'Higher'],
        'n_patients': 280, 'n_events': 96, 'group_sizes': {'Lower': 140, 'Higher': 140},
        'same_patients_and_groups_as_Panel_E': True, 'median_cut': float(k.median_cut),
        'logrank_p': float(k.logrank_p), 'five_year_DFS': five.to_dict(orient='records'),
        'adjusted_HR_per_SD': hr[['hr', 'ci_low', 'ci_high', 'p']].to_dict(),
        'risk_table_checked_from_patient_followup': True,
        'curve_coordinates_identical_to_saved_results': True, 'CI_and_censoring_retained': True,
        'models_refit': False, 'threshold_optimized': False, 'source_files_unchanged': True,
        'all_text_inside_canvas': True, 'group_colors': COLORS,
        'figure_size_inches': [width, height], 'x_range_years': [0, 8]})
    js(BASE / 'provenance/MANIFEST.json', [{'file': str(p.relative_to(BASE)), 'sha256': sha(p)}
       for p in sorted(BASE.rglob('*')) if p.is_file() and p.relative_to(BASE).parts[0] not in {'cache', 'tmp', 'versions'} and p.name != 'MANIFEST.json'])
    print('Panel F complete: fixed KM curves, 95% CI, censoring and risk table.', flush=True)

if __name__ == '__main__': main()
