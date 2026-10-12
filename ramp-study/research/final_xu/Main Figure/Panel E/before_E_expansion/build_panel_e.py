"""Common-unit summary of strict unequal-input order inversions."""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from figure_style import *
import numpy as np
import pandas as pd


def main():
    setup()
    order = ['Harmony-direct[source_scale]', 'Harmony-direct[fixed_arcsinh]',
             'cyCombine', 'CytoNorm', 'RAMP']
    original = pd.read_csv(track(ROOT / 'benchmark_results/exact_value_artifacts/summary/rank_integrity_by_unit_marker.csv.gz'))
    selected = original[original.method.isin(order)].copy()
    selected['drop'] = selected.maximum_strict_order_inversion_margin
    cyto = pd.read_csv(track(ROOT / 'algorithm_test/cytonorm_artifact_audit/rank_integrity_by_unit_marker.csv'))
    cyto['method'] = 'CytoNorm'; cyto['drop'] = cyto.max_inversion_native_drop
    keys = ['case', 'dataset', 'unit_key', 'marker']
    fields = [*keys, 'method', 'drop']
    data = pd.concat([selected[fields], cyto[fields]], ignore_index=True)
    assert not data.duplicated([*keys, 'method']).any()
    common = set(map(tuple, data[data.method.eq('RAMP')][keys].to_numpy()))
    assert len(common) == 1151
    for method in order:
        assert set(map(tuple, data[data.method.eq(method)][keys].to_numpy())) == common
    # Standardize the two existing audits to the same strict numerical tolerance.
    data['inversion'] = data['drop'] > 1e-12
    data.to_csv(HERE / 'rank_integrity_unit_marker.csv.gz', index=False)
    summary = data.groupby('method').inversion.agg(['sum', 'count']).loc[order]
    summary.columns = ['n_with_inversion', 'n_tested']
    summary['percent'] = 100*summary.n_with_inversion/summary.n_tested
    summary.to_csv(HERE / 'rank_integrity_summary.csv')
    data.groupby(['case', 'method']).inversion.agg(['sum', 'count']).to_csv(HERE / 'rank_integrity_by_benchmark.csv')
    fig = new_panel('E', 'Rank preservation')
    fig.text(.55, .827, 'All 4 benchmarks', ha='center', fontsize=10.5, weight='bold')
    ax = fig.add_axes([.35, .235, .605, .525])
    y = np.arange(5)
    ax.barh(y, 100, height=.63, color='#EBEFF0', zorder=1)
    ax.barh(y, summary.percent, height=.63, color='#CB765E', zorder=2)
    for i, row in enumerate(summary.itertuples()):
        pct = row.percent
        text = f'{pct:.1f}%\n{row.n_with_inversion:,}/{row.n_tested:,}'
        if pct > 80:
            ax.text(97, i, text, ha='right', va='center', color='white', fontsize=9.2, weight='bold')
        else:
            ax.text(min(pct+3, 65), i, text, ha='left', va='center', color='#222222', fontsize=9.2, weight='bold')
        if pct == 0: ax.scatter([0], [i], color='#087F5B', s=22, zorder=4, clip_on=False)
    ax.set_yticks(y, ['Harmony-\ndirect', 'Harmony-\narcsinh', 'cyCombine', 'CytoNorm', 'RAMP'])
    ax.tick_params(axis='y', length=0, labelsize=9, pad=5)
    ax.get_yticklabels()[-1].set_weight('bold'); ax.get_yticklabels()[-1].set_color('#087F5B')
    ax.set_xlim(0, 100); ax.set_ylim(4.6, -.6)
    ax.set_xticks([0, 50, 100]); ax.tick_params(axis='x', labelsize=9, length=2.2)
    ax.set_xlabel('Sample × marker combinations\nwith order inversion (%)', fontsize=9.3,
                  weight='bold', labelpad=7)
    for side in ['left', 'right', 'top']: ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color('#8C8C8C')
    save(fig, HERE, 'panel_e_rank_preservation', {
        'panel': 'E', 'methods': order, 'n_common_unit_markers': 1151,
        'n_units': 104, 'benchmark_unit_marker_counts': {'bone_marrow': 238, 'tnbc': 121, 'melanoma': 552, 'lung_cancer': 240},
        'definition': 'At least one x_i < x_j but y_i > y_j within the same sample and marker; input ties excluded; output difference >1e-12.',
        'comparison': 'Each method input versus its corrected marker output; exact-value outputs only.',
        'aggregation': 'Pooled sample×marker combinations, not equal weight per benchmark; numbers are affected combinations / tested combinations, not cell-pair counts.',
        'interpretation': 'Any strict numerical order reversal; this is not the count of biological marker artifacts and does not measure effect severity.',
        'excluded': 'Formal fastMNN and MaxFuse embeddings have no one-to-one marker axis; legacy fastMNN d=NA stress test excluded.',
    })


if __name__ == '__main__': main()
