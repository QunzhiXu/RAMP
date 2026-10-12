"""Compare two larger ridge smoothing bandwidths on frozen bone-marrow data."""
from pathlib import Path
import sys
import json
import os

HERE = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(HERE / '.cache/matplotlib'))
sys.path.insert(0, str(HERE.parent / 'benchmark_review'))
from build_ridge_review import (load_case, load_matrices, ridge, combined,
                                dataset_legend, SELECTED, sha)
import matplotlib.pyplot as plt
import pandas as pd


def main():
    sources, audit = {}, []
    meta, coords = load_case('bone_marrow', sources)
    markers, matrices = load_matrices('bone_marrow', meta, sources)
    plt.rcParams.update({'font.family': 'DejaVu Sans'})
    fig = plt.figure(figsize=(12.5, 11.4), facecolor='white')
    fig.text(.06, .984, 'Bone marrow | RAMP ridge smoothing',
             fontsize=17, weight='bold', va='top')
    dataset_legend(fig, meta, .925)
    grid = fig.add_gridspec(3, 3, left=.07, right=.985, top=.878,
                           bottom=.035, wspace=.18, hspace=.20)
    for row, marker in enumerate(SELECTED['bone_marrow']):
        for col, sigma in enumerate([2,4,6]):
            ax = fig.add_subplot(grid[row,col])
            ridge(ax, matrices['RAMP'][:,markers.index(marker)], meta,
                  'bone_marrow', 'RAMP', marker, audit, sigma_bins=sigma)
            if row == 0:
                ax.set_title({2:'Original | 1×',4:'Smoother | 2×',6:'Smoother | 3×'}[sigma],
                             fontsize=12, weight='bold', pad=10)
            if col == 0:
                ax.set_ylabel(marker, fontsize=14, weight='bold', labelpad=13)
    fig.savefig(HERE / 'bone_marrow_bandwidth_comparison.png', dpi=190)
    plt.close(fig)
    for sigma in [4,6]:
        combined('bone_marrow', meta, coords, markers, matrices, audit,
                 sigma_bins=sigma,
                 output_path=HERE / f'bone_marrow_ridge_umap_sigma{sigma}.png')
    pd.DataFrame(audit).drop_duplicates(
        ['benchmark','method','marker','unit_key','smoothing_sigma_bins']
    ).to_csv(HERE / 'ridge_display_audit.csv', index=False)
    assert all(sha(Path(path)) == value for path,value in sources.items())
    (HERE / 'manifest.json').write_text(json.dumps({
        'benchmark':'bone_marrow', 'input_sha256':sources, 'inputs_unchanged':True,
        'cells':len(meta), 'sample_units':int(meta.unit_key.nunique()),
        'markers':SELECTED['bone_marrow'], 'histogram_bins':160,
        'original_sigma_bins':2, 'compared_sigma_bins':[2,4,6],
        'same_bandwidth_rule_applied_to_all_methods':True,
        'ridge_range_quantiles':[.005,.995], 'range_unchanged_from_previous':True,
        'x_scale':'linear', 'smoothing_mode':'constant',
        'tail_values_clipped_into_edge_bins':False,
        'umap_refit':False, 'alignment_refit':False,
        'posthoc_sample_recentering_or_scaling':False,
        'original_review_images_preserved':True,
    },indent=2)+'\n')


if __name__ == '__main__':
    main()
