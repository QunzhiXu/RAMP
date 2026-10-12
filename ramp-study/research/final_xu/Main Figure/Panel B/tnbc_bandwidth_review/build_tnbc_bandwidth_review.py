"""TNBC ridge bandwidth comparison from frozen matrices and UMAP coordinates."""
from pathlib import Path
import sys
import os
import json

HERE=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(HERE.parent/'bandwidth_review/.cache/matplotlib'))
sys.path.insert(0,str(HERE.parent/'benchmark_review'))
from build_ridge_review import (load_case,load_matrices,ridge,combined,
                                dataset_legend,SELECTED,sha)
import matplotlib.pyplot as plt
import pandas as pd


def main():
    sources,audit={},[]
    meta,coords=load_case('tnbc',sources)
    markers,matrices=load_matrices('tnbc',meta,sources)
    plt.rcParams.update({'font.family':'DejaVu Sans'})
    fig=plt.figure(figsize=(12.5,9.8),facecolor='white')
    fig.text(.06,.983,'TNBC | RAMP ridge smoothing',fontsize=17,weight='bold',va='top')
    dataset_legend(fig,meta,.922)
    grid=fig.add_gridspec(3,3,left=.07,right=.985,top=.868,bottom=.045,wspace=.18,hspace=.21)
    for row,marker in enumerate(SELECTED['tnbc']):
        for col,sigma in enumerate([2,4,6]):
            ax=fig.add_subplot(grid[row,col])
            ridge(ax,matrices['RAMP'][:,markers.index(marker)],meta,
                  'tnbc','RAMP',marker,audit,sigma_bins=sigma)
            if row==0:
                ax.set_title({2:'Original | 1×',4:'Smoother | 2×',6:'Smoother | 3×'}[sigma],
                             fontsize=12,weight='bold',pad=10)
            if col==0:
                ax.set_ylabel(marker,fontsize=14,weight='bold',labelpad=13)
    fig.savefig(HERE/'tnbc_bandwidth_comparison.png',dpi=190)
    plt.close(fig)
    for sigma in [4,6]:
        combined('tnbc',meta,coords,markers,matrices,audit,sigma_bins=sigma,
                 raw_display='arcsinh',
                 output_path=HERE/f'tnbc_ridge_umap_sigma{sigma}.png')
    pd.DataFrame(audit).drop_duplicates(
        ['benchmark','method','marker','unit_key','smoothing_sigma_bins']
    ).to_csv(HERE/'ridge_display_audit.csv',index=False)
    assert all(sha(Path(path))==value for path,value in sources.items())
    (HERE/'manifest.json').write_text(json.dumps({
        'benchmark':'tnbc','input_sha256':sources,'inputs_unchanged':True,
        'cells':len(meta),'sample_units':int(meta.unit_key.nunique()),
        'markers':SELECTED['tnbc'],'histogram_bins':160,
        'original_sigma_bins':2,'compared_sigma_bins':[2,4,6],
        'same_bandwidth_rule_applied_to_all_methods':True,
        'ridge_range_quantiles':[.005,.995],
        'raw_ridge_display_transform':'arcsinh(source value / 1), identical for all Raw cells',
        'raw_display_transform_used_for_umap_or_alignment':False,
        'other_method_ridge_display_transform':'none; native saved output values',
        'smoothing_mode':'constant','tail_values_clipped_into_edge_bins':False,
        'umap_refit':False,'alignment_refit':False,
        'posthoc_sample_recentering_or_scaling':False,
        'original_review_images_preserved':True,
    },indent=2)+'\n')


if __name__=='__main__':
    main()
