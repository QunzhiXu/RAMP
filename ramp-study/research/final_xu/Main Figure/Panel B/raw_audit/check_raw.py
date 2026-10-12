"""Trace bone-marrow Raw to benchmark inputs and diagnose linear-ridge resolution."""
from pathlib import Path
import os
import sys
import json

HERE = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(HERE.parent / 'bandwidth_review/.cache/matplotlib'))
sys.path.insert(0, str(HERE.parent / 'benchmark_review'))
from build_ridge_review import (ROOT, load_case, load_matrices, ridge, dataset_legend, sha)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    hashes, plot_audit = {}, []
    meta, _coords = load_case('bone_marrow', hashes)
    markers, matrices = load_matrices('bone_marrow', meta, hashes)
    raw = matrices['Unaligned']
    source_root = ROOT / 'benchmark_data/bone_marrow'
    relabel_path = source_root / 'relabel/data/bone_marrow_88031_relabeled.csv.gz'
    hashes[str(relabel_path)] = sha(relabel_path)
    source = pd.read_csv(relabel_path, low_memory=False)
    assert np.array_equal(meta.cell_id.astype(str), source.cell_id.astype(str))
    assert np.array_equal(raw, source[markers].to_numpy())
    checks = []
    for dataset, folder in [('BM CODEX','bm_codex'),('BM CyTOF','bm_cytof'),
                            ('Triana AbSeq','triana_abseq')]:
        selected = meta.dataset == dataset
        for file in ['alignment_input_14.csv.gz', 'source_input_15.csv.gz']:
            path = source_root / folder / file
            hashes[str(path)] = sha(path)
            frame = pd.read_csv(path, low_memory=False)
            assert frame.cell_id.is_unique
            ids = meta.loc[selected,'cell_id'].astype(str)
            assert set(frame.cell_id.astype(str)) == set(ids)
            values = frame.set_index(frame.cell_id.astype(str)).loc[ids, markers].to_numpy()
            equal = np.array_equal(raw[selected], values)
            delta = float(np.abs(raw[selected]-values).max())
            assert equal, (dataset, file, delta)
            checks.append({'dataset':dataset, 'file':str(path), 'cells':len(ids),
                           'markers_compared':len(markers), 'exactly_equal':equal,
                           'max_absolute_difference':delta})
    stats = []
    chosen = ['CD3','CD4','CD8']
    for marker in chosen:
        values = raw[:,markers.index(marker)]
        lo, hi = np.quantile(values,[.005,.995])
        width = (hi-lo)/160
        for dataset in meta.dataset.unique():
            v = values[meta.dataset == dataset]
            stats.append({'marker':marker,'dataset':dataset,'cells':len(v),
                          'linear_display_low':lo,'linear_display_high':hi,
                          'linear_bin_width':width,'first_bin_upper':lo+width,
                          'sigma4_in_source_units':4*width,
                          'fraction_in_first_bin':float(((v>=lo)&(v<lo+width)).mean()),
                          'q10':float(np.quantile(v,.1)),'median':float(np.median(v)),
                          'q90':float(np.quantile(v,.9))})
    pd.DataFrame(stats).to_csv(HERE/'raw_scale_diagnostics.csv',index=False)
    fig = plt.figure(figsize=(12.7,8.6),facecolor='white')
    fig.text(.07,.985,'Bone marrow | Same Raw data, different display scales',
             fontsize=16,weight='bold',va='top')
    dataset_legend(fig,meta,.916)
    grid = fig.add_gridspec(2,3,left=.075,right=.985,top=.855,bottom=.065,
                           hspace=.23,wspace=.18)
    for row in range(2):
        for col, marker in enumerate(chosen):
            values = raw[:,markers.index(marker)]
            if row == 1:
                values = np.arcsinh(values)
            ax = fig.add_subplot(grid[row,col])
            ridge(ax,values,meta,'bone_marrow',
                  'Raw_linear' if row == 0 else 'Raw_arcsinh_display',
                  marker,plot_audit,sigma_bins=4)
            if row == 0:
                ax.set_title(marker,fontsize=13,weight='bold',pad=9)
            if col == 0:
                ax.set_ylabel('Raw | Linear' if row == 0 else 'Raw | Arcsinh display',
                              fontsize=12,weight='bold',labelpad=12)
            ax.set_xlabel('Source value' if row == 0 else 'arcsinh(source value / 1)',
                          fontsize=10,labelpad=5)
    fig.savefig(HERE/'raw_linear_vs_arcsinh.png',dpi=190)
    plt.close(fig)
    pd.DataFrame(plot_audit).to_csv(HERE/'display_audit.csv',index=False)
    input_manifest_path=ROOT/'benchmark_results/bone_marrow/ramp/alignment/alignment_manifest.json'
    original_manifest=json.loads(input_manifest_path.read_text())
    raw_path=ROOT/'benchmark_results/bone_marrow/ramp/alignment/input/raw.npy'
    raw_hash=sha(raw_path)
    assert raw_hash==original_manifest['inputs']['raw.npy']
    assert all(sha(Path(p))==s for p,s in hashes.items())
    (HERE/'provenance_audit.json').write_text(json.dumps({
        'raw_path':str(raw_path),'raw_sha256':raw_hash,
        'matches_alignment_manifest':True,
        'exactly_equal_to_relabel_source':True,
        'platform_source_comparisons':checks,
        'input_sha256':hashes,'inputs_unchanged':True,
        'cells':len(raw),'markers':len(markers),
        'provider_preprocessing':{
            'BM CODEX':'Provider-processed CODEX counts-slot cell-level intensity',
            'BM CyTOF':'Provider CCT-subtracted, normalized, pre-gated FCS event values',
            'Triana AbSeq':'Raw antibody UMI counts'},
        'raw_definition':'Unaligned benchmark source input, before RAMP KMEANSLOW',
        'display_transform':'arcsinh(x / 1), identical for all cells and datasets; display only',
        'sample_specific_rescaling':False,'benchmark_or_umap_refit':False,
    },indent=2)+'\n')
    print(json.dumps({'source_checks':checks, 'diagnostics':stats},indent=2),flush=True)


if __name__ == '__main__':
    main()
