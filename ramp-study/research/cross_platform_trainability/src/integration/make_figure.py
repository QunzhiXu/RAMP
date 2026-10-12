"""Two panels: the two-axis integration view, and the data-efficiency curves."""
import glob, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / 'scaling'))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import paths, summarise_scaling as SS

COLOR = {'RAMP': '#C0392B', 'Raw': '#7F8C8D', 'Harmony-direct': '#2980B9',
         'CytoNorm': '#16A085', 'cyCombine': '#8E44AD',
         'fastMNN-reconstructed': '#D68910', 'VirTues-style': '#2C3E50',
         'Spatium-style': '#B7950B'}
ORDER = list(COLOR)


def mm(s):
    return (s - s.min()) / (s.max() - s.min())


def main():
    d = pd.concat([pd.read_csv(f) for f in
                   sorted(glob.glob(str(paths.ROOT / 'results/integration/*.csv')))])
    t = d.groupby('arm')[['iLISI_norm', 'kBET_reject', 'ASW_batch', 'cLISI_norm',
                          'ARI', 'NMI', 'ASW_celltype']].mean()
    t['BATCH'] = (t.iLISI_norm + (1 - t.kBET_reject) + (1 - mm(t.ASW_batch))) / 3
    t['BIO'] = (mm(t.cLISI_norm) + mm(t.ARI) + mm(t.NMI) + mm(t.ASW_celltype)) / 4

    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.2))
    for arm in t.index:
        c = COLOR.get(arm, '#888')
        big = arm == 'RAMP'
        ax[0].scatter(t.loc[arm, 'BATCH'], t.loc[arm, 'BIO'], s=190 if big else 110,
                      color=c, zorder=3, edgecolor='k' if big else 'none', linewidth=1.2)
        ax[0].annotate(arm, (t.loc[arm, 'BATCH'], t.loc[arm, 'BIO']),
                       textcoords='offset points', xytext=(8, 5), fontsize=8.5,
                       fontweight='bold' if big else 'normal')
    ax[0].set_xlabel('batch mixing  (iLISI, 1−kBET, 1−ASW$_{batch}$)')
    ax[0].set_ylabel('biological conservation  (cLISI, ARI, NMI, ASW$_{cell type}$)')
    ax[0].set_title('Mixing against conservation, mean over 7 tasks')
    ax[0].grid(alpha=.25); ax[0].margins(.16)

    df = SS.load()
    c = SS.curves(df, 'job1_celltype')
    piv = c.groupby(['arm', 'per_class'])['macro_f1'].mean().unstack()
    for arm in [a for a in ORDER if a in piv.index]:
        ax[1].plot(piv.columns, piv.loc[arm], 'o-', color=COLOR[arm],
                   lw=2.4 if arm == 'RAMP' else 1.2, ms=6 if arm == 'RAMP' else 4,
                   label=arm, zorder=3 if arm == 'RAMP' else 2)
    ax[1].set_xscale('log')
    ax[1].set_xlabel('labelled training cells per class')
    ax[1].set_ylabel('Job 1 macro F1')
    ax[1].set_title('Data efficiency (S2 sweep, every point class-balanced)')
    ax[1].grid(alpha=.25); ax[1].legend(fontsize=7.5, ncol=2, frameon=False)
    for a in ax:
        for s in ('top', 'right'):
            a.spines[s].set_visible(False)
    fig.tight_layout()
    out = paths.ROOT / 'figures'
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'integration_and_scaling.{ext}', dpi=200)
    print('wrote', out / 'integration_and_scaling.png')


if __name__ == '__main__':
    main()
