"""Job 2 before and after refitting the alignment without the target marker.

Left: per-task Spearman at a fixed capacity, main grid vs leave-one-out.
Right: the change per arm, with each arm's leakage reduction beside it.
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import summarise_lomo as S

COLOR = {'RAMP': '#C0392B', 'Raw': '#7F8C8D', 'Harmony-direct': '#2980B9',
         'CytoNorm': '#16A085', 'cyCombine': '#8E44AD',
         'fastMNN-reconstructed': '#D68910', 'fastMNN-embedding': '#D68910',
         'VirTues-style': '#2C3E50', 'Spatium-style': '#B7950B'}
CAP = 'linear'


def main():
    main_df, lomo = S.tables()
    pl, pm = S.pivot(lomo, CAP), S.pivot(main_df, CAP)
    tasks = [t for t in pl.index if t in pm.index]
    arms = [a for a in S.ORDER if a in pl.columns]

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6),
                           gridspec_kw={'width_ratios': [1.9, 1]})
    x = np.arange(len(tasks))
    for a in arms:
        ax[0].plot(x, pl.loc[tasks, a], 'o-', color=COLOR.get(a, '#888'),
                   lw=2.2 if a == 'RAMP' else 1.2, ms=6 if a == 'RAMP' else 4,
                   label=a, zorder=3 if a == 'RAMP' else 2)
    ax[0].set_xticks(x); ax[0].set_xticklabels(tasks, rotation=20, ha='right')
    ax[0].set_ylabel(f'Job 2 Spearman ({CAP})')
    ax[0].set_title('Alignment refitted without the marker being predicted')
    ax[0].grid(alpha=.25); ax[0].legend(fontsize=7.5, ncol=2, frameon=False)

    delta = []
    for a in arms:
        m = S.PAIR.get(a, a)
        if m in pm.columns:
            delta.append((a, float((pl.loc[tasks, a] - pm.loc[tasks, m]).mean())))
    delta.sort(key=lambda t: t[1])
    ax[1].barh([d[0] for d in delta], [d[1] for d in delta],
               color=[COLOR.get(d[0], '#888') for d in delta])
    ax[1].axvline(0, color='k', lw=.8)
    ax[1].set_xlabel('change in Spearman after refitting')
    ax[1].set_title('Leave-one-out minus main grid')
    ax[1].grid(axis='x', alpha=.25)
    for s in ('top', 'right'):
        ax[0].spines[s].set_visible(False); ax[1].spines[s].set_visible(False)
    fig.tight_layout()
    out = HERE.parents[1] / 'figures'
    out.mkdir(exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'lomo_job2.{ext}', dpi=200)
    print('wrote', out / 'lomo_job2.png')


if __name__ == '__main__':
    main()
