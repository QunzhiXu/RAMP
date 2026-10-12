import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

res = pd.read_csv('composition_sensitivity_results.csv')
gt  = pd.read_csv('gate_transfer_results.csv')
COL = {'Raw': '#9aa6ad', 'RAMP': '#008866',
       'VirTues-style': '#cb506a', 'Spatium-style': '#5b6ebf'}
ORDER = ['Raw', 'RAMP', 'VirTues-style', 'Spatium-style']
SRC_FRAC = {'CD3': 0.209, 'CD4': 0.586, 'CD8': 0.318}

fig, axes = plt.subplots(1, 4, figsize=(13.4, 3.3))

# panel A: matched-composition transfer (the first experiment)
ax = axes[0]
x = np.arange(3); w = 0.2
for i, arm in enumerate(ORDER):
    v = [gt[(gt.arm == arm) & (gt.marker == m)].target_BA.iloc[0] for m in ['CD3','CD4','CD8']]
    ax.bar(x + (i - 1.5) * w, v, w, color=COL[arm], label=arm, lw=0)
ax.axhline(.5, color='#333', lw=.6, ls=(0, (3, 2)))
ax.set_xticks(x); ax.set_xticklabels(['CD3', 'CD4', 'CD8'])
ax.set_ylim(.45, 1); ax.set_ylabel('Balanced accuracy on target')
ax.set_title('A  Gate transfer at native composition', fontsize=9.5, loc='left')
ax.legend(fontsize=6.6, frameon=False, loc='lower right', ncol=2)

# panels B-D: composition sweep
for k, m in enumerate(['CD3', 'CD4', 'CD8']):
    ax = axes[k + 1]
    d = res[res.marker == m]
    keep = sorted(c for c in d.nominal.unique() if d[d.nominal == c].rep.nunique() == 5)
    for arm in ORDER:
        g = d[d.arm == arm].groupby('nominal').BA.agg(['mean', 'std']).loc[keep]
        ax.plot(keep, g['mean'], color=COL[arm], lw=1.5, marker='o', ms=3)
        ax.fill_between(keep, g['mean'] - g['std'], g['mean'] + g['std'],
                        color=COL[arm], alpha=.18, lw=0)
    ax.axvline(SRC_FRAC[m], color='#333', lw=.7, ls=(0, (2, 2)))
    ax.text(SRC_FRAC[m], .455, ' source', fontsize=6, color='#333', va='bottom')
    ax.axhline(.5, color='#333', lw=.6, ls=(0, (3, 2)))
    ax.set_xlabel('positive-cell fraction in target cohort')
    ax.set_ylim(.45, 1); ax.set_xlim(0, .85)
    ax.set_title(f'{"BCD"[k]}  {m}', fontsize=9.5, loc='left')
    if k: ax.set_yticklabels([])
axes[1].set_ylabel('Balanced accuracy on target')

for ax in axes:
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig('figure_fm_normalization.png', dpi=220)
fig.savefig('figure_fm_normalization.pdf')
print('written')
