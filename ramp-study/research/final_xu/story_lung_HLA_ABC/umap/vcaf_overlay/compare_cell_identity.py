"""Compare Cords CAF subtypes with Zheng author annotations on fixed coordinates."""
import plot_vcaf_overlay as rt
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main():
    paths = [rt.STORY / 'data_umap/data/CELL_METADATA.csv.gz',
             rt.STORY / 'umap/results/RAMP_UMAP.npy']
    hashes = {str(p): rt.sha(p) for p in paths}
    md = pd.read_csv(paths[0], low_memory=False)
    xy = np.load(paths[1])
    assert len(md) == len(xy) == 84000 and md.cell_key.is_unique
    cords = md.dataset.eq('Cords IMC')
    zheng = md.dataset.eq('Zheng mIF')
    masks = {
        'Cords other CAF': cords & md.cell_category.eq('Fibroblast') & ~md.provider_cell_type.eq('vCAF'),
        'Cords vCAF': cords & md.provider_cell_type.eq('vCAF'),
        'Zheng Fibroblast': zheng & md.provider_cell_type.eq('Fibroblast'),
        'Zheng Vascular': zheng & md.provider_cell_type.eq('Vascular'),
    }
    counts = {k: int(v.sum()) for k, v in masks.items()}
    assert list(counts.values()) == [4560, 646, 4805, 3871]
    purple, pink, teal = '#9869b0', '#e54285', '#35a7b4'
    fig, axes = plt.subplots(1, 3, figsize=(17.4, 6.6))
    fig.subplots_adjust(left=.036, right=.985, bottom=.18, top=.835, wspace=.14)
    lo, hi = xy.min(0), xy.max(0)
    pad = (hi-lo)*.04
    records = []
    for i, (ax, title) in enumerate(zip(axes, ['Cords: other CAF', 'Cords: vCAF', 'Zheng: author annotations'])):
        ax.scatter(xy[:, 0], xy[:, 1], s=.7, c='#cbd0d5', alpha=.14, linewidths=0, rasterized=True)
        ax.set(title=title, xlabel='UMAP 1', ylabel='UMAP 2', xticks=[], yticks=[],
               xlim=(lo[0]-pad[0], hi[0]+pad[0]), ylim=(lo[1]-pad[1], hi[1]+pad[1]))
        ax.set_aspect('equal', adjustable='box')
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['bottom', 'left']].set_color('#6c737d')
        ax.text(-.04, 1.045, 'ABC'[i], transform=ax.transAxes, weight='bold', fontsize=15)
    cases = [(0, 'Cords other CAF', purple, 3.4), (1, 'Cords vCAF', pink, 7),
             (2, 'Zheng Fibroblast', purple, 3.4), (2, 'Zheng Vascular', teal, 3.4)]
    handles = {0: [], 1: [], 2: []}
    for panel, name, color, size in cases:
        ids = np.random.default_rng(42).permutation(np.flatnonzero(masks[name]))
        points = xy[ids]
        a = axes[panel].scatter(points[:, 0], points[:, 1], s=size, c=color, alpha=.8,
                               linewidths=0, rasterized=True)
        assert np.array_equal(a.get_offsets(), points)
        short = name.replace('Cords ', '').replace('Zheng ', '')
        handles[panel].append(Line2D([], [], marker='o', ls='', color=color, markersize=5,
                                    label=f'{short} (n = {len(ids):,})'))
        records.append({'group': name, 'n': len(ids), 'n_patients': int(md.iloc[ids].patient_id.nunique()),
                        'coordinates_exact': True})
    for i, ax in enumerate(axes):
        ax.legend(handles=handles[i], loc='upper center', bbox_to_anchor=(.5, -.095),
                  frameon=False, ncol=1 if i<2 else 2, fontsize=10)
    fig.suptitle('Where do Cords vCAF sit relative to other CAF and Zheng cell identities?',
                 fontsize=17, y=.966)
    fig.text(.5, .035, 'Same fixed three-cohort UMAP in every panel | Original author annotations | All cells in each indicated reference group are shown',
             ha='center', fontsize=10, color='#596370')
    stem = rt.OUT / 'CAF_vCAF_and_Zheng_identity'
    fig.savefig(stem.with_suffix('.png'), dpi=230, facecolor='white')
    fig.savefig(stem.with_suffix('.svg'), facecolor='white')
    plt.close(fig)
    assert hashes == {str(p): rt.sha(p) for p in paths}
    (rt.OUT / 'CELL_IDENTITY_VERIFICATION.json').write_text(json.dumps({
        'status': 'PASS', 'input_sha256': hashes, 'inputs_unchanged': True, 'groups': records,
        'refit': False, 'axis_limits_identical': True, 'hla_values_used': False,
        'script_sha256': rt.sha(__file__), 'plot_runtime_sha256': rt.sha(rt.__file__),
        'figures': {p.name: rt.sha(p) for p in [stem.with_suffix('.png'), stem.with_suffix('.svg')]}
    }, indent=2)+'\n')
    print(json.dumps(counts, indent=2))


if __name__ == '__main__':
    main()
