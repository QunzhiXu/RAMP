#!/usr/bin/env python3
"""HNSCC: A+B on top, C+E below, vertical D spanning both rows.

Redraw compact B/C/E from frozen results and compose vector PDFs. No model,
UMAP, survival fit, or marker-selection step is rerun.
"""
from pathlib import Path
import hashlib
import json
import os

HERE = Path(__file__).resolve().parent
BIG = HERE.parent
HNSCC = BIG.parent
os.environ.setdefault('MPLCONFIGDIR', str(HERE / '.cache' / 'matplotlib'))
os.environ.setdefault('MPLBACKEND', 'Agg')

import fitz
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

S = HNSCC / 'survival_motivation/output'
SCAN = BIG / 'panel_E/group_fraction_scan_20260910'
INPUTS = {
    'A': BIG / 'panel_A/panel_A_niche_discovery.pdf',
    'D': BIG / 'panel_D/panel_D_vertical.pdf',
    'km': S / 'median_split_km_curve.csv',
    'risk': S / 'median_split_number_at_risk.csv',
    'survival_models': S / 'survival_models.csv',
    'umap': HNSCC / 'alignment/umap/full_94488_umap_coordinates.csv.gz',
    'molecular_summary': SCAN / 'equal_donor_contrasts.csv',
    'molecular_donors': SCAN / 'donor_contrasts.csv',
    'molecular_counts': SCAN / 'group_counts.csv',
}
INK, LOW, HIGH, GRID = '#182027', '#2775B6', '#D95F02', '#E7EBEE'
DATASET_COLORS = {'GSE212797_ADT': '#0072B2', 'Haist_CODEX': '#E69F00'}
DATASET_NAMES = {'GSE212797_ADT': 'Rahim CITE-seq', 'Haist_CODEX': 'Haist CODEX'}
DISPLAY_ORDER = ['Tumor / epithelial / stroma', 'Myeloid', 'Other', 'Vascular', 'NK / gdT', 'Plasma', 'B', 'CD8 T', 'CD4 T']
TYPE_COLORS = {'CD8 T': '#D55E00', 'CD4 T': '#CC79A7', 'B': '#0072B2', 'Plasma': '#56B4E9',
               'NK / gdT': '#009E73', 'Myeloid': '#E5C100', 'Tumor / epithelial / stroma': '#B2182B',
               'Vascular': '#17BECF', 'Other': '#BDBDBD'}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def header(fig, label, title, wide=False):
    fig.text(.015 if wide else .025, .976, label, fontsize=23, weight='bold', color=INK, ha='left', va='top')
    fig.text(.068 if wide else .145, .947, title, fontsize=13.0 if wide else 11.9,
             weight='bold', color=INK, ha='left', va='top')


def save(fig, label, stem):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = list(fig.texts)
    for ax in fig.axes:
        texts += [ax.title, ax._left_title, ax.xaxis.label, ax.yaxis.label]
        texts += list(ax.texts) + ax.get_xticklabels() + ax.get_yticklabels()
        if ax.get_legend():
            texts += ax.get_legend().get_texts()
    for item in texts:
        if not item.get_visible() or not item.get_text():
            continue
        b = item.get_window_extent(renderer)
        assert b.x0 >= -1 and b.y0 >= -1 and b.x1 <= fig.bbox.width+1 and b.y1 <= fig.bbox.height+1, (label, item.get_text())
    out = BIG / f'panel_{label}' / stem
    for fmt in ['png', 'pdf', 'svg']:
        fig.savefig(out.with_suffix('.'+fmt), dpi=320, facecolor='white')
    plt.close(fig)
    return out.with_suffix('.pdf')


def build_b():
    km, risk, models = [pd.read_csv(INPUTS[k]) for k in ['km', 'risk', 'survival_models']]
    model = models[models.model.eq('Primary: continuous spatial score')].iloc[0]
    assert model.patients == 74 and model.deaths == 24
    assert set(km.group) == {'Lower', 'Higher'} and km.survival.between(0, 1).all()
    assert risk[risk.time.eq(0)].n_risk.sum() == 74
    fig = plt.figure(figsize=(4.65, 3.45), facecolor='white')
    header(fig, 'B', 'SPATIAL NICHE AND SURVIVAL')
    fig.text(.555, .835, f'Continuous Cox: HR = {model.hazard_ratio:.2f} per SD\n95% CI {model.ci95_low:.2f}-{model.ci95_high:.2f}; P = {model.wald_p:.3f}',
             ha='center', va='center', fontsize=9.1, color=INK, linespacing=1.35)
    ax = fig.add_axes([.155, .310, .785, .400])
    colors = {'Lower': '#2B6CB0', 'Higher': '#C9433C'}
    for group in ['Lower', 'Higher']:
        b = km[km.group.eq(group)].sort_values('time')
        n = int(risk[risk.group.eq(group) & risk.time.eq(0)].n_risk.iloc[0])
        ax.step(b.time, b.survival, where='post', color=colors[group], lw=2.1,
                label=f'{group} niche score (n={n})')
    ax.set(xlim=(0, 120), ylim=(0, 1.05), xticks=[0, 24, 48, 72, 96, 120], yticks=[0, .25, .5, .75, 1])
    ax.set_xlabel('Overall survival (months)', fontsize=9.4, labelpad=4)
    ax.set_ylabel('Survival probability', fontsize=9.3, labelpad=5)
    ax.tick_params(labelsize=8.1, length=3)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='lower left', frameon=False, fontsize=8.3, handlelength=1.7, borderaxespad=.4)
    fig.text(.028, .147, 'At risk', ha='left', va='center', fontsize=8.4, weight='bold')
    for group, y in [('Lower', .102), ('Higher', .053)]:
        fig.text(.133, y, group, ha='right', va='center', fontsize=8.1, color=colors[group])
        vals = risk[risk.group.eq(group)].set_index('time').n_risk
        for x, t in zip(np.linspace(.155, .94, 6), [0, 24, 48, 72, 96, 120]):
            fig.text(x, y, str(int(vals.loc[t])), ha='center', va='center', fontsize=8.1, color=colors[group])
    return save(fig, 'B', 'panel_B_survival_compact'), model.to_dict()


def build_c():
    d = pd.read_csv(INPUTS['umap'], low_memory=False)
    assert len(d) == 94488 and d.cell_uid.notna().all()
    assert set(d.dataset) == set(DATASET_COLORS) and set(d.plot_label) == set(DISPLAY_ORDER)
    assert np.isfinite(d[['UMAP1', 'UMAP2']]).all().all()
    fig = plt.figure(figsize=(9.3, 3.45), facecolor='white')
    header(fig, 'C', 'RAMP ALIGNMENT OF HNSCC CITE-seq AND CODEX', wide=True)
    axes = [fig.add_axes([.012, .025, .483, .82]), fig.add_axes([.505, .025, .483, .82])]
    order = np.random.default_rng(20260902).permutation(len(d))
    axes[0].scatter(d.UMAP1.to_numpy()[order], d.UMAP2.to_numpy()[order],
                    c=d.dataset.map(DATASET_COLORS).to_numpy()[order], s=.78, alpha=.48, linewidths=0, rasterized=True)
    for i, label in enumerate(DISPLAY_ORDER):
        ix = np.flatnonzero(d.plot_label.eq(label))
        ix = ix[np.random.default_rng(20260902+i).permutation(len(ix))]
        axes[1].scatter(d.UMAP1.to_numpy()[ix], d.UMAP2.to_numpy()[ix], color=TYPE_COLORS[label],
                        s=.78, alpha=.50, linewidths=0, rasterized=True, label=label)
    axes[0].legend(handles=[Line2D([0], [0], marker='o', linestyle='', markersize=5.2,
                        markerfacecolor=color, markeredgewidth=0,
                        label=f'{DATASET_NAMES[key]} (n={int(d.dataset.eq(key).sum()):,})')
                        for key, color in DATASET_COLORS.items()],
                    loc='upper left', frameon=False, fontsize=7.9, borderaxespad=.1, handletextpad=.4)
    axes[1].legend(loc='upper left', ncol=2, frameon=False, fontsize=7.1,
                   markerscale=4.7, borderaxespad=.1, handletextpad=.30, columnspacing=.7)
    xspan, yspan = d.UMAP1.max()-d.UMAP1.min(), d.UMAP2.max()-d.UMAP2.min()
    for ax, title in zip(axes, ['DATASET', 'CELL TYPE']):
        ax.set_title(title, loc='left', fontsize=10.6, weight='bold', pad=4)
        ax.set(xlim=(d.UMAP1.min()-.02*xspan, d.UMAP1.max()+.02*xspan),
               ylim=(d.UMAP2.min()-.02*yspan, d.UMAP2.max()+.25*yspan), xticks=[], yticks=[])
        ax.set_aspect('equal', adjustable='box')
        for spine in ax.spines.values():
            spine.set_visible(False)
    return save(fig, 'C', 'panel_C_alignment_compact'), d.groupby('dataset').size().to_dict()


def build_e():
    s = pd.read_csv(INPUTS['molecular_summary'])
    s = s[s.fraction.eq(.3)].set_index(['modality', 'marker'])
    donor = pd.read_csv(INPUTS['molecular_donors'], dtype={'donor': str})
    donor = donor[donor.fraction.eq(.3)]
    counts = pd.read_csv(INPUTS['molecular_counts'])
    counts = counts[counts.fraction.eq(.3)]
    assert len(s) == 6 and counts.n_low.sum() == 1439 and counts.n_high.sum() == 1437
    features = [('ADT', 'CD223--LAG-3', 'LAG-3'), ('ADT', 'CD127--IL-7Ra', 'CD127'), ('ADT', 'CD62L', 'CD62L'),
                ('RNA', 'CTLA4', 'CTLA4'), ('RNA', 'CXCL13', 'CXCL13'), ('RNA', 'IL7R', 'IL7R')]
    rows = []
    for modality, marker, label in features:
        r = s.loc[(modality, marker)]
        b = donor[donor.modality.eq(modality) & donor.marker.eq(marker)]
        assert len(b) == 3 and r.donors_consistent == 3
        np.testing.assert_allclose(b[['low', 'high']].mean(), r[['low', 'high']].astype(float), rtol=0, atol=1e-12)
        rows.append(dict(modality=modality, marker=marker, label=label, lower=r.low, higher=r.high,
                         unit='mean_expression_percentile' if modality == 'ADT' else 'RNA_detection_percent'))
    table = pd.DataFrame(rows)
    fig = plt.figure(figsize=(4.65, 3.45), facecolor='white')
    header(fig, 'E', 'CD39/TIGIT STATE PHENOTYPE')
    fig.text(.54, .818, 'Measured CD39/TIGIT state', ha='center', va='center', fontsize=9.8)
    fig.legend(handles=[Line2D([0], [0], marker='o', linestyle='', color=c, markeredgewidth=0, markersize=5.5, label=t)
                         for t, c in [('Lower 30%', LOW), ('Higher 30%', HIGH)]],
               loc='center', bbox_to_anchor=(.54, .739), ncol=2, frameon=False, fontsize=9.4,
               columnspacing=1.2, handlelength=.7, handletextpad=.4)
    axes = [fig.add_axes([.16, .183, .320, .452]), fig.add_axes([.705, .183, .270, .452])]
    for ax, modality, title, xlabel in zip(axes, ['ADT', 'RNA'], ['Protein', 'RNA'], ['Mean expression\npercentile', 'RNA detected (%)']):
        b = table[table.modality.eq(modality)]
        for y, r in zip([2, 1, 0], b.itertuples(index=False)):
            ax.plot([r.lower, r.higher], [y, y], color='#BFC8CF', lw=1.5)
            ax.scatter(r.lower, y, s=36, color=LOW, edgecolors='white', linewidths=.5, zorder=3)
            ax.scatter(r.higher, y, s=36, color=HIGH, edgecolors='white', linewidths=.5, zorder=3)
        ax.set(xlim=(-4, 104), ylim=(-.55, 2.55), xticks=[0, 50, 100], yticks=[2, 1, 0])
        ax.set_yticklabels(b.label.tolist(), fontsize=9.1)
        ax.tick_params(axis='y', length=0, pad=4)
        ax.tick_params(axis='x', length=2.5, pad=3, labelsize=8.4)
        ax.set_title(title, fontsize=10.5, weight='bold', pad=8)
        ax.set_xlabel(xlabel, fontsize=9, labelpad=5)
        ax.grid(axis='x', color=GRID, lw=.65)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.spines['bottom'].set_color('#7A838A')
    table.to_csv(BIG / 'panel_E/panel_E_molecular_30pct_two_columns_values.csv', index=False)
    return save(fig, 'E', 'panel_E_molecular_30pct_two_columns'), rows


def main():
    hashes = {k: sha(p) for k, p in INPUTS.items()}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})
    b, survival = build_b()
    c, cells = build_c()
    e, molecular = build_e()
    # Slots retain native scale: each small plot occupies one 4.65-inch slot.
    # A and C occupy two slots; D extends through the 0.15-inch row gutter.
    unit, wide, gutter, row_h, row_gap = 4.65, 9.3, .08, 3.45, .15
    width, height = wide+unit*2+gutter*2, row_h*2+row_gap
    placements = {
        'A': (INPUTS['A'], (0, 0, wide, row_h)),
        'B': (b, (wide+gutter, 0, wide+gutter+unit, row_h)),
        'C': (c, (0, row_h+row_gap, wide, height)),
        'E': (e, (wide+gutter, row_h+row_gap, wide+gutter+unit, height)),
        'D': (INPUTS['D'], (wide+unit+gutter*2, 0, width, height)),
    }
    doc = fitz.open()
    page = doc.new_page(width=width*72, height=height*72)
    slot_info = {}
    for label, (path, bounds) in placements.items():
        src = fitz.open(path)
        assert len(src) == 1
        rect = fitz.Rect(*(v*72 for v in bounds))
        np.testing.assert_allclose([rect.width, rect.height], [src[0].rect.width, src[0].rect.height], atol=.01)
        page.show_pdf_page(rect, src, 0)
        slot_info[label] = dict(path=str(path), bounds_inches=bounds, pdf_sha256=sha(path))
        src.close()
    out = HERE / 'panels_ABCDE_two_rows'
    doc.set_metadata({'title': 'HNSCC spatial niche-guided biomarker discovery', 'subject': 'Two-row preview, vertical D and two-column E'})
    doc.save(out.with_suffix('.pdf'), deflate=True, garbage=4)
    page.get_pixmap(dpi=200, alpha=False).save(out.with_suffix('.png'))
    doc.close()
    assert hashes == {k: sha(p) for k, p in INPUTS.items()}
    manifest = dict(inputs={k: dict(path=str(INPUTS[k]), sha256=h) for k, h in hashes.items()}, inputs_unchanged=True,
                    layout='Top A+B; bottom C+E; right D spans both rows', size_inches=[width, height],
                    placements=slot_info, survival_primary_model=survival, umap_cells_by_dataset=cells,
                    umap_coordinates_recomputed=False, survival_refit=False,
                    molecular_group_fraction=.3, molecular_low_cells=1439, molecular_high_cells=1437,
                    molecular_aggregation='Equal-weight mean of three donor means', molecular_features=molecular,
                    biological_interpretation='Molecular phenotype, not causal proof of exhaustion or survival mediation')
    (HERE / 'layout_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(out.with_suffix('.png'))
    print(out.with_suffix('.pdf'))
    print(f'Figure: {width:.2f} x {height:.2f} inches. All original input hashes unchanged.')


if __name__ == '__main__':
    main()
