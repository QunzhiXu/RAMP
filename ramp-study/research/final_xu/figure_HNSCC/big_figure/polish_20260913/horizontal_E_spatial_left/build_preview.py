#!/usr/bin/env python3
"""Layout-only HNSCC trial: top A/B/D, bottom C/F/E. Frozen data, native PDFs."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os

HERE = Path(__file__).resolve().parent
TRIAL = HERE.parent
BIG = TRIAL.parent
ROOT = BIG.parent
os.environ.setdefault('MPLCONFIGDIR', str(HERE / '.cache/matplotlib'))
os.environ.setdefault('MPLBACKEND', 'Agg')
import fitz
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

INK, GRID = '#182027', '#E7EBEE'
LOW, HIGH = '#2B6CB0', '#C9433C'
ORANGE, TEAL = '#D95F02', '#238A8D'
HEIGHT, GUTTER, ROW_GAP = 3.45, .02, .08
WIDTHS = {'A': 8.1, 'B': 4.2, 'D': 4.65, 'C': 7., 'F': 7.35, 'E': 2.6}
DONORS = ['126', 'indiv_1', 'indiv_2']
FEATURES = [('ADT', 'CD223--LAG-3', 'LAG-3'), ('ADT', 'CD127--IL-7Ra', 'CD127'),
            ('ADT', 'CD62L', 'CD62L'), ('RNA', 'CTLA4', 'CTLA4'),
            ('RNA', 'CXCL13', 'CXCL13'), ('RNA', 'IL7R', 'IL7R')]
SCAN = BIG / 'panel_E/group_fraction_scan_20260910'
SPATIAL = ROOT / 'spatial_return_survival'
PLANE_ROOT = SPATIAL / 'input/planes'
RAW = ROOT / 'sampled_data/haist_codex/complete/HNSCCdiscovery_CODEX.csv'
RETURNED = SPATIAL / 'output/haist74_pt_cd8_tigit_cd39_return.csv.gz'
ROI, PATIENT, CROP = '522_S2reg054', '15219', (1205, 205, 1805, 670)
INPUTS = [SCAN / 'donor_contrasts.csv', SCAN / 'equal_donor_contrasts.csv',
          SCAN / 'group_counts.csv', ROOT / 'biomarker_discovery/screen/marker_donor_effects_long.csv',
          ROOT / 'biomarker_discovery/screen/marker_screen_all_107_all_variants.csv',
          ROOT / 'alignment/umap/full_94488_umap_coordinates.csv.gz', RAW, RETURNED,
          SPATIAL / 'output/spatial_validation_manifest.json',
          TRIAL / 'panel_A/panel_A_niche_discovery.pdf',
          BIG / 'panel_B/panel_B_survival_compact.pdf',
          BIG / 'combined_ABCDE_two_rows/build_combined_two_rows.py',
          TRIAL / 'panel_E/plotted_donor_values.csv']
INPUTS += [PLANE_ROOT / f'{m}.raw' for m in ['DAPI', 'CD8', 'FAP', 'CD68']]


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def header(fig, letter, title, font=12):
    w = fig.get_figwidth()
    fig.text(.08/w, .976, letter, fontsize=23, weight='bold', va='top', color=INK)
    fig.text(.51/w, .947, title, fontsize=font, weight='bold', va='top', color=INK)


def save(fig, letter, stem):
    out = HERE / f'panel_{letter}'
    out.mkdir(exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    text_items = list(fig.texts)
    for ax in fig.axes:
        text_items += list(ax.texts) + [ax.xaxis.label, ax.yaxis.label, ax.title, ax._left_title]
        text_items += ax.get_xticklabels() + ax.get_yticklabels()
        if ax.get_legend():
            text_items += ax.get_legend().get_texts()
    for legend in fig.legends:
        text_items += legend.get_texts()
    for t in text_items:
        if t.get_text() and t.get_visible():
            r = t.get_window_extent(renderer)
            assert r.x0 >= -1 and r.y0 >= -1 and r.x1 <= fig.bbox.width+1 and r.y1 <= fig.bbox.height+1, (letter, t.get_text(), tuple(r.bounds))
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(out / f'{stem}.{ext}', dpi=320, facecolor='white')
    plt.close(fig)
    return out / f'{stem}.pdf'


def build_e():
    d = pd.read_csv(SCAN / 'donor_contrasts.csv', dtype={'donor': str})
    d = d[d.fraction.eq(.3)]
    counts = pd.read_csv(SCAN / 'group_counts.csv')
    counts = counts[counts.fraction.eq(.3)]
    assert counts.n_low.sum() == 1439 and counts.n_high.sum() == 1437
    fig = plt.figure(figsize=(WIDTHS['E'], HEIGHT))
    header(fig, 'E', 'CD39/TIGIT\nstate phenotype', font=10.7)
    fig.legend(handles=[Line2D([], [], marker='o', linestyle='', color=c, markersize=3.5, label=l)
                        for c, l in [(LOW, 'Lower 30%'), (HIGH, 'Higher 30%')]],
               loc='center', bbox_to_anchor=(.535, .800), ncol=2, frameon=False, fontsize=8.,
               handlelength=.6, handletextpad=.3, columnspacing=.8)
    ax = fig.add_axes([.285, .153, .675, .568])
    centers = [6.1, 5.1, 4.1, 2.1, 1.1, .1]
    records = []
    for y, (modality, marker, label) in zip(centers, FEATURES):
        b = d[d.modality.eq(modality) & d.marker.eq(marker)].set_index('donor').loc[DONORS]
        assert len(b) == 3 and b.direction_consistent.all()
        for group, offset, color in [('low', .17, LOW), ('high', -.17, HIGH)]:
            values = b[group].to_numpy(float)
            ax.boxplot([values], positions=[y+offset], widths=.24, vert=False, patch_artist=True,
                       manage_ticks=False, whis=(0, 100), showfliers=False,
                       boxprops=dict(facecolor=color, edgecolor=color, alpha=.20, linewidth=.8),
                       medianprops=dict(color=color, linewidth=1.2),
                       whiskerprops=dict(color=color, linewidth=.8), capprops=dict(color=color, linewidth=.8))
            pts = ax.scatter(values, y+offset+np.array([-.055, 0, .055]), s=10,
                             color=color, edgecolor='white', linewidth=.3, zorder=4)
            np.testing.assert_array_equal(pts.get_offsets()[:, 0], values)
            for donor, value in zip(DONORS, values):
                records.append(dict(modality=modality, marker=marker, label=label, donor=donor,
                                    group=group, value=float(value)))
    ax.set(xlim=(0, 100), ylim=(-.6, 7.25), xticks=[0, 50, 100])
    ax.set_yticks(centers, [f[2] for f in FEATURES])
    for tick, f in zip(ax.get_yticklabels(), FEATURES):
        tick.set_fontstyle('italic' if f[0] == 'RNA' else 'normal')
    ax.tick_params(axis='y', labelsize=8.7, length=0, pad=4)
    ax.tick_params(axis='x', labelsize=8., length=2.5, pad=3)
    ax.xaxis.grid(True, color=GRID, linewidth=.6)
    ax.axhline(3.28, color='#D2D9DE', lw=.7)
    ax.text(0, 7.0, 'Protein', fontsize=9.1, weight='bold', va='center')
    ax.text(0, 3.0, 'RNA', fontsize=9.1, weight='bold', va='center',
            bbox=dict(facecolor='white', edgecolor='none', pad=.3))
    ax.set_axisbelow(True)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_color('#7A838A')
    ax.set_xlabel('Percentile / detected (%)', fontsize=8.4, labelpad=4)
    path = save(fig, 'E', 'panel_E_horizontal')
    values = pd.DataFrame(records)
    old = pd.read_csv(TRIAL / 'panel_E/plotted_donor_values.csv', dtype={'donor': str})
    keys = ['modality', 'marker', 'donor', 'group']
    a, b = values.set_index(keys).sort_index(), old.set_index(keys).sort_index()
    np.testing.assert_array_equal(a.value, b.value)
    values.to_csv(path.parent / 'plotted_donor_values.csv', index=False)
    return path


def build_d():
    source = ROOT / 'biomarker_discovery/screen'
    d = pd.read_csv(source / 'marker_donor_effects_long.csv', dtype={'donor': str})
    screen = pd.read_csv(source / 'marker_screen_all_107_all_variants.csv')
    features = [('TIGIT--VSTM3', 'TIGIT'), ('CD39', 'CD39'), ('CD224', 'CD224'),
                ('CD134--OX40', 'OX40'), ('TCR-A--TCR-B', 'TCRαβ'), ('CD83-1', 'CD83')]
    assert len(screen) == 107
    assert set(screen[screen.selected_compact_top5pct].marker_raw) == {'CD39', 'TIGIT--VSTM3'}
    d = d[d.variant.eq('paper_clr') & d.marker_raw.isin(dict(features))].copy()
    assert len(d) == 18
    fig = plt.figure(figsize=(WIDTHS['D'], HEIGHT))
    header(fig, 'D', 'BIOMARKER DISCOVERY', font=12.)
    fig.text(.54, .810, '107 proteins screened | 3 donors', ha='center', fontsize=9.5)
    ax = fig.add_axes([.154, .225, .811, .505])
    for x, (marker, name) in enumerate(features):
        b = d[d.marker_raw.eq(marker)].set_index('donor').loc[DONORS]
        v, color = b.association.to_numpy(float), ORANGE if x < 2 else TEAL
        ax.boxplot([v], positions=[x], widths=.47, manage_ticks=False,
                   patch_artist=True, showfliers=False, whis=(0, 100),
                   boxprops=dict(facecolor=color, edgecolor=color, alpha=.18, linewidth=1.),
                   medianprops=dict(color=color, linewidth=1.4),
                   whiskerprops=dict(color=color, linewidth=1.), capprops=dict(color=color, linewidth=1.))
        ax.scatter(x+np.array([-.10, 0, .10]), v, s=26, color=color, edgecolor='white', linewidth=.5, zorder=4)
    ax.set(xlim=(-.6, 5.6), ylim=(0, .36), yticks=[0, .1, .2, .3])
    ax.set_xticks(range(6), [f[1] for f in features])
    for t, c in zip(ax.get_xticklabels(), [ORANGE]*2+[TEAL]*4):
        t.set_color(c)
        t.set_weight('bold')
    ax.set_ylabel('Association with niche score\n(Spearman ρ)', fontsize=9.3, labelpad=5)
    ax.tick_params(axis='x', labelsize=9., length=0, pad=7)
    ax.tick_params(axis='y', labelsize=8.5, length=3)
    ax.yaxis.grid(True, color=GRID, lw=.6)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    for s in ['left', 'bottom']:
        ax.spines[s].set_color('#7A838A')
    fig.legend(handles=[Line2D([], [], marker='o', linestyle='', color=c, markersize=4.2, label=l)
                        for c,l in [(ORANGE,'Selected pair'),(TEAL,'Other candidates')]],
               loc='center', bbox_to_anchor=(.54,.095), ncol=2, frameon=False, fontsize=8.6,
               handlelength=.6, handletextpad=.4, columnspacing=1.2)
    path = save(fig, 'D', 'panel_D_donor_boxes')
    d[['marker_raw','donor','association']].to_csv(path.parent / 'plotted_donor_values.csv', index=False)
    return path


def build_c():
    # Reuse the frozen coordinate reader, category colors and drawing order.
    src = BIG / 'combined_ABCDE_two_rows/build_combined_two_rows.py'
    spec = importlib.util.spec_from_file_location('frozen_hnscc_layout', src)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    old.WIDE_WIDTH = WIDTHS['C']
    old.save = save  # All outputs redirected into this new trial folder.
    old.header = lambda fig, letter, title, wide=False: header(fig, letter, title, font=11.5)
    path, counts = old.build_c()
    assert sum(counts.values()) == 94488
    return path, counts


def spatial_data():
    # Identical global channel scales, gamma and blend as the frozen original.
    rgb = np.zeros((1440,1920,3), dtype=np.float32)
    specs = [('DAPI',99.80,.72,[.20,.22,.34]), ('CD8',99.55,.70,[0,.78,.92]),
             ('FAP',99.70,.74,[0,.76,0]), ('CD68',99.55,.70,[.88,0,.70])]
    for marker, percentile, gamma, color in specs:
        a = np.fromfile(PLANE_ROOT / f'{marker}.raw', dtype='>u2').reshape(1440,1920).astype(np.float32)
        upper = float(np.percentile(a[a > 0], percentile))
        a = np.power(np.clip(a / max(upper,1.), 0, 1), gamma)
        rgb += a[...,None] * np.array(color, dtype=np.float32)
    rgb = np.clip(np.power(np.clip(rgb,0,1), .93),0,1) * .60
    x0,y0,x1,y1 = CROP
    d = pd.read_csv(RETURNED, dtype={'patient': str})
    d = d[d.patient.eq(PATIENT) & d.unique_region.eq(ROI)].copy()
    d = d[d.x.between(x0,x1,inclusive='left') & d.y.between(y0,y1,inclusive='left')].copy()
    d['in_niche'], d['high_state'] = d.spatial_y.gt(0), d.within_patient_rank.ge(.6)
    assert [len(d), int(d.in_niche.sum()), int(d.high_state.sum()), int((d.in_niche & d.high_state).sum())] == [61,32,28,22]
    blocks = []
    for chunk in pd.read_csv(RAW, usecols=['unique_region','x','y','cell_types_mediumres'], chunksize=200_000):
        hit = chunk[chunk.unique_region.eq(ROI) & chunk.cell_types_mediumres.isin(['CAF','macrophages'])]
        hit = hit[hit.x.between(x0,x1,inclusive='left') & hit.y.between(y0,y1,inclusive='left')]
        if len(hit):
            blocks.append(hit)
    architecture = pd.concat(blocks, ignore_index=True)
    assert architecture.cell_types_mediumres.value_counts().to_dict() == {'macrophages': 56, 'CAF': 27}
    return rgb[y0:y1,x0:x1], d, architecture


def build_f():
    rgb, d, architecture = spatial_data()
    fig = plt.figure(figsize=(WIDTHS['F'], HEIGHT))
    header(fig, 'F', 'SPATIAL CORRESPONDENCE', font=12.)
    # Equal coordinate geometry; only membership of the yellow highlighted set changes.
    w, left, gap = 3.48, .065, .10
    h = w * 465/600
    y = .21
    axes = [fig.add_axes([x/WIDTHS['F'], y/HEIGHT, w/WIDTHS['F'], h/HEIGHT])
            for x in [left,left+w+gap]]
    for ax, state, title in zip(axes, ['in_niche','high_state'],
                               ['Observed spatial niche','Returned CD39/TIGIT state']):
        ax.imshow(rgb, extent=(CROP[0],CROP[2],CROP[3],CROP[1]), origin='upper', interpolation='nearest')
        for kind, color in [('CAF','#39EF72'),('macrophages','#FF43B5')]:
            b = architecture[architecture.cell_types_mediumres.eq(kind)]
            ax.scatter(b.x,b.y,s=4,color=color,alpha=.75,linewidths=0,zorder=3)
        ax.scatter(d.x,d.y,s=3.4,color='#B8BEC7',alpha=.52,linewidths=0,zorder=4)
        b = d[d[state]]
        ax.scatter(b.x,b.y,s=21,color='#FFD400',edgecolor='#241A00',linewidth=.65,zorder=5)
        ax.set(xlim=(CROP[0],CROP[2]),ylim=(CROP[3],CROP[1]),xticks=[],yticks=[])
        ax.set_title(title, fontsize=9.5, weight='bold', loc='left', pad=4)
        for s in ax.spines.values():
            s.set_visible(False)
    handles = [Line2D([],[],marker='o',linestyle='',color=color,markersize=3.6,label=name)
               for name,color in [('Highlighted CD8','#E4B900'),('Other CD8','#B8BEC7'),('CAF','#39C76B'),('Macrophage','#DF409D')]]
    fig.legend(handles=handles, loc='center', bbox_to_anchor=(.52,.025), ncol=4, frameon=False,
               fontsize=8., handlelength=.6, handletextpad=.35, columnspacing=1.1)
    path = save(fig,'F','panel_F_spatial_correspondence')
    d.to_csv(path.parent / 'plotted_cd8_cells.csv', index=False)
    architecture.to_csv(path.parent / 'plotted_caf_macrophage_cells.csv', index=False)
    return path


def compose(paths):
    width = sum(WIDTHS[k] for k in ['A','B','D']) + 2*GUTTER
    total_height = 2*HEIGHT+ROW_GAP
    assert abs(width - sum(WIDTHS[k] for k in ['C','F','E']) - 2*GUTTER) < 1e-10
    doc = fitz.open()
    page = doc.new_page(width=72*width, height=72*total_height)
    slots = {}
    for row, labels in enumerate([['A','B','D'],['C','F','E']]):
        x, y = 0., row*(HEIGHT+ROW_GAP)
        for label in labels:
            bounds = [x,y,x+WIDTHS[label],y+HEIGHT]
            rect = fitz.Rect(*[72*v for v in bounds])
            with fitz.open(paths[label]) as src:
                np.testing.assert_allclose([rect.width,rect.height],[src[0].rect.width,src[0].rect.height],atol=.01)
                page.show_pdf_page(rect,src,0)
            slots[label] = dict(bounds_inches=bounds,source=str(paths[label]),sha256=sha(paths[label]))
            x += WIDTHS[label]+GUTTER
    stem = HERE / 'hnscc_horizontal_E_spatial_left'
    doc.set_metadata({'title':'HNSCC layout trial: spatial correspondence before molecular phenotype'})
    doc.save(stem.with_suffix('.pdf'),garbage=4,deflate=True)
    page.get_pixmap(dpi=160,alpha=False).save(stem.with_suffix('.png'))
    page.get_pixmap(dpi=240,alpha=False,clip=fitz.Rect(0,(HEIGHT+ROW_GAP)*72,width*72,total_height*72)).save(HERE / 'second_row_preview.png')
    doc.close()
    return slots


def main():
    before = {str(p):sha(p) for p in INPUTS}
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','text.color':INK})
    paths = {'A':TRIAL / 'panel_A/panel_A_niche_discovery.pdf', 'B':BIG / 'panel_B/panel_B_survival_compact.pdf'}
    paths['E'] = build_e()
    print('Horizontal E: all 36 donor values unchanged.',flush=True)
    paths['D'] = build_d()
    paths['C'], counts = build_c()
    print('D compact; C uses all 94,488 unchanged coordinates.',flush=True)
    paths['F'] = build_f()
    print('Real CODEX pair: 61 CD8; sets 32/28; overlap 22.',flush=True)
    slots = compose(paths)
    assert all(sha(Path(p)) == h for p,h in before.items())
    audit = dict(status='PASS',inputs_sha256=before,inputs_unchanged=True,placements=slots,
                 trial_labels_retained=True,layout='A B D / C F E',size_inches=[16.99,6.98],
                 molecular_group_fraction=.3,molecular_boxes=12,molecular_donor_points=36,
                 molecular_values_identical_to_vertical_trial=True,
                 protein_unit='Mean within-donor expression percentile',rna_unit='RNA detected (%)',
                 box_center='Median of three donor summaries',whiskers='Full donor min/max',
                 spatial_roi=ROI,spatial_patient=PATIENT,spatial_crop=CROP,
                 spatial_high_state='within_patient_rank >= 0.60 (top 40%)',
                 spatial_counts=dict(cd8=61,in_niche=32,high_state=28,overlap=22),
                 identical_spatial_background=True,roi_or_threshold_selection_rerun=False,
                 umap_counts=counts,umap_coordinates_recomputed=False,models_refit=False,
                 formal_figure_replaced=False,manuscript_replaced=False)
    (HERE / 'LAYOUT_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
    print('PASS: sources unchanged; new trial only.',flush=True)


if __name__ == '__main__':
    main()
