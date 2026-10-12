"""Rotate the accepted donor summaries; keep values, scales, and other panels."""
from pathlib import Path
import hashlib
import json
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/hnscc_vertical_donor_E')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import fitz

HERE = Path(__file__).resolve().parent
TRIAL = HERE.parent
OLD_LAYOUT = TRIAL/'larger_LAG3_spatial/LAYOUT_AUDIT.json'
VALUES = TRIAL/'horizontal_E_spatial_left/panel_E/plotted_donor_values.csv'
LOW, HIGH, INK, GRID = '#2B6CB0', '#C9433C', '#182027', '#E7EBEE'
DONORS = ['126', 'indiv_1', 'indiv_2']


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def build_e():
    d = pd.read_csv(VALUES, dtype={'donor': str})
    assert len(d) == 36 and d.groupby(['modality', 'marker', 'group']).size().eq(3).all()
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none', 'text.color': INK})
    W, H = 2.6, 3.45
    fig = plt.figure(figsize=(W, H), facecolor='white')
    fig.text(.08/W, .976, 'E', fontsize=23, weight='bold', va='top')
    fig.text(.51/W, .947, 'CD39/TIGIT\nstate phenotype', fontsize=10.7, weight='bold', va='top')
    fig.legend(handles=[Line2D([], [], marker='o', linestyle='', color=c, markersize=3.5, label=l)
                        for c, l in [(LOW, 'Lower 30%'), (HIGH, 'Higher 30%')]],
               loc='center', bbox_to_anchor=(.55, 2.755/H), ncol=2, frameon=False, fontsize=8,
               handlelength=.6, handletextpad=.3, columnspacing=.8)
    records = []
    specifications = [
        ('ADT', 'Protein', ['LAG-3', 'CD127', 'CD62L'], 1.73, .78, 'Mean percentile'),
        ('RNA', 'RNA', ['CTLA4', 'CXCL13', 'IL7R'], .40, .86, 'Detected (%)')]
    for modality, title, markers, bottom, height, ylabel in specifications:
        ax = fig.add_axes([.51/W, bottom/H, 2.01/W, height/H])
        for x, label in enumerate(markers):
            for group, dx, color in [('low', -.19, LOW), ('high', .19, HIGH)]:
                b = d[d.modality.eq(modality) & d.label.eq(label) & d.group.eq(group)].set_index('donor').loc[DONORS]
                v = b.value.to_numpy(float)
                ax.boxplot([v], positions=[x+dx], widths=.25, patch_artist=True,
                           manage_ticks=False, whis=(0, 100), showfliers=False,
                           boxprops=dict(facecolor=color, edgecolor=color, alpha=.20, linewidth=.8),
                           medianprops=dict(color=color, linewidth=1.2),
                           whiskerprops=dict(color=color, linewidth=.8), capprops=dict(color=color, linewidth=.8))
                pts = ax.scatter(x+dx+np.array([-.055, 0, .055]), v, s=10, color=color,
                                 edgecolor='white', linewidth=.3, zorder=4)
                np.testing.assert_array_equal(pts.get_offsets()[:, 1], v)
                records.extend(b.reset_index().to_dict('records'))
        ax.set(xlim=(-.57, 2.57), ylim=(0, 100), yticks=[0, 50, 100])
        ax.set_xticks([0, 1, 2], markers)
        ax.set_title(title, fontsize=9.1, weight='bold', loc='left', pad=4)
        ax.set_ylabel(ylabel, fontsize=8, labelpad=3)
        ax.tick_params(axis='x', labelsize=8.3, length=0, pad=4)
        ax.tick_params(axis='y', labelsize=7.5, length=2, pad=2)
        if modality == 'RNA':
            for t in ax.get_xticklabels():
                t.set_fontstyle('italic')
        ax.yaxis.grid(True, color=GRID, linewidth=.6)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
        for side in ['bottom', 'left']:
            ax.spines[side].set_color('#7A838A')
            ax.spines[side].set_linewidth(.7)
    plotted = pd.DataFrame(records)
    keys = ['modality', 'marker', 'donor', 'group']
    pd.testing.assert_frame_equal(plotted.set_index(keys).sort_index()[['label', 'value']],
                                  d.set_index(keys).sort_index()[['label', 'value']])
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = list(fig.texts)
    for ax in fig.axes:
        texts += [ax.yaxis.label, ax._left_title]+ax.get_xticklabels()+ax.get_yticklabels()
    for legend in fig.legends:
        texts += legend.get_texts()
    for t in texts:
        box = t.get_window_extent(renderer)
        assert box.x0 >= -1 and box.y0 >= -1 and box.x1 <= fig.bbox.width+1 and box.y1 <= fig.bbox.height+1, t.get_text()
    out = HERE/'panel_E'
    out.mkdir(exist_ok=True)
    stem = out/'panel_E_vertical_donors'
    for ext in ['pdf', 'png', 'svg']:
        fig.savefig(stem.with_suffix('.'+ext), dpi=320, facecolor='white')
    plt.close(fig)
    plotted.to_csv(out/'plotted_donor_values.csv', index=False)
    with fitz.open(stem.with_suffix('.pdf')) as doc:
        doc[0].get_pixmap(dpi=250, alpha=False).save(out/'pdf_preview.png')
    return stem.with_suffix('.pdf')


def main():
    old = json.loads(OLD_LAYOUT.read_text())
    sources = [VALUES, OLD_LAYOUT]+[Path(p['source']) for p in old['placements'].values()]
    hashes = {str(p): sha(p) for p in sources}
    replacement = build_e()
    doc = fitz.open()
    page = doc.new_page(width=16.99*72, height=6.98*72)
    placements = {}
    for label, slot in old['placements'].items():
        source = replacement if label == 'E' else Path(slot['source'])
        rect = fitz.Rect(*[x*72 for x in slot['bounds_inches']])
        with fitz.open(source) as panel:
            np.testing.assert_allclose([rect.width, rect.height], [panel[0].rect.width, panel[0].rect.height], atol=.01)
            page.show_pdf_page(rect, panel, 0)
        placements[label] = dict(source=str(source), sha256=sha(source), bounds_inches=slot['bounds_inches'])
    doc.set_metadata({'title': 'HNSCC: vertical donor summaries with 343-cell LAG3 spatial example'})
    stem = HERE/'hnscc_vertical_donor_E'
    doc.save(stem.with_suffix('.pdf'), garbage=4, deflate=True)
    doc.close()
    with fitz.open(stem.with_suffix('.pdf')) as check:
        assert len(check) == 1
        text = check[0].get_text()
        assert 'Mean percentile' in text and 'Detected (%)' in text and 'mean Spearman' in text
        check[0].get_pixmap(dpi=180, alpha=False).save(stem.with_suffix('.png'))
        check[0].get_pixmap(dpi=220, alpha=False, clip=fitz.Rect(0, 3.53*72, 16.99*72, 6.98*72)).save(HERE/'second_row_preview.png')
    assert hashes == {p: sha(Path(p)) for p in hashes}
    (HERE/'LAYOUT_AUDIT.json').write_text(json.dumps(dict(
        status='PASS', placements=placements, source_layout=str(OLD_LAYOUT),
        changes='Panel E rotated to vertical boxes in two stacked modality axes',
        source_values_sha256=hashes[str(VALUES)], donor_points=36, boxes=12,
        all_donor_values_unchanged=True, axis_ranges_unchanged=[0, 100],
        protein_quantity='Mean within-donor expression percentile', rna_quantity='RNA detected (%)',
        other_panels_reused_without_changes=True, panel_F=old['panel_F'],
        size_inches=[16.99, 6.98], inputs_unchanged=hashes,
        formal_figure_replaced=False, manuscript_replaced=False), indent=2)+'\n')
    (HERE/'README_ZH.md').write_text('''# HNSCC：Panel E 竖向 donor 汇总试稿

整图：`hnscc_vertical_donor_E.png/.pdf`。Panel E 单图位于 `panel_E/`。

E 改为 Protein 在上、RNA 在下的两张竖向箱线图，每个小图的三个 marker 沿横轴排列。每个 marker/组仍为原三个 donor 的汇总值，共 36 个点、12 个箱体。上下轴均保留 0–100 范围，数值逐项与横向版一致。

Protein 为每位 donor 各组的平均表达百分位；RNA 为每位 donor 各组的检出百分比。两种测量分别标在各自纵轴上。原高低 30% 分组、细胞名单、均值与检出比例均未更改。

其余面板直接复用原 PDF；F 保留 343 个 CD8 的真实 DAPI 空间双图和 74 patients mean Spearman ρ = 0.52。整图尺寸及面板占位不变。原横向版、探索版和 manuscript 保留。
''')
    print('PASS: vertical Panel E and complete figure; all 36 donor values and other panels unchanged.')


if __name__ == '__main__':
    main()
