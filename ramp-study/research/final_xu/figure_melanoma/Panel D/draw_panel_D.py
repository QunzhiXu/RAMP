"""Horizontal melanoma workflow preview; outputs stay inside Panel D."""
from pathlib import Path
import os
import sys
import shutil
import hashlib
import json
import argparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ASSETS = HERE / 'assets'
ASSETS.mkdir(exist_ok=True)
os.environ['MPLCONFIGDIR'] = str(HERE / '.mpl-cache')
os.environ['MPLBACKEND'] = 'Agg'
sys.dont_write_bytecode = True

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, PathPatch
from matplotlib.path import Path as MplPath

SOURCE = ROOT / 'story_lung_HLA_ABC/Figure/Panel A/assets/ramp_publication_icon.png'
SOURCE_RECORD = SOURCE.with_name('SOURCE.json')
ICON = ASSETS / SOURCE.name


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(version=4):
    for source in [SOURCE, SOURCE_RECORD]:
        destination = ASSETS / source.name
        if destination.exists():
            assert sha(source) == sha(destination)
        else:
            shutil.copyfile(source, destination)

    alignment_path = HERE.parent / 'analysis/alignment/output/manifest.json'
    alignment_hash = sha(alignment_path)
    alignment = json.loads(alignment_path.read_text())
    assert alignment['marker_count'] == 15 and alignment['cells'] == 165686
    assert sum(alignment['dataset_cells'].values()) == alignment['cells']
    donor_count = sum(alignment['dataset_patients'].values())
    assert donor_count == 38
    previous_design = HERE / 'DESIGN.json'
    if previous_design.exists():
        previous = json.loads(previous_design.read_text())
        saved = HERE / f"DESIGN_v{previous['version']}.json"
        if not saved.exists():
            shutil.copyfile(previous_design, saved)

    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10.5,
        'svg.fonttype': 'none', 'savefig.facecolor': 'white',
        'text.color': '#252E35',
    })
    W, H = 6.16, 2.95
    colors = {'ink': '#252E35', 'edge': '#C4CDD3', 'arrow': '#53616C',
              'imc': '#3F78B5', 'cite': '#E68A2E', 'validation': '#7355A3'}
    fig = plt.figure(figsize=(W, H), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, W), ylim=(H, 0), aspect='equal')
    ax.axis('off')

    def label(x, y, value, size=10.5, color=None, weight='bold', **kwargs):
        return ax.text(x, y, value, ha='center', va='center',
                       fontsize=size, color=color or colors['ink'],
                       fontweight=weight, linespacing=1.25, zorder=5, **kwargs)

    def box(x, y, w, h, fill='white', edge=None):
        patch = FancyBboxPatch((x, y), w, h,
                              boxstyle='round,pad=0,rounding_size=0.07',
                              facecolor=fill, edgecolor=edge or colors['edge'],
                              linewidth=1.15, zorder=3)
        ax.add_patch(patch)
        return patch

    def arrow(start, end, color=None):
        patch = FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=9,
                                linewidth=1.25, color=color or colors['arrow'],
                                shrinkA=2, shrinkB=2, zorder=2)
        ax.add_patch(patch)

    ax.text(.10, .18, 'D.  Cross-platform spatial-state transfer',
            ha='left', va='center', fontsize=13.5, fontweight='bold')

    # Platform identities describe the shared alignment inputs.
    main_y = 1.49
    platform_x, platform_w = .15, 1.03
    if version == 1:
        box(platform_x, main_y-.47, platform_w, .94)
        ax.plot([platform_x+.10, platform_x+platform_w-.10], [main_y, main_y],
                color='#DEE4E8', lw=.8, zorder=4)
        label(platform_x+platform_w/2, main_y-.235, 'IMC', size=11.5, color=colors['imc'])
        label(platform_x+platform_w/2, main_y+.235, 'CITE-seq', size=10.8, color=colors['cite'])
    else:
        platform_centers = [1.12, 1.93]
        for cy, name, color, size in [
            (platform_centers[0], 'IMC', colors['imc'], 11.5),
            (platform_centers[1], 'CITE-seq', colors['cite'], 10.8),
        ]:
            box(platform_x, cy-.23, platform_w, .46)
            label(platform_x+platform_w/2, cy, name, size=size, color=color)

    # Reuse the publication bitmap without altering it. Hide only the old
    # pipe stubs with the same vector clipping geometry used in the lung figure.
    logo = plt.imread(ICON)
    ih, iw = logo.shape[:2]
    logo_x, logo_w = 1.46, 1.28
    logo_h = logo_w*ih/iw
    logo_y = main_y-.36*logo_h
    scale = logo_w/iw
    im = ax.imshow(logo, extent=(logo_x, logo_x+logo_w, logo_y+logo_h, logo_y), zorder=4)
    clip_pixels = np.array([
        [0, 0], [iw, 0], [iw, 170], [1458, 170], [1458, 430],
        [iw, 430], [iw, ih], [0, ih], [0, 430], [26, 430],
        [26, 170], [0, 170], [0, 0],
    ])
    clip_vertices = clip_pixels*scale+[logo_x, logo_y]
    im.set_clip_path(PathPatch(MplPath(clip_vertices), transform=ax.transData))
    if version == 1:
        label(logo_x+logo_w/2, 2.06, '15 shared proteins', size=9.2, weight='normal')
        arrow((platform_x+platform_w+.01, main_y), (logo_x+.01, main_y))
    else:
        label(logo_x+logo_w/2, 2.06,
              f"{alignment['marker_count']} shared proteins", size=9.2)
        if version < 4:
            label(logo_x+logo_w/2, 2.31, 'Align all cells', size=9.2, weight='normal')
        if version == 2:
            label(logo_x+logo_w/2, 2.53,
                  f"{alignment['cells']:,} cells", size=9.2, weight='normal')
        else:
            count_y = .86 if version >= 4 else .66
            label(logo_x+logo_w/2, count_y,
                  f"{alignment['cells']:,} cells", size=10.1, weight='normal')
            label(logo_x+logo_w/2, count_y+.24,
                  f'{donor_count} donors', size=10.1, weight='normal')
        for cy, target_y in zip(platform_centers, [main_y-.12, main_y+.12]):
            arrow((platform_x+platform_w+.01, cy), (logo_x+.015, target_y))

    model_x, model_w, model_y, model_h = 3.04, 1.41, 1.07, .84
    model_cx = model_x+model_w/2
    box(model_x, model_y, model_w, model_h, fill='#F2F6FA')
    label(model_cx, model_y+.27, 'Spatial-state\nmodel', size=11)
    label(model_cx, model_y+.67, 'Train on IMC CD8', size=8.9,
          color=colors['imc'], weight='normal')
    arrow((logo_x+logo_w+.01, main_y), (model_x-.01, main_y))

    # The supervised endpoint is supplied only by the source IMC cohort.
    source_w, source_h, source_y = 1.40, .43, .48
    box(model_cx-source_w/2, source_y, source_w, source_h)
    label(model_cx, source_y+source_h/2, 'IMC spatial labels', size=9.2)
    arrow((model_cx, source_y+source_h), (model_cx, model_y))

    query_x, query_w = 4.81, 1.20
    box(query_x, model_y, query_w, model_h)
    label(query_x+query_w/2, model_y+.25, 'CD8 state\nscores', size=10.5)
    label(query_x+query_w/2, model_y+.65, 'CITE-seq', size=10.4, color=colors['cite'])
    arrow((model_x+model_w+.01, main_y), (query_x-.01, main_y))

    validation_y, validation_w, validation_h = 2.25, 1.86, .52
    box(model_cx-validation_w/2, validation_y, validation_w, validation_h,
        fill='#F7F4FB')
    label(model_cx, validation_y+validation_h/2,
          'Patient-held-out\nvalidation', size=10.0)
    arrow((model_cx, model_y+model_h), (model_cx, validation_y), color=colors['validation'])

    for extension in ['png', 'svg']:
        fig.savefig(HERE / f'panel_D_workflow_horizontal_v{version}.{extension}', dpi=320,
                    facecolor='white')
    plt.close(fig)
    design = {
        'version': version, 'canvas_inches': [W, H],
        'title': 'Cross-platform spatial-state transfer',
        'icon_source': str(SOURCE), 'icon_sha256': sha(ICON),
        'icon_bitmap_unmodified': sha(SOURCE) == sha(ICON),
        'platform_boxes': 'two separate boxes, each with its own arrow to RAMP' if version >= 2 else 'one divided box',
        'alignment_scope_shown': version >= 2,
        'alignment_cells': alignment['cells'],
        'alignment_donors': donor_count,
        'cell_donor_counts_above_RAMP': version >= 3,
        'cell_donor_counts_lowered_inches': .20 if version >= 4 else 0,
        'align_all_cells_text_shown': version in [2, 3],
        'alignment_marker_count': alignment['marker_count'],
        'alignment_manifest': str(alignment_path),
        'alignment_manifest_sha256': alignment_hash,
        'main_steps': ['IMC and CITE-seq', 'RAMP alignment with 15 shared proteins',
                       'Spatial-state model trained on IMC CD8', 'CITE-seq CD8 state scores'],
        'source_label': 'Tc.ae AND prob.Ki67 >= 0.5 AND nearest melanoma centroid <= 30 um',
        'model': 'StandardScaler + balanced LogisticRegression(C=0.5)',
        'validation': 'Leave each source IMC patient out of classifier training in turn',
        'deployment': 'Refit on all source CD8 cells; score all 5768 target CITE-seq CD8 cells',
        'artwork_text': 'Abbreviated; exact label/model/fitting details are in README.',
        'source_scripts_read_only': [
            '../analysis/all_cd8_prediction/run_all_cd8_patient_heldout.py',
            '../analysis/all_cd8_prediction/build_all_cd8_citeseq_projection.py'],
        'existing_figure_modified': False,
    }
    assert sha(alignment_path) == alignment_hash
    for filename in ['DESIGN.json', f'DESIGN_v{version}.json']:
        (HERE / filename).write_text(json.dumps(design, indent=2, ensure_ascii=False)+'\n')
    print(HERE / f'panel_D_workflow_horizontal_v{version}.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=int, choices=[1, 2, 3, 4], default=4)
    main(parser.parse_args().version)
