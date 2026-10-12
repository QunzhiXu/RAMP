"""Compact Panel A: connect the existing RAMP gantry to platform and use blocks."""
from runtime import BASE, STORY
import datetime
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, PathPatch
from matplotlib.path import Path as MplPath
import numpy as np
from PIL import Image

W, H = 6.78, 3.45
C = {
    'ink': '#243745', 'rail': '#263138', 'rail_highlight': '#778087',
    'border': '#c8d1d5', 'blue': '#487c9d', 'green': '#368f83',
    'pink': '#c35b75', 'green_bg': '#eff7f4', 'pink_bg': '#fcf2f5',
}
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'svg.fonttype': 'none', 'savefig.facecolor': 'white',
})
fig, ax = plt.subplots(figsize=(W, H))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax.set(xlim=(0, W), ylim=(H, 0))
ax.axis('off')


def label(x, y, text, size=11, color=None, weight='bold', **kwargs):
    return ax.text(x, y, text, fontsize=size, color=color or C['ink'],
                   fontweight=weight, ha='center', va='center',
                   linespacing=1.4, zorder=5, **kwargs)


def box(x, y, w, h, fill='white', edge=None, radius=.10):
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle=f'round,pad=0,rounding_size={radius}',
        facecolor=fill, edgecolor=edge or C['border'], linewidth=1.1, zorder=3,
    )
    ax.add_patch(patch)
    return patch


def rail(points, width=2.1):
    """Dark rails with the same subtle metallic highlight as the earlier figure."""
    a = np.asarray(points)
    for color, lw in [(C['rail'], width), (C['rail_highlight'], .55)]:
        ax.plot(a[:, 0], a[:, 1], color=color, lw=lw,
                solid_capstyle='butt', solid_joinstyle='round', zorder=2)


label(.28, .31, 'A', size=22)

# One compact platform block provides a common border for the two gantry rails.
# Explicitly identify CyTOF as the source of the transferred HLA-ABC state.
platform_x, platform_y, platform_w, platform_h = .40, .70, 1.30, 2.40
box(platform_x, platform_y, platform_w, platform_h)
row_h = platform_h / 3
for i, (text, color) in enumerate([
    ('CyTOF', C['blue']), ('mIF', C['green']), ('IMC', C['pink']),
]):
    cy = platform_y + (i + .5) * row_h
    label(platform_x + platform_w / 2, cy - (.12 if i == 0 else 0),
          text, size=13, color=color)
    if i == 0:
        label(platform_x + platform_w / 2, cy + .17,
              'HLA-ABC state', size=8.6, color=color, weight='normal')
    if i < 2:
        yy = platform_y + (i + 1) * row_h
        ax.plot([platform_x + .12, platform_x + platform_w - .12], [yy, yy],
                color='#dce3e6', lw=.8, zorder=4)

# Keep the sign artwork at its original scale; clip off both raster pipe stubs.
# The replacement vector rails extend underneath the sign's dark outer border.
with Image.open(BASE / 'assets/ramp_publication_icon.png') as im:
    logo = np.asarray(im.convert('RGB'))
logo_x, logo_y, logo_w = 2.08, 1.35, 1.78
logo_h = logo_w * logo.shape[0] / logo.shape[1]
rail_y = [logo_y + logo_h * f for f in (.286, .450)]
pixel_scale = logo_w / logo.shape[1]
sign_left_px, sign_right_px = 26, 1458
sign_left = logo_x + sign_left_px * pixel_scale
sign_right = logo_x + sign_right_px * pixel_scale
pipe_span = sign_left - (platform_x + platform_w)
branch_x = sign_right + pipe_span
pipe_clip_top_px, pipe_clip_bottom_px = 170, 430
for yy in rail_y:
    rail([(platform_x + platform_w, yy),
          (logo_x + (sign_left_px + 4) * pixel_scale, yy)])
    rail([(logo_x + (sign_right_px - 4) * pixel_scale, yy), (branch_x, yy)])
logo_artist = ax.imshow(
    logo, extent=(logo_x, logo_x + logo_w, logo_y + logo_h, logo_y),
    aspect='equal', zorder=4,
)
ih, iw = logo.shape[:2]
clip_pixels = np.array([
    [0, 0], [iw, 0], [iw, pipe_clip_top_px],
    [sign_right_px, pipe_clip_top_px], [sign_right_px, pipe_clip_bottom_px],
    [iw, pipe_clip_bottom_px], [iw, ih], [0, ih],
    [0, pipe_clip_bottom_px], [sign_left_px, pipe_clip_bottom_px],
    [sign_left_px, pipe_clip_top_px], [0, pipe_clip_top_px], [0, 0],
])
clip_vertices = clip_pixels * pixel_scale + [logo_x, logo_y]
logo_artist.set_clip_path(PathPatch(MplPath(clip_vertices), transform=ax.transData))

# Place the transfer step directly beneath the raised RAMP icon.
transfer_cx = (sign_left + sign_right) / 2
transfer_y, transfer_w, transfer_h = 2.48, 2.08, .52
ax.plot([transfer_cx, transfer_cx], [logo_y + logo_h, transfer_y],
        color=C['border'], lw=1.0, zorder=2)
box(transfer_cx - transfer_w / 2, transfer_y, transfer_w, transfer_h,
    edge=C['blue'], radius=.08)
label(transfer_cx, transfer_y + transfer_h / 2,
      'HLA-ABC state transfer', size=9.5)

# Two concise, parallel uses of the same transfer. Detailed validation readouts
# and downstream analyses are described in the caption and later figure panels.
output_x, output_w = branch_x + .24, 1.92
validation_y, validation_h = .82, .70
study_y, study_h = 2.25, .70
validation_cy = validation_y + validation_h / 2
study_cy = study_y + study_h / 2
rail([(output_x, validation_cy), (branch_x, validation_cy),
      (branch_x, study_cy), (output_x, study_cy)], width=1.5)
box(output_x, validation_y, output_w, validation_h,
    fill=C['green_bg'], edge=C['green'])
label(output_x + output_w / 2, validation_cy,
      'Validate in mIF', size=11.3, color=C['green'])
box(output_x, study_y, output_w, study_h,
    fill=C['pink_bg'], edge=C['pink'])
label(output_x + output_w / 2, study_cy,
      'vCAF analysis in IMC', size=10.3, color=C['pink'])

fig.savefig(BASE / 'panel_A_workflow.png', dpi=300)
fig.savefig(BASE / 'panel_A_workflow.svg')
plt.close(fig)

source_files = ['data_umap/README_ZH.md', 'validation/README_ZH.md',
                'data_cords/README_ZH.md', 'summary_steps3_6/README_ZH.md']
inputs = {rel: hashlib.sha256((STORY / rel).read_bytes()).hexdigest()
          for rel in source_files}
design = {
    'version': 'v5',
    'time_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'purpose': 'Compact workflow block for a two-row lung story figure.',
    'language': 'English artwork; Chinese notes and bilingual caption.',
    'logic': [
        'CyTOF, mIF and IMC share the RAMP alignment space.',
        'One all-cell-types operator transfers relative HLA-ABC from CyTOF.',
        'mIF measured HLA-A provides auxiliary validation of the transferred state.',
        'IMC application studies author-annotated vCAF after full-cell projection.',
        'Downstream phenotype, immune context and DFS details are reserved for later panels.',
    ],
    'input_sha256': inputs,
    'icons': 'Original RAMP bitmap with both old side pipe stubs hidden by an SVG clip path; new vector rails connect directly to the sign border. Source asset remains unchanged.',
    'icon_provenance': json.loads((BASE / 'assets/SOURCE.json').read_text()),
    'rail_geometry': {
        'icon_extent_xywh': [logo_x, logo_y, logo_w, logo_h],
        'native_rail_fractions_from_top': [.286, .450],
        'extension_y': rail_y,
        'left_connects_to_platform_border': True,
        'left_arrow_removed': True,
        'transfer_box_below_RAMP': True,
        'CyTOF_source_state_label': 'HLA-ABC state',
        'visible_pipe_span_left': pipe_span,
        'visible_pipe_span_right': branch_x - sign_right,
        'output_node_label': 'vCAF analysis in IMC',
        'old_pipe_stubs_clipped': True,
        'clip_polygon_source_pixels': clip_pixels.tolist(),
        'sign_side_borders_source_pixels': [sign_left_px, sign_right_px],
        'new_rails_overlap_border_pixels': 4,
    },
    'outputs': ['panel_A_workflow.svg', 'panel_A_workflow.png'],
    'canvas_inches': [W, H],
    'previous_version': 'versions/v4',
}
(BASE / 'provenance/DESIGN.json').write_text(
    json.dumps(design, ensure_ascii=False, indent=2) + '\n')
print('Panel A v5 SVG and PNG generated.', flush=True)
