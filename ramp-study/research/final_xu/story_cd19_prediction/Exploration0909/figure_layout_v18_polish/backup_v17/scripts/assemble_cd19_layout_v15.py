"""Panel G: grey context, green intermediate predictions, matched yellow highs."""
from runtime import BASE, save, sha, now
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.transforms import Bbox
import assemble_cd19_layout_v14 as previous

v, v8 = previous.v, previous.v8
old = previous.previous.previous
source = old.source
OUT = BASE / 'figure_layout_v15'
for directory in [OUT, OUT / 'panels', OUT / 'data']:
    directory.mkdir(parents=True, exist_ok=True)
BOUNDS = (.5, 1.5)
GREEN_POSITION = .25
GREEN_RGB = np.array([.06, .62, .29])
YELLOW_RGB = np.array([1.0, .91, .12])
dapi = source.grey_dapi[source.crop]
texture = (.85 + .15*dapi)[..., None]
green = texture * GREEN_RGB
yellow = texture * YELLOW_RGB
green[source.boundary] *= .78
yellow[source.boundary] *= .78
reference = source.grey_rgb.copy()
reference[old.bmap] = yellow[old.bmap]
norm = Normalize(*BOUNDS, clip=True)
level = norm(np.nan_to_num(source.lookup[source.mask], nan=BOUNDS[0]))
prediction = source.grey_rgb.copy()
low = source.known & (level <= GREEN_POSITION)
high = source.known & (level > GREEN_POSITION)
a = (level[low]/GREEN_POSITION)[:, None]
prediction[low] = (1-a)*source.grey_rgb[low] + a*green[low]
a = ((level[high]-GREEN_POSITION)/(1-GREEN_POSITION))[:, None]
prediction[high] = (1-a)*green[high] + a*yellow[high]
at_lower = source.known & (level == 0)
at_upper = source.known & (level == 1)
assert np.array_equal(prediction[at_lower], source.grey_rgb[at_lower])
assert np.array_equal(prediction[at_upper], yellow[at_upper])
assert np.array_equal(reference[~old.bmap], source.grey_rgb[~old.bmap])
assert np.array_equal(reference[old.bmap], yellow[old.bmap])
assert np.array_equal(prediction[~source.known], source.grey_rgb[~source.known])
key_texture = .85 + .15*.5
CMAP = LinearSegmentedColormap.from_list('CD19_grey_green_yellow',
    [(0, np.repeat(.82*.5, 3)), (GREEN_POSITION, key_texture*GREEN_RGB),
     (1, key_texture*YELLOW_RGB)])


def panel_g(fig, rect, letter='G'):
    x, y, w, h = rect
    v.title(fig, rect, letter, 'Spatial distribution of predicted CD19')
    side = min((w-.66)/2, h-.88)
    start = x + (w-2*side-.62)/2
    for i, (image, title) in enumerate([(prediction, 'Predicted CD19'),
                                       (reference, 'Annotated B cells')]):
        ax = v.ax_at(fig, start+i*(side+.62), y+.51, side, side)
        source.draw_image(ax, image, title)
    cax = v.ax_at(fig, start+side+.08, y+1.17, .075, 1.60)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax,
                     ticks=[.5, 1.0, 1.5])
    cb.ax.set_yticklabels(['≤0.5', '1.0', '≥1.5'])
    cb.ax.tick_params(labelsize=8, length=2, pad=2)
    cb.outline.set_visible(False)
    v.text_at(fig, x+w/2, y+h-.23,
              f'Mean B-cell identity AUROC: {source.MEAN_AUC:.3f} (12 donors)',
              ha='center', fontsize=10.8, fontweight='bold')


def export(fig, name):
    for ext in ['png', 'svg', 'pdf']:
        fig.savefig(OUT / f'{name}.{ext}', dpi=300, facecolor='white')


def main():
    plt.rcParams['pdf.fonttype'] = 42
    fig = plt.figure(figsize=(6.82, 4.05), facecolor='white')
    panel_g(fig, (.05, .05, 6.72, 3.95))
    export(fig, 'panels/G_green_yellow')
    plt.close(fig)
    fig = plt.figure(figsize=(13.64, 4.35), facecolor='white')
    for x, label, draw in [(.05, 'Previous palette', old.panel_g),
                           (6.87, 'Green–yellow palette', panel_g)]:
        v.text_at(fig, x+3.36, .15, label, ha='center', fontsize=11, fontweight='bold')
        draw(fig, (x, .35, 6.72, 3.95))
    export(fig, 'panels/G_previous_vs_green_yellow')
    plt.close(fig)
    panels = [
        (v.panel_a, (.15, .13, 2.50, 2.55)),
        (v8.v6.panel_b, (2.76, .13, 5.70, 2.55)),
        (v8.v3.panel_c, (8.57, .13, 5.78, 2.55)),
        (v8.v3.panel_d, (.15, 2.79, 5.45, 3.50)),
        (previous.panel_e, (5.695, 2.79, 4.30, 3.50)),
        (v8.v7.panel_f, (10.09, 2.79, 4.26, 3.50)),
        (panel_g, (.15, 6.40, 6.72, 3.95)),
        (v8.v3.panel_h, (7.00, 6.40, 7.35, 3.95)),
    ]
    fig = plt.figure(figsize=(14.5, 10.5), facecolor='white')
    for draw, rect in panels:
        draw(fig, rect)
    export(fig, 'cd19_story_three_row_v15')
    fig.savefig(OUT / 'third_row_preview.png', dpi=300, facecolor='white',
                bbox_inches=Bbox.from_extents(.10, .08, 14.42, 10.5-6.35))
    plt.close(fig)
    for name, image in [('G_predicted_CD19_no_labels', prediction),
                         ('G_annotated_B_no_labels', reference)]:
        Image.fromarray(np.round(np.clip(image, 0, 1)*255).astype(np.uint8)).save(OUT / 'panels' / f'{name}.png')
    prior = json.loads((BASE / 'figure_layout_v14/LAYOUT_AUDIT.json').read_text())
    assert all(sha(path) == digest for path, digest in prior['source_sha256'].items())
    assert abs(source.MEAN_AUC-.9040993666381246) < 1e-14
    save(OUT / 'LAYOUT_AUDIT.json', {
        'status': 'PASS', 'time_utc': now(), 'script_sha256': sha(__file__),
        'source_sha256': prior['source_sha256'], 'changed_panel': 'G',
        'display_bounds': list(BOUNDS), 'display_bounds_unchanged': True,
        'color_control_points': {'lower': 'original neutral grey DAPI',
            'intermediate_normalized_position': GREEN_POSITION,
            'intermediate_base_RGB': GREEN_RGB.tolist(), 'upper_base_RGB': YELLOW_RGB.tolist()},
        'color_mapping': 'Continuous piecewise-linear grey to green to yellow; original DAPI modulates texture',
        'reference_B_cells_use_same_yellow': True,
        'low_prediction_and_non_B_reference_DAPI_unchanged': True,
        'same_real_pixels_masks_ROI_and_cell_predictions': True,
        'same_E_all_donor_CD8': True, 'all_input_files_unchanged': True,
        'mean_B_identity_AUROC': source.MEAN_AUC, 'no_model_or_UMAP_refit': True})
    print('PASS: green–yellow palette; matched yellow B-cell reference; fixed bounds and metrics.')
    print(OUT / 'cd19_story_three_row_v15.png')


if __name__ == '__main__':
    main()
