"""Make the grey, below-range category explicit in Panel G's legend."""
from runtime import BASE, save, sha, now
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox
import assemble_cd19_layout_v16 as previous

OUT = BASE / 'figure_layout_v17'
for directory in [OUT, OUT / 'panels']:
    directory.mkdir(parents=True, exist_ok=True)
v, v8 = previous.v, previous.v8
GREY = np.repeat(.82*.5, 3)


def panel_g(fig, rect, letter='G'):
    previous.panel_g(fig, rect, letter=letter)
    x, y, w, h = rect
    side = min((w-.66)/2, h-.88)
    start = x+(w-2*side-.62)/2
    cx = start+side+.08
    width, height = fig.get_size_inches()
    cax = fig.axes[-1]
    cax.set_position([cx/width, (height-y-1.17-1.34)/height, .075/width, 1.34/height])
    cax.set_yticks([1.0, 1.25, 1.5])
    cax.set_yticklabels(['1.0', '1.25', '≥1.5'])
    # A separated swatch is categorical: its size does not denote a numeric interval.
    sx, sy, size = cx-.0375, y+2.70, .15
    swatch = Rectangle((sx/width, (height-sy-size)/height), size/width, size/height,
        transform=fig.transFigure, facecolor=GREY, edgecolor='#AAB2B8', linewidth=.45)
    fig.add_artist(swatch)
    fig.text((cx+.13)/width, 1-(sy+size/2)/height, '≤1.0',
             va='center', ha='left', fontsize=8, color='#596671')


def export(fig, name):
    for ext in ['png', 'svg', 'pdf']:
        fig.savefig(OUT / f'{name}.{ext}', dpi=300, facecolor='white')


def main():
    plt.rcParams['pdf.fonttype'] = 42
    image, _ = previous.image_for(previous.BOUNDS)
    old_image = np.asarray(Image.open(BASE / 'figure_layout_v16/panels/G_predicted_CD19_no_labels.png'))
    assert np.array_equal(np.round(image*255).astype(np.uint8), old_image)
    fig = plt.figure(figsize=(6.82, 4.05), facecolor='white')
    panel_g(fig, (.05, .05, 6.72, 3.95))
    export(fig, 'panels/G_explicit_grey_legend')
    plt.close(fig)
    panels = [
        (v.panel_a, (.15, .13, 2.50, 2.55)),
        (v8.v6.panel_b, (2.76, .13, 5.70, 2.55)),
        (v8.v3.panel_c, (8.57, .13, 5.78, 2.55)),
        (v8.v3.panel_d, (.15, 2.79, 5.45, 3.50)),
        (previous.previous.previous.panel_e, (5.695, 2.79, 4.30, 3.50)),
        (v8.v7.panel_f, (10.09, 2.79, 4.26, 3.50)),
        (panel_g, (.15, 6.40, 6.72, 3.95)),
        (v8.v3.panel_h, (7.00, 6.40, 7.35, 3.95)),
    ]
    fig = plt.figure(figsize=(14.5, 10.5), facecolor='white')
    for draw, rect in panels:
        draw(fig, rect)
    export(fig, 'cd19_story_three_row_v17')
    fig.savefig(OUT / 'third_row_preview.png', dpi=300, facecolor='white',
                bbox_inches=Bbox.from_extents(.10, .08, 14.42, 10.5-6.35))
    plt.close(fig)
    prior = json.loads((BASE / 'figure_layout_v16/LAYOUT_AUDIT.json').read_text())
    assert all(sha(path) == digest for path, digest in prior['source_sha256'].items())
    save(OUT / 'LAYOUT_AUDIT.json', {'status': 'PASS', 'time_utc': now(),
        'script_sha256': sha(__file__), 'source_sha256': prior['source_sha256'],
        'changed_element': 'Panel G legend only', 'display_bounds': list(previous.BOUNDS),
        'grey_swatch_label': '≤1.0', 'grey_swatch_RGB': GREY.tolist(),
        'swatch_is_categorical_not_interval_length': True,
        'negative_standardized_scores_included_in_grey_category': True,
        'predicted_image_matches_v16_pixelwise': True, 'same_reference_image': True,
        'same_E_all_donor_CD8': True, 'mean_B_identity_AUROC': previous.source.MEAN_AUC,
        'all_inputs_unchanged': True, 'no_model_or_UMAP_refit': True})
    print('PASS: explicit grey ≤1.0 swatch; prediction pixels, labels and metrics unchanged.')
    print(OUT / 'cd19_story_three_row_v17.png')


if __name__ == '__main__':
    main()
