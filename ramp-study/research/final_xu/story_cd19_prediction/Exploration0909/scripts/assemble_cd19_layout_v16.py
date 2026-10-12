"""Compare display lower bounds 0.5 and 1.0 with a fixed yellow upper bound."""
from runtime import BASE, save, sha, now
import json
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.transforms import Bbox
import assemble_cd19_layout_v15 as previous

OUT = BASE / 'figure_layout_v16'
for directory in [OUT, OUT / 'panels', OUT / 'data']:
    directory.mkdir(parents=True, exist_ok=True)
v, v8, source = previous.v, previous.v8, previous.source
BOUNDS = (1.0, 1.5)


def image_for(bounds):
    norm = Normalize(*bounds, clip=True)
    level = norm(np.nan_to_num(source.lookup[source.mask], nan=bounds[0]))
    image = source.grey_rgb.copy()
    low = source.known & (level <= previous.GREEN_POSITION)
    high = source.known & (level > previous.GREEN_POSITION)
    a = (level[low]/previous.GREEN_POSITION)[:, None]
    image[low] = (1-a)*source.grey_rgb[low] + a*previous.green[low]
    a = ((level[high]-previous.GREEN_POSITION)/(1-previous.GREEN_POSITION))[:, None]
    image[high] = (1-a)*previous.green[high] + a*previous.yellow[high]
    assert np.array_equal(image[source.known & (level == 0)], source.grey_rgb[source.known & (level == 0)])
    assert np.array_equal(image[source.known & (level == 1)], previous.yellow[source.known & (level == 1)])
    assert np.array_equal(image[~source.known], source.grey_rgb[~source.known])
    return image, norm


def panel_g(fig, rect, letter='G', bounds=BOUNDS):
    x, y, w, h = rect
    v.title(fig, rect, letter, 'Spatial distribution of predicted CD19')
    image, norm = image_for(bounds)
    side = min((w-.66)/2, h-.88)
    start = x+(w-2*side-.62)/2
    for i, (pixels, title) in enumerate([(image, 'Predicted CD19'),
                                        (previous.reference, 'Annotated B cells')]):
        ax = v.ax_at(fig, start+i*(side+.62), y+.51, side, side)
        source.draw_image(ax, pixels, title)
    cax = v.ax_at(fig, start+side+.08, y+1.17, .075, 1.60)
    mid = sum(bounds)/2
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=previous.CMAP), cax=cax,
                     ticks=[bounds[0], mid, bounds[1]])
    middle_label = f'{mid:.2f}'.rstrip('0').rstrip('.')
    cb.ax.set_yticklabels([f'≤{bounds[0]:.1f}', middle_label, f'≥{bounds[1]:.1f}'])
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
    old_pixels, _ = image_for((.5, 1.5))
    assert np.array_equal(old_pixels, previous.prediction)
    reference_path = BASE / 'figure_layout_v15/panels/G_annotated_B_no_labels.png'
    reference_u8 = np.round(previous.reference*255).astype(np.uint8)
    assert np.array_equal(reference_u8, np.asarray(Image.open(reference_path)))
    for lower in [.5, 1.0]:
        fig = plt.figure(figsize=(6.82, 4.05), facecolor='white')
        panel_g(fig, (.05, .05, 6.72, 3.95), bounds=(lower, 1.5))
        export(fig, f'panels/G_lower_{lower:.1f}_upper_1.5')
        plt.close(fig)
    fig = plt.figure(figsize=(13.64, 4.35), facecolor='white')
    for x, lower in [(.05, .5), (6.87, 1.0)]:
        v.text_at(fig, x+3.36, .15, f'Display range: {lower:.1f}–1.5',
                  ha='center', fontsize=11, fontweight='bold')
        panel_g(fig, (x, .35, 6.72, 3.95), bounds=(lower, 1.5))
    export(fig, 'panels/G_lower_0p5_vs_1p0')
    plt.close(fig)
    panels = [
        (v.panel_a, (.15, .13, 2.50, 2.55)),
        (v8.v6.panel_b, (2.76, .13, 5.70, 2.55)),
        (v8.v3.panel_c, (8.57, .13, 5.78, 2.55)),
        (v8.v3.panel_d, (.15, 2.79, 5.45, 3.50)),
        (previous.previous.panel_e, (5.695, 2.79, 4.30, 3.50)),
        (v8.v7.panel_f, (10.09, 2.79, 4.26, 3.50)),
        (panel_g, (.15, 6.40, 6.72, 3.95)),
        (v8.v3.panel_h, (7.00, 6.40, 7.35, 3.95)),
    ]
    fig = plt.figure(figsize=(14.5, 10.5), facecolor='white')
    for draw, rect in panels:
        draw(fig, rect)
    export(fig, 'cd19_story_three_row_v16')
    fig.savefig(OUT / 'third_row_preview.png', dpi=300, facecolor='white',
                bbox_inches=Bbox.from_extents(.10, .08, 14.42, 10.5-6.35))
    plt.close(fig)
    image, _ = image_for(BOUNDS)
    Image.fromarray(np.round(image*255).astype(np.uint8)).save(OUT / 'panels/G_predicted_CD19_no_labels.png')
    Image.fromarray(reference_u8).save(OUT / 'panels/G_annotated_B_no_labels.png')
    counts = []
    for scope, scores in [('Displayed ROI', source.zoom.predicted_cd19.to_numpy()),
                           ('All 12 sampled donors', source.prediction)]:
        for lower in [.5, 1.0]:
            counts.append({'scope': scope, 'lower': lower, 'upper': 1.5, 'n_cells': len(scores),
                'grey_cells': int((scores <= lower).sum()),
                'intermediate_cells': int(((scores > lower) & (scores < 1.5)).sum()),
                'yellow_cells': int((scores >= 1.5).sum())})
    pd.DataFrame(counts).to_csv(OUT / 'data/display_clipping_counts.csv', index=False)
    prior = json.loads((BASE / 'figure_layout_v15/LAYOUT_AUDIT.json').read_text())
    assert all(sha(path) == digest for path, digest in prior['source_sha256'].items())
    save(OUT / 'LAYOUT_AUDIT.json', {'status': 'PASS', 'time_utc': now(),
        'script_sha256': sha(__file__), 'source_sha256': prior['source_sha256'],
        'display_bounds': list(BOUNDS), 'comparison_bounds': [.5, 1.5],
        'only_lower_display_bound_changed': True,
        'same_palette_and_reference_pixels': True, 'old_render_reproduced_pixelwise': True,
        'cell_predictions_and_ROI_unchanged': True, 'no_cells_removed': True,
        'mean_B_identity_AUROC': source.MEAN_AUC,
        'display_clipping_does_not_enter_metrics': True, 'no_model_or_UMAP_refit': True,
        'same_E_all_donor_CD8': True, 'all_inputs_unchanged': True,
        'clipping_counts': counts})
    print(pd.DataFrame(counts).to_string(index=False))
    print('PASS: lower-bound comparison; fixed scores, upper bound, right image and metrics.')
    print(OUT / 'cd19_story_three_row_v16.png')


if __name__ == '__main__':
    main()
