"""Match the continuous prediction's saturated color to annotated B cells."""
from runtime import BASE, save, sha, now
import json
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.transforms import Bbox
import assemble_cd19_layout_v11 as source

v, v8 = source.v, source.v8
OUT = BASE / 'figure_layout_v12'
for p in [OUT, OUT / 'panels', OUT / 'data']:
    p.mkdir(parents=True, exist_ok=True)
BOUNDS = (.5, 1.5)
positive_color = np.array(source.render['display']['cell_mask_positive_color'])
dapi = source.grey_dapi[source.crop]
blue = dapi[:, :, None] * np.array([.32, .45, .82])
highlight = np.clip(blue + positive_color, 0, 1)
highlight[source.boundary] *= .78
# The stored annotation map uses this exact highlight at each B-cell pixel.
lookup_b = np.zeros(len(source.lookup), bool)
lookup_b[source.full.CellID.to_numpy(int)] = source.full.annotated_b.to_numpy(bool)
bmap = lookup_b[source.mask]
assert np.array_equal(np.round(highlight[bmap] * 255).astype(np.uint8), source.reference[bmap])
assert np.array_equal(np.round(source.grey_rgb[~bmap] * 255).astype(np.uint8), source.reference[~bmap])
# A representative DAPI level shows the continuous tint in the color key.
key_dapi = .5
key_low = np.repeat(.82 * key_dapi, 3)
key_high = np.clip(key_dapi * np.array([.32, .45, .82]) + positive_color, 0, 1)
CMAP = LinearSegmentedColormap.from_list('CD19_matched_cyan', [key_low, key_high])


def image_for(bounds):
    norm = Normalize(*bounds, clip=True)
    level = norm(np.nan_to_num(source.lookup[source.mask], nan=bounds[0]))
    image = source.grey_rgb.copy()
    alpha = level[source.known, None]
    image[source.known] = ((1 - alpha) * source.grey_rgb[source.known] +
                          alpha * highlight[source.known])
    upper = source.known & (source.lookup[source.mask] >= bounds[1])
    lower = source.known & (source.lookup[source.mask] <= bounds[0])
    assert np.array_equal(image[upper], highlight[upper])
    assert np.array_equal(image[lower], source.grey_rgb[lower])
    return image, norm


def panel_g(fig, rect, letter='G', bounds=BOUNDS):
    x, y, w, h = rect
    v.title(fig, rect, letter, 'Spatial distribution of predicted CD19')
    image, norm = image_for(bounds)
    ax_side = min((w - .66)/2, h - .88)
    start = x + (w - 2 * ax_side - .62)/2
    for i, (img, title) in enumerate([(image, 'Predicted CD19'), (source.reference, 'Annotated B cells')]):
        ax = v.ax_at(fig, start + i * (ax_side + .62), y + .51, ax_side, ax_side)
        source.draw_image(ax, img, title)
    cax = v.ax_at(fig, start + ax_side + .08, y + 1.17, .075, 1.60)
    mid = 1.0
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax,
                     ticks=[bounds[0], mid, bounds[1]])
    cb.ax.set_yticklabels([f'≤{bounds[0]:.1f}', f'{mid:.1f}', f'≥{bounds[1]:.1f}'])
    cb.ax.tick_params(labelsize=8.0, length=2, pad=2)
    cb.outline.set_visible(False)
    v.text_at(fig, x + w/2, y + h - .23,
              f'Mean B-cell identity AUROC: {source.MEAN_AUC:.3f} (12 donors)',
              ha='center', fontsize=10.8, fontweight='bold')


def export(fig, name):
    for ext in ['png', 'svg', 'pdf']:
        fig.savefig(OUT / f'{name}.{ext}', dpi=300, facecolor='white')


def main():
    plt.rcParams['pdf.fonttype'] = 42
    for upper in [1.4, 1.5]:
        fig = plt.figure(figsize=(6.82, 4.05), facecolor='white')
        panel_g(fig, (.05, .05, 6.72, 3.95), bounds=(.5, upper))
        export(fig, f'panels/G_continuous_matched_color_upper_{upper:.1f}')
        plt.close(fig)
    fig = plt.figure(figsize=(13.64, 4.05), facecolor='white')
    panel_g(fig, (.05, .05, 6.72, 3.95), bounds=(.5, 1.4))
    panel_g(fig, (6.87, .05, 6.72, 3.95), bounds=(.5, 1.5))
    export(fig, 'panels/G_upper_1p4_vs_1p5')
    plt.close(fig)
    panels = [
        (v.panel_a, (.15, .13, 2.50, 2.55)),
        (v8.v6.panel_b, (2.76, .13, 5.70, 2.55)),
        (v8.v3.panel_c, (8.57, .13, 5.78, 2.55)),
        (v8.v3.panel_d, (.15, 2.79, 5.45, 3.50)),
        (source.old.previous.panel_e, (5.695, 2.79, 4.30, 3.50)),
        (v8.v7.panel_f, (10.09, 2.79, 4.26, 3.50)),
        (panel_g, (.15, 6.40, 6.72, 3.95)),
        (v8.v3.panel_h, (7.00, 6.40, 7.35, 3.95)),
    ]
    fig = plt.figure(figsize=(14.5, 10.5), facecolor='white')
    for draw, rect in panels:
        draw(fig, rect)
    export(fig, 'cd19_story_three_row_v12')
    fig.savefig(OUT / 'third_row_preview.png', dpi=300, facecolor='white',
                bbox_inches=Bbox.from_extents(.10, .08, 14.42, 10.5-6.35))
    plt.close(fig)
    image, _ = image_for(BOUNDS)
    Image.fromarray(np.round(image * 255).astype(np.uint8)).save(OUT / 'panels/G_predicted_CD19_no_labels.png')
    Image.fromarray(source.reference).save(OUT / 'panels/G_annotated_B_no_labels.png')
    source.zoom.to_csv(OUT / 'data/G_zoom_ROI_cells.csv.gz', index=False)
    source.donor_scores.to_csv(OUT / 'data/B_identity_metrics_by_donor.csv', index=False)
    pd.DataFrame(source.scopes).to_csv(OUT / 'data/B_identity_metrics_by_scope.csv', index=False)
    saturation = []
    for upper in [1.4, 1.5]:
        scores = source.zoom.predicted_cd19.to_numpy()
        saturation.append({'display_lower':.5,'display_upper':upper,'n_cells':len(scores),
            'n_at_lower_color':int((scores<=.5).sum()), 'n_at_upper_color':int((scores>=upper).sum()),
            'n_with_intermediate_color':int(((scores>.5)&(scores<upper)).sum())})
    pd.DataFrame(saturation).to_csv(OUT / 'data/display_saturation.csv', index=False)
    inputs = {**source.SOURCES, **source.old.SOURCES, **source.old.prior['source_sha256'],
              str(BASE/'scripts/assemble_cd19_layout_v11.py'):sha(BASE/'scripts/assemble_cd19_layout_v11.py')}
    assert all(sha(p) == h for p, h in inputs.items())
    save(OUT / 'LAYOUT_AUDIT.json', {
        'status':'PASS','time_utc':now(),'script_sha256':sha(__file__),'source_sha256':inputs,
        'canvas_inches':[14.5,10.5],'changed_panel':'G',
        'display_bounds':list(BOUNDS),'alternative_upper_bound':1.4,
        'display_rule':'Linear interpolation from grey DAPI to the exact B-annotation highlight; saturation at the upper bound',
        'upper_color_matches_reference_pixelwise':True,'lower_color_matches_reference_DAPI_pixelwise':True,
        'reference_raster_unchanged_except_previously_defined_zoom':True,
        'normalization':'Original predicted KMEANSLOW CD19 values; no rescaling or reranking',
        'display_bounds_are_not_classification_thresholds':True,
        'zoom':{'relative_bounds_xyxy':[source.cx,source.cy,source.cx+source.side,source.cy+source.side],
                'absolute_bounds_xyxy':[source.roi['x0']+source.cx,source.roi['y0']+source.cy,
                    source.roi['x0']+source.cx+source.side,source.roi['y0']+source.cy+source.side],
                'n_cells':len(source.zoom),'n_B':int(source.zoom.annotated_b.sum()),
                'pixel_size_um':source.roi['pixel_size_um'],'magnification':2,
                'selection':'Maximum cell-centroid density inside prior selected ROI; labels/predictions not used for zoom selection'},
        'mean_B_identity_AUROC':source.MEAN_AUC,'n_donors':12,'n_evaluation_cells':30000,
        'metric_computed_from_unclipped_continuous_predictions':True,
        'all_thresholded_precision_recall_results_retained_in_scope_table':True,
        'all_input_files_unchanged':True,'no_model_or_UMAP_refit':True,
    })
    print(pd.DataFrame(saturation).to_string(index=False))
    print('PASS: exact matched positive tint; frozen data and AUROC unchanged.')
    print(OUT / 'cd19_story_three_row_v12.png')


if __name__ == '__main__':
    main()
