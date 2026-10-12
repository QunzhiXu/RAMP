"""Continuous CD19 on a zoomed real ROI; whole-cohort B-identity ranking."""
from runtime import BASE, STORY, save, sha, now
import json
import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter, minimum_filter
from sklearn.metrics import roc_auc_score, average_precision_score
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.transforms import Bbox
import assemble_cd19_layout_v10 as old

v, v8 = old.v, old.v8
OUT = BASE / 'figure_layout_v11'
for p in [OUT, OUT / 'panels', OUT / 'data']:
    p.mkdir(parents=True, exist_ok=True)
REAL = BASE / 'real_spatial0913'
SOURCES = {}
CMAP = LinearSegmentedColormap.from_list('CD19_continuous',
        ['#D5D9DE', '#70AEB0', '#22AC87', '#F1D84B'])
BOUNDS = (.5, 2.5)


def track(p):
    SOURCES[str(p)] = sha(p)
    return p


def metrics(scope, truth, score):
    positive = score > 1.4
    tp = int((positive & truth).sum())
    return {'scope': scope, 'n_cells': len(truth), 'n_B': int(truth.sum()),
            'B_fraction': float(truth.mean()), 'auroc': float(roc_auc_score(truth, score)),
            'AP': float(average_precision_score(truth, score)),
            'precision_at_1p4': float(tp / positive.sum()),
            'recall_at_1p4': float(tp / truth.sum())}


md = pd.read_csv(track(STORY / 'sampled_data/bm_codex/locked_metadata.csv.gz'))
pm = pd.read_csv(track(STORY / 'prediction/output/codex_prediction_metadata.csv.gz'))
assert md.cell_id.astype(str).equals(pm.cell_id.astype(str))
with np.load(track(STORY / 'prediction/output/codex_cd19_prediction.npz')) as z:
    prediction = z['predicted_kmeanslow'].astype(float)
truth = md.cluster_anno_l1.eq('B-Cells').to_numpy()
donor_records = [metrics(donor, truth[ii], prediction[ii])
                 for donor, ii in md.groupby('donor_id', sort=True).indices.items()]
donor_scores = pd.DataFrame(donor_records)
assert len(donor_scores) == 12 and donor_scores.n_cells.eq(2500).all()
MEAN_AUC = float(donor_scores.auroc.mean())
full = pd.read_csv(track(REAL / 'data/H14_full_predictions.csv.gz'))
roi_cells = pd.read_csv(track(REAL / 'data/selected_roi_cells.csv.gz'))
roi = json.loads(track(REAL / 'results/selected_roi.json').read_text())
render = json.loads(track(REAL / 'results/render_audit.json').read_text())
grey = json.loads(track(REAL / 'results/gray_background_audit.json').read_text())
parent_mask = np.load(track(REAL / 'data/selected_roi_mask.npy'))
with np.load(track(REAL / 'data/selected_roi_measured_channels.npz')) as z:
    raw_dapi = z['DAPI']
xy = roi_cells[['x.coord', 'y.coord']].to_numpy() - np.array([roi['x0'], roi['y0']])
# Density-only zoom within the already selected ROI. Labels and scores are not used.
side, stride = 512, 16
windows = []
for yy in range(0, 1024 - side + 1, stride):
    for xx in range(0, 1024 - side + 1, stride):
        inside = ((xy[:, 0] >= xx) & (xy[:, 0] < xx + side) &
                  (xy[:, 1] >= yy) & (xy[:, 1] < yy + side))
        windows.append({'x0': xx, 'y0': yy, 'n_cells': int(inside.sum())})
windows = pd.DataFrame(windows).sort_values(['n_cells', 'y0', 'x0'], ascending=[False, True, True])
best = windows.iloc[0]
cx, cy = int(best.x0), int(best.y0)
selection = ((xy[:, 0] >= cx) & (xy[:, 0] < cx + side) &
             (xy[:, 1] >= cy) & (xy[:, 1] < cy + side))
zoom = roi_cells.loc[selection].copy()
assert (cx, cy, len(zoom)) == (384, 320, 251)
crop = np.s_[cy:cy + side, cx:cx + side]
mask = parent_mask[crop]
ids = full.CellID.to_numpy(int)
length = int(max(ids.max(), parent_mask.max())) + 1
lookup = np.full(length, np.nan)
lookup[ids] = full.predicted_cd19.to_numpy()
known = np.isfinite(lookup[mask]) & (mask > 0)
grey_dapi = np.clip(raw_dapi.astype(float) / render['display']['dapi_hi'], 0, 1) ** render['display']['dapi_gamma']
grey_rgb = np.repeat((grey_dapi * .82)[:, :, None], 3, axis=2)[crop]
boundary = (maximum_filter(parent_mask, 3) != minimum_filter(parent_mask, 3))[crop] & (mask > 0)
reference = np.asarray(Image.open(track(REAL / 'figures/H14_ROI_annotated_B_masks_gray_background.png')).convert('RGB'))[crop]
binary = np.asarray(Image.open(track(REAL / 'figures/H14_ROI_predicted_CD19_masks_gray_background.png')).convert('RGB'))[crop]
scopes = [metrics('12 donors pooled', truth, prediction),
          metrics('H14 full donor', full.annotated_b.to_numpy(bool), full.predicted_cd19.to_numpy()),
          metrics('H14 original ROI', roi_cells.annotated_b.to_numpy(bool), roi_cells.predicted_cd19.to_numpy()),
          metrics('H14 zoom ROI', zoom.annotated_b.to_numpy(bool), zoom.predicted_cd19.to_numpy())]


def continuous(bounds):
    norm = Normalize(*bounds, clip=True)
    image = grey_rgb.copy()
    color = CMAP(norm(np.nan_to_num(lookup[mask], nan=bounds[0])))[:, :, :3]
    image[known] = .55 * grey_rgb[known] + .65 * color[known]
    image[known & boundary] *= .88
    return np.clip(image, 0, 1), norm


def draw_image(ax, image, title):
    ax.imshow(image, interpolation='nearest')
    ax.axis('off')
    ax.set_title(title, fontsize=10.5, fontweight='bold', pad=6)
    length = 50 / roi['pixel_size_um']
    xx, yy = side - length - 24, side - 28
    ax.plot([xx, xx + length], [yy, yy], color='white', lw=2.4, solid_capstyle='butt')
    ax.text(xx + length/2, yy - 10, '50 µm', color='white', fontsize=9,
            fontweight='bold', ha='center', va='bottom')


def panel_g(fig, rect, letter='G', bounds=BOUNDS):
    x, y, w, h = rect
    v.title(fig, rect, letter, 'Spatial distribution of predicted CD19')
    image, norm = continuous(bounds)
    ax_side = min((w - .66)/2, h - .88)
    start = x + (w - 2 * ax_side - .62)/2
    for i, (img, title) in enumerate([(image, 'Predicted CD19'), (reference, 'Annotated B cells')]):
        ax = v.ax_at(fig, start + i * (ax_side + .62), y + .51, ax_side, ax_side)
        draw_image(ax, img, title)
    cax = v.ax_at(fig, start + ax_side + .08, y + 1.17, .075, 1.60)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax,
                      ticks=[bounds[0], (bounds[0] + bounds[1])/2, bounds[1]])
    cb.ax.set_yticklabels([f'≤{bounds[0]:.1f}', f'{(bounds[0]+bounds[1])/2:.1f}', f'≥{bounds[1]:.1f}'])
    cb.ax.tick_params(labelsize=8.0, length=2, pad=2)
    cb.outline.set_visible(False)
    v.text_at(fig, x + w/2, y + h - .23,
              f'B-cell identity AUROC: {MEAN_AUC:.3f} (12 donors)',
              ha='center', fontsize=11.2, fontweight='bold')


def export(fig, name):
    for ext in ['png', 'svg', 'pdf']:
        fig.savefig(OUT / f'{name}.{ext}', dpi=300, facecolor='white')


def main():
    plt.rcParams['pdf.fonttype'] = 42
    for bounds in [(0., 3.), (0., 2.5), (.5, 2.5)]:
        fig = plt.figure(figsize=(6.82, 4.05), facecolor='white')
        panel_g(fig, (.05, .05, 6.72, 3.95), bounds=bounds)
        export(fig, f'panels/G_continuous_{bounds[0]:.1f}_{bounds[1]:.1f}')
        plt.close(fig)
    fig = plt.figure(figsize=(10.5, 3.7), facecolor='white')
    image, norm = continuous(BOUNDS)
    for i, (img, title) in enumerate([(binary, 'Previous: predicted CD19 > 1.4'),
                                      (image, 'Continuous predicted CD19'),
                                      (reference, 'Annotated B cells')]):
        ax = v.ax_at(fig, .06 + 3.50*i, .38, 3.30, 3.30)
        draw_image(ax, img, title)
    export(fig, 'panels/G_binary_vs_continuous')
    plt.close(fig)
    panels = [
        (v.panel_a, (.15, .13, 2.50, 2.55)),
        (v8.v6.panel_b, (2.76, .13, 5.70, 2.55)),
        (v8.v3.panel_c, (8.57, .13, 5.78, 2.55)),
        (v8.v3.panel_d, (.15, 2.79, 5.45, 3.50)),
        (old.previous.panel_e, (5.695, 2.79, 4.30, 3.50)),
        (v8.v7.panel_f, (10.09, 2.79, 4.26, 3.50)),
        (panel_g, (.15, 6.40, 6.72, 3.95)),
        (v8.v3.panel_h, (7.00, 6.40, 7.35, 3.95)),
    ]
    fig = plt.figure(figsize=(14.5, 10.5), facecolor='white')
    for draw, rect in panels:
        draw(fig, rect)
    export(fig, 'cd19_story_three_row_v11')
    fig.savefig(OUT / 'third_row_preview.png', dpi=300, facecolor='white',
                bbox_inches=Bbox.from_extents(.10, .08, 14.42, 10.5-6.35))
    plt.close(fig)
    image, _ = continuous(BOUNDS)
    Image.fromarray(np.round(image*255).astype(np.uint8)).save(OUT / 'panels/G_continuous_CD19_no_labels.png')
    Image.fromarray(reference).save(OUT / 'panels/G_annotated_B_no_labels.png')
    windows.to_csv(OUT / 'data/geometry_only_zoom_windows.csv', index=False)
    zoom.to_csv(OUT / 'data/zoom_ROI_cells.csv.gz', index=False)
    donor_scores.to_csv(OUT / 'data/B_identity_metrics_by_donor.csv', index=False)
    pd.DataFrame(scopes).to_csv(OUT / 'data/B_identity_metrics_by_scope.csv', index=False)
    cutoff_stats = []
    for lo, hi in [(0., 3.), (0., 2.5), (.5, 2.5)]:
        cutoff_stats.append({'lower':lo,'upper':hi,
            'fraction_at_lower_color':float((zoom.predicted_cd19<=lo).mean()),
            'fraction_at_upper_color':float((zoom.predicted_cd19>=hi).mean())})
    pd.DataFrame(cutoff_stats).to_csv(OUT / 'data/display_saturation.csv', index=False)
    assert np.array_equal(reference, old.images[1][crop])
    assert np.all(np.isfinite(lookup[zoom.CellID.to_numpy(int)]))
    all_sources = {**SOURCES, **old.SOURCES, **old.prior['source_sha256']}
    assert all(sha(p) == h for p, h in all_sources.items())
    save(OUT / 'LAYOUT_AUDIT.json', {
        'status':'PASS','time_utc':now(),'script_sha256':sha(__file__),'source_sha256':all_sources,
        'canvas_inches':[14.5,10.5],'changed_panels_relative_to_v10':['G'],
        'zoom':{'side_pixels':side,'relative_x0':cx,'relative_y0':cy,
                'absolute_x0':roi['x0']+cx,'absolute_y0':roi['y0']+cy,
                'n_centroid_cells':len(zoom),'n_B':int(zoom.annotated_b.sum()),
                'selection':'Maximum centroid count among 512-pixel windows, stride16, inside prior ROI; no labels/scores used',
                'parent_ROI_selection':'Previously selected by F1 within H14',
                'pixel_size_um':roi['pixel_size_um'],'magnification_relative_to_1024px_ROI':2},
        'display':{'bounds':list(BOUNDS),'interpretation':'Color saturation bounds, not biological or B-cell classification cutoffs',
                   'scale':'Original predicted KMEANSLOW CD19; no ranks or re-normalization',
                   'DAPI':'Acquired H14 DAPI with grey display','cell_shapes':'Author Mesmer labels',
                   'prediction_smoothing':False,'all_overlapping_atlas_cells_colored':True,
                   'unknown_mask_objects':'Grey DAPI only','low_end_grey':'Lower-end displayed score; not proof of zero expression'},
        'metrics':{'label':'B-cell identity AUROC','displayed_value':MEAN_AUC,
                   'scope':'Arithmetic mean of 12 donor AUROCs, each on 2500 frozen cells',
                   'continuous_scores_used_before_display_clipping':True,
                   'AP_donor_mean':float(donor_scores.AP.mean()),'pooled_30000':scopes[0],
                   'all_scopes_in_csv':True,'prior_precision_recall_retained_in_csv':True},
        'reference_pixels_unchanged_except_crop':True,'all_input_files_unchanged':True,
        'no_model_or_UMAP_refit':True,
    })
    print(pd.DataFrame(scopes).to_string(index=False))
    print('Mean donor AUROC:',MEAN_AUC)
    print(OUT / 'cd19_story_three_row_v11.png')


if __name__ == '__main__':
    main()
