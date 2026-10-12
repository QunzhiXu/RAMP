"""Recolor DAPI-only context to neutral grey, preserving positive overlays."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import maximum_filter, minimum_filter

from render_real_roi import BASE, figure_images, normalized, save_rgb


def sha(path):
    with open(path, 'rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    sources = [BASE / name for name in [
        'results/selected_roi.json', 'results/render_audit.json',
        'data/H14_full_predictions.csv.gz', 'data/selected_roi_cells.csv.gz',
        'data/selected_roi_mask.npy', 'data/selected_roi_measured_channels.npz',
    ]]
    hashes = {str(path): sha(path) for path in sources}
    roi = json.loads(sources[0].read_text())
    audit = json.loads(sources[1].read_text())
    full = pd.read_csv(sources[2])
    selected = pd.read_csv(sources[3])
    mask = np.load(sources[4])
    with np.load(sources[5]) as data:
        raw = {name: data[name] for name in data.files}
    ids = full.CellID.to_numpy(int)
    length = int(max(mask.max(), ids.max())) + 1
    high = np.zeros(length, dtype=bool)
    annotated = np.zeros(length, dtype=bool)
    high[ids] = full.predicted_high.to_numpy(bool)
    annotated[ids] = full.annotated_b.to_numpy(bool)
    pmap, bmap = high[mask], annotated[mask]
    boundary = (maximum_filter(mask, 3) != minimum_filter(mask, 3)) & (mask > 0)

    display = audit['display']
    dapi, _ = normalized(raw['DAPI'], hi=display['dapi_hi'], gamma=display['dapi_gamma'])
    blue = dapi[:, :, None] * np.array([.32, .45, .82])
    grey = np.repeat((dapi * .82)[:, :, None], 3, axis=2)
    positive = np.array(display['cell_mask_positive_color'])
    pairs = []
    for name, positive_map in [('predicted_CD19_masks', pmap), ('annotated_B_masks', bmap)]:
        original = np.clip(blue + positive_map[:, :, None] * positive, 0, 1)
        original[positive_map & boundary] *= .78
        previous_png = np.asarray(Image.open(BASE / 'figures' / f'H14_ROI_{name}.png').convert('RGB'))
        assert np.array_equal(np.round(original * 255).astype(np.uint8), previous_png)
        recolored = original.copy()
        recolored[~positive_map] = grey[~positive_map]
        assert np.array_equal(recolored[positive_map], original[positive_map])
        assert np.array_equal(recolored[~positive_map, 0], recolored[~positive_map, 1])
        assert np.array_equal(recolored[~positive_map, 1], recolored[~positive_map, 2])
        save_rgb(recolored, f'H14_ROI_{name}_gray_background.png')
        pairs.append(recolored)

    save_rgb(grey, 'H14_ROI_DAPI_gray.png')
    figure_images(pairs, ['Predicted CD19 > 1.4', 'Annotated B cells'],
                  'H14_best_ROI_real_cells_pair_gray_background', roi,
                  [f'Precision {roi["precision"]:.1%}', f'Recall {roi["recall"]:.1%}'])
    # The reference measurement keeps its actual CD79A pixels; only its DAPI display is neutral.
    cd79, _ = normalized(raw['CD79A'], hi=display['cd79a_hi'])
    cd79_with_grey = np.clip(grey + cd79[:, :, None] * np.array([.10, .90, .45]), 0, 1)
    save_rgb(cd79_with_grey, 'H14_ROI_measured_CD79A_gray_DAPI.png')
    figure_images([cd79_with_grey] + pairs,
                  ['Measured CD79A + DAPI', 'Predicted CD19 > 1.4', 'Annotated B cells'],
                  'H14_best_ROI_true_image_and_predictions_gray_background', roi,
                  ['Original CODEX pixels', f'Precision {roi["precision"]:.1%}', f'Recall {roi["recall"]:.1%}'])

    # Verify the displayed membership and original evaluation are identical.
    a = selected.annotated_b.to_numpy(bool)
    p = selected.predicted_high.to_numpy(bool)
    for key, value in [('n_cells', len(selected)), ('tp', int((p & a).sum())),
                       ('fp', int((p & ~a).sum())), ('fn', int((~p & a).sum()))]:
        assert value == roi[key], (key, value, roi[key])
    assert hashes == {str(path): sha(path) for path in sources}
    record = {
        'status': 'PASS', 'request': 'Neutral white-grey DAPI context for non-highlighted cells',
        'dapi_display': {'upper': display['dapi_hi'], 'gamma': display['dapi_gamma'], 'rgb': [.82, .82, .82]},
        'positive_overlay_pixels_identical_to_previous_render': True,
        'all_non_highlighted_pixels_are_achromatic': True,
        'prediction_threshold': roi['threshold'], 'n_cells': len(selected),
        'tp': roi['tp'], 'fp': roi['fp'], 'fn': roi['fn'],
        'precision': roi['precision'], 'recall': roi['recall'], 'f1': roi['f1'],
        'interpretation': 'Grey intensity shows measured DAPI only, not low CD19 expression. '
                          'Left non-highlighted atlas cells are below the fixed prediction threshold; '
                          'right non-highlighted atlas cells are not annotated B cells. '
                          'Unmatched mask objects retain neutral measured DAPI without an inferred label.',
        'source_sha256': hashes, 'sources_unchanged': True,
        'source_blue_figures_preserved': True,
    }
    (BASE / 'results/gray_background_audit.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({k: v for k, v in record.items() if k != 'source_sha256'}, indent=2))


if __name__ == '__main__':
    main()
