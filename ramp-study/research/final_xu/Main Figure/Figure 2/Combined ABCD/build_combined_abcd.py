"""Assemble full-resolution Figure 2 without stretching or cropping any panel."""
from pathlib import Path
import hashlib
import json
from PIL import Image

HERE = Path(__file__).resolve().parent
FIGURE = HERE.parent
SOURCES = {
    'A': FIGURE/'Panel A/bone_marrow_ridge_umap_figure1_style.png',
    'B': FIGURE/'Panel B/umap_style_review/lung_cancer_ridge_umap_refined.png',
    'C': FIGURE/'Panel C/umap_style_review/melanoma_ridge_umap_refined.png',
    'D': FIGURE/'Panel D/panel_d_overall_benchmark_performance.png',
}


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    hashes = {label: sha(path) for label, path in SOURCES.items()}
    panels = {label: Image.open(path).convert('RGB') for label, path in SOURCES.items()}
    assert len({image.width for image in panels.values()}) == 1
    width = panels['A'].width
    gap_x, gap_y = 8, 24
    top_h = max(panels['A'].height, panels['B'].height)
    lower_h = max(panels['C'].height, panels['D'].height)
    canvas = Image.new('RGB', (2*width+gap_x, top_h+gap_y+lower_h), 'white')
    positions = {'A': (0, 0), 'B': (width+gap_x, 0),
                 'C': (0, top_h+gap_y), 'D': (width+gap_x, top_h+gap_y)}
    for label, image in panels.items(): canvas.paste(image, positions[label])
    output = HERE/'figure_2_ABCD.png'
    canvas.save(output, dpi=(210, 210), optimize=True)
    assert all(sha(SOURCES[label]) == digest for label, digest in hashes.items())
    (HERE/'manifest.json').write_text(json.dumps({
        'layout': 'A bone marrow | B lung cancer; C melanoma | D overall performance',
        'source_files': {label: str(path) for label, path in SOURCES.items()},
        'source_sha256': hashes, 'source_files_unchanged': True,
        'panel_pixel_sizes': {label: list(image.size) for label, image in panels.items()},
        'canvas_px': list(canvas.size), 'gap_x_px': gap_x, 'gap_y_px': gap_y,
        'positions_px': positions, 'rescaling': False, 'cropping': False,
        'top_row_padding_under_B_px': top_h-panels['B'].height,
    }, indent=2)+'\n')
    print(output)


if __name__ == '__main__': main()
