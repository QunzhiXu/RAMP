"""Compare A and B at equal width without stretching or forcing equal height."""
from pathlib import Path
import hashlib
import json
from PIL import Image

HERE = Path(__file__).resolve().parent
MAIN = HERE.parent
SOURCES = {
    'A': MAIN / 'Panel A/panel_a_ramp_method_overview.png',
    'B': MAIN / 'Panel B/tnbc_bandwidth_review/adtnorm_style_review/tnbc_ridge_umap_adtnorm_style.png',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    hashes = {label: sha(path) for label, path in SOURCES.items()}
    panels = {label: Image.open(path).convert('RGB') for label, path in SOURCES.items()}
    # Preserve B's native resolution and downsample A isotropically.
    target_width = panels['B'].width
    original_sizes = {label: list(im.size) for label, im in panels.items()}
    resized = {
        label: im.resize((target_width, round(im.height * target_width / im.width)),
                         Image.Resampling.LANCZOS)
        if im.width != target_width else im.copy()
        for label, im in panels.items()
    }
    gap = 32
    height = max(im.height for im in resized.values())
    canvas = Image.new('RGB', (2*target_width+gap, height), 'white')
    canvas.paste(resized['A'], (0, 0))
    canvas.paste(resized['B'], (target_width+gap, 0))
    output = HERE / 'panels_AB_equal_width.png'
    canvas.save(output, dpi=(210, 210), optimize=True)
    assert all(sha(SOURCES[label]) == digest for label, digest in hashes.items())
    ratio = (panels['A'].height/panels['A'].width)/(panels['B'].height/panels['B'].width)
    manifest = {
        'layout': 'A left; B right; equal width; top aligned; original aspect ratios',
        'source_files': {label: str(path) for label, path in SOURCES.items()},
        'source_sha256': hashes, 'source_files_unchanged': True,
        'original_pixel_sizes': original_sizes,
        'combined_panel_sizes_px': {label: list(im.size) for label, im in resized.items()},
        'gap_px': gap, 'canvas_size_px': list(canvas.size),
        'height_A_divided_by_B_at_equal_width': ratio,
        'A_taller_than_B_percent': 100*(ratio-1),
        'B_blank_below_px': height-resized['B'].height,
        'cropping': False, 'nonuniform_scaling': False,
        'panel_content_changed': False,
    }
    (HERE / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(output)
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
