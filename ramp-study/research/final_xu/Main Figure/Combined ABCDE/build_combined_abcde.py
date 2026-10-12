"""Assemble the current full-size A+B preview above C, D and E."""
from pathlib import Path
import hashlib
import json
from PIL import Image, ImageChops

HERE = Path(__file__).resolve().parent
MAIN = HERE.parent


def main():
    paths = [MAIN/'Combined AB/panels_AB_equal_width.png',
             MAIN/'Panel C/panel_c_benchmark_performance.png',
             MAIN/'Panel D/panel_d_marker_artifacts.png',
             MAIN/'Panel E/panel_e_rank_preservation.png']
    images = [Image.open(p).convert('RGB') for p in paths]
    top, *lower = images
    assert [im.width for im in lower] == [1590, 1590, 806]
    assert len(set(im.height for im in lower)) == 1
    gaps_x, gap_y = [16, 4], 24
    assert sum(im.width for im in lower)+sum(gaps_x) == top.width
    # Remove only the common white footer after the three visible bottoms align.
    content_bottom = max(ImageChops.difference(im, Image.new('RGB', im.size, 'white')).getbbox()[3]
                         for im in lower)
    lower_height = min(lower[0].height, content_bottom+20)
    for im in lower:
        footer = im.crop((0, lower_height, im.width, im.height))
        assert ImageChops.difference(footer, Image.new('RGB', footer.size, 'white')).getbbox() is None
    canvas = Image.new('RGB', (top.width, top.height+gap_y+lower_height), 'white')
    canvas.paste(top, (0, 0))
    x = 0
    for i, im in enumerate(lower):
        canvas.paste(im, (x, top.height+gap_y))
        x += im.width+(gaps_x[i] if i < len(gaps_x) else 0)
    output = HERE/'main_figure_ABCDE.png'
    canvas.save(output, dpi=(210, 210), optimize=True)
    (HERE/'manifest.json').write_text(json.dumps({
        'layout': 'A+B above C+D+E', 'source_pngs': [str(p) for p in paths],
        'input_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'canvas_px': list(canvas.size), 'bottom_panel_widths_px': [im.width for im in lower],
        'row_gap_px': gap_y, 'bottom_column_gaps_px': gaps_x,
        'panels_rescaled': False, 'panels_cropped': lower_height < lower[0].height,
        'cropping': 'Only the common all-white bottom footer; no plot content cropped.',
        'bottom_row_visible_height_px': lower_height,
        'bottom_white_footer_trim_px': lower[0].height-lower_height,
        'bottom_content_padding_px': 20,
    }, indent=2)+'\n')
    print(output)


if __name__ == '__main__': main()
