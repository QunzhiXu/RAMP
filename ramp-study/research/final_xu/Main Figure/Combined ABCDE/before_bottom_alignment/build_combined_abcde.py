"""Assemble the current full-size A+B preview above C, D and E."""
from pathlib import Path
import hashlib
import json
from PIL import Image

HERE = Path(__file__).resolve().parent
MAIN = HERE.parent


def main():
    paths = [MAIN/'Combined AB/panels_AB_equal_width.png',
             MAIN/'Panel C/panel_c_benchmark_performance.png',
             MAIN/'Panel D/panel_d_marker_artifacts.png',
             MAIN/'Panel E/panel_e_rank_preservation.png']
    images = [Image.open(p).convert('RGB') for p in paths]
    top, *lower = images
    assert [im.width for im in lower] == [1590, 1590, 794]
    assert len(set(im.height for im in lower)) == 1
    gap_x, gap_y = 16, 24
    assert sum(im.width for im in lower)+2*gap_x == top.width
    canvas = Image.new('RGB', (top.width, top.height+gap_y+lower[0].height), 'white')
    canvas.paste(top, (0, 0))
    x = 0
    for im in lower:
        canvas.paste(im, (x, top.height+gap_y)); x += im.width+gap_x
    output = HERE/'main_figure_ABCDE.png'
    canvas.save(output, dpi=(210, 210), optimize=True)
    (HERE/'manifest.json').write_text(json.dumps({
        'layout': 'A+B above C+D+E', 'source_pngs': [str(p) for p in paths],
        'input_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'canvas_px': list(canvas.size), 'bottom_panel_widths_px': [im.width for im in lower],
        'row_gap_px': gap_y, 'bottom_column_gap_px': gap_x,
        'panels_rescaled': False, 'panels_cropped': False,
    }, indent=2)+'\n')
    print(output)


if __name__ == '__main__': main()
