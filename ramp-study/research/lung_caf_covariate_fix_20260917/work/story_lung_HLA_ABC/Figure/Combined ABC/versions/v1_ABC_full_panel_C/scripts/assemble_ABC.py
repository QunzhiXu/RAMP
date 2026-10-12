"""Compose the approved Panel A, Panel B and Panel C SVGs without changing their source files."""
from pathlib import Path
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

BASE = Path(__file__).resolve().parents[1]
FIGURE = BASE.parent
sys.dont_write_bytecode = True


def allowed(path):
    if isinstance(path, int) or path is None:
        return
    path = Path(os.fsdecode(path)).resolve()
    if path != Path('/dev/null') and not path.is_relative_to(BASE):
        raise PermissionError(f'Write outside Combined ABC: {path}')


def audit(event, args):
    if event == 'open':
        path, mode, flags = args
        if (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (
            isinstance(flags, int) and flags & (
                os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
            )
        ):
            allowed(path)
    elif event in {'os.remove', 'os.rmdir', 'os.mkdir', 'os.chmod', 'os.utime', 'os.truncate'}:
        allowed(args[0])
    elif event in {'os.rename', 'os.replace'}:
        allowed(args[0]); allowed(args[1])


sys.addaudithook(audit)
for relative in ['inputs', 'provenance', 'cache/fontconfig', 'cache/matplotlib', 'tmp']:
    (BASE / relative).mkdir(parents=True, exist_ok=True)
os.environ['TMPDIR'] = str(BASE / 'tmp')
os.environ['XDG_CACHE_HOME'] = str(BASE / 'cache')
os.environ['MPLCONFIGDIR'] = str(BASE / 'cache/matplotlib')

SVG = 'http://www.w3.org/2000/svg'
XLINK = 'http://www.w3.org/1999/xlink'
ET.register_namespace('', SVG)
ET.register_namespace('xlink', XLINK)


def tag(local):
    return f'{{{SVG}}}{local}'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')


records = []
for panel, filename in [('A', 'panel_A_workflow.svg'), ('B', 'panel_B_alignment.svg'),
                        ('C', 'panel_C_vcaf.svg')]:
    source = FIGURE / f'Panel {panel}' / filename
    local = BASE / 'inputs' / filename
    digest = sha(source)
    if local.exists():
        if sha(local) != digest:
            archives = list((BASE / 'versions').glob(f'*/inputs/{filename}'))
            assert any(sha(p) == sha(local) for p in archives), 'Archive previous input first.'
            shutil.copyfile(source, local)
    else:
        shutil.copyfile(source, local)
    records.append({'panel': panel, 'source': str(source),
                    'local': str(local.relative_to(BASE)), 'sha256': digest})

def dimensions(record):
    svg = ET.parse(BASE / record['local']).getroot()
    _, _, w, h = map(float, svg.attrib['viewBox'].split())
    return w, h

a_source_width, a_source_height = dimensions(records[0])
b_width, b_height = dimensions(records[1])
c_width, c_height = dimensions(records[2])
assert b_height == c_height
# Crop the workflow's outer canvas margins and scale its body to the UMAP height.
# Bounds include the full left platform frame and the right output frames.
a_scale = .78
a_body_left, a_body_right, a_body_top = 28.0, 416.0, 50.4
a_left_margin, a_right_margin = 6.0, 4.0
a_width = (a_body_right - a_body_left) * a_scale + a_left_margin + a_right_margin
a_x = a_left_margin - a_body_left * a_scale
b_plot = json.loads((FIGURE / 'Panel B/provenance/VERIFICATION.json').read_text())
b_plot_top = b_height - 72 * (b_plot['plot_bottom_inches'] + b_plot['plot_size_inches'][1])
a_y = b_plot_top - a_body_top * a_scale
height, gap = b_height, 3.0
b_x = a_width + gap
c_x = b_x + b_width + gap
width = c_x + c_width
root = ET.Element(tag('svg'), {
    'width': f'{width}pt', 'height': f'{height}pt',
    'viewBox': f'0 0 {width} {height}', 'version': '1.1',
})
ET.SubElement(root, tag('title')).text = 'Panels A–C: HLA-ABC state transfer, alignment and vCAF definition'
ET.SubElement(root, tag('rect'), {
    'x': '0', 'y': '0', 'width': str(width), 'height': str(height), 'fill': 'white',
})


def add_panel(record, x, y, scale=1.0):
    original = ET.parse(BASE / record['local']).getroot()
    # Standardize headings separately and remove the original outer white canvas.
    removed = []
    for parent in original.iter():
        for child in list(parent):
            if child.tag == tag('text'):
                text = ''.join(child.itertext())
                if text == record['panel'] or text.startswith('Cross-platform') or text == 'vCAF Definition':
                    removed.append(text)
                    parent.remove(child)
            elif parent.attrib.get('id') == 'figure_1' and child.attrib.get('id') == 'patch_1':
                parent.remove(child)
    assert len(removed) == 2, removed

    prefix = f"panel{record['panel']}_"
    id_map = {element.attrib['id']: prefix + element.attrib['id']
              for element in original.iter() if 'id' in element.attrib}
    for element in original.iter():
        for key, value in list(element.attrib.items()):
            if key == 'id':
                element.set(key, id_map[value])
            elif key in {'href', f'{{{XLINK}}}href'} and value.startswith('#'):
                element.set(key, '#' + id_map[value[1:]])
            else:
                element.set(key, re.sub(r'url\(#([^)]+)\)',
                                       lambda m: f'url(#{id_map[m.group(1)]})', value))
    group = ET.SubElement(root, tag('g'), {
        'id': f"panel_{record['panel']}_body", 'transform': f'translate({x} {y}) scale({scale})',
    })
    for child in original:
        if child.tag not in {tag('metadata'), tag('title'), tag('desc')}:
            group.append(child)


# Align the workflow boxes vertically with the UMAP plotting region.
add_panel(records[0], a_x, a_y, a_scale)
add_panel(records[1], b_x, 0)
add_panel(records[2], c_x, 0)

for x, letter, title in [
    (0, 'A', 'Cross-platform HLA-ABC state transfer'),
    (b_x, 'B', 'Cross-platform alignment'),
    (c_x, 'C', 'vCAF Definition'),
]:
    for tx, ty, text, size in [
        (x + 6, 21.0, letter, 22),
        (x + 36, 18.5, title, 12.3),
    ]:
        element = ET.SubElement(root, tag('text'), {
            'x': str(tx), 'y': str(ty),
            'style': f"font-weight:700;font-size:{size}px;font-family:'DejaVu Sans';fill:#243745",
        })
        element.text = text

# Check all internal SVG references after namespacing the three source panels.
ids = [element.attrib['id'] for element in root.iter() if 'id' in element.attrib]
assert len(ids) == len(set(ids))
for element in root.iter():
    for key, value in element.attrib.items():
        for ref in re.findall(r'url\(#([^)]+)\)', value):
            assert ref in ids, ref
        if key in {'href', f'{{{XLINK}}}href'} and value.startswith('#'):
            assert value[1:] in ids, value
svg_path = BASE / 'panels_ABC_preview.svg'
ET.ElementTree(root).write(svg_path, encoding='utf-8', xml_declaration=True)

# Supply the same DejaVu family used by the original matplotlib figures to librsvg.
import matplotlib
font_dir = Path(matplotlib.get_data_path()) / 'fonts/ttf'
conf = ET.Element('fontconfig')
ET.SubElement(conf, 'dir').text = str(font_dir)
ET.SubElement(conf, 'cachedir').text = str(BASE / 'cache/fontconfig')
conf_path = BASE / 'cache/fonts.conf'
ET.ElementTree(conf).write(conf_path, encoding='utf-8', xml_declaration=True)
os.environ['FONTCONFIG_FILE'] = str(conf_path)
node = os.environ.get('PANEL_NODE',
    '/Users/qxu6/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node')
subprocess.run([node, str(BASE / 'scripts/render_svg.cjs')], check=True, env=os.environ.copy())

from PIL import Image
with Image.open(BASE / 'panels_ABC_preview.png') as im:
    size = list(im.size)
    im.load()
    im.save(BASE / 'panels_ABC_preview.png', dpi=(300, 300))
assert abs(size[0] - width * 300 / 72) <= 1
assert abs(size[1] - height * 300 / 72) <= 1
for record in records:
    assert sha(Path(record['source'])) == sha(BASE / record['local']) == record['sha256']
write_json(BASE / 'provenance/SOURCES.json', records)
c_plot = json.loads((FIGURE / 'Panel C/provenance/VERIFICATION.json').read_text())
assert c_plot['plot_bottom_inches'] == b_plot['plot_bottom_inches']
assert c_plot['plot_size_inches'][1] == b_plot['plot_size_inches'][1]
write_json(BASE / 'provenance/VERIFICATION.json', {
    'status': 'PASS', 'version': 'v1_ABC_full_panel_C', 'source_panels_unchanged': True,
    'layout': 'Compact horizontal A, B, C', 'body_scale_A': a_scale,
    'body_scale_B': 1, 'body_scale_C': 1, 'A_body_offset_pt': [a_x, a_y],
    'gap_pt': gap, 'common_title_size_pt': 12.3,
    'common_panel_letter_size_pt': 22, 'svg_ids_unique': True,
    'all_svg_internal_references_valid': True, 'PNG_dimensions': size, 'dpi': 300,
    'data_or_predictions_changed': False, 'B_views': ['Platform', 'Cell type'],
    'before_after_comparison': False,
    'panel_A_width_pt': a_width, 'panel_B_width_pt': b_width, 'panel_C_width_pt': c_width,
    'canvas_size_pt': [width, height], 'panel_origins_pt': {'A': [0, 0], 'B': [b_x, 0], 'C': [c_x, 0]},
    'B_legend_position': 'right of each UMAP, centered vertically',
    'B_n_display_groups': b_plot['n_display_groups'],
    'B_CD4_above_fibroblast_and_myeloid': b_plot['CD4_above_fibroblast_and_myeloid'],
    'C_source_version': c_plot['version'],
    'C_title': 'vCAF Definition', 'C_subtitle': 'Full-panel CAF UMAP',
    'C_paper_citation_displayed': False,
    'C_embedding': 'Original Cords measured 41-protein CAF-only UMAP',
    'B_C_plot_height_and_baseline_matched': True,
})
write_json(BASE / 'provenance/MANIFEST.json', [
    {'file': str(path.relative_to(BASE)), 'bytes': path.stat().st_size, 'sha256': sha(path)}
    for path in sorted(BASE.rglob('*')) if path.is_file()
    and path.relative_to(BASE).parts[0] not in {'cache', 'tmp', 'versions'}
    and path.name != 'MANIFEST.json'
])
print(f'Created {svg_path} and PNG {size}; source files unchanged.', flush=True)
