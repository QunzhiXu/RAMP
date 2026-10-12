"""Compose the approved Panel A and Panel B SVGs without changing their source files."""
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
        raise PermissionError(f'Write outside Combined AB: {path}')


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
for panel, filename in [('A', 'panel_A_workflow.svg'), ('B', 'panel_B_alignment.svg')]:
    source = FIGURE / f'Panel {panel}' / filename
    local = BASE / 'inputs' / filename
    digest = sha(source)
    if local.exists():
        if sha(local) != digest:
            archived = BASE / 'versions/v1_three_umaps/inputs' / filename
            assert archived.exists() and sha(archived) == sha(local), 'Archive previous input first.'
            shutil.copyfile(source, local)
    else:
        shutil.copyfile(source, local)
    records.append({'panel': panel, 'source': str(source),
                    'local': str(local.relative_to(BASE)), 'sha256': digest})

def dimensions(record):
    svg = ET.parse(BASE / record['local']).getroot()
    _, _, w, h = map(float, svg.attrib['viewBox'].split())
    return w, h

a_width, a_height = dimensions(records[0])
b_width, b_height = dimensions(records[1])
height, gap = max(a_height + 12, b_height), 12.0
b_x = a_width + gap
width = b_x + b_width
root = ET.Element(tag('svg'), {
    'width': f'{width}pt', 'height': f'{height}pt',
    'viewBox': f'0 0 {width} {height}', 'version': '1.1',
})
ET.SubElement(root, tag('title')).text = 'Panels A–B: HLA-ABC state transfer and alignment'
ET.SubElement(root, tag('rect'), {
    'x': '0', 'y': '0', 'width': str(width), 'height': str(height), 'fill': 'white',
})


def add_panel(record, x, y):
    original = ET.parse(BASE / record['local']).getroot()
    # Keep the body at its native size. Headings are standardized in the composite only.
    removed = []
    for parent in original.iter():
        for child in list(parent):
            if child.tag == tag('text'):
                text = ''.join(child.itertext())
                if text == record['panel'] or text.startswith('Cross-platform'):
                    removed.append(text)
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
        'id': f"panel_{record['panel']}_body", 'transform': f'translate({x} {y})',
    })
    for child in original:
        if child.tag not in {tag('metadata'), tag('title'), tag('desc')}:
            group.append(child)


# Align the workflow boxes vertically with the UMAP plotting region.
add_panel(records[0], 0, 12)
add_panel(records[1], b_x, 0)

for x, letter, title in [
    (0, 'A', 'Cross-platform HLA-ABC state transfer'),
    (b_x, 'B', 'Cross-platform alignment'),
]:
    for tx, ty, text, size in [
        (x + 10.8864, 28.390625, letter, 22),
        (x + 44.3232, 25.714031, title, 12.3),
    ]:
        element = ET.SubElement(root, tag('text'), {
            'x': str(tx), 'y': str(ty),
            'style': f"font-weight:700;font-size:{size}px;font-family:'DejaVu Sans';fill:#243745",
        })
        element.text = text

# Check all internal SVG references after namespacing both source panels.
ids = [element.attrib['id'] for element in root.iter() if 'id' in element.attrib]
assert len(ids) == len(set(ids))
for element in root.iter():
    for key, value in element.attrib.items():
        for ref in re.findall(r'url\(#([^)]+)\)', value):
            assert ref in ids, ref
        if key in {'href', f'{{{XLINK}}}href'} and value.startswith('#'):
            assert value[1:] in ids, value
svg_path = BASE / 'panels_AB_preview.svg'
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
with Image.open(BASE / 'panels_AB_preview.png') as im:
    size = list(im.size)
    im.load()
    im.save(BASE / 'panels_AB_preview.png', dpi=(300, 300))
assert abs(size[0] - width * 300 / 72) <= 1
assert abs(size[1] - height * 300 / 72) <= 1
for record in records:
    assert sha(Path(record['source'])) == sha(BASE / record['local']) == record['sha256']
write_json(BASE / 'provenance/SOURCES.json', records)
write_json(BASE / 'provenance/VERIFICATION.json', {
    'status': 'PASS', 'version': 'v2_aligned_two_views', 'source_panels_unchanged': True,
    'layout': 'A left; B right; one row', 'body_scale_A': 1, 'body_scale_B': 1,
    'A_body_y_offset_pt': 12, 'gap_pt': gap, 'common_title_size_pt': 12.3,
    'common_panel_letter_size_pt': 22, 'svg_ids_unique': True,
    'all_svg_internal_references_valid': True, 'PNG_dimensions': size, 'dpi': 300,
    'data_or_predictions_changed': False, 'B_views': ['Platform', 'Cell type'],
    'before_after_comparison': False,
})
write_json(BASE / 'provenance/MANIFEST.json', [
    {'file': str(path.relative_to(BASE)), 'bytes': path.stat().st_size, 'sha256': sha(path)}
    for path in sorted(BASE.rglob('*')) if path.is_file()
    and path.relative_to(BASE).parts[0] not in {'cache', 'tmp', 'versions'}
    and path.name != 'MANIFEST.json'
])
print(f'Created {svg_path} and PNG {size}; source files unchanged.', flush=True)
