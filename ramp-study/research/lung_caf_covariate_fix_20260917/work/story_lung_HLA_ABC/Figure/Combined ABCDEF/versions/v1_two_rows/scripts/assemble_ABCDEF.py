"""Compose the approved ABC row and the new D/E/F row without changing source panels."""
from runtime import BASE
from pathlib import Path
import hashlib, json, os, re, shutil, subprocess
import xml.etree.ElementTree as ET

FIGURE = BASE.parent
SVG = 'http://www.w3.org/2000/svg'
XLINK = 'http://www.w3.org/1999/xlink'
ET.register_namespace('', SVG); ET.register_namespace('xlink', XLINK)
def tag(s): return f'{{{SVG}}}{s}'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p, v): p.write_text(json.dumps(v, ensure_ascii=False, indent=2) + '\n')

for directory in ['inputs', 'provenance', 'cache/fontconfig']:
    (BASE / directory).mkdir(parents=True, exist_ok=True)
specs = [
    ('ABC', 'Combined ABC', 'panels_ABC_preview.svg', None),
    ('D', 'Panel D', 'panel_D_validation.svg', 'Validation in mIF'),
    ('E', 'Panel E', 'panel_E_immune_composition.svg', 'Immune composition'),
    ('F', 'Panel F', 'panel_F_survival.svg', 'vCAF HLA-ABC and DFS'),
]
records = []
for key, folder, filename, title in specs:
    source = FIGURE / folder / filename
    local = BASE / 'inputs' / filename
    if local.exists(): assert sha(local) == sha(source), 'Archive the previous composite before updating inputs.'
    else: shutil.copyfile(source, local)
    view = ET.parse(local).getroot().attrib['viewBox'].split()
    records.append({'key': key, 'title': title, 'source': str(source),
                    'local': str(local.relative_to(BASE)), 'sha256': sha(source),
                    'width_pt': float(view[2]), 'height_pt': float(view[3])})

width, row_height = records[0]['width_pt'], records[0]['height_pt']
assert all(r['height_pt'] == row_height for r in records)
gap, row_gap = 3.0, 8.0
second_width = sum(r['width_pt'] for r in records[1:]) + 2 * gap
assert abs(second_width - width) < 1e-8
second_y = row_height + row_gap
height = second_y + row_height
root = ET.Element(tag('svg'), {'width': f'{width}pt', 'height': f'{height}pt',
                  'viewBox': f'0 0 {width} {height}', 'version': '1.1'})
ET.SubElement(root, tag('title')).text = 'Lung HLA-ABC story: panels A–F'
ET.SubElement(root, tag('rect'), {'x': '0', 'y': '0', 'width': str(width), 'height': str(height), 'fill': 'white'})

def embed(record, x, y):
    source = ET.parse(BASE / record['local']).getroot()
    if record['key'] != 'ABC':
        removed = []
        for parent in source.iter():
            for child in list(parent):
                if child.tag == tag('text') and ''.join(child.itertext()) in {record['key'], record['title']}:
                    removed.append(''.join(child.itertext())); parent.remove(child)
                elif parent.attrib.get('id') == 'figure_1' and child.attrib.get('id') == 'patch_1':
                    parent.remove(child)
        assert len(removed) == 2, removed
    prefix = 'composite_' + record['key'] + '_'
    names = {e.attrib['id']: prefix + e.attrib['id'] for e in source.iter() if 'id' in e.attrib}
    for element in source.iter():
        for key, value in list(element.attrib.items()):
            if key == 'id': element.set(key, names[value])
            elif key in {'href', f'{{{XLINK}}}href'} and value.startswith('#'):
                element.set(key, '#' + names[value[1:]])
            else:
                element.set(key, re.sub(r'url\(#([^)]+)\)', lambda m: f'url(#{names[m.group(1)]})', value))
    group = ET.SubElement(root, tag('g'), {'id': f"body_{record['key']}", 'transform': f'translate({x} {y})'})
    for child in source:
        if child.tag not in {tag('metadata'), tag('title'), tag('desc')}: group.append(child)
    if record['key'] != 'ABC':
        for dx, dy, text, size in [(6, 21, record['key'], 22), (36, 18.5, record['title'], 12.3)]:
            element = ET.SubElement(root, tag('text'), {'x': str(x + dx), 'y': str(y + dy),
                'style': f"font-weight:700;font-size:{size}px;font-family:'DejaVu Sans';fill:#243745"})
            element.text = text

embed(records[0], 0, 0)
x = 0
positions = {'ABC': [0, 0]}
for record in records[1:]:
    embed(record, x, second_y)
    positions[record['key']] = [x, second_y]
    x += record['width_pt'] + gap
ids = [e.attrib['id'] for e in root.iter() if 'id' in e.attrib]
assert len(ids) == len(set(ids))
for element in root.iter():
    for key, value in element.attrib.items():
        for ref in re.findall(r'url\(#([^)]+)\)', value): assert ref in ids
        if key in {'href', f'{{{XLINK}}}href'} and value.startswith('#'): assert value[1:] in ids
output = BASE / 'panels_ABCDEF_preview.svg'
ET.ElementTree(root).write(output, encoding='utf-8', xml_declaration=True)

import matplotlib
fonts = ET.Element('fontconfig')
ET.SubElement(fonts, 'dir').text = str(Path(matplotlib.get_data_path()) / 'fonts/ttf')
ET.SubElement(fonts, 'cachedir').text = str(BASE / 'cache/fontconfig')
conf = BASE / 'cache/fonts.conf'
ET.ElementTree(fonts).write(conf, encoding='utf-8', xml_declaration=True)
os.environ['FONTCONFIG_FILE'] = str(conf)
node = os.environ.get('PANEL_NODE', '/Users/qxu6/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node')
subprocess.run([node, str(BASE / 'scripts/render_svg.cjs')], env=os.environ.copy(), check=True)
from PIL import Image
with Image.open(BASE / 'panels_ABCDEF_preview.png') as image:
    size = list(image.size); image.load()
    image.save(BASE / 'panels_ABCDEF_preview.png', dpi=(300, 300))
assert abs(size[0] - width * 300 / 72) <= 1 and abs(size[1] - height * 300 / 72) <= 1

# Check shared patient identity and color semantics directly from E and F packages.
import pandas as pd
e = pd.read_csv(FIGURE / 'Panel E/data/PLOT_PATIENTS.csv')
f = pd.read_csv(FIGURE / 'Panel F/data/KM_PATIENTS.csv')
joint = e.merge(f, on='patient_id', validate='one_to_one', suffixes=('_E', '_F'))
assert len(joint) == 280 and joint.group_E.equals(joint.group_F)
ec = json.loads((FIGURE / 'Panel E/provenance/VERIFICATION.json').read_text())['group_colors']
fc = json.loads((FIGURE / 'Panel F/provenance/VERIFICATION.json').read_text())['group_colors']
assert ec == fc
for r in records: assert sha(Path(r['source'])) == sha(BASE / r['local']) == r['sha256']
js(BASE / 'provenance/SOURCES.json', records)
js(BASE / 'provenance/VERIFICATION.json', {
    'status': 'PASS', 'version': 'v1_two_rows', 'rows': [['A', 'B', 'C'], ['D', 'E', 'F']],
    'row_1': 'Approved compact ABC v2 with narrower BC spacing',
    'panel_origins_pt': positions, 'row_gap_pt': row_gap, 'second_row_gap_pt': gap,
    'canvas_size_pt': [width, height], 'PNG_dimensions': size, 'dpi': 300,
    'native_source_panel_sizes_retained': True, 'common_title_size_pt': 12.3, 'common_letter_size_pt': 22,
    'source_files_unchanged': True, 'svg_ids_unique': True, 'internal_references_valid': True,
    'E_F_same_280_patients_and_groups': True, 'E_F_group_colors': ec,
    'statistics_or_models_refit': False})
js(BASE / 'provenance/MANIFEST.json', [{'file': str(p.relative_to(BASE)), 'sha256': sha(p)}
   for p in sorted(BASE.rglob('*')) if p.is_file() and p.relative_to(BASE).parts[0] not in {'cache', 'tmp', 'versions'} and p.name != 'MANIFEST.json'])
print(f'Created A–F two-row SVG and PNG {size}.', flush=True)
