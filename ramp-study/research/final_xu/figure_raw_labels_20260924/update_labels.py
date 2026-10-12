"""Publication-only label revision, preserving every non-label PDF drawing command."""
from pathlib import Path
import hashlib, json, re, shutil, math
import fitz
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MANUSCRIPT = ROOT.parent / 'bioarxiv_Ver2'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

def literal(text):
    return text.encode('utf-16be').replace(b'\\', b'\\\\').replace(b'(', b'\\(').replace(b')', b'\\)')

def labels(page, target):
    return [(line, span) for block in page.get_text('dict')['blocks']
            for line in block.get('lines', []) for span in line['spans']
            if span['text'] == target]

def edit_label(source, dest, old):
    doc = fitz.open(source)
    assert len(doc) == 1
    page = doc[0]
    spans = labels(page, old)
    assert len(spans) == 1, (source, spans)
    line, span = spans[0]
    bbox = fitz.Rect(span['bbox'])
    streams = page.get_contents()
    needle = b'[ (' + literal(old) + b') ] TJ'
    hits = [n for n in streams if needle in doc.xref_stream(n)]
    assert len(hits) == 1
    n = hits[0]
    content = doc.xref_stream(n)
    assert content.count(needle) == 1
    new_content = content.replace(needle, b'[ (' + literal('Raw') + b') ] TJ')
    centered = False
    if line['dir'] == (0.0, -1.0):
        # Keep the vertical main-figure method label centered in its original row.
        rawdict = page.get_text('rawdict')
        chars = [s['chars'] for b in rawdict['blocks'] for l in b.get('lines', [])
                 for s in l['spans'] if ''.join(c['c'] for c in s['chars']) == old][0]
        raw_height = chars[0]['bbox'][3] - chars[2]['bbox'][1]
        shift = (bbox.height - raw_height) / 2
        pattern = rb'(0 1 -1 0 [\d.]+ )([\d.]+)( cm\nBT\n/F1 6\.3 Tf\n0 0 Td\n)' + re.escape(needle)
        match = re.search(pattern, content)
        assert match
        replacement = match.group(1) + f'{float(match.group(2))+shift:.10f}'.encode() + match.group(3) + b'[ (' + literal('Raw') + b') ] TJ'
        new_content = content[:match.start()] + replacement + content[match.end():]
        centered = True
    else:
        assert line['dir'] == (1.0, 0.0)
    doc.update_stream(n, new_content)
    dest.parent.mkdir(parents=True, exist_ok=True)
    doc.save(dest, garbage=0, deflate=True)
    doc.close()
    before = fitz.open(source)
    after = fitz.open(dest)
    assert before[0].rect == after[0].rect
    assert old not in after[0].get_text()
    assert before[0].get_text().replace(old, 'Raw') == after[0].get_text()
    for other in streams:
        if other != n:
            assert before.xref_stream(other) == after.xref_stream(other)
    # Only the specified text operator and (for vertical Raw) its centering change.
    assert after.xref_stream(n) == new_content
    zoom = 2
    a = before[0].get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    b = after[0].get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    aa = np.frombuffer(a.samples, np.uint8).reshape(a.height, a.width, a.n)
    bb = np.frombuffer(b.samples, np.uint8).reshape(b.height, b.width, b.n)
    delta = np.any(aa != bb, axis=2)
    region = fitz.Rect(bbox.x0-1, bbox.y0-1, bbox.x1+1, bbox.y1+1)
    y0,y1 = max(0,math.floor(region.y0*zoom)),math.ceil(region.y1*zoom)
    x0,x1 = max(0,math.floor(region.x0*zoom)),math.ceil(region.x1*zoom)
    changed = int(delta.sum())
    delta[y0:y1,x0:x1] = False
    assert not delta.any(), (source, int(delta.sum()))
    png = HERE / 'output/png' / (dest.stem + '.png')
    png.parent.mkdir(parents=True, exist_ok=True)
    b.save(png)
    return dict(source=str(source), destination=str(dest), old_label=old, new_label='Raw',
                old_sha256=sha(source), sha256=sha(dest), old_label_bbox=list(bbox),
                vertical_center_preserved=centered, non_label_pixels_identical=True,
                density_and_umap_drawing_commands_unchanged=True, changed_pixels=changed)

def main():
    backup = HERE / 'backup_before_revision'
    assert not backup.exists(), 'Backups already exist; use preserved outputs rather than overwriting history.'
    backup.mkdir()
    hashes = {str(p.relative_to(MANUSCRIPT)):sha(p) for p in MANUSCRIPT.rglob('*') if p.is_file()}
    (HERE/'BEFORE_HASHES.json').write_text(json.dumps(hashes, indent=2)+'\n')
    for rel in ['main.tex','main.pdf','README.md','COMPILATION_CHECK.json','PROVENANCE.json','FILE_MANIFEST.csv']:
        shutil.copy2(MANUSCRIPT/rel, backup/rel)
    for rel in ['Main Figure/README.md','Supplementary Figures/README.md','Supplementary Figures/CURRENT_PUBLICATION.json']:
        out = backup/'local_docs'/rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/rel, out)
    prov = json.loads((MANUSCRIPT/'PROVENANCE.json').read_text())
    items = [dict(number='1',source=prov['figure1']['source'],destination='figure/figure1_ramp_overview.pdf',old='Raw (arcsinh)')]
    items += [dict(number=p['number'],source=p['source'],destination=p['destination'],old='Raw (assay-transformed)')
              for p in prov['supplementary'] if p['number'] in [f'S{i}' for i in range(1,8)]]
    assert len(items) == 8
    qa = []
    stage = HERE/'stage'
    (stage/'figure/supplementary').mkdir(parents=True)
    replaced = {p['destination'] for p in items}
    for p in (MANUSCRIPT/'figure').rglob('*'):
        if not p.is_file(): continue
        rel = str(p.relative_to(MANUSCRIPT))
        if rel not in replaced:
            out=stage/rel; out.parent.mkdir(parents=True,exist_ok=True);out.symlink_to(p)
    for item in items:
        source=Path(item['source'])
        assert sha(source) == sha(MANUSCRIPT/item['destination'])
        out=backup/item['destination'];out.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,out)
        output=HERE/'output/pdf'/source.name
        result=edit_label(source,output,item['old'])
        result.update(number=item['number'],manuscript_destination=item['destination'])
        qa.append(result)
        shutil.copy2(output,stage/item['destination'])
    for name in ['main.tex','bibliography.bib','ramp_numeric.bst','ramp_harvard.bst']:
        shutil.copy2(MANUSCRIPT/name,stage/name)
    (stage/'supplementary_data').symlink_to(MANUSCRIPT/'supplementary_data',target_is_directory=True)
    (HERE/'LABEL_QA.json').write_text(json.dumps(qa,indent=2)+'\n')
    (HERE/'SOURCE_MANIFEST.json').write_text(json.dumps(items,indent=2)+'\n')
    print(json.dumps({'figures_updated':len(qa),'all_non_label_pixels_identical':True,'stage':str(stage)},indent=2))

if __name__ == '__main__':main()
