"""Improve existing vector workflow labels and remove empty vertical bands."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import subprocess
import fitz
import numpy as np
from PIL import Image
from pypdf.generic import ContentStream, DecodedStreamObject, FloatObject

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'panel_a_ramp_method_overview.pdf'
PANEL_B = HERE.parents[1] / 'Panel B/tnbc_bandwidth_review/adtnorm_style_review/tnbc_ridge_umap_adtnorm_style.png'
FONT = Path('/Users/qxu6/Library/Python/3.14/lib/python/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans-Bold.ttf')
PDF = HERE / 'panel_a_ramp_method_overview_readable.pdf'
PNG = HERE / 'panel_a_ramp_method_overview_readable.png'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def spans(page):
    return [s for b in page.get_text('dict')['blocks'] if b['type'] == 0
            for line in b['lines'] for s in line['spans'] if s['text'].strip()]


def thicken_vector_strokes(doc, page):
    # Edit PDF graphics operators so existing opacity, colors and layering survive.
    xrefs = set(page.get_contents()) | {item[0] for item in page.get_xobjects()}
    changed = []
    for xref in sorted(xrefs):
        decoded = DecodedStreamObject()
        decoded.set_data(doc.xref_stream(xref))
        stream = ContentStream(decoded, None)
        count = 0
        for operands, operator in stream.operations:
            if operator == b'w':
                old = float(operands[0])
                if old > 0:
                    operands[0] = FloatObject(old*1.3)
                    count += 1
        if count:
            stream.operations = stream.operations
            doc.update_stream(xref, stream.get_data())
            changed.append({'xref': xref, 'linewidth_operators': count})
    # The original illustration stores most stroke widths in ExtGState dictionaries.
    for xref in range(1, doc.xref_length()):
        kind, value = doc.xref_get_key(xref, 'LW')
        if kind in ['int', 'float'] and float(value) > 0:
            doc.xref_set_key(xref, 'LW', f'{float(value)*1.3:.8f}')
            changed.append({'xref': xref, 'graphics_state_linewidth': float(value)})
    assert changed
    return changed


def main():
    source_hash, b_hash = sha(SOURCE), sha(PANEL_B)
    doc = fitz.open(SOURCE)
    page = doc[0]
    before_spans = spans(page)
    before_strokes = [d['width'] for d in page.get_drawings() if d['color'] is not None]
    stroke_changes = thicken_vector_strokes(doc, page)
    after_strokes = [d['width'] for d in page.get_drawings() if d['color'] is not None]
    assert len(before_strokes) == len(after_strokes)
    assert np.allclose(np.asarray(after_strokes), np.asarray(before_strokes)*1.3,
                       rtol=1e-5, atol=1e-5)
    # The source PDF used translucent ridges while the approved PNG used solid fill.
    # Match the approved PNG when exporting the edited vector version.
    doc.xref_set_key(page.xref, 'Resources/ExtGState/A2/ca', '1')
    font = fitz.Font(fontfile=str(FONT))
    replacements = []
    for span in before_spans:
        if not (435 < span['bbox'][1] < 690 and span['size'] <= 10.6):
            continue
        text = span['text']
        size = 11.0
        if text in ['Native', 'Preprocessed', 'Cell labels masked', 'Shared markers only']:
            size = 11.5
        if text in ['+', '✓', '×']:
            size = 11.5
        if text == 'Matching weight':
            size = 10.75
        width = font.text_length(text, fontsize=size)
        original_rect = fitz.Rect(span['bbox'])
        x = (original_rect.x0+original_rect.x1-width)/2
        if text in ['Cell labels masked', 'Matching weight']:
            x = span['origin'][0]
        baseline = span['origin'][1]
        new_rect = fitz.Rect(x, baseline-font.ascender*size,
                            x+width, baseline-font.descender*size)
        assert new_rect.x0 >= 0 and new_rect.x1 <= page.rect.width
        replacements.append({'text': text, 'old_size': span['size'], 'new_size': size,
                             'old_rect': list(original_rect), 'new_rect': list(new_rect),
                             'origin': [x, baseline], 'font': 'DejaVu Sans Bold'})
        page.add_redact_annot(original_rect, fill=False, cross_out=False)
    # Only remove original text; preserve drawings and other content behind it.
    page.apply_redactions(images=0, graphics=0, text=0)
    page.insert_font(fontname='ReadableBold', fontfile=str(FONT))
    for edit in replacements:
        page.insert_text(edit['origin'], edit['text'], fontsize=edit['new_size'],
                         fontname='ReadableBold', color=(.09, .09, .09))
    expected = Counter(s['text'] for s in before_spans)
    actual = Counter(s['text'] for s in spans(page))
    assert expected == actual, {'lost': expected-actual, 'added': actual-expected}
    # Check enlarged labels remain separate from each other and from existing text.
    all_spans = spans(page)
    overlaps = []
    for i, left in enumerate(all_spans):
        for right in all_spans[i+1:]:
            if not (435 < left['bbox'][1] < 690 or 435 < right['bbox'][1] < 690):
                continue
            intersection = fitz.Rect(left['bbox']) & fitz.Rect(right['bbox'])
            if not intersection.is_empty and intersection.get_area() > 1:
                overlaps.append((left['text'], right['text'], list(intersection)))
    assert not overlaps, overlaps

    with Image.open(PANEL_B) as im:
        b_size = list(im.size)
    target_height = page.rect.width*b_size[1]/b_size[0]
    remove_total = page.rect.height-target_height
    cuts = [(44., 62.), (363., 375.), (683., 693.), (724., 732.)]
    removed_interior = sum(high-low for low, high in cuts)
    cuts.append((page.rect.height-(remove_total-removed_interior), page.rect.height))
    # Verify each removed band is actually white after the lettering/stroke edits.
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    for low, high in cuts:
        strip = array[int(np.ceil(low*2)):int(np.floor(high*2))]
        assert strip.min() >= 248, (low, high, int(strip.min()))
    output = fitz.open()
    new_page = output.new_page(width=page.rect.width, height=target_height)
    start, output_y = 0., 0.
    retained = []
    for low, high in cuts:
        if low > start:
            source_rect = fitz.Rect(0, start, page.rect.width, low)
            dest_rect = fitz.Rect(0, output_y, page.rect.width, output_y+low-start)
            new_page.show_pdf_page(dest_rect, doc, 0, clip=source_rect)
            retained.append({'source': list(source_rect), 'destination': list(dest_rect)})
            output_y += low-start
        start = high
    assert abs(output_y-target_height) < 1e-5
    output.set_metadata({'title': 'Panel A: Overview of RAMP',
                         'subject': 'Readable matching labels and compact vertical layout'})
    output.save(PDF, garbage=4, deflate=True)
    output.close()
    # Render the complete final PDF using Poppler for visual review and composition.
    width_px = 3600
    height_px = round(width_px*b_size[1]/b_size[0])
    subprocess.run(['/opt/homebrew/bin/pdftoppm', '-png', '-singlefile',
                    '-scale-to-x', str(width_px), '-scale-to-y', str(height_px),
                    str(PDF), str(PNG.with_suffix(''))], check=True)
    assert sha(SOURCE) == source_hash and sha(PANEL_B) == b_hash
    manifest = {
        'source_pdf': str(SOURCE), 'source_pdf_sha256': source_hash,
        'panel_b_png': str(PANEL_B), 'panel_b_png_sha256': b_hash,
        'source_files_unchanged': True, 'wording_unchanged': True,
        'matching_rank_label_added': False,
        'font_file': str(FONT), 'text_changes': replacements,
        'stroke_width_multiplier': 1.3, 'stroke_stream_changes': stroke_changes,
        'section_1_ridge_fill': 'Solid fill to match the previously approved source PNG',
        'stroke_paths_checked': len(before_strokes),
        'original_pdf_size_pt': list(page.rect)[2:],
        'new_pdf_size_pt': [page.rect.width, target_height],
        'removed_height_pt': remove_total,
        'height_reduction_percent': 100*remove_total/page.rect.height,
        'removed_blank_bands_pt': cuts, 'retained_vector_regions': retained,
        'content_stretching': False, 'new_png_size_px': [width_px, height_px],
        'all_original_text_retained': True, 'text_overlaps': overlaps,
        'removed_bands_verified_blank': True,
    }
    (HERE / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'pdf': str(PDF), 'png': str(PNG),
                      'labels_changed': len(replacements),
                      'stroke_paths_boldened': len(before_strokes),
                      'height_reduction_percent': manifest['height_reduction_percent']}, indent=2))


if __name__ == '__main__':
    main()
