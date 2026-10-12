"""Update publication labels/layout only; frozen scores and source arrays stay intact."""
from pathlib import Path
import importlib.util
import hashlib
import json
import sys
import os
import xml.etree.ElementTree as ET
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent / '.mplconfig'))
os.environ.setdefault('MPLBACKEND', 'Agg')
import fitz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LIVE = ROOT.parent / 'bioRxiv_Ver1'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def figure3():
    base = ROOT / 'cross_platform_trainability_v2_20260919/Figure'
    out = base / 'Combined ABC'
    out.mkdir(exist_ok=True)
    m = module(base / 'build_panels_bc.py', 'bc')
    m.HERE = out
    m.LABELS['harmony_direct'] = 'Harmony'
    summary, raw, benchmark, error = m.load_scores()
    plt.rcParams.update({'font.family':'Arial', 'font.size':10,
        'text.color':'#202C32', 'axes.labelcolor':'#202C32',
        'xtick.color':'#202C32', 'ytick.color':'#202C32',
        'svg.fonttype':'none', 'pdf.fonttype':42, 'savefig.facecolor':'white'})
    for letter in 'BC':
        folder, frame = m.write_panel_data(letter, raw, benchmark, summary, error)
        for filename in ['per_benchmark_scores.csv', 'per_model_scores.csv']:
            old = pd.read_csv(base / f'Panel {letter}' / filename)
            new = pd.read_csv(folder / filename)
            pd.testing.assert_frame_equal(old, new)
        fig = plt.figure(figsize=m.PANEL_SIZE, facecolor='white')
        m.draw_panel(fig, letter, frame)
        for ext in ['pdf','png','svg']:
            fig.savefig(folder / f"{m.SETTINGS[letter]['stem']}.{ext}", dpi=320)
        plt.close(fig)
    code = (base / 'build_panels_def.py').read_text()
    start = code.index('def combine_vector_files(')
    end = code.index('\ndef ', start+4)
    namespace = dict(fitz=fitz, ET=ET, json=json, sha=sha)
    exec(code[start:end], namespace)
    sources = [base/'Panel A/panel_a_cross_platform_learning',
               out/'Panel B/panel_b_cell_type_annotation',
               out/'Panel C/panel_c_missing_marker_prediction']
    rects = [[0,0,479.52,289.44], [480.52,0,935.56,289.44], [936.56,0,1391.6,289.44]]
    namespace['combine_vector_files'](sources, rects, 1391.6, 289.44,
                                       out/'figure_cross_platform_learning_ABC')
    layout = json.loads((out/'LAYOUT.json').read_text())
    layout.update(draft_layout=False, panels=['A','B','C'], removed_panels=['D','E','F'],
                  frozen_b_c_scores_unchanged=True, display_name='Harmony')
    (out/'LAYOUT.json').write_text(json.dumps(layout,indent=2)+'\n')
    print('Figure 3 A-C rebuilt; B/C data unchanged.', flush=True)

def supplementary():
    out = ROOT / 'Supplementary Figures/harmony_display_20260922'
    out.mkdir(exist_ok=True)
    (out/'pdf').mkdir(exist_ok=True)
    (out/'png').mkdir(exist_ok=True)
    (out/'svg').mkdir(exist_ok=True)
    sources = json.loads((LIVE/'SUPPLEMENTARY_FIGURE_SOURCES.json').read_text())
    fontpath = '/System/Library/Fonts/Supplemental/Arial Bold.ttf'
    font = fitz.Font(fontfile=fontpath)
    records = []
    for item in sources:
        src = Path(item.get('previous_source', item['source']))
        assert sha(src) == item.get('previous_sha256', item['sha256'])
        d = fitz.open(src)
        p = d[0]
        spans = [s for b in p.get_text('dict')['blocks'] for ln in b.get('lines',[])
                 for s in ln['spans'] if s['text'] in ['Harmony-direct','Harmony direct']]
        assert spans
        number = int(item['number'][1:])
        for s in spans:
            r = fitz.Rect(s['bbox'])
            p.add_redact_annot(r, fill=False, cross_out=False)
        p.apply_redactions(images=0, graphics=0)
        for s in spans:
            x,y = s['origin']
            if number >= 8:  # metric plots use right-aligned method tick labels
                x = s['bbox'][2] - font.text_length('Harmony', fontsize=s['size'])
            p.insert_text((x,y),'Harmony',fontfile=fontpath,fontname='HarmonyLabel',
                          fontsize=s['size'],color=fitz.sRGB_to_pdf(s['color']))
        target = out/'pdf'/Path(item['destination']).name
        d.save(target, garbage=4, deflate=True)
        d.close()
        before, after = fitz.open(src), fitz.open(target)
        text = after[0].get_text()
        assert 'Harmony-direct' not in text and 'Harmony direct' not in text
        assert text.count('Harmony') == len(spans)
        a,b = before[0].get_pixmap(dpi=144),after[0].get_pixmap(dpi=144)
        pa=np.frombuffer(a.samples,dtype=np.uint8).reshape(a.height,a.width,a.n)
        pb=np.frombuffer(b.samples,dtype=np.uint8).reshape(b.height,b.width,b.n)
        changed=np.any(pa!=pb,axis=2)
        allowed=np.zeros(changed.shape,dtype=bool)
        for s in spans:
            x0,y0,x1,y1=s['bbox']
            allowed[max(0,int(y0*2)-4):int(y1*2)+5,max(0,int(x0*2)-4):int(x1*2)+5]=True
        assert not np.any(changed & ~allowed), ('Non-label visual change',item['number'])
        after[0].get_pixmap(dpi=150).save(out/'png'/f"{target.stem}.png")
        # Corresponding editable SVG: exact text substitution, anchors untouched.
        original = src
        candidates=[original.with_suffix('.svg'),original.parent.parent/(original.stem+'.svg')]
        svg = next((q for q in candidates if q.is_file()),None)
        assert svg, ('Missing source SVG',original)
        content=svg.read_text().replace('>Harmony-direct<','>Harmony<').replace('>Harmony direct<','>Harmony<')
        assert '>Harmony-direct<' not in content and '>Harmony direct<' not in content
        (out/'svg'/f'{target.stem}.svg').write_text(content)
        records.append(dict(number=item['number'],previous_source=str(src),previous_pdf=str(src),
                            previous_sha256=sha(src),source=str(target),sha256=sha(target),
                            destination=item['destination'],case=item['case'],labels_changed=len(spans),
                            pixels_outside_label_boxes_identical=True))
        print(item['number']+': labels updated; all plot pixels preserved.',flush=True)
    (out/'MANIFEST.json').write_text(json.dumps(records,indent=2)+'\n')

if __name__ == '__main__':
    if sys.argv[1] == 'figure3': figure3()
    elif sys.argv[1] == 'supplementary': supplementary()
