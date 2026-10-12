"""Check staged PDF layout and regression against the current manuscript."""
from pathlib import Path
import fitz, json, hashlib

HERE=Path(__file__).resolve().parent
DEST=Path('/Users/qxu6/Documents/New project/manuscript')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    qa={}
    for stem in ['manuscript','manuscript_figure1_polished']:
        doc=fitz.open(HERE/'staged'/f'{stem}.pdf')
        old=fitz.open(DEST/f'{stem}.pdf')
        assert len(doc)==len(old)==18
        identical=[];changed=[]
        # Render before text inspection, using the same sequence on both PDFs.
        for i in range(len(doc)):
            a=doc[i].get_pixmap(dpi=80,alpha=False)
            b=old[i].get_pixmap(dpi=80,alpha=False)
            (identical if a.samples==b.samples else changed).append(i+1)
        assert changed==[4,16,17], (stem,changed)
        doc.close();doc=fitz.open(HERE/'staged'/f'{stem}.pdf')
        txt=doc[3].get_text()
        assert all(s in txt for s in ['Louvain ARI','Figure 2:','0.835','not enter D.'])
        assert 'embedding-only fastMNN' not in txt
        log=(HERE/'staged'/f'{stem}.log').read_text()
        assert 'Overfull' not in log and 'undefined references' not in log
        for page in doc:
            for block in page.get_text('dict')['blocks']:
                if 'lines' not in block:continue
                x0,y0,x1,y1=block['bbox']
                assert x0>=-1 and y0>=-1 and x1<=page.rect.width+1 and y1<=page.rect.height+1
        for n in [16,17]:
            doc[n-1].get_pixmap(dpi=120,alpha=False).save(HERE/f'{stem}_changed_page{n}.png')
        qa[stem]={'pages':18,'figure2_page':4,'caption_on_same_page':True,
            'no_overfull_or_undefined_reference':True,'all_text_within_page_bounds':True,
            'pixel_identical_pages':identical,'changed_pages':changed}
        print(stem, 'changed pages:',changed)
    original=json.loads((HERE/'original_manuscript_hashes.json').read_text())
    assert all(sha(k)==v for k,v in original.items())
    qa['original_manuscript_unchanged_before_install']=True
    qa['other_figure_assets_unchanged']=all(sha(f)==sha(DEST/'figure'/f.name)
        for f in (HERE/'staged/figure').iterdir()
        if f.is_file() and not f.name.startswith('figure2_benchmark.'))
    assert qa['other_figure_assets_unchanged']
    qa['visual_review']={'figure2_page4':'PASS: full figure and caption fit at original width and font size.',
                         'changed_methods_pages':'Pending inspection'}
    (HERE/'QA.json').write_text(json.dumps(qa,indent=2)+'\n')

if __name__=='__main__':main()
