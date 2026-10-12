from pathlib import Path
import fitz, json
JOB=Path(__file__).resolve().parent
results={}
for stem in ['manuscript','manuscript_figure1_polished']:
    old=fitz.open(JOB/'backup'/f'{stem}.pdf')
    new=fitz.open(JOB/'stage'/f'{stem}.pdf')
    assert len(old)==len(new)==18
    changed=[i+1 for i in range(len(new)) if old[i].get_pixmap(matrix=fitz.Matrix(1,1)).samples!=new[i].get_pixmap(matrix=fitz.Matrix(1,1)).samples]
    assert changed==[6], changed
    assert (JOB/'backup'/f'{stem}.tex').read_bytes()==(JOB/'stage'/f'{stem}.tex').read_bytes()
    log=(JOB/'stage'/f'{stem}.log').read_text()
    assert 'Overfull' not in log
    for page in [6,7]:
        new[page-1].get_pixmap(matrix=fitz.Matrix(2.5,2.5),alpha=False).save(JOB/'qa'/f'{stem}_page_{page}.png')
    results[stem]={'pages':len(new),'rendered_pages':[6,7],'changed_render_pages':changed,'manuscript_tex_unchanged':True,'no_overfull_boxes':True}
(JOB/'qa/PDF_CHECK.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
