"""Read-only inspection of 20 fixed figure assets, with small rendered previews.

No historical script is imported or executed, and no figure is regenerated.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import stat

HERE=Path(__file__).resolve().parent
STUDY=HERE.parents[1]
ROOT=STUDY.parent
INDEX=STUDY/'inventory_20261010/evidence/publication_assets.json'
INDEX_SHA='3da152d0db83b757c422819b6462ac7d7c7f0987909c2f3e6d69cb04cc2d7771'
PAPER=ROOT/'ramp_paper/RAMP_bioRxiv_2026.09.25.754510v1.pdf'
PAPER_SHA='522e51e422f3e8dee8ce6c4e7ae020902d5cfa7fb0bf703ec162e4910ca709b3'
MAIN_PAGES=[6,8,11,12,14,16,18]


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def rec(p):
    p=Path(p)
    if not stat.S_ISREG(p.lstat().st_mode):raise ValueError('Not a regular source: '+str(p))
    return {'bytes':p.stat().st_size,'sha256':sha(p)}


def write(p,value):
    with Path(p).open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')


def read(p):return json.loads(Path(p).read_text())


def source_index():
    if sha(INDEX)!=INDEX_SHA or sha(PAPER)!=PAPER_SHA:raise ValueError('Unrecognized figure or paper anchor')
    items=read(INDEX)
    expected={f'Figure {i}' for i in range(1,8)}|{f'S{i}' for i in range(1,14)}
    if len(items)!=20 or {v['label'] for v in items}!=expected:raise ValueError('Figure set differs')
    for item in items:
        for path in [item['local_manuscript_asset'],*item['byte_identical_local_candidates']]:
            if rec(ROOT/path)!={'bytes':item['bytes'],'sha256':item['sha256']}:raise ValueError('Asset changed: '+path)
    return items


def prior():
    result={}
    for p in sorted(STUDY.rglob('*')):
        rel=p.relative_to(STUDY)
        if p.is_file() and not p.is_symlink() and not p.is_relative_to(HERE) and not any(
                x in ('runs','.venv','__pycache__','.cache','.git') or x.startswith('.') for x in rel.parts):
            if p.stat().st_size<=20_000_000:result[str(rel)]=rec(p)
    return result


def capture(output):
    import fitz
    from PIL import Image
    items=source_index()
    out=Path(output).absolute()
    if out.is_symlink() or any(p.is_symlink() for p in out.parents) or not out.is_relative_to(HERE/'runs'):
        raise ValueError('Output must be a new physical directory in this module runs/')
    if shutil.disk_usage(HERE).free<20_000_000_000:raise ValueError('Insufficient space')
    before=prior()
    out.mkdir(exist_ok=False)
    try:
        write(out/'REQUEST.json',{'controller':rec(Path(__file__)),'index':rec(INDEX),'published_paper':rec(PAPER),
                                 'scope':'source equality, actual-asset rendering and text inspection; no fitting or regeneration'})
        write(out/'PRIOR.json',before)
        (out/'previews').mkdir();records=[]
        with fitz.open(PAPER) as paper:
            write(out/'paper_text.json',{str(i+1):p.get_text() for i,p in enumerate(paper)})
            if len(paper)!=61:raise ValueError('Published page count changed')
            for item in items:
                label=item['label'];name=label.replace(' ','_');path=ROOT/item['local_manuscript_asset']
                record={**item,'preview':f'previews/{name}.png','text':'','page_count':1,'text_outside_page':[]}
                if path.suffix=='.pdf':
                    with fitz.open(path) as pdf:
                        if len(pdf)!=1:raise ValueError('Unexpected multipage figure: '+label)
                        page=pdf[0];scale=2200/max(page.rect.width,page.rect.height)
                        page.get_pixmap(matrix=fitz.Matrix(scale,scale),alpha=False).save(out/record['preview'])
                        record['size_points']=list(page.rect);record['text']=page.get_text()
                        spans=[s for block in page.get_text('dict')['blocks'] if 'lines' in block for line in block['lines'] for s in line['spans']]
                        record['font_sizes_points']=sorted({round(s['size'],3) for s in spans})
                        for s in spans:
                            if s['text'].strip() and not (page.rect+(-.5,-.5,.5,.5)).contains(fitz.Rect(s['bbox'])):
                                record['text_outside_page'].append({'text':s['text'],'bbox':s['bbox']})
                else:
                    with Image.open(path) as image:
                        record['size_pixels']=list(image.size);image.thumbnail((2200,2200));image.convert('RGB').save(out/record['preview'])
                number=int(label.split()[-1]) if label.startswith('Figure') else int(label[1:])
                page_no=MAIN_PAGES[number-1] if label.startswith('Figure') else 48+number
                page=paper[page_no-1];scale=1600/max(page.rect.width,page.rect.height)
                preview=f'previews/published_{name}.png'
                page.get_pixmap(matrix=fitz.Matrix(scale,scale),alpha=False).save(out/preview)
                record.update(published_page=page_no,published_preview=preview)
                records.append(record)
                print('RENDERED '+label,flush=True)
        write(out/'assets.json',records)
        after=prior()
        if before!=after:raise ValueError('Previously indexed study files changed')
        source_index()
        write(out/'COMPLETE.json',{'status':'inspection_assets_ready_not_full_qa', 'assets_sha256':sha(out/'assets.json'),
                                  'prior_sha256':sha(out/'PRIOR.json'),'figures':len(records),'prior_files_unchanged':len(before)})
        return {'figures':len(records),'prior_files':len(before),'new_bytes':sum(p.stat().st_size for p in out.rglob('*') if p.is_file())}
    except BaseException as exc:
        write(out/'FAILED.json',{'error':repr(exc)});raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    print(json.dumps(capture(a.out),indent=2))
