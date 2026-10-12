from pathlib import Path
from datetime import datetime, timezone
import json, hashlib, re, csv, shutil, math
import fitz
import numpy as np
from PIL import Image, ImageDraw

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;LIVE=ROOT.parent/'bioarxiv_Ver2'
stage=HERE/'stage';build=HERE/'build';backup=HERE/'backup_before_revision'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
qa=json.loads((HERE/'LABEL_QA.json').read_text())
log=(build/'main.log').read_text()
assert not re.search(r'Warning|Overfull|Float too large',log)
assert (build/'main.bbl').read_bytes()==(ROOT/'Main Figure/ridge_compact_20260924/build/main.bbl').read_bytes()
assert sha(stage/'main.tex')==sha(backup/'main.tex')
old=fitz.open(backup/'main.pdf');new=fitz.open(build/'main.pdf')
assert len(old)==len(new)==64
pages=[5,52,53,54,55,56,57,58]
review=[]
for i,(a,b) in enumerate(zip(old,new),1):
    before=a.get_text();after=b.get_text()
    assert before.replace('Raw (arcsinh)','Raw').replace('Raw (assay-transformed)','Raw')==after,i
    assert a.rect==b.rect
    if i in pages:
        spans=[s for block in a.get_text('dict')['blocks'] for line in block.get('lines',[])
               for s in line['spans'] if s['text'] in ['Raw (arcsinh)','Raw (assay-transformed)']]
        assert len(spans)==1
        box=fitz.Rect(spans[0]['bbox']);box+=(-1,-1,1,1)
        # Warm the same font/text extraction path in both documents before raster QA.
        b.get_text('dict')
        x=a.get_pixmap(matrix=fitz.Matrix(1.5,1.5),alpha=False)
        y=b.get_pixmap(matrix=fitz.Matrix(1.5,1.5),alpha=False)
        xx=np.frombuffer(x.samples,np.uint8).reshape(x.height,x.width,x.n)
        yy=np.frombuffer(y.samples,np.uint8).reshape(y.height,y.width,y.n)
        delta=np.any(xx!=yy,axis=2)
        delta[math.floor(box.y0*1.5):math.ceil(box.y1*1.5),math.floor(box.x0*1.5):math.ceil(box.x1*1.5)]=False
        assert not delta.any(),i
        out=HERE/'renders'/f'manuscript_page_{i:02d}.png';out.parent.mkdir(exist_ok=True);y.save(out)
        review.append(dict(page=i,non_label_pixels_identical=True))
    for block in b.get_text('dict')['blocks']:
        for line in block.get('lines',[]):
            for s in line['spans']:
                x0,y0,x1,y1=s['bbox']
                assert min(x0,y0)>=-1 and x1<=b.rect.width+1 and y1<=b.rect.height+1,(i,s['text'])

# Compact visual review of the changed supplementary titles and plot starts.
contact=Image.new('RGB',(1280,4*220),'white');draw=ImageDraw.Draw(contact)
for n,item in enumerate(qa[1:]):
    d=fitz.open(item['destination'])
    pix=d[0].get_pixmap(matrix=fitz.Matrix(2,2),clip=fitz.Rect(270,116,575,211),alpha=False)
    im=Image.frombytes('RGB',(pix.width,pix.height),pix.samples)
    x=(n%2)*640;y=(n//2)*220
    draw.text((x+10,y+3),item['number'],fill='black');contact.paste(im,(x+10,y+22))
contact.save(HERE/'renders/supplementary_raw_labels.png')

prov=json.loads((backup/'PROVENANCE.json').read_text())
rev=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),change='Figure 1 and S1-S7 ridge labels unified to Raw; exact transformations remain in unchanged captions and Methods.',
         backup=str(backup),manuscript_tex_unchanged=True,non_label_pixels_identical=True,
         labels_changed=8,figure1_sox10_left_limit=-1.5,overleaf_zip_updated=False,
         compiled_pdf_sha256=sha(build/'main.pdf'))
prov['figure1'].update(previous_source=prov['figure1']['source'],previous_sha256=prov['figure1']['sha256'],
                      source=qa[0]['destination'],sha256=qa[0]['sha256'])
for item in qa[1:]:
    r=next(p for p in prov['supplementary'] if p['number']==item['number'])
    r.update(previous_source=r['source'],previous_sha256=r['sha256'],source=item['destination'],sha256=item['sha256'],
             change='Ridge row label unified to Raw; all other pixels, data transformations and display ranges unchanged.')
prov['raw_labels_revision_20260924']=rev
(stage/'PROVENANCE.json').write_text(json.dumps(prov,indent=2,ensure_ascii=False)+'\n')
check=json.loads((backup/'COMPILATION_CHECK.json').read_text())
check.update(revision='Raw label harmonization, Figure 1 and S1-S7, 2026-09-24',
             changed_figure_sources=[p['manuscript_destination'] for p in qa],
             compiled_pdf_sha256=sha(build/'main.pdf'),main_tex_unchanged=True,
             manuscript_body_methods_and_captions_unchanged=True,text_changes_only=[],
             visual_review_detail_pages=[5],visual_review_contact_pages=list(range(52,59)),
             visual_review_findings='Eight Raw labels unified. Every pixel outside each original label rectangle is identical in the standalone figures and compiled manuscript. Main Raw is recentered vertically. All page text otherwise identical and within page bounds.',
             supplementary_figure_pages_pixel_identical=False,
             supplementary_figure_pages_pixels_identical_except_raw_label=True,
             backup=str(backup),all_source_data_files_unchanged=True,overleaf_zip_updated=False)
(stage/'COMPILATION_CHECK.json').write_text(json.dumps(check,indent=2)+'\n')
readme=(backup/'README.md').read_text()
readme+='\n\n## Raw 图内命名统一（2026-09-24）\n\nFigure 1 和 S1–S7 的 ridge 行名已统一为 **Raw**。原来的 arcsinh 或 assay-transformed 括号文字已移除，caption 和 Methods 中的具体变换说明完整保留。仅图内行名改变，数据、曲线、坐标范围和其余面板不变；共 64 页。Overleaf ZIP 未更新。\n\n图源、可重复执行的文字修改脚本及备份：\n`'+str(HERE)+'`\n'
(stage/'README.md').write_text(readme)
replacements={p['manuscript_destination']:Path(p['destination']) for p in qa}
replacements['main.pdf']=build/'main.pdf'
for name in ['README.md','COMPILATION_CHECK.json','PROVENANCE.json']:replacements[name]=stage/name
with (backup/'FILE_MANIFEST.csv').open(newline='') as f:rows=list(csv.DictReader(f))
for row in rows:
    p=replacements.get(row['relative_path'],LIVE/row['relative_path'])
    row.update(bytes=p.stat().st_size,sha256=sha(p))
with (stage/'FILE_MANIFEST.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['relative_path','bytes','sha256']);w.writeheader();w.writerows(rows)
replacements['FILE_MANIFEST.csv']=stage/'FILE_MANIFEST.csv'

local={}
mainreadme=(backup/'local_docs/Main Figure/README.md').read_text()
mainreadme=mainreadme.replace('ridge_compact_20260924/figure_1_melanoma.png','../figure_raw_labels_20260924/output/png/figure_1_melanoma.png')
mainreadme=mainreadme.replace('ridge_compact_20260924/figure_1_melanoma.pdf','../figure_raw_labels_20260924/output/pdf/figure_1_melanoma.pdf')
mainreadme=mainreadme.replace('[重画入口与记录](ridge_compact_20260924/README.md)','[重画入口与记录](../figure_raw_labels_20260924/README.md)')
mainreadme+='\n2026-09-24：图内 Raw 行名已统一，具体显示变换仍见 caption；密度范围沿用上述紧凑版本。最新发布图经过 `../figure_raw_labels_20260924/update_labels.py` 仅修改文字，原绘图脚本及输出保留。\n'
(HERE/'MAIN_README_NEXT.md').write_text(mainreadme)
local['Main Figure/README.md']=str(HERE/'MAIN_README_NEXT.md')
supreadme=(backup/'local_docs/Supplementary Figures/README.md').read_text()
supreadme=supreadme.replace('## 当前发布显示版本（2026-09-22）','## 当前发布显示版本（2026-09-24）')
supreadme=supreadme.replace('S1–S7 最新 PDF/PNG/SVG 位于 [ridge_full_range_update_20260922/published](ridge_full_range_update_20260922/published/)，已修复密度截断；',
    'S1–S7 最新 PDF/PNG 位于 [figure_raw_labels_20260924/output](../figure_raw_labels_20260924/output/)，图内行名统一为 **Raw**，保留此前完整密度范围；')
supreadme=supreadme.replace('当前版本均为各目录的 `ridge_full_range_20260922/`，入口见各自 `CURRENT_PREVIEW.json`。',
    '底层绘图脚本保存在各目录的 `ridge_full_range_20260922/`，其输出经文字修改脚本统一 Raw 名称；最新图入口见各自 `CURRENT_PREVIEW.json`。')
supreadme+='\nRaw 行名统一仅影响图内文字，caption 保留每个平台的具体变换。脚本、原图与旧指针备份见 `../figure_raw_labels_20260924/`；旧 SVG 和历史图保持原样。此次同步至 bioarxiv_Ver2，未更新历史 Ver1 或 Overleaf ZIP。\n'
(HERE/'SUP_README_NEXT.md').write_text(supreadme)
local['Supplementary Figures/README.md']=str(HERE/'SUP_README_NEXT.md')
(HERE/'CURRENT_PUBLICATION_NEXT.json').write_text(json.dumps(prov['supplementary'],indent=2)+'\n')
local['Supplementary Figures/CURRENT_PUBLICATION.json']=str(HERE/'CURRENT_PUBLICATION_NEXT.json')
for item in qa[1:]:
    r=next(p for p in prov['supplementary'] if p['number']==item['number'])
    rel='Supplementary Figures/'+r['case']+'/CURRENT_PREVIEW.json'
    path=ROOT/rel
    out=backup/'local_docs'/rel;out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,out)
    d=json.loads(path.read_text())
    d.update(previous_folder=d['folder'],version='raw_labels_20260924',folder=str(HERE),
             pdf=item['destination'],png=str(HERE/'output/png'/(Path(item['destination']).stem+'.png')),
             revision='Raw label only; full density range and all other pixels unchanged.')
    out=HERE/(item['number']+'_CURRENT_PREVIEW_NEXT.json');out.write_text(json.dumps(d,indent=2)+'\n')
    local[rel]=str(out)
(HERE/'INSTALL_PLAN.json').write_text(json.dumps({'manuscript':{r:str(p) for r,p in replacements.items()},'local':local},indent=2)+'\n')
(HERE/'README.md').write_text('''# Raw display-label revision — 2026-09-24

Current Figure 1 and supplementary S1–S7 publication PDFs and PNGs are under `output/`. Only their ridge-row labels change from `Raw (arcsinh)` or `Raw (assay-transformed)` to `Raw`. The vertically rotated main-figure label remains centered. Fonts, curves, data transformations, limits and UMAPs are unchanged. Figure 1 retains the accepted RAMP SOX10 left limit of −1.5.

`SOURCE_MANIFEST.json` records original inputs. `update_labels.py` edits exactly one text-showing operation in each source PDF (and the main label's centering), without recreating the scientific plots. `LABEL_QA.json` verifies that every other pixel is identical. Use the original renderers to recreate the source PDFs, then apply `edit_label(source, destination, old_label)` from this script to reproduce the publication naming. Original sources remain unchanged.

The main TeX, captions, Methods, numeric results and Overleaf ZIP remain unchanged. `stage/` and `build/` hold the manuscript build. `backup_before_revision/` holds the prior manuscript and eight figure PDFs. `MANUSCRIPT_QA.json` records checks of the compiled 64-page result.
''')
(HERE/'MANUSCRIPT_QA.json').write_text(json.dumps({'pages':64,'no_compile_warnings':True,'tex_unchanged':True,'bibliography_unchanged':True,'changed_pages':review,'other_page_text_unchanged':True},indent=2)+'\n')
print(json.dumps({'pages':len(new),'labels_updated':8,'ready_to_deploy':True},indent=2))
