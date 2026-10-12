from pathlib import Path
import hashlib,json,re,zipfile,fitz
from PIL import Image,ImageDraw
HERE=Path(__file__).resolve().parent;LIVE=HERE.parent.parent/'bioarxiv_Ver2';BUILD=HERE/'zip_compile'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
log=(BUILD/'main.log').read_text()
assert not re.search(r'Warning|Overfull|Float too large',log)
aux=(BUILD/'main.aux').read_text()
labels={k:dict(number=n,page=int(p)) for k,n,p in re.findall(r'\\newlabel\{(fig:[^}]+)\}\{\{([^}]+)\}\{(\d+)\}',aux)}
prior=json.loads((LIVE/'COMPILATION_CHECK.json').read_text())
assert labels==prior['labels'];assert len(re.findall(r'\\bibcite\{',aux))==63
oldbbl=HERE.parent/'Supplementary Figures/ridge_compact_preview_20260924/manuscript_integration/build/main.bbl'
assert sha(BUILD/'main.bbl')==sha(oldbbl)
assert sha(BUILD/'main.tex')==sha(LIVE/'main.tex')
# All project-specific inputs must come from the extracted archive, not our working folders.
external=[]
for line in (BUILD/'main.fls').read_text().splitlines():
    if line.startswith('INPUT '):
        raw=line[6:];p=(BUILD/raw).resolve()
        if str(p).startswith('/Users/qxu6/Documents/New project/') and not p.is_relative_to(BUILD):external.append(str(p))
assert not external,external
canonical=fitz.open(LIVE/'main.pdf');rebuilt=fitz.open(BUILD/'main.pdf')
assert len(canonical)==len(rebuilt)==64
for i,(a,b) in enumerate(zip(canonical,rebuilt),1):
    assert a.rect==b.rect and a.get_text()==b.get_text(),i
    a.get_text('dict');b.get_text('dict')
    assert a.get_pixmap(alpha=False).samples==b.get_pixmap(alpha=False).samples,('pixel mismatch',i)
renders=HERE/'renders';renders.mkdir(exist_ok=True)
contact=Image.new('RGB',(1260,1780),'#eeeeee');draw=ImageDraw.Draw(contact)
for n,i in enumerate([5,10,48,52,54,58]):
    pix=rebuilt[i-1].get_pixmap(matrix=fitz.Matrix(1.3,1.3),alpha=False)
    im=Image.frombytes('RGB',(pix.width,pix.height),pix.samples);im.thumbnail((405,835))
    x=(n%3)*420+5;y=(n//3)*890+30;contact.paste(im,(x,y));draw.text((x,y-21),'Page '+str(i),fill='black')
    if i==48:pix.save(renders/'caption_page_48.png')
contact.save(renders/'package_compile_contact.png')
manifest=json.loads((HERE/'PACKAGE_MANIFEST.json').read_text())
with zipfile.ZipFile(HERE/'bioarxiv_Ver2_Overleaf.zip') as z:
    assert z.testzip() is None
    assert set(z.namelist())=={x['path'] for x in manifest}
    for item in manifest:assert hashlib.sha256(z.read(item['path'])).hexdigest()==item['sha256']
    assert not any(Path(n).suffix in ['.aux','.log','.bbl','.blg','.zip'] or n in ['main.pdf','ramp_harvard.bst'] for n in z.namelist())
qa=dict(status='passed',total_pages=64,main_figures=7,supplementary_figures=13,reference_count=63,numeric_citations=True,all_labels_unchanged=True,tex_and_captions_unchanged=True,all_64_pages_pixel_identical_to_canonical=True,no_compile_warnings=True,no_external_project_dependencies=True,compile_entry='main.tex',compiler='pdfLaTeX',bibliography_engine='BibTeX',archive_files=len(manifest),source_data_and_record_files=sum(x['path'].startswith('supplementary_data/') for x in manifest),archive_bytes=(HERE/'bioarxiv_Ver2_Overleaf.zip').stat().st_size,archive_sha256=sha(HERE/'bioarxiv_Ver2_Overleaf.zip'),canonical_main_pdf_sha256=sha(LIVE/'main.pdf'),zip_compile_pdf_sha256=sha(BUILD/'main.pdf'),canonical_main_tex_sha256=sha(LIVE/'main.tex'),visual_review_pages=[5,10,48,52,54,58],archive_excludes=['compiled main.pdf','old bibliography style','intermediate files','local backups'])
(HERE/'OVERLEAF_PACKAGE_CHECK.json').write_text(json.dumps(qa,indent=2)+'\n')
print(json.dumps(qa,indent=2))
