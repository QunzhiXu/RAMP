"""Verify an icon-only figure revision and synchronize the two current manuscripts."""
from pathlib import Path
import datetime,hashlib,json,shutil,sys
import fitz
from PIL import Image,ImageChops

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'shared_cd4_h1_20260914'
DEST=HERE.parents[2]/'manuscript'
WORK=HERE/'manuscript_sync'
STAGE=WORK/'stage'
STEMS=['manuscript','manuscript_figure1_polished']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')

def stage():
    assert not (WORK/'original_hashes.json').exists()
    (STAGE/'figure').mkdir(parents=True,exist_ok=True)
    original={str(p):sha(p) for p in [*DEST.iterdir(),*(DEST/'figure').iterdir()] if p.is_file()}
    js(WORK/'original_hashes.json',original)
    source=Path('/Users/qxu6/Downloads/ChatGPT Image Sep 14, 2026, 06_45_46 PM.png')
    assert sha(source)==sha(HERE/'assets/ramp_full_figure1.png')
    assets=json.loads((HERE/'original_asset_hashes.json').read_text())
    assert all(sha(p)==h for p,h in assets.items())
    unchanged=[]
    for label in ['B','C','D','CD']:
        rel=Path(f'Panel {label}')/f'panel_{label.lower()}.png'
        a=Image.open(OLD/rel).convert('RGB');b=Image.open(HERE/rel).convert('RGB')
        assert a.size==b.size and ImageChops.difference(a,b).getbbox() is None,label
        unchanged.append(label)
    a=Image.open(OLD/'figure_1_shared_cd4.png').convert('RGB')
    b=Image.open(HERE/'figure_1_icon_updated.png').convert('RGB')
    assert a.size==b.size
    box=ImageChops.difference(a,b).getbbox()
    assert box[0]>=31 and box[1]>=254 and box[2]<=912 and box[3]<=697,box
    a.crop((20,230,935,735)).save(WORK/'icon_before.png')
    b.crop((20,230,935,735)).save(WORK/'icon_after.png')
    (HERE/'qa').mkdir(exist_ok=True)
    d=fitz.open(HERE/'figure_1_icon_updated.pdf')
    assert len(d)==1
    d[0].get_pixmap(dpi=180).save(HERE/'qa/pdf_preview.png')
    text_old=fitz.open(OLD/'figure_1_shared_cd4.pdf')[0].get_text()
    assert d[0].get_text()==text_old,'Vector labels changed'
    js(WORK/'figure_checks.json',{'status':'PASS','icon_asset_byte_identical_to_user_file':True,
        'global_icon_files_unchanged':True,'old_figure_files_unchanged':True,
        'unchanged_panels_pixel_identical':unchanged,'A2_pixel_identical':True,
        'changed_pixel_bounds':box,'only_A1_image_region_changed':True,
        'all_vector_labels_identical':True,'canvas_pixels':b.size})
    for p in (DEST/'figure').iterdir():
        if not p.is_file():continue
        target=STAGE/'figure'/p.name
        if p.name in ['figure1_ramp_overview.pdf','figure1_ramp_overview.png','figure1_ramp_overview.svg','figure1_ramp_overview_polished.pdf']:
            shutil.copy2(HERE/f'figure_1_icon_updated{p.suffix}',target)
        else:target.symlink_to(p)
    for stem in STEMS:shutil.copy2(DEST/f'{stem}.tex',STAGE/f'{stem}.tex')
    print('PASS: only A.1 image pixels changed; manuscript staged with unchanged TeX.')

def verify():
    qa={}
    for stem in STEMS:
        old=fitz.open(DEST/f'{stem}.pdf');new=fitz.open(STAGE/f'{stem}.pdf')
        assert len(old)==len(new)==18
        changed=[i+1 for i in range(18) if old[i].get_pixmap().samples!=new[i].get_pixmap().samples]
        assert changed==[2],changed
        assert sha(DEST/f'{stem}.tex')==sha(STAGE/f'{stem}.tex')
        log=(STAGE/f'{stem}.log').read_text()
        assert 'Overfull' not in log and 'undefined references' not in log
        if stem=='manuscript':new[1].get_pixmap(matrix=fitz.Matrix(1.6,1.6),alpha=False).save(WORK/'figure1_page.png')
        qa[stem]={'pages':18,'changed_pages':changed,'TeX_byte_identical':True,'other_pages_pixel_identical':True,'no_overflow_or_undefined_references':True}
    qa['figure_checks']=json.loads((WORK/'figure_checks.json').read_text())
    js(WORK/'QA.json',qa)
    print(json.dumps({s:qa[s] for s in STEMS},indent=2))

def install():
    original=json.loads((WORK/'original_hashes.json').read_text())
    assert all(sha(p)==h for p,h in original.items()),'Manuscript changed since staging'
    qa=json.loads((WORK/'QA.json').read_text())
    assert (WORK/'VISUAL_QA.md').read_text().startswith('PASS\n')
    files=[(STAGE/f'{stem}.pdf',DEST/f'{stem}.pdf') for stem in STEMS]
    files += [(STAGE/'figure'/name,DEST/'figure'/name) for name in ['figure1_ramp_overview.pdf','figure1_ramp_overview.png','figure1_ramp_overview.svg','figure1_ramp_overview_polished.pdf']]
    files += [(WORK/'figure1_page.png',DEST/name) for name in ['figure1_latest_page.png','figure1_polished_page.png']]
    files += [(WORK/name,DEST/name) for name in ['FIGURE1_UPDATE.md','FIGURE1_POLISH_PREVIEW.md']]
    backup=DEST/'backups'/('figure1_icon_before_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    records=[]
    for source,target in files:
        saved=backup/target.relative_to(DEST);saved.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(target,saved);assert sha(saved)==sha(target)
        records.append({'target':str(target),'before':sha(saved),'after':sha(source)})
    manifest={'status':'backed_up','backup':str(backup),'files':records}
    js(backup/'MANIFEST.json',manifest)
    for source,target in files:
        tmp=target.with_name(target.name+'.icon_update_tmp')
        shutil.copy2(source,tmp);assert sha(tmp)==sha(source)
        tmp.replace(target);assert sha(target)==sha(source)
    changed={str(t) for _,t in files}
    assert all(sha(p)==h for p,h in original.items() if p not in changed)
    assert all(sha(p)==h for p,h in json.loads((HERE/'original_asset_hashes.json').read_text()).items())
    manifest.update(status='PASS',qa=qa)
    js(backup/'MANIFEST.json',manifest);js(WORK/'INSTALL_VERIFY.json',manifest)
    print(json.dumps({'status':'PASS','backup':str(backup),'files_installed':len(files)},indent=2))

if __name__=='__main__':{'stage':stage,'verify':verify,'install':install}[sys.argv[1]]()
