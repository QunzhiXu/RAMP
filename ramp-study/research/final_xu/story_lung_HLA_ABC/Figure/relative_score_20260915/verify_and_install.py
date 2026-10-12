"""Verify staged files, then install only after visual QA with hash-checked backups."""
from pathlib import Path
import datetime, hashlib, json, shutil, sys
import fitz

HERE=Path(__file__).resolve().parent
DEST=HERE.parents[3]/'manuscript'
STAGE=HERE/'stage'
STEMS=['manuscript','manuscript_figure1_polished']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def verify():
    qa={}
    for stem in STEMS:
        old=fitz.open(DEST/f'{stem}.pdf');new=fitz.open(STAGE/f'{stem}.pdf')
        assert len(old)==len(new)==18,(stem,len(old),len(new))
        changed=[i+1 for i in range(18) if old[i].get_pixmap().samples!=new[i].get_pixmap().samples]
        assert changed==[9,18],changed
        log=(STAGE/f'{stem}.log').read_text()
        assert 'Overfull' not in log and 'undefined references' not in log
        qa[stem]={'pages':18,'changed_pages':changed,'other_pages_pixel_identical':True,'no_overfull_or_undefined_references':True}
        if stem=='manuscript':
            for page in changed:
                new[page-1].get_pixmap(matrix=fitz.Matrix(1.6,1.6),alpha=False).save(HERE/f'page{page}.png')
        assert 'relative ranks' not in new[8].get_text()
        assert 'relative scores' in new[8].get_text()
        assert 'Lung vCAF HLA-ABC inference' in new[17].get_text()
    qa['figure_checks']=json.loads((HERE/'figure_checks.json').read_text())
    for p in (DEST/'figure').iterdir():
        if p.is_file() and p.name not in {'figure5_lung_hla.png','figure5_lung_hla.svg'}:
            assert sha(p)==sha(STAGE/'figure'/p.name)
    qa['other_figure_assets_unchanged']=True
    (HERE/'QA.json').write_text(json.dumps(qa,indent=2)+'\n')
    print(json.dumps(qa,indent=2))

def install():
    original=json.loads((HERE/'original_hashes.json').read_text())
    assert all(sha(p)==h for p,h in original.items()),'Live manuscript changed since staging'
    qa=json.loads((HERE/'QA.json').read_text())
    assert qa['other_figure_assets_unchanged']
    assert (HERE/'VISUAL_QA.md').read_text().startswith('PASS\n')
    files=[Path(f'{stem}.{ext}') for stem in STEMS for ext in ['tex','pdf']]
    files += [Path(f'figure/figure5_lung_hla.{ext}') for ext in ['png','svg']]
    backup=DEST/'backups'/('lung_relative_score_before_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    records=[]
    for rel in files:
        target=DEST/rel;saved=backup/rel
        saved.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(target,saved);assert sha(target)==sha(saved)
        records.append({'file':str(rel),'before':sha(saved),'after':sha(STAGE/rel)})
    manifest={'status':'backed_up','backup':str(backup),'files':records}
    (backup/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for rel in files:
        tmp=(DEST/rel).with_name((DEST/rel).name+'.relative_score_tmp')
        shutil.copy2(STAGE/rel,tmp);assert sha(tmp)==sha(STAGE/rel)
        tmp.replace(DEST/rel)
        assert sha(DEST/rel)==sha(STAGE/rel)
    changed={str(DEST/p) for p in files}
    assert all(sha(p)==h for p,h in original.items() if p not in changed)
    manifest.update(status='PASS',qa=qa)
    (backup/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (HERE/'INSTALL_VERIFY.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'status':'PASS','backup':str(backup),'files_installed':len(files)},indent=2))

if __name__=='__main__':
    {'verify':verify,'install':install}[sys.argv[1]]()
