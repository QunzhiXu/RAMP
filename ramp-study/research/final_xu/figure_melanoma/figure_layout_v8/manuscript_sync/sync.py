"""Stage, verify, and back up the approved Figure 4 Panel F manuscript update."""
from pathlib import Path
import datetime,difflib,hashlib,json,shutil,sys
import fitz

HERE=Path(__file__).resolve().parent
FIGURE=HERE.parent
STORY=FIGURE.parent
DEST=STORY.parent.parent/'manuscript'
STAGE=HERE/'stage'
STEMS=['manuscript','manuscript_figure1_polished']
FILES=[Path(f'{stem}.{ext}') for stem in STEMS for ext in ['tex','pdf']]+[Path(f'figure/figure4_melanoma.{ext}') for ext in ['png','svg']]
CAPTION=r'''\textbf{(F) Cross-patient state enrichment.} Horizontal box plots show AUROC
(upper) and target enrichment (lower), with independent scales, for the same
14 evaluable IMC patients with at least 20 candidate CD8 cells and both classes.
Enrichment is the target fraction in the highest-scoring 20\% divided by the
target fraction among all candidate CD8 cells in that patient; selection
uses the whole patient image, as in E. Median AUROC is 0.707 and median
enrichment is 1.98-fold; all 14 patients exceed one-fold enrichment. Dashed
references are 0.5 and one-fold, respectively. Boxes show quartiles and medians,
whiskers extend to the most extreme observations within 1.5 interquartile
ranges, and dots show every patient. Standardization and classifier fitting
excluded the evaluated patient; RAMP alignment was frozen from the full cohort.
'''

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')

def stage():
    assert not (HERE/'original_hashes.json').exists(),'Already staged'
    STAGE.mkdir(parents=True)
    (STAGE/'figure').mkdir()
    paths=[DEST/f'{stem}.{ext}' for stem in STEMS for ext in ['tex','pdf']]
    paths += [p for p in (DEST/'figure').iterdir() if p.is_file()]
    js(HERE/'original_hashes.json',{str(p):sha(p) for p in paths})
    assert json.loads((FIGURE/'REVISION_QA.json').read_text())['status']=='PASS'
    for p in (DEST/'figure').iterdir():
        if not p.is_file():continue
        target=STAGE/'figure'/p.name
        if p.name in ['figure4_melanoma.png','figure4_melanoma.svg']:
            shutil.copy2(FIGURE/f'melanoma_three_row_redesign_v8{p.suffix}',target)
        else:target.symlink_to(p)
    for stem in STEMS:
        original=(DEST/f'{stem}.tex').read_text()
        first=original.index(r'\includegraphics[width=\textwidth]{figure4_melanoma.png}')
        start=original.index(r'\textbf{(F)',first)
        end=original.index(r'\textbf{(G)',start)
        revised=original[:start]+CAPTION+original[end:]
        assert revised[:start]==original[:start] and revised[start+len(CAPTION):]==original[end:]
        (STAGE/f'{stem}.tex').write_text(revised)
        (HERE/f'{stem}.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),revised.splitlines(True),fromfile=str(DEST/f'{stem}.tex'),tofile=str(STAGE/f'{stem}.tex'))))
    print('Staged Figure 4 PNG/SVG and Panel F caption in both manuscripts.')

def verify():
    qa={}
    for stem in STEMS:
        old=fitz.open(DEST/f'{stem}.pdf');new=fitz.open(STAGE/f'{stem}.pdf')
        assert len(old)==len(new)==18,(stem,len(old),len(new))
        changed=[]
        for i in range(18):
            if old[i].get_pixmap().samples!=new[i].get_pixmap().samples:changed.append(i+1)
        assert changed==[7,8],changed
        log=(STAGE/f'{stem}.log').read_text()
        assert 'Overfull' not in log and 'undefined references' not in log
        if stem=='manuscript':
            for page in changed:
                new[page-1].get_pixmap(matrix=fitz.Matrix(1.6,1.6),alpha=False).save(HERE/f'page{page}.png')
        assert 'Cross-patient state enrichment' in new[7].get_text()
        assert '1.98-fold' in new[7].get_text()
        qa[stem]={'pages':18,'changed_pages':changed,'other_pages_pixel_identical':True,'no_overfull_or_undefined_references':True}
    for p in (DEST/'figure').iterdir():
        if p.is_file() and p.name not in ['figure4_melanoma.png','figure4_melanoma.svg']:
            assert sha(p)==sha(STAGE/'figure'/p.name)
    qa['other_figure_assets_unchanged']=True
    qa['source_figure_checks']=json.loads((FIGURE/'REVISION_QA.json').read_text())
    qa['data_checks']=json.loads((FIGURE/'data/DATA_QA.json').read_text())
    js(HERE/'QA.json',qa)
    print(json.dumps({s:qa[s] for s in STEMS},indent=2))

def install():
    original=json.loads((HERE/'original_hashes.json').read_text())
    assert all(sha(p)==h for p,h in original.items()),'Manuscript changed during staging'
    qa=json.loads((HERE/'QA.json').read_text())
    assert qa['other_figure_assets_unchanged']
    assert (HERE/'VISUAL_QA.md').read_text().startswith('PASS\n')
    backup=DEST/'backups'/('melanoma_v8_before_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    records=[]
    for rel in FILES:
        saved=backup/rel;saved.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(DEST/rel,saved);assert sha(saved)==sha(DEST/rel)
        records.append({'file':str(rel),'before':sha(saved),'after':sha(STAGE/rel)})
    manifest={'status':'backed_up','backup':str(backup),'files':records}
    js(backup/'MANIFEST.json',manifest)
    for rel in FILES:
        tmp=(DEST/rel).with_name((DEST/rel).name+'.melanoma_v8_tmp')
        shutil.copy2(STAGE/rel,tmp);assert sha(tmp)==sha(STAGE/rel)
        tmp.replace(DEST/rel);assert sha(DEST/rel)==sha(STAGE/rel)
    changed={str(DEST/rel) for rel in FILES}
    assert all(sha(p)==h for p,h in original.items() if p not in changed)
    manifest.update(status='PASS',qa=qa)
    js(backup/'MANIFEST.json',manifest);js(HERE/'INSTALL_VERIFY.json',manifest)
    print(json.dumps({'status':'PASS','files_installed':len(FILES),'backup':str(backup)},indent=2))

if __name__=='__main__':{'stage':stage,'verify':verify,'install':install}[sys.argv[1]]()
