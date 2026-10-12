"""Back up and install the already compiled and visually reviewed Figure 1."""
from pathlib import Path
import datetime
import hashlib
import json
import shutil

HERE = Path(__file__).resolve().parent
DEST = Path('/Users/qxu6/Documents/New project/manuscript')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    original=json.loads((HERE/'original_manuscript_hashes.json').read_text())
    assert all(sha(p)==s for p,s in original.items()), 'Source manuscript changed during staging'
    qa=json.loads((HERE/'QA.json').read_text())
    for stem in ['manuscript','manuscript_figure1_polished']:
        q=qa[stem]
        assert q['pages']==18 and q['figure1_page']==2 and q['figure1_caption_page']==3
        assert q['figure2_page']==4 and q['figure2_text_unchanged']
        assert q['visual_review_passed'] and q['no_overfull_or_undefined_reference']
        assert q['both_variants_visually_identical']
    pairs=[]
    for stem in ['manuscript','manuscript_figure1_polished']:
        for ext in ['tex','pdf']:
            pairs.append((HERE/'staged'/f'{stem}.{ext}',DEST/f'{stem}.{ext}'))
    for name in ['figure1_ramp_overview.pdf','figure1_ramp_overview.png',
                 'figure1_ramp_overview.svg','figure1_ramp_overview_polished.pdf']:
        pairs.append((HERE/'staged/figure'/name,DEST/'figure'/name))
    for name in ['figure1_latest_page.png','figure1_polished_page.png']:
        pairs.append((HERE/'figure1_page-02.png',DEST/name))
    for name in ['FIGURE1_UPDATE.md','FIGURE1_POLISH_PREVIEW.md']:
        pairs.append((HERE/name,DEST/name))
    assert all(source.is_file() for source,_ in pairs)
    timestamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    backup=DEST/'backups'/f'figure1_before_{timestamp}'
    backup.mkdir(parents=True,exist_ok=False)
    records=[]
    # Complete every backup before replacing any of the manuscript files.
    for source,target in pairs:
        previous=sha(target) if target.exists() else None
        if previous is not None:
            saved=backup/target.relative_to(DEST)
            saved.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(target,saved)
            assert sha(saved)==previous
        records.append(dict(source=str(source),file=str(target),old_sha256=previous,new_sha256=sha(source)))
    manifest=dict(status='backed_up',timestamp=timestamp,backup=str(backup),files=records,
                  pages=18,figure1_page=2,caption_page=3,source_figure_version=str(HERE.parent),qa=qa)
    (backup/'SYNC_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for source,target in pairs:
        shutil.copy2(source,target)
        assert sha(source)==sha(target)
    targets={str(target) for _,target in pairs}
    untouched={p:s for p,s in original.items() if p not in targets}
    assert all(sha(p)==s for p,s in untouched.items())
    manifest.update(status='PASS',untouched_original_files=len(untouched))
    (backup/'SYNC_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (HERE/'INSTALL_VERIFY.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',backup=str(backup),installed=len(pairs),
                         untouched_original_files=len(untouched),figure1_page=2),indent=2))


if __name__=='__main__':
    main()
