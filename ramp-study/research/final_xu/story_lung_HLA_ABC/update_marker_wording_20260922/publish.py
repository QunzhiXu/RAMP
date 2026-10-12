from pathlib import Path
import hashlib,json,shutil,sys,fitz

HERE=Path(__file__).resolve().parent
STAGE=HERE/'manuscript_stage'
TARGET=HERE.parents[2]/'bioRxiv_Ver1'
assert TARGET==Path('/Users/qxu6/Documents/New project/bioRxiv_Ver1')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

if '--prepare' in sys.argv:
    pdf=STAGE/'build/template.pdf';doc=fitz.open(pdf)
    log=(STAGE/'build/template.log').read_text()
    assert 'Overfull' not in log and 'undefined' not in log.lower()
    tex=(STAGE/'template.tex').read_text()
    assert 'vWF + CD31' in tex and 'CD45RA + CD45R0' in tex
    assert 'nine shared proteins.' not in tex[tex.index(r'\paperfig{figure6_lung_hla}'):tex.index(r'}{fig:lunghla}')]
    cap=(STAGE/'FIGURE6_CAPTION.md').read_text().split('```latex\n')[1].split('\n```')[0]
    assert cap in tex
    figure_pages={}
    for i,p in enumerate(doc):
        for n in range(1,8):
            if f'Figure {n}:' in p.get_text():figure_pages[str(n)]=i+1
    assert len(figure_pages)==7
    for p in (STAGE/'figure').iterdir():
        assert sha(p)==sha(TARGET/'figure'/p.name)
    for i in [33,38,40,49]:
        if i<len(doc):doc[i].get_pixmap(matrix=fitz.Matrix(1.2,1.2),alpha=False).save(HERE/'qa'/f'page_{i+1}.png')
    shutil.copy2(pdf,STAGE/'template.pdf')
    for name in ['INTEGRATION_CHECKS.json','LAYOUT_CHECKS.json']:
        d=json.loads((TARGET/name).read_text());d['updated']='2026-09-22 marker-channel wording'
        d['pages']=len(doc);d['figure_pages']=figure_pages
        if name=='LAYOUT_CHECKS.json':d.update(pages_before=49,pages_after=len(doc),full_page_review=[16,33,34,39,40,41,50])
        d.update(marker_channel_wording_corrected=True,all_figure_images_unchanged=True)
        (STAGE/name).write_text(json.dumps(d,indent=2)+'\n')
    p=STAGE/'README.md';p.write_text(p.read_text().replace('Current manuscript: 49 pages','Current manuscript: 50 pages'))
    names=['template.tex','template.pdf','FIGURE6_CAPTION.md','CROSS_PLATFORM_LEARNING_SECTION.md',
        'README.md','LUNG_MARKER_CHANNEL_UPDATE_20260922.md',
        'supplementary_data/lung_marker_channel_mapping.csv','INTEGRATION_CHECKS.json','LAYOUT_CHECKS.json']
    plan=[dict(file=n,new_sha256=sha(STAGE/n),old_sha256=sha(TARGET/n) if (TARGET/n).exists() else None) for n in names]
    (HERE/'PUBLISH_PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
    (HERE/'QA.json').write_text(json.dumps(dict(pages=len(doc),figure_pages=figure_pages,caption_matches_tex=True,
        all_figure_files_unchanged=True,no_overfull=True,no_undefined_references=True,
        changes='Methods channel correspondence, Figure 6 caption, complete marker mapping, data-reference wording'),indent=2)+'\n')
    print('Prepared',len(names),'files;',len(doc),'pages')
else:
    plan=json.loads((HERE/'PUBLISH_PLAN.json').read_text())
    backup=TARGET/'_backups/lung_marker_channels_20260922'
    assert not backup.exists()
    for item in plan:
        dst=TARGET/item['file']
        assert (sha(dst) if dst.exists() else None)==item['old_sha256']
        assert sha(STAGE/item['file'])==item['new_sha256']
    backup.mkdir(parents=True)
    for item in plan:
        rel=Path(item['file']);dst=TARGET/rel
        if dst.exists():
            b=backup/rel;b.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(dst,b)
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(STAGE/rel,dst)
        assert sha(dst)==item['new_sha256']
    shutil.copy2(HERE/'PUBLISH_PLAN.json',backup/'PUBLISH_PLAN.json')
    (HERE/'PUBLISHED.json').write_text(json.dumps(dict(destination=str(TARGET),files=len(plan),
        backup=str(backup),manuscript_sha256=sha(TARGET/'template.pdf')),indent=2)+'\n')
    print('Published',len(plan),'files; backups:',backup)
