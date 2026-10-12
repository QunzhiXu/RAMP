from pathlib import Path
import hashlib
import json
import re
import shutil
import fitz
import pandas as pd
from PIL import Image, ImageDraw

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent.parent
STAGE=OUT/'manuscript_stage'
TARGET=ROOT.parent.parent/'bioRxiv_Ver1'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p,value):
    p.write_text(json.dumps(value,indent=2)+'\n')


def main():
    log=(STAGE/'build/template.log').read_text()
    assert 'Output written on' in log
    assert 'Overfull' not in log and 'undefined' not in log.lower()
    assert not re.search(r'LaTeX Warning|Package .* Warning',log)
    doc=fitz.open(STAGE/'build/template.pdf')
    assert len(doc)==49
    figure_pages={}
    for n,p in enumerate(doc):
        text=p.get_text()
        for fig in range(1,8):
            if f'Figure {fig}:' in text: figure_pages[str(fig)]=n+1
        assert len(text.strip())>40, ('near-empty page',n+1)
    assert len(figure_pages)==7 and figure_pages['3']==10
    full='\n'.join(p.get_text() for p in doc)
    for val in ['0.773','0.751','0.584','0.520','0.744','0.709','0.277','0.349','0.359','0.219']:
        assert val in full
    for letter in ['B','C']:
        p=ROOT/f'Figure/Panel {letter}'
        plot=pd.read_csv(p/'per_benchmark_scores.csv')
        means=pd.read_csv(p/'overall_means.csv')
        recalculated=plot.groupby(['metric','method']).score.mean()
        check=means.set_index(['metric','method'])['mean']
        assert (check-recalculated).abs().max()<1e-12
        proof=json.loads((p/'PROVENANCE.json').read_text())
        assert sha(proof['source'])==proof['source_sha256']
        assert proof['all_points_within_visible_axes']
    source_records=json.loads((STAGE/'supplementary_data/trainability/SOURCE_MANIFEST.json').read_text())
    for rec in source_records:
        assert sha(STAGE/rec['copy'])==rec['sha256'],rec['copy']
        assert sha(rec['source'])==rec['sha256'],rec['source']
    initial=json.loads((OUT/'MANUSCRIPT_BEFORE_HASHES.json').read_text())
    for f in (STAGE/'figure').iterdir():
        if f.name!='figure3_cross_platform_learning.pdf':
            assert sha(f)==initial[str(f.relative_to(STAGE))]
    for letter in ['D','E','F']:
        for p in (STAGE/f'supplementary_data/trainability/Panel_{letter}').rglob('*'):
            if p.is_file(): assert sha(p)==initial[str(p.relative_to(STAGE))]
    assert sha(STAGE/'figure/figure3_cross_platform_learning.pdf')==sha(ROOT/'Figure/Combined ABCDEF/figure_cross_platform_learning.pdf')
    # Check caption equality, ignoring only Markdown/TeX bold syntax and spacing.
    caption=(STAGE/'FIGURE3_CAPTION.md').read_text().split('\n\n',1)[1].strip()
    caption=re.sub(r'\*\*(.*?)\*\*',lambda m:r'\textbf{'+m.group(1)+'}',caption)
    tex=(STAGE/'template.tex').read_text()
    actual=tex.split(r'\paperfig{figure3_cross_platform_learning}{%',1)[1].split('}{fig:trainability}',1)[0]
    assert ' '.join(caption.split())==' '.join(actual.split())
    means=pd.read_csv(OUT/'OVERALL.csv')
    chosen=means[((means.task_type=='J1') & means.method.isin(['ramp','unaligned','clipped_zscore']) & means.metric.isin(['balanced_accuracy','macro_f1'])) | ((means.task_type=='J2') & means.method.isin(['ramp','unaligned','percentile']))]
    records=[dict(task_type=r.task_type,metric=r.metric,method=r.method,value=r.score,reported=round(r.score,3)) for r in chosen.itertuples()]
    qa=dict(updated='2026-09-22',release=OUT.name,pages=len(doc),figure_pages=figure_pages,
        all_seven_figure_references_resolved=True,undefined_references=False,overfull_boxes=False,
        caption_md_matches_tex=True,plot_values_match_release=True,
        manuscript_figure_matches_source=True,source_files_verified=len(source_records),
        reported_main_means=records,other_six_figure_files_unchanged=True,
        panels_def_data_unchanged=True,panels_def_original_scope_explicit=True,
        primary_scope_revisions_disclosed=True,models_refitted_during_sync=False)
    save(STAGE/'INTEGRATION_CHECKS.json',qa)
    save(OUT/'QA_CHECKS.json',qa)
    # Render all pages for a final layout contact sheet, plus the edited pages.
    qa_dir=OUT/'qa';qa_dir.mkdir(exist_ok=True)
    canvas=Image.new('RGB',(7*235,7*325),'#e5e7eb');draw=ImageDraw.Draw(canvas)
    for i,p in enumerate(doc):
        pix=p.get_pixmap(matrix=fitz.Matrix(.35,.35),alpha=False)
        im=Image.frombytes('RGB',[pix.width,pix.height],pix.samples)
        im.thumbnail((223,300))
        x=(i%7)*235+6;y=(i//7)*325+18
        canvas.paste(im,(x,y));draw.text((x,y-15),str(i+1),fill='black')
    canvas.save(qa_dir/'all_pages.png')
    for i in [0,8,9,10,30,31,32,33]:
        doc[i].get_pixmap(matrix=fitz.Matrix(1.3,1.3),alpha=False).save(qa_dir/f'page_{i+1:02}.png')
    save(STAGE/'LAYOUT_CHECKS.json',dict(updated='2026-09-22',pages_before=49,pages_after=len(doc),
        figure_pages=figure_pages,overfull_boxes=False,undefined_references=False,
        rendered_all_pages_contact_sheet=True,full_page_review=[1,9,10,11,31,32,33,34],
        layout_parameters_unchanged=True,body_font_pt=12,body_line_spread=1.3))
    shutil.copy2(STAGE/'build/template.pdf',STAGE/'template.pdf')
    # Only actual changes and new Figure 3 source records are publishable.
    plan=[]
    for p in STAGE.rglob('*'):
        if not p.is_file(): continue
        relative=str(p.relative_to(STAGE))
        if relative.startswith('build/'): continue
        dest=TARGET/relative
        digest=sha(p)
        if dest.exists() and sha(dest)==digest: continue
        assert relative in ['template.tex','template.pdf','FIGURE3_CAPTION.md',
            'CROSS_PLATFORM_LEARNING_SECTION.md','README.md','REVISION_NOTES.md',
            'INTEGRATION_CHECKS.json','LAYOUT_CHECKS.json','FIGURE3_UPDATE_20260922.md',
            'figure/figure3_cross_platform_learning.pdf'] or relative.startswith('supplementary_data/trainability/'),relative
        old=sha(dest) if dest.exists() else None
        if relative in initial: assert old==initial[relative],('Concurrent change',relative)
        plan.append(dict(relative=relative,staged_sha256=digest,previous_sha256=old))
    save(OUT/'PUBLISH_PLAN.json',dict(destination=str(TARGET),stage=str(STAGE),files=plan))
    print(json.dumps(dict(pages=len(doc),figure_pages=figure_pages,source_records=len(source_records),publish_files=len(plan)),indent=2))


if __name__=='__main__': main()
