from pathlib import Path
import json,hashlib,math,shutil
import numpy as np
import pandas as pd
import fitz
from PIL import Image,ImageDraw,ImageFont

HERE=Path(__file__).resolve().parent
plan=json.loads((HERE/'PLAN.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
MM=72/25.4
METHODS=['Raw','Harmony-direct','cyCombine','CytoNorm','fastMNN','RAMP']
out=HERE/'output';(out/'pdf').mkdir(parents=True,exist_ok=True);(out/'png').mkdir(exist_ok=True)
compiled=fitz.open();reports=[];summaries=[];previews=[]
fontpath='/System/Library/Fonts/Supplemental/Arial Bold.ttf'
font=ImageFont.truetype(fontpath,23);small=ImageFont.truetype(fontpath,21)
for item in plan:
    folder=Path(item['output']);original=Path(item['source'])
    old=pd.read_csv(original/'ridge_display_audit.csv');new=pd.read_csv(folder/'ridge_display_audit.csv')
    keys=['method','marker','dataset','unit_key']
    a=old.set_index(keys).sort_index();b=new.set_index(keys).sort_index()
    assert a.index.equals(b.index)
    for col in ['density_sha256','n_cells','histogram_count','bins','sigma_bins','bin_width','ridge_color','top_to_bottom_index']:
        pd.testing.assert_series_equal(a[col],b[col])
    pdf=next((folder/'output/pdf').glob('*.pdf'))
    doc=fitz.open(pdf);before=fitz.open(item['old_publication'])
    assert len(doc)==len(before)==1 and doc[0].rect==before[0].rect
    # Warm identical font extraction paths before a pixel-wise rendering comparison.
    before[0].get_text('dict');doc[0].get_text('dict')
    x=before[0].get_pixmap(matrix=fitz.Matrix(2,2),alpha=False)
    y=doc[0].get_pixmap(matrix=fitz.Matrix(2,2),alpha=False)
    aa=np.frombuffer(x.samples,np.uint8).reshape(x.height,x.width,x.n)
    bb=np.frombuffer(y.samples,np.uint8).reshape(y.height,y.width,y.n)
    delta=np.any(aa!=bb,axis=2);total=int(delta.sum())
    for method in item['targets']:
        i=METHODS.index(method);top=46.5+i*37.24
        # Includes axes frames and x ticks; labels, other rows and the UMAP block are excluded.
        r=fitz.Rect(97*MM,(top+3.3)*MM,203.5*MM,(top+37.0)*MM)
        delta[math.floor(r.y0*2):math.ceil(r.y1*2),math.floor(r.x0*2):math.ceil(r.x1*2)]=False
    assert not delta.any(),(item['number'],int(delta.sum()))
    assert 'Raw (assay-transformed)' not in doc[0].get_text()
    compiled.insert_pdf(doc)
    number=item['number'].lower()
    doc[0].get_pixmap(matrix=fitz.Matrix(2,2),alpha=False).save(out/'png'/f'{number}_full_preview.png')
    for (method,marker),g in new.groupby(['method','marker'],sort=False):
        if not g.display_window_changed.iloc[0]:continue
        oldg=old[old.method.eq(method)&old.marker.eq(marker)]
        left,right=int(g.n_below.sum()),int(g.n_above.sum());n=int(g.n_cells.sum())
        summaries.append(dict(figure=item['number'],case=item['case'],method=method.replace('-direct',''),marker=marker,
            old_left=float(oldg.display_low.iloc[0]),old_right=float(oldg.display_high.iloc[0]),
            new_left=float(g.display_low.iloc[0]),new_right=float(g.display_high.iloc[0]),
            n_cells=n,left_of_view=left,right_of_view=right,percent_in_view=100*(1-(left+right)/n),
            max_boundary_density_percent=100*float(g[['display_endpoint_density_left','display_endpoint_density_right']].to_numpy().max())))
    markers=list(new.marker.unique())
    for method in item['targets']:
        i=METHODS.index(method);top=46.5+i*37.24
        clip=fitz.Rect(97*MM,(top-.1)*MM,203.5*MM,(top+37.0)*MM)
        pix=doc[0].get_pixmap(matrix=fitz.Matrix(2.4,2.4),clip=clip,alpha=False)
        im=Image.frombytes('RGB',(pix.width,pix.height),pix.samples)
        canvas=Image.new('RGB',(im.width+32,im.height+75),'white');draw=ImageDraw.Draw(canvas)
        draw.text((16,5),item['number']+' | '+item['case'],font=font,fill='#263238')
        for j,marker in enumerate(markers):
            cx=16+((115.5+j*34.5)*MM-clip.x0)*2.4
            draw.text((cx,38),marker,font=small,fill='#263238',anchor='mt')
        canvas.paste(im,(16,72));previews.append((item['number'],method,canvas))
    reports.append(dict(figure=item['number'],density_hashes_verified=len(new),
        all_original_counts_and_bandwidths_identical=True,non_target_pixels_identical=True,
        changed_pixels=total,targets=item['targets'],source_pdf=item['old_publication'],preview_pdf=str(pdf),
        preview_only=True))

compiled.set_toc([[1,item['number']+' - '+item['case'],i+1] for i,item in enumerate(plan)])
compiled.save(out/'pdf/s1_s7_ridge_preview.pdf',garbage=3,deflate=True)
for name,entries in [('s1_s4_changed_rows',[r for r in previews if r[0] in ['S1','S2','S3','S4']]),
                     ('s5_s7_changed_rows',[r for r in previews if r[0] in ['S5','S6','S7']])]:
    width=max(im.width for _,_,im in entries);height=sum(im.height+12 for _,_,im in entries)
    sheet=Image.new('RGB',(width,height),'white');y=0
    for _,_,im in entries:sheet.paste(im,(0,y));y+=im.height+12
    sheet.save(out/'png'/f'{name}.png')
pd.DataFrame(summaries).to_csv(HERE/'DISPLAY_WINDOW_SUMMARY.csv',index=False)
(HERE/'QA.json').write_text(json.dumps(reports,indent=2)+'\n')
for path,digest in json.loads((HERE/'PROTECTED_HASHES.json').read_text()).items():assert sha(Path(path))==digest,path
(HERE/'README.md').write_text('''# S1-S7 compact ridge previews (2026-09-24)

Preview only; no manuscript, Overleaf ZIP, current-publication pointer or previous figure was changed.

The combined seven-page A4 preview is `output/pdf/s1_s7_ridge_preview.pdf`. Full-page PNGs and two contact sheets of changed rows are in `output/png/`. Individual vector PDFs and editable generated scripts are under S1-S7.

Only these rows change: S1 Harmony; S2 Harmony; S3 RAMP; S4 Harmony; S5/S6/S7 Harmony and RAMP. The same display rule is applied to all targeted panels: take the union of positions where any displayed sample's peak-normalized density is at least 0.5% of its own peak, then pad both sides by 4% of that span, capped at the full histogram grid. S3 RAMP SOX10 uses the already approved main-figure left limit of -1.5. All other rows retain their previous view.

All cells from selected units enter the original full-grid histograms. No values are trimmed, no density is refitted with a different bandwidth, and no artificial zeros are inserted at the viewport boundary. The window is based on actually displayed patients; the full-cohort histogram origin and original bin width remain unchanged. Sparse tails may fall outside the viewport. Exact ranges, outside-view cell counts and boundary densities are in `DISPLAY_WINDOW_SUMMARY.csv` and the per-figure `ridge_display_audit.csv` files.

`QA.json` verifies every density hash, sample count, bin width and color against the previous full-range files. Pixels outside the targeted axes and x ticks are identical, including all UMAPs. `PROTECTED_HASHES.json` records unchanged manuscript/publication files. `prepare.py` and `render_all.py` reproduce the isolated preview scripts; the original builders remain read-only.
''')
print(json.dumps({'pages':len(compiled),'changed_panels':len(summaries),'density_hashes_verified':sum(x['density_hashes_verified'] for x in reports),
                  'min_percent_in_view':min(x['percent_in_view'] for x in summaries),'manuscript_unchanged':True},indent=2))
