#!/usr/bin/env python3
"""Replace only the A icon geometry in the preceding layout trial."""
from pathlib import Path
import hashlib,json
import fitz
import numpy as np
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'horizontal_E_spatial_left'
manifest=json.loads((OLD/'LAYOUT_AUDIT.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
doc=fitz.open();page=doc.new_page(width=16.99*72,height=6.98*72)
slots={}
for key,row in manifest['placements'].items():
    source=HERE/'panel_A/panel_A_niche_discovery.pdf' if key=='A' else Path(row['source'])
    before=sha(source)
    rect=fitz.Rect(*[v*72 for v in row['bounds_inches']])
    with fitz.open(source) as src:
        np.testing.assert_allclose([rect.width,rect.height],[src[0].rect.width,src[0].rect.height],atol=.01)
        page.show_pdf_page(rect,src,0)
    assert sha(source)==before
    slots[key]={'source':str(source),'sha256':before,'bounds_inches':row['bounds_inches']}
doc.set_metadata({'title':'HNSCC layout with RAMP pipes meeting both frame borders'})
doc.save(HERE/'hnscc_fitted_ramp_trial.pdf',deflate=True,garbage=4)
page.get_pixmap(dpi=160,alpha=False).save(HERE/'hnscc_fitted_ramp_trial.png')
doc.close()
qa=HERE/'qa';qa.mkdir(exist_ok=True)
with fitz.open(HERE/'panel_A/panel_A_niche_discovery.pdf') as src:
    src[0].get_pixmap(dpi=200,alpha=False).save(qa/'panel_A_fitted_icon.png')
(HERE/'LAYOUT_AUDIT.json').write_text(json.dumps({'status':'PASS','changes':'A logo width 1.36 -> 1.47 inches; x=2.78 to 4.25','source_layout':str(OLD),'placements':slots,'scientific_results_changed':False,'manuscript_replaced':False},indent=2)+'\n')
(HERE/'README_ZH.md').write_text('# RAMP 两侧水管贴框的试稿\n\nPanel A 使用同一原图，保持比例，宽度由 1.36 增至 1.47 英寸；左端 x=2.78，右端 x=4.25，正好对应两侧框边。\n\n整图位于 hnscc_fitted_ramp_trial.png/.pdf；这里只替换 A 的图标大小，其余 panel 和科学结果与上版一致。扩展 program 的独立比较保存在 figure_HNSCC/program_return_exploration_20260913/，未将任何新 program 写入主图。旧版、正式图和 manuscript 均保留。\n')
print('PASS: A fitted to frame edges, whole trial composed.')
