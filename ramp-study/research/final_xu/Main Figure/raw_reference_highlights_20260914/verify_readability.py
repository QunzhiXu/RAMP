"""Compare the display-only revision against its preserved predecessor."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from PIL import Image
import fitz
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'five_celltypes_readability_20260914'
qa=HERE/'qa';qa.mkdir(exist_ok=True)
report={}
for filename in ['artifact_box_statistics.csv','artifact_auc.csv','median_shift_box_statistics.csv','panel_d_display_overflow.csv']:
    before=pd.read_csv(OLD/filename,float_precision='round_trip')
    after=pd.read_csv(HERE/filename,float_precision='round_trip')
    pd.testing.assert_frame_equal(before,after,check_exact=True)
    report[filename]={'unchanged':True,'rows':len(after)}
for panel in ['A','B','D']:
    relative=Path(f'Panel {panel}')/f'panel_{panel.lower()}.png'
    a=np.asarray(Image.open(OLD/relative));b=np.asarray(Image.open(HERE/relative))
    assert np.array_equal(a,b),panel
    report[f'panel_{panel}_pixel_identical']=True
manifest=json.loads((HERE/'manifest.json').read_text())
for name,digest in {**manifest['all_source_hashes'],**manifest['prior_main_figure_sha256']}.items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
report['source_and_baseline_hashes_unchanged']=True
assert json.loads((qa/'text_collisions.json').read_text())==[]
report['text_collisions']=0
stats=pd.read_csv(HERE/'artifact_box_statistics.csv')
cd8=stats[(stats.marker=='CD8')&(stats.method=='Raw')&(stats.celltype=='CD8 T')].iloc[0]
report['raw_cd8_t_box']={k:float(cd8[k]) for k in ['q1','median','q3']}
report['common_y_display_limits']=[0,1.04]
report['raw_reference_highlights']=manifest['raw_reference_highlights']
report['reference_arrow_endpoint']='Raw Q3'
report['manuscript_replaced']=False
report['artificial_iqr_expansion']=False
cd14=stats[(stats.marker=='CD14')&(stats.method=='Raw')]
report['raw_cd14_median_percentile']=dict(zip(cd14.celltype,cd14['median']))
with fitz.open(HERE/'figure_1_five_celltypes.pdf') as doc:
    assert len(doc)==1
    report['pdf_page_count']=len(doc)
    report['pdf_dimensions_mm']=[doc[0].rect.width/72*25.4,doc[0].rect.height/72*25.4]
    doc[0].get_pixmap(dpi=180).save(qa/'pdf_render.png')
    # Render Panel C directly from the exported PDF, not the source PNG.
    clip=fitz.Rect(58/25.4*72,125/25.4*72,145/25.4*72,188/25.4*72)
    doc[0].get_pixmap(dpi=420,clip=clip).save(qa/'panel_c_pdf_render.png')
(qa/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
