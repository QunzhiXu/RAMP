from pathlib import Path
import json, hashlib, ast
import pandas as pd
import numpy as np
import fitz
HERE=Path(__file__).resolve().parent
PLAN=json.loads((HERE/'REVISION_PLAN.json').read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def functions(p):return {f.name:ast.dump(f,include_attributes=False) for f in ast.parse(p.read_text()).body if isinstance(f,ast.FunctionDef)}
reports=[];ranges=[];hashes={}
for item in PLAN:
    old,new=Path(item['old']),Path(item['new'])
    before=json.loads((old/'MANIFEST.json').read_text());after=json.loads((new/'MANIFEST.json').read_text());cfg=json.loads((new/'RIDGE_CONFIG.json').read_text())
    a=pd.read_csv(old/'ridge_display_audit.csv');b=pd.read_csv(new/'ridge_display_audit.csv')
    keys=['method','marker','dataset','unit_key'];j=b.merge(a,on=keys,validate='one_to_one',suffixes=('_new','_old'))
    assert len(a)==len(b)==len(j)==cfg['ridge_units']*18
    for key in ['n_cells','ridge_color','top_to_bottom_index','fill_alpha','sigma_bins']:
        assert np.array_equal(j[key+'_new'],j[key+'_old']),(item['name'],key)
    assert sha(old/'ridge_unit_selection.csv')==sha(new/'ridge_unit_selection.csv')
    assert np.allclose(j.bin_width,(j.display_high_old-j.display_low_old)/160,rtol=2e-6,atol=1e-9)
    assert b.n_below.eq(0).all() and b.n_above.eq(0).all()
    assert b.histogram_count.eq(b.n_cells).all()
    assert b.endpoint_density_left.eq(0).all() and b.endpoint_density_right.eq(0).all()
    assert ((b.frozen_value_min-b.display_low)>=4*b.bandwidth_expression_units).all()
    assert ((b.display_high-b.frozen_value_max)>=4*b.bandwidth_expression_units).all()
    for method,g in b.groupby(['method','marker']):
        row=g.iloc[0];ranges.append(dict(case=item['name'],method=method[0],marker=method[1],old_low=row.old_display_low,old_high=row.old_display_high,new_low=row.display_low,new_high=row.display_high,bins=row.bins,bandwidth=row.bandwidth_expression_units,width_ratio=(row.display_high-row.display_low)/(row.old_display_high-row.old_display_low),restored_tail_observations=int((g.old_n_below+g.old_n_above).sum())))
    fa,fb=functions(old/'build_figure.py'),functions(new/'build_figure.py')
    for name in ['scatter','point_order','point_orders','load']:
        if name in fa:assert fa[name]==fb[name],name
    for key in ['cells','dataset_counts','display_cells','umap_protocols','dataset_palette','celltype_palette','draw_order','layout']:
        if key in before:assert before[key]==after[key],key
    for path,want in {**item['old_hashes'],**after['input_sha256']}.items():
        key=str(Path(path).resolve())
        if key not in hashes:hashes[key]=sha(path)
        assert hashes[key]==want,path
    for rel,want in after['output_sha256'].items():assert sha(new/rel)==want,rel
    pdf=new/'output/pdf'/Path(item['png']).with_suffix('.pdf')
    olddoc=fitz.open(old/'output/pdf'/pdf.name);doc=fitz.open(pdf);page=doc[0]
    assert len(doc)==1 and np.allclose([page.rect.width,page.rect.height],[210*72/25.4,297*72/25.4],atol=.01)
    assert 'Harmony-direct' not in page.get_text() and 'Harmony' in page.get_text()
    for geom in after['layout']:
        if geom['block']=='UMAP':
            rect=fitz.Rect(geom['x_mm'],geom['y_mm'],geom['x_mm']+geom['width_mm'],geom['y_mm']+geom['height_mm'])*72/25.4
            assert olddoc[0].get_pixmap(dpi=120,clip=rect).samples==page.get_pixmap(dpi=120,clip=rect).samples,(item['name'],geom)
    for x0,y0,x1,y1,*_ in page.get_text('words'):assert 0<=x0<=x1<=page.rect.width and 0<=y0<=y1<=page.rect.height
    qa=HERE/'qa';qa.mkdir(exist_ok=True)
    page.get_pixmap(dpi=110).save(qa/(new.parent.name+'.png'))
    report=dict(case=item['name'],status='passed',ridges=len(b),displayed_cells_per_method_marker=cfg['ridge_cells'],displayed_units=cfg['ridge_units'],all_displayed_observations_counted=True,zero_tail_observations_omitted=True,all_curve_endpoints_at_baseline=True,original_bandwidth_and_selection_preserved=True,umap_pixels_identical=True,source_data_and_old_outputs_unchanged=True,all_pdf_text_inside_page=True,max_range_expansion=float(((b.display_high-b.display_low)/(b.old_display_high-b.old_display_low)).max()))
    (new/'QA.json').write_text(json.dumps(report,indent=2)+'\n');reports.append(report)
    print('Verified',item['name'],'max expansion',round(report['max_range_expansion'],2),flush=True)
pd.DataFrame(ranges).to_csv(HERE/'RANGE_AUDIT.csv',index=False)
(HERE/'QA_SUMMARY.json').write_text(json.dumps(reports,indent=2)+'\n')
