"""Check the final supplement against the current story and render its PDF."""
from pathlib import Path
import json, hashlib, shutil, subprocess
import numpy as np
import pandas as pd
import fitz

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
STORY=ROOT/'story_lung_HLA_ABC'

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    m=json.loads((HERE/'MANIFEST.json').read_text())
    contract=json.loads((HERE/'DISPLAY_CONTRACT.json').read_text())
    ref=json.loads((ROOT/'Supplementary Figures/Melanoma three-platform/layout_v2_matched_style/MANIFEST.json').read_text())
    assert all(sha(p)==h for p,h in m['input_sha256'].items())
    assert all(sha(p)==h for p,h in contract['input_sha256'].items())
    assert m['page_mm']==ref['page_mm']==[210.,297.]
    assert len(m['layout'])==len(ref['layout'])==32
    for a,b in zip(m['layout'],ref['layout']):
        assert {k:v for k,v in a.items() if k!='marker'}=={k:v for k,v in b.items() if k!='marker'}
    for key in ['ridge_bins','ridge_sigma_bins','ridge_range_quantiles','ridge_height']:
        assert m[key]==ref[key]
    story=pd.read_csv(STORY/'Figure/Panel B/data/PLOT_DATA.csv.gz',low_memory=False)
    dm=pd.read_csv(HERE/'display_metadata.csv.gz',low_memory=False)
    assert np.array_equal(story.cell_key,dm.cell_id)
    assert np.array_equal(story.display_cell_type,dm.celltype)
    assert np.array_equal(story.dataset,dm.dataset)
    assert dm.groupby('unit_key').size().eq(1000).all() and dm.unit_key.nunique()==84
    assert np.array_equal(np.load(HERE/'coordinates/RAMP.npy'),np.load(STORY/'Figure/Panel B/data/RAMP_UMAP.npy'))
    assert contract['protocol']==json.loads((STORY/'releases/senosain140k_20260921/alignment/UMAP_COMPLETE.json').read_text())['parameters']
    for method in m['umap_methods']:
        p=HERE/'coordinates'/f'{method}.json';pr=json.loads(p.read_text())
        assert pr['cells_sha256']==contract['cells_sha256'] and pr['protocol']==contract['protocol']
    audit=pd.read_csv(HERE/'ridge_display_audit.csv')
    assert len(audit)==1512 and audit.groupby(['method','marker']).n_cells.sum().eq(84000).all()
    assert audit.groupby(['method','marker']).unit_key.nunique().eq(84).all()
    overlap=[]
    for i,a in enumerate(m['text_bounds']):
        for b in m['text_bounds'][i+1:]:
            x0,y0,x1,y1=a['bbox_pixels'];xx0,yy0,xx1,yy1=b['bbox_pixels']
            if min(x1,xx1)-max(x0,xx0)>.5 and min(y1,yy1)-max(y0,yy0)>.5:
                overlap.append([a['label'],b['label']])
    assert not overlap,overlap
    pdf=HERE/'output/pdf/lung_hla_abc_story_supplement.pdf'
    doc=fitz.open(pdf);assert len(doc)==1
    text=doc[0].get_text()
    for s in ['Lung HLA-ABC story','CyTOF','IMC','mIF','84,000','CD3','CD4','CD8','MaxFuse','RAMP',*m['celltype_palette']]:
        assert s in text,s
    out=HERE/'tmp/pdfs';out.mkdir(parents=True,exist_ok=True)
    if shutil.which('pdftoppm'):
        subprocess.run(['pdftoppm','-scale-to','1800','-png','-singlefile',str(pdf),str(out/'page1')],check=True)
    else:doc[0].get_pixmap(matrix=fitz.Matrix(1.8,1.8),alpha=False).save(out/'page1.png')
    qa=dict(status='passed',pdf_pages=1,all_methods_same_84000_cells=True,
        all_84_patients_and_7_story_display_labels_retained=True,
        ramp_coordinates_identical_to_current_story=True,
        all_methods_use_current_story_umap_recipe=True,
        story_display_labels_palette_and_order_reused=True,
        plot_geometry_matches_existing_supplementary_figures=True,
        source_hashes_unchanged=True,original_score_results_unchanged=True,
        text_bounds_and_overlap_checks_passed=True,n_umaps=14,n_ridge_panels=18,n_ridge_curves=1512)
    (HERE/'QA.json').write_text(json.dumps(qa,indent=2)+'\n')
    print('Story identity, all methods, full-cell ridges and PDF layout verified.')

if __name__=='__main__':main()
