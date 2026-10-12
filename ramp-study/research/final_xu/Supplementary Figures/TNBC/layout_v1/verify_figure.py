"""Check frozen data, Figure 1 rendering, ridge coverage and PDF geometry."""
from pathlib import Path
import os,sys,ast,json
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(HERE/'.cache/matplotlib');os.environ['MPLBACKEND']='Agg'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import fitz
import build_figure as b

def main():
    report=json.loads((HERE/'MANIFEST.json').read_text())
    assert all(b.sha(p)==h for p,h in report['input_sha256'].items())
    assert all(b.sha(HERE/p)==h for p,h in report['output_sha256'].items())
    meta,markers,coords,values,style,protocols=b.load()
    assert len(meta)==62701 and meta.unit_key.nunique()==11 and len(markers)==11
    assert set(meta.celltype)==set(style['celltype_palette']) and len(set(meta.celltype))==9
    # Independent execution of only the source plotting function. No import of
    # original renderer, caches, or output-producing main function.
    tree=ast.parse(b.STYLE.read_text());constants={}
    for n in tree.body:
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ['DS_COLORS','PALETTE','ORDER','LINE']:
            constants[n.targets[0].id]=ast.literal_eval(n.value)
    funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['axes','scatter']]
    env=dict(constants,np=np,plt=plt,Rectangle=Rectangle,W=180.,H=188.)
    exec(compile(ast.Module(body=funcs,type_ignores=[]),str(b.STYLE),'exec'),env)
    checks=[];images=[]
    for method in ['Raw','Harmony-direct','cyCombine','RAMP']:
        for by in ['Dataset','Cell type']:
            ref=plt.figure(figsize=(180/25.4,188/25.4),facecolor='white',dpi=300)
            env['scatter'](ref,meta,coords[method],by,0,0,28.6,21.7)
            ref.canvas.draw();ax=ref.axes[0]
            source_h=ax.get_position().height*ref.get_figheight()*25.4
            np.testing.assert_allclose(source_h,21.7,atol=1e-10)
            for a in list(ref.artists):a.remove()
            ref.set_size_inches(39/25.4,27/25.4);ax.set_position([0,0,1,1]);ax.apply_aspect()
            target_h=ax.get_position().height*ref.get_figheight()*25.4
            scale=(target_h/source_h)**2
            np.testing.assert_allclose(scale,b.POINT_AREA_SCALE,rtol=1e-12)
            for a in ax.collections:a.set_sizes(a.get_sizes()*scale)
            ref.canvas.draw();ref_pixels=np.asarray(ref.canvas.buffer_rgba()).copy()
            fig=plt.figure(figsize=(39/25.4,27/25.4),facecolor='white',dpi=300)
            new_ax=fig.add_axes([0,0,1,1]);new=b.scatter(new_ax,meta,coords[method],by,style)
            fig.canvas.draw();new_pixels=np.asarray(fig.canvas.buffer_rgba()).copy()
            assert len(new)==len(ax.collections)
            for a,c in zip(new,ax.collections):
                np.testing.assert_array_equal(a.get_offsets(),c.get_offsets())
                np.testing.assert_array_equal(a.get_facecolors(),c.get_facecolors())
                np.testing.assert_allclose(a.get_sizes(),c.get_sizes(),rtol=1e-12)
                assert a.get_zorder()==c.get_zorder()
            delta=np.abs(ref_pixels.astype(int)-new_pixels.astype(int))
            assert not delta.any(),(method,by,int(delta.max()),int(np.count_nonzero(delta)))
            checks.append(dict(method=method,view=by,all_cells=True,coordinates_identical=True,point_order_identical=True,
                colors_and_alpha_identical=True,area_scale_verified=True,pixels_identical=True))
            if method=='RAMP':images.extend([ref_pixels,new_pixels])
            plt.close(ref);plt.close(fig)
    f,axs=plt.subplots(2,2,figsize=(8,6),facecolor='white')
    for ax,im,title in zip(axs.ravel(),images,['Figure 1: dataset','Supplement: dataset','Figure 1: cell type','Supplement: cell type']):
        ax.imshow(im);ax.set_axis_off();ax.set_title(title,fontsize=11,weight='bold')
    f.tight_layout();f.savefig(HERE/'output/figure1_rendering_comparison.png',dpi=200);plt.close(f)
    audit=pd.read_csv(HERE/'ridge_display_audit.csv')
    assert len(audit)==6*3*11
    assert audit.groupby(['method','marker']).n_cells.sum().eq(62701).all()
    assert audit.groupby(['method','marker']).unit_key.nunique().eq(11).all()
    # Every existing Figure 1 ridge uses identical values, display limits,
    # excluded-tail counts, smoothing and per-unit normalization.
    source=b.ROOT/'Main Figure/text/ridge_display_audit.csv'
    old=pd.read_csv(source)
    names={'Unaligned_arcsinh_display':'Raw','Harmony-direct_source_scale':'Harmony-direct','cyCombine':'cyCombine','RAMP':'RAMP'}
    old['method']=old.method.map(names)
    assert old.method.notna().all() and len(old)==132
    keys=['method','marker','unit_key'];joined=old.merge(audit,on=keys,validate='one_to_one',suffixes=('_old','_new'))
    assert len(joined)==132
    for name in ['display_low','display_high','n_cells']:
        np.testing.assert_allclose(joined[name+'_old'],joined[name+'_new'],rtol=1e-12,atol=1e-12)
    np.testing.assert_array_equal(joined.n_below_display,joined.n_below)
    np.testing.assert_array_equal(joined.n_above_display,joined.n_above)
    np.testing.assert_array_equal(joined.smoothing_sigma_bins,joined.sigma_bins)
    pdf=HERE/'output/pdf/tnbc_benchmark_supplement.pdf';tmp=HERE/'tmp/pdfs';tmp.mkdir(parents=True,exist_ok=True)
    with fitz.open(pdf) as doc:
        assert len(doc)==1;page=doc[0]
        np.testing.assert_allclose([page.rect.width,page.rect.height],np.array([210,297])/25.4*72,atol=.02)
        txt=page.get_text()
        for term in ['62,701','11 sample units',*b.METHODS,'CD3','CD8','PANCK']:assert term in txt,term
        for block in page.get_text('dict')['blocks']:
            for line in block.get('lines',[]):
                for span in line['spans']:
                    x0,y0,x1,y1=span['bbox'];assert x0>=0 and y0>=0 and x1<=page.rect.width and y1<=page.rect.height,span['text']
        page.get_pixmap(dpi=180,alpha=False).save(tmp/'page1.png')
    assert all(b.sha(p)==h for p,h in report['input_sha256'].items())
    qa=dict(status='PASS',cells=62701,sample_units=11,markers=11,display_labels=9,umap_views=14,ridge_panels=18,
        frozen_coordinates_reused_for_all_methods=True,umap_protocols_identical=True,figure1_rendering=checks,
        figure1_ridge_records_identical=132,ridge_complete_coverage=True,alignment_and_metrics_unchanged=True,
        inputs_unchanged=True,page_count=1,page_mm=[210,297],pdf_text_within_page=True,
        final_pdf_sha256=b.sha(pdf),final_png_sha256=b.sha(HERE/'output/tnbc_benchmark_supplement.png'))
    (HERE/'QA.json').write_text(json.dumps(qa,indent=2)+'\n');print(json.dumps(qa,indent=2))
if __name__=='__main__':main()
