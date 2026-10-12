"""Validate provenance, complete coverage, and exact story rendering at matched size."""
from pathlib import Path
import ast,json,hashlib,os,sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(HERE/'.cache/matplotlib');os.environ['MPLBACKEND']='Agg'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import fitz
import build_figure as b

def main():
    m=json.loads((HERE/'MANIFEST.json').read_text());style=json.loads((HERE/'STORY_STYLE.json').read_text())
    contract=json.loads((HERE/'DISPLAY_CONTRACT.json').read_text())
    assert all(b.sha(p)==h for p,h in m['input_sha256'].items())
    assert all(b.sha(p)==h for p,h in contract['input_sha256'].items())
    assert all(b.sha(HERE/p)==h for p,h in m['output_sha256'].items())
    meta=pd.read_csv(HERE/'display_metadata.csv.gz',low_memory=False,dtype={'unit_key':str})
    original=pd.read_csv(b.STORY/'alignment/umap/full_94488_umap_coordinates.csv.gz',low_memory=False)
    xy=np.load(HERE/'coordinates/RAMP.npy')
    np.testing.assert_array_equal(meta.cell_id,original.cell_uid)
    np.testing.assert_array_equal(meta.celltype,original.plot_label)
    np.testing.assert_array_equal(xy,original[['UMAP1','UMAP2']].to_numpy())
    assert len(meta)==94488 and meta.unit_key.nunique()==77
    for method in b.METHODS:
        p=json.loads((HERE/'coordinates'/f'{method}.json').read_text())
        assert p['cells_sha256']==contract['cells_sha256'] and p['protocol']==contract['protocol']
        assert np.load(HERE/'coordinates'/f'{method}.npy').shape==(94488,2)
    # Full-active MaxFuse row reassembly must match exactly, not just in size.
    mf=b.BASE/'maxfuse_full_panel/methods/maxfuse'
    ids=np.concatenate([np.load(mf/f'cca_embedding{i}_cell_ids.npy',allow_pickle=True) for i in [1,2]])
    values=np.vstack([np.load(mf/f'cca_embedding{i}.npy') for i in [1,2]])
    assert len(set(ids))==94488 and set(ids)==set(meta.cell_id)
    order=pd.Index(ids).get_indexer(meta.cell_id)
    np.testing.assert_allclose(np.load(mf/'values.npy'),values[order],atol=0,rtol=0)
    raw=np.load(b.BASE/'input/raw.npy');ready=np.load(b.BASE/'input/analysis_ready.npy')
    adt=meta.dataset.eq('GSE212797_ADT').to_numpy()
    np.testing.assert_allclose(ready[adt],np.log1p(raw[adt]),rtol=0,atol=0)
    np.testing.assert_array_equal(ready[~adt],raw[~adt])
    audit=pd.read_csv(HERE/'ridge_display_audit.csv')
    assert len(audit)==77*6*3
    assert audit.groupby(['method','marker']).n_cells.sum().eq(94488).all()
    assert audit.groupby(['method','marker']).unit_key.nunique().eq(77).all()
    assert not audit.method.eq('MaxFuse').any()
    # Use the original story function, without importing any source script.
    script=Path(style['style_source']);tree=ast.parse(script.read_text());const={}
    for n in tree.body:
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ['DATASET_COLORS','DATASET_NAMES','DISPLAY_ORDER','TYPE_COLORS']:
            const[n.targets[0].id]=ast.literal_eval(n.value)
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='build_c')
    captured={}
    def capture(fig,*args):captured['fig']=fig;return 'memory'
    env=dict(const,np=np,pd=pd,plt=plt,Line2D=Line2D,WIDE_WIDTH=7.,ROW_HEIGHT=3.45,
             INPUTS={'umap':b.STORY/'alignment/umap/full_94488_umap_coordinates.csv.gz'},
             header=lambda *a,**k:None,save=capture)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(script),'exec'),env)
    b.SCATTER_AREA=m['scatter_area_pt2']
    images=[];checks=[]
    for j,by in enumerate(['Dataset','Cell type']):
        env['build_c']();ref=captured['fig'];ref.canvas.draw()
        chosen=ref.axes[j]
        source_h=chosen.get_position().height*ref.get_figheight()*25.4
        assert np.isclose(source_h,style['source_plot_height_mm'])
        for ax in list(ref.axes):
            if ax is not chosen:ax.remove()
        if chosen.get_legend():chosen.get_legend().remove()
        chosen.set_title('',loc='left');chosen.set_axis_off()
        ref.set_size_inches(39/25.4,27/25.4);ref.set_dpi(300)
        chosen.set_position([0,0,1,1]);chosen.apply_aspect()
        new_h=chosen.get_position().height*ref.get_figheight()*25.4
        new_area=.78*(new_h/source_h)**2
        assert np.isclose(new_area,b.SCATTER_AREA)
        for artist in chosen.collections:artist.set_sizes([new_area])
        ref.canvas.draw();ref_image=np.asarray(ref.canvas.buffer_rgba()).copy()
        fresh=plt.figure(figsize=(39/25.4,27/25.4),dpi=300,facecolor='white')
        ax=fresh.add_axes([0,0,1,1]);artists=b.scatter(ax,meta,xy,by,style)
        fresh.canvas.draw();fresh_image=np.asarray(fresh.canvas.buffer_rgba()).copy()
        assert len(artists)==len(chosen.collections)
        for x,y in zip(artists,chosen.collections):
            np.testing.assert_array_equal(x.get_offsets(),y.get_offsets())
            np.testing.assert_array_equal(x.get_facecolors(),y.get_facecolors())
            np.testing.assert_allclose(x.get_sizes(),y.get_sizes(),rtol=1e-12,atol=1e-12)
        delta=np.abs(ref_image.astype(int)-fresh_image.astype(int))
        assert not delta.any(),(by,int(delta.max()),int(np.count_nonzero(delta)))
        checks.append(dict(view=by,per_cell_order_identical=True,colors_identical=True,alpha_identical=True,
                           point_area_scale_verified=True,all_pixels_identical=True,max_channel_difference=int(delta.max())))
        images.extend([ref_image,fresh_image]);plt.close(ref);plt.close(fresh)
    fig,axs=plt.subplots(2,2,figsize=(8,6),facecolor='white')
    for ax,im,title in zip(axs.ravel(),images,['Story: dataset','Supplement: dataset','Story: cell type','Supplement: cell type']):
        ax.imshow(im);ax.set_axis_off();ax.set_title(title,fontsize=11,weight='bold')
    fig.tight_layout();fig.savefig(HERE/'output/story_rendering_comparison.png',dpi=200);plt.close(fig)
    pdf=HERE/'output/pdf/hnscc_story_supplement.pdf'
    tmp=HERE/'tmp/pdfs';tmp.mkdir(parents=True,exist_ok=True)
    with fitz.open(pdf) as doc:
        assert len(doc)==1;page=doc[0]
        np.testing.assert_allclose([page.rect.width,page.rect.height],np.array([210,297])/25.4*72,atol=.02)
        txt=page.get_text()
        for term in ['94,488','22 shared proteins',*b.METHODS,'CD3','CD4','CD8']:assert term in txt,term
        for block in page.get_text('dict')['blocks']:
            for line in block.get('lines',[]):
                for span in line['spans']:
                    x0,y0,x1,y1=span['bbox']
                    assert x0>=0 and y0>=0 and x1<=page.rect.width and y1<=page.rect.height,span['text']
        page.get_pixmap(dpi=180,alpha=False).save(tmp/'page1.png')
    report=dict(status='PASS',cells=94488,sample_units=77,shared_markers=22,umap_views=14,ridge_panels=18,
        ramp_story_coordinates_exact=True,identical_cells_across_methods=True,full_active_maxfuse_identity_verified=True,
        ridge_complete_coverage=True,raw_assay_transforms_verified=True,story_rendering=checks,
        original_inputs_and_metrics_unchanged=True,page_count=1,page_mm=[210,297],pdf_text_within_page=True,
        final_pdf_sha256=b.sha(pdf),final_png_sha256=b.sha(HERE/'output/hnscc_story_supplement.png'))
    (HERE/'QA.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
