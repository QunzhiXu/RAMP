"""Full-height A with the supplied three-lane icon; expanded artifact controls."""
from pathlib import Path
import json
import rendering_helpers as h
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D, Bbox
from matplotlib import font_manager
from matplotlib.cbook import boxplot_stats

HERE=Path(__file__).resolve().parent
PREVIOUS=HERE.parent / "redesign_full_icon_20260914"
EXAMPLES=[
    dict(case='tnbc',unit='Meyer IMC|B12.64192',marker='CD8',
         cells=['CD8 T','CD4 T','Tumor / epithelial'],labels=['CD8 T','CD4 T','Tumor'],
         title='CD8',drivers=['Harmony-direct','cyCombine']),
    dict(case='bone_marrow',unit='BM CyTOF|H1',marker='CD4',
         cells=['CD4 T','Mature / CODEX B','CD8 T'],labels=['CD4 T','B cells','CD8 T'],
         title='CD4',drivers=['CytoNorm'])]
METHODS=['Raw','Harmony-direct','cyCombine','CytoNorm','RAMP']
COLORS=[h.BLUE,h.ORANGE,'#AAB5BB']


def build_a(fig):
    # Reuse the accepted method schematic, giving it more vertical room.
    # Only native vector positions change; all labels and algorithm steps remain.
    h.build_a(fig)
    remove=fig.texts[:2]
    for artist in remove:
        h.TEXT.remove(artist)
        artist.remove()
    scale=(182.-54.)/(121.-12.)
    offset=54.-12.*scale
    transform=Affine2D().scale(1,scale).translate(0,1-scale-offset/h.H)
    for artist in fig.artists:
        artist.set_transform(transform+fig.transFigure)
    for artist in fig.texts:
        x,y=artist.get_position()
        artist.set_position((x,1-(scale*((1-y)*h.H)+offset)/h.H))
    for ax in fig.axes:
        x,y,w,height=ax.get_position().bounds
        top=(1-y-height)*h.H
        newtop=scale*top+offset
        ax.set_position([x,1-newtop/h.H-height*scale,w,height*scale])
    h.panel_title(fig,'A','RAMP overview',2,4.2)
    h.text(fig,2,11.5,'A.1',size=7.2,bold=True,color=h.GREEN)
    h.text(fig,10,11.5,'Cross-platform alignment',size=7.2,bold=True)
    icon=h.track(h.ROOT/'icon/ramp_full.png')
    im=plt.imread(icon)
    ax=h.axes(fig,2,15.5,53,26.5)
    ax.imshow(im);ax.axis('off')
    h.line(fig,2,44.5,54,44.5,color='#CBD1D4',lw=.4)
    h.text(fig,2,48.5,'A.2',size=7.2,bold=True,color=h.GREEN)
    h.text(fig,10,48.5,'Method overview',size=7.2,bold=True)


def build_c(fig):
    h.panel_title(fig,'C','Artifacts in comparison methods',59,128)
    d=pd.read_csv(h.track(PREVIOUS/'artifact_cells.csv.gz'),float_precision='round_trip')
    saved=[]
    for row,ex in enumerate(EXAMPLES):
        y=[135.5,164][row]
        h.text(fig,65,y-2.6,ex['title'],size=6.2,bold=True)
        for x,label,color in zip([103,116,129],ex['labels'],COLORS):
            h.mini_legend(fig,x,y-2.6,[label],[color],0,size=5.5)
        ax=h.statistical_axes(fig,65,y,78,[20,18.5][row])
        subset=d[d.benchmark.eq(ex['case'])&d.unit_key.eq(ex['unit'])&d.marker.eq(ex['marker'])]
        for method in ex['drivers']:
            i=METHODS.index(method)
            ax.axvspan(i-.46,i+.46,color='#FBECE6',zorder=0)
        for i,method in enumerate(METHODS):
            g=subset[subset.method.eq(method)]
            arrays=[g[g.celltype.eq(label)].percentile.to_numpy() for label in ex['cells']]
            assert min(map(len,arrays))>=100
            b=ax.boxplot(arrays,positions=i+np.array([-.28,0,.28]),widths=.22,
                patch_artist=True,showfliers=False,manage_ticks=False,
                medianprops={'color':h.INK,'linewidth':.6},
                whiskerprops={'color':h.LINE,'linewidth':.45},
                capprops={'color':h.LINE,'linewidth':.45},
                boxprops={'edgecolor':h.LINE,'linewidth':.45})
            for patch,color in zip(b['boxes'],COLORS):patch.set_facecolor(color)
            for label,array in zip(ex['cells'],arrays):
                stat=boxplot_stats(array,whis=1.5)[0]
                saved.append({'case':ex['case'],'unit_key':ex['unit'],'marker':ex['marker'],
                    'method':method,'celltype':label,'n_cells':len(array),'scope_cells':len(g),
                    'mean_percentile':array.mean(),'q1':stat['q1'],'median':stat['med'],
                    'q3':stat['q3'],'whislo':stat['whislo'],'whishi':stat['whishi']})
        ax.set_xlim(-.5,4.5);ax.set_ylim(0,1);ax.set_yticks([0,.5,1])
        ax.set_ylabel(f"{ex['marker']} percentile",fontsize=5.8,labelpad=2)
        ax.set_xticks(range(5),['Raw','Harmony-\ndirect','cyCombine','CytoNorm','RAMP'])
        ax.tick_params(axis='x',labelsize=5.5)
        for label,method in zip(ax.get_xticklabels(),METHODS):
            if method in ex['drivers']:label.set_color('#A64736')
            if method=='RAMP':label.set_color(h.GREEN);label.set_weight('bold')
    out=pd.DataFrame(saved)
    old=pd.read_csv(h.track(h.MAIN/'Panel D/artifact_box_statistics.csv'),float_precision='round_trip')
    for r in old.itertuples():
        new=out[out.case.eq(r.case)&out.marker.eq(r.marker)&out.method.eq(r.method)&out.celltype.eq(r.celltype)].iloc[0]
        for field in ['mean_percentile','q1','median','q3']:
            assert np.isclose(new[field],getattr(r,field),atol=1e-12,rtol=0)
    out.to_csv(HERE/'artifact_plotted_statistics.csv',index=False)


def build_d(fig):
    h.panel_title(fig,'D','Cell-type shifts',147,128)
    h.text(fig,164,133,'All 4 benchmarks',size=6,ha='center')
    h.mini_legend(fig,151,137,['Panel C examples'],[h.BLUE],0,size=5.4,marker='D')
    d=pd.read_csv(h.track(HERE/'celltype_quartile_shifts_display.csv.gz'),float_precision='round_trip')
    methods=['Harmony-direct[source_scale]','Harmony-direct[fixed_arcsinh]',
             'cyCombine','CytoNorm','RAMP']
    labels=['Harmony-direct','Harmony-arcsinh','cyCombine','CytoNorm','RAMP']
    ax=h.axes(fig,148,140,30,39.5)
    rng=np.random.default_rng(20260914)
    records=[]
    highlights=[]
    example_rows={methods[0]:('tnbc','Meyer IMC|B12.64192','CD8','CD4 T'),
                  'cyCombine':('tnbc','Meyer IMC|B12.64192','CD8','CD4 T'),
                  'CytoNorm':('bone_marrow','BM CyTOF|H1','CD4','Mature / CODEX B')}
    n_expected=json.loads((HERE/'computation_manifest.json').read_text())['display_groups_per_method']
    for i,(method,label) in enumerate(zip(methods,labels)):
        sub=d[d.method.eq(method)]
        v=sub.abs_delta_median_pp.to_numpy()
        assert len(v)==n_expected and np.isfinite(v).all()
        color=h.GREEN if method=='RAMP' else h.ORANGE
        # Every combination is displayed, including zero values and long tails.
        ax.scatter(v, i+.43+rng.uniform(-.18,.18,len(v)),s=.45,
                   color=color,alpha=.12,linewidths=0,rasterized=True,zorder=2)
        b=ax.boxplot([v],positions=[i+.43],vert=False,widths=.24,patch_artist=True,
                     showfliers=False,manage_ticks=False,
                     boxprops={'edgecolor':h.LINE,'linewidth':.5},
                     whiskerprops={'color':h.LINE,'linewidth':.5},
                     capprops={'color':h.LINE,'linewidth':.5},
                     medianprops={'color':h.INK,'linewidth':.75},zorder=4)
        b['boxes'][0].set_facecolor(color)
        b['boxes'][0].set_alpha(.83)
        ax.text(0,i,label,va='center',fontsize=5.6,color=h.GREEN if method=='RAMP' else h.INK,
                weight='bold' if method=='RAMP' else 'normal')
        stat=boxplot_stats(v,whis=1.5)[0]
        records.append(dict(method=method,n=len(v),q1=stat['q1'],median=stat['med'],
                            q3=stat['q3'],whislo=stat['whislo'],whishi=stat['whishi']))
        if method in example_rows:
            case,unit,marker,celltype=example_rows[method]
            selected=sub[sub.case.eq(case)&sub.unit_key.eq(unit)&sub.marker.eq(marker)&sub.celltype.eq(celltype)]
            assert len(selected)==1
            r=selected.iloc[0]
            ax.scatter([r.abs_delta_median_pp],[i+.43],s=10,marker='D',color=h.BLUE,
                       edgecolors='white',linewidths=.35,zorder=6)
            highlights.append(r.to_dict())
        if method=='RAMP':
            assert np.median(v)==0
            ax.text(4,i+.43,'Median 0',fontsize=5.5,color=h.GREEN,va='center')
    # Linear scale, zero included; no clipping of the observed tails.
    assert d.abs_delta_median_pp.max()<95
    ax.set_xlim(-1.3,95);ax.set_ylim(4.8,-.16)
    ax.set_yticks([]);ax.set_xticks([0,30,60,90])
    ax.tick_params(axis='x',labelsize=5.5,length=1.5,pad=1.5,width=.45)
    for side in ['top','right','left']:ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(h.LINE);ax.spines['bottom'].set_linewidth(.45)
    h.text(fig,163,185,'Absolute median shift\n(percentile points)',size=5.7,ha='center')
    pd.DataFrame(records).to_csv(HERE/'median_shift_box_statistics.csv',index=False)
    pd.DataFrame(highlights).to_csv(HERE/'highlighted_panel_c_examples.csv',index=False)


def verify_layout(fig):
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    text=list(h.TEXT)
    for ax in fig.axes:
        if ax.axison:
            xmin,xmax=sorted(ax.get_xlim());ymin,ymax=sorted(ax.get_ylim())
            text += [t for loc,t in zip(ax.get_xticks(),ax.get_xticklabels()) if xmin<=loc<=xmax]
            text += [t for loc,t in zip(ax.get_yticks(),ax.get_yticklabels()) if ymin<=loc<=ymax]
            text += [ax.xaxis.label,ax.yaxis.label]
        text += list(ax.texts)
    text=[t for t in text if t.get_visible() and t.get_text()]
    collisions=[]
    for i,left in enumerate(text):
        a=left.get_window_extent(renderer)
        assert a.x0>=-1 and a.y0>=-1 and a.x1<=fig.bbox.width+1 and a.y1<=fig.bbox.height+1,(left.get_text(),a)
        for right in text[i+1:]:
            b=right.get_window_extent(renderer)
            dx=min(a.x1,b.x1)-max(a.x0,b.x0);dy=min(a.y1,b.y1)-max(a.y0,b.y0)
            area=dx*dy*(25.4/fig.dpi)**2
            if dx>0 and dy>0 and area>.08:
                collisions.append({'left':left.get_text(),'right':right.get_text(),'area_mm2':area})
    (HERE/'qa').mkdir(exist_ok=True)
    (HERE/'qa/text_collisions.json').write_text(json.dumps(collisions,indent=2)+'\n')
    assert not collisions,collisions


def main():
    for filename in ['Arial.ttf','Arial Bold.ttf']:
        font_manager.fontManager.addfont('/System/Library/Fonts/Supplemental/'+filename)
    plt.rcParams.update({'font.family':'Arial','font.size':6.2,'text.color':h.INK,
        'axes.labelcolor':h.INK,'svg.fonttype':'none','pdf.fonttype':42,
        'ps.fonttype':42,'axes.linewidth':.45,'savefig.facecolor':'white'})
    meta,coords=h.load_case('tnbc',h.READS)
    markers,matrices=h.load_matrices('tnbc',meta,h.READS)
    assert len(meta)==62701 and meta.unit_key.nunique()==11
    fig=plt.figure(figsize=(h.W/25.4,h.H/25.4),facecolor='white',dpi=160)
    build_a(fig)
    h.build_b(fig,meta,coords,markers,matrices)
    build_c(fig)
    build_d(fig)
    old=pd.read_csv(h.track(h.MAIN/'text/ridge_display_audit.csv'))
    pd.testing.assert_frame_equal(pd.DataFrame(h.AUDIT),old,check_exact=False,rtol=1e-12,atol=1e-12)
    pd.DataFrame(h.AUDIT).to_csv(HERE/'ridge_display_audit.csv',index=False)
    audit=json.loads((HERE/'audit_manifest.json').read_text())
    assert all(h.sha(p)==s for p,s in audit['source_sha256'].items())
    verify_layout(fig)
    for ext,dpi in [('png',420),('pdf',600),('svg',600)]:
        fig.savefig(HERE/f'figure_1_celltype_percentile_shift.{ext}',dpi=dpi)
    # Native figure exports of individual panel regions, preserving physical sizes.
    panels={'A':(0,0,56,188),'B':(58,0,122,123),
            'C':(58,125,87,63),'D':(146,125,34,63)}
    for letter,(x,y,w,ht) in panels.items():
        folder=HERE/f'Panel {letter}';folder.mkdir(exist_ok=True)
        bbox=Bbox.from_bounds(x/25.4,(h.H-y-ht)/25.4,w/25.4,ht/25.4)
        fig.savefig(folder/f'panel_{letter.lower()}.png',dpi=420,bbox_inches=bbox,pad_inches=0)
    plt.close(fig)
    oldhash=json.loads((HERE/'previous_version_hashes.json').read_text())
    assert all(h.sha(p)==s for p,s in oldhash.items())
    prior=json.loads((HERE/'prior_draft_sha256.json').read_text())
    assert all(h.sha(p)==s for p,s in prior.items())
    assert all(h.sha(p)==s for p,s in h.READS.items())
    manifest={'canvas_mm':[180,188],
        'layout':'Full-height A at left; B upper right; expanded C (old D) and D (old E) lower right.',
        'panel_a':'A.1 supplied full three-lane RAMP icon; A.2 accepted three-stage vector method overview.',
        'old_panel_c':'Removed; the two metrics are already in Figure 2.',
        'artifact_added_celltypes':{'tnbc_CD8':'Tumor / epithelial','bone_marrow_CD4':'CD8 T'},
        'artifact_axes_width_mm':78,'previous_artifact_axes_width_mm':62,
        'artifact_examples':EXAMPLES,'artifact_methods':METHODS,
        'rank_scope':'All cells of each selected sample within method; no re-ranking among just the three displayed types.',
        'original_artifact_quartiles_unchanged':True,'old_ten_auc_values_unchanged':True,
        'ridge_display_audit_unchanged':True,'benchmark_cells':62701,'umap_refit':False,
        'rank_statistic':'Absolute median percentile shift per eligible sample x cell type x marker; all shared markers, four benchmarks','input_sha256':h.READS,'artifact_input_sha256':audit['source_sha256'],
        'previous_version_unchanged':True,'manuscript_replaced':False,'text_collisions':[],
        'source_script_sha256':h.sha(__file__)}
    (HERE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(HERE/'figure_1_celltype_percentile_shift.png')


if __name__=='__main__':main()
