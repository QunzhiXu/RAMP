"""Three examples: shared H1 CD4 for Harmony-direct and CytoNorm, plus CD8 and CD14."""
from pathlib import Path
import os,sys,json,importlib.util
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
MAIN=HERE.parent
ROOT=MAIN.parent
PREVIOUS=MAIN/'celltype_percentile_shift_20260914'
AUDIT=MAIN/'fastmnn_reconstructed_audit_20260914'
os.environ['MPLCONFIGDIR']=str(HERE/'.cache/matplotlib')
import rendering_helpers as h
spec=importlib.util.spec_from_file_location('accepted_figure',PREVIOUS/'build_figure.py')
accepted=importlib.util.module_from_spec(spec);spec.loader.exec_module(accepted)
accepted.HERE=HERE
sys.path.insert(0,str(MAIN/'pair_inversion_boxplot_20260914'))
import compute_pair_inversions as frozen
os.environ['NUMBA_CACHE_DIR']=str(HERE/'.cache/numba')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from matplotlib import font_manager
from matplotlib.colors import to_rgb
from matplotlib.cbook import boxplot_stats
from scipy.stats import rankdata

CELLS=['CD4 T','CD8 T','B cells']
COLORS=[h.BLUE,h.ORANGE,'#AAB5BB']
MONOCYTE_COLOR='#8571A8'
MACROPHAGE_COLOR='#C7A43C'
LEGEND_CELLS=CELLS+['Monocytes','Macrophages']
LEGEND_COLORS=COLORS+[MONOCYTE_COLOR,MACROPHAGE_COLOR]
METHOD_MAP={'Raw':'Raw','Harmony-direct[source_scale]':'Harmony-direct',
    'Harmony-direct[fixed_arcsinh]':'Harmony-arcsinh','cyCombine':'cyCombine',
    'CytoNorm':'CytoNorm','RAMP':'RAMP','fastMNN reconstructed':'fastMNN'}
EXAMPLES=[
    dict(case='bone_marrow',unit='BM CyTOF|H1',marker='CD4',
        rect=[65,138.5,78,17],methods=['Raw','Harmony-direct','CytoNorm','RAMP'],drivers=['Harmony-direct','CytoNorm']),
    dict(case='tnbc',unit='Shiao CODEX|h17',marker='CD8',
        rect=[65,164,33.5,18.5],methods=['Raw','cyCombine','RAMP'],drivers=['cyCombine']),
    dict(case='bone_marrow',unit='BM CyTOF|H2',marker='CD14',
        rect=[109.5,164,33.5,18.5],methods=['Raw','fastMNN','RAMP'],drivers=['fastMNN'])]

for ex in EXAMPLES:
    ex['celltypes']=CELLS+['Monocytes']
    ex['colors']=COLORS+[MONOCYTE_COLOR]
    if ex['case']=='tnbc':
        ex['celltypes']=['CD4 T','CD8 T','Macrophages']
        ex['colors']=[h.BLUE,h.ORANGE,MACROPHAGE_COLOR]

def prepare():
    cell_rows=[];stats=[];all_auc=[]
    for case in ['tnbc','bone_marrow']:
        p=ROOT/'benchmark_results'/case/'ramp/alignment/input'
        meta=pd.read_csv(frozen.track(p/'cell_metadata.csv.gz'),low_memory=False,dtype={'cell_id':str})
        markers=frozen.track(p/'markers.txt').read_text().splitlines()
        arrays={'Raw':np.load(frozen.track(p/'raw.npy'))}
        for name,_,z in frozen.load_methods(case,meta.cell_id.to_numpy(),markers):arrays[METHOD_MAP[name]]=z
        arrays['fastMNN']=np.load(frozen.track(AUDIT/case/'reconstructed.npy'))
        for ex in [x for x in EXAMPLES if x['case']==case]:
            ix=np.flatnonzero(meta.unit_key.eq(ex['unit']))
            native=meta.iloc[ix].celltype.to_numpy()
            display_cells=ex['celltypes'];display_colors=ex['colors']
            label_map={'CD4 T':'CD4 T','CD8 T':'CD8 T',
                'Mature / CODEX B':'B cells','Monocyte':'Monocytes','Macrophage':'Macrophages'}
            lab=np.array([label_map.get(v,'not displayed') for v in native],dtype=object)
            if case=='tnbc':
                # Respect the provider labels: Plasma is not relabelled as B cells,
                # and Mac is not relabelled as Monocytes.
                provider=meta.iloc[ix].provider_label_l1
                lab[provider.isin(['B_cell','Bcell']).to_numpy()]='B cells'
                lab[provider.eq('Monocyte').to_numpy()]='Monocytes'
                lab[provider.eq('Mac').to_numpy()]='Macrophages'
                assert not np.any(provider.eq('Plasma').to_numpy() & (lab=='B cells'))
            col=markers.index(ex['marker'])
            assert min((lab==v).sum() for v in display_cells)>=50
            for method,z in arrays.items():
                # All sample cells define each method's percentile reference.
                values=z[ix,col];r=rankdata(values)/(len(ix)+1)
                for label,color in zip(display_cells,display_colors):
                    take=lab==label;v=r[take];ss=boxplot_stats(v,whis=1.5)[0]
                    stats.append(dict(case=case,unit_key=ex['unit'],marker=ex['marker'],method=method,celltype=label,
                        n_cells=len(v),mean_percentile=float(v.mean()),q1=ss['q1'],median=ss['med'],q3=ss['q3'],
                        whislo=ss['whislo'],whishi=ss['whishi'],displayed=method in ex['methods']))
                    for cid,val,pct in zip(meta.cell_id.to_numpy()[ix][take],values[take],v):
                        cell_rows.append(dict(case=case,unit_key=ex['unit'],marker=ex['marker'],method=method,
                            celltype=label,cell_id=cid,value=val,percentile=pct,displayed=method in ex['methods']))
                pairs=[('CD4 T','CD8 T'),('CD4 T','B cells'),('CD8 T','B cells'),
                    ('Monocytes','CD4 T'),('Monocytes','CD8 T'),('Monocytes','B cells'),
                    ('CD4 T','Macrophages'),('CD8 T','Macrophages'),('B cells','Macrophages'),
                    ('Monocytes','Macrophages')]
                for positive,negative in pairs:
                    if positive not in display_cells or negative not in display_cells:continue
                    aa=values[lab==positive];bb=values[lab==negative]
                    assert min(len(aa),len(bb))>=50
                    rr=rankdata(np.r_[aa,bb])
                    auc=(rr[:len(aa)].sum()-len(aa)*(len(aa)+1)/2)/(len(aa)*len(bb))
                    all_auc.append(dict(case=case,unit_key=ex['unit'],marker=ex['marker'],method=method,
                        positive=positive,negative=negative,n_positive=len(aa),n_negative=len(bb),auc=auc))
    cells=pd.DataFrame(cell_rows);s=pd.DataFrame(stats)
    cells.to_csv(HERE/'artifact_cells.csv.gz',index=False)
    s.to_csv(HERE/'artifact_box_statistics.csv',index=False)
    pd.DataFrame(all_auc).to_csv(HERE/'artifact_auc.csv',index=False)
    # Original CytoNorm/CD4 example is numerically unchanged despite compacting.
    old=pd.read_csv(h.track(MAIN/'redesign_full_icon_20260914/all_celltype_statistics.csv'))
    checked=0
    for row in old[old.benchmark.eq('bone_marrow')&old.marker.eq('CD4')&old.unit_key.eq('BM CyTOF|H1')&old.celltype.isin(['CD4 T','CD8 T','Mature / CODEX B'])].itertuples():
        label='B cells' if row.celltype=='Mature / CODEX B' else row.celltype
        new=s[s.case.eq('bone_marrow')&s.marker.eq('CD4')&s.unit_key.eq(row.unit_key)&s.method.eq(row.method)&s.celltype.eq(label)].iloc[0]
        for q in ['q1','median','q3']:
            assert abs(new[q]-getattr(row,q))<1e-12
            checked+=1
    assert checked>0
    return cells,s,checked

def build_c(fig,cells):
    h.panel_title(fig,'C','Artifacts in comparison methods',59,128)
    # Five distinct legend identities, with only adequately represented groups plotted.
    for x,label,color in zip([65,78,92,107,125],LEGEND_CELLS,LEGEND_COLORS):h.mini_legend(fig,x,132.9,[label],[color],0,size=5.5)
    for ex in EXAMPLES:
        x,y,w,ht=ex['rect'];methods=ex['methods']
        display_cells=ex.get('celltypes',CELLS);display_colors=ex.get('colors',COLORS)
        offsets=np.array([-.28,0,.28]) if len(display_cells)==3 else np.array([-.30,-.10,.10,.30])
        width=.22 if len(display_cells)==3 else .16
        h.text(fig,x,y-2.6,ex['marker'],size=6.2,bold=True)
        ax=h.statistical_axes(fig,x,y,w,ht)
        sub=cells[cells.case.eq(ex['case'])&cells.unit_key.eq(ex['unit'])&cells.marker.eq(ex['marker'])]
        for driver in ex['drivers']:
            i=methods.index(driver);ax.axvspan(i-.46,i+.46,color='#FBECE6',zorder=0)
        for i,method in enumerate(methods):
            g=sub[sub.method.eq(method)]
            arrays=[g[g.celltype.eq(label)].percentile.to_numpy() for label in display_cells]
            b=ax.boxplot(arrays,positions=i+offsets,widths=width,
                patch_artist=True,showfliers=False,manage_ticks=False,
                medianprops={'linewidth':.32},whiskerprops={'linewidth':.28},
                capprops={'linewidth':.28},boxprops={'linewidth':.28})
            # Preserve the actual narrow IQRs: colored thin strokes avoid a black bar
            # when the median and box boundaries occupy almost the same position.
            for j,(patch,color) in enumerate(zip(b['boxes'],display_colors)):
                edge=tuple(.82*v for v in to_rgb(color))
                median=tuple(.72*v for v in to_rgb(color))
                patch.set_facecolor(color);patch.set_edgecolor(edge)
                b['medians'][j].set_color(median)
                for artist in b['whiskers'][2*j:2*j+2]+b['caps'][2*j:2*j+2]:
                    artist.set_color(edge)
        # A shared display margin separates high-percentile boxes from the frame.
        ax.set_xlim(-.5,len(methods)-.5);ax.set_ylim(0,1.04);ax.set_yticks([0,.5,1])
        # Mark the marker-positive Raw population as a biological reference.
        # The label points at its real upper quartile; no box is enlarged.
        reference={'CD4':'CD4 T','CD8':'CD8 T','CD14':'Monocytes'}[ex['marker']]
        j=display_cells.index(reference)
        raw=sub[sub.method.eq('Raw')&sub.celltype.eq(reference)].percentile.to_numpy()
        color=display_colors[j]
        # Separate text from the leader so text-overlap QA measures text only.
        ax.text(8.04/w,1.155,reference,transform=ax.transAxes,ha='left',va='center',
            fontsize=5.4,fontweight='bold',color=color,clip_on=False,zorder=6)
        tail_x=(11.2225 if reference=='Monocytes' else 7.5375)/w
        ax.annotate('',xy=(offsets[j],float(np.quantile(raw,.75))),
            xycoords='data',xytext=(tail_x,1.06),textcoords='axes fraction',
            arrowprops=dict(arrowstyle='-|>',lw=.48,color=color,
                mutation_scale=4.0,shrinkA=0,shrinkB=1.4),
            annotation_clip=False,zorder=6)
        ax.set_ylabel(ex['marker']+' percentile',fontsize=5.8,labelpad=2)
        ax.set_xticks(range(len(methods)),['Harmony-\ndirect' if m=='Harmony-direct' and w<50 else m for m in methods])
        ax.tick_params(axis='x',labelsize=5.5)
        for t,m in zip(ax.get_xticklabels(),methods):
            if m in ex['drivers']:t.set_color('#A64736')
            if m=='RAMP':t.set_color(h.GREEN);t.set_weight('bold')

def build_d(fig):
    h.panel_title(fig,'D','Cell-type shifts',147,128)
    h.text(fig,164,133,'All 4 benchmarks',size=6,ha='center')
    d=pd.read_csv(h.track(AUDIT/'comparison_celltype_shifts.csv.gz'),low_memory=False,float_precision='round_trip')
    methods=['Harmony-direct[source_scale]','Harmony-direct[fixed_arcsinh]','cyCombine','CytoNorm','fastMNN reconstructed','RAMP']
    labels=['Harmony-direct','Harmony-arcsinh','cyCombine','CytoNorm','fastMNN (recon.)','RAMP']
    ax=h.axes(fig,148,135.5,30,44);rng=np.random.default_rng(20260914)
    stats=[];over=[]
    for i,(method,label) in enumerate(zip(methods,labels)):
        v=d[d.method.eq(method)].abs_delta_median_pp.to_numpy();assert len(v)==6173
        color=h.GREEN if method=='RAMP' else h.ORANGE
        yy=i+.43+rng.uniform(-.18,.18,len(v));within=v<=60
        ax.scatter(v[within],yy[within],s=.9,color=color,alpha=.12,linewidths=0,rasterized=True,zorder=2)
        ax.scatter(np.full((~within).sum(),60.),yy[~within],s=1.4,marker='>',color=color,alpha=.35,lw=0,zorder=3,clip_on=False)
        b=ax.boxplot(v,positions=[i+.43],vert=False,widths=.24,patch_artist=True,showfliers=False,manage_ticks=False,
            boxprops={'edgecolor':h.LINE,'lw':.5},whiskerprops={'color':h.LINE,'lw':.5},capprops={'color':h.LINE,'lw':.5},
            medianprops={'color':h.INK,'lw':.75},zorder=4)
        b['boxes'][0].set_facecolor(color);b['boxes'][0].set_alpha(.83)
        ax.text(0,i,label,va='center',fontsize=5.6,color=h.GREEN if method=='RAMP' else h.INK,
            weight='bold' if method=='RAMP' else 'normal')
        ss=boxplot_stats(v)[0];stats.append(dict(method=method,n=len(v),median=ss['med'],q1=ss['q1'],q3=ss['q3']))
        over.append(dict(method=method,n_above_60=int((v>60).sum()),maximum=float(v.max())))
    ax.set_xlim(-.8,60);ax.set_ylim(5.8,-.16);ax.set_xticks([0,20,40,60]);ax.set_yticks([])
    ax.tick_params(axis='x',labelsize=5.5,length=1.5,pad=1.5,width=.45)
    ax.spines[['top','right','left']].set_visible(False);ax.spines['bottom'].set_color(h.LINE);ax.spines['bottom'].set_linewidth(.45)
    h.text(fig,163,185,'Absolute median shift\n(percentile points)',size=5.7,ha='center')
    pd.DataFrame(stats).to_csv(HERE/'median_shift_box_statistics.csv',index=False)
    pd.DataFrame(over).to_csv(HERE/'panel_d_display_overflow.csv',index=False)

def main():
    for filename in ['Arial.ttf','Arial Bold.ttf']:font_manager.fontManager.addfont('/System/Library/Fonts/Supplemental/'+filename)
    plt.rcParams.update({'font.family':'Arial','font.size':6.2,'text.color':h.INK,'axes.labelcolor':h.INK,
        'svg.fonttype':'none','pdf.fonttype':42,'ps.fonttype':42,'axes.linewidth':.45,'savefig.facecolor':'white'})
    h.track(__file__);h.track(HERE/'rendering_helpers.py');h.track(PREVIOUS/'build_figure.py')
    old_files=[PREVIOUS/f'figure_1_celltype_percentile_shift.{e}' for e in ['png','pdf','svg']]
    old_hash={str(p):h.sha(p) for p in old_files}
    cells=pd.read_csv(h.track(MAIN/'shared_cd4_h1_20260914/artifact_cells.csv.gz'),float_precision='round_trip')
    stats=pd.read_csv(h.track(MAIN/'shared_cd4_h1_20260914/artifact_box_statistics.csv'),float_precision='round_trip')
    checked=json.loads((MAIN/'shared_cd4_h1_20260914/manifest.json').read_text())['old_cd4_quartile_checks']
    meta,coords=h.load_case('tnbc',h.READS);markers,matrices=h.load_matrices('tnbc',meta,h.READS)
    fig=plt.figure(figsize=(h.W/25.4,h.H/25.4),facecolor='white',dpi=160)
    accepted.build_a(fig)
    icon_axes=[a for a in fig.axes if len(a.images)]
    assert len(icon_axes)==1
    icon_artist=icon_axes[0].images[0]
    old_shape=icon_artist.get_array().shape
    replacement=plt.imread(h.track(HERE/'assets/ramp_full_figure1.png'))
    assert abs(old_shape[1]/old_shape[0]-replacement.shape[1]/replacement.shape[0])<1e-12
    icon_artist.set_data(replacement)
    h.build_b(fig,meta,coords,markers,matrices);build_c(fig,cells);build_d(fig)
    prior_ridge=pd.read_csv(h.track(MAIN/'text/ridge_display_audit.csv'))
    pd.testing.assert_frame_equal(pd.DataFrame(h.AUDIT),prior_ridge,check_exact=False,rtol=1e-12,atol=1e-12)
    accepted.verify_layout(fig)
    for ext,dpi in [('png',420),('pdf',600),('svg',600)]:fig.savefig(HERE/f'figure_1_icon_updated.{ext}',dpi=dpi)
    panels={'A':(0,0,56,188),'B':(58,0,122,123),'C':(58,125,87,63),'D':(146,125,34,63),'CD':(58,125,122,63)}
    for label,(x,y,w,ht) in panels.items():
        folder=HERE/f'Panel {label}';folder.mkdir(exist_ok=True)
        bbox=Bbox.from_bounds(x/25.4,(h.H-y-ht)/25.4,w/25.4,ht/25.4)
        fig.savefig(folder/f'panel_{label.lower()}.png',dpi=420,bbox_inches=bbox,pad_inches=0)
    plt.close(fig)
    assert all(h.sha(p)==s for p,s in old_hash.items())
    assert all(h.sha(p)==s for p,s in h.READS.items())
    assert all(frozen.sha(p)==s for p,s in frozen.READS.items())
    (HERE/'manifest.json').write_text(json.dumps(dict(canvas_mm=[180,188],panel_c_mm=[87,63],
        artifact_layout='One wide upper CD4 H1 plot (Raw, Harmony-direct, CytoNorm, RAMP); lower CD8 Shiao h17 and CD14 H2 plots.',
        merged_cd4_example='Both comparison methods use exactly the same H1 cells, marker and Raw reference; H2 CD4 removed.',
        legend_celltypes=LEGEND_CELLS,legend_colors=LEGEND_COLORS,examples=EXAMPLES,
        raw_reference_highlights={'CD4':'CD4 T','CD8':'CD8 T','CD14':'Monocytes'},
        reference_annotation='Cell-type-colored label and arrow to the Raw population upper quartile; quartiles unchanged.',
        readability=dict(box_and_whisker_linewidth_pt=.28,median_linewidth_pt=.32,
            stroke_color='Cell-type hue, darkened to 82% for edges and 72% for medians',
            shared_ylim=[0,1.04],ticks=[0,.5,1],data_and_box_statistics_unchanged=True),
        minimum_cells_per_displayed_type=50,old_cd4_quartile_checks=checked,old_cd4_quartiles_unchanged=True,
        cd8_example='Shiao h17 CD8: 164 CD4 T, 120 CD8 T and 420 Macrophages. The unified legend lists five actual cell identities, not five compulsory groups in every sample.',
        celltype_scope='Bone-marrow H1/H2 display CD4 T, CD8 T, B cells and Monocytes; Shiao h17 displays CD4 T, CD8 T and Macrophages. Monocytes remain purple and Macrophages are gold. No Plasma is displayed or renamed. All original sample cells remain in the percentile reference.',
        fastmnn='Standard PCA-fastMNN reconstructed marker output, not d=NA. Full reconstruction diagnostics retained in the sibling audit directory.',
        panel_d='Original boxplot and 0-60 axis, with reconstructed fastMNN added; unchanged statistics for the other five methods.',
        missing_groups='Absent cell types are omitted, never imputed as zero-valued boxes or filled from another sample.',panel_a1_icon_replaced=True,panels_a2_b_c_d_unchanged=True,umap_refit=False,old_main_figure_unchanged=True,manuscript_replaced=False,
        all_source_hashes={**h.READS,**frozen.READS},prior_main_figure_sha256=old_hash),indent=2)+'\n')
    print(HERE/'figure_1_icon_updated.png',flush=True)

if __name__=='__main__':main()
