"""Three marker examples in the original two-row Panel C footprint."""
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
from matplotlib.cbook import boxplot_stats
from scipy.stats import rankdata

CELLS=['CD4 T','CD8 T','B cells']
COLORS=[h.BLUE,h.ORANGE,'#AAB5BB']
METHOD_MAP={'Raw':'Raw','Harmony-direct[source_scale]':'Harmony-direct',
    'Harmony-direct[fixed_arcsinh]':'Harmony-arcsinh','cyCombine':'cyCombine',
    'CytoNorm':'CytoNorm','RAMP':'RAMP','fastMNN reconstructed':'fastMNN'}
EXAMPLES=[dict(case='tnbc',unit='Meyer IMC|B12.02518',marker='CD8',
    rect=[65,135.5,78,20],methods=['Raw','Harmony-direct','cyCombine','CytoNorm','RAMP'],drivers=['Harmony-direct','cyCombine']),
    dict(case='bone_marrow',unit='BM CyTOF|H1',marker='CD4',
    rect=[65,164,33.5,18.5],methods=['Raw','CytoNorm','RAMP'],drivers=['CytoNorm']),
    dict(case='bone_marrow',unit='BM CyTOF|H2',marker='CD14',
    rect=[109.5,164,33.5,18.5],methods=['Raw','fastMNN','RAMP'],drivers=['fastMNN'])]

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
            b='B lineage' if case=='tnbc' else 'Mature / CODEX B'
            label_map={'CD4 T':'CD4 T','CD8 T':'CD8 T',b:'B cells'}
            lab=np.array([label_map.get(v,'not displayed') for v in native])
            col=markers.index(ex['marker'])
            assert min((lab==v).sum() for v in CELLS)>=50
            for method,z in arrays.items():
                # All sample cells define each method's percentile reference.
                values=z[ix,col];r=rankdata(values)/(len(ix)+1)
                for label,color in zip(CELLS,COLORS):
                    take=lab==label;v=r[take];ss=boxplot_stats(v,whis=1.5)[0]
                    stats.append(dict(case=case,unit_key=ex['unit'],marker=ex['marker'],method=method,celltype=label,
                        n_cells=len(v),mean_percentile=float(v.mean()),q1=ss['q1'],median=ss['med'],q3=ss['q3'],
                        whislo=ss['whislo'],whishi=ss['whishi'],displayed=method in ex['methods']))
                    for cid,val,pct in zip(meta.cell_id.to_numpy()[ix][take],values[take],v):
                        cell_rows.append(dict(case=case,unit_key=ex['unit'],marker=ex['marker'],method=method,
                            celltype=label,cell_id=cid,value=val,percentile=pct,displayed=method in ex['methods']))
                for positive,negative in [('CD4 T','CD8 T'),('CD4 T','B cells'),('CD8 T','B cells')]:
                    aa=values[lab==positive];bb=values[lab==negative];rr=rankdata(np.r_[aa,bb])
                    auc=(rr[:len(aa)].sum()-len(aa)*(len(aa)+1)/2)/(len(aa)*len(bb))
                    all_auc.append(dict(case=case,unit_key=ex['unit'],marker=ex['marker'],method=method,
                        positive=positive,negative=negative,auc=auc))
    cells=pd.DataFrame(cell_rows);s=pd.DataFrame(stats)
    cells.to_csv(HERE/'artifact_cells.csv.gz',index=False)
    s.to_csv(HERE/'artifact_box_statistics.csv',index=False)
    pd.DataFrame(all_auc).to_csv(HERE/'artifact_auc.csv',index=False)
    # Original CytoNorm/CD4 example is numerically unchanged despite compacting.
    old=pd.read_csv(h.track(MAIN/'redesign_full_icon_20260914/all_celltype_statistics.csv'))
    checked=0
    for row in old[old.benchmark.eq('bone_marrow')&old.marker.eq('CD4')&old.celltype.isin(['CD4 T','CD8 T','Mature / CODEX B'])].itertuples():
        label='B cells' if row.celltype=='Mature / CODEX B' else row.celltype
        new=s[s.case.eq('bone_marrow')&s.marker.eq('CD4')&s.method.eq(row.method)&s.celltype.eq(label)].iloc[0]
        for q in ['q1','median','q3']:
            assert abs(new[q]-getattr(row,q))<1e-12
            checked+=1
    assert checked>0
    return cells,s,checked

def build_c(fig,cells):
    h.panel_title(fig,'C','Artifacts in comparison methods',59,128)
    # One shared legend: identical cell identities/colors in all three subplots.
    for x,label,color in zip([103,116,129],CELLS,COLORS):h.mini_legend(fig,x,132.9,[label],[color],0,size=5.5)
    for ex in EXAMPLES:
        x,y,w,ht=ex['rect'];methods=ex['methods']
        h.text(fig,x,y-2.6,ex['marker'],size=6.2,bold=True)
        ax=h.statistical_axes(fig,x,y,w,ht)
        sub=cells[cells.case.eq(ex['case'])&cells.unit_key.eq(ex['unit'])&cells.marker.eq(ex['marker'])]
        for driver in ex['drivers']:
            i=methods.index(driver);ax.axvspan(i-.46,i+.46,color='#FBECE6',zorder=0)
        for i,method in enumerate(methods):
            g=sub[sub.method.eq(method)]
            arrays=[g[g.celltype.eq(label)].percentile.to_numpy() for label in CELLS]
            b=ax.boxplot(arrays,positions=i+np.array([-.28,0,.28]),widths=.22,
                patch_artist=True,showfliers=False,manage_ticks=False,
                medianprops={'color':h.INK,'linewidth':.6},whiskerprops={'color':h.LINE,'linewidth':.45},
                capprops={'color':h.LINE,'linewidth':.45},boxprops={'edgecolor':h.LINE,'linewidth':.45})
            for patch,color in zip(b['boxes'],COLORS):patch.set_facecolor(color)
        ax.set_xlim(-.5,len(methods)-.5);ax.set_ylim(0,1);ax.set_yticks([0,.5,1])
        ax.set_ylabel(ex['marker']+' percentile',fontsize=5.8,labelpad=2)
        ax.set_xticks(range(len(methods)),['Harmony-\ndirect' if m=='Harmony-direct' else m for m in methods])
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
    cells,stats,checked=prepare()
    meta,coords=h.load_case('tnbc',h.READS);markers,matrices=h.load_matrices('tnbc',meta,h.READS)
    fig=plt.figure(figsize=(h.W/25.4,h.H/25.4),facecolor='white',dpi=160)
    accepted.build_a(fig);h.build_b(fig,meta,coords,markers,matrices);build_c(fig,cells);build_d(fig)
    prior_ridge=pd.read_csv(h.track(MAIN/'text/ridge_display_audit.csv'))
    pd.testing.assert_frame_equal(pd.DataFrame(h.AUDIT),prior_ridge,check_exact=False,rtol=1e-12,atol=1e-12)
    accepted.verify_layout(fig)
    for ext,dpi in [('png',420),('pdf',600),('svg',600)]:fig.savefig(HERE/f'figure_1_compact_artifacts.{ext}',dpi=dpi)
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
        artifact_layout='Exactly two rows: CD8 full width; CD4 and CD14 side by side in the second row.',
        common_celltypes=CELLS,common_colors=COLORS,examples=EXAMPLES,
        minimum_cells_per_displayed_type=50,old_cd4_quartile_checks=checked,old_cd4_quartiles_unchanged=True,
        first_example_replaced='Previous B12.64192 contains only one B cell. New B12.02518 contains 404 CD4 T, 389 CD8 T and 286 B cells.',
        cd14_scope='Only the three requested expected-low lymphoid groups are plotted. The separate monocyte-vs-B AUROC is not labelled on this plot.',
        fastmnn='Standard PCA-fastMNN reconstructed marker output, not d=NA. Full reconstruction diagnostics retained in the sibling audit directory.',
        panel_d='Original boxplot and 0-60 axis, with reconstructed fastMNN added; unchanged statistics for the other five methods.',
        panels_a_b_unchanged=True,umap_refit=False,old_main_figure_unchanged=True,manuscript_replaced=False,
        all_source_hashes={**h.READS,**frozen.READS},prior_main_figure_sha256=old_hash),indent=2)+'\n')
    print(HERE/'figure_1_compact_artifacts.png',flush=True)

if __name__=='__main__':main()
