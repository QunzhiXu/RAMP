#!/usr/bin/env python3
"""All selected cells as points, pooled boxes, and donor means; layout-only trial."""
from pathlib import Path
import hashlib,json,os
HERE=Path(__file__).resolve().parent
TRIAL=HERE.parent;BIG=TRIAL.parent;ROOT=BIG.parent
os.environ.setdefault('MPLCONFIGDIR',str(HERE/'.cache/matplotlib'))
os.environ.setdefault('MPLBACKEND','Agg')
import fitz
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import rankdata

SOURCE=ROOT/'program_return_exploration_20260913/output/source_features_and_programs.csv.gz'
OLD_DONORS=BIG/'panel_E/group_fraction_scan_20260910/donor_contrasts.csv'
FORMAL_PAIR=ROOT/'biomarker_discovery/screen/rahim_tumor_cd8_biomarker_programs.csv.gz'
INPUTS=[SOURCE,OLD_DONORS,FORMAL_PAIR]
LOW,HIGH,INK,GRID='#2B6CB0','#C9433C','#182027','#E7EBEE'
DONORS=['126','indiv_1','indiv_2']
FEATURES={'E':[('LAG3','LAG-3','CD223--LAG-3'),('CD127','CD127','CD127--IL-7Ra'),('CD62L','CD62L','CD62L')],
          'F':[('CTLA4','CTLA4','CTLA4'),('CXCL13','CXCL13','CXCL13'),('IL7R','IL7R','IL7R')]}
WIDTHS={'A':8.1,'B':4.2,'C':8.1,'D':4.65,'E':4.2,'F':4.65}
HEIGHT=3.45


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def prepare():
    d=pd.read_csv(SOURCE,dtype={'cell_uid':str,'donor':str})
    formal=pd.read_csv(FORMAL_PAIR,dtype={'cell_uid':str,'donor':str}).set_index('cell_uid').loc[d.cell_uid]
    np.testing.assert_array_equal(formal.donor.to_numpy(),d.donor.to_numpy())
    np.testing.assert_allclose(formal.compact_top5pct_clr_program,d.pair,atol=1e-14,rtol=0)
    assert len(d)==4788 and set(d.donor)==set(DONORS)
    d['state_rank']=d.groupby('donor').pair.transform(lambda x:(rankdata(x)-.5)/len(x))
    d['group']=np.select([d.state_rank<=.30,d.state_rank>=.70],['Lower','Higher'],default='Middle')
    assert d.group.value_counts().to_dict()=={'Middle':1912,'Lower':1439,'Higher':1437}
    # Protein axes show the original expression direction, not signed state score.
    for key,_,_ in FEATURES['E']:
        d[key+'_display']=d.groupby('donor')[key+'_measured'].transform(lambda x:100*(rankdata(x)-.5)/len(x))
    for key,_,_ in FEATURES['F']:d[key+'_display']=d[key+'_measured']
    d[['cell_uid','donor','pair','state_rank','group']].to_csv(HERE/'group_assignments.csv.gz',index=False)
    d.groupby(['donor','group']).size().rename('cells').to_csv(HERE/'group_counts.csv')
    return d


def build_panel(d,label,old):
    protein=label=='E';width=WIDTHS[label]
    fig=plt.figure(figsize=(width,HEIGHT),facecolor='white')
    fig.text(.08/width,.976,label,fontsize=23,weight='bold',va='top',color=INK)
    fig.text(.51/width,.947,'PROTEIN PHENOTYPE' if protein else 'RNA PHENOTYPE',
             fontsize=12.,weight='bold',va='top',color=INK)
    fig.legend(handles=[Line2D([],[],marker='o',linestyle='',markersize=4.2,color=LOW,label='Lower 30%'),
                        Line2D([],[],marker='o',linestyle='',markersize=4.2,color=HIGH,label='Higher 30%'),
                        Line2D([],[],marker='D',linestyle='',markersize=4.,markerfacecolor=INK,markeredgecolor='white',
                               markeredgewidth=.45,label='Donor mean')],
               loc='center',bbox_to_anchor=(.55,.839),ncol=3,frameon=False,fontsize=8.6,
               handlelength=.7,handletextpad=.35,columnspacing=.9)
    ax=fig.add_axes([.155,.160,.814,.575])
    cell_rows,mean_rows,box_rows=[],[],[]
    point_count=0
    for j,(key,name,raw_name) in enumerate(FEATURES[label]):
        for g,dx,color in [('Lower',-.215,LOW),('Higher',.215,HIGH)]:
            b=d[d.group.eq(g)].copy();v=b[key+'_display'].to_numpy(float);x=j+dx
            assert len(v)==(1439 if g=='Lower' else 1437) and np.isfinite(v).all()
            assert v.min()>=0
            # A faint density silhouette and every measured cell at its true y value.
            vio=ax.violinplot([v],positions=[x],widths=.39,showmeans=False,showextrema=False,
                              points=180,bw_method=.22)
            for body in vio['bodies']:
                body.set_facecolor(color);body.set_edgecolor(color);body.set_linewidth(.45);body.set_alpha(.09)
                body.set_zorder(1)
            jitter=np.random.default_rng(20260913+100*j+(0 if g=='Lower' else 1)).uniform(-.095,.095,len(v))
            points=ax.scatter(x+jitter,v,s=2.0,color=color,alpha=.13,linewidths=0,rasterized=True,zorder=2)
            np.testing.assert_array_equal(np.asarray(points.get_offsets())[:,1],v)
            point_count+=len(v)
            ax.boxplot([v],positions=[x],widths=.14,patch_artist=True,manage_ticks=False,
                       showfliers=False,whis=1.5,
                       boxprops=dict(facecolor='white',edgecolor=color,linewidth=.9,alpha=.85),
                       medianprops=dict(color=color,linewidth=1.2),
                       whiskerprops=dict(color=color,linewidth=.8),capprops=dict(color=color,linewidth=.8),zorder=3)
            means=b.groupby('donor')[key+'_display'].mean().loc[DONORS]
            ax.scatter(x+np.array([-.037,0,.037]),means.to_numpy(),s=18,marker='D',facecolor=INK,
                       edgecolor='white',linewidth=.5,zorder=5)
            quart=np.quantile(v,[.25,.5,.75]);iqr=quart[2]-quart[0]
            box_rows.append(dict(panel=label,marker=key,group=g,cells=len(v),q1=quart[0],median=quart[1],q3=quart[2],
                minimum=v.min(),maximum=v.max(),whisker_low=v[v>=quart[0]-1.5*iqr].min(),
                whisker_high=v[v<=quart[2]+1.5*iqr].max(),pooled_cell_mean=v.mean(),equal_donor_mean=means.mean()))
            for donor,mean in means.items():
                sub=b[b.donor.eq(donor)]
                ref=old[old.donor.eq(donor)&old.marker.eq(raw_name)].iloc[0]
                previous=ref['low' if g=='Lower' else 'high']
                old_quantity=mean if protein else 100*(sub[key+'_measured']>0).mean()
                np.testing.assert_allclose(old_quantity,previous,atol=1e-10,rtol=0)
                mean_rows.append(dict(panel=label,marker=key,donor=donor,group=g,cells=len(sub),mean=mean,
                    original_panel_quantity=old_quantity,original_quantity_matches=True,
                    rna_detection_fraction=np.nan if protein else (sub[key+'_measured']>0).mean()))
            for uid,donor,value in zip(b.cell_uid,b.donor,v):
                cell_rows.append(dict(panel=label,marker=key,cell_uid=uid,donor=donor,group=g,value=value))
    assert point_count==8628
    ax.set_xlim(-.65,2.65);ax.set_xticks([0,1,2],[f[1] for f in FEATURES[label]])
    if protein:
        ax.set(ylim=(-4,104),yticks=[0,25,50,75,100]);ax.set_ylabel('Expression percentile',fontsize=9.5,labelpad=5)
    else:
        maximum=d[d.group.ne('Middle')][[f[0]+'_display' for f in FEATURES[label]]].to_numpy().max()
        ax.set(ylim=(-.24,max(6.3,maximum+.20)),yticks=[0,2,4,6]);ax.set_ylabel('RNA expression',fontsize=9.5,labelpad=5)
        for tick in ax.get_xticklabels():tick.set_fontstyle('italic')
    ax.tick_params(axis='x',labelsize=10.2,length=0,pad=7)
    ax.tick_params(axis='y',labelsize=8.6,length=3,pad=3)
    ax.yaxis.grid(True,color=GRID,lw=.6);ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
    for side in ['left','bottom']:ax.spines[side].set_color('#7A838A');ax.spines[side].set_linewidth(.8)
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    items=list(fig.texts)+list(ax.texts)+[ax.xaxis.label,ax.yaxis.label]+ax.get_xticklabels()+ax.get_yticklabels()
    for legend in fig.legends:items+=legend.get_texts()
    for t in items:
        if t.get_visible() and t.get_text():
            bb=t.get_window_extent(renderer)
            assert bb.x0>=-1 and bb.y0>=-1 and bb.x1<=fig.bbox.width+1 and bb.y1<=fig.bbox.height+1,(label,t.get_text())
    folder=HERE/f'panel_{label}';folder.mkdir(exist_ok=True)
    stem=folder/f'panel_{label}_cell_distributions'
    for ext in ['png','pdf','svg']:fig.savefig(stem.with_suffix('.'+ext),dpi=320,facecolor='white')
    plt.close(fig)
    pd.DataFrame(cell_rows).to_csv(folder/'plotted_cell_values.csv.gz',index=False)
    pd.DataFrame(mean_rows).to_csv(folder/'donor_means.csv',index=False)
    pd.DataFrame(box_rows).to_csv(folder/'pooled_cell_box_summaries.csv',index=False)
    return stem.with_suffix('.pdf')


def compose(e,f):
    paths={'A':TRIAL/'program_return_trial/panel_A/panel_A_niche_discovery.pdf',
           'B':BIG/'panel_B/panel_B_survival_compact.pdf',
           'C':BIG/'panel_C/panel_C_alignment_compact.pdf',
           'D':TRIAL/'horizontal_E_spatial_left/panel_D/panel_D_donor_boxes.pdf','E':e,'F':f}
    hashes={k:sha(p) for k,p in paths.items()}
    doc=fitz.open();page=doc.new_page(width=16.99*72,height=6.98*72)
    slots={}
    for row,labels in enumerate([['A','B','D'],['C','E','F']]):
        x=0.;y=row*(HEIGHT+.08)
        for label in labels:
            rect=fitz.Rect(x*72,y*72,(x+WIDTHS[label])*72,(y+HEIGHT)*72)
            with fitz.open(paths[label]) as src:
                np.testing.assert_allclose([rect.width,rect.height],[src[0].rect.width,src[0].rect.height],atol=.01)
                page.show_pdf_page(rect,src,0)
            slots[label]=dict(path=str(paths[label]),bounds_inches=[v/72 for v in rect],sha256=hashes[label])
            x+=WIDTHS[label]+.02
    doc.set_metadata({'title':'HNSCC trial: full cell distributions and donor means; no spatial return panel'})
    doc.save(HERE/'hnscc_cell_distribution_trial.pdf',deflate=True,garbage=4)
    page.get_pixmap(dpi=180,alpha=False).save(HERE/'hnscc_cell_distribution_trial.png')
    page.get_pixmap(dpi=230,alpha=False,clip=fitz.Rect(8.12*72,3.53*72,16.99*72,6.98*72)).save(HERE/'protein_rna_preview.png')
    doc.close()
    assert hashes=={k:sha(p) for k,p in paths.items()}
    return slots


def main():
    hashes={str(p):sha(p) for p in INPUTS}
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','text.color':INK})
    d=prepare();old=pd.read_csv(OLD_DONORS,dtype={'donor':str});old=old[old.fraction.eq(.30)]
    e=build_panel(d,'E',old);f=build_panel(d,'F',old)
    slots=compose(e,f)
    assert hashes=={str(p):sha(p) for p in INPUTS}
    audit=dict(status='PASS',inputs_sha256=hashes,inputs_unchanged=True,placements=slots,
        source_cells=4788,lower_cells=1439,higher_cells=1437,middle_cells_not_in_original_tail_comparison=1912,
        markers=6,cell_points_per_panel=8628,total_cell_marker_points=17256,donor_mean_points=36,
        grouping='Original measured CD39/TIGIT within-donor midrank <=0.30 and >=0.70',
        displayed_protein='Within-donor expression percentile over all 4788 source cells, then subset tails',
        displayed_rna='Provider continuous expression from raw.X; unchanged zero values',
        boxes='Pooled individual cells: quartiles, median, 1.5 IQR whiskers; all cell values shown as points',
        point_jitter='Horizontal only; deterministic; no jitter to expression values',
        violin='Pooled cell KDE at bw_method=0.22; support restricted to observed min/max',
        donor_means='Three original source donor means per group and marker; dark diamonds',
        original_36_donor_group_quantities_verified=True,cell_level_inferential_tests=False,
        donor_equal_box_weighting=False,full_figure_inches=[16.99,6.98],
        old_spatial_F_removed_in_this_trial=True,new_F='RNA phenotype',formal_figure_replaced=False,manuscript_replaced=False)
    (HERE/'PLOT_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
    qa=HERE/'qa';qa.mkdir(exist_ok=True)
    for letter,path in [('E',e),('F',f)]:
        with fitz.open(path) as pdf:pdf[0].get_pixmap(dpi=220,alpha=False).save(qa/f'panel_{letter}_pdf.png')
    print(json.dumps({'status':'PASS','cell_marker_points':17256,'donor_mean_points':36,'figure_size':[16.99,6.98]}))


if __name__=='__main__':main()
