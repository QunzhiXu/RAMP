"""Independent Figure 3 layout preview. Reads frozen endpoints; never fits a model."""
from pathlib import Path
import os, json, hashlib, shutil
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
os.environ['MPLCONFIGDIR']=str(HERE/'tmp/mplconfig')
os.environ['MPLBACKEND']='Agg'
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb

SRC=ROOT/'cross_platform_trainability_v2_20260919/Figure/Combined ABC'
MANUSCRIPT=ROOT.parent/'bioarxiv_Ver2'
SOURCE_ICON=ROOT/'cross_platform_trainability_v2_20260919/Figure/Panel A/assets/ramp_sign_no_pipes.png'
METHODS=['ramp','unaligned','harmony_direct','cycombine','cytonorm','fastmnn','maxfuse','clipped_zscore','minmax','percentile']
LABELS=['RAMP','Unaligned','Harmony','cyCombine','CytoNorm','fastMNN','MaxFuse','Clipped z-score','Min–max','Percentile']
COLORS=['#007C78','#656565','#A4A44D','#4C8DAE','#C77638','#5266AB','#AF5580','#A07617','#AB7474','#8557A4']
TASKS=['bone_marrow','tnbc','melanoma','lung_cancer','lung_story','melanoma_story','hnscc_story']
INK='#202C32'; EDGE='#B9C3C9'; BLUE='#347CAB'; ORANGE='#DE913D'; ARROW='#687780'
W,H=6.4,5.0
plt.rcParams.update({'font.family':'Arial','font.size':8,'font.weight':'bold',
 'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,
 'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    tracked=[MANUSCRIPT/'main.tex',MANUSCRIPT/'main.pdf',MANUSCRIPT/'figure/figure3_cross_platform_learning.pdf']
    before={str(p):sha(p) for p in tracked}
    fig=plt.figure(figsize=(W,H),facecolor='white')
    ax=fig.add_axes([0,0,1,1]); ax.set(xlim=(0,W),ylim=(H,0),aspect='equal');ax.axis('off')
    text_records=[]
    def text(x,y,s,size=8,ha='center',color=INK,**kw):
        t=ax.text(x,y,s,fontsize=size,fontweight='bold',ha=ha,va='center',
           color=color,linespacing=1.08,zorder=9,**kw);text_records.append(t);return t
    def line(x,y,**kw):return ax.plot(x,y,color=kw.pop('color',ARROW),lw=kw.pop('lw',.72),zorder=3,**kw)
    def box(x,y,w,h):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=0.045',
          facecolor='white',edgecolor=EDGE,lw=.7,zorder=2))
    def arrow(a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=7,
          shrinkA=0,shrinkB=0,color=ARROW,lw=.8,zorder=4))
    def dot(x,y,r,c):ax.add_patch(Circle((x,y),r,facecolor=c,edgecolor='white',lw=.28,zorder=6))
    def heat(x,y,c,vals,dx=.063,dy=.059):
        for i,row in enumerate(vals):
            for j,v in enumerate(row):
                ax.add_patch(Rectangle((x+j*dx,y+i*dy),dx-.007,dy-.007,
                    fc=1-v*(1-np.array(to_rgb(c))),ec='none',zorder=4))
    # A: vertically arranged workflow at final publication size.
    text(.035,.19,'A',11.5,ha='left')
    text(.25,.23,'Cross-platform\nlearning',10.1,ha='left')
    left,width=.07,1.40; mid=left+width/2
    box(left,.60,width,.92)
    text(mid,.75,'Cross-platform data',8.2)
    text(.43,.95,'Source(s)',8,color=BLUE);text(1.115,.95,'Target',8,color=ORANGE)
    vals=np.array([[.25,.68,.92,.32],[.8,.2,.52,.77],[.62,.87,.23,.48],[.36,.52,.78,.92],[.91,.33,.64,.21]])
    heat(.31,1.055,BLUE,vals);heat(.995,1.055,ORANGE,np.roll(vals,1,axis=0)*.85)
    text(mid,1.425,'Unlabeled expression',7.9)
    arrow((mid,1.525),(mid,1.685))
    box(left,1.70,width,.85)
    text(mid,1.81,'Preprocessing',8.6)
    icon=HERE/'assets/ramp_sign_no_pipes.png';shutil.copy2(SOURCE_ICON,icon)
    logo=plt.imread(icon); iw=.80; ih=iw*logo.shape[0]/logo.shape[1]
    ax.imshow(logo,extent=(mid-iw/2,mid+iw/2,1.92+ih,1.92),zorder=4)
    text(mid,2.46,'vs. 9 control pipelines',7.6)
    arrow((mid,2.555),(mid,2.655))
    box(left,2.67,width,.86)
    text(mid,2.805,'Model training',8.6)
    # Linear fit, kNN and NN icons drawn in vector form.
    line([.19,.19,.50],[2.945,3.25,3.25],lw=.6)
    line([.225,.48],[3.20,2.965],color=BLUE,lw=1)
    for x,y in [(.23,3.15),(.29,3.18),(.34,3.06),(.41,3.07),(.465,2.995)]:dot(x,y,.019,BLUE)
    text(.35,3.395,'Linear',7.8)
    ax.add_patch(Circle((.77,3.10),.108,fc='#F2F7FA',ec=BLUE,ls=(0,(2,2)),lw=.6,zorder=3))
    for x,y in [(.70,3.065),(.81,3.05),(.78,3.17),(.625,3.0),(.9,3.02),(.65,3.235),(.88,3.25)]:dot(x,y,.021,BLUE)
    dot(.77,3.10,.031,ORANGE);text(.77,3.395,'kNN',7.8)
    layers=[[(1.05,y) for y in [2.98,3.10,3.22]],[(1.185,y) for y in [2.96,3.045,3.13,3.215]],[(1.32,y) for y in [3.02,3.18]]]
    for a,b in zip(layers[:-1],layers[1:]):
        for p in a:
            for q in b:line([p[0],q[0]],[p[1],q[1]],color='#BAC6CE',lw=.4)
    for k,layer in enumerate(layers):
        for x,y in layer:dot(x,y,.023,ORANGE if k==2 else BLUE)
    text(1.185,3.395,'Neural\nnetwork',7.4)
    # Branch explicitly to both tasks; no task-to-task arrow.
    line([mid,mid,.12,.12],[3.535,3.69,3.69,4.64],lw=.8)
    text(.80,3.725,'Apply to target cells',7.9,bbox=dict(fc='white',ec='none',pad=1.6))
    for y in (3.98,4.46):
        box(.25,y,1.22,.40);arrow((.12,y+.2),(.24,y+.2))
    for j,c in enumerate([BLUE,'#A873A3',ORANGE]):
        for i in range(2):
            cx,cy=.355+j*.085,4.115+i*.105;dot(cx,cy,.036,c)
            ax.add_patch(Circle((cx+.005,cy-.004),.011,fc='white',ec='none',alpha=.75,zorder=7))
    text(1.00,4.18,'Cell-type\nannotation',8.1)
    # Source marker A and target predicted marker A; not paired cells.
    text(.365,4.545,'A',6.6,color=BLUE);text(.57,4.545,'A',6.6,color=ORANGE)
    heat(.337,4.595,BLUE,np.array([[.25],[.78],[.45],[.93]]),dx=.056,dy=.039)
    heat(.542,4.595,ORANGE,np.array([[.76],[.25],[.6],[.91],[.42]]),dx=.056,dy=.031)
    ax.add_patch(Rectangle((.534,4.585),.067,.17,fc='none',ec=ORANGE,ls=(0,(2,2)),lw=.55,zorder=5))
    arrow((.405,4.67),(.518,4.67));text(1.035,4.66,'Missing-marker\nprediction',7.9)

    datasets={}; scores=[]
    configs={
      'B':dict(kind='J1',heading='Cell-type annotation',n=7,ty=.18,top=.69,bottom=2.20,
        metrics=['balanced_accuracy','macro_f1'],titles=['Balanced accuracy ↑','Macro-F1 ↑'],
        limits=[(.2,1),(.1,1)],ticks=[[.2,.4,.6,.8,1],[.1,.3,.5,.7,.9,1]]),
      'C':dict(kind='J2',heading='Missing-marker prediction',n=6,ty=2.58,top=3.08,bottom=4.59,
        metrics=['effective_spearman','mae'],titles=['Spearman ρ ↑','Relative-expression MAE ↓'],
        limits=[(.1,.55),(.15,.45)],ticks=[[.1,.2,.3,.4,.5],[.15,.25,.35,.45]])}
    axes=[]
    for letter,cfg in configs.items():
        source=SRC/f'Panel {letter}/per_benchmark_scores.csv'
        shutil.copy2(source,HERE/f'data/panel_{letter.lower()}_per_benchmark_scores.csv')
        frame=pd.read_csv(source);datasets[letter]={'path':str(source),'sha256':sha(source)}
        assert set(frame.method)==set(METHODS)
        task_order=[t for t in TASKS if t in set(frame.task)]
        assert len(task_order)==cfg['n']
        offsets=dict(zip(task_order,np.linspace(-.16,.16,len(task_order))))
        text(1.655,cfg['ty'],letter,11.5,ha='left')
        text(1.875,cfg['ty'],cfg['heading'],10.1,ha='left')
        text(6.25,cfg['ty'],f"{cfg['n']} benchmarks",7.7,ha='right')
        for col,(metric,title,limits,ticks) in enumerate(zip(cfg['metrics'],cfg['titles'],cfg['limits'],cfg['ticks'])):
            x=[2.49,4.49][col]; w=[1.73,1.76][col]
            a=fig.add_axes([x/W,(H-cfg['bottom'])/H,w/W,(cfg['bottom']-cfg['top'])/H]);axes.append(a)
            a.set_title(title,fontsize=8.1,fontweight='bold',pad=8,color=INK)
            a.axhspan(-.45,.45,facecolor=COLORS[0],alpha=.055,zorder=0)
            for j,(method,color) in enumerate(zip(METHODS,COLORS)):
                d=frame[(frame.method==method)&(frame.metric==metric)].set_index('task').loc[task_order]
                values=d.score.to_numpy();assert len(values)==cfg['n']
                assert np.all((values>=limits[0])&(values<=limits[1]))
                bp=a.boxplot([values],positions=[j],orientation='horizontal',widths=.57,patch_artist=True,
                  whis=1.5,showfliers=False,manage_ticks=False,
                  boxprops=dict(fc=color,ec=color,lw=.75),medianprops=dict(color=color,lw=1.2),
                  whiskerprops=dict(color=color,lw=.7),capprops=dict(color=color,lw=.7))
                bp['boxes'][0].set_facecolor((*to_rgb(color),.15))
                a.scatter(values,[j+offsets[t] for t in task_order],s=10.8,fc=color,ec='white',lw=.24,alpha=.84,zorder=4)
                a.scatter([values.mean()],[j],s=21,marker='D',fc='white',ec=INK,lw=.72,zorder=6)
                scores.append({'panel':letter,'metric':metric,'method':method,'mean':float(values.mean()),'n':len(values)})
            a.set_xlim(*limits);a.set_xticks(ticks);a.set_xticklabels([f'{t:g}' for t in ticks],fontsize=8,fontweight='bold')
            a.set_ylim(9.55,-.55);a.set_yticks(range(10))
            a.set_yticklabels(LABELS if col==0 else [],fontsize=8,fontweight='bold')
            if col==0:a.get_yticklabels()[0].set_color(COLORS[0])
            a.tick_params(axis='y',length=0,pad=5);a.tick_params(axis='x',length=2.5,pad=3,color=ARROW)
            a.set_axisbelow(True);a.grid(axis='x',color='#E6EBEE',lw=.5)
            for sp in ['top','right','left']:a.spines[sp].set_visible(False)
            a.spines['bottom'].set_color('#718089');a.spines['bottom'].set_linewidth(.7)
    handles=[Line2D([],[],ls='none',marker='o',color='#718089',markersize=3.5,label='Benchmark (3-model mean)'),
             Line2D([],[],ls='none',marker='D',mfc='white',mec=INK,markersize=4,label='Overall mean')]
    legend=fig.legend(handles=handles,loc='center',bbox_to_anchor=(4.27/W,.14/H),ncol=2,frameon=False,
       prop={'size':7.6,'weight':'bold'},handletextpad=.45,columnspacing=1.1,borderaxespad=0)
    fig.canvas.draw()
    ren=fig.canvas.get_renderer()
    # Check all nonempty text stays inside figure (icons contain only a few smaller letters).
    bounds=fig.bbox
    for txt in fig.findobj(plt.Text):
        if not txt.get_text() or not txt.get_visible():continue
        bb=txt.get_window_extent(ren)
        assert bb.x0>=bounds.x0-.5 and bb.y0>=bounds.y0-.5 and bb.x1<=bounds.x1+.5 and bb.y1<=bounds.y1+.5,(txt.get_text(),bb)
    out=HERE/'output/pdf/figure3_readability_preview.pdf'
    fig.savefig(out,dpi=600)
    fig.savefig(HERE/'output/figure3_readability_preview.png',dpi=260)
    fig.savefig(HERE/'output/figure3_readability_preview.svg')
    pd.DataFrame(scores).to_csv(HERE/'data/overall_means.csv',index=False)
    after={str(p):sha(p) for p in tracked};assert before==after
    proof={'status':'preview_only_not_deployed','canvas_inches':[W,H],'layout':'left vertical workflow; right B above C; two axes per results panel',
      'tick_font_pt':8,'method_label_font_pt':8,'source_files':datasets,'manuscript_hashes_before_and_after':after,
      'all_methods_and_benchmarks_retained':True,'score_aggregation':'unchanged frozen benchmark mean across the three model families',
      'axis_limits_unchanged':True,'no_refitting':True,'icon_sha256':sha(icon),'output_pdf':str(out),'output_pdf_sha256':sha(out)}
    (HERE/'PROVENANCE.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(proof,indent=2))
if __name__=='__main__':main()
