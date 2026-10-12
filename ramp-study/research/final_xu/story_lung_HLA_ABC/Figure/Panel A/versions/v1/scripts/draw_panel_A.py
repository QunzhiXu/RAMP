"""Vector workflow schematic for the lung vCAF HLA-ABC story."""
from runtime import BASE,STORY
import json,hashlib,datetime
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, PathPatch
from matplotlib.path import Path as MPath

W,H=16.0,8.2
C={'ink':'#243745','muted':'#607582','line':'#a9b7bf','soft':'#eef2f4',
   'senosain':'#487c9d','senosain_bg':'#f0f5f8','zheng':'#368f83','zheng_bg':'#eff7f4',
   'cords':'#c35b75','cords_bg':'#fcf2f5','white':'#ffffff','model':'#f3f5f7'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'svg.fonttype':'none','savefig.facecolor':'white'})
fig,ax=plt.subplots(figsize=(W,H))
fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
ax.set(xlim=(0,W),ylim=(H,0));ax.axis('off')

def txt(x,y,s,size=12,color=None,weight='normal',ha='left',va='center',**kw):
    return ax.text(x,y,s,fontsize=size,color=color or C['ink'],fontweight=weight,ha=ha,va=va,linespacing=1.35,**kw)
def box(x,y,w,h,fill='white',edge=None,r=.14,lw=1.1):
    p=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',facecolor=fill,edgecolor=edge or C['line'],linewidth=lw,zorder=1)
    ax.add_patch(p);return p
def arrow(a,b,color=None,lw=1.55,style='-',curve=0):
    p=FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,linewidth=lw,color=color or C['muted'],linestyle=style,connectionstyle=f'arc3,rad={curve}',shrinkA=0,shrinkB=0,zorder=2)
    ax.add_patch(p);return p
def line(points,color=None,lw=1.45,ls='-'):
    pts=np.array(points);ax.plot(pts[:,0],pts[:,1],color=color or C['line'],lw=lw,ls=ls,solid_capstyle='round',zorder=2)
def cells(cx,cy,color,spatial=False):
    points=[(-.27,-.19),(.04,-.23),(.28,-.08),(-.22,.14),(.11,.13),(.34,.27),(-.08,.36)]
    if spatial:box(cx-.48,cy-.44,.99,1.00,'white',edge=color,r=.05,lw=.8)
    for i,(dx,dy) in enumerate(points):
        ax.add_patch(Circle((cx+dx,cy+dy),.085 if i%2 else .10,facecolor=color,alpha=.40+.075*(i%4),edgecolor='none',zorder=3))
def fibro(cx,cy,scale=.25,color=None):
    coords=np.array([[-1.0,-.25],[-.28,-.20],[-.05,-.82],[.15,-.22],[.94,-.52],[.43,.02],[1.04,.57],[.20,.33],[-.13,.78],[-.25,.26],[-.97,.48],[-.55,.06],[-1.0,-.25]])*scale+[cx,cy]
    ax.add_patch(PathPatch(MPath(coords,[MPath.MOVETO]+[MPath.LINETO]*11+[MPath.CLOSEPOLY]),facecolor=color or C['cords'],edgecolor='none',alpha=.8,zorder=3))
    ax.add_patch(Circle((cx,cy),scale*.17,facecolor='white',edgecolor='none',zorder=4))

# Panel identifier and reading order.
txt(.34,.40,'A',27,weight='bold')
txt(.91,.43,'Cross-platform transfer of HLA-ABC states',20,weight='bold')
for x,s in [(.52,'LUNG CANCER COHORTS'),(4.60,'ALIGN & TRANSFER'),(9.10,'VALIDATE & INTERPRET')]:
    txt(x,1.00,s,11.2,color=C['muted'],weight='bold')

# Source cohorts: markers are readouts, not model feature lists.
input_cards=[('Senosain','CyTOF','Measured HLA-ABC','senosain',1.42,False),
             ('Zheng','mIF','Measured HLA-A','zheng',3.20,True),
             ('Cords','IMC','HLA-ABC to infer','cords',4.98,True)]
for name,platform,readout,key,y,spatial in input_cards:
    box(.52,y,3.35,1.40,C[key+'_bg'],edge=C[key],lw=1.2)
    cells(1.13,y+.67,C[key],spatial)
    txt(1.86,y+.35,name,16,weight='bold',color=C[key])
    txt(1.86,y+.76,platform,12.5,weight='bold')
    txt(1.86,y+1.08,readout,11.5,color=C['muted'])

# The shared features from all three cohorts feed one reference alignment.
for y in [2.12,3.90,5.68]:line([(3.87,y),(4.16,y)])
line([(4.16,2.12),(4.16,5.68)])
arrow((4.16,2.78),(4.58,2.78))
box(4.60,1.69,3.68,2.24,C['model'],edge='#b7c4cb')
txt(6.44,2.05,'RAMP alignment',17,weight='bold',ha='center')
txt(6.44,2.51,'9 shared protein markers',13,ha='center')
line([(4.98,2.83),(7.90,2.83)],color='#d3dce1',lw=1)
txt(6.44,3.14,'28 patients per cohort',12.5,ha='center')
txt(6.44,3.52,'1,000 cells per patient',12.5,ha='center')
arrow((6.44,3.93),(6.44,4.49))
txt(6.65,4.23,'Aligned profiles',10.5,color=C['muted'])

# One protein transfer operator, followed by two parallel uses.
box(4.60,4.52,3.68,1.75,'white',edge=C['senosain'],lw=1.35)
txt(6.44,4.89,'HLA-ABC state transfer',15,ha='center',weight='bold')
txt(6.44,5.34,'Senosain HLA-ABC reference',11.7,ha='center',color=C['senosain'])
txt(6.44,5.74,'Distance-weighted neighbors · k = 30',10.7,ha='center')
txt(6.44,6.67,'All cell types → all cell types',12.3,ha='center',weight='bold',color=C['muted'])

# Parallel target branches. Zheng is validation, not a second training source.
line([(8.28,5.38),(8.69,5.38),(8.69,2.39),(9.08,2.39)])
arrow((8.91,2.39),(9.08,2.39),color=C['zheng'])
line([(8.69,5.38),(8.69,4.36),(9.08,4.36)])
arrow((8.91,4.36),(9.08,4.36),color=C['cords'])
box(9.11,1.42,6.37,1.80,C['zheng_bg'],edge=C['zheng'])
txt(9.43,1.84,'Validate in Zheng',16,weight='bold',color=C['zheng'])
txt(9.43,2.27,'Predicted HLA state vs measured HLA-A',13)
txt(9.43,2.75,'Spatial concordance · 28 patients / 119 cores',11.7,color=C['muted'])

box(9.11,3.63,6.37,1.35,C['cords_bg'],edge=C['cords'])
txt(9.43,4.02,'Project to all Cords cells',16,weight='bold',color=C['cords'])
txt(9.43,4.56,'Frozen mapping · 527 patients · 2.85 million cells',11.8)
arrow((12.30,4.98),(12.30,5.34),color=C['cords'])
fibro(11.01,5.58,.28)
txt(11.48,5.59,'Author-annotated vCAF',15,weight='bold')

# Cell-level molecular co-variation and patient-level associations are separate.
line([(12.30,5.89),(12.30,6.04),(10.18,6.04)])
arrow((10.18,6.04),(10.18,6.34),color=C['cords'])
box(9.11,6.36,2.16,1.08,'white',edge='#d5a3b1')
txt(10.19,6.68,'Cell-level phenotype',11.5,ha='center',weight='bold')
txt(10.19,7.10,'CD73 / IDO',13,ha='center',color=C['cords'])

line([(12.30,6.04),(13.56,6.04)])
arrow((13.56,6.04),(13.56,6.25),color=C['cords'])
txt(13.57,6.49,'Patient mean predicted HLA-ABC',11.5,ha='center',weight='bold')
line([(13.57,6.74),(13.57,6.90),(12.64,6.90)])
line([(13.57,6.90),(14.70,6.90)])
arrow((12.64,6.90),(12.64,7.03),color=C['cords'])
arrow((14.70,6.90),(14.70,7.03),color=C['cords'])
box(11.60,7.05,2.12,.78,C['cords_bg'],edge='#e4c5ce',r=.10)
txt(12.66,7.29,'Immune context',11.2,ha='center',weight='bold')
txt(12.66,7.61,'Composition / TCF1',10.7,ha='center')
box(13.99,7.05,1.49,.78,C['cords_bg'],edge='#e4c5ce',r=.10)
txt(14.74,7.29,'DFS',12.5,ha='center',weight='bold')
txt(14.74,7.61,'280 patients',10.7,ha='center')

# Short scientific qualifier belongs to the feature stage, not a warning box.
txt(.52,7.49,'HLA-A and HLA-ABC are excluded\nfrom the shared alignment markers.',11.5,color=C['muted'])

fig.savefig(BASE/'panel_A_workflow.png',dpi=230)
fig.savefig(BASE/'panel_A_workflow.svg')
plt.close(fig)

source_files=['data_umap/README_ZH.md','validation/README_ZH.md','data_cords/README_ZH.md','summary_steps3_6/README_ZH.md']
inputs={rel:hashlib.sha256((STORY/rel).read_bytes()).hexdigest() for rel in source_files}
(BASE/'provenance/DESIGN.json').write_text(json.dumps({'version':'v1','time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'purpose':'Publication panel A workflow schematic; quantitative findings belong in subsequent panels.','language':'English artwork; Chinese notes and captions.','logic':['Three balanced cohort references contribute the shared protein space.','RAMP alignment uses nine shared proteins, excluding HLA-A and HLA-ABC.','One all-cell-types transfer operator uses Senosain measured relative HLA-ABC.','Zheng validation and Cords application are parallel branches of that same operator.','Cords full-cell projection precedes author-vCAF selection.','Within-vCAF CD73/IDO phenotype and patient-mean HLA associations are separate readouts.'],'input_sha256':inputs,'icons':'Schematic cohort cells and fibroblast symbol; no synthetic UMAP, tissue measurements or survival curves.','outputs':['panel_A_workflow.svg','panel_A_workflow.png'],'canvas_inches':[W,H]},ensure_ascii=False,indent=2)+'\n')
print('Panel A v1 SVG and PNG generated.',flush=True)
