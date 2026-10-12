"""Compact workflow with editable labels and the existing publication RAMP icon."""
from runtime import BASE,STORY
import json,hashlib,datetime
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, PathPatch
from matplotlib.path import Path as MPath
from PIL import Image

W,H=14.3,7.15
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
def fibro(cx,cy,scale=.25,color=None):
    coords=np.array([[-1.0,-.25],[-.28,-.20],[-.05,-.82],[.15,-.22],[.94,-.52],[.43,.02],[1.04,.57],[.20,.33],[-.13,.78],[-.25,.26],[-.97,.48],[-.55,.06],[-1.0,-.25]])*scale+[cx,cy]
    ax.add_patch(PathPatch(MPath(coords,[MPath.MOVETO]+[MPath.LINETO]*11+[MPath.CLOSEPOLY]),facecolor=color or C['cords'],edgecolor='none',alpha=.8,zorder=3))
    ax.add_patch(Circle((cx,cy),scale*.17,facecolor='white',edgecolor='none',zorder=4))

# Panel identifier and reading order.
txt(.34,.40,'A',27,weight='bold')
txt(.91,.43,'Cross-platform transfer of HLA-ABC states',20,weight='bold')
for x,s in [(.52,'PLATFORMS'),(3.11,'ALIGN & TRANSFER'),(7.41,'VALIDATE & INTERPRET')]:
    txt(x,1.00,s,11.2,color=C['muted'],weight='bold')

# Technology names only; cohort and model details remain in the caption.
for platform,key,cy in [('CyTOF','senosain',1.70),('mIF','zheng',2.65),('IMC','cords',3.60)]:
    box(.52,cy-.35,1.65,.70,C[key+'_bg'],edge=C[key],lw=1.2)
    txt(1.345,cy,platform,16,weight='bold',color=C[key],ha='center')
    line([(2.17,cy),(2.56,cy)])
line([(2.56,1.70),(2.56,3.60)])
arrow((2.56,2.10),(3.48,2.10))

# Exact existing publication icon; retain its native aspect ratio.
with Image.open(BASE/'assets/ramp_publication_icon.png') as im:
    logo=np.asarray(im.convert('RGB'))
logo_x,logo_y,logo_w=3.59,1.30,2.72
logo_h=logo_w*logo.shape[0]/logo.shape[1]
ax.imshow(logo,extent=(logo_x,logo_x+logo_w,logo_y+logo_h,logo_y),aspect='equal',zorder=4)
arrow((4.95,logo_y+logo_h+.06),(4.95,3.10))

# One protein transfer operator, followed by two parallel uses.
box(3.11,3.13,3.68,.88,'white',edge=C['senosain'],lw=1.35)
txt(4.95,3.57,'HLA-ABC state transfer',15,ha='center',weight='bold')

# Parallel target branches. Zheng is validation, not a second training source.
line([(6.79,3.57),(7.07,3.57),(7.07,1.86)])
arrow((7.07,1.86),(7.38,1.86),color=C['zheng'])
arrow((7.07,3.57),(7.38,3.57),color=C['cords'])
box(7.41,1.42,6.37,.88,C['zheng_bg'],edge=C['zheng'])
txt(10.595,1.86,'Validate HLA-A in mIF',16,weight='bold',color=C['zheng'],ha='center')

box(7.41,3.13,6.37,.88,C['cords_bg'],edge=C['cords'])
txt(10.595,3.57,'Project to all IMC cells',16,weight='bold',color=C['cords'],ha='center')
arrow((10.60,4.01),(10.60,4.34),color=C['cords'])
fibro(9.31,4.58,.28)
txt(9.78,4.59,'Author-annotated vCAF',15,weight='bold')

# Cell-level molecular co-variation and patient-level associations are separate.
line([(10.60,4.89),(10.60,5.04),(8.48,5.04)])
arrow((8.48,5.04),(8.48,5.34),color=C['cords'])
box(7.41,5.36,2.16,1.08,'white',edge='#d5a3b1')
txt(8.49,5.68,'Cell-level phenotype',11.5,ha='center',weight='bold')
txt(8.49,6.10,'CD73 / IDO',13,ha='center',color=C['cords'])

line([(10.60,5.04),(11.86,5.04)])
arrow((11.86,5.04),(11.86,5.25),color=C['cords'])
txt(11.87,5.49,'Patient mean predicted HLA-ABC',11.5,ha='center',weight='bold')
line([(11.87,5.74),(11.87,5.90),(10.94,5.90)])
line([(11.87,5.90),(13.00,5.90)])
arrow((10.94,5.90),(10.94,6.03),color=C['cords'])
arrow((13.00,5.90),(13.00,6.03),color=C['cords'])
box(9.90,6.05,2.12,.78,C['cords_bg'],edge='#e4c5ce',r=.10)
txt(10.96,6.29,'Immune context',11.2,ha='center',weight='bold')
txt(10.96,6.61,'Composition / TCF1',10.7,ha='center')
box(12.29,6.05,1.49,.78,C['cords_bg'],edge='#e4c5ce',r=.10)
txt(13.04,6.29,'DFS',12.5,ha='center',weight='bold')
txt(13.04,6.61,'280 patients',10.7,ha='center')

fig.savefig(BASE/'panel_A_workflow.png',dpi=230)
fig.savefig(BASE/'panel_A_workflow.svg')
plt.close(fig)

source_files=['data_umap/README_ZH.md','validation/README_ZH.md','data_cords/README_ZH.md','summary_steps3_6/README_ZH.md']
inputs={rel:hashlib.sha256((STORY/rel).read_bytes()).hexdigest() for rel in source_files}
(BASE/'provenance/DESIGN.json').write_text(json.dumps({'version':'v2','time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'purpose':'Simplified publication workflow following user edits.','language':'English artwork; Chinese notes and captions.','logic':['Three balanced cohort references contribute the shared protein space.','RAMP alignment uses nine shared proteins, excluding HLA-A and HLA-ABC; technical detail is in the caption.','One all-cell-types transfer operator uses Senosain measured relative HLA-ABC.','Zheng validation and Cords application are parallel branches, labeled by mIF and IMC.','Cords full-cell projection precedes author-vCAF selection.','Within-vCAF CD73/IDO phenotype and patient-mean HLA associations are separate readouts.'],'input_sha256':inputs,'icons':'Original RAMP publication bitmap embedded; fibroblast and all labels/connectors are vector elements.','icon_provenance':json.loads((BASE/'assets/SOURCE.json').read_text()),'outputs':['panel_A_workflow.svg','panel_A_workflow.png'],'canvas_inches':[W,H]},ensure_ascii=False,indent=2)+'\n')
print('Panel A v2 SVG and PNG generated.',flush=True)
