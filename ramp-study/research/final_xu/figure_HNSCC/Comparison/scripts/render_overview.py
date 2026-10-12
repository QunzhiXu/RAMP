"""Compact full-cohort UMAP overview; fixed coordinates and identical drawing rules."""
import os,sys,json,hashlib
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parents[1];OUT=HERE/'evaluation'
os.environ['MPLBACKEND']='Agg';os.environ['MPLCONFIGDIR']=str(HERE/'.cache/evaluation/matplotlib')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
CFG=json.loads((HERE/'run_config.json').read_text())
METHODS=[('raw','Unaligned'),('ramp','RAMP'),('harmony','Harmony-direct'),('cycombine','cyCombine'),('cytonorm','CytoNorm'),('fastmnn','fastMNN')]+([(f'maxfuse_hub{i}',f'MaxFuse hub {i+1}') for i in range(3)] if len(CFG['datasets'])==3 else [('maxfuse','MaxFuse')])
def safe(s):
 import re
 return re.sub(r'[^A-Za-z0-9_.-]+','_',s).strip('_')
def main():
 md=pd.read_csv(HERE/'input/evaluation_metadata.csv.gz');coordinates={}
 for key,label in METHODS:
  p=OUT/'umap/coordinates'/(safe(label)+'.npy')
  if p.exists():coordinates[key]=np.load(p)
 if (HERE/'COMPLETE.json').exists():assert len(coordinates)==len(METHODS),'Completed comparison must have every UMAP'
 order=np.random.default_rng(42).permutation(len(md))
 palette=['#4477AA','#EEAA33','#CC6677','#009E73','#AA4499','#44AA99','#88CCEE','#DDCC77','#882255','#332288','#669900','#CCBB44']
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none'})
 for col in ['dataset','celltype']:
  cats=sorted(md[col].unique());colors={c:palette[i%len(palette)] for i,c in enumerate(cats)}
  for c in cats:
   if c=='Other' or any(s in c.lower() for s in ['ambiguous','uncertain','non-joint']):colors[c]='#BBBBBB'
  cellcolors=md[col].map(colors).to_numpy()
  cols=4 if len(METHODS)==7 else 3;rows=int(np.ceil(len(METHODS)/cols))
  fig,axes=plt.subplots(rows,cols,figsize=(cols*2.9,rows*2.65+1.0));axes=np.asarray(axes).reshape(-1)
  for ax,(key,label) in zip(axes,METHODS):
   if key in coordinates:
    xy=coordinates[key]
    ax.scatter(xy[order,0],xy[order,1],s=.23,c=cellcolors[order],alpha=.62,edgecolors='none',rasterized=True)
   else:ax.text(.5,.5,'Pending',ha='center',va='center',transform=ax.transAxes,color='#999999')
   ax.set_title(label,fontweight='bold',fontsize=11);ax.set_xticks([]);ax.set_yticks([])
   for sp in ax.spines.values():sp.set_visible(False)
  for ax in axes[len(METHODS):]:ax.set_visible(False)
  handles=[Line2D([],[],ls='',marker='o',color=colors[c],markersize=4,label=c) for c in cats]
  if len(METHODS)==7:
   axes[-1].set_visible(True);axes[-1].axis('off')
   axes[-1].legend(handles=handles,loc='center left',frameon=False,fontsize=9,title='Dataset' if col=='dataset' else 'Cell type')
  else:fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.005),ncol=min(5,len(cats)),frameon=False,fontsize=8)
  fig.suptitle({'hnscc_story':'HNSCC','lung_story':'Lung','melanoma_story':'Melanoma'}[CFG['case']]+f' · {len(md):,} cells · '+('platform' if col=='dataset' else 'cell type'),fontweight='bold',y=.99)
  fig.subplots_adjust(left=.015,right=.99,bottom=.03 if len(METHODS)==7 else .14,top=.93,hspace=.23,wspace=.06)
  for ext in ['png','svg']:fig.savefig(OUT/(f'umap_overview_{col}.'+ext),dpi=200,bbox_inches='tight')
  plt.close(fig)
 (OUT/'umap/overview_rendering.json').write_text(json.dumps({'draw_order':'One fixed random permutation (seed 42), all cells, shared across methods. No changes to coordinates.','methods_drawn':list(coordinates),'cell_count':len(md)},indent=2)+'\n')
 print('Overview rendered',len(coordinates),'/',len(METHODS))
if __name__=='__main__':main()
