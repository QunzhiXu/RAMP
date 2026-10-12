"""Display only CODEX cells using the frozen three-platform UMAP."""
from runtime import BASE,save,sha
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT=BASE/'CD4_NN_UMAP_validation/joint_umap'
d=pd.read_csv(OUT/'codex_predictions_on_joint_umap.csv.gz')
assert len(d)==30000 and d.dataset.eq('BM CODEX').all() and d.donor.nunique()==12
xy=d[['UMAP1','UMAP2']].to_numpy()
high=d.celltype.eq('CD4 T').to_numpy()
assert high.sum()==1426
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','text.color':'#26333E'})
fig=plt.figure(figsize=(6.5,4.7))
ax=fig.add_axes([.025,.12,.74,.72])
ax.scatter(xy[~high,0],xy[~high,1],s=.9,c='#CBD2D8',alpha=.5,lw=0,rasterized=True)
ax.scatter(xy[high,0],xy[high,1],s=3,c='#266BAE',alpha=.9,lw=0,rasterized=True)
ax.set_aspect('equal'); ax.axis('off')
fig.text(.035,.96,'CODEX CD4 T cells',fontsize=16,fontweight='bold',va='top')
fig.text(.035,.89,'Coordinates from the joint three-platform UMAP',fontsize=10.1,color='#586873')
fig.legend([Line2D([0],[0],marker='o',ls='',color='#266BAE',ms=5),
            Line2D([0],[0],marker='o',ls='',color='#CBD2D8',ms=5)],
           ['CD4 T\n(1,426 cells)','Other CODEX\ncells'],loc='center left',
           bbox_to_anchor=(.755,.56),frameon=False,fontsize=9.5,labelspacing=1.0,handletextpad=.5)
fig.text(.035,.064,'30,000 CODEX cells from all 12 donors; reference cells are hidden.',fontsize=8.7)
fig.text(.035,.028,'RAMP and joint UMAP use 13 shared markers, excluding CD4 and CD19.',fontsize=8.7)
for ext in ['png','svg']:
    fig.savefig(OUT/f'CODEX_CD4_on_joint_UMAP.{ext}',dpi=280,facecolor='white')
plt.close(fig)
save(OUT/'CODEX_CD4_focus_audit.json',{'displayed_dataset':'BM CODEX','n_cells':30000,
    'n_donors':12,'highlight':'author-derived CD4 T label','n_highlighted':1426,
    'coordinates':'subset of frozen all-platform UMAP; no refit or smoothing',
    'coordinate_sha256':sha(OUT/'umap_all.npy'),'reference_cells_displayed':False})
